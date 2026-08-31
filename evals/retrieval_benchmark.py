import argparse
import json
import sys
import time

sys.path.insert(0, ".")

from src.services import embeddings, reranker
from src.services import opensearch_service as oss
from src.services.experiments import DEFAULT_MATRIX, build_benchmark_index
from src.services.retrieval import reciprocal_rank_fusion


def evaluate_index(index_name: str, mode: str, cases: list[dict], k: int = 5, embed_model: str | None = None) -> dict:
    hits_at_k = 0
    reciprocal_ranks = []
    latencies: list[float] = []
    for case in cases:
        query = case["question"]
        expected = case["arxiv_id"]
        start = time.perf_counter()
        if mode == "bm25":
            results = oss.bm25_search(query, top_k=10, index_name=index_name)
        elif mode == "hybrid_rerank":
            vec = embeddings.embed_query(query, model=embed_model)
            results = reciprocal_rank_fusion(
                [
                    oss.bm25_search(query, top_k=10, index_name=index_name),
                    oss.vector_search(vec, top_k=10, index_name=index_name),
                ],
                top_k=10,
            )
            results = reranker.rerank(query, results, top_k=10)
        else:
            vec = embeddings.embed_query(query, model=embed_model)
            if mode == "vector":
                results = oss.vector_search(vec, top_k=10, index_name=index_name)
            else:
                results = reciprocal_rank_fusion(
                    [
                        oss.bm25_search(query, top_k=10, index_name=index_name),
                        oss.vector_search(vec, top_k=10, index_name=index_name),
                    ],
                    top_k=10,
                )
        latencies.append((time.perf_counter() - start) * 1000)
        doc_ids = [r.get("arxiv_id", "") for r in results]
        rank = next((i + 1 for i, d in enumerate(doc_ids) if d == expected), None)
        if rank and rank <= k:
            hits_at_k += 1
            reciprocal_ranks.append(1.0 / rank)

    n = max(1, len(cases))
    avg_ms = round(sum(latencies) / len(latencies), 1) if latencies else 0
    return {"hit_at_k": round(hits_at_k / n, 3), "mrr": round(sum(reciprocal_ranks) / n, 3), "latency_ms": avg_ms}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark chunking strategies x retrieval modes")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--configs", nargs="*", help="Subset of config names to run")
    parser.add_argument("--rebuild", action="store_true", help="Force re-chunk + re-index (cache makes it cheap)")
    parser.add_argument(
        "--embed-model", default=None, help="Pin one embedding model for all configs (overrides fallback chain)"
    )
    parser.add_argument("--eval-only", action="store_true", help="Skip index builds; evaluate existing indices")
    args = parser.parse_args()

    with open("evals/golden_retrieval.jsonl", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    print(f"Evaluating {len(cases)} golden questions at k={args.k}\n")

    configs = [c for c in DEFAULT_MATRIX if not args.configs or c.name in args.configs]
    chain = [args.embed_model] if args.embed_model else embeddings.fallback_models()
    current_model = chain[0]
    remaining_models = chain[1:]

    rows = []
    for config in configs:
        if args.eval_only:
            index_name = f"bench_{config.name}"
            if not oss.get_client().indices.exists(index=index_name):
                print(f"skipping {config.name}: no existing index")
                continue
        else:
            while True:
                try:
                    start = time.time()
                    index_name = build_benchmark_index(config, force=args.rebuild, embed_model=current_model or None)
                    print(f"built {index_name} in {round(time.time() - start, 1)}s (model: {current_model})")
                    break
                except RuntimeError as exc:
                    if "Daily" in str(exc) and remaining_models:
                        current_model = remaining_models.pop(0)
                        print(f"\n>>> switching embedding model to {current_model} and retrying {config.name}\n")
                        continue
                    raise
        for mode in ("bm25", "vector", "hybrid", "hybrid_rerank"):
            metrics = evaluate_index(index_name, mode, cases, k=args.k, embed_model=current_model or None)
            rows.append(
                {
                    "config": config.name, "strategy": config.strategy,
                    "size": config.size, "overlap": config.overlap,
                    "mode": mode, "embed_model": current_model, **metrics,
                }
            )
            print(
                f"{config.name:<16} {mode:<13} hit@{args.k}={metrics['hit_at_k']:.2f}  "
                f"mrr={metrics['mrr']:.2f}  {metrics['latency_ms']}ms"
            )
        print()

    rows.sort(key=lambda r: (-r["hit_at_k"], -r["mrr"]))
    best = rows[0]
    print("=" * 60)
    print(
        f"BEST CONFIG: {best['config']} ({best['mode']})  "
        f"hit@{args.k}={best['hit_at_k']}  mrr={best['mrr']}  {best['latency_ms']}ms"
    )
    print("=" * 60)

    with open("evals/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print("Saved to evals/benchmark_results.json")


if __name__ == "__main__":
    main()
