import argparse
import json
import sys
import time

sys.path.insert(0, ".")

from evals.metrics import compute_latency_summary, evaluate_retrieval_case
from src.services import embeddings, reranker
from src.services import opensearch_service as oss
from src.services.experiments import DEFAULT_MATRIX, build_benchmark_index
from src.services.retrieval import reciprocal_rank_fusion


def evaluate_index(index_name: str, mode: str, cases: list[dict], k: int = 5, embed_model: str | None = None) -> dict:
    case_results = []
    latencies: list[float] = []

    for case in cases:
        query = case["question"]
        expected = [case["arxiv_id"]]
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
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        doc_ids = [r.get("arxiv_id", "") for r in results]
        case_res = evaluate_retrieval_case(query, doc_ids, expected, latency_ms=latency_ms)
        case_results.append(case_res)

    n = max(1, len(cases))
    lat_summary = compute_latency_summary(latencies)

    avg_mrr = sum(c.mrr for c in case_results) / n
    avg_hit_1 = sum(c.hit_at_1 for c in case_results) / n
    avg_hit_3 = sum(c.hit_at_3 for c in case_results) / n
    avg_hit_5 = sum(c.hit_at_5 for c in case_results) / n
    avg_hit_10 = sum(c.hit_at_10 for c in case_results) / n
    avg_p5 = sum(c.precision_at_5 for c in case_results) / n
    avg_r5 = sum(c.recall_at_5 for c in case_results) / n
    avg_ndcg5 = sum(c.ndcg_at_5 for c in case_results) / n

    return {
        "hit_at_k": round(avg_hit_5, 3),  # Backwards compatibility
        "hit_at_1": round(avg_hit_1, 3),
        "hit_at_3": round(avg_hit_3, 3),
        "hit_at_5": round(avg_hit_5, 3),
        "hit_at_10": round(avg_hit_10, 3),
        "mrr": round(avg_mrr, 3),
        "precision_at_5": round(avg_p5, 3),
        "recall_at_5": round(avg_r5, 3),
        "ndcg_at_5": round(avg_ndcg5, 3),
        "latency_ms": lat_summary.mean,  # Backwards compatibility
        "latency_median_ms": lat_summary.median,
        "latency_p95_ms": lat_summary.p95,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark chunking strategies x retrieval modes")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--configs", nargs="*", help="Subset of config names to run")
    parser.add_argument("--rebuild", action="store_true", help="Force re-chunk + re-index (cache makes it cheap)")
    parser.add_argument(
        "--embed-model", default=None, help="Pin one embedding model for all configs (overrides fallback chain)"
    )
    parser.add_argument("--eval-only", action="store_true", help="Skip index builds; evaluate existing indices")
    parser.add_argument("--baseline", type=str, default=None, help="Path to baseline benchmark JSON to compare against")
    parser.add_argument("--out", type=str, default="evals/benchmark_results.json", help="Path to save benchmark JSON")
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
                    "config": config.name,
                    "strategy": config.strategy,
                    "size": config.size,
                    "overlap": config.overlap,
                    "mode": mode,
                    "embed_model": current_model,
                    **metrics,
                }
            )
            print(
                f"{config.name:<16} {mode:<13} hit@1={metrics['hit_at_1']:.2f}  "
                f"hit@{args.k}={metrics['hit_at_5']:.2f}  mrr={metrics['mrr']:.2f}  "
                f"p@5={metrics['precision_at_5']:.2f}  r@5={metrics['recall_at_5']:.2f}  "
                f"lat_p95={metrics['latency_p95_ms']}ms"
            )
        print()

    rows.sort(key=lambda r: (-r["hit_at_5"], -r["mrr"]))
    best = rows[0]
    print("=" * 60)
    print(
        f"BEST CONFIG: {best['config']} ({best['mode']})  "
        f"hit@1={best['hit_at_1']}  hit@{args.k}={best['hit_at_5']}  "
        f"mrr={best['mrr']}  lat_p95={best['latency_p95_ms']}ms"
    )
    print("=" * 60)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved to {args.out}")

    if args.baseline:
        try:
            from evals.regression import compare_benchmarks, format_markdown_report

            with open(args.baseline, encoding="utf-8") as f:
                base_data = json.load(f)
            report = compare_benchmarks(base_data, rows)
            print("\n" + format_markdown_report(report))
        except Exception as exc:
            print(f"Warning: Baseline comparison failed: {exc}")


if __name__ == "__main__":
    main()
