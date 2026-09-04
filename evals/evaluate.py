import argparse
import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, ".")

import httpx

from evals.metrics import (
    compute_latency_summary,
    detect_refusal,
    evaluate_citations,
    keyword_recall,
    parse_judge_response,
    summarize_faithfulness,
)

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def call_rag(question: str) -> dict[str, Any]:
    response = httpx.post(f"{API_URL}/api/v1/ask", json={"question": question, "top_k": 6}, timeout=120)
    response.raise_for_status()
    return response.json()


def judge_faithfulness_structured(question: str, context: str, answer: str) -> dict[str, Any]:
    from src.config import get_settings
    from src.services.ai_client import get_gemini_client

    prompt = (
        "You are an impartial scientific evaluation judge. Evaluate how faithfully the ANSWER "
        "is grounded in the provided CONTEXT excerpts.\n\n"
        "Rules:\n"
        "- Score 2: Every claim in the answer is directly supported by the context.\n"
        "- Score 1: The answer is mostly supported but contains minor extrapolation.\n"
        "- Score 0: The answer contradicts the context or invents unsupported facts.\n\n"
        f"QUESTION: {question}\n\n"
        f"CONTEXT (excerpts):\n{context[:6000]}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Reply with ONLY a JSON object:\n"
        '{"score": <0, 1, or 2>, "reason": "<max 15 words>", "claims_supported": <int>, "total_claims": <int>}'
    )
    client = get_gemini_client()
    model_name = get_settings().gemini_chat_model
    try:
        response = client.models.generate_content(model=model_name, contents=prompt)
        verdict = parse_judge_response(response.text or "")
        return {
            "score": verdict.score,
            "reason": verdict.reason,
            "claims_supported": verdict.claims_supported,
            "total_claims": verdict.total_claims,
            "judge_model": model_name,
            "is_valid": verdict.is_valid,
        }
    except Exception as exc:
        return {
            "score": -1.0,
            "reason": f"Judge error: {str(exc)[:60]}",
            "claims_supported": 0,
            "total_claims": 0,
            "judge_model": model_name,
            "is_valid": False,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG system on the golden dataset")
    parser.add_argument("--dataset", default="evals/golden_dataset.jsonl")
    parser.add_argument(
        "--skip-judge", action="store_true", help="Skip LLM faithfulness judging (keyword metrics only)"
    )
    parser.add_argument("--baseline", type=str, default=None, help="Path to baseline results JSON to compare against")
    parser.add_argument("--out", type=str, default="evals/results.json", help="Path to save evaluation results")
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    rows = []
    latencies_s: list[float] = []

    for i, case in enumerate(cases, 1):
        start = time.time()
        try:
            result = call_rag(case["question"])
        except Exception as exc:
            print(f"[{i}/{len(cases)}] FAILED: {exc}")
            continue

        latency = round(time.time() - start, 2)
        latencies_s.append(latency)
        citations_list = result.get("citations", [])
        citations_text = "\n".join(c.get("snippet", "") for c in citations_list)
        ans = result.get("answer", "")

        # Citation quality
        cite_eval = evaluate_citations(ans, citations_list)
        kw_recall = keyword_recall(ans, case.get("expected_keywords", []))
        is_refusal = detect_refusal(ans)

        row: dict[str, Any] = {
            "question": case["question"],
            "latency_s": latency,
            "n_sources": len(citations_list),
            "keyword_recall": kw_recall,
            "has_citations": cite_eval.has_citations,
            "citation_count": cite_eval.citation_count,
            "valid_citation_ratio": cite_eval.valid_citation_ratio,
            "citation_precision": cite_eval.citation_precision,
            "citation_overlap": cite_eval.mean_token_overlap,
            "is_refusal": is_refusal,
        }

        if not args.skip_judge and citations_text and not is_refusal:
            judge_res = judge_faithfulness_structured(case["question"], citations_text, ans)
            if judge_res.get("is_valid", False) and judge_res.get("score", -1.0) >= 0.0:
                row["faithfulness"] = judge_res["score"]
                row["judge_status"] = "JUDGED"
            else:
                row["faithfulness"] = None
                row["judge_status"] = "UNJUDGED"
            row["judge_reason"] = judge_res.get("reason", "")
        else:
            row["faithfulness"] = None
            row["judge_status"] = "SKIPPED" if args.skip_judge else ("REFUSAL" if is_refusal else "NO_CONTEXT")

        rows.append(row)
        faith_str = f"faithfulness={row['faithfulness']}" if row.get("faithfulness") is not None else ""
        print(f"[{i}/{len(cases)}] {case['question'][:50]}... recall={row['keyword_recall']} {faith_str}")

    if not rows:
        print("No results. Is the API running?")
        return

    def avg(key: str) -> float:
        values = [r[key] for r in rows if r.get(key) is not None and isinstance(r[key], (int, float))]
        return round(sum(values) / max(1, len(values)), 2)

    lat_summary = compute_latency_summary([r["latency_s"] * 1000 for r in rows])
    faith_summary = summarize_faithfulness(rows)

    summary = {
        "cases": len(rows),
        "avg_keyword_recall": avg("keyword_recall"),
        "avg_faithfulness_0_to_1": faith_summary.avg_faithfulness,
        "avg_latency_s": avg("latency_s"),
        "latency_median_s": round(lat_summary.median / 1000, 2),
        "latency_p95_s": round(lat_summary.p95 / 1000, 2),
        "avg_valid_citation_ratio": avg("valid_citation_ratio"),
        "avg_citation_precision": avg("citation_precision"),
        "grounding_failures": faith_summary.grounding_failures,
        "unjudged_cases": faith_summary.unjudged_cases,
        "clean_refusals": sum(1 for r in rows if r.get("is_refusal")),
    }

    print("\n=== EVALUATION SUMMARY ===")
    print(json.dumps(summary, indent=2))

    payload = {"summary": summary, "details": rows}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nDetails saved to {args.out}")

    if args.baseline:
        try:
            from evals.regression import compare_single_metric

            with open(args.baseline, encoding="utf-8") as f:
                base_data = json.load(f)
            base_summary = base_data.get("summary", base_data)

            metrics_to_compare = (
                "avg_keyword_recall",
                "avg_faithfulness_0_to_1",
                "avg_latency_s",
                "grounding_failures",
                "unjudged_cases",
            )
            for metric_key in metrics_to_compare:
                if (
                    metric_key in base_summary
                    and metric_key in summary
                    and isinstance(base_summary[metric_key], (int, float))
                    and isinstance(summary[metric_key], (int, float))
                ):
                    diff = compare_single_metric(metric_key, base_summary[metric_key], summary[metric_key])
                    status_map = {
                        "REGRESSED": "[REGRESSED]",
                        "IMPROVED": "[IMPROVED]",
                        "STABLE": "[STABLE]",
                    }
                    status_icon = status_map.get(diff.status, "[STABLE]")
                    print(
                        f"{metric_key:<25} base={diff.baseline:<6} cand={diff.candidate:<6} "
                        f"rel={diff.relative_change_pct:+.1f}%  {status_icon} {diff.status}"
                    )
        except Exception as exc:
            print(f"Warning: Baseline comparison failed: {exc}")


if __name__ == "__main__":
    main()
