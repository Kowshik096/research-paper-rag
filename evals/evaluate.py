import argparse
import json
import re
import sys
import time

sys.path.insert(0, ".")

import httpx

API_URL = "http://localhost:8000"


def call_rag(question: str) -> dict:
    response = httpx.post(f"{API_URL}/api/v1/ask", json={"question": question, "top_k": 6}, timeout=120)
    response.raise_for_status()
    return response.json()


def keyword_hit(answer: str, keywords: list[str]) -> float:
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if re.search(r"\b" + re.escape(kw.lower()), answer_lower))
    return hits / len(keywords)


def judge_faithfulness(question: str, context: str, answer: str) -> float:
    from src.config import get_settings
    from src.services.llm import get_client

    prompt = (
        "You are an evaluator. Rate how faithful the ANSWER is to the CONTEXT, "
        "where 0 = contradicts or fabricates information not in context, "
        "1 = mostly supported with minor extrapolation, "
        "2 = every claim supported by the context.\n\n"
        f"QUESTION: {question}\n\nCONTEXT (excerpts):\n{context[:6000]}\n\nANSWER:\n{answer}\n\n"
        'Reply with only a JSON object: {"score": <number>}'
    )
    client = get_client()
    response = client.models.generate_content(model=get_settings().gemini_chat_model, contents=prompt)
    match = re.search(r"\d", response.text or "")
    return int(match.group()) / 2 if match else -1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG system on the golden dataset")
    parser.add_argument("--dataset", default="evals/golden_dataset.jsonl")
    parser.add_argument(
        "--skip-judge", action="store_true", help="Skip LLM faithfulness judging (keyword metrics only)"
    )
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    rows = []
    for i, case in enumerate(cases, 1):
        start = time.time()
        try:
            result = call_rag(case["question"])
        except Exception as exc:
            print(f"[{i}/{len(cases)}] FAILED: {exc}")
            continue

        latency = round(time.time() - start, 1)
        citations = "\n".join(c["snippet"] for c in result.get("citations", []))
        row = {
            "question": case["question"],
            "latency_s": latency,
            "n_sources": len(result.get("citations", [])),
            "keyword_recall": round(keyword_hit(result["answer"], case["expected_keywords"]), 2),
        }
        if not args.skip_judge and citations:
            row["faithfulness"] = judge_faithfulness(case["question"], citations, result["answer"])
        rows.append(row)
        print(f"[{i}/{len(cases)}] {case['question'][:60]}... recall={row['keyword_recall']} "
              f"{row.get('faithfulness', '')}")

    if not rows:
        print("No results. Is the API running?")
        return

    def avg(key: str) -> float:
        values = [r[key] for r in rows if key in r]
        return round(sum(values) / max(1, len(values)), 2)

    summary = {
        "cases": len(rows),
        "avg_keyword_recall": avg("keyword_recall"),
        "avg_faithfulness_0_to_1": avg("faithfulness"),
        "avg_latency_s": avg("latency_s"),
        "grounding_failures": sum(1 for r in rows if r.get("faithfulness") is not None and r["faithfulness"] < 0.5),
    }
    print("\n=== EVALUATION SUMMARY ===")
    print(json.dumps(summary, indent=2))
    with open("evals/results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": rows}, f, indent=2)
    print("\nDetails saved to evals/results.json")


if __name__ == "__main__":
    main()
