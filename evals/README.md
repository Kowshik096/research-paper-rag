# RAG Evaluation & Regression Framework

An objective, decoupled evaluation framework designed to measure retrieval quality, citation validity, answer grounding, and latency across RAG pipeline configurations.

---

## 1. Metrics Overview

### A. Retrieval Metrics (`evals/metrics.py`)
* **Chunk-Level Evaluation (`Hit@k`, `MRR`):** $k$ refers to the top-$k$ retrieved chunks returned by OpenSearch/reranker.
  * **Hit@k ($k \in \{1, 3, 5, 10\}$):** Binary indicator (1.0 or 0.0) measuring whether any chunk from the ground-truth target research paper is present within the top-$k$ retrieved chunks.
  * **MRR (Mean Reciprocal Rank):** Reciprocal rank of the first relevant chunk ($1/\text{rank}$) averaged across queries.
* **Paper-Level Evaluation (`Precision@k`, `Recall@k`, `NDCG@k`):** $k$ refers to the top-$k$ unique research papers after deduplicating contiguous/repeated chunks by their paper ID while strictly preserving first-appearance rank order.
  * **Precision@k:** Fraction of the top-$k$ unique retrieved papers that are relevant. Deduplication ensures that multiple chunks from the same paper (e.g. `[A, A, A, B, C]`) do NOT inflate paper-level precision (yielding 1/5 = 0.20 rather than an invalid 3/5).
  * **Recall@k:** Fraction of all relevant papers that appear in the top-$k$ unique retrieved papers.
  * **NDCG@k:** Evaluates paper ranking quality with logarithmic rank discounting.

### B. Citation Quality Metrics
* **Citation Presence:** Whether the answer contains numerical citation tags (e.g., `[1]`, `[2]`).
* **Citation Validity Ratio:** Fraction of cited indices that correspond to actual citations provided in the response metadata (`1 <= idx <= len(citations)`). Flags hallucinated citation numbers.
* **Citation Precision:** Fraction of returned sources that were actually referenced in the answer text (identifies unused retrieval baggage).
* **Token Overlap:** Mean unigram overlap between each citing sentence and its corresponding citation snippet (lexical verification of relevance).

### C. Answer Quality & Faithfulness
* **Keyword Recall:** Fraction of expected domain keywords present in the generated answer text.
* **Clean Refusal Detection:** Detects whether the system appropriately responded with an honest refusal (*"could not find this in the indexed papers"*) rather than fabricating claims.
* **Structured Faithfulness (LLM-as-a-Judge):** Independent judge evaluating whether every claim in the answer is grounded in the retrieved excerpts:
  * `Score 2` (1.0): Every claim directly supported by context.
  * `Score 1` (0.5): Mostly supported with minor extrapolation.
  * `Score 0` (0.0): Contradicts or fabricates facts not found in context.
  * *Safety:* Enforces structured JSON output (`score`, `claims_supported`, `total_claims`, `reason`) with deterministic fallback parsing.
* **Grounding Failure vs. Unjudged Cases:**
  * **Grounding Failure:** A valid judge verdict where `score < 0.5` (contradiction or hallucination).
  * **Unjudged Case (`unjudged_cases`):** Cases where the LLM judge failed, timed out, or returned malformed output. Unjudged cases are strictly tracked in `unjudged_cases`, are **NOT** counted as grounding failures, and are **excluded** from `avg_faithfulness_0_to_1` to prevent API network errors from falsifying RAG quality metrics.

### D. Latency Statistics
* Evaluates `mean`, `median` ($p50$), and $p95$ percentiles in milliseconds to distinguish typical response times from tail latency spikes.

---

## 2. Regression Detection (`evals/regression.py`)

Compares candidate benchmark results against a baseline run using configurable relative tolerance thresholds:
* **Higher-is-better metrics (`mrr`, `hit_at_k`, `precision`, `recall`):**
  * Degradation $> 5\%$ is flagged as `[REGRESSED]` (triggers exit code 1 in automated checks).
  * Improvement $> 5\%$ is flagged as `[IMPROVED]`.
  * Within $\pm 5\%$ is flagged as `[STABLE]`.
* **Lower-is-better metrics (`latency_ms`):**
  * Latency increase $> 25\%$ is flagged as `[REGRESSED]`.
  * Latency decrease $> 25\%$ is flagged as `[IMPROVED]`.

---

## 3. How to Run

### 1. Run Offline Retrieval Benchmark
Evaluates 6 chunking configurations across 4 retrieval modes (`bm25`, `vector`, `hybrid`, `hybrid_rerank`):
```bash
uv run python evals/retrieval_benchmark.py --embed-model "local:BAAI/bge-base-en-v1.5"
```
To automatically compare against an existing baseline run:
```bash
uv run python evals/retrieval_benchmark.py --baseline evals/benchmark_results.json
```

### 2. Run End-to-End QA Evaluation (requires API running on port 8000)
```bash
# Full evaluation with structured LLM-as-a-judge
uv run python evals/evaluate.py

# Fast offline keyword & citation check without LLM judging
uv run python evals/evaluate.py --skip-judge

# Compare against baseline results
uv run python evals/evaluate.py --baseline evals/results.json
```

### 3. Run Standalone Regression Comparison
```bash
uv run python -m evals.regression evals/benchmark_results.json candidate_results.json
```

### 4. Run Deterministic Unit Tests
```bash
uv run pytest tests/test_evaluation_framework.py -v
```

---

## 4. Current Implementation vs. Future Extensions

| Feature | Currently Implemented | Planned Future Extension |
|---|---|---|
| **Retrieval Quality** | Hit@1, Hit@3, Hit@5, Hit@10, MRR, Precision@5, Recall@5, NDCG@5 | Dense vs. sparse failure mode cluster analysis |
| **Citations** | Numerical index extraction, validity ratio, source precision, lexical snippet overlap | Exact PDF character bounding-box coordinate matching |
| **Answer Quality** | Word-boundary keyword recall, domain refusal detection | Multi-aspect semantic question-answering F1 |
| **Faithfulness Judge**| Structured JSON extraction, score normalization, claim counts, fallback parsing | Multi-judge ensemble with calibration against human labels |
| **Regression Tool** | Automated threshold comparison, markdown tables, CLI diff, JSON diff | Automated PR comment bot in GitHub Actions |
| **Golden Datasets** | 8 retrieval golden queries, 7 QA golden queries | Scaled synthetic benchmark corpus (100+ papers, 500+ QAs) |

---

## 5. Honest Limitations

1. **LLM-as-a-Judge:** While structured prompts and regex fallbacks improve stability, LLM judges can still exhibit position bias, verbosity bias, and self-enhancement bias. Judge scores should be treated as informative signals, not infallible truths.
2. **Dataset Size:** The current golden dataset contains 8 curated retrieval queries and 7 QA pairs. While ideal for rapid local regression testing, broader statistical significance requires expanding the benchmark dataset.
