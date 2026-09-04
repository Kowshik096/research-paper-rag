"""Unit tests for the RAG evaluation and regression framework.

100% deterministic, zero external API or network dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.metrics import (
    compute_latency_summary,
    detect_refusal,
    evaluate_citations,
    evaluate_retrieval_case,
    extract_citation_indices,
    hit_at_k,
    keyword_recall,
    ndcg_at_k,
    parse_judge_response,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    summarize_faithfulness,
)
from evals.regression import (
    compare_benchmarks,
    compare_single_metric,
    format_markdown_report,
)

# ==============================================================================
# 1. Retrieval Metrics Tests
# ==============================================================================


def test_hit_at_k():
    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d", "doc_e"]

    # Target at rank 1
    assert hit_at_k(retrieved, ["doc_a"], k=1) == 1.0
    assert hit_at_k(retrieved, ["doc_a"], k=5) == 1.0

    # Target at rank 3
    assert hit_at_k(retrieved, ["doc_c"], k=1) == 0.0
    assert hit_at_k(retrieved, ["doc_c"], k=3) == 1.0
    assert hit_at_k(retrieved, ["doc_c"], k=5) == 1.0

    # Target not in retrieved
    assert hit_at_k(retrieved, ["doc_z"], k=5) == 0.0

    # Edge cases
    assert hit_at_k([], ["doc_a"], k=5) == 0.0
    assert hit_at_k(retrieved, [], k=5) == 0.0
    assert hit_at_k(retrieved, ["doc_a"], k=0) == 0.0
    assert hit_at_k(retrieved, ["doc_a"], k=-1) == 0.0


def test_reciprocal_rank():
    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d"]

    assert reciprocal_rank(retrieved, ["doc_a"]) == 1.0
    assert reciprocal_rank(retrieved, ["doc_b"]) == 0.5
    assert reciprocal_rank(retrieved, ["doc_c"]) == 1.0 / 3.0
    assert reciprocal_rank(retrieved, ["doc_d"]) == 0.25
    assert reciprocal_rank(retrieved, ["doc_z"]) == 0.0
    assert reciprocal_rank([], ["doc_a"]) == 0.0
    assert reciprocal_rank(retrieved, []) == 0.0


def test_precision_at_k():
    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d", "doc_e"]
    relevant = {"doc_a", "doc_c", "doc_e"}

    assert precision_at_k(retrieved, relevant, k=1) == 1.0  # doc_a
    assert precision_at_k(retrieved, relevant, k=2) == 0.5  # doc_a (1/2)
    assert precision_at_k(retrieved, relevant, k=3) == 2.0 / 3.0  # doc_a, doc_c (2/3)
    assert precision_at_k(retrieved, relevant, k=5) == 3.0 / 5.0  # doc_a, doc_c, doc_e (3/5)

    # Edge cases
    assert precision_at_k([], relevant, k=5) == 0.0
    assert precision_at_k(retrieved, relevant, k=0) == 0.0


def test_precision_at_k_deduplication():
    # 1. Repeated chunks from the same paper: ["A", "A", "A", "B", "C"] with relevant ["A"]
    # MUST NOT produce 3/5 = 0.60; must produce 1/5 = 0.20!
    chunks = ["paper_a", "paper_a", "paper_a", "paper_b", "paper_c"]
    assert precision_at_k(chunks, ["paper_a"], k=5) == 0.20
    assert recall_at_k(chunks, ["paper_a"], k=5) == 1.0
    assert hit_at_k(chunks, ["paper_a"], k=5) == 1.0
    assert reciprocal_rank(chunks, ["paper_a"]) == 1.0

    # Explicit non-deduplicated mode verifies the difference
    assert precision_at_k(chunks, ["paper_a"], k=5, dedup=False) == 3.0 / 5.0

    # 2. Multiple unique papers: ["A", "B", "C", "D", "E"] with relevant ["A", "B"]
    assert precision_at_k(["paper_a", "paper_b", "paper_c", "paper_d", "paper_e"], ["paper_a", "paper_b"], k=5) == 0.40
    assert recall_at_k(["paper_a", "paper_b", "paper_c", "paper_d", "paper_e"], ["paper_a", "paper_b"], k=5) == 1.0

    # 3. Repeated relevant paper across all slots: ["A", "A", "A", "A", "A"] with relevant ["A"]
    assert precision_at_k(["paper_a"] * 5, ["paper_a"], k=5) == 0.20
    assert recall_at_k(["paper_a"] * 5, ["paper_a"], k=5) == 1.0
    assert hit_at_k(["paper_a"] * 5, ["paper_a"], k=1) == 1.0

    # 4. No relevant paper found
    assert precision_at_k(["paper_a", "paper_a", "paper_b", "paper_c"], ["paper_z"], k=5) == 0.0
    assert recall_at_k(["paper_a", "paper_a", "paper_b", "paper_c"], ["paper_z"], k=5) == 0.0
    assert hit_at_k(["paper_a", "paper_a", "paper_b", "paper_c"], ["paper_z"], k=5) == 0.0

    # 5. Empty retrieval
    assert precision_at_k([], ["paper_a"], k=5) == 0.0
    assert recall_at_k([], ["paper_a"], k=5) == 0.0

    # 6. k larger than available unique papers: ["A", "A", "B"] (only 2 unique papers), k=5
    assert precision_at_k(["paper_a", "paper_a", "paper_b"], ["paper_a"], k=5) == 0.20
    assert recall_at_k(["paper_a", "paper_a", "paper_b"], ["paper_a"], k=5) == 1.0

    # 7. Deduplicated NDCG verifies ranking
    assert ndcg_at_k(["paper_a", "paper_a", "paper_b"], ["paper_a"], k=3) == 1.0


def test_recall_at_k():
    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d"]
    relevant = {"doc_a", "doc_c", "doc_z"}  # 3 total relevant

    assert recall_at_k(retrieved, relevant, k=1) == 1.0 / 3.0  # found doc_a
    assert recall_at_k(retrieved, relevant, k=3) == 2.0 / 3.0  # found doc_a, doc_c
    assert recall_at_k(retrieved, relevant, k=4) == 2.0 / 3.0  # doc_z was never retrieved

    # Edge cases
    assert recall_at_k([], relevant, k=5) == 0.0
    assert recall_at_k(retrieved, set(), k=5) == 0.0


def test_ndcg_at_k():
    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d"]

    # Perfect ranking: relevant doc at rank 1
    assert ndcg_at_k(retrieved, ["doc_a"], k=3) == 1.0

    # Relevant doc at rank 2: DCG = 1/log2(3) = 0.6309, IDCG = 1/log2(2) = 1.0
    score_rank2 = ndcg_at_k(retrieved, ["doc_b"], k=3)
    assert 0.63 < score_rank2 < 0.64

    # No hits
    assert ndcg_at_k(retrieved, ["doc_z"], k=3) == 0.0


def test_evaluate_retrieval_case():
    res = evaluate_retrieval_case(
        query="test query",
        retrieved_ids=["doc_1", "doc_2", "doc_3"],
        expected_ids=["doc_2"],
        latency_ms=25.0,
    )
    assert res.hit_at_1 == 0.0
    assert res.hit_at_3 == 1.0
    assert res.hit_at_5 == 1.0
    assert res.mrr == 0.5
    assert res.latency_ms == 25.0


# ==============================================================================
# 2. Citation Metrics Tests
# ==============================================================================


def test_extract_citation_indices():
    assert extract_citation_indices("According to [1], eddies transport heat [2].") == [1, 2]
    assert extract_citation_indices("Multiple studies [1][2][10] agree.") == [1, 2, 10]
    assert extract_citation_indices("No citations in this text.") == []
    assert extract_citation_indices("") == []


def test_evaluate_citations_valid():
    answer = "Mesoscale eddies contribute to heat transport [1], especially in the Southern Ocean [2]."
    citations = [
        {
            "index": 1,
            "title": "Paper 1",
            "snippet": "Mesoscale eddies contribute significantly to meridional heat transport.",
        },
        {
            "index": 2,
            "title": "Paper 2",
            "snippet": "Observations in the Southern Ocean demonstrate eddy kinetic energy.",
        },
    ]
    eval_res = evaluate_citations(answer, citations)

    assert eval_res.has_citations is True
    assert eval_res.citation_count == 2
    assert eval_res.unique_citations == [1, 2]
    assert eval_res.valid_citation_ratio == 1.0
    assert eval_res.unreferenced_sources == []
    assert eval_res.citation_precision == 1.0
    assert eval_res.mean_token_overlap > 0.0


def test_evaluate_citations_invalid_and_unreferenced():
    # Model cites [1] and non-existent [4]
    answer = "Eddies mix water [1] and impact salinity [4]."
    citations = [
        {"index": 1, "snippet": "Eddies mix water and transport tracers."},
        {"index": 2, "snippet": "Atmospheric coupling in tropical zones."},
        {"index": 3, "snippet": "Deep sea bathymetry effects."},
    ]
    eval_res = evaluate_citations(answer, citations)

    assert eval_res.has_citations is True
    assert eval_res.citation_count == 2
    # [1] is valid, [4] is invalid -> 1/2 = 0.5
    assert eval_res.valid_citation_ratio == 0.5
    # Sources 2 and 3 were never cited
    assert eval_res.unreferenced_sources == [2, 3]
    # Precision: 1 cited out of 3 total sources = 1/3
    assert abs(eval_res.citation_precision - 0.333) < 0.01


def test_evaluate_citations_no_citations():
    answer = "I could not find this in the indexed papers."
    citations = [{"index": 1, "snippet": "Some text."}]
    eval_res = evaluate_citations(answer, citations)

    assert eval_res.has_citations is False
    assert eval_res.citation_count == 0
    assert eval_res.valid_citation_ratio == 0.0
    assert eval_res.unreferenced_sources == [1]


# ==============================================================================
# 3. Answer Metrics Tests
# ==============================================================================


def test_keyword_recall():
    answer = "The Agulhas Current leakage transports warm saline water to the Atlantic overturning system."
    keywords = ["Agulhas", "leakage", "overturning", "saline"]
    assert keyword_recall(answer, keywords) == 1.0

    partial_keywords = ["Agulhas", "cyclone", "typhoon"]
    assert abs(keyword_recall(answer, partial_keywords) - 1.0 / 3.0) < 0.01

    assert keyword_recall("", ["test"]) == 0.0
    assert keyword_recall(answer, []) == 1.0


def test_detect_refusal():
    assert detect_refusal("I could not find this in the indexed papers.") is True
    assert detect_refusal("Question is outside the research domain of the indexed papers.") is True
    assert detect_refusal("Empty question provided.") is True
    assert detect_refusal("The Agulhas Current transports 15 Sv of water.") is False


def test_parse_judge_response():
    # Perfect JSON with 0-2 scale
    raw_1 = '{"score": 2, "claims_supported": 3, "total_claims": 3, "reason": "All claims supported"}'
    v1 = parse_judge_response(raw_1)
    assert v1.is_valid is True
    assert v1.score == 1.0
    assert v1.claims_supported == 3
    assert v1.total_claims == 3

    # Normalized 0.0-1.0 scale
    raw_2 = '{"score": 0.5, "claims_supported": 1, "total_claims": 2, "reason": "Partially supported"}'
    v2 = parse_judge_response(raw_2)
    assert v2.is_valid is True
    assert v2.score == 0.5

    # Markdown wrapped JSON
    raw_3 = '```json\n{"score": 0, "reason": "Hallucination"}\n```'
    v3 = parse_judge_response(raw_3)
    assert v3.is_valid is True
    assert v3.score == 0.0

    # Fallback to plain digit
    raw_4 = "I rate this answer a 2 because it matches the context."
    v4 = parse_judge_response(raw_4)
    assert v4.is_valid is True
    assert v4.score == 1.0

    # Malformed garbage
    raw_5 = "No score provided here."
    v5 = parse_judge_response(raw_5)
    assert v5.is_valid is False
    assert v5.score == -1.0


def test_summarize_faithfulness():
    # 1. Valid score 0 -> grounding failure
    res_score_0 = summarize_faithfulness([{"faithfulness": 0.0, "is_valid": True}])
    assert res_score_0.grounding_failures == 1
    assert res_score_0.avg_faithfulness == 0.0
    assert res_score_0.unjudged_cases == 0
    assert res_score_0.valid_judged_count == 1

    # 2. Valid score 1 (normalized 0.5) -> valid judged result, NOT a grounding failure
    res_score_1 = summarize_faithfulness([{"faithfulness": 0.5, "is_valid": True}])
    assert res_score_1.grounding_failures == 0
    assert res_score_1.avg_faithfulness == 0.5
    assert res_score_1.unjudged_cases == 0
    assert res_score_1.valid_judged_count == 1

    # 3. Valid score 2 (normalized 1.0) -> valid judged result
    res_score_2 = summarize_faithfulness([{"faithfulness": 1.0, "is_valid": True}])
    assert res_score_2.grounding_failures == 0
    assert res_score_2.avg_faithfulness == 1.0
    assert res_score_2.unjudged_cases == 0
    assert res_score_2.valid_judged_count == 1

    # 4. Judge timeout/error -> unjudged, NOT a grounding failure
    res_timeout = summarize_faithfulness([{"faithfulness": None, "judge_status": "UNJUDGED", "is_valid": False}])
    assert res_timeout.grounding_failures == 0
    assert res_timeout.unjudged_cases == 1
    assert res_timeout.avg_faithfulness is None
    assert res_timeout.valid_judged_count == 0

    # 5. Malformed judge response -> unjudged, NOT a grounding failure
    res_malformed = summarize_faithfulness([{"faithfulness": -1.0, "is_valid": False}])
    assert res_malformed.grounding_failures == 0
    assert res_malformed.unjudged_cases == 1
    assert res_malformed.avg_faithfulness is None
    assert res_malformed.valid_judged_count == 0

    # 6, 7, 8. Mixed scenario: 3 valid judged cases (1.0, 0.5, 0.0) and 1 failed judge
    mixed_records = [
        {"faithfulness": 1.0, "is_valid": True, "judge_status": "JUDGED"},
        {"faithfulness": 0.5, "is_valid": True, "judge_status": "JUDGED"},
        {"faithfulness": 0.0, "is_valid": True, "judge_status": "JUDGED"},
        {"faithfulness": None, "is_valid": False, "judge_status": "UNJUDGED"},
    ]
    mixed_summary = summarize_faithfulness(mixed_records)

    # 6. Faithfulness average excludes unjudged cases: (1.0 + 0.5 + 0.0) / 3 = 0.50
    assert mixed_summary.avg_faithfulness == 0.50
    # 7. Grounding_failures strictly excludes unjudged cases (only the 0.0 score is a failure)
    assert mixed_summary.grounding_failures == 1
    # 8. Unjudged_cases correctly increments
    assert mixed_summary.unjudged_cases == 1
    assert mixed_summary.valid_judged_count == 3


# ==============================================================================
# 4. Latency Statistics Tests
# ==============================================================================


def test_compute_latency_summary():
    empty_res = compute_latency_summary([])
    assert empty_res.count == 0
    assert empty_res.mean == 0.0

    single_res = compute_latency_summary([100.0])
    assert single_res.count == 1
    assert single_res.mean == 100.0
    assert single_res.median == 100.0
    assert single_res.p95 == 100.0

    samples = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    res = compute_latency_summary(samples)
    assert res.count == 10
    assert res.mean == 55.0
    assert res.median == 55.0
    assert res.min == 10.0
    assert res.max == 100.0
    assert res.p95 >= 95.0


# ==============================================================================
# 5. Regression Detection Tests
# ==============================================================================


def test_compare_single_metric():
    # Higher is better: improvement
    m1 = compare_single_metric("mrr", baseline_val=0.80, candidate_val=0.90, threshold=0.05)
    assert m1.status == "IMPROVED"
    assert m1.relative_change_pct > 5.0

    # Higher is better: regression (drop > 5%)
    m2 = compare_single_metric("mrr", baseline_val=0.80, candidate_val=0.70, threshold=0.05)
    assert m2.status == "REGRESSED"
    assert m2.relative_change_pct < -5.0

    # Higher is better: stable (within 5%)
    m3 = compare_single_metric("mrr", baseline_val=0.80, candidate_val=0.82, threshold=0.05)
    assert m3.status == "STABLE"

    # Lower is better (latency): candidate takes longer -> regression
    m4 = compare_single_metric("latency_ms", baseline_val=100.0, candidate_val=140.0, threshold=0.25)
    assert m4.status == "REGRESSED"  # 40% increase > 25% tolerance

    # Lower is better (latency): candidate is faster -> improvement
    m5 = compare_single_metric("latency_ms", baseline_val=100.0, candidate_val=70.0, threshold=0.25)
    assert m5.status == "IMPROVED"


def test_compare_benchmarks_pass():
    baseline = [
        {"config": "para_1200_o150", "mode": "hybrid", "mrr": 0.90, "hit_at_5": 1.0, "latency_ms": 100.0},
    ]
    # Candidate improves MRR by +8.9% (exceeding 5% threshold)
    candidate = [
        {"config": "para_1200_o150", "mode": "hybrid", "mrr": 0.98, "hit_at_5": 1.0, "latency_ms": 95.0},
    ]
    report = compare_benchmarks(baseline, candidate)
    assert report.verdict == "PASS"
    assert report.regressions_count == 0
    assert report.improvements_count >= 1

    md = format_markdown_report(report)
    assert "PASS" in md
    assert "para_1200_o150 (hybrid)" in md


def test_compare_benchmarks_regression():
    baseline = [
        {"config": "para_1200_o150", "mode": "hybrid", "mrr": 0.95, "hit_at_5": 1.0, "latency_ms": 100.0},
    ]
    # Candidate drops MRR from 0.95 to 0.70 (> 25% drop)
    candidate = [
        {"config": "para_1200_o150", "mode": "hybrid", "mrr": 0.70, "hit_at_5": 0.80, "latency_ms": 100.0},
    ]
    report = compare_benchmarks(baseline, candidate)
    assert report.verdict == "REGRESSION_DETECTED"
    assert report.regressions_count >= 1
