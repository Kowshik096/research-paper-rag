"""Evaluation metrics library for RAG systems.

Provides decoupled, objective metric calculations for:
1. Retrieval quality (Hit@k, MRR, Precision@k, Recall@k, NDCG@k)
2. Citation quality (presence, validity, precision, snippet overlap)
3. Answer quality (keyword recall, refusal detection, structured faithfulness)
4. Latency statistics (mean, median, p95, min, max)

Zero external API dependencies: purely deterministic and fully unit-testable.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

# ==============================================================================
# 1. Retrieval Metrics
# ==============================================================================


def hit_at_k(retrieved_ids: list[str], relevant_ids: list[str] | set[str], k: int) -> float:
    """Calculate Hit@k (binary: 1.0 if any relevant doc is within the top-k retrieved, else 0.0)."""
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    top_k_ids = retrieved_ids[:k]
    return 1.0 if any(doc_id in relevant_set for doc_id in top_k_ids) else 0.0


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str] | set[str], max_k: int = 100) -> float:
    """Calculate Reciprocal Rank (1 / rank of first relevant doc, or 0.0 if none found within max_k)."""
    if not retrieved_ids or not relevant_ids or max_k <= 0:
        return 0.0
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids[:max_k], 1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str] | set[str],
    k: int,
    dedup: bool = True,
) -> float:
    """Calculate Precision@k: fraction of top-k retrieved docs that are relevant.

    If dedup is True (default for paper-level evaluation), duplicate document/paper
    IDs are collapsed preserving their first-appearance rank order before taking the
    top-k cutoff. This evaluates top-k unique papers and prevents multiple chunks
    from the same paper from inflating precision.
    """
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    eval_ids = list(dict.fromkeys(retrieved_ids)) if dedup else retrieved_ids
    top_k_ids = eval_ids[:k]
    hits = sum(1 for doc_id in top_k_ids if doc_id in relevant_set)
    return hits / k


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str] | set[str],
    k: int,
    dedup: bool = True,
) -> float:
    """Calculate Recall@k: fraction of all relevant docs that appear in top-k retrieved.

    If dedup is True (default for paper-level evaluation), duplicate document/paper
    IDs are collapsed preserving their first-appearance rank order before taking the
    top-k cutoff.
    """
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    eval_ids = list(dict.fromkeys(retrieved_ids)) if dedup else retrieved_ids
    top_k_ids = eval_ids[:k]
    hits = sum(1 for doc_id in top_k_ids if doc_id in relevant_set)
    return hits / len(relevant_set)


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str] | set[str],
    k: int,
    dedup: bool = True,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k (binary relevance).

    If dedup is True (default for paper-level evaluation), duplicate document/paper
    IDs are collapsed preserving their first-appearance rank order, ensuring a paper
    is only credited once.
    """
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    eval_ids = list(dict.fromkeys(retrieved_ids)) if dedup else retrieved_ids
    top_k_ids = eval_ids[:k]

    dcg = 0.0
    for rank, doc_id in enumerate(top_k_ids, 1):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(k, len(relevant_set))
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0.0 else 0.0


@dataclass
class RetrievalCaseResult:
    query: str
    expected_ids: list[str]
    retrieved_ids: list[str]
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0
    mrr: float = 0.0
    precision_at_5: float = 0.0
    recall_at_5: float = 0.0
    ndcg_at_5: float = 0.0
    latency_ms: float = 0.0


def evaluate_retrieval_case(
    query: str, retrieved_ids: list[str], expected_ids: list[str], latency_ms: float = 0.0
) -> RetrievalCaseResult:
    """Evaluate a single query's retrieved document list against ground truth."""
    return RetrievalCaseResult(
        query=query,
        expected_ids=expected_ids,
        retrieved_ids=retrieved_ids,
        hit_at_1=hit_at_k(retrieved_ids, expected_ids, 1),
        hit_at_3=hit_at_k(retrieved_ids, expected_ids, 3),
        hit_at_5=hit_at_k(retrieved_ids, expected_ids, 5),
        hit_at_10=hit_at_k(retrieved_ids, expected_ids, 10),
        mrr=reciprocal_rank(retrieved_ids, expected_ids, max_k=20),
        precision_at_5=precision_at_k(retrieved_ids, expected_ids, 5),
        recall_at_5=recall_at_k(retrieved_ids, expected_ids, 5),
        ndcg_at_5=ndcg_at_k(retrieved_ids, expected_ids, 5),
        latency_ms=latency_ms,
    )


# ==============================================================================
# 2. Citation Metrics
# ==============================================================================

_CITATION_REGEX = re.compile(r"\[(\d+)\]")


def extract_citation_indices(text: str) -> list[int]:
    """Extract all integer citation markers like [1], [2] from text in appearance order."""
    if not text:
        return []
    return [int(m) for m in _CITATION_REGEX.findall(text)]


@dataclass
class CitationEvaluation:
    citation_count: int = 0
    unique_citations: list[int] = field(default_factory=list)
    has_citations: bool = False
    valid_citation_ratio: float = 0.0
    unreferenced_sources: list[int] = field(default_factory=list)
    citation_precision: float = 0.0
    mean_token_overlap: float = 0.0


def evaluate_citations(
    answer: str,
    citations: list[dict[str, Any]],
) -> CitationEvaluation:
    """Evaluate citation quality of a generated answer against the citations payload.

    Args:
        answer: The text generated by the LLM.
        citations: The list of citation dicts provided in the response envelope
                   (expected to have 1-based indexing, e.g. index=1..N, snippet=...).
    """
    cited_indices = extract_citation_indices(answer)
    unique_cited = sorted(set(cited_indices))
    total_sources = len(citations)

    if total_sources == 0:
        return CitationEvaluation(
            citation_count=len(cited_indices),
            unique_citations=unique_cited,
            has_citations=bool(cited_indices),
            valid_citation_ratio=0.0 if cited_indices else 1.0,
            unreferenced_sources=[],
            citation_precision=0.0,
            mean_token_overlap=0.0,
        )

    valid_indices = {i for i in range(1, total_sources + 1)}
    valid_cited = [idx for idx in cited_indices if idx in valid_indices]
    valid_ratio = len(valid_cited) / len(cited_indices) if cited_indices else 0.0

    unreferenced = [i for i in range(1, total_sources + 1) if i not in unique_cited]
    precision = (total_sources - len(unreferenced)) / total_sources

    overlaps: list[float] = []
    snippet_by_index: dict[int, str] = {}
    for i, c in enumerate(citations, 1):
        idx = c.get("index", i)
        snippet_by_index[idx] = c.get("snippet", "") or c.get("content", "")

    sentences = re.split(r"(?<=[.!?])\s+", answer)
    for sentence in sentences:
        indices_in_sentence = extract_citation_indices(sentence)
        if not indices_in_sentence:
            continue
        sentence_words = set(re.findall(r"\w+", sentence.lower()))
        for idx in indices_in_sentence:
            snippet = snippet_by_index.get(idx, "")
            if not snippet or not sentence_words:
                continue
            snippet_words = set(re.findall(r"\w+", snippet.lower()))
            common = sentence_words & snippet_words
            overlap = len(common) / len(sentence_words) if sentence_words else 0.0
            overlaps.append(overlap)

    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0

    return CitationEvaluation(
        citation_count=len(cited_indices),
        unique_citations=unique_cited,
        has_citations=len(cited_indices) > 0,
        valid_citation_ratio=round(valid_ratio, 3),
        unreferenced_sources=unreferenced,
        citation_precision=round(precision, 3),
        mean_token_overlap=round(mean_overlap, 3),
    )


# ==============================================================================
# 3. Answer Metrics & Quality
# ==============================================================================

REFUSAL_PHRASES = (
    "could not find this in the indexed papers",
    "outside the research domain",
    "no indexed papers matched",
    "empty question provided",
    "disallowed instructions",
)


def detect_refusal(answer: str) -> bool:
    """Return True if the answer is a recognized domain refusal or 'not found' response."""
    lowered = answer.lower()
    return any(phrase in lowered for phrase in REFUSAL_PHRASES)


def keyword_recall(answer: str, expected_keywords: list[str]) -> float:
    """Calculate the fraction of expected keywords found in the answer string (word-boundary regex)."""
    if not expected_keywords:
        return 1.0
    if not answer or not answer.strip():
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if re.search(r"\b" + re.escape(kw.lower()) + r"\b", answer_lower))
    return round(hits / len(expected_keywords), 3)


@dataclass
class StructuredJudgeVerdict:
    score: float
    claims_supported: int = 0
    total_claims: int = 0
    reason: str = ""
    raw_response: str = ""
    is_valid: bool = True


def parse_judge_response(raw_text: str) -> StructuredJudgeVerdict:
    """Parse and validate structured JSON judge output with graceful regex fallbacks."""
    if not raw_text or not raw_text.strip():
        return StructuredJudgeVerdict(score=-1.0, reason="Empty judge response", is_valid=False)

    import json

    match = re.search(r"\{[^{}]*\}", raw_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            score_val = data.get("score")
            if score_val is not None:
                numeric_score = float(score_val)
                if numeric_score > 1.0:
                    numeric_score = numeric_score / 2.0
                return StructuredJudgeVerdict(
                    score=min(max(round(numeric_score, 2), 0.0), 1.0),
                    claims_supported=int(data.get("claims_supported", 0)),
                    total_claims=int(data.get("total_claims", 0)),
                    reason=str(data.get("reason", ""))[:120],
                    raw_response=raw_text[:200],
                    is_valid=True,
                )
        except Exception:
            pass

    digit_match = re.search(r"\b([0-2])\b", raw_text)
    if digit_match:
        val = int(digit_match.group(1)) / 2.0
        return StructuredJudgeVerdict(
            score=val,
            reason="Regex extracted digit",
            raw_response=raw_text[:200],
            is_valid=True,
        )

    return StructuredJudgeVerdict(
        score=-1.0, reason="Failed to parse score", raw_response=raw_text[:200], is_valid=False
    )


@dataclass
class FaithfulnessSummary:
    avg_faithfulness: float | None
    grounding_failures: int
    unjudged_cases: int
    valid_judged_count: int


def summarize_faithfulness(records: list[dict[str, Any]]) -> FaithfulnessSummary:
    """Summarize faithfulness judge results across a set of QA evaluation records.

    Distinguishes:
    - VALID JUDGED CASE: score in [0.0, 1.0] (corresponding to original 0, 1, or 2).
    - GROUNDING FAILURE: valid score < 0.5 (score 0 indicates contradiction or unsupported claims).
    - UNJUDGED CASE: judge failed, timed out, was skipped, or output was malformed.

    Unjudged cases are strictly EXCLUDED from avg_faithfulness and are NOT
    counted as grounding failures.
    """
    valid_scores: list[float] = []
    grounding_failures = 0
    unjudged_cases = 0

    for r in records:
        score = r.get("faithfulness")
        judge_status = r.get("judge_status", "")
        # A case is valid judged if faithfulness is a float/int >= 0.0 and judge_status is not UNJUDGED
        if (
            score is not None
            and isinstance(score, (int, float))
            and score >= 0.0
            and judge_status != "UNJUDGED"
            and r.get("is_valid", True) is not False
        ):
            valid_scores.append(float(score))
            if float(score) < 0.5:
                grounding_failures += 1
        else:
            # If judge was attempted and failed/invalid or explicitly marked UNJUDGED
            if judge_status == "UNJUDGED" or score == -1.0 or r.get("is_valid") is False:
                unjudged_cases += 1

    avg_val = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else None

    return FaithfulnessSummary(
        avg_faithfulness=avg_val,
        grounding_failures=grounding_failures,
        unjudged_cases=unjudged_cases,
        valid_judged_count=len(valid_scores),
    )


# ==============================================================================
# 4. Latency Statistics
# ==============================================================================


@dataclass
class LatencySummary:
    count: int = 0
    mean: float = 0.0
    median: float = 0.0
    p95: float = 0.0
    min: float = 0.0
    max: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_latency_summary(samples_ms: list[float]) -> LatencySummary:
    """Compute statistical percentiles for a list of latency samples (in milliseconds)."""
    if not samples_ms:
        return LatencySummary()

    sorted_samples = sorted(samples_ms)
    n = len(sorted_samples)
    mean_val = round(sum(sorted_samples) / n, 1)
    median_val = round(statistics.median(sorted_samples), 1)

    if n == 1:
        p95_val = sorted_samples[0]
    else:
        k = 0.95 * (n - 1)
        f = math.floor(k)
        c = math.ceil(k)
        p95_val = sorted_samples[int(k)] if f == c else sorted_samples[f] * (c - k) + sorted_samples[c] * (k - f)

    return LatencySummary(
        count=n,
        mean=mean_val,
        median=median_val,
        p95=round(p95_val, 1),
        min=round(sorted_samples[0], 1),
        max=round(sorted_samples[-1], 1),
    )
