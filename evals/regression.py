"""Regression detection framework for RAG evaluation benchmarks.

Compares candidate evaluation metrics against a baseline run and flags:
- IMPROVEMENT: Metric improved beyond the improvement threshold.
- REGRESSION: Metric degraded beyond the regression tolerance threshold.
- STABLE: Metric change remains within expected statistical variance.

Supports both retrieval benchmark comparisons and end-to-end QA evaluation comparisons.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Default relative tolerance thresholds (fraction of baseline)
DEFAULT_THRESHOLDS: dict[str, float] = {
    "mrr": 0.05,  # 5% relative drop in MRR is a regression
    "hit_at_1": 0.05,  # 5% relative drop in Hit@1
    "hit_at_3": 0.05,
    "hit_at_5": 0.05,
    "hit_at_10": 0.05,
    "precision_at_5": 0.05,
    "recall_at_5": 0.05,
    "latency_ms": 0.25,  # 25% increase in latency is a regression
    "keyword_recall": 0.08,  # 8% drop in keyword recall
    "faithfulness": 0.10,  # 10% drop in faithfulness
}

LOWER_IS_BETTER: set[str] = {
    "latency_ms",
    "latency_s",
    "avg_latency_s",
    "grounding_failures",
    "unjudged_cases",
}


@dataclass
class MetricDiff:
    metric: str
    baseline: float
    candidate: float
    absolute_change: float
    relative_change_pct: float
    status: str  # "IMPROVED", "REGRESSED", "STABLE"
    threshold_pct: float


@dataclass
class ConfigDiff:
    config_name: str
    mode: str
    metrics: dict[str, MetricDiff] = field(default_factory=dict)
    has_regression: bool = False


@dataclass
class RegressionReport:
    verdict: str  # "PASS" or "REGRESSION_DETECTED"
    total_metrics_compared: int = 0
    regressions_count: int = 0
    improvements_count: int = 0
    stable_count: int = 0
    configurations: list[ConfigDiff] = field(default_factory=list)
    overall_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_single_metric(
    metric_name: str,
    baseline_val: float,
    candidate_val: float,
    threshold: float | None = None,
) -> MetricDiff:
    """Compare a single numerical metric between baseline and candidate."""
    tol = threshold if threshold is not None else DEFAULT_THRESHOLDS.get(metric_name, 0.05)
    abs_change = candidate_val - baseline_val

    # Avoid zero division
    if abs(baseline_val) < 1e-6:
        rel_change = 0.0 if abs(candidate_val) < 1e-6 else (100.0 if candidate_val > 0 else -100.0)
    else:
        rel_change = (abs_change / baseline_val) * 100.0

    is_lower_better = metric_name in LOWER_IS_BETTER
    threshold_pct = tol * 100.0

    if is_lower_better:
        # For latency/errors: positive change means regression, negative means improvement
        if rel_change > threshold_pct:
            status = "REGRESSED"
        elif rel_change < -threshold_pct:
            status = "IMPROVED"
        else:
            status = "STABLE"
    else:
        # For quality: negative change means regression, positive means improvement
        if rel_change < -threshold_pct:
            status = "REGRESSED"
        elif rel_change > threshold_pct:
            status = "IMPROVED"
        else:
            status = "STABLE"

    return MetricDiff(
        metric=metric_name,
        baseline=round(baseline_val, 3),
        candidate=round(candidate_val, 3),
        absolute_change=round(abs_change, 3),
        relative_change_pct=round(rel_change, 1),
        status=status,
        threshold_pct=round(threshold_pct, 1),
    )


def compare_benchmarks(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    custom_thresholds: dict[str, float] | None = None,
) -> RegressionReport:
    """Compare full benchmark datasets (lists of configuration rows).

    Matches records on (config, mode) key.
    """
    thresholds = {**DEFAULT_THRESHOLDS, **(custom_thresholds or {})}

    # Key by (config, mode) or fallback to config name / index
    def make_key(rec: dict[str, Any], idx: int) -> str:
        cfg = rec.get("config") or rec.get("name") or str(idx)
        mode = rec.get("mode") or ""
        return f"{cfg}::{mode}" if mode else str(cfg)

    baseline_map = {make_key(r, i): r for i, r in enumerate(baseline_records)}
    candidate_map = {make_key(r, i): r for i, r in enumerate(candidate_records)}

    config_diffs: list[ConfigDiff] = []
    total_metrics = 0
    regressions = 0
    improvements = 0
    stables = 0

    all_keys = sorted(set(baseline_map.keys()) & set(candidate_map.keys()))

    for key in all_keys:
        base_row = baseline_map[key]
        cand_row = candidate_map[key]

        cfg_name = base_row.get("config", key.split("::")[0])
        mode = base_row.get("mode", "")
        cfg_diff = ConfigDiff(config_name=cfg_name, mode=mode)

        # Identify numerical metrics present in both
        common_metrics = [
            k
            for k in base_row
            if k in cand_row
            and isinstance(base_row[k], (int, float))
            and isinstance(cand_row[k], (int, float))
            and k not in ("size", "overlap")
        ]

        for m in sorted(common_metrics):
            diff = compare_single_metric(m, float(base_row[m]), float(cand_row[m]), thresholds.get(m))
            cfg_diff.metrics[m] = diff
            total_metrics += 1
            if diff.status == "REGRESSED":
                regressions += 1
                cfg_diff.has_regression = True
            elif diff.status == "IMPROVED":
                improvements += 1
            else:
                stables += 1

        config_diffs.append(cfg_diff)

    verdict = "REGRESSION_DETECTED" if regressions > 0 else "PASS"

    return RegressionReport(
        verdict=verdict,
        total_metrics_compared=total_metrics,
        regressions_count=regressions,
        improvements_count=improvements,
        stable_count=stables,
        configurations=config_diffs,
        overall_summary={
            "baseline_configs": len(baseline_records),
            "candidate_configs": len(candidate_records),
            "matched_configs": len(all_keys),
        },
    )


def format_markdown_report(report: RegressionReport) -> str:
    """Format a human-readable Markdown diff table from a RegressionReport."""
    lines = [
        f"# RAG Regression Evaluation Report: **{report.verdict}**",
        "",
        f"- **Verdict:** `{report.verdict}`",
        f"- **Total Metrics Checked:** {report.total_metrics_compared}",
        f"- **Regressions:** {report.regressions_count}",
        f"- **Improvements:** {report.improvements_count}",
        f"- **Stable:** {report.stable_count}",
        "",
    ]

    for cfg in report.configurations:
        title = f"{cfg.config_name} ({cfg.mode})" if cfg.mode else cfg.config_name
        lines.append(f"### Configuration: `{title}`")
        lines.append("| Metric | Baseline | Candidate | Rel Change | Status |")
        lines.append("|---|---|---|---|---|")
        for m_name, diff in cfg.metrics.items():
            status_tag = (
                "[REGRESSED]"
                if diff.status == "REGRESSED"
                else ("[IMPROVED]" if diff.status == "IMPROVED" else "[STABLE]")
            )
            lines.append(
                f"| {m_name} | {diff.baseline} | {diff.candidate} | {diff.relative_change_pct:+.1f}% | {status_tag} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare candidate RAG benchmark against baseline")
    parser.add_argument("baseline", type=Path, help="Path to baseline JSON results")
    parser.add_argument("candidate", type=Path, help="Path to candidate JSON results")
    parser.add_argument("--tolerance-mrr", type=float, default=0.05, help="Relative MRR tolerance (default: 0.05)")
    parser.add_argument(
        "--tolerance-latency", type=float, default=0.25, help="Relative latency tolerance (default: 0.25)"
    )
    parser.add_argument("--out-json", type=Path, default=None, help="Save diff report JSON")
    args = parser.parse_args()

    if not args.baseline.exists():
        sys.exit(f"Error: Baseline file not found: {args.baseline}")
    if not args.candidate.exists():
        sys.exit(f"Error: Candidate file not found: {args.candidate}")

    with open(args.baseline, encoding="utf-8") as f:
        base_data = json.load(f)
    with open(args.candidate, encoding="utf-8") as f:
        cand_data = json.load(f)

    # Normalize single summary dicts to list
    base_rows = base_data if isinstance(base_data, list) else [base_data.get("summary", base_data)]
    cand_rows = cand_data if isinstance(cand_data, list) else [cand_data.get("summary", cand_data)]

    custom_tols = {
        "mrr": args.tolerance_mrr,
        "latency_ms": args.tolerance_latency,
    }

    report = compare_benchmarks(base_rows, cand_rows, custom_tols)
    md_output = format_markdown_report(report)
    print(md_output)

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nSaved report to {args.out_json}")

    if report.verdict == "REGRESSION_DETECTED":
        sys.exit(1)


if __name__ == "__main__":
    main()
