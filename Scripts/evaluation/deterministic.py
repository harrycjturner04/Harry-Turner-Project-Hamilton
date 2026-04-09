"""Deterministic (non-LLM) metrics for pipeline run evaluation.

Ingests:
- Gate scores (quality_score 0-1) from debug/__gate.json files
- Verdict check pass rates from debug/__verdict.json files
- Report-level metrics from report markdown and data_profile.json
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Stages that the gate/critic system covers
_GATED_STAGES = ("cleaning", "analysis", "cross_validation")

# Pattern: {dataset}__{stage}[__{retry_label}]__gate.json
_GATE_RE = re.compile(
    r"^(?P<dataset>.+?)__(?P<stage>cleaning|analysis|cross_validation)"
    r"(?:__(?P<retry>retry\d+))?__gate\.json$"
)
_VERDICT_RE = re.compile(
    r"^(?P<dataset>.+?)__(?P<stage>cleaning|analysis|cross_validation)"
    r"(?:__(?P<retry>retry\d+))?__verdict\.json$"
)


# ── Gate / Verdict ingestion ──────────────────────────────────────────────────

def _parse_gate_file(path: Path) -> Optional[Dict[str, Any]]:
    """Parse a gate JSON file. Returns None on failure."""
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception as exc:
        logger.warning("Could not parse gate file %s: %s", path, exc)
        return None


def _parse_verdict_file(path: Path) -> Optional[Dict[str, Any]]:
    """Parse a verdict JSON file. Returns None on failure."""
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception as exc:
        logger.warning("Could not parse verdict file %s: %s", path, exc)
        return None


def _compute_check_pass_rates(
    verdict: Dict[str, Any],
) -> Tuple[Optional[float], Optional[float], int]:
    """Compute check pass rates from a verdict dict.

    Returns:
        (must_fix_pass_rate, should_fix_pass_rate, total_checks)
    """
    all_checks: List[Dict[str, Any]] = []
    for key in ("structural_checks", "content_checks",
                "plot_checks", "claim_checks"):
        checks = verdict.get(key)
        if isinstance(checks, list):
            all_checks.extend(checks)

    if not all_checks:
        return None, None, 0

    must_fix = [c for c in all_checks if c.get("severity") == "MUST_FIX"]
    should_fix = [c for c in all_checks if c.get("severity") == "SHOULD_FIX"]

    must_rate = (
        sum(1 for c in must_fix if c.get("passed")) / len(must_fix)
        if must_fix else None
    )
    should_rate = (
        sum(1 for c in should_fix if c.get("passed")) / len(should_fix)
        if should_fix else None
    )
    return must_rate, should_rate, len(all_checks)


def ingest_gate_metrics(run_dir: Path, run_id: str, db: Any) -> int:
    """Scan debug/ directory and ingest gate/verdict metrics into the DB.

    Args:
        run_dir: Path to the run output directory.
        run_id: Run identifier (directory name).
        db: EvalDB instance.

    Returns:
        Number of gate records ingested.
    """
    debug_dir = run_dir / "debug"
    if not debug_dir.is_dir():
        logger.warning("No debug/ directory in %s", run_dir)
        return 0

    # Group gate files by (dataset, stage)
    # Key: (dataset, stage) -> {attempt_label: gate_path}
    gate_files: Dict[Tuple[str, str], Dict[str, Path]] = {}
    verdict_files: Dict[Tuple[str, str, str], Path] = {}

    for f in debug_dir.iterdir():
        m = _GATE_RE.match(f.name)
        if m:
            dataset = m.group("dataset")
            stage = m.group("stage")
            retry = m.group("retry") or ""
            gate_files.setdefault((dataset, stage), {})[retry] = f
            continue

        m = _VERDICT_RE.match(f.name)
        if m:
            dataset = m.group("dataset")
            stage = m.group("stage")
            retry = m.group("retry") or ""
            verdict_files[(dataset, stage, retry)] = f

    count = 0
    for (dataset, stage), attempts in gate_files.items():
        # Count retries for this stage
        num_retries = sum(1 for k in attempts if k.startswith("retry"))

        for attempt_label, gate_path in sorted(attempts.items()):
            gate = _parse_gate_file(gate_path)
            if gate is None:
                continue

            # Look up corresponding verdict
            verdict_path = verdict_files.get((dataset, stage, attempt_label))
            must_rate = should_rate = None
            total_checks = 0
            if verdict_path:
                verdict = _parse_verdict_file(verdict_path)
                if verdict:
                    must_rate, should_rate, total_checks = (
                        _compute_check_pass_rates(verdict)
                    )

            db.upsert_stage_gate_metric(
                run_id=run_id,
                dataset_name=dataset,
                stage=stage,
                attempt_label=attempt_label,
                quality_score=gate.get("quality_score"),
                status=gate.get("status"),
                quality_breakdown=gate.get("quality_breakdown"),
                check_pass_rate_must_fix=must_rate,
                check_pass_rate_should_fix=should_rate,
                total_checks=total_checks,
                num_retries=num_retries if attempt_label == "" else None,
            )
            count += 1

    logger.info(
        "Ingested %d gate metric records for run %s", count, run_id
    )
    return count


# ── Report-level deterministic metrics ───────────────────────────────────────

def _get_column_names(run_dir: Path, dataset_name: str) -> List[str]:
    """Extract dataset column names from data_profile.json, falling back to
    the cleaned parquet when schema profiling was not enabled."""
    profile_path = (
        run_dir / "files" / dataset_name / "analysis" / "data_profile.json"
    )
    if profile_path.exists():
        try:
            profile = json.loads(profile_path.read_text("utf-8"))
            columns = profile.get("columns", [])
            names = [c["name"] for c in columns if isinstance(c, dict) and "name" in c]
            if names:
                return names
        except Exception as exc:
            logger.warning("Could not parse data_profile.json: %s", exc)

    # Fallback: read column names directly from the cleaned parquet
    parquet_path = (
        run_dir / "files" / dataset_name / "cleaning"
        / f"{dataset_name}__cleaned.parquet"
    )
    if parquet_path.exists():
        try:
            import pandas as pd
            return list(pd.read_parquet(parquet_path, columns=None).columns)
        except Exception as exc:
            logger.warning(
                "Could not read columns from cleaned parquet %s: %s",
                parquet_path, exc,
            )
    return []


def _get_report_text(run_dir: Path, dataset_name: str) -> Optional[str]:
    """Load the report markdown for a dataset."""
    report_path = run_dir / "reports" / f"{dataset_name}__report.md"
    if not report_path.exists():
        return None
    try:
        return report_path.read_text("utf-8")
    except Exception as exc:
        logger.warning("Could not read report: %s", exc)
        return None


def _count_report_figures(run_dir: Path, dataset_name: str) -> int:
    """Count PNG figures generated for a dataset's report."""
    fig_dir = run_dir / "reports" / "report_figures" / dataset_name
    if not fig_dir.is_dir():
        return 0
    return sum(1 for f in fig_dir.iterdir() if f.suffix.lower() == ".png")


def compute_data_coverage_rate(
    report_text: str, column_names: List[str]
) -> Optional[float]:
    """Fraction of dataset column names referenced in the report body.

    Measures how thoroughly the report engages with the raw data dimensions.
    Only meaningful when column_names is non-empty.
    """
    if not column_names or not report_text:
        return None
    found = sum(1 for col in column_names if col in report_text)
    return round(found / len(column_names), 4)


def compute_figure_reference_rate(
    report_text: str, total_figures: int
) -> Optional[float]:
    """Fraction of generated figures explicitly referenced in the report.

    Counts "Figure N", "Fig. N", or markdown image links.
    A figure_reference_rate < 1.0 means the report generated figures
    it did not discuss.
    """
    if total_figures == 0:
        return None
    # Match "Figure 1", "Fig. 1", "fig1", "fig_1", markdown ![]() links
    patterns = [
        r"\bFig(?:ure)?\.?\s*\d+",
        r"\bfig\d+",
        r"!\[.*?\]\(",
    ]
    references = set()
    for pattern in patterns:
        for m in re.finditer(pattern, report_text, re.IGNORECASE):
            references.add(m.group(0).lower().strip())

    # Cap at total_figures — one reference per figure is the ideal
    referenced = min(len(references), total_figures)
    return round(referenced / total_figures, 4)


def compute_quantitative_grounding_rate(report_text: str) -> Optional[float]:
    """Fraction of analytical paragraphs containing at least one number.

    Measures how data-grounded the narrative is.
    Paragraphs in the intro/methods are excluded (heuristic: only paragraphs
    after the first section header containing '##').
    """
    if not report_text:
        return None

    # Find the start of analytical content (after the introduction)
    # Use a heuristic: skip until we hit the first ## section that looks
    # like analysis (contains 'result', 'finding', 'analysis', 'data', etc.)
    lines = report_text.split("\n")
    analytical_start = 0
    for i, line in enumerate(lines):
        if line.startswith("## ") and any(
            kw in line.lower()
            for kw in ("result", "finding", "data", "analysis",
                       "discussion", "quality", "material", "method")
        ):
            analytical_start = i
            break

    analytical_text = "\n".join(lines[analytical_start:])
    paragraphs = [
        p.strip() for p in analytical_text.split("\n\n")
        if len(p.strip()) > 50  # skip short fragments
        and not p.strip().startswith("#")  # skip headers
        and not p.strip().startswith("|")  # skip tables
    ]

    if not paragraphs:
        return None

    # A paragraph is "quantitative" if it contains at least one number
    _num_re = re.compile(r"\b\d+[\.,]?\d*\b")
    grounded = sum(1 for p in paragraphs if _num_re.search(p))
    return round(grounded / len(paragraphs), 4)


def compute_report_metrics(
    run_dir: Path, dataset_name: str
) -> Dict[str, Optional[float]]:
    """Compute all deterministic report-level metrics for one dataset.

    Returns dict with keys:
        data_coverage_rate, figure_reference_rate, quantitative_grounding_rate
    """
    # dataset_name may include a file extension (e.g. "foo.parquet") from
    # judge_input.json, but filesystem paths use the stem only.
    dataset_stem = Path(dataset_name).stem
    report_text = _get_report_text(run_dir, dataset_stem)
    if report_text is None:
        logger.warning(
            "No report found for dataset %s in %s", dataset_name, run_dir
        )
        return {
            "data_coverage_rate": None,
            "figure_reference_rate": None,
            "quantitative_grounding_rate": None,
        }

    column_names = _get_column_names(run_dir, dataset_stem)
    total_figures = _count_report_figures(run_dir, dataset_stem)

    return {
        "data_coverage_rate": compute_data_coverage_rate(
            report_text, column_names
        ),
        "figure_reference_rate": compute_figure_reference_rate(
            report_text, total_figures
        ),
        "quantitative_grounding_rate": compute_quantitative_grounding_rate(
            report_text
        ),
    }


# ── High-level ingestion entry point ─────────────────────────────────────────

def ingest_run_deterministic(run_record: Any, db: Any) -> None:
    """Ingest all deterministic metrics for a single run.

    Idempotent — safe to call multiple times.

    Args:
        run_record: RunRecord instance (from registry).
        db: EvalDB instance.
    """
    run_id = run_record.run_id
    run_dir = Path(run_record.run_dir)

    # 1. Gate metrics from debug/
    ingest_gate_metrics(run_dir, run_id, db)

    # 2. Report-level metrics — update structural_metrics rows
    for dataset_name in run_record.datasets:
        metrics = compute_report_metrics(run_dir, dataset_name)

        # Read existing structural_metrics for this run/dataset to
        # preserve existing counts
        existing = db.get_structural_metrics(
            run_id=run_id, dataset_name=dataset_name
        )
        if existing:
            row = existing[0]
            db.upsert_structural_metrics(
                run_id=run_id,
                dataset_name=dataset_name,
                plot_count=row.get("plot_count") or 0,
                verified_claims_count=row.get("verified_claims_count") or 0,
                cv_gaps_count=row.get("cv_gaps_count") or 0,
                stages={
                    "cleaning": bool(row.get("stages_cleaning")),
                    "analysis": bool(row.get("stages_analysis")),
                    "cross_validation": bool(
                        row.get("stages_cross_validation")
                    ),
                    "report": bool(row.get("stages_report")),
                },
                data_coverage_rate=metrics["data_coverage_rate"],
                figure_reference_rate=metrics["figure_reference_rate"],
                quantitative_grounding_rate=metrics[
                    "quantitative_grounding_rate"
                ],
            )
        else:
            # No existing row — insert with zeros for counts
            db.upsert_structural_metrics(
                run_id=run_id,
                dataset_name=dataset_name,
                plot_count=0,
                verified_claims_count=0,
                cv_gaps_count=0,
                stages={},
                data_coverage_rate=metrics["data_coverage_rate"],
                figure_reference_rate=metrics["figure_reference_rate"],
                quantitative_grounding_rate=metrics[
                    "quantitative_grounding_rate"
                ],
            )
        logger.info(
            "Deterministic report metrics for %s/%s: %s",
            run_id, dataset_name, metrics,
        )
