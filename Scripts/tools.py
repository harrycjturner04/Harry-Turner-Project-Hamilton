"""Utility functions for the biologics analysis pipeline.

No AG2/autogen dependencies — pure Python, pandas, and standard library only.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger("captain_pipeline")


# ──────────────────────────────────────────────────────────────────────
# Gated review data structures
# ──────────────────────────────────────────────────────────────────────


class Severity(str, Enum):
    """Severity of a failed quality check."""
    MUST_FIX = "must_fix"
    SHOULD_FIX = "should_fix"


class CheckCategory(str, Enum):
    """Category of a quality check — determines routing for corrective action."""
    STRUCTURAL = "structural"
    CONTENT_QUALITY = "content_quality"
    PLOT_QUALITY = "plot_quality"
    CLAIM_EVIDENCE = "claim_evidence"


@dataclass
class CheckResult:
    """Result of a single quality check (structural, content, plot, or claim)."""
    name: str                                # e.g. "interpretation_depth", "min_plots"
    passed: bool
    severity: Optional[Severity] = None      # None when passed=True
    category: CheckCategory = CheckCategory.STRUCTURAL
    detail: str = ""
    fix_instruction: str = ""
    ref: Optional[str] = None                # finding/plot identifier


@dataclass
class StageVerdict:
    """Aggregated verdict for a stage attempt — all checks from all evaluators."""
    stage: str
    attempt: int
    structural_checks: List[CheckResult] = field(default_factory=list)
    content_checks: List[CheckResult] = field(default_factory=list)
    plot_checks: List[CheckResult] = field(default_factory=list)
    claim_checks: List[CheckResult] = field(default_factory=list)
    overall_passed: bool = False
    gate_decision: str = "retry"  # "pass" | "retry" | "pass_with_warnings"
    evaluators_ran: Dict[str, bool] = field(default_factory=dict)
    # e.g. {"structural": True, "content": True, "plot_quality": False}

    def must_fix_failures(self) -> List["CheckResult"]:
        all_c = (self.structural_checks + self.content_checks
                 + self.plot_checks + self.claim_checks)
        return [c for c in all_c if not c.passed and c.severity == Severity.MUST_FIX]

    def should_fix_failures(self) -> List["CheckResult"]:
        all_c = (self.structural_checks + self.content_checks
                 + self.plot_checks + self.claim_checks)
        return [c for c in all_c if not c.passed and c.severity == Severity.SHOULD_FIX]

    def plot_only_failures(self) -> List["CheckResult"]:
        return [c for c in self.plot_checks
                if not c.passed and c.severity == Severity.MUST_FIX]

    def content_failures(self) -> List["CheckResult"]:
        return [c for c in (self.structural_checks + self.content_checks)
                if not c.passed and c.severity == Severity.MUST_FIX]


@dataclass
class GateResult:
    """Deterministic gate decision based on a StageVerdict."""
    status: str                              # "passed" | "passed_degraded" | "failed"
    verdict: StageVerdict
    warnings: List[str] = field(default_factory=list)
    retry_tier: Optional[str] = None         # "full" | "plot_fix" | "finding_fix" | "gap_fill" | None
    retry_instructions: str = ""
    quality_score: Optional[float] = None    # WP-2: numeric quality score for convergence
    # WP-C2: Targeted refinement directives
    refinement_directives: List["RefinementDirective"] = field(default_factory=list)
    quality_breakdown: Optional["QualityScoreBreakdown"] = None  # WP-C1: per-dimension scores
    # Pass C: Regression firewall — set when a retry caused net regression
    net_regression: bool = False              # True if more checks regressed than improved
    regressed_checks: List[str] = field(default_factory=list)  # names of PASS→FAIL checks


# ──────────────────────────────────────────────────────────────────────
# WP-C2: Targeted refinement data structures
# ──────────────────────────────────────────────────────────────────────


class RefinementScope(str, Enum):
    """Scope of a targeted refinement action — from most targeted to broadest."""
    PLOT_FIX = "plot_fix"            # fix specific plots (existing)
    ALIGNMENT_FIX = "alignment_fix"  # deterministic figure_ref patching
    FINDING_FIX = "finding_fix"      # regenerate specific findings
    GAP_FILL = "gap_fill"            # add missing analytical dimension
    FULL_RERUN = "full"              # full CaptainAgent re-run (existing)


@dataclass
class RefinementDirective:
    """A targeted refinement action to be attempted before full retry."""
    scope: RefinementScope
    target_ref: Optional[str] = None       # e.g. "finding_3", "07_heatmap.png"
    check_results: List[CheckResult] = field(default_factory=list)
    instructions: str = ""                 # human-readable fix instructions
    budget: int = 1                        # max attempts for this directive


@dataclass
class IssueTracker:
    """Track per-issue persistence across retry attempts.

    If an issue persists for ``max_consecutive`` consecutive attempts
    without resolution, it is downgraded from MUST_FIX to SHOULD_FIX
    to prevent infinite cycling.
    """
    issue_name: str
    first_seen_attempt: int
    consecutive_count: int = 1
    max_consecutive: int = 2
    resolved: bool = False

    @property
    def should_downgrade(self) -> bool:
        return self.consecutive_count >= self.max_consecutive and not self.resolved


@dataclass
class QualityScoreBreakdown:
    """Per-dimension quality scores for enhanced trajectory tracking."""
    overall: float = 1.0       # weighted composite (current behaviour)
    structural: float = 1.0    # structural checks only
    content: float = 1.0       # content quality checks only
    visual: float = 1.0        # plot quality checks only


# ──────────────────────────────────────────────────────────────────────
# WP-2: Numeric quality scoring for convergence control
# ──────────────────────────────────────────────────────────────────────

# Expected criteria counts per stage (from rubric definitions in prompts.py)
_EXPECTED_CRITERIA_COUNT: Dict[str, int] = {
    "cleaning": 4,        # justification, conservatism, preservation, informativeness
    "analysis": 7,        # per_group_depth, domain_methods, plot_diversity,
                          # domain_interpretation, statistical_rigor,
                          # domain_contribution, context_coverage
    "cross_validation": 3, # recomputation, tolerance, claim_selection
}


def finding_text(finding) -> str:
    """Extract plain text from a finding (dict or legacy string).

    Findings may be plain strings (legacy format) or dicts with ``text``
    and ``figure_ref`` keys (new schema).  This helper normalises both.
    """
    if isinstance(finding, dict):
        return finding.get("text", str(finding))
    return str(finding)


def _score_check_list(checks: List[CheckResult]) -> float:
    """Score a list of checks: passed=1.0, should_fix=0.5, must_fix=0.0."""
    if not checks:
        return 1.0
    total = 0.0
    for c in checks:
        if c.passed:
            total += 1.0
        elif c.severity == Severity.SHOULD_FIX:
            total += 0.5
        # MUST_FIX → 0.0
    return round(total / len(checks), 4)


def compute_quality_score(
    verdict: StageVerdict,
    evaluators_ran: Optional[Dict[str, bool]] = None,
) -> float:
    """Compute a numeric quality score (0.0–1.0) from a StageVerdict.

    Scoring:
      - passed check = 1.0
      - should_fix check = 0.5
      - must_fix check = 0.0
    Structural checks get 1x weight, content checks get 2x weight.
    VLM plot checks get 2x weight normalised by the number of criteria
    per plot: each plot's complete VLM evaluation contributes exactly 2.0
    to the total weight regardless of how many sub-criteria were assessed.
    This prevents VLM criterion counts (11–12 per plot × N plots) from
    numerically swamping the 7 content criteria and masking content failures.

    If *evaluators_ran* is provided and key LLM/VLM evaluators were skipped,
    a coverage penalty is applied (-0.05 per skipped evaluator) to prevent
    inflated scores from incomplete evaluation.

    Returns the weighted mean across all checks. Returns 1.0 if no checks exist.
    Must-fix enforcement is handled by the quality gate's explicit must-fix
    check — the score reflects actual quality distribution across dimensions.
    """
    all_checks = (
        verdict.structural_checks
        + verdict.content_checks
        + verdict.plot_checks
        + verdict.claim_checks
    )
    if not all_checks:
        return 1.0

    # Pre-compute per-plot criterion counts for VLM normalisation.
    # Group PLOT_QUALITY checks by their ref (plot filename).  Checks without
    # a ref are grouped under the sentinel None and treated as a single plot.
    from collections import Counter as _Counter
    _plot_criteria_counts: dict = _Counter(
        c.ref for c in all_checks if c.category == CheckCategory.PLOT_QUALITY
    )
    _n_plot_refs = len(_plot_criteria_counts) or 1

    total_weight = 0.0
    weighted_sum = 0.0
    for check in all_checks:
        if check.category == CheckCategory.PLOT_QUALITY:
            # Normalise: the whole VLM evaluation of one plot = 2.0 weight.
            # Each criterion within a plot gets weight = 2.0 / criteria_count.
            _criteria_for_this_plot = _plot_criteria_counts.get(check.ref, 1)
            weight = 2.0 / _criteria_for_this_plot
        elif check.category == CheckCategory.CONTENT_QUALITY:
            weight = 2.0
        else:
            weight = 1.0

        if check.passed:
            score = 1.0
        elif check.severity == Severity.SHOULD_FIX:
            score = 0.5
        else:  # MUST_FIX or unknown
            score = 0.0

        weighted_sum += weight * score
        total_weight += weight

    raw_score = round(weighted_sum / total_weight, 4) if total_weight > 0 else 1.0

    # Coverage penalty: if LLM/VLM evaluators were expected but did not run,
    # the score is inflated by their absence.  Apply -0.05 per skipped
    # evaluator to prevent convergent early-stop from accepting incomplete
    # evaluation.
    if evaluators_ran:
        _EXPECTED_EVALUATORS = {"content", "analytical_depth", "plot_quality"}
        skipped = [
            e for e in _EXPECTED_EVALUATORS
            if evaluators_ran.get(e) is False
        ]
        if skipped:
            coverage_penalty = 0.05 * len(skipped)
            raw_score = max(0.0, raw_score - coverage_penalty)

    return raw_score


def compute_quality_breakdown(
    verdict: StageVerdict,
    evaluators_ran: Optional[Dict[str, bool]] = None,
) -> QualityScoreBreakdown:
    """Compute per-dimension quality scores for enhanced trajectory tracking."""
    # Cosmetic plot checks always pass (passed=True) and dilute the visual score,
    # inflating it even when MUST_FIX visual failures are present.  Score only on
    # scientific_validity and statistical_completeness criteria.
    _non_cosmetic_plot_checks = [
        c for c in verdict.plot_checks
        if not c.detail.startswith("[cosmetic]")
    ]
    return QualityScoreBreakdown(
        overall=compute_quality_score(verdict, evaluators_ran=evaluators_ran),
        structural=_score_check_list(verdict.structural_checks),
        content=_score_check_list(verdict.content_checks),
        visual=_score_check_list(_non_cosmetic_plot_checks),
    )


def evaluator_coverage_ratio(
    content_checks: List[CheckResult],
    stage_name: str,
) -> float:
    """Compute the fraction of expected rubric criteria that the evaluator returned.

    Returns criteria_returned / criteria_expected. Returns 1.0 if the stage
    has no expected criteria count defined.
    """
    expected = _EXPECTED_CRITERIA_COUNT.get(stage_name, 0)
    if expected == 0:
        return 1.0
    returned = len(content_checks)
    return round(min(returned / expected, 1.0), 4)


# ──────────────────────────────────────────────────────────────────────
# Think-token handling (Qwen3 models)
# ──────────────────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_tokens(text: str) -> str:
    """Remove ``<think>…</think>`` blocks emitted by Qwen3-family models."""
    if not isinstance(text, str):
        return text
    cleaned = _THINK_RE.sub("", text).strip()
    return cleaned if cleaned else text


# ──────────────────────────────────────────────────────────────────────
# JSON utilities
# ──────────────────────────────────────────────────────────────────────


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def extract_json_payload(text: str) -> str:
    """Extract the first JSON object or array from *text* (fenced or bare)."""
    if text is None:
        raise ValueError("Empty response")
    if isinstance(text, dict):
        return json.dumps(text)

    # Fast path: if the stripped text is already valid JSON (object or array),
    # return it directly.  This correctly handles JSON arrays like [{...}, ...].
    stripped = text.strip()
    if stripped and stripped[0] in ("[", "{"):
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            pass

    # Try fenced blocks (handle both objects and arrays)
    if "```" in text:
        fences = text.split("```")
        for i in range(1, len(fences), 2):
            block = fences[i].strip()
            if block.startswith("json"):
                block = block[4:].strip()
            if (block.startswith("{") and block.endswith("}")) or \
               (block.startswith("[") and block.endswith("]")):
                return block

    # Bracket-matching for JSON arrays
    depth = 0
    start = None
    for idx, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "]":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start : idx + 1]

    # Brace-matching fallback for JSON objects
    depth = 0
    start = None
    for idx, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start : idx + 1]
    return text


def parse_json_tolerant(text: str) -> Dict[str, Any] | None:
    """Parse JSON from *text* tolerantly; return ``None`` on failure."""
    try:
        return json.loads(extract_json_payload(text))
    except Exception:
        return None


def safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def tolerant_json_payload(
    text: str, fallback_key: str = "raw_text"
) -> Dict[str, Any]:
    parsed = parse_json_tolerant(text)
    return parsed if isinstance(parsed, dict) else {fallback_key: text or ""}


def tolerant_validation_payload(text: str) -> Dict[str, Any]:
    parsed = parse_json_tolerant(text)
    if not isinstance(parsed, dict):
        text_preview = (text or "")[:200].replace("\n", " ")
        logger.warning(
            "tolerant_validation_payload: JSON parse failed. "
            "type=%s, len=%d, preview='%s'",
            type(text).__name__, len(text or ""), text_preview,
        )
        return {
            "ok": False,
            "issues": ["invalid_or_missing_json"],
            "should_retry": False,
            "retry_instructions": "",
            "parse_failure_preview": text_preview,
        }
    return {
        "ok": bool(parsed.get("ok", False)),
        "issues": (
            _ensure_list(parsed.get("issues"))
            or _ensure_list(parsed.get("issue"))
            or []
        ),
        "should_retry": bool(parsed.get("should_retry", False)),
        "retry_instructions": str(parsed.get("retry_instructions", "")),
        "raw": parsed,
    }


# ──────────────────────────────────────────────────────────────────────
# File I/O
# ──────────────────────────────────────────────────────────────────────


def list_inputs(input_dir: str | Path) -> List[str]:
    """Return sorted list of parquet file paths under *input_dir*."""
    path = Path(input_dir)
    if path.is_file():
        return [str(path)]
    if not path.exists():
        return []
    files = sorted(path.glob("*.parquet")) + sorted(path.glob("*.pq"))
    return [str(p) for p in files]


def load_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if ext in {".csv", ".txt"}:
        return pd.read_csv(p)
    raise ValueError(f"Unsupported extension: {ext}")


def inspect_table(path: str | Path, max_rows: int = 5) -> Dict[str, Any]:
    """Lightweight table inspection — columns, dtypes, missingness, preview."""
    result: Dict[str, Any] = {
        "data_path": str(path),
        "columns": [],
        "dtypes": {},
        "missingness": {},
        "preview_rows": [],
        "numeric_columns": [],
        "numeric_describe": {},
        "row_count": 0,
        "error": None,
    }
    try:
        df = load_table(path)
    except Exception as exc:
        result["error"] = f"inspect failed: {exc}"
        return result

    result["columns"] = list(df.columns)
    result["dtypes"] = {col: str(dt) for col, dt in df.dtypes.items()}
    result["row_count"] = int(len(df))

    missing = df.isna().sum()
    result["missingness"] = {
        col: {
            "missing_count": int(missing[col]),
            "missing_pct": round(float(missing[col]) / len(df), 4) if len(df) else 0.0,
        }
        for col in df.columns
    }

    result["preview_rows"] = json.loads(df.head(max_rows).to_json(orient="records"))

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    result["numeric_columns"] = numeric_cols
    if numeric_cols:
        # Cap to 15 most-variable columns to keep payload small
        cols = numeric_cols[:15]
        result["numeric_describe"] = df[cols].describe().to_dict()

    return result


def write_text(path: str | Path, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return str(p)


# ──────────────────────────────────────────────────────────────────────
# Domain detection
# ──────────────────────────────────────────────────────────────────────


CHROM_KEYWORDS = frozenset({
    "retention_time", "rt", "peak_area", "peak_height",
    "absorbance", "mau", "au", "elution_volume", "wavelength",
    "chromatogram", "hplc", "sec", "iex", "hic",
})

MS_KEYWORDS = frozenset({
    "m/z", "mz", "charge_state", "charge", "molecular_weight",
    "mw", "da", "dalton", "tic", "bpc", "spectrum", "scan",
    "mass_spec", "deconvolution", "intensity",
})

# Confidence thresholds for domain detection
_STRONG_CONFIDENCE = 0.2   # ≥20% of keywords match → strong match
_AMBIGUOUS_FLOOR = 0.05    # 5-20% → ambiguous, may benefit from LLM


def detect_data_domains(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Inspect column names to suggest which domain experts are relevant.

    Returns a dict with boolean domain flags, confidence scores, and an
    ``ambiguous`` flag indicating when keyword detection is inconclusive.
    """
    columns_lower = [c.lower() for c in (evidence.get("columns") or [])]
    col_text = " ".join(columns_lower)

    chrom_hits = sum(1 for kw in CHROM_KEYWORDS if kw in col_text)
    ms_hits = sum(1 for kw in MS_KEYWORDS if kw in col_text)
    chrom_confidence = chrom_hits / len(CHROM_KEYWORDS) if CHROM_KEYWORDS else 0
    ms_confidence = ms_hits / len(MS_KEYWORDS) if MS_KEYWORDS else 0

    has_chrom = chrom_confidence >= _AMBIGUOUS_FLOOR
    has_ms = ms_confidence >= _AMBIGUOUS_FLOOR

    # Ambiguous when at least one domain has a weak (non-zero) signal
    # but neither reaches the strong threshold
    max_conf = max(chrom_confidence, ms_confidence)
    ambiguous = _AMBIGUOUS_FLOOR <= max_conf < _STRONG_CONFIDENCE

    row_count = evidence.get("row_count", 0)
    n_numeric = len(evidence.get("numeric_columns", []))

    return {
        "chromatography": has_chrom,
        "mass_spectrometry": has_ms,
        "statistics": n_numeric >= 2,
        "ml_modeling": row_count > 30 and n_numeric >= 3,
        "general_eda": not has_chrom and not has_ms,
        "has_run_groups": any(k in columns_lower for k in ("run_no", "run", "run_id")),
        "has_stage_groups": "chromatography_stage" in columns_lower,
        "has_sample_groups": "sample_code" in columns_lower,
        # Confidence scoring (additive fields)
        "chromatography_confidence": chrom_confidence,
        "mass_spectrometry_confidence": ms_confidence,
        "ambiguous": ambiguous,
    }


# ──────────────────────────────────────────────────────────────────────
# Context / metadata loading
# ──────────────────────────────────────────────────────────────────────

MAX_CONTEXT_CHARS = 4000
MAX_METADATA_CHARS = 4000


def trim_text(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 50].rstrip() + "\n...[truncated]"


def _summarize_evidence_for_query(evidence: Dict[str, Any], raw_path: Path) -> str:
    columns = (evidence.get("columns") or [])[:20]
    dtypes = evidence.get("dtypes") or {}
    dtype_pairs = [f"{k}:{v}" for k, v in list(dtypes.items())[:12]]
    return " | ".join(
        filter(None, [
            f"file={raw_path.name}",
            f"columns={', '.join(columns)}",
            f"dtypes={', '.join(dtype_pairs)}",
        ])
    )


def load_context_text(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {"context_path": None, "context_text": "", "context_error": "not_provided"}
    if not path.exists():
        return {"context_path": str(path), "context_text": "", "context_error": "not_found"}
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return {
            "context_path": str(path),
            "context_text": trim_text(text, MAX_CONTEXT_CHARS),
            "context_error": None,
        }
    except Exception as exc:
        return {"context_path": str(path), "context_text": "", "context_error": str(exc)}


def build_metadata_context(
    evidence: Dict[str, Any],
    raw_path: Path,
    metadata_db_dir: Optional[Path],
    top_k: int = 6,
) -> Dict[str, Any]:
    if metadata_db_dir is None:
        return {"metadata_context": [], "metadata_context_text": ""}
    if not metadata_db_dir.exists():
        return {"metadata_context": [], "metadata_context_text": ""}
    try:
        from database import search_metadata

        query = _summarize_evidence_for_query(evidence, raw_path)
        matches = search_metadata(query, metadata_db_dir, top_k=top_k)
        if matches is None or matches.empty:
            return {"metadata_context": [], "metadata_context_text": ""}
        cols = [c for c in ("source_file", "text_representation") if c in matches.columns]
        if not cols:
            cols = list(matches.columns[:6])
        records = matches[cols].to_dict(orient="records")
        context_text = "\n".join(
            trim_text(str(r.get("text_representation", "")), 500) for r in records
        )
        return {
            "metadata_context": records,
            "metadata_context_text": trim_text(context_text, MAX_METADATA_CHARS),
        }
    except Exception as exc:
        logger.warning("Metadata search failed: %s", exc)
        return {"metadata_context": [], "metadata_context_text": ""}


# ──────────────────────────────────────────────────────────────────────
# Group summary and artifact management
# ──────────────────────────────────────────────────────────────────────

KNOWN_GROUP_COLUMNS = ["run_no", "run", "chromatography_stage", "Sample_Code", "column"]

MAX_ARTIFACT_BYTES = 500_000  # 500 KB


def get_group_summary(
    parquet_path: str | Path,
    extra_columns: List[str] | None = None,
) -> Dict[str, Any]:
    """Return a compact summary of data groups in a parquet file.

    Detects known grouping columns and reports unique values and counts
    for each.  Does NOT return per-row data.

    Args:
        parquet_path: Path to the parquet file.
        extra_columns: Additional column names (e.g. from context
            ``grouping_columns``) to include alongside KNOWN_GROUP_COLUMNS.
    """
    try:
        df = pd.read_parquet(Path(parquet_path))
    except Exception as exc:
        return {"error": str(exc), "total_rows": 0, "groups": {}}

    # Merge known columns with any context-configured extras (deduped, order-preserving)
    cols_to_check: List[str] = list(KNOWN_GROUP_COLUMNS)
    if extra_columns:
        seen = set(cols_to_check)
        for c in extra_columns:
            if c not in seen:
                cols_to_check.append(c)
                seen.add(c)

    summary: Dict[str, Any] = {"total_rows": len(df), "groups": {}}
    for col in cols_to_check:
        if col in df.columns:
            vc = df[col].dropna().value_counts()
            _nan_count = int(df[col].isna().sum())
            summary["groups"][col] = {
                "unique_count": int(vc.shape[0]),
                "nan_count": _nan_count,
                "nan_pct": round(float(_nan_count / len(df) * 100), 1) if len(df) > 0 else 0.0,
                "values": {str(k): int(v) for k, v in vc.head(30).items()},
            }
    return summary


def load_splits_manifest(splits_dir: str | Path) -> Dict[str, Any]:
    """Load splits_manifest.json from *splits_dir*; return empty dict on failure."""
    manifest_path = Path(splits_dir) / "splits_manifest.json"
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text("utf-8"))
        except Exception:
            pass
    return {}


def _truncate_json_values(data: Any, max_list_len: int = 100) -> Any:
    """Recursively truncate lists in a JSON-like structure."""
    if isinstance(data, dict):
        return {k: _truncate_json_values(v, max_list_len) for k, v in data.items()}
    if isinstance(data, list):
        if len(data) > max_list_len:
            return data[:max_list_len] + [f"... truncated {len(data) - max_list_len} items"]
        return [_truncate_json_values(item, max_list_len) for item in data]
    return data


def cap_artifact_size(path: Path) -> bool:
    """Truncate an oversized JSON artifact in-place.  Returns True if truncated."""
    if not path.exists():
        return False
    size = path.stat().st_size
    if size <= MAX_ARTIFACT_BYTES:
        return False
    try:
        data = json.loads(path.read_text("utf-8"))
        truncated = _truncate_json_values(data, max_list_len=100)
        if isinstance(truncated, dict):
            truncated["_truncated"] = True
            truncated["_original_size_bytes"] = size
        path.write_text(json.dumps(truncated, indent=2, default=str), "utf-8")
        logger.info("Truncated oversized artifact %s (%d → %d bytes)", path.name, size, path.stat().st_size)
        return True
    except Exception as exc:
        logger.warning("Failed to truncate artifact %s: %s", path, exc)
        return False


# ──────────────────────────────────────────────────────────────────────
# Token budget management
# ──────────────────────────────────────────────────────────────────────


def estimate_tokens(text: str) -> int:
    """Rough estimate: ~3.2 characters per token for structured JSON.

    The previous 4-char heuristic overestimated token count by ~20-30%%
    for JSON payloads, triggering premature truncation.
    """
    return int(len(text) / 3.2)


def trim_payload_to_budget(
    payload: Dict[str, Any], max_input_tokens: int = 55_000
) -> Dict[str, Any]:
    """Progressively trim payload fields to fit within token budget.

    Default budget of 55K leaves headroom for system
    prompt (~2K), agent library (~3K), conversation history (~60K), and
    format suffixes (~2K) within the model's 131K+ effective context window.

    Trimming order (least → most analytically valuable):
      0. metadata_context
      1. evidence bulk (preview_rows, numeric_describe)
      2. context_text (shorten)
      3. cleaning_summary subfields
      4. data_profile.phase_b (recoverable from parquet)
      5. group_summary values (top 10)
      6. analysis_contexts (multi-angle guidance — preserve as long as possible)
      7. instructions (hard-truncate to 12K chars — last resort)

    Returns a (potentially trimmed) copy of *payload*.
    """
    text = json.dumps(payload, default=str)
    initial_est = estimate_tokens(text)
    if initial_est <= max_input_tokens:
        return payload

    trimmed = dict(payload)  # shallow copy
    _levels_hit: list[str] = []

    def _fits() -> bool:
        return estimate_tokens(json.dumps(trimmed, default=str)) <= max_input_tokens

    # Level 0 — remove verbose metadata records (least analytical value)
    for field in ("metadata_context", "metadata_context_text"):
        if field in trimmed:
            trimmed[field] = "[trimmed]"
    if _fits():
        _levels_hit.append("L0:metadata")
        logger.info("Payload trimming stopped at level 0 (metadata). Levels: %s", _levels_hit)
        return trimmed
    _levels_hit.append("L0:metadata")

    # Level 1 — strip evidence bulk (preview_rows, numeric_describe)
    if "evidence" in trimmed and isinstance(trimmed["evidence"], dict):
        ev = dict(trimmed["evidence"])
        ev.pop("preview_rows", None)
        ev.pop("numeric_describe", None)
        trimmed["evidence"] = ev
        if _fits():
            _levels_hit.append("L1:evidence_bulk")
            logger.info("Payload trimming stopped at level 1. Levels: %s", _levels_hit)
            return trimmed
    _levels_hit.append("L1:evidence_bulk")

    # Level 2 — shorten context text
    if isinstance(trimmed.get("context_text"), str):
        trimmed["context_text"] = trimmed["context_text"][:2000] + "...[trimmed]"
        if _fits():
            _levels_hit.append("L2:context_text")
            logger.info("Payload trimming stopped at level 2. Levels: %s", _levels_hit)
            return trimmed
    _levels_hit.append("L2:context_text")

    # Level 3 — strip bulky cleaning_summary subfields (low analytical value)
    if "cleaning_summary" in trimmed and isinstance(trimmed["cleaning_summary"], dict):
        cs = dict(trimmed["cleaning_summary"])
        for bulky_key in ("sample_values", "numeric_summary", "categorical_columns"):
            cs.pop(bulky_key, None)
        trimmed["cleaning_summary"] = cs
        if _fits():
            _levels_hit.append("L3:cleaning_summary")
            logger.info("Payload trimming stopped at level 3. Levels: %s", _levels_hit)
            return trimmed
    _levels_hit.append("L3:cleaning_summary")

    # Level 4 — strip Phase B from data_profile (distributions +
    #   technique recommendations — valuable but recoverable from parquet)
    dp = trimmed.get("data_profile")
    if isinstance(dp, dict) and "phase_b" in dp:
        dp = dict(dp)
        dp["phase_b"] = "[trimmed — recompute normality from cleaned parquet if needed]"
        trimmed["data_profile"] = dp
        if _fits():
            _levels_hit.append("L4:phase_b")
            logger.info("Payload trimming stopped at level 4. Levels: %s", _levels_hit)
            return trimmed
    _levels_hit.append("L4:phase_b")

    # Level 5 — truncate group_summary values (top 10 instead of 30)
    if "group_summary" in trimmed and isinstance(trimmed["group_summary"], dict):
        gs = dict(trimmed["group_summary"])
        for gk, gv in gs.get("groups", {}).items():
            if isinstance(gv, dict) and "values" in gv:
                gv["values"] = dict(list(gv["values"].items())[:10])
        trimmed["group_summary"] = gs
        if _fits():
            _levels_hit.append("L5:group_summary")
            logger.info("Payload trimming stopped at level 5. Levels: %s", _levels_hit)
            return trimmed
    _levels_hit.append("L5:group_summary")

    # Level 6 — strip analysis_contexts from dimensional_structure
    #   (multi-angle analysis guidance — analytically important, trim late)
    dp = trimmed.get("data_profile")
    if isinstance(dp, dict) and isinstance(dp.get("dimensional_structure"), dict):
        dp = dict(dp)
        ds = dict(dp["dimensional_structure"])
        ds.pop("analysis_contexts", None)
        dp["dimensional_structure"] = ds
        trimmed["data_profile"] = dp
        if _fits():
            _levels_hit.append("L6:analysis_contexts")
            logger.info("Payload trimming stopped at level 6. Levels: %s", _levels_hit)
            return trimmed
    _levels_hit.append("L6:analysis_contexts")

    # Level 7 — hard-truncate instructions (last resort — 12K chars)
    if "instructions" in trimmed and isinstance(trimmed["instructions"], str):
        trimmed["instructions"] = trimmed["instructions"][:12000] + "...[trimmed]"
    _levels_hit.append("L7:instructions")

    final_est = estimate_tokens(json.dumps(trimmed, default=str))
    logger.warning(
        "Payload trimming exhausted all levels. initial=%d tokens, final=%d tokens, "
        "budget=%d. Levels hit: %s",
        initial_est, final_est, max_input_tokens, _levels_hit,
    )

    return trimmed


# ──────────────────────────────────────────────────────────────────────
# Signal processing utilities (reference implementations for LLM agents)
# ──────────────────────────────────────────────────────────────────────


def baseline_als(y, lam: float = 1e6, p: float = 0.01, niter: int = 10):
    """Asymmetric Least Squares baseline estimation (Eilers & Boelens 2005).

    Parameters:
        y: 1-D signal array (numpy array or list of floats).
        lam: smoothness — larger values produce smoother baselines.
        p: asymmetry — smaller values ignore peaks more aggressively.
        niter: number of re-weighting iterations.
    Returns:
        Estimated baseline array (same length as *y*).
    """
    import numpy as np
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve

    y = np.asarray(y, dtype=float)
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for _ in range(niter):
        W = diags(w, 0, shape=(L, L))
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def chromatographic_resolution(t1: float, w1: float, t2: float, w2: float) -> float:
    """Chromatographic resolution between two adjacent peaks.

    Rs = 2 * |t2 - t1| / (w1 + w2)
    where t = retention time at peak max, w = peak width at base.
    Rs > 1.5 = baseline resolved, 1.0-1.5 = partial, < 1.0 = unresolved.
    """
    denom = w1 + w2
    if denom == 0:
        return float("inf")
    return 2.0 * abs(t2 - t1) / denom


def theoretical_plates(retention_time: float, peak_width_base: float) -> float:
    """USP theoretical plate count: N = 16 * (tR / W)^2."""
    if peak_width_base == 0:
        return 0.0
    return 16.0 * (retention_time / peak_width_base) ** 2


def mass_accuracy_ppm(observed: float, expected: float) -> float:
    """Mass accuracy in parts per million."""
    if expected == 0:
        return float("inf")
    return abs(observed - expected) / expected * 1e6


# ──────────────────────────────────────────────────────────────────────
# Gated review: structural gate (pure Python, no LLM)
# ──────────────────────────────────────────────────────────────────────


def structural_gate(
    stage_name: str,
    expected_paths: Dict[str, str],
    output_dir_listing: List[Dict[str, Any]],
    quality_min_plots: int = 3,
    quality_min_findings: int = 3,
    quality_require_per_group: bool = True,
    quality_png_min_bytes: int = 5000,
    require_figure_references: bool = False,
) -> List[CheckResult]:
    """Run all deterministic structural checks for a pipeline stage.

    Consolidates file-existence, JSON-validity, min-plot-count,
    per_run_per_stage presence, min-findings, and PNG-size checks
    into a single function.  No LLM call is involved.

    Returns a list of CheckResult (all with category=STRUCTURAL).
    """
    checks: List[CheckResult] = []

    # ── 1. Expected files exist and are non-trivial ──
    for key, path_str in expected_paths.items():
        p = Path(path_str)
        if not p.exists():
            checks.append(CheckResult(
                name=f"file_exists__{key}",
                passed=False,
                severity=Severity.MUST_FIX,
                category=CheckCategory.STRUCTURAL,
                detail=f"Expected file missing: {path_str}",
                fix_instruction=f"Execute code to produce {p.name} and confirm it exists on disk.",
                ref=path_str,
            ))
        elif p.is_file() and p.stat().st_size < 100:
            checks.append(CheckResult(
                name=f"file_size__{key}",
                passed=False,
                severity=Severity.MUST_FIX,
                category=CheckCategory.STRUCTURAL,
                detail=f"File too small ({p.stat().st_size} bytes): {p.name}",
                fix_instruction=f"Regenerate {p.name} with actual content.",
                ref=path_str,
            ))
        else:
            checks.append(CheckResult(
                name=f"file_exists__{key}", passed=True,
                category=CheckCategory.STRUCTURAL,
                detail=f"{p.name} exists ({p.stat().st_size} bytes)",
                ref=path_str,
            ))

    # ── 2. PNG plot count (analysis stage) ──
    if stage_name == "analysis":
        png_files = [
            item for item in output_dir_listing
            if isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and item["path"].lower().endswith(".png")
        ]
        png_count = len(png_files)
        if png_count < quality_min_plots:
            checks.append(CheckResult(
                name="min_plots",
                passed=False,
                severity=Severity.MUST_FIX,
                category=CheckCategory.STRUCTURAL,
                detail=f"Only {png_count} plots generated (minimum: {quality_min_plots})",
                fix_instruction=(
                    f"Generate at least {quality_min_plots} PNG plots answering "
                    "distinct analytical questions (overlays, boxplots, heatmaps, scatter)."
                ),
            ))
        else:
            checks.append(CheckResult(
                name="min_plots", passed=True,
                category=CheckCategory.STRUCTURAL,
                detail=f"{png_count} plots generated (minimum: {quality_min_plots})",
            ))

        # ── 3. PNG size checks ──
        for item in png_files:
            size = item.get("size", 0) or 0
            fname = Path(item["path"]).name
            if size < quality_png_min_bytes:
                checks.append(CheckResult(
                    name=f"png_size__{fname}",
                    passed=False,
                    severity=Severity.MUST_FIX,
                    category=CheckCategory.STRUCTURAL,
                    detail=f"Plot '{fname}' is {size} bytes (minimum: {quality_png_min_bytes})",
                    fix_instruction=f"Regenerate '{fname}' — current file is likely corrupt or empty.",
                    ref=fname,
                ))

    # ── 4. analysis_summary.json content checks (analysis stage) ──
    asp = expected_paths.get("analysis_summary_path", "")
    if stage_name == "analysis" and asp:
        asp_path = Path(asp)
        if asp_path.exists():
            try:
                data = json.loads(asp_path.read_text("utf-8"))

                # findings count
                findings = data.get("findings", [])
                if len(findings) < quality_min_findings:
                    checks.append(CheckResult(
                        name="min_findings",
                        passed=False,
                        severity=Severity.MUST_FIX,
                        category=CheckCategory.STRUCTURAL,
                        detail=(
                            f"Only {len(findings)} finding(s) in analysis_summary.json "
                            f"(need >= {quality_min_findings} with numeric values)"
                        ),
                        fix_instruction=(
                            f"Populate 'findings' array with >= {quality_min_findings} entries. "
                            "Format: 'Run R{{X}} shows {{N}}% deviation vs group mean of {{M}} "
                            "in metric {{K}} (p={{P}})'."
                        ),
                    ))
                else:
                    checks.append(CheckResult(
                        name="min_findings", passed=True,
                        category=CheckCategory.STRUCTURAL,
                        detail=f"{len(findings)} findings present",
                    ))

                # ── WP-3A: claim-figure traceability ──
                if require_figure_references and findings:
                    # Collect PNG filenames present on disk
                    disk_pngs = {
                        Path(item["path"]).name
                        for item in output_dir_listing
                        if isinstance(item, dict)
                        and isinstance(item.get("path"), str)
                        and item["path"].lower().endswith(".png")
                    }
                    orphaned = []
                    for idx, f in enumerate(findings):
                        if isinstance(f, dict):
                            # Accept both 'figure_ref' (new) and 'figure' (legacy)
                            fig_ref = f.get("figure_ref") or f.get("figure", "")
                            if not fig_ref:
                                orphaned.append(f"finding[{idx}]: missing 'figure_ref' key")
                            elif fig_ref not in disk_pngs:
                                # Also try stem matching (ref may omit .png)
                                ref_stem = Path(fig_ref).stem.lower()
                                disk_stems = {Path(p).stem.lower() for p in disk_pngs}
                                if ref_stem not in disk_stems:
                                    orphaned.append(
                                        f"finding[{idx}]: references '{fig_ref}' "
                                        "which does not exist on disk"
                                    )
                        # If findings are plain strings (baseline format),
                        # skip traceability — no breakage when disabled
                    if orphaned:
                        checks.append(CheckResult(
                            name="figure_references",
                            passed=False,
                            severity=Severity.SHOULD_FIX,
                            category=CheckCategory.STRUCTURAL,
                            detail=(
                                f"{len(orphaned)} finding(s) have missing or "
                                f"invalid figure references: {'; '.join(orphaned[:5])}"
                            ),
                            fix_instruction=(
                                "Each finding in analysis_summary.json should be a dict with "
                                "a 'figure_ref' key pointing to an existing PNG filename. "
                                "Format: {\"text\": \"...\", \"figure_ref\": \"01_overlay.png\"}"
                            ),
                        ))
                    else:
                        # All dict-format findings have valid references
                        dict_count = sum(1 for f in findings if isinstance(f, dict))
                        if dict_count > 0:
                            checks.append(CheckResult(
                                name="figure_references", passed=True,
                                category=CheckCategory.STRUCTURAL,
                                detail=f"{dict_count} findings have valid figure references",
                            ))

                # per_run_per_stage / per_group key
                has_prps = bool(data.get("per_run_per_stage"))
                has_pg = bool(data.get("per_group"))
                if quality_require_per_group and not has_prps and not has_pg:
                    checks.append(CheckResult(
                        name="per_run_per_stage_present",
                        passed=False,
                        severity=Severity.MUST_FIX,
                        category=CheckCategory.STRUCTURAL,
                        detail="analysis_summary.json missing per_run_per_stage/per_group key",
                        fix_instruction=(
                            "Add 'per_run_per_stage' key with one entry per run x stage "
                            "combination: {'R1__E5': {'peak_count': N, 'max_uv_280': V, ...}}."
                        ),
                    ))
                else:
                    checks.append(CheckResult(
                        name="per_run_per_stage_present", passed=True,
                        category=CheckCategory.STRUCTURAL,
                        detail="per_run_per_stage or per_group key present",
                    ))

                # domain_reasoning completeness is handled by StructuralCritic._check_domain_reasoning()

            except json.JSONDecodeError:
                checks.append(CheckResult(
                    name="json_valid__analysis_summary",
                    passed=False,
                    severity=Severity.MUST_FIX,
                    category=CheckCategory.STRUCTURAL,
                    detail="analysis_summary.json is not valid JSON",
                    fix_instruction="Rewrite analysis_summary.json as valid JSON with default=str.",
                ))

    # ── 5. cleaning_summary.json content checks (cleaning stage) ──
    sp = expected_paths.get("summary_path", "")
    if stage_name == "cleaning" and sp:
        sp_path = Path(sp)
        if sp_path.exists():
            try:
                data = json.loads(sp_path.read_text("utf-8"))
                if "rows_before" not in data and "rows_after" not in data:
                    checks.append(CheckResult(
                        name="cleaning_row_counts",
                        passed=False,
                        severity=Severity.MUST_FIX,
                        category=CheckCategory.STRUCTURAL,
                        detail="cleaning_summary.json missing row counts (rows_before/rows_after)",
                        fix_instruction=(
                            "Include 'rows_before' and 'rows_after' keys in cleaning_summary.json."
                        ),
                    ))
                else:
                    checks.append(CheckResult(
                        name="cleaning_row_counts", passed=True,
                        category=CheckCategory.STRUCTURAL,
                        detail="Row counts present in cleaning summary",
                    ))
            except json.JSONDecodeError:
                checks.append(CheckResult(
                    name="json_valid__cleaning_summary",
                    passed=False,
                    severity=Severity.MUST_FIX,
                    category=CheckCategory.STRUCTURAL,
                    detail="cleaning_summary.json is not valid JSON",
                    fix_instruction="Rewrite cleaning_summary.json as valid JSON.",
                ))

    return checks


# ──────────────────────────────────────────────────────────────────────
# Gated review: quality gate (deterministic decision logic)
# ──────────────────────────────────────────────────────────────────────


def quality_gate(
    verdict: StageVerdict,
    max_retries: int,
    previous_verdict: Optional[StageVerdict] = None,
    sf_accumulation_threshold: int = 4,
    stall_count: int = 0,
    issue_stall_max_consecutive: int = 2,
) -> GateResult:
    """Deterministic gate decision based on aggregated check results.

    - No must_fix failures → passed (unless SHOULD_FIX accumulation threshold hit)
    - Only should_fix → passed with warnings
    - must_fix but attempt >= max_retries → passed_degraded
    - Stall detection: same must_fix as previous attempt → passed_degraded
    - Plot-only must_fix → retry_tier="plot_fix"
    - Content/structural must_fix → retry_tier="full"

    WP-2: Also computes and attaches a numeric quality_score to the GateResult.
    """
    must_fix = verdict.must_fix_failures()
    should_fix = verdict.should_fix_failures()
    _evaluators_ran = getattr(verdict, "evaluators_ran", None)
    _quality_score = compute_quality_score(verdict, evaluators_ran=_evaluators_ran)

    _quality_breakdown = compute_quality_breakdown(verdict, evaluators_ran=_evaluators_ran)

    # Helper: attach quality_score and breakdown before returning any GateResult
    def _make_result(**kwargs: Any) -> GateResult:
        result = GateResult(**kwargs)
        result.quality_score = _quality_score
        result.quality_breakdown = _quality_breakdown
        return result

    # All checks passed — but check if evaluators actually ran.
    # Content and plot quality evaluate orthogonal dimensions:
    #   content  → analysis JSON correctness (per_group_depth, findings, etc.)
    #   plot_quality → visual correctness of PNG figures
    # A VLM plot pass does NOT substitute for a content evaluator pass.
    if not must_fix and not should_fix:
        ran = verdict.evaluators_ran
        coverage_warnings: List[str] = []
        content_ran = ran.get("content", True) if ran else True
        plot_ran = ran.get("plot_quality", True) if ran else True

        if not content_ran:
            coverage_warnings.append(
                "Content evaluator did not run or produced no results — "
                "content quality not verified"
            )
        if not plot_ran:
            coverage_warnings.append(
                "Plot quality evaluator did not run"
            )

        # If content evaluator specifically did not run → passed_degraded.
        # Content is the more important dimension — it verifies that the
        # analysis JSON has correct metrics, which the VLM cannot assess.
        if not content_ran:
            verdict.overall_passed = True
            verdict.gate_decision = "pass_with_warnings"
            return _make_result(
                status="passed_degraded", verdict=verdict,
                warnings=coverage_warnings,
            )

        # Content ran; pass with any plot-coverage warnings
        verdict.overall_passed = True
        verdict.gate_decision = "pass" if not coverage_warnings else "pass_with_warnings"
        return _make_result(
            status="passed", verdict=verdict,
            warnings=coverage_warnings,
        )

    # Only advisory issues — but check accumulation threshold first.
    # Too many SHOULD_FIX issues collectively indicate inadequate quality.
    # PLOT_QUALITY SHOULD_FIX are excluded from this count: they are noisy
    # VLM aesthetic opinions on individual plots (one per plot), and with
    # large plot counts they will always exceed any reasonable threshold.
    # Plot-quality issues are already handled by the MUST_FIX / plot-fix
    # path when they are severe; SHOULD_FIX plot opinions are advisory only.
    if not must_fix:
        non_plot_sf = [
            c for c in should_fix
            if c.category != CheckCategory.PLOT_QUALITY
        ]
        sf_count = len(non_plot_sf)
        if sf_count >= sf_accumulation_threshold and verdict.attempt < max_retries:
            verdict.overall_passed = False
            verdict.gate_decision = "retry_accumulated_sf"
            return _make_result(
                status="failed", verdict=verdict,
                retry_tier="full",
                retry_instructions=_build_retry_text(non_plot_sf),
                warnings=[
                    f"{sf_count} non-plot SHOULD_FIX issues accumulated "
                    f"(threshold: {sf_accumulation_threshold}; "
                    f"{len(should_fix) - sf_count} plot-quality SHOULD_FIX excluded)"
                ],
            )
        # Below threshold or budget exhausted — pass with warnings
        verdict.overall_passed = True
        verdict.gate_decision = "pass_with_warnings"
        return _make_result(
            status="passed", verdict=verdict,
            warnings=[c.detail for c in should_fix],
        )

    # Must-fix failures exist but retry budget exhausted
    if verdict.attempt >= max_retries:
        verdict.overall_passed = True
        verdict.gate_decision = "pass_with_warnings"
        return _make_result(
            status="passed_degraded", verdict=verdict,
            warnings=[
                f"Max retries ({max_retries}) reached with {len(must_fix)} "
                "outstanding must_fix issue(s)"
            ] + [c.detail for c in must_fix],
        )

    # Stall detection: same must_fix checks failed across consecutive attempts.
    # Only accept degraded after issue_stall_max_consecutive identical failures
    # (tracked via stall_count passed in from run_stage_gated).
    if stall_count >= issue_stall_max_consecutive:
        verdict.overall_passed = True
        verdict.gate_decision = "pass_with_warnings"
        return _make_result(
            status="passed_degraded", verdict=verdict,
            warnings=[
                f"Stall detected: identical must_fix failures for "
                f"{stall_count} consecutive attempts (threshold: "
                f"{issue_stall_max_consecutive})"
            ] + [c.detail for c in must_fix],
        )

    # ── Pass C: Regression detection ──
    # Compare against previous verdict to detect net regression (more checks
    # went PASS→FAIL than FAIL→PASS).  This signals the retry loop to
    # consider rolling back artifacts rather than continuing with worse output.
    _net_regression = False
    _regressed_checks: List[str] = []
    if previous_verdict is not None:
        _prev_all = (previous_verdict.structural_checks + previous_verdict.content_checks
                     + previous_verdict.plot_checks + previous_verdict.claim_checks)
        _curr_all = (verdict.structural_checks + verdict.content_checks
                     + verdict.plot_checks + verdict.claim_checks)
        _prev_passed = {c.name for c in _prev_all if c.passed}
        _prev_failed = {c.name for c in _prev_all if not c.passed}
        _curr_passed = {c.name for c in _curr_all if c.passed}
        _curr_failed = {c.name for c in _curr_all if not c.passed}
        _regressed = _prev_passed & _curr_failed
        _improved = _prev_failed & _curr_passed
        if _regressed and len(_regressed) > len(_improved):
            _net_regression = True
            _regressed_checks = sorted(_regressed)

    # Route to appropriate retry tier and build refinement directives
    plot_only = verdict.plot_only_failures()
    content = verdict.content_failures()

    # WP-C2: Build targeted refinement directives (most targeted first)
    directives: List[RefinementDirective] = []

    # 1. Plot-specific fixes
    if plot_only:
        directives.append(RefinementDirective(
            scope=RefinementScope.PLOT_FIX,
            check_results=plot_only,
            instructions=_build_retry_text(plot_only),
        ))

    # 1.5 Plot-finding alignment fix (deterministic, no LLM)
    _alignment_fixes = [
        c for c in must_fix
        if c.name == "exec__plot_finding_alignment"
    ]
    if _alignment_fixes:
        directives.append(RefinementDirective(
            scope=RefinementScope.ALIGNMENT_FIX,
            check_results=_alignment_fixes,
            instructions=_build_retry_text(_alignment_fixes),
        ))

    # 2. Finding-specific fixes (content MUST_FIX — exclude alignment which
    #    needs a structural patch, not an LLM finding rewrite)
    _finding_fixes = [
        c for c in must_fix
        if c.category == CheckCategory.CONTENT_QUALITY
        and c.name != "exec__plot_finding_alignment"
    ]
    if _finding_fixes:
        directives.append(RefinementDirective(
            scope=RefinementScope.FINDING_FIX,
            check_results=_finding_fixes,
            instructions=_build_retry_text(_finding_fixes),
        ))

    # 3. Analytical gap fills (depth critic checks with name prefix "depth__")
    _gap_fills = [
        c for c in must_fix
        if c.name.startswith("depth__")
    ]
    if _gap_fills:
        directives.append(RefinementDirective(
            scope=RefinementScope.GAP_FILL,
            check_results=_gap_fills,
            instructions=_build_retry_text(_gap_fills),
        ))

    # 4. Full re-run is always available as final escalation
    directives.append(RefinementDirective(
        scope=RefinementScope.FULL_RERUN,
        check_results=must_fix,
        instructions=_build_retry_text(must_fix),
    ))

    # Select the most targeted applicable tier
    if content:
        # Content/structural failures: check if targeted fix is possible
        if _finding_fixes:
            retry_tier = RefinementScope.FINDING_FIX.value
        else:
            retry_tier = RefinementScope.FULL_RERUN.value
        retry_text = _build_retry_text(must_fix)
    elif plot_only:
        retry_tier = RefinementScope.PLOT_FIX.value
        retry_text = _build_retry_text(plot_only)
    else:
        retry_tier = RefinementScope.FULL_RERUN.value
        retry_text = _build_retry_text(must_fix)

    verdict.overall_passed = False
    verdict.gate_decision = "retry"
    _result = _make_result(
        status="failed", verdict=verdict,
        retry_tier=retry_tier,
        retry_instructions=retry_text,
        refinement_directives=directives,
    )
    _result.net_regression = _net_regression
    _result.regressed_checks = _regressed_checks
    return _result


_CATEGORY_PRIORITY = {
    CheckCategory.STRUCTURAL: 0,
    CheckCategory.CONTENT_QUALITY: 1,
    CheckCategory.PLOT_QUALITY: 2,
}

_MAX_RETRY_ITEMS = 10

# Pass C: Check redundancy map — when a higher-priority check is present,
# the lower-priority check's retry instruction is suppressed to avoid
# contradictory guidance.  Key = suppressed check, value = set of checks
# that subsume it.
_CHECK_SUBSUMPTION: Dict[str, set] = {
    "depth__bare_deviations": {"domain_interpretation"},
    "depth__missing_group_comparison": {"per_group_depth", "statistical_rigor"},
    "exec__finding_data_mismatch": {"domain_interpretation"},
}


def _build_retry_text(failures: List[CheckResult]) -> str:
    """Build structured retry instructions from a list of failing checks.

    Caps to the top 10 most critical issues (structural > content > visual)
    and instructs the agent to preserve passing aspects.  Pass C: suppresses
    redundant checks when a higher-priority check covers the same dimension.
    """
    if not failures:
        return ""

    # Prioritise: structural > content > visual, then by severity
    sorted_failures = sorted(
        failures,
        key=lambda c: (
            _CATEGORY_PRIORITY.get(c.category, 99),
            0 if c.severity == Severity.MUST_FIX else 1,
        ),
    )

    # Pass C: Deduplicate — suppress checks that are subsumed by
    # higher-priority checks already in the failure set.
    _present_names = {c.name for c in sorted_failures}
    _deduplicated = []
    _suppressed = []
    for c in sorted_failures:
        _subsumers = _CHECK_SUBSUMPTION.get(c.name, set())
        if _subsumers & _present_names:
            _suppressed.append(c.name)
            continue
        _deduplicated.append(c)
    if _suppressed:
        import logging as _logging
        _logging.getLogger("captain_pipeline").debug(
            "Retry text: suppressed redundant checks %s (subsumed by present checks)",
            _suppressed,
        )

    top = _deduplicated[:_MAX_RETRY_ITEMS]
    omitted = len(_deduplicated) - len(top)

    lines = [f"FOCUS: Address these {len(top)} critical issues. "
             "All other aspects were acceptable — preserve them."]
    for i, f in enumerate(top, 1):
        lines.append(f"  {i}. [{f.name}] {f.fix_instruction}")
        if f.ref:
            lines.append(f"     Affected: {f.ref}")
    if omitted:
        lines.append(f"\n({omitted} lower-priority issues omitted — "
                     "fix the above first.)")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Gated review: serialization helpers
# ──────────────────────────────────────────────────────────────────────


def check_to_dict(c: CheckResult) -> Dict[str, Any]:
    """Serialize a CheckResult for JSON output."""
    return {
        "name": c.name,
        "passed": c.passed,
        "severity": c.severity.value if c.severity else None,
        "category": c.category.value,
        "detail": c.detail,
        "fix_instruction": c.fix_instruction,
        "ref": c.ref,
    }


def verdict_to_dict(v: StageVerdict) -> Dict[str, Any]:
    """Serialize a StageVerdict for JSON output."""
    return {
        "stage": v.stage,
        "attempt": v.attempt,
        "overall_passed": v.overall_passed,
        "gate_decision": v.gate_decision,
        "evaluators_ran": dict(v.evaluators_ran),
        "structural_checks": [check_to_dict(c) for c in v.structural_checks],
        "content_checks": [check_to_dict(c) for c in v.content_checks],
        "plot_checks": [check_to_dict(c) for c in v.plot_checks],
        "claim_checks": [check_to_dict(c) for c in v.claim_checks],
    }


def gate_to_dict(g: GateResult) -> Dict[str, Any]:
    """Serialize a GateResult for JSON output."""
    d: Dict[str, Any] = {
        "status": g.status,
        "retry_tier": g.retry_tier,
        "retry_instructions": g.retry_instructions[:500] if g.retry_instructions else "",
        "warnings": g.warnings,
        "quality_score": g.quality_score,
    }
    # WP-C2: Include refinement directives
    if g.refinement_directives:
        d["refinement_directives"] = [
            {
                "scope": rd.scope.value if hasattr(rd.scope, "value") else str(rd.scope),
                "target_ref": rd.target_ref,
                "instructions": rd.instructions[:300],
                "budget": rd.budget,
                "check_count": len(rd.check_results),
            }
            for rd in g.refinement_directives
        ]
    # WP-C1: Include quality breakdown
    if g.quality_breakdown:
        d["quality_breakdown"] = {
            "overall": g.quality_breakdown.overall,
            "structural": g.quality_breakdown.structural,
            "content": g.quality_breakdown.content,
            "visual": g.quality_breakdown.visual,
        }
    return d
