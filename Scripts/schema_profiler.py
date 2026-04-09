"""Schema-aware data profiler for the biologics analysis pipeline.

WP-1A: Column role classification and grouping inference.
WP-1B: Distribution profiling and technique recommendations.

Pure Python + pandas + scipy — no LLM calls. Produces a deterministic data
profile that enables adaptive analysis, grouping, and technique selection
based on dataset structure rather than hardcoded column names or keyword
matching.

Toggle: ``run_config.schema_profiling`` in context.md
  - "disabled": bypass profiler, use legacy inspect_table + get_group_summary
  - "roles_only": Phase A only (column roles + grouping inference)
  - "full": Phase A + B (distributions + technique recommendations)
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from itertools import combinations, product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("captain_pipeline")

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

# Cardinality thresholds for column role classification
_MAX_CATEGORICAL_UNIQUE = 50        # columns with <= this many unique values → candidate group
_CATEGORICAL_RATIO_CEIL = 0.05      # unique/row_count must be below this for categorical
_IDENTIFIER_RATIO_FLOOR = 0.95      # unique/row_count above this → likely identifier
_METADATA_TEXT_MIN_MEDIAN_LEN = 20  # median string length above this → metadata text

# Grouping inference
_MIN_GROUPS = 2                     # minimum useful group count (2 = binary/pairwise designs)
_MAX_GROUPS = 500                   # maximum before grouping becomes unwieldy
_MIN_ROWS_PER_GROUP = 5             # each group should have at least this many rows
_IDEAL_GROUP_RANGE = (5, 200)       # preferred range for group count scoring
_MAX_GROUPING_COMBO_COLUMNS = 3     # max columns in a compound grouping key
_CANDIDATE_NULL_CEILING = 0.8       # columns below this null% are eligible as grouping candidates
_COVERAGE_PENALTY_WEIGHT = 0.15     # score penalty weight for partial-coverage grouping columns

# Ordinal detection — patterns indicating ordered/sequential values
_ORDINAL_PATTERNS = [
    re.compile(r"^[A-Za-z]*\d+$"),              # E1, E2, stage1, step3
    re.compile(r"^(step|stage|phase|round|pass|cycle|day|hour|time[_\s]?point)\s*\d*$", re.I),
    re.compile(r"^\d+(\.\d+)?$"),                # pure numeric strings (orderable)
]

# Phase B: Distribution profiling
_NORMALITY_ALPHA = 0.05          # significance level for normality tests
_MIN_SAMPLES_NORMALITY = 20     # D'Agostino-Pearson requires n >= 20
_MIN_SAMPLES_SHAPIRO = 8        # Shapiro-Wilk minimum
_KDE_GRID_POINTS = 512          # resolution for KDE-based modality detection
_PEAK_PROMINENCE_FACTOR = 0.10  # fraction of max density for peak detection
_HIGH_SKEW_THRESHOLD = 1.0      # |skewness| above this → suggest transform
_EXTREME_SKEW_THRESHOLD = 2.0   # |skewness| above this → log transform


# ──────────────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ColumnProfile:
    """Profile for a single column."""
    name: str
    dtype: str
    role: str  # identifier | categorical_group | ordinal_stage | continuous_measurement | metadata_text | datetime | constant
    cardinality: int
    cardinality_ratio: float  # unique_values / row_count
    null_count: int
    null_pct: float
    # Populated for categorical/ordinal columns
    sample_values: List[str] = field(default_factory=list)
    value_counts_top10: Dict[str, int] = field(default_factory=dict)
    # Populated for continuous columns
    stats: Optional[Dict[str, float]] = None  # min, max, mean, std, median
    # Phase A+: Semantic purpose (populated by dimensional structure analysis)
    semantic_purpose: str = "uncategorised"


@dataclass
class ColumnRelationship:
    """Pairwise relationship between two discrete columns."""
    col_a: str
    col_b: str
    relationship: str  # "redundant" | "nested_a_in_b" | "nested_b_in_a" | "crossed" | "partial_overlap"
    strength: float    # 0-1
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DimensionDescriptor:
    """A single dimension in the dataset's structural hierarchy."""
    name: str                           # Human-readable dimension name (canonical column)
    columns: List[str]                  # Column(s) encoding this dimension (first = canonical)
    cardinality: int
    semantic_purpose: str               # experimental_unit, experimental_condition, process_phase, etc.
    nesting_parent: Optional[str] = None  # Name of the dimension this nests within (None = outermost)
    redundant_with: List[str] = field(default_factory=list)
    sample_values: List[str] = field(default_factory=list)


@dataclass
class AnalysisContext:
    """A suggested analytical context: a combination of dimensions for a class of questions."""
    name: str                           # e.g. "process_trend"
    description: str                    # Human-readable description
    group_by: List[str]                 # Columns to group by
    compare_across: str                 # The dimension being compared
    hold_fixed: List[str]              # Dimensions held constant
    expected_groups: int
    use_case: str                       # "comparison" | "trend" | "replicate_assessment"


@dataclass
class DimensionalStructure:
    """Complete dimensional structure of the dataset."""
    dimensions: List[DimensionDescriptor] = field(default_factory=list)
    relationships: List[ColumnRelationship] = field(default_factory=list)
    redundancy_groups: List[List[str]] = field(default_factory=list)
    hierarchy_levels: List[str] = field(default_factory=list)  # dimension names, coarsest → finest
    analysis_contexts: List[AnalysisContext] = field(default_factory=list)
    recommended_default_grouping: Optional["GroupingCandidate"] = None


@dataclass
class GroupingCandidate:
    """A candidate compound grouping key with quality scores."""
    columns: List[str]
    group_count: int
    min_group_size: int
    median_group_size: float
    size_cv: float  # coefficient of variation of group sizes (lower = more even)
    score: float    # composite quality score (higher = better)
    rows_covered_pct: float  # % of rows in non-null groups


@dataclass
class DistributionProfile:
    """Distribution characteristics for a continuous column (Phase B)."""
    skewness: float
    kurtosis: float                       # excess kurtosis (Fisher definition)
    normality_stat: Optional[float]       # test statistic (D'Agostino or Shapiro)
    normality_pvalue: Optional[float]     # p-value from normality test
    normality_test: str                   # "dagostino" | "shapiro" | "insufficient_data"
    is_normal: bool                       # True if p > _NORMALITY_ALPHA
    modality: str                         # "unimodal" | "bimodal" | "multimodal"
    n_modes: int                          # number of detected density peaks
    suggested_transform: Optional[str]    # "log" | "sqrt" | None


@dataclass
class TechniqueRecommendation:
    """A recommended analytical technique based on data characteristics (Phase B)."""
    technique: str        # e.g. "Kruskal-Wallis", "one-way ANOVA"
    rationale: str        # why this technique fits the data
    columns: List[str]    # which columns it applies to
    priority: str         # "primary" | "exploratory"


@dataclass
class DataProfile:
    """Complete data profile for a cleaned dataset."""
    # Phase A: Column roles and grouping
    file_path: str
    row_count: int
    column_count: int
    columns: List[ColumnProfile] = field(default_factory=list)
    grouping_candidates: List[GroupingCandidate] = field(default_factory=list)
    recommended_grouping: Optional[GroupingCandidate] = None
    # Extended grouping: a user-directed finer grouping for the subset
    # of rows where extension columns are non-null.
    extended_grouping: Optional[GroupingCandidate] = None
    # Role summaries for quick access
    identifiers: List[str] = field(default_factory=list)
    categorical_groups: List[str] = field(default_factory=list)
    ordinal_stages: List[str] = field(default_factory=list)
    continuous_measurements: List[str] = field(default_factory=list)
    metadata_text_cols: List[str] = field(default_factory=list)
    datetime_cols: List[str] = field(default_factory=list)
    constant_cols: List[str] = field(default_factory=list)
    # Phase B: Distributions and technique recommendations
    distributions: Dict[str, DistributionProfile] = field(default_factory=dict)
    technique_recommendations: List[TechniqueRecommendation] = field(default_factory=list)
    # Phase A+: Dimensional structure (relationships, hierarchy, analysis contexts)
    dimensional_structure: Optional[DimensionalStructure] = None
    # Profiler warnings (e.g. context override rejected)
    profiler_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to dict for JSON output and payload injection."""
        return asdict(self)

    def to_compact_dict(self) -> Dict[str, Any]:
        """Compact representation for LLM payload injection.

        Omits per-column stats and value_counts to keep payload small.
        Phase B content (distributions, technique_recommendations) is
        included under a separate ``"phase_b"`` key so that
        ``trim_payload_to_budget()`` can strip it first when under
        pressure — Phase A is always retained.
        """
        result: Dict[str, Any] = {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "column_roles": {},
            "grouping_candidates": [],
            "recommended_grouping": None,
        }

        # Column roles as compact role → [col_names] mapping
        role_map: Dict[str, List[str]] = {}
        for col in self.columns:
            role_map.setdefault(col.role, []).append(col.name)
        result["column_roles"] = role_map

        # Top-5 grouping candidates (compact) — expanded from 3 so
        # agents see alternative groupings beyond the recommended one.
        for gc in self.grouping_candidates[:5]:
            result["grouping_candidates"].append({
                "columns": gc.columns,
                "group_count": gc.group_count,
                "median_group_size": round(gc.median_group_size, 1),
                "score": round(gc.score, 3),
                "rows_covered_pct": gc.rows_covered_pct,
            })

        if self.recommended_grouping:
            result["recommended_grouping"] = {
                "columns": self.recommended_grouping.columns,
                "group_count": self.recommended_grouping.group_count,
                "score": round(self.recommended_grouping.score, 3),
            }

        if self.extended_grouping:
            result["extended_grouping"] = {
                "columns": self.extended_grouping.columns,
                "group_count": self.extended_grouping.group_count,
                "rows_covered_pct": self.extended_grouping.rows_covered_pct,
                "note": (
                    f"For the {self.extended_grouping.rows_covered_pct}% of rows "
                    f"where {[c for c in self.extended_grouping.columns if c not in (self.recommended_grouping.columns if self.recommended_grouping else [])]} "
                    f"are non-null, use this finer grouping for sub-analysis."
                ),
            }

        # Profiler warnings (context override rejection, partial override notes)
        if self.profiler_warnings:
            result["profiler_warnings"] = self.profiler_warnings

        # Phase A+ — dimensional structure (under dedicated key for trimming)
        if self.dimensional_structure:
            ds = self.dimensional_structure
            result["dimensional_structure"] = {
                "dimensions": [
                    {
                        "name": d.name,
                        "columns": d.columns,
                        "cardinality": d.cardinality,
                        "purpose": d.semantic_purpose,
                        "nesting_parent": d.nesting_parent,
                    }
                    for d in ds.dimensions
                ],
                "redundancy_groups": ds.redundancy_groups,
                "hierarchy": ds.hierarchy_levels,
                "analysis_contexts": [
                    {
                        "name": ac.name,
                        "description": ac.description,
                        "group_by": ac.group_by,
                        "compare_across": ac.compare_across,
                        "expected_groups": ac.expected_groups,
                        "use_case": ac.use_case,
                    }
                    for ac in ds.analysis_contexts
                ],
            }

        # Phase B — distributions and technique recommendations
        # Stored under a dedicated key for tiered trimming.
        if self.distributions or self.technique_recommendations:
            phase_b: Dict[str, Any] = {}
            if self.distributions:
                phase_b["distributions"] = {
                    col_name: {
                        "skewness": round(dp.skewness, 3),
                        "kurtosis": round(dp.kurtosis, 3),
                        "is_normal": dp.is_normal,
                        "modality": dp.modality,
                        "n_modes": dp.n_modes,
                        "suggested_transform": dp.suggested_transform,
                    }
                    for col_name, dp in self.distributions.items()
                }
            if self.technique_recommendations:
                phase_b["technique_recommendations"] = [
                    {
                        "technique": tr.technique,
                        "rationale": tr.rationale,
                        "columns": tr.columns,
                        "priority": tr.priority,
                    }
                    for tr in self.technique_recommendations
                ]
            result["phase_b"] = phase_b

        return result


# ──────────────────────────────────────────────────────────────────────
# Phase A: Column role classification
# ──────────────────────────────────────────────────────────────────────

def _classify_column_role(
    series: pd.Series,
    col_name: str,
    row_count: int,
) -> str:
    """Classify a single column's role based on structural properties.

    Returns one of: identifier, categorical_group, ordinal_stage,
    continuous_measurement, metadata_text, datetime, constant.
    """
    if row_count == 0:
        return "constant"

    # Handle datetime columns
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    non_null = series.dropna()
    n_unique = non_null.nunique()

    # Constant column (0 or 1 unique value)
    if n_unique <= 1:
        return "constant"

    ratio = n_unique / row_count if row_count > 0 else 0.0

    # Numeric columns
    if pd.api.types.is_numeric_dtype(series):
        # Low-cardinality numeric → likely categorical group or ordinal
        if n_unique <= _MAX_CATEGORICAL_UNIQUE and ratio < _CATEGORICAL_RATIO_CEIL:
            if _is_ordinal(non_null):
                return "ordinal_stage"
            return "categorical_group"
        # High-cardinality numeric with near-unique values → possible identifier
        if ratio >= _IDENTIFIER_RATIO_FLOOR and pd.api.types.is_integer_dtype(series):
            return "identifier"
        # Default for numeric: continuous measurement
        if non_null.std() > 0:
            return "continuous_measurement"
        return "constant"

    # String / object columns
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        # Try parsing as datetime
        if _looks_like_datetime(non_null):
            return "datetime"

        # Near-unique strings → identifier or metadata
        if ratio >= _IDENTIFIER_RATIO_FLOOR:
            # Short strings → identifier; long strings → metadata
            median_len = non_null.astype(str).str.len().median()
            if median_len >= _METADATA_TEXT_MIN_MEDIAN_LEN:
                return "metadata_text"
            return "identifier"

        # Low-cardinality strings → categorical group or ordinal
        if n_unique <= _MAX_CATEGORICAL_UNIQUE and ratio < _CATEGORICAL_RATIO_CEIL:
            if _is_ordinal(non_null):
                return "ordinal_stage"
            return "categorical_group"

        # Medium-cardinality strings
        median_len = non_null.astype(str).str.len().median()
        if median_len >= _METADATA_TEXT_MIN_MEDIAN_LEN:
            return "metadata_text"

        # Fallback: treat medium-cardinality short strings as categorical
        if n_unique <= _MAX_CATEGORICAL_UNIQUE:
            return "categorical_group"
        return "metadata_text"

    # Fallback for other dtypes (bool, category, etc.)
    if pd.api.types.is_bool_dtype(series):
        return "categorical_group"

    return "metadata_text"


def _is_ordinal(series: pd.Series) -> bool:
    """Heuristic: does a low-cardinality series look like ordered stages?

    Checks if the unique values match ordinal naming patterns (E1, E2, step_1, etc.)
    or are numeric and sequential.
    """
    unique_vals = series.unique()
    if len(unique_vals) < 2:
        return False

    # If numeric, check if values are roughly sequential integers
    if pd.api.types.is_numeric_dtype(series):
        try:
            sorted_vals = sorted(unique_vals)
            diffs = [sorted_vals[i+1] - sorted_vals[i] for i in range(len(sorted_vals) - 1)]
            if all(d > 0 for d in diffs):
                return True
        except (TypeError, ValueError):
            pass
        return False

    # For strings, check against ordinal patterns
    str_vals = [str(v) for v in unique_vals[:20]]  # sample up to 20
    matches = sum(
        1 for v in str_vals
        if any(pat.match(v.strip()) for pat in _ORDINAL_PATTERNS)
    )
    return matches / len(str_vals) >= 0.6


def _looks_like_datetime(series: pd.Series) -> bool:
    """Quick heuristic: do the string values look like dates/times?"""
    sample = series.dropna().head(10).astype(str)
    if len(sample) == 0:
        return False
    try:
        parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
        return parsed.notna().sum() / len(sample) >= 0.7
    except Exception:
        return False


def _profile_column(series: pd.Series, col_name: str, row_count: int) -> ColumnProfile:
    """Build a ColumnProfile for a single column."""
    role = _classify_column_role(series, col_name, row_count)
    non_null = series.dropna()
    n_unique = non_null.nunique()
    null_count = int(series.isna().sum())

    profile = ColumnProfile(
        name=col_name,
        dtype=str(series.dtype),
        role=role,
        cardinality=n_unique,
        cardinality_ratio=round(n_unique / row_count, 4) if row_count > 0 else 0.0,
        null_count=null_count,
        null_pct=round(null_count / row_count, 4) if row_count > 0 else 0.0,
    )

    # Add sample values for categorical/ordinal columns
    if role in ("categorical_group", "ordinal_stage"):
        vc = non_null.value_counts()
        profile.sample_values = [str(v) for v in vc.index[:10]]
        profile.value_counts_top10 = {str(k): int(v) for k, v in vc.head(10).items()}

    # Add stats for continuous columns
    if role == "continuous_measurement":
        desc = non_null.describe()
        profile.stats = {
            "min": float(desc.get("min", 0)),
            "max": float(desc.get("max", 0)),
            "mean": float(desc.get("mean", 0)),
            "std": float(desc.get("std", 0)),
            "median": float(desc.get("50%", 0)),
        }

    return profile


# ──────────────────────────────────────────────────────────────────────
# Phase A: Grouping inference
# ──────────────────────────────────────────────────────────────────────

def _score_grouping(
    group_count: int,
    min_group_size: int,
    median_group_size: float,
    size_cv: float,
    rows_covered_pct: float,
    *,
    discriminability: float = 0.0,
    has_ordinal: bool = False,
    n_columns: int = 1,
) -> float:
    """Score a candidate grouping on a 0-1 scale.

    Higher = better. Considers:
    - group_count in ideal range (2-200)
    - minimum group size (>= _MIN_ROWS_PER_GROUP)
    - discriminability: CV of measurement means across groups (high = good —
      the grouping separates the data along a dimension with meaningful
      variation). Falls back to evenness (low size_cv) when no measurement
      data is available.
    - row coverage (high = good)
    - ordinal bonus: groupings containing ordinal stages receive a 0.15
      bonus (capped at 1.0) because ordinal dimensions typically encode
      process phases with analytically meaningful progression.
    - dimensionality bonus: multi-column groupings receive +0.10 per
      additional column beyond 1 (capped at +0.30 for 4+ columns),
      rewarding deeper analytical slicing.
    """
    score = 0.0

    # Group count score (0-0.35): prefer 5-200 groups, accept down to 2
    if _IDEAL_GROUP_RANGE[0] <= group_count <= _IDEAL_GROUP_RANGE[1]:
        score += 0.35
    elif _MIN_GROUPS <= group_count <= _MAX_GROUPS:
        # Linearly decay outside ideal range
        if group_count < _IDEAL_GROUP_RANGE[0]:
            score += 0.35 * (group_count / _IDEAL_GROUP_RANGE[0])
        else:
            decay = (group_count - _IDEAL_GROUP_RANGE[1]) / (_MAX_GROUPS - _IDEAL_GROUP_RANGE[1])
            score += 0.35 * max(0, 1.0 - decay)
    # else: 0 points for < _MIN_GROUPS or > _MAX_GROUPS

    # Minimum group size score (0-0.25)
    if min_group_size >= _MIN_ROWS_PER_GROUP:
        score += 0.25
    elif min_group_size >= 2:
        score += 0.25 * (min_group_size / _MIN_ROWS_PER_GROUP)

    # Discriminability score (0-0.20): does this grouping separate
    # the data along a dimension with meaningful measurement variation?
    # When discriminability is available (CV of group measurement means),
    # a higher CV means the grouping captures real analytical structure.
    # Falls back to evenness (inverse of size_cv) when no measurement data.
    if discriminability > 0:
        # Saturates at CV=1.0 (100% variation explained)
        disc_score = min(discriminability, 1.0)
        score += 0.20 * disc_score
    else:
        # Fallback: evenness (low size_cv is better)
        evenness = max(0, 1.0 - size_cv) if size_cv < 2.0 else 0.0
        score += 0.20 * evenness

    # Coverage score (0-0.20)
    score += 0.20 * (rows_covered_pct / 100.0)

    # Ordinal bonus: ordinal stages typically encode process phases
    # with analytically meaningful progression (e.g. chromatography
    # stages E1→E2→W1→W2). Boost these groupings.
    if has_ordinal:
        score += 0.15

    # Dimensionality bonus: multi-column groupings capture richer
    # analytical structure.  +0.10 per column beyond the first,
    # capped at +0.30 (i.e. full bonus at 4+ columns).
    if n_columns > 1:
        score += min(0.10 * (n_columns - 1), 0.30)

    return round(min(score, 1.0), 4)


def _evaluate_grouping(
    df: pd.DataFrame,
    columns: List[str],
    *,
    max_groups: int = _MAX_GROUPS,
    measurement_cols: Optional[List[str]] = None,
    column_roles: Optional[Dict[str, str]] = None,
) -> Optional[GroupingCandidate]:
    """Evaluate a specific set of columns as a compound grouping key.

    Returns None if the grouping is invalid (too few/many groups, all NaN, etc.).

    Coverage-aware: columns with partial null values are evaluated on the
    subset of rows where they are present.  The coverage percentage is
    factored into the quality score so that higher-null groupings are
    ranked lower (not eliminated) compared to fully-populated ones.

    Args:
        max_groups: Override the upper group-count limit.  The default
            ``_MAX_GROUPS`` (200) is appropriate for speculative candidate
            enumeration.  For user-directed context extensions, callers
            can pass a higher ceiling so that analytically meaningful
            but higher-cardinality groupings are not rejected.
        measurement_cols: Optional list of continuous measurement column
            names.  When provided, computes discriminability (CV of group
            means) to reward groupings that separate the data along
            dimensions with meaningful measurement variation.
        column_roles: Optional mapping of column name → role string.
            Used to detect ordinal_stage columns for the scoring bonus.
    """
    # Check all columns exist
    missing = [c for c in columns if c not in df.columns]
    if missing:
        return None

    # Drop rows where any grouping column is NaN
    subset = df[columns].dropna()
    if len(subset) == 0:
        return None

    rows_covered_pct = round(len(subset) / len(df) * 100, 1) if len(df) > 0 else 0.0

    # Count groups
    grouped = subset.groupby(columns, sort=False)
    group_sizes = grouped.size()
    group_count = len(group_sizes)

    if group_count < _MIN_GROUPS or group_count > max_groups:
        return None

    min_size = int(group_sizes.min())
    median_size = float(group_sizes.median())
    mean_size = float(group_sizes.mean())
    std_size = float(group_sizes.std()) if group_count > 1 else 0.0
    cv = std_size / mean_size if mean_size > 0 else 0.0

    # Compute discriminability: how much do measurement means vary across
    # groups?  High CV of group means = grouping separates real structure.
    discriminability = 0.0
    if measurement_cols and group_count >= 2:
        usable_mcols = [c for c in measurement_cols if c in df.columns]
        if usable_mcols:
            try:
                # Use the rows that survived the grouping-column NaN filter
                meas_df = df.loc[subset.index, usable_mcols + columns]
                group_means = meas_df.groupby(columns, sort=False)[usable_mcols].mean()
                # CV of group means for each measurement, then take the median
                disc_per_col = []
                for mc in usable_mcols:
                    gm = group_means[mc].dropna()
                    if len(gm) >= 2 and gm.mean() != 0:
                        disc_per_col.append(abs(gm.std() / gm.mean()))
                if disc_per_col:
                    discriminability = float(np.median(disc_per_col))
            except Exception:
                pass  # fall back to evenness in scorer

    # Detect if any grouping column is ordinal
    has_ordinal = False
    if column_roles:
        has_ordinal = any(column_roles.get(c) == "ordinal_stage" for c in columns)

    score = _score_grouping(
        group_count, min_size, median_size, cv, rows_covered_pct,
        discriminability=discriminability,
        has_ordinal=has_ordinal,
        n_columns=len(columns),
    )

    # Apply coverage penalty for partial-null grouping columns so that
    # higher-coverage groupings are preferred when scores are close, but
    # analytically valuable columns with moderate nulls are not discarded.
    max_col_null = max(df[c].isna().mean() for c in columns)
    if max_col_null > 0.1:
        coverage_penalty = _COVERAGE_PENALTY_WEIGHT * max_col_null
        score = round(max(score - coverage_penalty, 0.0), 4)

    return GroupingCandidate(
        columns=columns,
        group_count=group_count,
        min_group_size=min_size,
        median_group_size=round(median_size, 1),
        size_cv=round(cv, 3),
        score=score,
        rows_covered_pct=rows_covered_pct,
    )


def _infer_grouping_candidates(
    df: pd.DataFrame,
    column_profiles: List[ColumnProfile],
) -> List[GroupingCandidate]:
    """Enumerate and rank candidate compound grouping keys.

    Only considers columns classified as categorical_group or ordinal_stage.
    Evaluates all 1-, 2-, and 3-column combinations and returns them
    sorted by score (descending).

    When semantic purposes are available on the column profiles, a
    multiplier is applied so that groupings covering analytically
    meaningful dimensions rank higher than statistically equivalent
    but shallow alternatives.
    """
    # Candidate columns: categorical_group and ordinal_stage only.
    # Use relaxed null ceiling (_CANDIDATE_NULL_CEILING) so that columns
    # with moderate missingness are still evaluated as grouping candidates;
    # the coverage penalty in _evaluate_grouping() handles score adjustment.
    candidate_cols = [
        cp.name for cp in column_profiles
        if cp.role in ("categorical_group", "ordinal_stage")
        and cp.null_pct < _CANDIDATE_NULL_CEILING
    ]

    if not candidate_cols:
        return []

    # Build auxiliary data for discriminability scoring
    measurement_cols = [
        cp.name for cp in column_profiles
        if cp.role == "continuous_measurement"
    ]
    column_roles = {cp.name: cp.role for cp in column_profiles}

    candidates: List[GroupingCandidate] = []

    # Evaluate all combinations of 1, 2, and 3 columns
    for n_cols in range(1, min(_MAX_GROUPING_COMBO_COLUMNS + 1, len(candidate_cols) + 1)):
        for combo in combinations(candidate_cols, n_cols):
            gc = _evaluate_grouping(
                df, list(combo),
                measurement_cols=measurement_cols,
                column_roles=column_roles,
            )
            if gc is not None:
                # Apply semantic quality multiplier if purposes are classified
                mult = _semantic_quality_multiplier(list(combo), column_profiles)
                if mult != 1.0:
                    gc.score = round(min(gc.score * mult, 1.0), 4)
                candidates.append(gc)

    # Sort by score descending
    candidates.sort(key=lambda c: c.score, reverse=True)

    return candidates


# ──────────────────────────────────────────────────────────────────────
# Semantic grouping composition
# ──────────────────────────────────────────────────────────────────────

# Weights for each semantic purpose when scoring grouping compositions.
# Higher weight = more analytically meaningful for the primary per_group
# output.  Process phases and experimental conditions capture the
# scientific variation; experimental units (runs) are important but can
# serve as a secondary comparison dimension; technical replicates are
# the least informative for primary grouping.
_SEMANTIC_GROUPING_WEIGHTS: Dict[str, int] = {
    "process_phase": 3,
    "experimental_condition": 3,
    "experimental_unit": 2,
    "technical_replicate": 1,
}

# Purposes considered for semantic composition, in evaluation order.
_SEMANTIC_PURPOSES = list(_SEMANTIC_GROUPING_WEIGHTS.keys())

# Maximum column alternatives to try per purpose (limits combinatorial explosion).
_MAX_ALTS_PER_PURPOSE = 2


def _compose_semantic_grouping(
    df: pd.DataFrame,
    col_profiles: List[ColumnProfile],
) -> Optional[GroupingCandidate]:
    """Compose a recommended grouping from semantic purposes.

    Mimics how a domain scientist would slice the data: pick the columns
    that capture the most analytically-distinct dimensions (process phases,
    experimental conditions, experimental units) while keeping the group
    count within practical limits.

    Algorithm:
      1. Collect usable discrete columns per semantic purpose.
      2. Enumerate all valid compositions (subsets of purposes × column
         alternatives) and evaluate each via ``_evaluate_grouping``.
      3. Score each valid composition by the sum of purpose weights
         (primary) and grouping quality score (tiebreaker).
      4. Return the highest-scoring composition.

    Returns None if no columns have recognised semantic purposes.
    """
    # ── Collect usable columns per purpose ──
    # Use relaxed null ceiling so analytically important columns with
    # moderate missingness participate in semantic composition; the
    # coverage penalty in _evaluate_grouping() adjusts scores.
    purpose_cols: Dict[str, List[ColumnProfile]] = {}
    for cp in col_profiles:
        if (cp.semantic_purpose in _SEMANTIC_GROUPING_WEIGHTS
                and cp.role in ("categorical_group", "ordinal_stage")
                and cp.null_pct < _CANDIDATE_NULL_CEILING):
            purpose_cols.setdefault(cp.semantic_purpose, []).append(cp)

    # Sort each purpose by cardinality descending (most informative first)
    # and keep only top alternatives to limit search space.
    for purpose in purpose_cols:
        purpose_cols[purpose] = sorted(
            purpose_cols[purpose], key=lambda c: c.cardinality, reverse=True,
        )[:_MAX_ALTS_PER_PURPOSE]

    available = [p for p in _SEMANTIC_PURPOSES if p in purpose_cols]
    if not available:
        return None

    # Build auxiliary data for discriminability scoring
    measurement_cols = [
        cp.name for cp in col_profiles
        if cp.role == "continuous_measurement"
    ]
    column_roles = {cp.name: cp.role for cp in col_profiles}

    best_gc: Optional[GroupingCandidate] = None
    best_semantic_score = -1.0

    # Try all subsets of available purposes (from largest to smallest).
    for n_purposes in range(len(available), 0, -1):
        for purpose_combo in combinations(available, n_purposes):
            # Semantic weight of this combination
            semantic_weight = sum(_SEMANTIC_GROUPING_WEIGHTS[p] for p in purpose_combo)

            # Skip if this combination can't beat what we already have
            if best_gc is not None and semantic_weight < best_semantic_score - 1.0:
                continue

            # Try all column alternatives for the selected purposes
            alt_lists = [
                [cp.name for cp in purpose_cols[p]] for p in purpose_combo
            ]
            for col_combo in product(*alt_lists):
                gc = _evaluate_grouping(
                    df, list(col_combo),
                    measurement_cols=measurement_cols,
                    column_roles=column_roles,
                )
                if gc is None:
                    continue
                # Composite score: amplified semantic weight + quality (0-1).
                # The 1.5× multiplier ensures that analytically richer
                # groupings (more purposes) can outscore statistically
                # cleaner but shallower alternatives.
                composite = semantic_weight * 1.5 + gc.score
                if composite > best_semantic_score:
                    best_semantic_score = composite
                    best_gc = gc

    if best_gc is not None:
        logger.info(
            "Semantic grouping composed: %s (%d groups, score=%.3f)",
            best_gc.columns, best_gc.group_count, best_gc.score,
        )
    return best_gc


def _semantic_quality_multiplier(
    columns: List[str],
    col_profiles: List[ColumnProfile],
) -> float:
    """Return a scoring multiplier (0.7 – 1.5) based on the semantic
    quality of the columns in a grouping candidate.

    Groupings that span multiple analytically-distinct purposes get a
    bonus; groupings that use only low-value purposes (e.g. technical
    replicate alone) get a penalty.  The ceiling is set high enough
    (1.5 for 3+ purposes) that a rich multi-dimensional grouping can
    outscore a statistically cleaner but analytically shallow one.
    """
    profile_map = {cp.name: cp for cp in col_profiles}
    purposes_present: set[str] = set()
    for col in columns:
        cp = profile_map.get(col)
        if cp and cp.semantic_purpose in _SEMANTIC_GROUPING_WEIGHTS:
            purposes_present.add(cp.semantic_purpose)

    if not purposes_present:
        return 1.0  # no semantic info → neutral

    weight_sum = sum(_SEMANTIC_GROUPING_WEIGHTS.get(p, 0) for p in purposes_present)
    n_purposes = len(purposes_present)

    # Base multiplier from number of purposes covered.
    # Ceiling raised to 1.5 for 3+ purposes so that analytically rich
    # groupings can offset group-count scoring penalties.
    if n_purposes >= 3:
        multiplier = 1.5
    elif n_purposes == 2:
        multiplier = 1.25
    elif weight_sum >= 3:
        # Single high-value purpose (process_phase or condition)
        multiplier = 1.10
    elif weight_sum >= 2:
        # Single moderate purpose (experimental_unit)
        multiplier = 1.0
    else:
        # Only technical_replicate
        multiplier = 0.7

    return multiplier


# ──────────────────────────────────────────────────────────────────────
# Phase A+: Relationship graph and dimensional structure
# ──────────────────────────────────────────────────────────────────────

# Cardinality ceiling for pairwise relationship analysis
_RELATIONSHIP_MAX_CARDINALITY = 5000
# Fill ratio threshold distinguishing crossed from partial overlap
_CROSSING_FILL_THRESHOLD = 0.8
# CV threshold for identifying protocol metadata (highly uneven groups)
_PROTOCOL_METADATA_CV_THRESHOLD = 1.5
# Maximum cardinality for linking identifier classification
_LINKING_ID_MAX_CARDINALITY = 5000

# Name-pattern regexes for semantic purpose (soft signals, refined by structure)
_PURPOSE_PATTERNS = {
    "experimental_unit": re.compile(
        r"run|batch|lot|experiment[_\s]?(?:id|no|num)?$|campaign", re.I,
    ),
    "experimental_condition": re.compile(
        r"column|method|treatment|condition|media|resin|type|mode|format", re.I,
    ),
    "process_phase": re.compile(
        r"stage|phase|step|fraction|cycle|pass|elution|wash", re.I,
    ),
    "technical_replicate": re.compile(
        r"replicate|rep(?:_|\b)|technical|repeat", re.I,
    ),
    "linking_identifier": re.compile(
        r"sample[_\s]?(?:code|id)|unique[_\s]?(?:\w+[_\s])?id|fraction[_\s]?(?:\w+[_\s])?id|peak[_\s]?id", re.I,
    ),
}


def _detect_redundancy(
    df: pd.DataFrame, col_a: str, col_b: str,
) -> Optional[ColumnRelationship]:
    """Test whether *col_a* and *col_b* partition the data identically."""
    a, b = df[col_a], df[col_b]
    if a.nunique() != b.nunique():
        return None

    pair = df[[col_a, col_b]].dropna()
    if pair.empty:
        return None

    # Each A value maps to exactly one B value?
    a_to_b = pair.groupby(col_a, sort=False)[col_b].nunique()
    if a_to_b.max() != 1:
        return None

    # And vice versa?
    b_to_a = pair.groupby(col_b, sort=False)[col_a].nunique()
    if b_to_a.max() != 1:
        return None

    return ColumnRelationship(
        col_a=col_a, col_b=col_b,
        relationship="redundant", strength=1.0,
        details={"cardinality": int(a.nunique())},
    )


def _detect_nesting(
    df: pd.DataFrame, col_a: str, col_b: str,
) -> Optional[ColumnRelationship]:
    """Test whether one column is nested within the other (functional dependency).

    *nested_a_in_b*: every value of A appears within exactly one value of B
    (A is finer-grained, B is coarser). Equivalently: A → B.
    """
    pair = df[[col_a, col_b]].dropna()
    if pair.empty:
        return None

    a_to_b = pair.groupby(col_a, sort=False)[col_b].nunique()
    b_to_a = pair.groupby(col_b, sort=False)[col_a].nunique()

    a_determines_b = int(a_to_b.max()) == 1  # A → B
    b_determines_a = int(b_to_a.max()) == 1  # B → A

    if a_determines_b and b_determines_a:
        return None  # Redundancy, handled separately

    if a_determines_b:
        # A is finer (nested within B)
        return ColumnRelationship(
            col_a=col_a, col_b=col_b,
            relationship="nested_a_in_b", strength=1.0,
            details={
                "finer": col_a, "coarser": col_b,
                "finer_cardinality": int(a_to_b.shape[0]),
                "coarser_cardinality": int(b_to_a.shape[0]),
            },
        )

    if b_determines_a:
        # B is finer (nested within A)
        return ColumnRelationship(
            col_a=col_a, col_b=col_b,
            relationship="nested_b_in_a", strength=1.0,
            details={
                "finer": col_b, "coarser": col_a,
                "finer_cardinality": int(b_to_a.shape[0]),
                "coarser_cardinality": int(a_to_b.shape[0]),
            },
        )

    return None


def _detect_crossing(
    df: pd.DataFrame, col_a: str, col_b: str,
) -> ColumnRelationship:
    """Classify a non-nested, non-redundant relationship as crossed or partial."""
    pair = df[[col_a, col_b]].dropna()
    n_a = pair[col_a].nunique()
    n_b = pair[col_b].nunique()
    max_combos = n_a * n_b
    if max_combos == 0:
        return ColumnRelationship(
            col_a=col_a, col_b=col_b,
            relationship="partial_overlap", strength=0.0,
            details={"fill_ratio": 0.0, "observed_combos": 0},
        )

    observed = pair.drop_duplicates().shape[0]
    fill_ratio = observed / max_combos

    rel_type = "crossed" if fill_ratio >= _CROSSING_FILL_THRESHOLD else "partial_overlap"
    return ColumnRelationship(
        col_a=col_a, col_b=col_b,
        relationship=rel_type, strength=round(fill_ratio, 4),
        details={"fill_ratio": round(fill_ratio, 4), "observed_combos": observed},
    )


def _build_relationship_graph(
    df: pd.DataFrame,
    column_profiles: List[ColumnProfile],
) -> List[ColumnRelationship]:
    """Build pairwise relationship graph for all discrete columns."""
    discrete_cols = [
        cp.name for cp in column_profiles
        if cp.role in ("categorical_group", "ordinal_stage", "identifier", "metadata_text")
        and cp.cardinality < _RELATIONSHIP_MAX_CARDINALITY
        and cp.null_pct < 50.0
    ]

    relationships: List[ColumnRelationship] = []
    for i, col_a in enumerate(discrete_cols):
        for col_b in discrete_cols[i + 1:]:
            rel = _detect_redundancy(df, col_a, col_b)
            if rel is None:
                rel = _detect_nesting(df, col_a, col_b)
            if rel is None:
                rel = _detect_crossing(df, col_a, col_b)
            if rel is not None:
                relationships.append(rel)

    n_red = sum(1 for r in relationships if r.relationship == "redundant")
    n_nest = sum(1 for r in relationships if r.relationship.startswith("nested"))
    n_cross = sum(1 for r in relationships if r.relationship == "crossed")
    logger.info(
        "Relationship graph: %d relationships (%d redundant, %d nested, %d crossed)",
        len(relationships), n_red, n_nest, n_cross,
    )
    return relationships


# ── Semantic purpose classification ──────────────────────────────────

def _initial_purpose(cp: ColumnProfile) -> str:
    """Pass 1: Assign semantic purpose using per-column heuristics (name + cardinality)."""
    name_lower = cp.name.lower()

    if cp.role in ("continuous_measurement", "datetime", "constant"):
        return "uncategorised"

    # Try name-pattern matching (soft signal)
    for purpose, pattern in _PURPOSE_PATTERNS.items():
        if pattern.search(name_lower):
            # Validate: linking_identifier needs medium-high cardinality
            if purpose == "linking_identifier" and cp.cardinality < 30:
                continue
            # Validate: technical_replicate should be low cardinality
            if purpose == "technical_replicate" and cp.cardinality > 10:
                continue
            # Validate: process_phase should be low cardinality (not identifiers)
            if purpose == "process_phase" and cp.role in ("metadata_text", "identifier"):
                continue
            return purpose

    # Role-based defaults when no name pattern fires
    if cp.role == "ordinal_stage":
        return "process_phase"  # ordinal defaults to phase
    if cp.role == "identifier":
        return "linking_identifier"
    if cp.role == "metadata_text" and cp.cardinality_ratio < 0.5 and cp.cardinality > 30:
        return "linking_identifier"

    return "uncategorised"


def _refine_purposes(
    column_profiles: List[ColumnProfile],
    relationships: List[ColumnRelationship],
    df: pd.DataFrame,
) -> None:
    """Pass 2: Refine semantic purposes using the relationship graph.

    Mutates ``column_profiles[].semantic_purpose`` in place.
    """
    by_name: Dict[str, ColumnProfile] = {cp.name: cp for cp in column_profiles}

    # Build quick lookup: which columns are redundant with each other
    redundant_partners: Dict[str, List[str]] = {}
    for rel in relationships:
        if rel.relationship == "redundant":
            redundant_partners.setdefault(rel.col_a, []).append(rel.col_b)
            redundant_partners.setdefault(rel.col_b, []).append(rel.col_a)

    # Build nesting info: finer → coarser
    nested_within: Dict[str, str] = {}  # finer_col → coarser_col
    for rel in relationships:
        if rel.relationship == "nested_a_in_b":
            nested_within[rel.col_a] = rel.col_b
        elif rel.relationship == "nested_b_in_a":
            nested_within[rel.col_b] = rel.col_a

    # Build crossing info
    crossed_with: Dict[str, List[str]] = {}
    for rel in relationships:
        if rel.relationship == "crossed":
            crossed_with.setdefault(rel.col_a, []).append(rel.col_b)
            crossed_with.setdefault(rel.col_b, []).append(rel.col_a)

    # Identify outermost columns (nothing nests them within another column)
    all_discrete = {
        cp.name for cp in column_profiles
        if cp.role in ("categorical_group", "ordinal_stage")
        and cp.cardinality < _RELATIONSHIP_MAX_CARDINALITY
        and cp.null_pct < 50.0
    }
    # Columns that appear as "coarser" in a nesting relationship
    has_finer = {coarser for coarser in nested_within.values()}
    # Columns that appear as "finer" in a nesting relationship
    is_finer = set(nested_within.keys())
    outermost = all_discrete - is_finer  # not nested within anything

    for cp in column_profiles:
        if cp.semantic_purpose != "uncategorised":
            continue
        if cp.name not in all_discrete:
            continue

        # Heuristic: outermost level with moderate cardinality → experimental_unit
        if cp.name in outermost and cp.cardinality >= 2:
            # Check if this column is crossed with others at the same level
            # (crossed outermost columns are likely unit × condition)
            partners = crossed_with.get(cp.name, [])
            has_crossed_partner_with_purpose = any(
                by_name[p].semantic_purpose in ("experimental_unit", "experimental_condition")
                for p in partners if p in by_name
            )
            if has_crossed_partner_with_purpose:
                # The partner already claimed unit or condition; guess the other
                partner_purposes = {by_name[p].semantic_purpose for p in partners if p in by_name}
                if "experimental_unit" in partner_purposes:
                    cp.semantic_purpose = "experimental_condition"
                else:
                    cp.semantic_purpose = "experimental_unit"
            elif cp.cardinality <= 20:
                cp.semantic_purpose = "experimental_condition"
            else:
                cp.semantic_purpose = "experimental_unit"
            continue

        # Heuristic: column nested within identified structure → process_phase or sub-condition
        if cp.name in is_finer:
            coarser = nested_within[cp.name]
            if cp.role == "ordinal_stage":
                cp.semantic_purpose = "process_phase"
            elif cp.cardinality <= 5:
                cp.semantic_purpose = "technical_replicate"
            else:
                cp.semantic_purpose = "process_phase"
            continue

    # Protocol metadata refinement: highly uneven categorical columns.
    # This overrides assignments made by the outermost heuristic because
    # columns with very uneven group sizes (high CV) are operational
    # metadata, not experimental structure — regardless of nesting level.
    for cp in column_profiles:
        if cp.role not in ("categorical_group", "ordinal_stage"):
            continue
        if cp.name not in all_discrete:
            continue
        # Skip columns with strong name-based purpose from Pass 1
        # (those were classified by name, not by the outermost heuristic)
        if cp.semantic_purpose in ("process_phase", "technical_replicate",
                                   "linking_identifier"):
            continue
        try:
            sizes = df.groupby(cp.name, sort=False).size()
            if len(sizes) > 1:
                cv = float(sizes.std() / sizes.mean())
                if cv > _PROTOCOL_METADATA_CV_THRESHOLD:
                    cp.semantic_purpose = "protocol_metadata"
        except Exception:
            pass


# ── Dimensional structure builder ────────────────────────────────────

def _merge_redundancy_groups(
    column_profiles: List[ColumnProfile],
    relationships: List[ColumnRelationship],
) -> Tuple[List[List[str]], Dict[str, str]]:
    """Identify sets of mutually redundant columns and pick a canonical member.

    Returns (redundancy_groups, canonical_map) where canonical_map maps
    every column name to its canonical representative.
    """
    # Union-Find for redundancy
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for rel in relationships:
        if rel.relationship == "redundant":
            union(rel.col_a, rel.col_b)

    # Group by root
    groups: Dict[str, List[str]] = {}
    all_cols_in_rels = set()
    for rel in relationships:
        if rel.relationship == "redundant":
            all_cols_in_rels.add(rel.col_a)
            all_cols_in_rels.add(rel.col_b)
    for col in all_cols_in_rels:
        root = find(col)
        groups.setdefault(root, []).append(col)

    # Pick canonical: prefer shorter name, then numeric dtype
    by_name = {cp.name: cp for cp in column_profiles}
    canonical_map: Dict[str, str] = {}
    redundancy_groups: List[List[str]] = []

    for members in groups.values():
        if len(members) < 2:
            continue
        # Sort: shorter name first, then prefer numeric
        def _sort_key(c: str) -> Tuple[int, int, str]:
            cp = by_name.get(c)
            is_numeric = 0 if cp and "int" in cp.dtype or "float" in cp.dtype else 1
            return (len(c), is_numeric, c)

        members_sorted = sorted(set(members), key=_sort_key)
        canonical = members_sorted[0]
        redundancy_groups.append(members_sorted)
        for m in members_sorted:
            canonical_map[m] = canonical

    # Columns not in any redundancy group map to themselves
    for cp in column_profiles:
        if cp.name not in canonical_map:
            canonical_map[cp.name] = cp.name

    return redundancy_groups, canonical_map


def _build_hierarchy(
    dimensions: List[DimensionDescriptor],
    relationships: List[ColumnRelationship],
    canonical_map: Dict[str, str],
) -> List[str]:
    """Topological sort of dimensions by nesting depth (coarsest first).

    Returns a list of dimension names from coarsest to finest.
    """
    dim_by_name: Dict[str, DimensionDescriptor] = {d.name: d for d in dimensions}

    # Build adjacency from nesting relationships (coarser → finer)
    children: Dict[str, List[str]] = {d.name: [] for d in dimensions}
    in_degree: Dict[str, int] = {d.name: 0 for d in dimensions}

    for rel in relationships:
        if rel.relationship in ("nested_a_in_b", "nested_b_in_a"):
            finer = rel.details.get("finer", "")
            coarser = rel.details.get("coarser", "")
            # Map to canonical dimension names
            finer_dim = canonical_map.get(finer, finer)
            coarser_dim = canonical_map.get(coarser, coarser)
            if finer_dim in dim_by_name and coarser_dim in dim_by_name and finer_dim != coarser_dim:
                if finer_dim not in children.get(coarser_dim, []):
                    children.setdefault(coarser_dim, []).append(finer_dim)
                    in_degree[finer_dim] = in_degree.get(finer_dim, 0) + 1
                    # Record nesting parent
                    dim_by_name[finer_dim].nesting_parent = coarser_dim

    # Topological sort (Kahn's algorithm)
    from collections import deque
    queue: deque[str] = deque()
    for d in dimensions:
        if in_degree.get(d.name, 0) == 0:
            queue.append(d.name)

    hierarchy: List[str] = []
    while queue:
        node = queue.popleft()
        hierarchy.append(node)
        for child in children.get(node, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    # Append any dimensions not reached (disconnected from nesting)
    for d in dimensions:
        if d.name not in hierarchy:
            hierarchy.append(d.name)

    return hierarchy


def _generate_analysis_contexts(
    dimensions: List[DimensionDescriptor],
    hierarchy: List[str],
    df: pd.DataFrame,
) -> List[AnalysisContext]:
    """Generate 3-5 context-appropriate grouping strategies."""
    contexts: List[AnalysisContext] = []
    dim_by_purpose: Dict[str, List[DimensionDescriptor]] = {}
    for d in dimensions:
        dim_by_purpose.setdefault(d.semantic_purpose, []).append(d)

    units = dim_by_purpose.get("experimental_unit", [])
    conditions = dim_by_purpose.get("experimental_condition", [])
    phases = dim_by_purpose.get("process_phase", [])
    replicates = dim_by_purpose.get("technical_replicate", [])

    # Helper: count groups for a set of columns
    def _count_groups(cols: List[str]) -> int:
        valid = [c for c in cols if c in df.columns]
        if not valid:
            return 0
        try:
            return df[valid].dropna().drop_duplicates().shape[0]
        except Exception:
            return 0

    # Context 1: Condition comparison (group by unit × condition)
    if conditions and units:
        unit_col = units[0].columns[0]
        cond_col = conditions[0].columns[0]
        group_by = [unit_col, cond_col]
        contexts.append(AnalysisContext(
            name="condition_comparison",
            description=(
                f"Compare {conditions[0].name} types within each "
                f"{units[0].name}"
            ),
            group_by=group_by,
            compare_across=cond_col,
            hold_fixed=[unit_col],
            expected_groups=_count_groups(group_by),
            use_case="comparison",
        ))
    elif conditions:
        cond_col = conditions[0].columns[0]
        contexts.append(AnalysisContext(
            name="condition_comparison",
            description=f"Compare across {conditions[0].name} types",
            group_by=[cond_col],
            compare_across=cond_col,
            hold_fixed=[],
            expected_groups=_count_groups([cond_col]),
            use_case="comparison",
        ))

    # Context 2: Process trend (unit × condition × phase)
    if phases:
        phase_col = phases[0].columns[0]
        outer: List[str] = []
        hold: List[str] = []
        if units:
            outer.append(units[0].columns[0])
            hold.append(units[0].columns[0])
        if conditions:
            outer.append(conditions[0].columns[0])
            hold.append(conditions[0].columns[0])
        group_by = outer + [phase_col]
        contexts.append(AnalysisContext(
            name="process_trend",
            description=(
                f"Track {phases[0].name} progression"
                + (f" within each {' × '.join(hold)}" if hold else "")
            ),
            group_by=group_by,
            compare_across=phase_col,
            hold_fixed=hold,
            expected_groups=_count_groups(group_by),
            use_case="trend",
        ))

    # Context 3: Run / unit comparison
    if units:
        unit_col = units[0].columns[0]
        contexts.append(AnalysisContext(
            name="run_comparison",
            description=(
                f"Compare across {units[0].name} "
                "(aggregate or stratify by other dimensions)"
            ),
            group_by=[unit_col],
            compare_across=unit_col,
            hold_fixed=[],
            expected_groups=_count_groups([unit_col]),
            use_case="comparison",
        ))

    # Context 4: Replicate consistency
    if replicates:
        rep_col = replicates[0].columns[0]
        outer = []
        hold = []
        if units:
            outer.append(units[0].columns[0])
            hold.append(units[0].columns[0])
        if conditions:
            outer.append(conditions[0].columns[0])
            hold.append(conditions[0].columns[0])
        group_by = outer + [rep_col]
        contexts.append(AnalysisContext(
            name="replicate_assessment",
            description=f"Assess consistency across {replicates[0].name}",
            group_by=group_by,
            compare_across=rep_col,
            hold_fixed=hold,
            expected_groups=_count_groups(group_by),
            use_case="replicate_assessment",
        ))

    # Context 5: Stage overview (aggregate across units)
    if phases and (units or conditions):
        phase_col = phases[0].columns[0]
        contexts.append(AnalysisContext(
            name="stage_overview",
            description=(
                f"Overview of {phases[0].name} (aggregated across "
                f"{units[0].name if units else 'all data'})"
            ),
            group_by=[phase_col],
            compare_across=phase_col,
            hold_fixed=[],
            expected_groups=_count_groups([phase_col]),
            use_case="comparison",
        ))

    # Hierarchy-driven contexts: for dimensions in the hierarchy that
    # aren't already covered by the template-based contexts above.
    # The hierarchy goes from coarsest to finest. For each adjacent pair,
    # generate a "drill-down" context: group by the coarser dimension,
    # compare across the finer dimension.
    if len(hierarchy) >= 2:
        _covered_cols = set()
        for ctx in contexts:
            _covered_cols.update(ctx.group_by)
            if ctx.compare_across:
                _covered_cols.add(ctx.compare_across)

        dim_lookup = {d.name: d for d in dimensions}
        for i in range(len(hierarchy) - 1):
            coarse_name = hierarchy[i]
            fine_name = hierarchy[i + 1]
            coarse_dim = dim_lookup.get(coarse_name)
            fine_dim = dim_lookup.get(fine_name)
            if coarse_dim is None or fine_dim is None:
                continue
            coarse_col = coarse_dim.columns[0]
            fine_col = fine_dim.columns[0]
            # Only add if this pair isn't already covered
            pair = {coarse_col, fine_col}
            if pair <= _covered_cols:
                continue
            group_by = [coarse_col, fine_col]
            n_groups = _count_groups(group_by)
            if n_groups < 2:
                continue
            contexts.append(AnalysisContext(
                name=f"hierarchy_{coarse_name}_to_{fine_name}",
                description=(
                    f"Drill from {coarse_name} into {fine_name} "
                    f"({coarse_dim.semantic_purpose} → {fine_dim.semantic_purpose})"
                ),
                group_by=group_by,
                compare_across=fine_col,
                hold_fixed=[coarse_col],
                expected_groups=n_groups,
                use_case="drill_down",
            ))

    # Context 6: Interaction analysis — cross 3+ dimensions to detect
    # interaction effects (e.g. does column effect vary by stage?).
    # Combines unit × condition × phase when all three are available.
    if units and conditions and phases:
        unit_col = units[0].columns[0]
        cond_col = conditions[0].columns[0]
        phase_col = phases[0].columns[0]
        group_by = [unit_col, cond_col, phase_col]
        n_groups = _count_groups(group_by)
        if n_groups >= _MIN_GROUPS:
            contexts.append(AnalysisContext(
                name="interaction_analysis",
                description=(
                    f"Detect interaction effects between {conditions[0].name} "
                    f"and {phases[0].name} within each {units[0].name}"
                ),
                group_by=group_by,
                compare_across=cond_col,
                hold_fixed=[unit_col, phase_col],
                expected_groups=n_groups,
                use_case="interaction",
            ))

    # Context 7: Deep process trend — full hierarchy including replicates
    # when available (unit × condition × phase × replicate).
    if phases and replicates and (units or conditions):
        rep_col = replicates[0].columns[0]
        outer: List[str] = []
        hold_deep: List[str] = []
        if units:
            outer.append(units[0].columns[0])
            hold_deep.append(units[0].columns[0])
        if conditions:
            outer.append(conditions[0].columns[0])
            hold_deep.append(conditions[0].columns[0])
        phase_col = phases[0].columns[0]
        group_by = outer + [phase_col, rep_col]
        n_groups = _count_groups(group_by)
        if n_groups >= _MIN_GROUPS:
            contexts.append(AnalysisContext(
                name="deep_process_trend",
                description=(
                    f"Full-depth process trend: {phases[0].name} progression "
                    f"with {replicates[0].name} granularity"
                    + (f" within {' × '.join(hold_deep)}" if hold_deep else "")
                ),
                group_by=group_by,
                compare_across=phase_col,
                hold_fixed=hold_deep + [rep_col],
                expected_groups=n_groups,
                use_case="trend",
            ))

    return contexts


def _build_dimensional_structure(
    df: pd.DataFrame,
    column_profiles: List[ColumnProfile],
    relationships: List[ColumnRelationship],
) -> Optional[DimensionalStructure]:
    """Build the complete dimensional structure from classified columns and relationships."""
    # Step 1: Merge redundancy groups
    redundancy_groups, canonical_map = _merge_redundancy_groups(column_profiles, relationships)

    # Step 2: Build dimensions (one per canonical column)
    by_name = {cp.name: cp for cp in column_profiles}
    seen_canonical: set = set()
    dimensions: List[DimensionDescriptor] = []

    for cp in column_profiles:
        if cp.role not in ("categorical_group", "ordinal_stage"):
            continue
        if cp.cardinality >= _RELATIONSHIP_MAX_CARDINALITY:
            continue
        if cp.null_pct >= 50.0:
            continue

        canonical = canonical_map.get(cp.name, cp.name)
        if canonical in seen_canonical:
            continue
        seen_canonical.add(canonical)

        # Gather all columns in the redundancy group
        redundant_cols = []
        for grp in redundancy_groups:
            if canonical in grp:
                redundant_cols = [c for c in grp if c != canonical]
                break

        dimensions.append(DimensionDescriptor(
            name=canonical,
            columns=[canonical] + redundant_cols,
            cardinality=by_name[canonical].cardinality,
            semantic_purpose=by_name[canonical].semantic_purpose,
            nesting_parent=None,  # filled by _build_hierarchy
            redundant_with=redundant_cols,
            sample_values=by_name[canonical].sample_values[:8],
        ))

    # Also add linking identifiers as dimensions
    for cp in column_profiles:
        if cp.semantic_purpose == "linking_identifier" and cp.name not in seen_canonical:
            canonical = canonical_map.get(cp.name, cp.name)
            if canonical in seen_canonical:
                continue
            seen_canonical.add(canonical)
            dimensions.append(DimensionDescriptor(
                name=canonical,
                columns=[canonical],
                cardinality=cp.cardinality,
                semantic_purpose="linking_identifier",
                sample_values=cp.sample_values[:8],
            ))

    if not dimensions:
        return None

    # Step 3: Build hierarchy
    hierarchy = _build_hierarchy(dimensions, relationships, canonical_map)

    # Step 4: Generate analysis contexts
    analysis_contexts = _generate_analysis_contexts(dimensions, hierarchy, df)

    # Step 5: Assemble
    ds = DimensionalStructure(
        dimensions=dimensions,
        relationships=relationships,
        redundancy_groups=redundancy_groups,
        hierarchy_levels=hierarchy,
        analysis_contexts=analysis_contexts,
    )

    logger.info(
        "Dimensional structure: %d dimensions, %d hierarchy levels, %d analysis contexts, "
        "%d redundancy groups",
        len(dimensions), len(hierarchy), len(analysis_contexts), len(redundancy_groups),
    )
    for d in dimensions:
        logger.info(
            "  Dimension %s (%d levels) → %s%s",
            d.name, d.cardinality, d.semantic_purpose,
            f" [redundant with: {', '.join(d.redundant_with)}]" if d.redundant_with else "",
        )

    return ds


# ──────────────────────────────────────────────────────────────────────
# Phase B: Distribution profiling
# ──────────────────────────────────────────────────────────────────────

def _detect_modality(values: np.ndarray) -> Tuple[str, int]:
    """Detect number of modes in a continuous distribution using KDE + peak finding.

    Returns (modality_label, n_peaks).
    """
    from scipy.signal import find_peaks
    from scipy.stats import gaussian_kde

    if len(values) < _MIN_SAMPLES_SHAPIRO:
        return ("unimodal", 1)

    try:
        kde = gaussian_kde(values, bw_method="scott")
    except (np.linalg.LinAlgError, ValueError):
        return ("unimodal", 1)

    x_grid = np.linspace(values.min(), values.max(), _KDE_GRID_POINTS)
    density = kde(x_grid)
    prominence = density.max() * _PEAK_PROMINENCE_FACTOR

    peaks, _ = find_peaks(density, prominence=prominence)
    n_peaks = max(1, len(peaks))

    if n_peaks == 1:
        return ("unimodal", 1)
    elif n_peaks == 2:
        return ("bimodal", 2)
    else:
        return ("multimodal", n_peaks)


def _profile_distribution(series: pd.Series) -> Optional[DistributionProfile]:
    """Compute distribution characteristics for a continuous column.

    Returns None if insufficient data.
    """
    from scipy import stats as sp_stats

    values = series.dropna().values.astype(float)
    if len(values) < _MIN_SAMPLES_SHAPIRO:
        return None

    skewness = float(sp_stats.skew(values, bias=False))
    kurtosis = float(sp_stats.kurtosis(values, bias=False))  # excess (Fisher)

    # Normality test — prefer D'Agostino-Pearson (n >= 20), fall back to Shapiro
    normality_stat: Optional[float] = None
    normality_pvalue: Optional[float] = None
    normality_test = "insufficient_data"

    if len(values) >= _MIN_SAMPLES_NORMALITY:
        try:
            stat, pval = sp_stats.normaltest(values)
            normality_stat = float(stat)
            normality_pvalue = float(pval)
            normality_test = "dagostino"
        except Exception:
            pass

    if normality_test == "insufficient_data" and len(values) >= _MIN_SAMPLES_SHAPIRO:
        try:
            stat, pval = sp_stats.shapiro(values[:5000])  # Shapiro caps at ~5000
            normality_stat = float(stat)
            normality_pvalue = float(pval)
            normality_test = "shapiro"
        except Exception:
            pass

    is_normal = (
        normality_pvalue > _NORMALITY_ALPHA
        if normality_pvalue is not None
        else False
    )

    # Modality detection
    modality, n_modes = _detect_modality(values)

    # Suggest transform for skewed data
    suggested_transform: Optional[str] = None
    abs_skew = abs(skewness)
    if abs_skew >= _EXTREME_SKEW_THRESHOLD and (values > 0).all():
        suggested_transform = "log"
    elif abs_skew >= _HIGH_SKEW_THRESHOLD:
        suggested_transform = "sqrt" if (values >= 0).all() else None

    return DistributionProfile(
        skewness=round(skewness, 4),
        kurtosis=round(kurtosis, 4),
        normality_stat=round(normality_stat, 4) if normality_stat is not None else None,
        normality_pvalue=round(normality_pvalue, 6) if normality_pvalue is not None else None,
        normality_test=normality_test,
        is_normal=is_normal,
        modality=modality,
        n_modes=n_modes,
        suggested_transform=suggested_transform,
    )


# ──────────────────────────────────────────────────────────────────────
# Phase B: Technique recommendations
# ──────────────────────────────────────────────────────────────────────

def _recommend_techniques(profile: DataProfile) -> List[TechniqueRecommendation]:
    """Generate analytical technique recommendations from the data profile.

    Rule-based engine — no LLM calls. Considers:
    - Number of groups and group sizes
    - Normality of continuous columns
    - Presence of ordinal stages (trend analysis)
    - Number of continuous columns (multivariate techniques)
    - Modality of distributions (clustering)
    """
    recs: List[TechniqueRecommendation] = []
    continuous_cols = profile.continuous_measurements
    has_groups = profile.recommended_grouping is not None
    n_groups = profile.recommended_grouping.group_count if has_groups else 0
    has_ordinal = len(profile.ordinal_stages) > 0

    # Count how many continuous columns are normal vs non-normal
    normal_cols = [c for c in continuous_cols if c in profile.distributions
                   and profile.distributions[c].is_normal]
    non_normal_cols = [c for c in continuous_cols if c in profile.distributions
                       and not profile.distributions[c].is_normal]

    # --- Group comparison techniques ---
    if has_groups and continuous_cols:
        if n_groups == 2:
            if normal_cols:
                recs.append(TechniqueRecommendation(
                    technique="independent t-test",
                    rationale=f"{n_groups} groups with normally distributed metrics — "
                              "parametric two-sample comparison is appropriate",
                    columns=normal_cols,
                    priority="primary",
                ))
            if non_normal_cols:
                recs.append(TechniqueRecommendation(
                    technique="Mann-Whitney U test",
                    rationale=f"{n_groups} groups with non-normal metrics — "
                              "non-parametric two-sample comparison avoids normality assumption",
                    columns=non_normal_cols,
                    priority="primary",
                ))
        elif n_groups >= 3:
            if normal_cols:
                recs.append(TechniqueRecommendation(
                    technique="one-way ANOVA",
                    rationale=f"{n_groups} groups with normally distributed metrics — "
                              "parametric multi-group comparison with post-hoc Tukey HSD",
                    columns=normal_cols,
                    priority="primary",
                ))
            if non_normal_cols:
                recs.append(TechniqueRecommendation(
                    technique="Kruskal-Wallis test",
                    rationale=f"{n_groups} groups with non-normal metrics — "
                              "non-parametric multi-group comparison with post-hoc Dunn's test",
                    columns=non_normal_cols,
                    priority="primary",
                ))

    # --- Trend analysis for ordinal stages ---
    if has_ordinal and continuous_cols:
        recs.append(TechniqueRecommendation(
            technique="Spearman rank correlation",
            rationale=f"ordinal stages ({', '.join(profile.ordinal_stages)}) present — "
                      "test for monotonic trends across process stages",
            columns=continuous_cols,
            priority="primary",
        ))

    # --- Correlation analysis ---
    if len(continuous_cols) >= 2:
        if len(normal_cols) >= 2:
            recs.append(TechniqueRecommendation(
                technique="Pearson correlation matrix",
                rationale=f"{len(continuous_cols)} continuous columns with normal "
                          "subsets — identify linear relationships between metrics",
                columns=continuous_cols,
                priority="primary" if len(continuous_cols) <= 8 else "exploratory",
            ))
        if len(non_normal_cols) >= 2:
            recs.append(TechniqueRecommendation(
                technique="Spearman correlation matrix",
                rationale=f"non-normal continuous columns present — "
                          "rank-based correlation is robust to skewness and outliers",
                columns=continuous_cols,
                priority="primary",
            ))

    # --- Multivariate techniques ---
    if len(continuous_cols) >= 4:
        recs.append(TechniqueRecommendation(
            technique="PCA (principal component analysis)",
            rationale=f"{len(continuous_cols)} continuous columns — "
                      "dimensionality reduction to identify dominant variation patterns",
            columns=continuous_cols,
            priority="exploratory",
        ))

    # --- Clustering for multimodal distributions ---
    bimodal_cols = [c for c in continuous_cols if c in profile.distributions
                    and profile.distributions[c].n_modes >= 2]
    if bimodal_cols:
        recs.append(TechniqueRecommendation(
            technique="Gaussian mixture model / k-means clustering",
            rationale=f"multimodal distributions detected in {', '.join(bimodal_cols)} — "
                      "clustering may reveal distinct sub-populations or process modes",
            columns=bimodal_cols,
            priority="exploratory",
        ))

    # --- Transform suggestions ---
    transform_cols = [c for c in continuous_cols if c in profile.distributions
                      and profile.distributions[c].suggested_transform is not None]
    if transform_cols:
        transforms = {c: profile.distributions[c].suggested_transform
                      for c in transform_cols}
        transform_summary = ", ".join(f"{c} → {t}" for c, t in transforms.items())
        recs.append(TechniqueRecommendation(
            technique="variance-stabilising transform",
            rationale=f"highly skewed columns detected — apply transforms before "
                      f"parametric tests: {transform_summary}",
            columns=transform_cols,
            priority="primary",
        ))

    return recs


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def profile_dataset(
    parquet_path: str | Path,
    context_grouping_columns: Optional[List[str]] = None,
    context_extend_columns: Optional[List[str]] = None,
    mode: str = "roles_only",
) -> DataProfile:
    """Profile a cleaned dataset and return a structured DataProfile.

    Args:
        parquet_path: Path to the cleaned parquet file.
        context_grouping_columns: Grouping columns from context.md (override).
            If provided, these take precedence over inferred grouping but
            inferred candidates are still computed for reference.
        context_extend_columns: Additional grouping columns from context.md
            (``grouping_extend_when_present``).  These are appended to
            ``context_grouping_columns`` when the data supports them
            (i.e. when they exist and have < _CANDIDATE_NULL_CEILING nulls).
        mode: Profiling depth — "roles_only" (Phase A) or "full" (Phase A+B).

    Returns:
        DataProfile with column classifications, grouping candidates, and
        (if mode="full") distribution characterisations and technique
        recommendations.
    """
    path = Path(parquet_path)
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.error("Schema profiler failed to read %s: %s", path, exc)
        return DataProfile(file_path=str(path), row_count=0, column_count=0)

    row_count = len(df)
    col_count = len(df.columns)

    # Stash extend columns for later — they are tried *after* the base
    # context grouping is validated so that a failing extension never
    # blocks acceptance of the core grouping the user specified.
    _extend_columns: List[str] = []
    if context_extend_columns and context_grouping_columns:
        for ext_col in context_extend_columns:
            if ext_col in df.columns and ext_col not in context_grouping_columns:
                null_pct = df[ext_col].isna().mean()
                if null_pct < _CANDIDATE_NULL_CEILING:
                    _extend_columns.append(ext_col)
                    logger.info(
                        "Context extension candidate '%s' eligible "
                        "(%.1f%% null, below ceiling %.0f%%)",
                        ext_col, null_pct * 100, _CANDIDATE_NULL_CEILING * 100,
                    )
                else:
                    logger.info(
                        "Skipping context extension column '%s' "
                        "(%.1f%% null, above ceiling %.0f%%)",
                        ext_col, null_pct * 100, _CANDIDATE_NULL_CEILING * 100,
                    )

    # Phase A: Classify columns
    col_profiles: List[ColumnProfile] = []
    for col_name in df.columns:
        cp = _profile_column(df[col_name], col_name, row_count)
        col_profiles.append(cp)

    # Phase A+: Relationship graph, semantic purposes, dimensional structure
    # (moved before grouping so semantic multiplier and composition are available)
    relationships = _build_relationship_graph(df, col_profiles)

    # Semantic purpose classification (two-pass)
    for cp in col_profiles:
        cp.semantic_purpose = _initial_purpose(cp)
    _refine_purposes(col_profiles, relationships, df)

    for cp in col_profiles:
        if cp.semantic_purpose != "uncategorised":
            logger.info(
                "  %s → %s (%s)", cp.name, cp.semantic_purpose, cp.role,
            )

    # Phase A: Infer grouping candidates (now with semantic multiplier)
    grouping_candidates = _infer_grouping_candidates(df, col_profiles)

    # Compose a semantic grouping from dimensional purposes
    semantic_gc = _compose_semantic_grouping(df, col_profiles)

    # Determine recommended grouping.
    # Priority: (1) full context override, (1b) partial context override,
    #           (2) semantic composition, (3) top inferred
    recommended: Optional[GroupingCandidate] = None
    _extended_gc: Optional[GroupingCandidate] = None
    _context_override_rejected: Optional[str] = None
    _partial_override_notes: List[str] = []
    if context_grouping_columns:
        # Context override: evaluate the specified columns as a grouping
        override_gc = _evaluate_grouping(df, context_grouping_columns)
        if override_gc is not None:
            recommended = override_gc
            logger.info(
                "Using context.md grouping override: %s (%d groups)",
                context_grouping_columns, override_gc.group_count,
            )
            # Try extending with additional columns from context.md.
            # Extensions produce a *separate* finer grouping for the
            # subset of rows where the extension columns are non-null.
            # The base recommended grouping is never replaced because
            # extension columns can have high null rates (dropping
            # coverage) and produce many more groups.
            if _extend_columns:
                ext_trial = list(context_grouping_columns)
                for ext_col in _extend_columns:
                    trial = ext_trial + [ext_col]
                    trial_gc = _evaluate_grouping(
                        df, trial,
                        max_groups=_MAX_GROUPS * 3,
                    )
                    if trial_gc is not None:
                        ext_trial = trial
                        _extended_gc = trial_gc
                        logger.info(
                            "Extended grouping with '%s': %s (%d groups, "
                            "coverage=%.1f%%)",
                            ext_col, ext_trial, trial_gc.group_count,
                            trial_gc.rows_covered_pct,
                        )
                    else:
                        _partial_override_notes.append(
                            f"Extension column '{ext_col}' was eligible "
                            f"(null < {_CANDIDATE_NULL_CEILING*100:.0f}%) "
                            f"but extending the grouping to {trial} produced "
                            f"too many groups. Consider sub-group analysis on "
                            f"'{ext_col}' within individual "
                            f"{', '.join(context_grouping_columns)} groups."
                        )
                        logger.info(
                            "Extension column '%s' failed evaluation, "
                            "not included in extended grouping",
                            ext_col,
                        )
        else:
            # Diagnose why the override failed
            missing = [c for c in context_grouping_columns if c not in df.columns]
            high_null = [
                c for c in context_grouping_columns
                if c in df.columns and df[c].isna().mean() > _CANDIDATE_NULL_CEILING
            ]
            usable = [
                c for c in context_grouping_columns
                if c in df.columns and c not in missing and c not in high_null
            ]
            reasons = []
            if missing:
                reasons.append(f"columns not found: {missing}")
            if high_null:
                reasons.append(f"columns >{_CANDIDATE_NULL_CEILING*100:.0f}%% null: {high_null}")

            # Attempt partial override: use the usable subset of context
            # columns so that the user's intent is honoured as far as
            # the data allows.
            if usable and len(usable) < len(context_grouping_columns):
                partial_gc = _evaluate_grouping(df, usable)
                if partial_gc is not None:
                    recommended = partial_gc
                    dropped = [c for c in context_grouping_columns if c not in usable]
                    _partial_override_notes.append(
                        f"Context requested grouping by {context_grouping_columns} "
                        f"but columns {dropped} were unusable ({'; '.join(reasons)}). "
                        f"Using partial override: {usable} "
                        f"({partial_gc.group_count} groups, {partial_gc.rows_covered_pct}% coverage). "
                        f"For the subset of rows where {dropped} are present, "
                        f"consider extending the grouping to include them."
                    )
                    logger.warning(
                        "Context grouping_columns %s partial override: using %s "
                        "(dropped %s: %s).",
                        context_grouping_columns, usable, dropped,
                        "; ".join(reasons),
                    )

            if recommended is None:
                if not reasons:
                    reasons.append("too few/many groups or insufficient data")
                _context_override_rejected = (
                    f"context.md requested grouping by {context_grouping_columns} "
                    f"but override failed ({'; '.join(reasons)})"
                )
                logger.warning(
                    "Context grouping_columns %s override rejected: %s. "
                    "Falling back to semantic inference.",
                    context_grouping_columns, "; ".join(reasons),
                )

    if recommended is None and semantic_gc is not None:
        recommended = semantic_gc
        logger.info(
            "Using semantic grouping: %s (%d groups, score=%.3f)",
            semantic_gc.columns, semantic_gc.group_count, semantic_gc.score,
        )
    elif recommended is None and grouping_candidates:
        recommended = grouping_candidates[0]
        logger.info(
            "Using top inferred grouping: %s (%d groups, score=%.3f)",
            recommended.columns, recommended.group_count, recommended.score,
        )

    # Build dimensional structure
    dim_structure = _build_dimensional_structure(df, col_profiles, relationships)
    if dim_structure is not None:
        dim_structure.recommended_default_grouping = recommended

    # Build role summary lists
    profile = DataProfile(
        file_path=str(path),
        row_count=row_count,
        column_count=col_count,
        columns=col_profiles,
        grouping_candidates=grouping_candidates[:10],  # keep top 10
        recommended_grouping=recommended,
        extended_grouping=_extended_gc,
        identifiers=[cp.name for cp in col_profiles if cp.role == "identifier"],
        categorical_groups=[cp.name for cp in col_profiles if cp.role == "categorical_group"],
        ordinal_stages=[cp.name for cp in col_profiles if cp.role == "ordinal_stage"],
        continuous_measurements=[cp.name for cp in col_profiles if cp.role == "continuous_measurement"],
        metadata_text_cols=[cp.name for cp in col_profiles if cp.role == "metadata_text"],
        datetime_cols=[cp.name for cp in col_profiles if cp.role == "datetime"],
        constant_cols=[cp.name for cp in col_profiles if cp.role == "constant"],
        dimensional_structure=dim_structure,
        profiler_warnings=(
            ([_context_override_rejected] if _context_override_rejected else [])
            + _partial_override_notes
        ),
    )

    # Phase B: Distribution profiling and technique recommendations
    if mode == "full":
        distributions: Dict[str, DistributionProfile] = {}
        for cp in col_profiles:
            if cp.role == "continuous_measurement":
                dist = _profile_distribution(df[cp.name])
                if dist is not None:
                    distributions[cp.name] = dist
        profile.distributions = distributions

        profile.technique_recommendations = _recommend_techniques(profile)

        # Post-Phase B: Inject bimodal/multimodal analysis contexts.
        # When a continuous column is bimodal, this is strong evidence of a
        # latent grouping dimension (e.g. two process modes, two populations).
        # Generate sub-group exploration contexts for these columns.
        if dim_structure is not None:
            bimodal_cols = [
                col_name for col_name, dp in distributions.items()
                if dp.n_modes >= 2
            ]
            if bimodal_cols:
                for bc in bimodal_cols[:3]:  # cap at 3 to avoid noise
                    dp = distributions[bc]
                    dim_structure.analysis_contexts.append(AnalysisContext(
                        name=f"subgroup_exploration_{bc}",
                        description=(
                            f"{bc} has a {dp.modality} distribution "
                            f"({dp.n_modes} modes) — investigate whether a "
                            f"latent categorical split explains the multimodality"
                        ),
                        group_by=[c for c in (recommended.columns if recommended else [])],
                        compare_across=bc,
                        hold_fixed=[],
                        expected_groups=dp.n_modes,
                        use_case="subgroup_exploration",
                    ))
                logger.info(
                    "Phase B: added %d bimodal sub-group exploration context(s) "
                    "for columns: %s",
                    len(bimodal_cols[:3]), bimodal_cols[:3],
                )

        logger.info(
            "Schema profiler Phase B: %d/%d columns profiled, "
            "%d technique recommendations",
            len(distributions), len(profile.continuous_measurements),
            len(profile.technique_recommendations),
        )

    logger.info(
        "Schema profiler [%s]: %d rows, %d cols → %d identifiers, "
        "%d categorical, %d ordinal, %d continuous, %d grouping candidates "
        "(recommended: %s)",
        mode, row_count, col_count,
        len(profile.identifiers), len(profile.categorical_groups),
        len(profile.ordinal_stages), len(profile.continuous_measurements),
        len(grouping_candidates),
        recommended.columns if recommended else "none",
    )

    return profile


def save_profile(profile: DataProfile, output_dir: str | Path) -> Path:
    """Save a DataProfile to data_profile.json in the output directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile_path = out / "data_profile.json"
    profile_path.write_text(
        json.dumps(profile.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Data profile saved to %s", profile_path)
    return profile_path


def build_profile_instructions(profile: DataProfile) -> str:
    """Build an LLM-readable instruction block from a DataProfile.

    Replaces the legacy build_grouping_instructions() when schema profiling
    is enabled. Provides column role context, dimensional structure, and
    multi-context grouping guidance.
    """
    lines: List[str] = []
    ds = profile.dimensional_structure

    # ── Profiler warnings (e.g. context override rejection) ──
    if profile.profiler_warnings:
        for w in profile.profiler_warnings:
            lines.append(f"WARNING: {w}")
        lines.append("")

    lines.append("DATA PROFILE (from schema profiler):")
    lines.append(f"Dataset: {profile.row_count} rows, {profile.column_count} columns.")
    lines.append("")

    # ── Column roles (enriched with semantic purpose when available) ──
    lines.append("COLUMN ROLES:")

    if ds and ds.dimensions:
        # Group columns by semantic purpose for richer output
        purpose_cols: Dict[str, List[str]] = {}
        for d in ds.dimensions:
            label = d.semantic_purpose
            info = f"{d.name} ({d.cardinality} unique"
            if d.redundant_with:
                info += f"; redundant with: {', '.join(d.redundant_with)}"
            info += ")"
            purpose_cols.setdefault(label, []).append(info)

        purpose_labels = {
            "experimental_unit": "Experimental units",
            "experimental_condition": "Experimental conditions",
            "process_phase": "Process phases",
            "technical_replicate": "Technical replicates",
            "linking_identifier": "Linking identifiers",
            "protocol_metadata": "Protocol metadata",
            "uncategorised": "Other discrete",
        }
        for purpose, label in purpose_labels.items():
            if purpose in purpose_cols:
                lines.append(f"  {label}: {', '.join(purpose_cols[purpose])}")
    else:
        # Fallback: original role-based listing
        if profile.identifiers:
            lines.append(f"  Identifiers: {', '.join(profile.identifiers)}")
        if profile.categorical_groups:
            lines.append(f"  Categorical groups: {', '.join(profile.categorical_groups)}")
        if profile.ordinal_stages:
            lines.append(f"  Ordinal stages: {', '.join(profile.ordinal_stages)}")

    if profile.continuous_measurements:
        lines.append(f"  Continuous measurements: {', '.join(profile.continuous_measurements)}")
    if profile.metadata_text_cols:
        # Exclude columns already shown as linking identifiers
        shown_as_linking = set()
        if ds:
            for d in ds.dimensions:
                if d.semantic_purpose == "linking_identifier":
                    shown_as_linking.update(d.columns)
        remaining_meta = [c for c in profile.metadata_text_cols if c not in shown_as_linking]
        if remaining_meta:
            lines.append(f"  Metadata/text: {', '.join(remaining_meta)}")
    if profile.datetime_cols:
        lines.append(f"  Datetime: {', '.join(profile.datetime_cols)}")
    if profile.constant_cols:
        lines.append(f"  Constant (ignore): {', '.join(profile.constant_cols)}")
    lines.append("")

    # ── Dimensional structure ──
    if ds and ds.dimensions:
        # Redundancy warnings
        if ds.redundancy_groups:
            lines.append("REDUNDANT COLUMNS (use the first as canonical):")
            for grp in ds.redundancy_groups:
                lines.append(f"  {' = '.join(grp)}")
            lines.append("  Do NOT include both members of a redundant pair in groupby operations.")
            lines.append("")

        # Hierarchy
        if ds.hierarchy_levels:
            lines.append("DIMENSIONAL STRUCTURE (hierarchy from coarsest to finest):")
            for dim_name in ds.hierarchy_levels:
                dim = next((d for d in ds.dimensions if d.name == dim_name), None)
                if dim is None:
                    continue
                parent_str = f" (within {dim.nesting_parent})" if dim.nesting_parent else " (outermost)"
                vals_str = ""
                if dim.sample_values:
                    vals_str = f" — e.g. {', '.join(str(v) for v in dim.sample_values[:6])}"
                lines.append(
                    f"  {dim.name}: {dim.cardinality} levels "
                    f"[{dim.semantic_purpose}]{parent_str}{vals_str}"
                )
            lines.append("")

    # ── Ordinal stage trajectory plot guidance ──
    if profile.ordinal_stages:
        lines.append("TRAJECTORY PLOTS (required when ordinal stages are present):")
        lines.append(
            f"  These columns represent sequential process phases: "
            f"{', '.join(profile.ordinal_stages)}."
        )
        lines.append(
            "  For each ordinal stage column, you MUST include at least one "
            "trajectory plot: x-axis = stage order, y-axis = a key measurement, "
            "with individual experiments distinguishable by colour or facet."
        )
        lines.append(
            "  Distribution plots (boxplots by stage) do NOT replace trajectory "
            "plots — they cannot show within-run progression or between-run "
            "alignment across stages."
        )
        lines.append("")

    # ── Analysis contexts (multi-context grouping guidance) ──
    if ds and ds.analysis_contexts:
        lines.append("ANALYSIS CONTEXTS — use the appropriate context for each analysis question:")
        lines.append("  These are NOT optional extras; they represent analytically distinct ways")
        lines.append("  to slice the data. If a question involves process trends across stages,")
        lines.append("  use the PROCESS_TREND context. If comparing conditions, use CONDITION_COMPARISON.")
        lines.append("  Extend the default grouping when your analysis question requires finer")
        lines.append("  discrimination (e.g., per-stage or per-sample breakdown).")
        for i, ac in enumerate(ds.analysis_contexts, 1):
            key_str = ", ".join(ac.group_by)
            lines.append(
                f"  {i}. {ac.name.upper()}: Group by ({key_str}). "
                f"Compare across {ac.compare_across}."
            )
            lines.append(f"     {ac.description}.")
            lines.append(
                f"     ~{ac.expected_groups} groups. Use for: {ac.use_case}."
            )
        lines.append("")

    # ── Default grouping (mandatory for per_group output structure) ──
    if profile.recommended_grouping:
        rg = profile.recommended_grouping
        key_str = ", ".join(rg.columns)
        compound_example = "__".join(f"<{c}>" for c in rg.columns)
        lines.append("DEFAULT DATA GROUPING (for per_group output structure):")
        lines.append(f"  Group by the compound key: ({key_str}).")
        lines.append(f"  This produces {rg.group_count} analytical groups "
                      f"(median size: {rg.median_group_size} rows, "
                      f"coverage: {rg.rows_covered_pct}%).")
        if rg.group_count == 2:
            lines.append("  NOTE: 2 groups detected — use t-test or pairwise comparison "
                          "(not ANOVA). Report effect size (Cohen's d) alongside p-value.")
        lines.append(f"  Store per-group results using compound keys joined by '__':")
        lines.append(f"    e.g. \"{compound_example}\"")
        lines.append("  Use a 'per_group' key in analysis_summary.json.")
        if ds and ds.analysis_contexts:
            lines.append("  You SHOULD use analysis contexts above when your analysis question")
            lines.append("  requires additional grouping dimensions (e.g., adding process_phase")
            lines.append("  for stage-level trends). The default grouping is the MINIMUM;")
            lines.append("  extend it when the data structure warrants finer discrimination.")
        lines.append("")

        # ── Required secondary analyses for uncovered meaningful dimensions ──
        _MEANINGFUL_PURPOSES = {
            "process_phase", "experimental_condition", "experimental_unit"
        }
        rg_cols = set(rg.columns)
        uncovered_dims: List[Any] = []
        if ds and ds.dimensions:
            for _dim in ds.dimensions:
                if _dim.semantic_purpose in _MEANINGFUL_PURPOSES:
                    _dim_cols = set(_dim.columns)
                    if not _dim_cols & rg_cols:
                        uncovered_dims.append(_dim)

        if uncovered_dims:
            _purpose_labels = {
                "experimental_unit": "experimental unit",
                "experimental_condition": "experimental condition",
                "process_phase": "process phase",
            }
            lines.append(
                "REQUIRED SECONDARY ANALYSES "
                "(dimensions NOT covered by the primary key):"
            )
            lines.append(
                "  Your primary grouping key does NOT include all meaningful "
                "analytical dimensions.  You MUST produce a secondary analysis "
                "for each dimension listed below — these are NOT optional extras."
            )
            lines.append(
                "  Keep the primary compound-key analysis intact.  Add SEPARATE "
                "secondary analyses using each missing dimension and store results "
                "under the 'secondary_analysis' key in analysis_summary.json, "
                "using the dimension name as the sub-key."
            )
            for _dim in uncovered_dims:
                _plabel = _purpose_labels.get(_dim.semantic_purpose, _dim.semantic_purpose)
                _vals_str = ""
                if _dim.sample_values:
                    _vals_str = (
                        f" — e.g. {', '.join(str(v) for v in _dim.sample_values[:4])}"
                    )
                lines.append(
                    f"  • {_dim.name} [{_plabel}, {_dim.cardinality} levels"
                    f"{_vals_str}]"
                )
                if _dim.semantic_purpose == "experimental_condition":
                    lines.append(
                        f"    → Compare key metrics between {_dim.name} groups "
                        f"(box/violin plot).  "
                        f"secondary_analysis['{_dim.name}'] = "
                        f"{{{{'<value>': {{'metric1': v, ...}}}}}}"
                    )
                elif _dim.semantic_purpose == "experimental_unit":
                    lines.append(
                        f"    → Summarise metric CVs per {_dim.name} and flag "
                        f"outlier values.  "
                        f"secondary_analysis['{_dim.name}'] = "
                        f"{{{{'<run_id>': {{'cv_percent': v, ...}}}}}}"
                    )
                else:
                    lines.append(
                        f"    → Show how key metrics change across {_dim.name} "
                        f"phases.  "
                        f"secondary_analysis['{_dim.name}'] = "
                        f"{{{{'<phase>': {{'mean': v, ...}}}}}}"
                    )
            lines.append("")

        # ── Per-run × per-stage cross-tabulation ──
        # Required when both experimental_unit and process_phase dimensions exist.
        _run_dims = (
            [d for d in ds.dimensions if d.semantic_purpose == "experimental_unit"]
            if ds and ds.dimensions else []
        )
        _stage_dims = (
            [d for d in ds.dimensions if d.semantic_purpose == "process_phase"]
            if ds and ds.dimensions else []
        )
        if _run_dims and _stage_dims:
            _run_col = _run_dims[0].columns[0]
            _stage_col = _stage_dims[0].columns[0]
            lines.append("PER_RUN_PER_STAGE CROSS-TABULATION (mandatory for this dataset):")
            lines.append(
                f"  This dataset has both {_run_dims[0].name} (experimental unit) "
                f"and {_stage_dims[0].name} (process phase) dimensions."
            )
            lines.append(
                "  You MUST compute a run × stage cross-tabulation of key metrics "
                "and store it under the 'per_run_per_stage' key in "
                "analysis_summary.json."
            )
            lines.append(
                f"  Format: {{'<{_run_col}>__<{_stage_col}>': "
                "{'metric1': value, 'metric2': value, 'row_count': N}, ...}}"
            )
            lines.append(
                "  Example: group df by ['"
                + _run_col + "', '" + _stage_col
                + "'] and compute per-group metric summaries."
            )
            lines.append(
                "  This cross-tabulation is required for cross-validation and "
                "longitudinal quality tracking across runs and stages."
            )
            lines.append("")

        # ── Alternative grouping candidates (top 5) ──
        alternatives = [
            gc for gc in profile.grouping_candidates
            if gc.columns != rg.columns
        ][:4]
        if alternatives:
            lines.append("ALTERNATIVE GROUPING CANDIDATES (ranked by quality):")
            for gc in alternatives:
                alt_str = ", ".join(gc.columns)
                lines.append(
                    f"  ({alt_str}): {gc.group_count} groups, "
                    f"score={gc.score:.3f}, coverage={gc.rows_covered_pct}%"
                )
            lines.append("  Consider these if the default grouping is too coarse for")
            lines.append("  your analysis question or if you need finer subgroups.")
            lines.append("")

        # ── Extended grouping for subset analysis ──
        if profile.extended_grouping:
            eg = profile.extended_grouping
            ext_cols = [c for c in eg.columns if c not in rg.columns]
            ext_str = ", ".join(eg.columns)
            lines.append("EXTENDED GROUPING (for subset analysis):")
            lines.append(f"  When analysing the {eg.rows_covered_pct}% of rows where "
                          f"{', '.join(ext_cols)} {'is' if len(ext_cols) == 1 else 'are'} "
                          f"non-null, use the finer grouping: ({ext_str}).")
            lines.append(f"  This produces {eg.group_count} groups "
                          f"(median size: {eg.median_group_size} rows).")
            lines.append(f"  Use this for stage-level or sample-level breakdown "
                          f"within each ({key_str}) group.")
            lines.append(f"  The remaining {round(100 - eg.rows_covered_pct, 1)}% of rows "
                          f"(where {', '.join(ext_cols)} {'is' if len(ext_cols) == 1 else 'are'} "
                          f"null) should still be analysed using the default grouping above.")
            lines.append("")
    else:
        lines.append("GROUPING GUIDANCE:")
        lines.append("  No strong grouping structure detected in this dataset.")
        lines.append("  Inspect categorical columns to identify potential groupings.")
        if profile.categorical_groups:
            lines.append(f"  Candidate columns: {', '.join(profile.categorical_groups)}")
        lines.append("")

    # Measurement columns to analyse
    if profile.continuous_measurements:
        lines.append("MEASUREMENT COLUMNS (analyse these):")
        for cp in profile.columns:
            if cp.role == "continuous_measurement" and cp.stats:
                lines.append(
                    f"  {cp.name}: range [{cp.stats['min']:.4g}, {cp.stats['max']:.4g}], "
                    f"mean={cp.stats['mean']:.4g}, std={cp.stats['std']:.4g}"
                )
        lines.append("")

    # ── Phase B: Distribution characteristics ──
    if profile.distributions:
        lines.append("DISTRIBUTION CHARACTERISTICS:")
        for col_name, dp in profile.distributions.items():
            normal_tag = "normal" if dp.is_normal else "non-normal"
            parts = [
                f"  {col_name}: {dp.modality}",
                f"{normal_tag}",
                f"skew={dp.skewness:.2f}",
                f"kurtosis={dp.kurtosis:.2f}",
            ]
            if dp.normality_pvalue is not None:
                parts.append(f"p={dp.normality_pvalue:.4f}")
            if dp.suggested_transform:
                parts.append(f"→ consider {dp.suggested_transform} transform")
            lines.append(", ".join(parts))
        lines.append("")

    # ── Phase B: Technique recommendations ──
    if profile.technique_recommendations:
        lines.append("RECOMMENDED ANALYTICAL TECHNIQUES:")
        primary = [r for r in profile.technique_recommendations if r.priority == "primary"]
        exploratory = [r for r in profile.technique_recommendations if r.priority == "exploratory"]
        if primary:
            lines.append("  Primary:")
            for rec in primary:
                cols_str = ", ".join(rec.columns[:5])
                if len(rec.columns) > 5:
                    cols_str += f" (+{len(rec.columns) - 5} more)"
                lines.append(f"    - {rec.technique}: {rec.rationale}")
                lines.append(f"      Columns: {cols_str}")
        if exploratory:
            lines.append("  Exploratory:")
            for rec in exploratory:
                cols_str = ", ".join(rec.columns[:5])
                if len(rec.columns) > 5:
                    cols_str += f" (+{len(rec.columns) - 5} more)"
                lines.append(f"    - {rec.technique}: {rec.rationale}")
                lines.append(f"      Columns: {cols_str}")
        lines.append("")

    return "\n".join(lines)
