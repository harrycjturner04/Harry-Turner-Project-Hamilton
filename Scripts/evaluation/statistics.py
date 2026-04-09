"""Statistical analysis: bootstrap CI, significance tests, effect sizes."""

import logging
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReplicationStats:
    """Statistics for N replicated runs of a single configuration."""

    config_label: str
    metric_name: str
    n_runs: int
    values: List[float]
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    median: float


@dataclass
class ComparisonResult:
    """Statistical comparison between two configurations."""

    config_a: str
    config_b: str
    metric_name: str
    mean_diff: float
    effect_size_cohens_d: float
    mann_whitney_u: float
    mann_whitney_p: float
    permutation_p: float
    n_permutations: int
    significant_at_005: bool
    correction_method: str
    values_a: List[float] = None  # type: ignore[assignment]
    values_b: List[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.values_a is None:
            self.values_a = []
        if self.values_b is None:
            self.values_b = []


def bootstrap_ci(
    values: np.ndarray,
    stat_func: Callable = np.mean,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval.

    Uses percentile method (suitable for small N=5-10).
    Falls back gracefully for N=1.
    """
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        val = float(values[0]) if len(values) == 1 else 0.0
        return (val, val)

    rng = np.random.default_rng(random_state)
    boot_stats = np.array([
        stat_func(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = 1 - ci_level
    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return (lower, upper)


def compute_replication_stats(
    scores: List[float],
    config_label: str,
    metric_name: str,
    ci_level: float = 0.95,
    n_bootstrap: int = 10000,
) -> ReplicationStats:
    """Compute summary statistics with bootstrap confidence intervals."""
    arr = np.array(scores, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    median = float(np.median(arr))
    ci_lower, ci_upper = bootstrap_ci(
        arr, ci_level=ci_level, n_bootstrap=n_bootstrap
    )

    return ReplicationStats(
        config_label=config_label,
        metric_name=metric_name,
        n_runs=n,
        values=scores,
        mean=round(mean, 4),
        std=round(std, 4),
        ci_lower=round(ci_lower, 4),
        ci_upper=round(ci_upper, 4),
        median=round(median, 4),
    )


def mann_whitney_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
) -> Tuple[float, float]:
    """Two-sided Mann-Whitney U test.

    Appropriate for small N, non-parametric.
    Returns (U statistic, p-value).
    """
    from scipy.stats import mannwhitneyu

    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)

    if len(a) < 2 or len(b) < 2:
        return (0.0, 1.0)

    try:
        u_stat, p_value = mannwhitneyu(a, b, alternative="two-sided")
        return (float(u_stat), float(p_value))
    except ValueError:
        return (0.0, 1.0)


def permutation_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_permutations: int = 10000,
    random_state: int = 42,
) -> float:
    """Two-sided permutation test for difference in means.

    Returns p-value.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)

    if len(a) < 2 or len(b) < 2:
        return 1.0

    observed_diff = abs(float(np.mean(a) - np.mean(b)))
    combined = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.default_rng(random_state)

    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_diff = abs(
            float(np.mean(combined[:n_a]) - np.mean(combined[n_a:]))
        )
        if perm_diff >= observed_diff:
            count += 1

    return count / n_permutations


def cohens_d(
    values_a: np.ndarray,
    values_b: np.ndarray,
) -> float:
    """Cohen's d effect size with Hedges' g correction for small samples.

    Interpretation: |d| < 0.2 negligible, 0.2-0.5 small,
    0.5-0.8 medium, > 0.8 large.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    n_a, n_b = len(a), len(b)

    if n_a < 2 or n_b < 2:
        return 0.0

    pooled_std = math.sqrt(
        ((n_a - 1) * float(np.var(a, ddof=1))
         + (n_b - 1) * float(np.var(b, ddof=1)))
        / (n_a + n_b - 2)
    )

    if pooled_std == 0:
        return 0.0

    d = (float(np.mean(a)) - float(np.mean(b))) / pooled_std
    # Hedges' g correction
    correction = 1 - 3 / (4 * (n_a + n_b) - 9)
    return round(d * correction, 4)


def compare_configurations(
    stats_a: ReplicationStats,
    stats_b: ReplicationStats,
    n_permutations: int = 10000,
) -> ComparisonResult:
    """Full statistical comparison between two configurations."""
    a = np.array(stats_a.values)
    b = np.array(stats_b.values)

    mean_diff = stats_a.mean - stats_b.mean
    effect = cohens_d(a, b)
    u_stat, u_p = mann_whitney_test(a, b)
    perm_p = permutation_test(a, b, n_permutations=n_permutations)

    return ComparisonResult(
        config_a=stats_a.config_label,
        config_b=stats_b.config_label,
        metric_name=stats_a.metric_name,
        mean_diff=round(mean_diff, 4),
        effect_size_cohens_d=effect,
        mann_whitney_u=u_stat,
        mann_whitney_p=round(u_p, 6),
        permutation_p=round(perm_p, 6),
        n_permutations=n_permutations,
        significant_at_005=u_p < 0.05,
        correction_method="none",
        values_a=list(stats_a.values),
        values_b=list(stats_b.values),
    )


def coefficient_of_variation(values: List[float]) -> Optional[float]:
    """Coefficient of variation (std / mean).

    Returns None if N < 2 or mean is zero.
    Interpretation: CoV < 0.15 is stable; > 0.25 is high variance.
    """
    if len(values) < 2:
        return None
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    if mean == 0:
        return None
    std = float(np.std(arr, ddof=1))
    return round(std / mean, 4)


def krippendorff_alpha(
    ratings: List[List[Optional[float]]],
    level_of_measurement: str = "interval",
) -> float:
    """Compute Krippendorff's alpha for inter-rater reliability.

    Measures agreement between multiple raters (judges) across units (reports).

    Args:
        ratings: Matrix of shape (n_raters, n_units).
                 Use None for missing values.
        level_of_measurement: 'interval' (default) or 'ordinal'.
            'interval' assumes equal spacing between score values.
            'ordinal' uses rank-based distance.

    Returns:
        Alpha value in [-1, 1]. Alpha >= 0.8 is good, 0.6-0.8 is
        acceptable, < 0.6 is unreliable for the given criterion.

    Reference:
        Krippendorff, K. (2004). Content analysis: An introduction to
        its methodology (2nd ed.). Sage.
    """
    if not ratings or not ratings[0]:
        return float("nan")

    n_raters = len(ratings)
    n_units = len(ratings[0])

    # Collect coincidence matrix
    # Only consider pairs where both raters rated the same unit
    values: List[float] = []
    for unit in range(n_units):
        for rater in range(n_raters):
            v = ratings[rater][unit]
            if v is not None:
                values.append(v)

    if len(values) < 2:
        return float("nan")

    unique_vals = sorted(set(values))
    n_vals = len(unique_vals)
    val_idx = {v: i for i, v in enumerate(unique_vals)}

    if n_vals == 1:
        return 1.0  # Perfect agreement (trivially)

    # Compute coincidence matrix o[g][k]
    o = np.zeros((n_vals, n_vals), dtype=float)
    n_pairable = 0

    for unit in range(n_units):
        unit_ratings = [
            ratings[r][unit]
            for r in range(n_raters)
            if ratings[r][unit] is not None
        ]
        m_u = len(unit_ratings)
        if m_u < 2:
            continue
        n_pairable += m_u * (m_u - 1)
        for r1 in range(m_u):
            for r2 in range(m_u):
                if r1 != r2:
                    g = val_idx[unit_ratings[r1]]
                    k = val_idx[unit_ratings[r2]]
                    o[g][k] += 1.0 / (m_u - 1)

    if n_pairable == 0:
        return float("nan")

    # Distance function
    def _distance(g: int, k: int) -> float:
        vg = unique_vals[g]
        vk = unique_vals[k]
        if level_of_measurement == "interval":
            return (vg - vk) ** 2
        elif level_of_measurement == "ordinal":
            # Ordinal: sum frequencies between g and k
            n_g = sum(o[g, :]) + sum(o[:, g])
            n_k = sum(o[k, :]) + sum(o[:, k])
            between = sum(
                (sum(o[c, :]) + sum(o[:, c]))
                for c in range(min(g, k), max(g, k) + 1)
            )
            return (between - (n_g + n_k) / 2) ** 2
        else:
            return float(g != k)  # nominal

    # Observed disagreement D_o
    n = float(sum(sum(o[g, :]) for g in range(n_vals)))
    D_o = 0.0
    for g in range(n_vals):
        for k in range(n_vals):
            D_o += o[g][k] * _distance(g, k)
    if n > 0:
        D_o /= n

    # Expected disagreement D_e (marginal distribution)
    n_g_total = np.array([sum(o[g, :]) for g in range(n_vals)])
    n_total = float(np.sum(n_g_total))
    D_e = 0.0
    for g in range(n_vals):
        for k in range(n_vals):
            D_e += n_g_total[g] * n_g_total[k] * _distance(g, k)
    if n_total > 1:
        D_e /= n_total * (n_total - 1)

    if D_e == 0:
        return 1.0 if D_o == 0 else 0.0

    alpha = 1.0 - D_o / D_e
    return round(float(alpha), 4)


def compute_krippendorff_per_criterion(
    judgments_by_judge: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Compute Krippendorff's alpha for each criterion across all judges.

    Args:
        judgments_by_judge: {judge_model: {unit_key: score}}.
            unit_key = f"{run_id}__{dataset}__{criterion}"

    Returns:
        {criterion_name: alpha}.
    """
    # Collect all unit keys and judges
    all_units: List[str] = sorted(
        {
            "__".join(k.split("__")[:-1])  # strip criterion suffix
            for v in judgments_by_judge.values()
            for k in v
        }
    )

    # Get criterion names from keys
    all_criteria: List[str] = sorted(
        {k.split("__")[-1] for v in judgments_by_judge.values() for k in v}
    )

    judges = list(judgments_by_judge.keys())
    results: Dict[str, float] = {}

    for criterion in all_criteria:
        # Build ratings matrix: rows=judges, cols=units
        ratings: List[List[Optional[float]]] = []
        for judge in judges:
            row: List[Optional[float]] = []
            for unit in all_units:
                key = f"{unit}__{criterion}"
                row.append(judgments_by_judge[judge].get(key))
            ratings.append(row)

        results[criterion] = krippendorff_alpha(ratings)

    return results


def _spearman_rho(xs: List[float], ys: List[float]) -> Optional[float]:
    """Spearman rank correlation with average-tie handling."""
    n = len(xs)
    if n < 3:
        return None

    def _rank(vals: List[float]) -> List[float]:
        sorted_idx = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[sorted_idx[j + 1]] == vals[sorted_idx[j]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _rank(xs), _rank(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return round(num / (dx * dy), 4)


def compute_spearman_per_criterion(
    judgments_by_judge: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Compute mean pairwise Spearman ρ per criterion across all judges.

    Unlike Krippendorff's α, Spearman ρ measures rank agreement and is
    insensitive to systematic calibration offsets between judges.

    Args:
        judgments_by_judge: {judge_model: {unit_key: score}}.
            unit_key = f"{run_id}__{dataset}__{criterion}"

    Returns:
        {criterion_name: mean_spearman_rho}
    """
    from itertools import combinations

    all_units: List[str] = sorted(
        {
            "__".join(k.split("__")[:-1])
            for v in judgments_by_judge.values()
            for k in v
        }
    )
    all_criteria: List[str] = sorted(
        {k.split("__")[-1] for v in judgments_by_judge.values() for k in v}
    )
    judges = list(judgments_by_judge.keys())
    results: Dict[str, float] = {}

    for criterion in all_criteria:
        pairwise_rhos: List[float] = []
        for j1, j2 in combinations(judges, 2):
            paired = [
                (
                    judgments_by_judge[j1][f"{u}__{criterion}"],
                    judgments_by_judge[j2][f"{u}__{criterion}"],
                )
                for u in all_units
                if f"{u}__{criterion}" in judgments_by_judge[j1]
                and f"{u}__{criterion}" in judgments_by_judge[j2]
            ]
            if len(paired) < 3:
                continue
            rho = _spearman_rho([p[0] for p in paired], [p[1] for p in paired])
            if rho is not None:
                pairwise_rhos.append(rho)
        if pairwise_rhos:
            results[criterion] = round(
                sum(pairwise_rhos) / len(pairwise_rhos), 4
            )

    return results


def holm_bonferroni(
    comparisons: List[ComparisonResult],
    alpha: float = 0.05,
) -> List[ComparisonResult]:
    """Apply Holm-Bonferroni step-down correction.

    Updates significant_at_005 and correction_method in place.
    Returns the same list with corrected significance.
    """
    if not comparisons:
        return comparisons

    n = len(comparisons)
    # Sort by p-value (using Mann-Whitney p)
    indexed = sorted(
        enumerate(comparisons),
        key=lambda x: x[1].mann_whitney_p,
    )

    for rank, (orig_idx, comp) in enumerate(indexed):
        adjusted_alpha = alpha / (n - rank)
        comp.significant_at_005 = comp.mann_whitney_p < adjusted_alpha
        comp.correction_method = "holm"

    return comparisons
