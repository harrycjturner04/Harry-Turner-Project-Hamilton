"""Statistical analysis: bootstrap CI, significance tests, effect sizes."""

import logging
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

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
    )


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
