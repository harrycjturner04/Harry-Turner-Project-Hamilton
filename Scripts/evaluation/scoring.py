"""Score aggregation across judges, passes, and datasets."""

import logging
from typing import Any, Dict, List, Optional

from .judge import JudgmentResult
from .rubrics import get_criterion_names, get_criterion_weights

logger = logging.getLogger(__name__)


def compute_weighted_score(
    criterion_scores: Dict[str, float],
    rubric_version: str = "v1",
) -> float:
    """Compute weighted overall score from criterion scores."""
    weights = get_criterion_weights(rubric_version)
    total = 0.0
    for name, weight in weights.items():
        total += weight * criterion_scores.get(name, 0.0)
    return round(total, 2)


def aggregate_judgments(
    judgments: List[JudgmentResult],
    method: str = "weighted_mean",
) -> Dict[str, Any]:
    """Aggregate scores from multiple judges and passes.

    Methods:
        weighted_mean: Weight by JudgeModelSpec.weight, average across passes.
        median: Median across all judgments.
        trimmed_mean: Drop highest and lowest, mean of rest (needs >= 4).

    Returns:
        Dict with per-criterion aggregated scores and 'overall'.
    """
    if not judgments:
        return {}

    criteria = get_criterion_names(judgments[0].rubric_version)

    if method == "weighted_mean":
        return _weighted_mean(judgments, criteria)
    elif method == "median":
        return _median(judgments, criteria)
    elif method == "trimmed_mean":
        return _trimmed_mean(judgments, criteria)
    else:
        raise ValueError(f"Unknown aggregation method: {method!r}")


def _weighted_mean(
    judgments: List[JudgmentResult],
    criteria: List[str],
) -> Dict[str, Any]:
    """Weighted mean aggregation."""
    # Collect weight per judge model (from config, default 1.0)
    # Since JudgmentResult doesn't carry weight, use uniform here.
    # The CLI layer can pass weighted judgments if needed.
    result = {}
    for name in criteria:
        values = [j.criterion_scores.get(name, 0.0) for j in judgments]
        n = len(values)
        mean = sum(values) / n if n > 0 else 0.0
        std = (
            (sum((v - mean) ** 2 for v in values) / n) ** 0.5
            if n > 1 else 0.0
        )
        result[name] = {
            "mean": round(mean, 2),
            "std": round(std, 2),
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
            "n": n,
        }

    overall_values = [j.overall_score for j in judgments]
    n = len(overall_values)
    overall_mean = sum(overall_values) / n if n > 0 else 0.0
    overall_std = (
        (sum((v - overall_mean) ** 2 for v in overall_values) / n) ** 0.5
        if n > 1 else 0.0
    )
    result["overall"] = {
        "mean": round(overall_mean, 2),
        "std": round(overall_std, 2),
        "min": min(overall_values) if overall_values else 0.0,
        "max": max(overall_values) if overall_values else 0.0,
        "n": n,
    }
    return result


def _median(
    judgments: List[JudgmentResult],
    criteria: List[str],
) -> Dict[str, Any]:
    """Median aggregation (robust to outlier judges)."""
    result = {}
    for name in criteria:
        values = sorted(
            j.criterion_scores.get(name, 0.0) for j in judgments
        )
        n = len(values)
        if n == 0:
            med = 0.0
        elif n % 2 == 1:
            med = values[n // 2]
        else:
            med = (values[n // 2 - 1] + values[n // 2]) / 2
        result[name] = {
            "mean": round(med, 2),  # 'mean' key used for consistency
            "std": 0.0,
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
            "n": n,
        }

    overall_values = sorted(j.overall_score for j in judgments)
    n = len(overall_values)
    if n == 0:
        med = 0.0
    elif n % 2 == 1:
        med = overall_values[n // 2]
    else:
        med = (overall_values[n // 2 - 1] + overall_values[n // 2]) / 2
    result["overall"] = {
        "mean": round(med, 2),
        "std": 0.0,
        "min": min(overall_values) if overall_values else 0.0,
        "max": max(overall_values) if overall_values else 0.0,
        "n": n,
    }
    return result


def _trimmed_mean(
    judgments: List[JudgmentResult],
    criteria: List[str],
) -> Dict[str, Any]:
    """Trimmed mean: drop highest and lowest, mean of rest."""
    if len(judgments) < 4:
        logger.warning(
            "trimmed_mean requires >=4 judgments, got %d; "
            "falling back to weighted_mean",
            len(judgments),
        )
        return _weighted_mean(judgments, criteria)

    result = {}
    for name in criteria:
        values = sorted(
            j.criterion_scores.get(name, 0.0) for j in judgments
        )
        trimmed = values[1:-1]
        n = len(trimmed)
        mean = sum(trimmed) / n if n > 0 else 0.0
        result[name] = {
            "mean": round(mean, 2),
            "std": 0.0,
            "min": min(trimmed) if trimmed else 0.0,
            "max": max(trimmed) if trimmed else 0.0,
            "n": len(judgments),
        }

    overall_values = sorted(j.overall_score for j in judgments)
    trimmed = overall_values[1:-1]
    n = len(trimmed)
    mean = sum(trimmed) / n if n > 0 else 0.0
    result["overall"] = {
        "mean": round(mean, 2),
        "std": 0.0,
        "min": min(trimmed) if trimmed else 0.0,
        "max": max(trimmed) if trimmed else 0.0,
        "n": len(judgments),
    }
    return result
