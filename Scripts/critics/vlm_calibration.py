"""VLM Calibration Framework — offline tool for validating VLM plot reviews.

WP-C4: NOT called during pipeline execution.  Used for experimental validation
to measure VLM reviewer accuracy against human-labelled ground truth plots.

Usage:
    python -m critics.vlm_calibration \\
        --ground-truth calibration_set.json \\
        --vlm-url http://localhost:8000 \\
        --vlm-model Qwen/Qwen3.5-27B \\
        --output vlm_calibration_results.json

Ground truth format (calibration_set.json):
    [
        {
            "plot_path": "/path/to/plot.png",
            "human_ratings": {
                "chart_type_appropriateness": "good",
                "axis_scaling": "acceptable",
                "grouping_correctness": "good",
                "statistical_annotations": "poor",
                ...
            },
            "overall_human_score": "acceptable"
        },
        ...
    ]
"""
from __future__ import annotations

import base64
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("captain_pipeline")

_SCORE_VALUES = {"good": 2, "acceptable": 1, "poor": 0}


def run_calibration(
    ground_truth_path: Path,
    vlm_url: str,
    vlm_model: str,
    review_prompt: str,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run VLM calibration against human-labelled ground truth.

    Returns calibration results including per-criterion confusion matrices,
    accuracy, and systematic bias indicators.
    """
    import requests

    with open(ground_truth_path) as fh:
        ground_truth = json.load(fh)

    if not isinstance(ground_truth, list):
        raise ValueError("Ground truth must be a JSON array of plot entries")

    results: List[Dict[str, Any]] = []
    per_criterion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # per_criterion[criterion_name]["TP_good"] etc.

    for entry in ground_truth:
        plot_path = Path(entry["plot_path"])
        human_ratings = entry.get("human_ratings", {})

        if not plot_path.exists():
            logger.warning("Calibration: plot not found: %s", plot_path)
            continue

        # Send to VLM
        with open(plot_path, "rb") as fh:
            img_b64 = base64.b64encode(fh.read()).decode()

        try:
            resp = requests.post(
                f"{vlm_url}/chat/completions",
                json={
                    "model": vlm_model,
                    "messages": [
                        {"role": "system", "content": review_prompt},
                        {"role": "user", "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                            {"type": "text",
                             "text": f"Review this plot: {plot_path.name}"},
                        ]},
                    ],
                    "max_tokens": 800,
                },
                timeout=120,
            )
            msg = resp.json()["choices"][0]["message"]["content"]

            # Parse VLM response
            # Strip think tokens if present
            import re
            msg = re.sub(r"<think>.*?</think>", "", msg, flags=re.DOTALL).strip()
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', msg, re.DOTALL)
            if not json_match:
                logger.warning("Calibration: no JSON in VLM response for %s", plot_path.name)
                continue
            parsed = json.loads(json_match.group())

        except Exception as exc:
            logger.warning("Calibration: VLM call failed for %s: %s", plot_path.name, exc)
            continue

        # Compare VLM ratings to human ratings
        vlm_ratings: Dict[str, str] = {}
        for criterion in parsed.get("criteria", []):
            if isinstance(criterion, dict):
                vlm_ratings[criterion.get("name", "")] = criterion.get("score", "")

        entry_result = {
            "plot": plot_path.name,
            "human": human_ratings,
            "vlm": vlm_ratings,
            "matches": {},
        }

        for criterion_name, human_score in human_ratings.items():
            vlm_score = vlm_ratings.get(criterion_name, "")
            match = human_score.lower() == vlm_score.lower()
            entry_result["matches"][criterion_name] = match

            # Build confusion matrix entries
            key = f"{human_score.lower()}_vs_{vlm_score.lower()}"
            per_criterion[criterion_name][key] += 1

            if match:
                per_criterion[criterion_name]["correct"] += 1
            else:
                per_criterion[criterion_name]["incorrect"] += 1
            per_criterion[criterion_name]["total"] += 1

        results.append(entry_result)

    # Compute summary statistics
    summary: Dict[str, Any] = {
        "total_plots": len(results),
        "per_criterion_accuracy": {},
        "per_criterion_confusion": {},
        "systematic_biases": [],
    }

    for criterion_name, counts in per_criterion.items():
        total = counts.get("total", 0)
        correct = counts.get("correct", 0)
        accuracy = correct / total if total > 0 else 0.0
        summary["per_criterion_accuracy"][criterion_name] = round(accuracy, 3)
        summary["per_criterion_confusion"][criterion_name] = dict(counts)

        # Detect systematic biases
        # Count how often VLM is more lenient vs more strict
        lenient = 0
        strict = 0
        for key, count in counts.items():
            if "_vs_" in key:
                human, vlm = key.split("_vs_")
                h_val = _SCORE_VALUES.get(human, -1)
                v_val = _SCORE_VALUES.get(vlm, -1)
                if h_val >= 0 and v_val >= 0:
                    if v_val > h_val:
                        lenient += count
                    elif v_val < h_val:
                        strict += count

        if total > 0:
            lenient_rate = lenient / total
            strict_rate = strict / total
            if lenient_rate > 0.3:
                summary["systematic_biases"].append(
                    f"{criterion_name}: VLM is too lenient ({lenient_rate:.0%} overrates)"
                )
            if strict_rate > 0.3:
                summary["systematic_biases"].append(
                    f"{criterion_name}: VLM is too strict ({strict_rate:.0%} underrates)"
                )

    overall_accuracy = (
        sum(c.get("correct", 0) for c in per_criterion.values())
        / max(sum(c.get("total", 0) for c in per_criterion.values()), 1)
    )
    summary["overall_accuracy"] = round(overall_accuracy, 3)

    calibration_result = {
        "summary": summary,
        "detailed_results": results,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(calibration_result, fh, indent=2)
        logger.info("Calibration results written to %s", output_path)

    return calibration_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VLM Calibration Framework")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth JSON")
    parser.add_argument("--vlm-url", required=True, help="VLM endpoint URL")
    parser.add_argument("--vlm-model", default="Qwen/Qwen3.5-27B", help="VLM model name")
    parser.add_argument("--output", default="vlm_calibration_results.json", help="Output path")
    args = parser.parse_args()

    from prompts import SCIENTIFIC_VISUAL_REVIEW_PROMPT

    result = run_calibration(
        ground_truth_path=Path(args.ground_truth),
        vlm_url=args.vlm_url,
        vlm_model=args.vlm_model,
        review_prompt=SCIENTIFIC_VISUAL_REVIEW_PROMPT,
        output_path=Path(args.output),
    )
    print(f"Calibration complete: {result['summary']['overall_accuracy']:.1%} overall accuracy")
    for bias in result["summary"]["systematic_biases"]:
        print(f"  BIAS: {bias}")
