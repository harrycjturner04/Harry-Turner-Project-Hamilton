"""Automated ablation analysis: comparison generation and reporting."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .registry import RunRecord
from .rubrics import get_criterion_names
from .statistics import (
    ComparisonResult,
    ReplicationStats,
    coefficient_of_variation,
    compare_configurations,
    compute_krippendorff_per_criterion,
    compute_replication_stats,
    holm_bonferroni,
)
from .storage import EvalDB
from .visualization import (
    export_results_table_latex,
    export_significance_table_latex,
    plot_ablation_bar_chart,
    plot_effect_size_forest,
    plot_metric_distributions,
    plot_repeatability_violin,
    plot_rubric_radar,
    plot_score_heatmap,
    plot_stage_quality_progression,
)

logger = logging.getLogger(__name__)


def generate_comparison_pairs(
    config_labels: List[str],
    baseline_label: str,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Generate comparison pairs for ablation study.

    Returns:
        baseline_pairs: [(baseline, other), ...] for significance testing.
        all_pairs: all unique pairs for pairwise ranking.
    """
    baseline_pairs = []
    for label in config_labels:
        if label != baseline_label:
            baseline_pairs.append((baseline_label, label))

    all_pairs = [
        (a, b)
        for i, a in enumerate(config_labels)
        for b in config_labels[i + 1:]
    ]
    return baseline_pairs, all_pairs


def collect_config_scores(
    db: EvalDB,
    config_groups: Dict[str, List[RunRecord]],
    rubric_version: str = "v1",
) -> Dict[str, Dict[str, List[float]]]:
    """Collect per-config, per-metric score lists from the database.

    Returns:
        {config_label: {metric_name: [score_per_run, ...]}}.
    """
    criteria = get_criterion_names(rubric_version) + ["overall"]
    result = {}  # type: Dict[str, Dict[str, List[float]]]

    for label, runs in config_groups.items():
        result[label] = {c: [] for c in criteria}
        for run in runs:
            agg = db.get_aggregated_scores(run_id=run.run_id)
            if not agg:
                continue
            for row in agg:
                metric = row["metric_name"]
                if metric in result[label]:
                    result[label][metric].append(row["mean_score"])

    return result


def compute_all_replication_stats(
    config_scores: Dict[str, Dict[str, List[float]]],
) -> Dict[str, Dict[str, ReplicationStats]]:
    """Compute replication stats for all configs and metrics.

    Returns:
        {config_label: {metric_name: ReplicationStats}}.
    """
    result = {}  # type: Dict[str, Dict[str, ReplicationStats]]
    for label, metrics in config_scores.items():
        result[label] = {}
        for metric, values in metrics.items():
            if values:
                result[label][metric] = compute_replication_stats(
                    values, label, metric
                )
    return result


def run_statistical_comparisons(
    all_stats: Dict[str, Dict[str, ReplicationStats]],
    pairs: List[Tuple[str, str]],
    metric_name: str = "overall",
) -> List[ComparisonResult]:
    """Run significance tests for all configuration pairs."""
    comparisons = []
    for config_a, config_b in pairs:
        stats_a = all_stats.get(config_a, {}).get(metric_name)
        stats_b = all_stats.get(config_b, {}).get(metric_name)
        if stats_a is None or stats_b is None:
            logger.warning(
                "Missing stats for %s or %s on metric %s",
                config_a, config_b, metric_name,
            )
            continue
        if stats_a.n_runs < 2 or stats_b.n_runs < 2:
            logger.warning(
                "Insufficient runs for statistical comparison: "
                "%s has %d, %s has %d",
                config_a, stats_a.n_runs, config_b, stats_b.n_runs,
            )
            continue
        comp = compare_configurations(stats_a, stats_b)
        comparisons.append(comp)

    if comparisons:
        holm_bonferroni(comparisons)

    return comparisons


def collect_gate_metrics(
    db: EvalDB,
    config_groups: Dict[str, List[RunRecord]],
) -> Dict[str, Dict[str, List[float]]]:
    """Collect gate quality scores per config per stage (cleaning and analysis only).

    Cross-validation does not produce gate scores — it uses a different execution
    path without a retry/gate loop.  Its quality is captured separately via
    collect_cv_structural_metrics().

    Returns:
        {config_label: {stage: [mean_quality_score_per_run]}}
    Each entry in the inner list is the mean quality_score across all
    datasets for a single run (initial-attempt records only).
    """
    stages = ["cleaning", "analysis"]
    result: Dict[str, Dict[str, List[float]]] = {}

    for label, runs in config_groups.items():
        result[label] = {s: [] for s in stages}
        for run in runs:
            metrics = db.get_stage_gate_metrics(
                run_id=run.run_id, final_only=True
            )
            if not metrics:
                continue
            stage_scores: Dict[str, List[float]] = {s: [] for s in stages}
            for m in metrics:
                s = m.get("stage")
                q = m.get("quality_score")
                if s in stage_scores and q is not None:
                    stage_scores[s].append(float(q))
            for s in stages:
                if stage_scores[s]:
                    result[label][s].append(
                        sum(stage_scores[s]) / len(stage_scores[s])
                    )

    return result


def collect_cv_structural_metrics(
    db: EvalDB,
    config_groups: Dict[str, List[RunRecord]],
) -> Dict[str, Dict[str, List[float]]]:
    """Collect cross-validation structural metrics per config.

    Cross-validation quality is measured via verified_claims_count
    (higher = more claims verified) and cv_gaps_count (lower = fewer gaps).

    Returns:
        {config_label: {"verified_claims": [...], "cv_gaps": [...]}}
    Each list contains one mean value per run (averaged across datasets).
    """
    result: Dict[str, Dict[str, List[float]]] = {}

    _report_metrics = [
        "data_coverage_rate",
        "figure_reference_rate",
        "quantitative_grounding_rate",
    ]

    for label, runs in config_groups.items():
        result[label] = {
            "verified_claims": [],
            "cv_gaps": [],
            **{m: [] for m in _report_metrics},
        }
        for run in runs:
            rows = db.get_structural_metrics(run_id=run.run_id)
            if not rows:
                continue
            claims = [
                float(r["verified_claims_count"])
                for r in rows
                if r.get("verified_claims_count") is not None
            ]
            gaps = [
                float(r["cv_gaps_count"])
                for r in rows
                if r.get("cv_gaps_count") is not None
            ]
            if claims:
                result[label]["verified_claims"].append(
                    sum(claims) / len(claims)
                )
            if gaps:
                result[label]["cv_gaps"].append(
                    sum(gaps) / len(gaps)
                )
            for metric in _report_metrics:
                vals = [
                    float(r[metric])
                    for r in rows
                    if r.get(metric) is not None
                ]
                if vals:
                    result[label][metric].append(sum(vals) / len(vals))

    return result


def cross_wp_correlation(
    db: EvalDB,
    config_groups: Dict[str, List[RunRecord]],
    metric_name: str = "overall",
) -> Dict[str, Dict[str, Any]]:
    """Compute cross-WP correlation for each pair of configurations.

    For each config pair (A, B), computes Pearson r of per-dataset mean
    scores across shared datasets.  High r means dataset difficulty is
    a consistent factor across conditions — supports generalisation claims.

    Returns:
        {f"{config_a} vs {config_b}": {
            "pearson_r": float | None,
            "p_value": float | None,
            "n_datasets": int,
        }}
    """
    try:
        from scipy.stats import pearsonr as _pearsonr
    except ImportError:
        logger.warning("scipy not available; skipping cross-WP correlation")
        return {}

    # Build per-config per-dataset score lists across runs
    config_ds: Dict[str, Dict[str, List[float]]] = {}
    for label, runs in config_groups.items():
        config_ds[label] = {}
        for run in runs:
            for row in db.get_aggregated_scores(run_id=run.run_id):
                if row.get("metric_name") != metric_name:
                    continue
                ds = row.get("dataset_name") or "__run__"
                config_ds[label].setdefault(ds, []).append(
                    float(row["mean_score"])
                )

    # Average across runs for each dataset per config
    config_mean: Dict[str, Dict[str, float]] = {
        label: {
            ds: sum(vals) / len(vals)
            for ds, vals in ds_scores.items()
            if vals
        }
        for label, ds_scores in config_ds.items()
    }

    configs = list(config_groups.keys())
    result: Dict[str, Dict[str, Any]] = {}

    for i, config_a in enumerate(configs):
        for config_b in configs[i + 1:]:
            means_a = config_mean.get(config_a, {})
            means_b = config_mean.get(config_b, {})
            shared = sorted(set(means_a) & set(means_b))
            key = f"{config_a} vs {config_b}"

            if len(shared) < 3:
                result[key] = {
                    "pearson_r": None,
                    "p_value": None,
                    "n_datasets": len(shared),
                }
                continue

            a_vals = [means_a[ds] for ds in shared]
            b_vals = [means_b[ds] for ds in shared]
            try:
                r, p = _pearsonr(a_vals, b_vals)
                result[key] = {
                    "pearson_r": round(float(r), 4),
                    "p_value": round(float(p), 4),
                    "n_datasets": len(shared),
                }
            except Exception as exc:
                logger.warning("pearsonr failed for %s: %s", key, exc)
                result[key] = {
                    "pearson_r": None,
                    "p_value": None,
                    "n_datasets": len(shared),
                }

    return result


def aggregate_rankings_by_config(
    run_rankings: Dict[str, float],
    runs: List[RunRecord],
) -> Dict[str, float]:
    """Aggregate run-level ranking scores to config-level means.

    Args:
        run_rankings: {run_id: strength_score} from BT or Elo.
        runs: All RunRecord objects so we can map run_id -> config_label.

    Returns:
        {config_label: mean_strength_score}
    """
    id_to_label = {r.run_id: (r.run_label or r.run_id) for r in runs}
    config_scores: Dict[str, List[float]] = {}
    for run_id, score in run_rankings.items():
        label = id_to_label.get(run_id, run_id)
        config_scores.setdefault(label, []).append(score)
    return {
        label: round(sum(vals) / len(vals), 4)
        for label, vals in config_scores.items()
    }


def generate_ablation_report(
    baseline_label: str,
    all_stats: Dict[str, Dict[str, ReplicationStats]],
    comparisons: List[ComparisonResult],
    rankings: Optional[Dict[str, Dict[str, float]]],
    output_dir: Path,
    gate_stats: Optional[Dict[str, Dict[str, List[float]]]] = None,
    cross_wp_corr: Optional[Dict[str, Dict[str, Any]]] = None,
    cv_structural: Optional[Dict[str, Dict[str, List[float]]]] = None,
    per_dataset_scores: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    per_judge_scores: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    judge_agreement: Optional[Dict[str, Dict[str, float]]] = None,
) -> Path:
    """Generate a Markdown ablation analysis report.

    Returns path to the generated report.
    """
    import statistics as _stats

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "ablation_report.md"

    lines = [
        "# Ablation Study Report",
        "",
        f"**Baseline configuration:** `{baseline_label}`",
        f"**Configurations evaluated:** {len(all_stats)}",
        "",
        "---",
        "",
        "## 1. Overview (LLM Judge Scores)",
        "",
    ]

    # Overview table
    criteria = []
    if all_stats:
        first = next(iter(all_stats.values()))
        criteria = list(first.keys())

    if criteria:
        header = "| Configuration | " + " | ".join(criteria) + " |"
        sep = "|---" * (len(criteria) + 1) + "|"
        lines.extend([header, sep])

        for label, metrics in all_stats.items():
            cells = [f"`{label}`"]
            for c in criteria:
                if c in metrics:
                    s = metrics[c]
                    if s.std > 0:
                        cells.append(f"{s.mean:.2f} ± {s.std:.2f}")
                    else:
                        cells.append(f"{s.mean:.2f}")
                else:
                    cells.append("N/A")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # Statistical comparisons
    if comparisons:
        lines.extend([
            "## 2. Statistical Comparisons",
            "",
            "| Comparison | Mean Diff | Cohen's d | p-value | Sig. |",
            "|---|---|---|---|---|",
        ])
        for c in comparisons:
            p = c.mann_whitney_p
            if p < 0.001:
                sig = "***"
            elif p < 0.01:
                sig = "**"
            elif p < 0.05:
                sig = "*"
            else:
                sig = "n.s."
            lines.append(
                f"| {c.config_a} vs {c.config_b} | "
                f"{c.mean_diff:+.3f} | {c.effect_size_cohens_d:.3f} | "
                f"{p:.4f} | {sig} |"
            )
        lines.extend([
            "",
            f"*Correction method: {comparisons[0].correction_method}*",
            "",
        ])

    # Rankings — config-level aggregated scores
    if rankings:
        lines.extend([
            "## 3. Rankings (Config-Level)",
            "",
            "*Scores aggregated from run-level Bradley-Terry / Elo by averaging "
            "across replications per configuration.*",
            "",
        ])
        for method, scores in rankings.items():
            lines.append(f"### {method}")
            lines.append("")
            sorted_scores = sorted(
                scores.items(), key=lambda x: x[1], reverse=True
            )
            lines.append("| Rank | Configuration | Mean Strength |")
            lines.append("|---|---|---|")
            for rank, (label, score) in enumerate(sorted_scores, 1):
                lines.append(f"| {rank} | `{label}` | {score:.4f} |")
            lines.append("")

    # Gate quality scores — cleaning and analysis only
    if gate_stats:
        stages = ["cleaning", "analysis"]
        lines.extend([
            "## 4. Gate Quality Scores (Deterministic — Cleaning & Analysis)",
            "",
            "*Mean ± std of pipeline-internal quality scores (0–1) across "
            "replications. Initial gate attempt only. Cross-validation does not "
            "use a gate loop — see Section 4b for its quality proxy.*",
            "",
            "| Configuration | Cleaning | Analysis |",
            "|---|---|---|",
        ])
        for label, stage_data in gate_stats.items():
            cells = [f"`{label}`"]
            for s in stages:
                vals = stage_data.get(s, [])
                if vals:
                    m = sum(vals) / len(vals)
                    sd = (_stats.stdev(vals) if len(vals) > 1 else 0.0)
                    cells.append(
                        f"{m:.3f} ± {sd:.3f} (N={len(vals)})"
                        if sd > 0
                        else f"{m:.3f} (N={len(vals)})"
                    )
                else:
                    cells.append("N/A")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # Cross-validation structural quality proxy
    if cv_structural:
        lines.extend([
            "### 4b. Cross-Validation Quality (Structural Metrics)",
            "",
            "*Cross-validation runs without a retry/gate loop. Quality is "
            "measured by verified_claims_count (higher = better) and "
            "cv_gaps_count (lower = better), averaged across datasets per run.*",
            "",
            "| Configuration | Verified Claims (mean) | CV Gaps (mean) |",
            "|---|---|---|",
        ])

        def _fmt(vals: List[float], decimals: int = 1) -> str:
            if not vals:
                return "N/A"
            m = sum(vals) / len(vals)
            sd = (_stats.stdev(vals) if len(vals) > 1 else 0.0)
            fmt = f"{{:.{decimals}f}}"
            return (
                f"{fmt.format(m)} ± {fmt.format(sd)} (N={len(vals)})"
                if sd > 0
                else f"{fmt.format(m)} (N={len(vals)})"
            )

        for label, data in cv_structural.items():
            lines.append(
                f"| `{label}` | {_fmt(data.get('verified_claims', []))} "
                f"| {_fmt(data.get('cv_gaps', []))} |"
            )
        lines.append("")

        # Report-level deterministic metrics (4c)
        report_metric_keys = [
            "data_coverage_rate",
            "figure_reference_rate",
            "quantitative_grounding_rate",
        ]
        has_report_metrics = any(
            cv_structural[lbl].get(m)
            for lbl in cv_structural
            for m in report_metric_keys
        )
        if has_report_metrics:
            lines.extend([
                "### 4c. Report Quality Metrics (Deterministic)",
                "",
                "*Computed from report text without LLM calls. "
                "data_coverage_rate = fraction of dataset column names "
                "mentioned in the report (breadth); "
                "figure_reference_rate = fraction of generated figures "
                "explicitly cited (none = orphaned plots); "
                "quantitative_grounding_rate = fraction of analytical "
                "paragraphs containing at least one number (depth).*",
                "",
                "| Configuration | Data Coverage | Figure Reference "
                "| Quant. Grounding |",
                "|---|---|---|---|",
            ])
            for label, data in cv_structural.items():
                lines.append(
                    f"| `{label}` "
                    f"| {_fmt(data.get('data_coverage_rate', []), 3)} "
                    f"| {_fmt(data.get('figure_reference_rate', []), 3)} "
                    f"| {_fmt(data.get('quantitative_grounding_rate', []), 3)} |"
                )
            lines.append("")

    # Per-dataset breakdown
    if per_dataset_scores:
        lines.extend([
            "## 5. Per-Dataset Score Breakdown",
            "",
            "*Overall scores per configuration per dataset, averaged across "
            "all judges and replications.*",
            "",
        ])
        all_datasets = sorted({
            ds
            for cfg_data in per_dataset_scores.values()
            for ds in cfg_data
        })
        header = "| Configuration | " + " | ".join(all_datasets) + " |"
        sep = "|---" * (len(all_datasets) + 1) + "|"
        lines.extend([header, sep])
        for label, ds_scores in per_dataset_scores.items():
            cells = [f"`{label}`"]
            for ds in all_datasets:
                score = ds_scores.get(ds)
                cells.append(f"{score:.2f}" if score is not None else "N/A")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # Per-judge breakdown
    if per_judge_scores:
        lines.extend([
            "## 6. Per-Judge Score Breakdown",
            "",
            "*Overall scores per configuration per judge model, averaged across "
            "all datasets and replications. Large divergence between judges "
            "indicates calibration differences.*",
            "",
        ])
        all_judges = sorted({
            j
            for cfg_data in per_judge_scores.values()
            for j in cfg_data
        })
        header = "| Configuration | " + " | ".join(all_judges) + " |"
        sep = "|---" * (len(all_judges) + 1) + "|"
        lines.extend([header, sep])
        for label, j_scores in per_judge_scores.items():
            cells = [f"`{label}`"]
            for j in all_judges:
                score = j_scores.get(j)
                cells.append(f"{score:.2f}" if score is not None else "N/A")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # Inter-judge agreement
    if judge_agreement:
        lines.extend([
            "## 7. Inter-Judge Agreement",
            "",
            "α (Krippendorff): ≥ 0.8 = good, 0.6–0.8 = acceptable, "
            "< 0.6 = unreliable. Measures absolute score agreement.",
            "ρ (Spearman): rank correlation; insensitive to calibration "
            "offsets. ρ > 0 means judges agree on relative ordering even "
            "when absolute scores differ.",
            "",
            "| Criterion | α (Krippendorff) | ρ (Spearman) |",
            "|---|---|---|",
        ])
        for crit, vals in sorted(judge_agreement.items()):
            alpha = vals.get("alpha", float("nan"))
            spearman = vals.get("spearman", float("nan"))
            a_str = f"{alpha:.3f}" if alpha == alpha else "N/A"
            s_str = f"{spearman:.3f}" if spearman == spearman else "N/A"
            lines.append(f"| {crit} | {a_str} | {s_str} |")
        lines.append("")

    # Repeatability (CoV per metric)
    if all_stats:
        lines.extend([
            "## 8. Repeatability (Coefficient of Variation)",
            "",
            "CoV = std / mean across replications.  "
            "CoV < 0.15 = stable; > 0.25 = high variance.",
            "",
            "| Metric | Configuration | Mean | Std | CoV | Stable? |",
            "|---|---|---|---|---|---|",
        ])
        for metric in criteria:
            for label, metrics in all_stats.items():
                if metric not in metrics:
                    continue
                s = metrics[metric]
                cov = coefficient_of_variation(s.values)
                if cov is None:
                    cov_str, stable = "N/A", "—"
                else:
                    cov_str = f"{cov:.3f}"
                    stable = "Yes" if cov < 0.15 else (
                        "Moderate" if cov < 0.25 else "No"
                    )
                lines.append(
                    f"| {metric} | `{label}` | {s.mean:.3f} | "
                    f"{s.std:.3f} | {cov_str} | {stable} |"
                )
        lines.append("")

    # Cross-WP correlation
    if cross_wp_corr:
        lines.extend([
            "## 9. Cross-Dataset Correlation",
            "",
            "Pearson r of per-dataset mean scores across configurations.  "
            "High r (> 0.6) indicates dataset difficulty is a consistent "
            "factor (supports generalisation).",
            "",
            "| Pair | Pearson r | p-value | N datasets | Interpretation |",
            "|---|---|---|---|---|",
        ])
        for pair, data in cross_wp_corr.items():
            r = data.get("pearson_r")
            p = data.get("p_value")
            n = data.get("n_datasets", 0)
            if r is None:
                interp = "Insufficient data"
                r_str, p_str = "N/A", "N/A"
            else:
                r_str = f"{r:.3f}"
                p_str = f"{p:.3f}" if p is not None else "N/A"
                if abs(r) > 0.6:
                    interp = "High generalisation"
                elif abs(r) > 0.3:
                    interp = "Moderate generalisation"
                else:
                    interp = "Low generalisation"
            lines.append(
                f"| {pair} | {r_str} | {p_str} | {n} | {interp} |"
            )
        lines.append("")

    # Figure references
    lines.extend([
        "## 10. Figures",
        "",
        "See the `figures/` directory for:",
        "- `radar_chart.png` — Rubric criterion comparison",
        "- `score_heatmap.png` — Configuration x Criterion heatmap",
        "- `overall_bar_chart.png` — Overall scores with 95% CI",
        "- `overall_distributions.png` — Score distributions per config",
        "- `effect_size_forest.png` — Effect sizes with 95% CIs for each comparison",
        "- `stage_quality_progression.png` — Gate quality per stage (cleaning & analysis)",
        "- `repeatability_violin.png` — Score distributions across replications",
        "- `judge_agreement.png` — Krippendorff's α inter-judge agreement heatmap",
        "- `rankings.png` — Config-level ranking comparison",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved ablation report to %s", report_path)
    return report_path


def collect_per_dataset_scores(
    db: EvalDB,
    config_groups: Dict[str, List[RunRecord]],
    metric_name: str = "overall",
) -> Dict[str, Dict[str, float]]:
    """Collect per-dataset mean scores per config.

    Queries judgment rows directly so dataset_name is always populated.

    Returns:
        {config_label: {dataset_name: mean_score}}
    """
    result: Dict[str, Dict[str, Any]] = {}

    for label, runs in config_groups.items():
        ds_scores: Dict[str, List[float]] = {}
        for run in runs:
            for row in db.get_judgments(run_id=run.run_id):
                ds = row.get("dataset_name") or "__global__"
                scores_json = row.get("criterion_scores_json", "{}")
                try:
                    crit_scores = json.loads(scores_json)
                except Exception:
                    crit_scores = {}
                if metric_name == "overall":
                    score = row.get("overall_score")
                else:
                    score = crit_scores.get(metric_name)
                if score is not None:
                    ds_scores.setdefault(ds, []).append(float(score))
        result[label] = {
            ds: round(sum(vals) / len(vals), 3)
            for ds, vals in ds_scores.items()
            if vals
        }

    return result


def collect_per_judge_scores(
    db: EvalDB,
    config_groups: Dict[str, List[RunRecord]],
    metric_name: str = "overall",
) -> Dict[str, Dict[str, float]]:
    """Collect per-judge mean overall scores per config.

    Returns:
        {config_label: {judge_model: mean_score}}
    """
    result: Dict[str, Dict[str, Any]] = {}

    for label, runs in config_groups.items():
        judge_scores: Dict[str, List[float]] = {}
        for run in runs:
            for row in db.get_judgments(run_id=run.run_id):
                judge = row.get("judge_model", "unknown")
                if metric_name == "overall":
                    score = row.get("overall_score")
                else:
                    try:
                        crit_scores = json.loads(
                            row.get("criterion_scores_json", "{}")
                        )
                        score = crit_scores.get(metric_name)
                    except Exception:
                        score = None
                if score is not None:
                    judge_scores.setdefault(judge, []).append(float(score))
        result[label] = {
            judge: round(sum(vals) / len(vals), 3)
            for judge, vals in judge_scores.items()
            if vals
        }

    return result


def collect_judge_agreement(
    db: EvalDB,
    config_groups: Dict[str, List[RunRecord]],
) -> Dict[str, Dict[str, float]]:
    """Compute Krippendorff's α and Spearman ρ per criterion across all judges.

    Only uses artifact_type="report" judgments to avoid mixing in global report
    criteria (which use a different rubric and would contaminate the criterion
    list with unrelated keys).

    Returns:
        {criterion_name: {"alpha": float, "spearman": float}}
        Empty dict if < 2 judges found.
    """
    from .statistics import (
        compute_krippendorff_per_criterion,
        compute_spearman_per_criterion,
    )

    # Build {judge_model: {unit_key: score}} — report artifact only
    judgments_by_judge: Dict[str, Dict[str, float]] = {}

    for runs in config_groups.values():
        for run in runs:
            for row in db.get_judgments(
                run_id=run.run_id, artifact_type="report"
            ):
                judge = row.get("judge_model", "unknown")
                ds = row.get("dataset_name") or "__global__"
                try:
                    crit_scores = json.loads(
                        row.get("criterion_scores_json", "{}")
                    )
                except Exception:
                    crit_scores = {}
                for crit, score in crit_scores.items():
                    if score is not None:
                        unit_key = f"{run.run_id}__{ds}__{crit}"
                        judgments_by_judge.setdefault(
                            judge, {}
                        )[unit_key] = float(score)
                overall = row.get("overall_score")
                if overall is not None:
                    unit_key = f"{run.run_id}__{ds}__overall"
                    judgments_by_judge.setdefault(
                        judge, {}
                    )[unit_key] = float(overall)

    if len(judgments_by_judge) < 2:
        logger.warning(
            "Only %d judge(s) found; inter-judge agreement requires >= 2.",
            len(judgments_by_judge),
        )
        return {}

    alpha_by_crit = compute_krippendorff_per_criterion(judgments_by_judge)
    spearman_by_crit = compute_spearman_per_criterion(judgments_by_judge)

    result: Dict[str, Dict[str, float]] = {}
    for crit in set(alpha_by_crit) | set(spearman_by_crit):
        result[crit] = {
            "alpha": alpha_by_crit.get(crit, float("nan")),
            "spearman": spearman_by_crit.get(crit, float("nan")),
        }
    return result


def cross_wp_correlation_from_judgments(
    db: EvalDB,
    config_groups: Dict[str, List[RunRecord]],
    metric_name: str = "overall",
) -> Dict[str, Dict[str, Any]]:
    """Compute cross-dataset correlation using per-dataset judgment scores.

    Queries judgment rows directly (which always have dataset_name populated)
    rather than aggregated_scores rows (which may have NULL dataset_name).

    Returns same structure as cross_wp_correlation().
    """
    try:
        from scipy.stats import pearsonr as _pearsonr
    except ImportError:
        logger.warning("scipy not available; skipping cross-dataset correlation")
        return {}

    # Build per-config per-dataset score lists
    config_ds: Dict[str, Dict[str, List[float]]] = {}
    for label, runs in config_groups.items():
        config_ds[label] = {}
        for run in runs:
            for row in db.get_judgments(run_id=run.run_id):
                ds = row.get("dataset_name")
                if not ds or ds == "__global__":
                    continue
                if metric_name == "overall":
                    score = row.get("overall_score")
                else:
                    try:
                        score = json.loads(
                            row.get("criterion_scores_json", "{}")
                        ).get(metric_name)
                    except Exception:
                        score = None
                if score is not None:
                    config_ds[label].setdefault(ds, []).append(float(score))

    # Average per dataset per config
    config_mean: Dict[str, Dict[str, float]] = {
        label: {
            ds: sum(vals) / len(vals)
            for ds, vals in ds_scores.items()
            if vals
        }
        for label, ds_scores in config_ds.items()
    }

    configs = list(config_groups.keys())
    result: Dict[str, Dict[str, Any]] = {}

    for i, config_a in enumerate(configs):
        for config_b in configs[i + 1:]:
            means_a = config_mean.get(config_a, {})
            means_b = config_mean.get(config_b, {})
            shared = sorted(set(means_a) & set(means_b))
            key = f"{config_a} vs {config_b}"

            if len(shared) < 3:
                result[key] = {
                    "pearson_r": None,
                    "p_value": None,
                    "n_datasets": len(shared),
                }
                continue

            a_vals = [means_a[ds] for ds in shared]
            b_vals = [means_b[ds] for ds in shared]
            try:
                r, p = _pearsonr(a_vals, b_vals)
                result[key] = {
                    "pearson_r": round(float(r), 4),
                    "p_value": round(float(p), 4),
                    "n_datasets": len(shared),
                }
            except Exception as exc:
                logger.warning("pearsonr failed for %s: %s", key, exc)
                result[key] = {
                    "pearson_r": None,
                    "p_value": None,
                    "n_datasets": len(shared),
                }

    return result


def generate_all_figures(
    all_stats: Dict[str, Dict[str, ReplicationStats]],
    config_scores: Dict[str, Dict[str, List[float]]],
    comparisons: List[ComparisonResult],
    rankings: Optional[Dict[str, Dict[str, float]]],
    output_dir: Path,
    gate_stats: Optional[Dict[str, Dict[str, List[float]]]] = None,
    judge_agreement: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """Generate all visualization figures for the ablation study."""
    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Radar chart — mean scores per criterion per config
    criteria = get_criterion_names()
    radar_data = {}
    for label, metrics in all_stats.items():
        radar_data[label] = {
            c: metrics[c].mean for c in criteria if c in metrics
        }
    if radar_data:
        plot_rubric_radar(radar_data, fig_dir / "radar_chart.png")

    # 2. Score heatmap
    configs = list(all_stats.keys())
    if configs and criteria:
        matrix = np.array([
            [all_stats[c].get(m, ReplicationStats(
                c, m, 0, [], 0, 0, 0, 0, 0
            )).mean for m in criteria]
            for c in configs
        ])
        plot_score_heatmap(
            configs, criteria, matrix, fig_dir / "score_heatmap.png"
        )

    # 3. Bar chart — overall scores with CI (y-axis floor auto-adjusted)
    bar_data = {}
    for label, metrics in all_stats.items():
        if "overall" in metrics:
            s = metrics["overall"]
            bar_data[label] = {
                "mean": s.mean,
                "ci_lower": s.ci_lower,
                "ci_upper": s.ci_upper,
            }
    if bar_data:
        plot_ablation_bar_chart(bar_data, fig_dir / "overall_bar_chart.png")

    # 4. Distribution plots
    if "overall" in next(iter(config_scores.values()), {}):
        dist_data = {
            label: vals.get("overall", [])
            for label, vals in config_scores.items()
            if vals.get("overall")
        }
        if dist_data:
            plot_metric_distributions(
                dist_data, "overall",
                fig_dir / "overall_distributions.png",
            )

    # 5. Forest plot with CIs
    if comparisons:
        plot_effect_size_forest(
            comparisons, fig_dir / "effect_size_forest.png"
        )

    # 6. Rankings (config-level)
    if rankings:
        from .visualization import plot_ranking_comparison
        plot_ranking_comparison(rankings, fig_dir / "rankings.png")

    # 7. Stage quality progression (cleaning & analysis only)
    if gate_stats:
        plot_stage_quality_progression(
            gate_stats,
            fig_dir / "stage_quality_progression.png",
        )

    # 8. Repeatability violin
    if config_scores:
        plot_repeatability_violin(
            config_scores,
            fig_dir / "repeatability_violin.png",
        )

    # 9. Inter-judge agreement heatmap (uses alpha values only)
    if judge_agreement:
        from .visualization import plot_judge_agreement_heatmap
        agreement_data = {
            crit: {"all_judges": vals["alpha"]}
            for crit, vals in judge_agreement.items()
            if isinstance(vals, dict)
            and vals.get("alpha") == vals.get("alpha")  # skip NaN
        }
        if agreement_data:
            plot_judge_agreement_heatmap(
                agreement_data, fig_dir / "judge_agreement.png"
            )

    logger.info("Generated all figures in %s", fig_dir)


def generate_latex_exports(
    all_stats: Dict[str, Dict[str, ReplicationStats]],
    comparisons: List[ComparisonResult],
    output_dir: Path,
) -> None:
    """Generate LaTeX tables for dissertation inclusion."""
    export_dir = Path(output_dir) / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Results table
    table_data = {}
    for label, metrics in all_stats.items():
        table_data[label] = {
            m: {"mean": s.mean, "std": s.std}
            for m, s in metrics.items()
        }
    if table_data:
        export_results_table_latex(
            table_data, export_dir / "results_table.tex"
        )

    # Significance table
    if comparisons:
        export_significance_table_latex(
            comparisons, export_dir / "significance_table.tex"
        )
