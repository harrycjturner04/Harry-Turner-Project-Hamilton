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
    compare_configurations,
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
    plot_rubric_radar,
    plot_score_heatmap,
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
    baseline_pairs: List[Tuple[str, str]],
    metric_name: str = "overall",
) -> List[ComparisonResult]:
    """Run significance tests for baseline vs each other config."""
    comparisons = []
    for config_a, config_b in baseline_pairs:
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


def generate_ablation_report(
    baseline_label: str,
    all_stats: Dict[str, Dict[str, ReplicationStats]],
    comparisons: List[ComparisonResult],
    rankings: Optional[Dict[str, Dict[str, float]]],
    output_dir: Path,
) -> Path:
    """Generate a Markdown ablation analysis report.

    Returns path to the generated report.
    """
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
        "## 1. Overview",
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
                        cells.append(f"{s.mean:.2f} +/- {s.std:.2f}")
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

    # Rankings
    if rankings:
        lines.extend(["## 3. Rankings", ""])
        for method, scores in rankings.items():
            lines.append(f"### {method}")
            lines.append("")
            sorted_scores = sorted(
                scores.items(), key=lambda x: x[1], reverse=True
            )
            lines.append("| Rank | Configuration | Strength |")
            lines.append("|---|---|---|")
            for rank, (label, score) in enumerate(sorted_scores, 1):
                lines.append(f"| {rank} | `{label}` | {score:.4f} |")
            lines.append("")

    # Figure references
    lines.extend([
        "## 4. Figures",
        "",
        "See the `figures/` directory for:",
        "- `radar_chart.png` — Rubric criterion comparison",
        "- `score_heatmap.png` — Configuration x Criterion heatmap",
        "- `overall_bar_chart.png` — Overall scores with 95% CI",
        "- `overall_distributions.png` — Score distributions per config",
        "- `effect_size_forest.png` — Effect sizes for each comparison",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved ablation report to %s", report_path)
    return report_path


def generate_all_figures(
    all_stats: Dict[str, Dict[str, ReplicationStats]],
    config_scores: Dict[str, Dict[str, List[float]]],
    comparisons: List[ComparisonResult],
    rankings: Optional[Dict[str, Dict[str, float]]],
    output_dir: Path,
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

    # 3. Bar chart — overall scores with CI
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

    # 5. Forest plot
    if comparisons:
        plot_effect_size_forest(
            comparisons, fig_dir / "effect_size_forest.png"
        )

    # 6. Rankings
    if rankings:
        from .visualization import plot_ranking_comparison
        plot_ranking_comparison(rankings, fig_dir / "rankings.png")

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
