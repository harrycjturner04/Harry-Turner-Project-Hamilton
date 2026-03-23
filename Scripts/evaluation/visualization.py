"""Visualization module for ablation study figures."""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_rubric_radar(
    config_scores: Dict[str, Dict[str, float]],
    output_path: Path,
    title: str = "Rubric Score Comparison",
) -> None:
    """Radar/spider chart of rubric scores per configuration.

    Args:
        config_scores: {config_label: {criterion: mean_score}}.
    """
    import matplotlib.pyplot as plt

    if not config_scores:
        return

    _ensure_dir(output_path)

    # Get criteria from first config
    criteria = list(next(iter(config_scores.values())).keys())
    n_criteria = len(criteria)
    angles = np.linspace(0, 2 * np.pi, n_criteria, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for label, scores in config_scores.items():
        values = [scores.get(c, 0) for c in criteria]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=label)
        ax.fill(angles, values, alpha=0.1)

    ax.set_thetagrids(
        np.degrees(angles[:-1]), criteria, fontsize=9
    )
    ax.set_ylim(0, 10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved radar chart to %s", output_path)


def plot_ablation_bar_chart(
    config_stats: Dict[str, Dict[str, float]],
    output_path: Path,
    title: str = "Ablation Study: Overall Scores",
    ylabel: str = "Score",
) -> None:
    """Grouped bar chart with error bars (95% CI).

    Args:
        config_stats: {config_label: {mean, ci_lower, ci_upper}}.
    """
    import matplotlib.pyplot as plt

    if not config_stats:
        return

    _ensure_dir(output_path)

    labels = list(config_stats.keys())
    means = [config_stats[l]["mean"] for l in labels]
    ci_low = [
        config_stats[l]["mean"] - config_stats[l].get("ci_lower", 0)
        for l in labels
    ]
    ci_high = [
        config_stats[l].get("ci_upper", 0) - config_stats[l]["mean"]
        for l in labels
    ]
    errors = [ci_low, ci_high]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=errors, capsize=5, color="steelblue",
                  edgecolor="black", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, 10)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"{mean:.2f}", ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved bar chart to %s", output_path)


def plot_score_heatmap(
    config_names: List[str],
    criterion_names: List[str],
    scores_matrix: np.ndarray,
    output_path: Path,
    title: str = "Score Heatmap",
) -> None:
    """Heatmap of scores: rows = configurations, columns = criteria.

    Args:
        scores_matrix: shape (n_configs, n_criteria), values 0-10.
    """
    import matplotlib.pyplot as plt

    if scores_matrix.size == 0:
        return

    _ensure_dir(output_path)

    fig, ax = plt.subplots(
        figsize=(max(8, len(criterion_names) * 1.2),
                 max(4, len(config_names) * 0.8))
    )
    im = ax.imshow(scores_matrix, cmap="RdYlGn", vmin=0, vmax=10,
                   aspect="auto")

    ax.set_xticks(np.arange(len(criterion_names)))
    ax.set_yticks(np.arange(len(config_names)))
    ax.set_xticklabels(criterion_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(config_names, fontsize=10)

    # Annotate cells
    for i in range(len(config_names)):
        for j in range(len(criterion_names)):
            val = scores_matrix[i, j]
            color = "white" if val < 4 or val > 8 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    color=color, fontsize=9)

    ax.set_title(title, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Score (0-10)")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heatmap to %s", output_path)


def plot_pairwise_win_matrix(
    config_names: List[str],
    win_matrix: np.ndarray,
    output_path: Path,
    title: str = "Pairwise Win Rates",
) -> None:
    """NxN matrix showing pairwise win rates.

    Args:
        win_matrix: shape (N, N), cell (i,j) = fraction of times i beat j.
    """
    import matplotlib.pyplot as plt

    if win_matrix.size == 0:
        return

    _ensure_dir(output_path)

    n = len(config_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n)))
    im = ax.imshow(win_matrix, cmap="RdBu_r", vmin=0, vmax=1,
                   aspect="auto")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(config_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(config_names, fontsize=10)

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "-", ha="center", va="center", fontsize=10)
            else:
                val = win_matrix[i, j]
                color = "white" if abs(val - 0.5) > 0.3 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=9)

    ax.set_xlabel("Opponent")
    ax.set_ylabel("Configuration")
    ax.set_title(title, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Win Rate")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved win matrix to %s", output_path)


def plot_effect_size_forest(
    comparisons: List[Any],
    output_path: Path,
    title: str = "Effect Sizes (Cohen's d)",
) -> None:
    """Forest plot of effect sizes with CIs.

    Args:
        comparisons: List of ComparisonResult objects.
    """
    import matplotlib.pyplot as plt

    if not comparisons:
        return

    _ensure_dir(output_path)

    labels = [
        f"{c.config_a} vs {c.config_b}"
        for c in comparisons
    ]
    effects = [c.effect_size_cohens_d for c in comparisons]
    significants = [c.significant_at_005 for c in comparisons]

    fig, ax = plt.subplots(figsize=(8, max(4, len(comparisons) * 0.6)))
    y_pos = np.arange(len(comparisons))

    colors = ["forestgreen" if s else "gray" for s in significants]
    ax.barh(y_pos, effects, color=colors, edgecolor="black", alpha=0.7,
            height=0.6)
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")

    # Add effect size thresholds
    for threshold, label in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        ax.axvline(x=threshold, color="gray", linewidth=0.5,
                   linestyle=":", alpha=0.5)
        ax.axvline(x=-threshold, color="gray", linewidth=0.5,
                   linestyle=":", alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Cohen's d (Hedges' g corrected)")
    ax.set_title(title, fontweight="bold")
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="forestgreen", alpha=0.7),
            plt.Rectangle((0, 0), 1, 1, color="gray", alpha=0.7),
        ],
        labels=["Significant (p<0.05)", "Not significant"],
        loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved forest plot to %s", output_path)


def plot_metric_distributions(
    config_values: Dict[str, List[float]],
    metric_name: str,
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    """Violin + strip plot showing score distributions per configuration.

    Essential for N=5-10 where individual data points matter.
    """
    import matplotlib.pyplot as plt

    if not config_values:
        return

    _ensure_dir(output_path)

    labels = list(config_values.keys())
    data = [config_values[l] for l in labels]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))

    # Violin plot
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.3)

    # Strip plot (individual points)
    for i, vals in enumerate(data, 1):
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter, vals,
            color="steelblue", alpha=0.7, s=30, zorder=3,
        )

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(
        title or f"Score Distribution: {metric_name}",
        fontweight="bold",
    )
    ax.set_ylim(0, 10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved distribution plot to %s", output_path)


def plot_ranking_comparison(
    rankings: Dict[str, Dict[str, float]],
    output_path: Path,
    title: str = "Configuration Rankings",
) -> None:
    """Bar chart comparing Bradley-Terry and Elo rankings side by side.

    Args:
        rankings: {method_name: {config_label: score}}.
    """
    import matplotlib.pyplot as plt

    if not rankings:
        return

    _ensure_dir(output_path)

    methods = list(rankings.keys())
    all_configs = set()
    for scores in rankings.values():
        all_configs.update(scores.keys())
    configs = sorted(all_configs)

    n_methods = len(methods)
    n_configs = len(configs)
    x = np.arange(n_configs)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(max(8, n_configs * 1.5), 5))

    for i, method in enumerate(methods):
        scores = [rankings[method].get(c, 0) for c in configs]
        ax.bar(x + i * width, scores, width, label=method, alpha=0.8)

    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(configs, rotation=45, ha="right")
    ax.set_ylabel("Strength Score")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ranking chart to %s", output_path)


# ── LaTeX Export ─────────────────────────────────────────────────────────────


def export_results_table_latex(
    config_stats: Dict[str, Dict[str, Any]],
    output_path: Path,
    caption: str = "Ablation Study Results",
    label: str = "tab:ablation-results",
) -> None:
    """Export LaTeX table of mean +/- std per config x metric.

    Args:
        config_stats: {config_label: {metric_name: {mean, std}}}.
    """
    _ensure_dir(output_path)

    configs = list(config_stats.keys())
    if not configs:
        return

    metrics = list(config_stats[configs[0]].keys())

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\toprule",
        "Configuration & " + " & ".join(
            m.replace("_", r"\_") for m in metrics
        ) + r" \\",
        r"\midrule",
    ]

    for config in configs:
        cells = [config.replace("_", r"\_")]
        for metric in metrics:
            stats = config_stats[config].get(metric, {})
            mean = stats.get("mean", 0)
            std = stats.get("std", 0)
            if std > 0:
                cells.append(f"${mean:.2f} \\pm {std:.2f}$")
            else:
                cells.append(f"${mean:.2f}$")
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved LaTeX table to %s", output_path)


def export_significance_table_latex(
    comparisons: List[Any],
    output_path: Path,
    caption: str = "Statistical Comparisons",
    label: str = "tab:significance",
) -> None:
    """Export LaTeX significance table with stars."""
    _ensure_dir(output_path)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{llcccl}",
        r"\toprule",
        r"Config A & Config B & $\Delta\bar{x}$ & Cohen's $d$ "
        r"& $p$-value & Sig. \\",
        r"\midrule",
    ]

    for c in comparisons:
        p = c.mann_whitney_p
        if p < 0.001:
            stars = "***"
        elif p < 0.01:
            stars = "**"
        elif p < 0.05:
            stars = "*"
        else:
            stars = "n.s."

        ca = c.config_a.replace("_", r"\_")
        cb = c.config_b.replace("_", r"\_")
        lines.append(
            f"{ca} & {cb} & ${c.mean_diff:+.3f}$ & "
            f"${c.effect_size_cohens_d:.3f}$ & "
            f"${p:.4f}$ & {stars} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\multicolumn{6}{l}{"
        r"\footnotesize * $p<0.05$, ** $p<0.01$, *** $p<0.001$, "
        r"n.s. = not significant} \\",
        r"\end{tabular}",
        r"\end{table}",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved significance table to %s", output_path)
