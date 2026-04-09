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

    Y-axis starts at a sensible floor (min CI lower - padding) rather than 0,
    so small differences between configs are visually distinguishable.

    Args:
        config_stats: {config_label: {mean, ci_lower, ci_upper}}.
    """
    import matplotlib.pyplot as plt

    if not config_stats:
        return

    _ensure_dir(output_path)

    labels = list(config_stats.keys())
    means = [config_stats[l]["mean"] for l in labels]
    ci_lowers = [config_stats[l].get("ci_lower", config_stats[l]["mean"])
                 for l in labels]
    ci_uppers = [config_stats[l].get("ci_upper", config_stats[l]["mean"])
                 for l in labels]
    ci_low_err = [m - lo for m, lo in zip(means, ci_lowers)]
    ci_high_err = [hi - m for m, hi in zip(means, ci_uppers)]
    errors = [ci_low_err, ci_high_err]

    # Y-axis: start at floor that shows differences, cap at 10
    y_floor = max(0.0, min(ci_lowers) - 1.0)
    y_ceil = min(10.0, max(ci_uppers) + 1.0)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    x = np.arange(len(labels))
    bars = ax.bar(
        x, means, yerr=errors, capsize=5,
        bottom=0,  # bars always start at 0 visually for correct proportions
        color="steelblue", edgecolor="black", alpha=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(y_floor, y_ceil)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels above the error bar cap, not the bar top
    for bar, mean, hi_err in zip(bars, means, ci_high_err):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + hi_err + 0.08,
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


def _bootstrap_cohens_d_ci(
    values_a: List[float],
    values_b: List[float],
    n_bootstrap: int = 5000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> tuple:
    """Bootstrap 95% CI for Cohen's d (Hedges' g corrected).

    Returns (ci_lower, ci_upper).
    """
    import math

    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)

    if len(a) < 2 or len(b) < 2:
        return (0.0, 0.0)

    rng = np.random.default_rng(random_state)
    boot_ds = []
    for _ in range(n_bootstrap):
        a_s = rng.choice(a, size=len(a), replace=True)
        b_s = rng.choice(b, size=len(b), replace=True)
        n_a, n_b = len(a_s), len(b_s)
        pooled = math.sqrt(
            ((n_a - 1) * float(np.var(a_s, ddof=1))
             + (n_b - 1) * float(np.var(b_s, ddof=1)))
            / (n_a + n_b - 2)
        )
        if pooled == 0:
            boot_ds.append(0.0)
        else:
            d = (float(np.mean(a_s)) - float(np.mean(b_s))) / pooled
            correction = 1 - 3 / (4 * (n_a + n_b) - 9)
            boot_ds.append(d * correction)

    alpha = 1 - ci_level
    return (
        float(np.percentile(boot_ds, 100 * alpha / 2)),
        float(np.percentile(boot_ds, 100 * (1 - alpha / 2))),
    )


def plot_effect_size_forest(
    comparisons: List[Any],
    output_path: Path,
    title: str = "Effect Sizes (Cohen's d, Hedges' g corrected)",
) -> None:
    """Forest plot of effect sizes with 95% bootstrap CIs.

    Args:
        comparisons: List of ComparisonResult objects, each must have
            a ``values_a`` and ``values_b`` attribute (raw score lists)
            for CI computation, or falls back to point-only display.
    """
    import matplotlib.pyplot as plt

    if not comparisons:
        return

    _ensure_dir(output_path)

    labels = [f"{c.config_a} vs {c.config_b}" for c in comparisons]
    effects = [c.effect_size_cohens_d for c in comparisons]
    significants = [c.significant_at_005 for c in comparisons]

    # Compute bootstrap CIs where raw values are available
    ci_lowers, ci_uppers = [], []
    for c in comparisons:
        va = getattr(c, "values_a", None)
        vb = getattr(c, "values_b", None)
        if va and vb and len(va) >= 2 and len(vb) >= 2:
            lo, hi = _bootstrap_cohens_d_ci(va, vb)
        else:
            lo, hi = c.effect_size_cohens_d, c.effect_size_cohens_d
        ci_lowers.append(lo)
        ci_uppers.append(hi)

    fig, ax = plt.subplots(figsize=(9, max(4, len(comparisons) * 1.5)))
    y_pos = np.arange(len(comparisons))

    colors = ["forestgreen" if s else "steelblue" for s in significants]

    for i, (y, d, lo, hi, col) in enumerate(
        zip(y_pos, effects, ci_lowers, ci_uppers, colors)
    ):
        # CI line
        ax.plot([lo, hi], [y, y], color=col, linewidth=2, alpha=0.7)
        # Point estimate
        ax.scatter([d], [y], color=col, s=80, zorder=3,
                   marker="D" if significants[i] else "o")

    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")

    # Effect size threshold guidelines (x-axis annotation, not floating text)
    for threshold in [0.2, 0.5, 0.8]:
        for sign in [1, -1]:
            ax.axvline(
                x=sign * threshold, color="gray",
                linewidth=0.5, linestyle=":", alpha=0.5,
            )
    # Threshold labels on x-axis via secondary tick labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_vals = [0.2, 0.5, 0.8]
    ax2.set_xticks(tick_vals)
    ax2.set_xticklabels(["small", "medium", "large"], fontsize=7, color="gray")
    ax2.tick_params(length=0)

    # Tight y-limits around the data
    ax.set_ylim(-0.6, len(y_pos) - 0.4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Cohen's d (Hedges' g corrected) with 95% bootstrap CI")
    ax.set_title(title, fontweight="bold", pad=20)
    ax.legend(
        handles=[
            plt.scatter([], [], color="forestgreen", marker="D", s=60),
            plt.scatter([], [], color="steelblue", marker="o", s=60),
        ],
        labels=["Significant (p<0.05)", "Not significant"],
        loc="lower right",
        fontsize=8,
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

    min_n_violin = 5  # violin shape is meaningless below this

    # Violin plot — only when enough data points
    if all(len(d) >= min_n_violin for d in data):
        parts = ax.violinplot(data, showmeans=True, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("steelblue")
            pc.set_alpha(0.3)

    # Strip plot (individual points) — always shown
    for i, vals in enumerate(data, 1):
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter, vals,
            color="steelblue", alpha=0.8, s=50, zorder=3,
        )
        # Mean line when no violin
        if len(vals) < min_n_violin:
            mean_val = float(np.mean(vals))
            ax.hlines(mean_val, i - 0.25, i + 0.25,
                      colors="steelblue", linewidths=2, zorder=4)
            ax.text(i + 0.3, mean_val, f"{mean_val:.2f}",
                    va="center", fontsize=8, color="steelblue")

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
    """Bar chart comparing ranking methods, one subplot per method.

    Bradley-Terry (log-scale, ~0) and Elo (~1000) are on different scales
    so they get separate subplots rather than a shared axis.

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
    x = np.arange(len(configs))
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    n_methods = len(methods)
    fig, axes = plt.subplots(
        1, n_methods,
        figsize=(max(5, len(configs) * 1.5) * n_methods, 5),
        squeeze=False,
    )

    for col, method in enumerate(methods):
        ax = axes[0][col]
        scores = [rankings[method].get(c, 0) for c in configs]
        bar_colors = [colors[i % len(colors)] for i in range(len(configs))]
        bars = ax.bar(x, scores, color=bar_colors, edgecolor="black",
                      alpha=0.8, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha="right")
        ax.set_ylabel("Strength Score")
        ax.set_title(method, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(y=0, color="black", linewidth=0.6, linestyle="--")

        # Value labels
        for bar, score in zip(bars, scores):
            va = "bottom" if score >= 0 else "top"
            offset = 0.01 * (max(scores) - min(scores) or 1)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                score + (offset if score >= 0 else -offset),
                f"{score:.3f}", ha="center", va=va, fontsize=8,
            )

    fig.suptitle(title, fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ranking chart to %s", output_path)


def plot_stage_quality_progression(
    stage_data: Dict[str, Dict[str, List[float]]],
    output_path: Path,
    title: str = "Stage Quality Progression",
) -> None:
    """Line chart: mean gate quality_score per stage per configuration.

    Analogous to Figure 5 in Innes 2025 — shows how intermediate stage
    quality differs between the full-critic and no-critic conditions.

    Args:
        stage_data: {config_label: {stage_name: [quality_score_per_run]}}.
            stage_name order: cleaning, analysis, cross_validation.
    """
    import matplotlib.pyplot as plt

    if not stage_data:
        return

    _ensure_dir(output_path)

    # Only plot stages that have data in at least one config
    _candidate_stages = ["cleaning", "analysis", "cross_validation"]
    _candidate_labels = ["Cleaning", "Analysis", "Cross-Validation"]
    stages = []
    stage_labels = []
    for s, sl in zip(_candidate_stages, _candidate_labels):
        if any(stage_data[cfg].get(s) for cfg in stage_data):
            stages.append(s)
            stage_labels.append(sl)

    if not stages:
        return

    fig, ax = plt.subplots(figsize=(max(5, len(stages) * 2.5), 5))

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for i, (label, stage_scores) in enumerate(stage_data.items()):
        means = []
        stds = []
        for stage in stages:
            vals = stage_scores.get(stage, [])
            if vals:
                arr = np.array(vals, dtype=float)
                means.append(float(np.mean(arr)))
                stds.append(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0)
            else:
                means.append(float("nan"))
                stds.append(0.0)

        x = np.arange(len(stages))
        color = colors[i % len(colors)]
        ax.plot(x, means, "o-", label=label, color=color, linewidth=2,
                markersize=7)
        ax.fill_between(
            x,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.15, color=color,
        )

    ax.set_xticks(np.arange(len(stages)))
    ax.set_xticklabels(stage_labels, fontsize=11)
    ax.set_ylabel("Gate Quality Score (0\u20131)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=1.0, color="gray", linewidth=0.8, linestyle="--",
               label="_nolegend_")

    # Annotate each point with mean
    for label, stage_scores in stage_data.items():
        for j, stage in enumerate(stages):
            vals = stage_scores.get(stage, [])
            if vals:
                mean = float(np.mean(vals))
                if not np.isnan(mean):
                    ax.annotate(
                        f"{mean:.2f}", (j, mean),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color="gray",
                    )

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved stage quality progression plot to %s", output_path)


def plot_repeatability_violin(
    config_values: Dict[str, Dict[str, List[float]]],
    output_path: Path,
    title: str = "Repeatability: Score Distribution Across Replications",
    metrics: Optional[List[str]] = None,
) -> None:
    """Violin plot showing score spread across N replications per config.

    Narrow violins indicate stable/reproducible output; wide violins indicate
    high stochastic variance between identical runs.

    Args:
        config_values: {config_label: {metric_name: [score_per_run]}}.
        metrics: Which metrics to include. Defaults to all metrics in data.
    """
    import matplotlib.pyplot as plt

    if not config_values:
        return

    _ensure_dir(output_path)

    # Determine metrics to show
    all_metrics: List[str] = []
    for vals in config_values.values():
        for m in vals:
            if m not in all_metrics:
                all_metrics.append(m)
    if metrics:
        plot_metrics = [m for m in metrics if m in all_metrics]
    else:
        plot_metrics = all_metrics

    if not plot_metrics:
        return

    configs = list(config_values.keys())
    n_metrics = len(plot_metrics)
    n_configs = len(configs)

    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(max(6, n_metrics * 3), max(4, n_configs * 0.8 + 2)),
        sharey=False,
    )
    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for ax, metric in zip(axes, plot_metrics):
        data = []
        xlabels = []
        for i, cfg in enumerate(configs):
            vals = config_values[cfg].get(metric, [])
            if vals:
                data.append(vals)
                xlabels.append(cfg)

        if not data:
            ax.set_visible(False)
            continue

        min_n_violin = 5  # violin shape is meaningless below this

        # Violin — only when all groups have enough points
        if all(len(d) >= min_n_violin for d in data):
            parts = ax.violinplot(data, showmeans=True, showmedians=True)
            for j, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors[j % len(colors)])
                pc.set_alpha(0.3)

        # Strip plot — always shown
        for j, vals in enumerate(data, 1):
            jitter = np.random.default_rng(42 + j).uniform(
                -0.08, 0.08, size=len(vals)
            )
            ax.scatter(
                np.full(len(vals), j) + jitter, vals,
                color=colors[(j - 1) % len(colors)],
                alpha=0.8, s=45, zorder=3,
            )
            # Mean line when no violin
            if len(vals) < min_n_violin:
                mean_val = float(np.mean(vals))
                ax.hlines(
                    mean_val, j - 0.25, j + 0.25,
                    colors=colors[(j - 1) % len(colors)],
                    linewidths=2, zorder=4,
                )

        # CoV annotation
        for j, vals in enumerate(data, 1):
            arr = np.array(vals, dtype=float)
            if np.mean(arr) > 0 and len(arr) > 1:
                cov = float(np.std(arr, ddof=1) / np.mean(arr))
                ax.text(
                    j, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else max(vals),
                    f"CoV={cov:.2f}", ha="center", va="bottom",
                    fontsize=7, color="dimgray",
                )

        ax.set_xticks(range(1, len(xlabels) + 1))
        ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=9)
        ax.set_title(metric.replace("_", " ").title(), fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved repeatability violin plot to %s", output_path)


def plot_judge_agreement_heatmap(
    agreement_data: Dict[str, Dict[str, float]],
    output_path: Path,
    title: str = "Inter-Judge Agreement (Krippendorff\u2019s \u03b1)",
) -> None:
    """Heatmap of Krippendorff's alpha per criterion per judge pair.

    Args:
        agreement_data: {criterion: {judge_pair_label: alpha_value}}.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if not agreement_data:
        return

    _ensure_dir(output_path)

    criteria = list(agreement_data.keys())
    pairs = list({p for vals in agreement_data.values() for p in vals})
    pairs.sort()

    matrix = np.full((len(criteria), len(pairs)), np.nan)
    for i, criterion in enumerate(criteria):
        for j, pair in enumerate(pairs):
            val = agreement_data[criterion].get(pair)
            if val is not None:
                matrix[i, j] = val

    # Colour map: red < 0.6, amber 0.6-0.8, green >= 0.8
    cmap = LinearSegmentedColormap.from_list(
        "agreement",
        [(0.0, "#d73027"), (0.6, "#fee090"), (0.8, "#91cf60"), (1.0, "#1a9850")],
    )

    fig, ax = plt.subplots(
        figsize=(max(6, len(pairs) * 1.5), max(3, len(criteria) * 0.7))
    )
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(pairs)))
    ax.set_yticks(np.arange(len(criteria)))
    ax.set_xticklabels(pairs, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(criteria, fontsize=9)

    for i in range(len(criteria)):
        for j in range(len(pairs)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.4 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=9)
            else:
                ax.text(j, i, "N/A", ha="center", va="center",
                        color="gray", fontsize=8)

    ax.set_title(title, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Krippendorff's α")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved judge agreement heatmap to %s", output_path)


def plot_deterministic_vs_llm_scatter(
    data_points: List[Dict[str, Any]],
    output_path: Path,
    x_metric: str = "check_pass_rate_must_fix",
    y_metric: str = "claim_evidence_consistency",
    title: Optional[str] = None,
) -> None:
    """Scatter plot: deterministic metric (x) vs LLM judge metric (y).

    Used to validate LLM judges against objective signals. A positive
    correlation (r > 0.5) supports judge validity.

    Args:
        data_points: List of dicts with keys for x_metric and y_metric.
        x_metric: Key for x-axis (deterministic).
        y_metric: Key for y-axis (LLM judge score).
    """
    import matplotlib.pyplot as plt

    xs = [d[x_metric] for d in data_points
          if d.get(x_metric) is not None and d.get(y_metric) is not None]
    ys = [d[y_metric] for d in data_points
          if d.get(x_metric) is not None and d.get(y_metric) is not None]

    if len(xs) < 3:
        logger.warning(
            "Insufficient data points for scatter plot (%d)", len(xs)
        )
        return

    _ensure_dir(output_path)

    xs_arr = np.array(xs, dtype=float)
    ys_arr = np.array(ys, dtype=float)

    # Pearson r
    from scipy.stats import pearsonr
    r, p = pearsonr(xs_arr, ys_arr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(xs_arr, ys_arr, alpha=0.7, color="steelblue", s=50, zorder=3)

    # Regression line
    m, b = np.polyfit(xs_arr, ys_arr, 1)
    x_line = np.linspace(xs_arr.min(), xs_arr.max(), 50)
    ax.plot(x_line, m * x_line + b, "r--", linewidth=1.5, alpha=0.8)

    ax.set_xlabel(x_metric.replace("_", " ").title(), fontsize=11)
    ax.set_ylabel(y_metric.replace("_", " ").title(), fontsize=11)
    ax.set_title(
        title or f"Deterministic vs LLM: {x_metric} vs {y_metric}",
        fontweight="bold",
    )
    p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
    ax.annotate(
        f"Pearson r={r:.2f}, {p_str}",
        xy=(0.05, 0.95), xycoords="axes fraction",
        fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                  edgecolor="gray", alpha=0.8),
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved scatter plot to %s", output_path)


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
