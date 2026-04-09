"""CLI entry point for the evaluation framework.

Usage::

    # Ingest gate scores and report metrics (no LLM calls, run first)
    python -m Scripts.evaluation.cli deterministic --glob "Outputs/slurm_*"

    # Evaluate all runs with LLM judges
    python -m Scripts.evaluation.cli evaluate --glob "Outputs/slurm_*"

    # Pairwise comparisons
    python -m Scripts.evaluation.cli pairwise --glob "Outputs/slurm_*"

    # Ablation analysis (includes gate metrics and cross-WP correlation)
    python -m Scripts.evaluation.cli ablation --baseline "baseline-v2" \\
        --glob "Outputs/slurm_*"

    # Repeatability (CoV) table
    python -m Scripts.evaluation.cli repeatability

    # Export for dissertation
    python -m Scripts.evaluation.cli export --format latex

    # Full pipeline (deterministic → evaluate → pairwise → ablation)
    python -m Scripts.evaluation.cli full --glob "Outputs/slurm_*" \\
        --baseline "baseline-v2"
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ablation import (
    aggregate_rankings_by_config,
    collect_config_scores,
    collect_cv_structural_metrics,
    collect_gate_metrics,
    collect_judge_agreement,
    collect_per_dataset_scores,
    collect_per_judge_scores,
    compute_all_replication_stats,
    cross_wp_correlation_from_judgments,
    generate_ablation_report,
    generate_all_figures,
    generate_comparison_pairs,
    generate_latex_exports,
    run_statistical_comparisons,
)
from .deterministic import ingest_run_deterministic
from .artifacts import index_run_artifacts
from .config import EvalConfig, build_openrouter_client, load_config
from .judge import (
    JudgmentResult,
    multi_judge_evaluate,
    multi_judge_evaluate_global,
    response_hash,
    save_raw_judgment,
)
from .pairwise import (
    PairwiseResult,
    compute_bradley_terry,
    compute_elo_ratings,
    pairwise_compare,
    pairwise_compare_with_swap,
)
from .pairwise import response_hash as pairwise_response_hash
from .registry import RunRecord, discover_runs, group_by_config
from .rubrics import get_criterion_names
from .scoring import aggregate_judgments
from .storage import EvalDB

logger = logging.getLogger(__name__)


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _resolve_runs(
    args: argparse.Namespace,
) -> List[RunRecord]:
    """Discover runs from CLI args."""
    run_dirs = getattr(args, "run_dirs", None) or []
    glob_pattern = getattr(args, "glob", None)

    if run_dirs:
        return discover_runs(".", run_dirs=[Path(d) for d in run_dirs])
    elif glob_pattern:
        import glob as _glob
        dirs = sorted(
            Path(d) for d in _glob.glob(glob_pattern) if Path(d).is_dir()
        )
        return discover_runs(".", run_dirs=dirs)
    else:
        return discover_runs("Outputs")


def _register_runs(db: EvalDB, runs: List[RunRecord]) -> None:
    """Register all discovered runs in the database."""
    for r in runs:
        db.upsert_run(
            run_id=r.run_id,
            run_dir=str(r.run_dir),
            run_label=r.run_label,
            pipeline_mode=r.pipeline_mode,
            model=r.model,
            timestamp=r.timestamp,
            batch_id=r.batch_id,
            run_config=r.run_config,
            datasets=r.datasets,
            file_count=r.file_count,
            stages_all_complete=r.stages_all_complete,
            context_md_hash=r.context_md_hash,
            git_commit_hash=r.git_commit_hash,
        )
        # Register structural metrics
        if r.judge_input:
            for f in r.judge_input.get("files", []):
                stages = f.get("stages_completed", {})
                db.upsert_structural_metrics(
                    run_id=r.run_id,
                    dataset_name=f.get("name", ""),
                    plot_count=f.get("plot_count", 0),
                    verified_claims_count=f.get(
                        "verified_claims_count", 0
                    ),
                    cv_gaps_count=len(f.get("cv_gaps", [])),
                    stages=stages,
                )


def _store_judgments(
    db: EvalDB,
    results: List[JudgmentResult],
    config: EvalConfig,
) -> None:
    """Store judgment results in the database and save raw JSON."""
    for result in results:
        rhash = response_hash(result.raw_response)
        db.insert_judgment(
            run_id=result.run_id,
            dataset_name=result.dataset_name,
            artifact_type=result.artifact_type,
            judge_model=result.judge_model,
            judge_pass=result.judge_pass,
            rubric_version=result.rubric_version,
            overall_score=result.overall_score,
            criterion_scores=result.criterion_scores,
            criterion_explanations=result.criterion_explanations,
            raw_response_hash=rhash,
            latency_ms=result.latency_ms,
            tokens_used=result.tokens_used,
        )
        save_raw_judgment(result, config.output_dir)


def _store_aggregated(
    db: EvalDB,
    run_id: str,
    results: List[JudgmentResult],
    method: str = "weighted_mean",
) -> None:
    """Aggregate and store scores for a run."""
    if not results:
        return
    agg = aggregate_judgments(results, method=method)
    for metric_name, stats in agg.items():
        db.upsert_aggregated_score(
            run_id=run_id,
            metric_name=metric_name,
            mean_score=stats["mean"],
            n_judgments=stats["n"],
            aggregation_method=method,
            std_score=stats.get("std"),
            min_score=stats.get("min"),
            max_score=stats.get("max"),
        )


# ── Commands ─────────────────────────────────────────────────────────────────


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run LLM-as-judge evaluation on pipeline outputs."""
    config = load_config(args.config)
    runs = _resolve_runs(args)
    if not runs:
        print("No runs found.", file=sys.stderr)
        return 1

    print(f"Evaluating {len(runs)} runs with "
          f"{len(config.judge_models)} judge(s)...")

    client = build_openrouter_client(config)

    with EvalDB(config.evaluation_db_path) as db:
        _register_runs(db, runs)

        for run in runs:
            idx = index_run_artifacts(run.run_dir)

            for ds_name, arts in idx.datasets.items():
                # Skip if already evaluated
                if args.skip_existing:
                    all_exist = all(
                        db.judgment_exists(
                            run.run_id, ds_name, "report",
                            jm.model_id, p,
                            config.rubric_version,
                        )
                        for jm in config.judge_models
                        for p in range(config.num_judge_passes)
                    )
                    if all_exist:
                        print(f"  Skipping {run.run_id}/{ds_name} "
                              f"(already evaluated)")
                        continue

                print(f"  Evaluating {run.run_id}/{ds_name}...")
                results = multi_judge_evaluate(
                    run_id=run.run_id,
                    dataset_name=ds_name,
                    artifacts=arts,
                    config=config,
                    client=client,
                )
                if results:
                    _store_judgments(db, results, config)
                    _store_aggregated(db, run.run_id, results)
                    scores = results[0].criterion_scores
                    print(f"    Overall: {results[0].overall_score:.2f} "
                          f"({len(results)} judgment(s))")

            # ── Global report evaluation ────────────────────────────
            if idx.global_report_path:
                if args.skip_existing:
                    all_global_exist = all(
                        db.judgment_exists(
                            run.run_id, "__global__",
                            "global_report",
                            jm.model_id, p,
                            "global_v1",
                        )
                        for jm in config.judge_models
                        for p in range(config.num_judge_passes)
                    )
                    if all_global_exist:
                        print(f"  Skipping {run.run_id}/global "
                              f"(already evaluated)")
                        continue

                print(f"  Evaluating {run.run_id}/global report...")
                global_results = multi_judge_evaluate_global(
                    run_id=run.run_id,
                    index=idx,
                    config=config,
                    client=client,
                )
                if global_results:
                    _store_judgments(db, global_results, config)
                    _store_aggregated(
                        db, run.run_id, global_results
                    )
                    print(
                        f"    Global overall: "
                        f"{global_results[0].overall_score:.2f} "
                        f"({len(global_results)} judgment(s))"
                    )

    print("Evaluation complete.")
    return 0


def cmd_pairwise(args: argparse.Namespace) -> int:
    """Run pairwise comparisons between runs."""
    config = load_config(args.config)
    runs = _resolve_runs(args)
    if len(runs) < 2:
        print("Need at least 2 runs for pairwise comparison.",
              file=sys.stderr)
        return 1

    print(f"Running pairwise comparisons across {len(runs)} runs...")

    client = build_openrouter_client(config)

    with EvalDB(config.evaluation_db_path) as db:
        _register_runs(db, runs)

        # Generate all pairs
        max_pairs = getattr(args, "max_pairs", None)
        pairs = [
            (runs[i], runs[j])
            for i in range(len(runs))
            for j in range(i + 1, len(runs))
        ]
        if max_pairs and len(pairs) > max_pairs:
            pairs = pairs[:max_pairs]

        for run_a, run_b in pairs:
            idx_a = index_run_artifacts(run_a.run_dir)
            idx_b = index_run_artifacts(run_b.run_dir)

            # Compare on shared datasets
            shared = set(idx_a.datasets) & set(idx_b.datasets)
            for ds_name in shared:
                from .artifacts import load_text_artifact
                report_a = load_text_artifact(
                    idx_a.datasets[ds_name].report_path,
                    config.max_report_chars,
                )
                report_b = load_text_artifact(
                    idx_b.datasets[ds_name].report_path,
                    config.max_report_chars,
                )
                if not report_a or not report_b:
                    continue

                for judge_model in config.judge_models:
                    print(f"  Comparing {run_a.run_id} vs "
                          f"{run_b.run_id} on {ds_name} "
                          f"({judge_model.display_name})...")
                    try:
                        result = pairwise_compare_with_swap(
                            run_a_id=run_a.run_id,
                            run_b_id=run_b.run_id,
                            report_a=report_a,
                            report_b=report_b,
                            dataset_name=ds_name,
                            judge_model=judge_model,
                            client=client,
                            max_chars=config.max_report_chars,
                        )
                        rhash = pairwise_response_hash(
                            result.primary.raw_response
                        )
                        explanation = result.primary.explanation
                        if result.inconsistency_note:
                            explanation = (
                                f"{explanation}\n\n"
                                f"[Position-swap: {result.inconsistency_note}]"
                            )
                        db.insert_pairwise(
                            run_a_id=result.primary.run_a_id,
                            run_b_id=result.primary.run_b_id,
                            dataset_name=ds_name,
                            judge_model=result.primary.judge_model,
                            winner=result.winner,
                            confidence=result.primary.confidence,
                            explanation=explanation,
                            criterion_preferences=result.primary.criterion_preferences,
                            raw_response_hash=rhash,
                            presentation_order="swap_check",
                        )
                        status = (
                            f"consistent → {result.winner}"
                            if result.consistent
                            else "inconsistent → tie"
                        )
                        print(f"    {status} "
                              f"(confidence: {result.primary.confidence:.2f})")
                    except Exception as exc:
                        logger.error("Pairwise failed: %s", exc)

        # Compute rankings
        pairwise_results_raw = db.get_pairwise_comparisons()
        if pairwise_results_raw:
            pw_results = [
                PairwiseResult(
                    run_a_id=r["run_a_id"],
                    run_b_id=r["run_b_id"],
                    dataset_name=r["dataset_name"],
                    judge_model=r["judge_model"],
                    winner=r["winner"],
                    confidence=r["confidence"],
                    explanation=r["explanation"],
                    criterion_preferences=json.loads(
                        r["criterion_preferences_json"]
                    ) if r["criterion_preferences_json"] else None,
                    raw_response="",
                    presentation_order=r["presentation_order"],
                )
                for r in pairwise_results_raw
            ]

            set_hash = hashlib.sha256(
                json.dumps(
                    [(r.run_a_id, r.run_b_id, r.winner)
                     for r in pw_results]
                ).encode()
            ).hexdigest()[:16]

            bt = compute_bradley_terry(pw_results)
            elo = compute_elo_ratings(pw_results)

            db.upsert_rankings(bt, "bradley_terry", set_hash)
            db.upsert_rankings(elo, "elo", set_hash)

            print("\nBradley-Terry rankings:")
            for label, score in sorted(
                bt.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  {label}: {score:.4f}")

            print("\nElo ratings:")
            for label, score in sorted(
                elo.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  {label}: {score:.1f}")

    print("Pairwise comparisons complete.")
    return 0


def cmd_ablation(args: argparse.Namespace) -> int:
    """Generate ablation comparison report."""
    config = load_config(args.config)
    runs = _resolve_runs(args)
    if not runs:
        print("No runs found.", file=sys.stderr)
        return 1

    baseline = args.baseline
    groups = group_by_config(runs)

    if baseline not in groups:
        print(f"Baseline label '{baseline}' not found. "
              f"Available: {list(groups.keys())}", file=sys.stderr)
        return 1

    output_dir = Path(getattr(args, "output_dir", None)
                      or config.output_dir)

    with EvalDB(config.evaluation_db_path) as db:
        _register_runs(db, runs)

        config_scores = collect_config_scores(
            db, groups, config.rubric_version
        )
        all_stats = compute_all_replication_stats(config_scores)

        config_labels = list(groups.keys())
        baseline_pairs, all_pairs = generate_comparison_pairs(
            config_labels, baseline
        )

        comparisons = run_statistical_comparisons(
            all_stats, all_pairs
        )

        # Get run-level rankings from DB and aggregate to config-level
        rankings = {}
        bt_raw = db.get_rankings("bradley_terry")
        if bt_raw:
            bt_run_scores = {r["run_label"]: r["strength_score"] for r in bt_raw}
            rankings["Bradley-Terry"] = aggregate_rankings_by_config(
                bt_run_scores, runs
            )
        elo_raw = db.get_rankings("elo")
        if elo_raw:
            elo_run_scores = {r["run_label"]: r["strength_score"] for r in elo_raw}
            rankings["Elo"] = aggregate_rankings_by_config(
                elo_run_scores, runs
            )

        # Collect deterministic gate metrics (cleaning & analysis only)
        gate_stats = collect_gate_metrics(db, groups)
        # Cross-validation structural quality proxy
        cv_structural = collect_cv_structural_metrics(db, groups)
        # Cross-dataset correlation using per-dataset judgments
        corr = cross_wp_correlation_from_judgments(
            db, groups, metric_name="overall"
        )
        # Per-dataset and per-judge breakdowns
        per_dataset = collect_per_dataset_scores(db, groups)
        per_judge = collect_per_judge_scores(db, groups)
        # Inter-judge agreement
        judge_agreement = collect_judge_agreement(db, groups)

        # Store stats in DB
        for label, metrics in all_stats.items():
            for metric_name, stats in metrics.items():
                db.upsert_replication_stat(
                    config_label=label,
                    metric_name=metric_name,
                    n_runs=stats.n_runs,
                    mean_val=stats.mean,
                    std_val=stats.std,
                    ci_lower=stats.ci_lower,
                    ci_upper=stats.ci_upper,
                    median_val=stats.median,
                )

        for comp in comparisons:
            db.upsert_statistical_comparison(
                config_a=comp.config_a,
                config_b=comp.config_b,
                metric_name=comp.metric_name,
                mean_diff=comp.mean_diff,
                effect_size_cohens_d=comp.effect_size_cohens_d,
                mann_whitney_u=comp.mann_whitney_u,
                mann_whitney_p=comp.mann_whitney_p,
                permutation_p=comp.permutation_p,
                significant_after_correction=comp.significant_at_005,
                correction_method=comp.correction_method,
            )

    # Generate report
    report_path = generate_ablation_report(
        baseline, all_stats, comparisons,
        rankings if rankings else None, output_dir,
        gate_stats=gate_stats or None,
        cross_wp_corr=corr or None,
        cv_structural=cv_structural or None,
        per_dataset_scores=per_dataset or None,
        per_judge_scores=per_judge or None,
        judge_agreement=judge_agreement or None,
    )

    # Generate figures
    generate_all_figures(
        all_stats, config_scores, comparisons,
        rankings if rankings else None, output_dir,
        gate_stats=gate_stats or None,
        judge_agreement=judge_agreement or None,
    )

    # Generate LaTeX exports
    generate_latex_exports(all_stats, comparisons, output_dir)

    print(f"\nAblation report: {report_path}")
    print(f"Figures: {output_dir / 'figures'}")
    print(f"LaTeX: {output_dir / 'exports'}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export results for dissertation."""
    config = load_config(args.config)
    output_dir = Path(getattr(args, "output_dir", None)
                      or config.output_dir)

    with EvalDB(config.evaluation_db_path) as db:
        rep_stats = db.get_replication_stats()
        comparisons_raw = db.get_statistical_comparisons()

    if not rep_stats:
        print("No replication stats found. "
              "Run 'ablation' command first.", file=sys.stderr)
        return 1

    # Reconstruct stats for export
    from .statistics import ReplicationStats, ComparisonResult

    all_stats = {}  # type: Dict[str, Dict[str, Any]]
    for row in rep_stats:
        label = row["config_label"]
        metric = row["metric_name"]
        all_stats.setdefault(label, {})[metric] = {
            "mean": row["mean_val"],
            "std": row["std_val"] or 0.0,
        }

    comparisons = []
    for row in comparisons_raw:
        comparisons.append(ComparisonResult(
            config_a=row["config_a"],
            config_b=row["config_b"],
            metric_name=row["metric_name"],
            mean_diff=row["mean_diff"],
            effect_size_cohens_d=row["effect_size_cohens_d"] or 0.0,
            mann_whitney_u=row["mann_whitney_u"] or 0.0,
            mann_whitney_p=row["mann_whitney_p"] or 1.0,
            permutation_p=row["permutation_p"] or 1.0,
            n_permutations=10000,
            significant_at_005=bool(
                row["significant_after_correction"]
            ),
            correction_method=row["correction_method"] or "none",
        ))

    fmt = getattr(args, "format", "latex")
    if fmt == "latex":
        generate_latex_exports_from_dict(all_stats, comparisons, output_dir)
    elif fmt == "csv":
        _export_csv(all_stats, comparisons, output_dir)
    elif fmt == "json":
        _export_json(all_stats, comparisons, output_dir)

    print(f"Exported to {output_dir / 'exports'}")
    return 0


def generate_latex_exports_from_dict(
    all_stats: Dict[str, Dict[str, Any]],
    comparisons: List[Any],
    output_dir: Path,
) -> None:
    """Generate LaTeX from dict-based stats."""
    export_dir = output_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    from .visualization import (
        export_results_table_latex,
        export_significance_table_latex,
    )

    if all_stats:
        export_results_table_latex(all_stats, export_dir / "results_table.tex")
    if comparisons:
        export_significance_table_latex(
            comparisons, export_dir / "significance_table.tex"
        )


def _export_csv(
    all_stats: Dict[str, Dict[str, Any]],
    comparisons: List[Any],
    output_dir: Path,
) -> None:
    """Export as CSV files."""
    import csv

    export_dir = output_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    if all_stats:
        configs = list(all_stats.keys())
        metrics = list(all_stats[configs[0]].keys()) if configs else []
        with open(export_dir / "results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["config"] + metrics)
            for config in configs:
                row = [config]
                for m in metrics:
                    s = all_stats[config].get(m, {})
                    mean = s.get("mean", 0)
                    std = s.get("std", 0)
                    row.append(f"{mean:.3f} +/- {std:.3f}")
                writer.writerow(row)

    if comparisons:
        with open(export_dir / "comparisons.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "config_a", "config_b", "mean_diff",
                "cohens_d", "p_value", "significant",
            ])
            for c in comparisons:
                writer.writerow([
                    c.config_a, c.config_b, f"{c.mean_diff:.4f}",
                    f"{c.effect_size_cohens_d:.4f}",
                    f"{c.mann_whitney_p:.6f}",
                    "yes" if c.significant_at_005 else "no",
                ])


def _export_json(
    all_stats: Dict[str, Dict[str, Any]],
    comparisons: List[Any],
    output_dir: Path,
) -> None:
    """Export as JSON."""
    export_dir = output_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "replication_stats": all_stats,
        "comparisons": [
            {
                "config_a": c.config_a,
                "config_b": c.config_b,
                "metric": c.metric_name,
                "mean_diff": c.mean_diff,
                "cohens_d": c.effect_size_cohens_d,
                "p_value": c.mann_whitney_p,
                "significant": c.significant_at_005,
            }
            for c in comparisons
        ],
    }
    path = export_dir / "results.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def cmd_deterministic(args: argparse.Namespace) -> int:
    """Ingest deterministic metrics (gate scores, report metrics) for all runs.

    Scans debug/ directories for __gate.json and __verdict.json files and
    computes data_coverage_rate, figure_reference_rate, and
    quantitative_grounding_rate for each report.  Idempotent.
    """
    config = load_config(args.config)
    runs = _resolve_runs(args)
    if not runs:
        print("No runs found.", file=sys.stderr)
        return 1

    print(f"Ingesting deterministic metrics for {len(runs)} run(s)...")

    with EvalDB(config.evaluation_db_path) as db:
        _register_runs(db, runs)
        for run in runs:
            print(f"  {run.run_id}...")
            ingest_run_deterministic(run, db)

    print("Deterministic ingestion complete.")
    return 0


def cmd_repeatability(args: argparse.Namespace) -> int:
    """Print a repeatability (CoV) table for all configurations and metrics."""
    from .statistics import coefficient_of_variation

    config = load_config(args.config)

    with EvalDB(config.evaluation_db_path) as db:
        rep_stats = db.get_replication_stats()

    if not rep_stats:
        print("No replication stats found. Run 'ablation' command first.",
              file=sys.stderr)
        return 1

    # Group by (config_label, metric_name)
    from collections import defaultdict
    grouped: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for row in rep_stats:
        grouped[row["config_label"]][row["metric_name"]] = row

    # Determine metric order
    all_metrics: List[str] = sorted(
        {row["metric_name"] for row in rep_stats}
    )

    print(f"\n{'Metric':<32} {'Config':<24} {'Mean':>7} {'Std':>7} "
          f"{'CoV':>7} {'Stable?':<10}")
    print("-" * 90)

    for metric in all_metrics:
        for label in sorted(grouped.keys()):
            row = grouped[label].get(metric)
            if row is None:
                continue
            mean = row["mean_val"] or 0.0
            std = row["std_val"] or 0.0
            # Reconstruct values list for CoV — approximate from std/mean
            # (exact values not stored, use std/mean ratio directly)
            if mean != 0 and std is not None:
                cov = round(std / mean, 4)
                stable = (
                    "Yes" if cov < 0.15
                    else "Moderate" if cov < 0.25
                    else "No"
                )
                cov_str = f"{cov:.3f}"
            else:
                cov_str = "N/A"
                stable = "—"
            print(
                f"{metric:<32} {label:<24} {mean:>7.3f} {std:>7.3f} "
                f"{cov_str:>7} {stable:<10}"
            )

    return 0


def cmd_rankings(args: argparse.Namespace) -> int:
    """Show current configuration rankings."""
    config = load_config(args.config)

    with EvalDB(config.evaluation_db_path) as db:
        method = getattr(args, "method", "both")
        methods = (
            ["bradley_terry", "elo"] if method == "both"
            else [method]
        )

        for m in methods:
            rankings = db.get_rankings(m)
            if not rankings:
                print(f"No {m} rankings found.")
                continue
            print(f"\n{m.replace('_', ' ').title()} Rankings:")
            print(f"{'Rank':<6}{'Configuration':<30}{'Score':<10}")
            print("-" * 46)
            for r in rankings:
                print(
                    f"{r['rank_position']:<6}"
                    f"{r['run_label']:<30}"
                    f"{r['strength_score']:<10.4f}"
                )

    return 0


def cmd_full(args: argparse.Namespace) -> int:
    """Run the full evaluation pipeline."""
    print("=" * 60)
    print("FULL EVALUATION PIPELINE")
    print("=" * 60)

    # Step 0: Ingest deterministic metrics (gate scores, report metrics)
    print("\n--- Step 0: Deterministic Metric Ingestion ---")
    ret = cmd_deterministic(args)
    if ret != 0:
        print("  (deterministic ingestion failed — continuing anyway)")

    # Step 1: Evaluate
    print("\n--- Step 1: LLM-as-Judge Evaluation ---")
    ret = cmd_evaluate(args)
    if ret != 0:
        return ret

    # Step 2: Pairwise (if enabled and >1 run)
    config = load_config(args.config)
    runs = _resolve_runs(args)
    if config.pairwise_enabled and len(runs) > 1:
        print("\n--- Step 2: Pairwise Comparisons ---")
        ret = cmd_pairwise(args)
        if ret != 0:
            return ret
    else:
        print("\n--- Step 2: Pairwise Comparisons (skipped) ---")

    # Step 3: Ablation analysis
    if hasattr(args, "baseline") and args.baseline:
        print("\n--- Step 3: Ablation Analysis ---")
        ret = cmd_ablation(args)
        if ret != 0:
            return ret

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    return 0


# ── CLI Parser ───────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluation",
        description="Evaluation framework for automated ablation studies",
    )
    parser.add_argument(
        "--config", "-c",
        default="evaluation_config.yaml",
        help="Path to evaluation config YAML",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    subparsers = parser.add_subparsers(dest="command")

    # evaluate
    eval_p = subparsers.add_parser(
        "evaluate", help="Run LLM-as-judge evaluation"
    )
    eval_p.add_argument("run_dirs", nargs="*")
    eval_p.add_argument("--glob", "-g")
    eval_p.add_argument(
        "--skip-existing", action="store_true",
        help="Skip runs already evaluated",
    )

    # pairwise
    pair_p = subparsers.add_parser(
        "pairwise", help="Run pairwise comparisons"
    )
    pair_p.add_argument("run_dirs", nargs="*")
    pair_p.add_argument("--glob", "-g")
    pair_p.add_argument("--max-pairs", type=int)

    # ablation
    abl_p = subparsers.add_parser(
        "ablation", help="Generate ablation analysis"
    )
    abl_p.add_argument("run_dirs", nargs="*")
    abl_p.add_argument("--glob", "-g")
    abl_p.add_argument("--baseline", required=True)
    abl_p.add_argument("--output-dir")

    # export
    exp_p = subparsers.add_parser(
        "export", help="Export results for dissertation"
    )
    exp_p.add_argument(
        "--format", choices=["latex", "csv", "json"], default="latex"
    )
    exp_p.add_argument("--output-dir")

    # deterministic
    det_p = subparsers.add_parser(
        "deterministic",
        help="Ingest gate scores and report metrics (no LLM calls)",
    )
    det_p.add_argument("run_dirs", nargs="*")
    det_p.add_argument("--glob", "-g")

    # repeatability
    subparsers.add_parser(
        "repeatability",
        help="Print CoV repeatability table for all configurations",
    )

    # rankings
    rank_p = subparsers.add_parser(
        "rankings", help="Show configuration rankings"
    )
    rank_p.add_argument(
        "--method",
        choices=["bradley_terry", "elo", "both"],
        default="both",
    )

    # full
    full_p = subparsers.add_parser(
        "full", help="Run full evaluation pipeline"
    )
    full_p.add_argument("run_dirs", nargs="*")
    full_p.add_argument("--glob", "-g")
    full_p.add_argument("--baseline", required=True)
    full_p.add_argument(
        "--skip-existing", action="store_true",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    _setup_logging(args.log_level)

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "evaluate": cmd_evaluate,
        "pairwise": cmd_pairwise,
        "ablation": cmd_ablation,
        "export": cmd_export,
        "rankings": cmd_rankings,
        "deterministic": cmd_deterministic,
        "repeatability": cmd_repeatability,
        "full": cmd_full,
    }

    cmd_func = commands.get(args.command)
    if cmd_func is None:
        parser.print_help()
        return 1

    return cmd_func(args)


if __name__ == "__main__":
    raise SystemExit(main())
