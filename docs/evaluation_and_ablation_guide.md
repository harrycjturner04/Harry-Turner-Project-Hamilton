# Evaluation & Ablation Guide

This guide covers the pipeline's ablation study system and the external evaluation framework. For pipeline architecture and configuration, see the main [README](../README.md).

---

## Ablation Studies & Run Comparison

The pipeline includes a built-in system for running controlled experiments. You can toggle individual improvements on or off via `context.md`, run the pipeline, and then compare results across runs using `compare_runs.py`. Every run self-documents its configuration in the manifest, so you always know exactly what was enabled.

### How Ablation Works

```
1. Edit context.md         ->  Set run_config toggles
2. Run the pipeline        ->  sbatch Batch_Script.sh  (or manual)
3. Output folder created   ->  Outputs/slurm_XXXX/manifest_*.json contains run_config
4. Repeat with different   ->  Change run_config, run again
   toggles
5. Compare results         ->  python Scripts/compare_runs.py --glob "Outputs/slurm_*"
```

The key insight: **only `context.md` changes between runs**. The code stays the same. Each feature is gated behind a toggle in the `run_config` block, so the same codebase can produce baseline or enhanced output depending on configuration.

### Configuring Ablation Runs

All toggles live under `pipeline.run_config` in your `context.md` YAML block. To switch between configurations, just edit this section before each run.

**Baseline run** (reproduces pre-improvement behaviour):
```yaml
pipeline:
  run_config:
    run_label: "baseline"
  # Everything else uses safe defaults: v1 prompts, critic auto-accepts,
  # tautological cross-validation, all figures included, etc.
```

**All improvements enabled**:
```yaml
pipeline:
  run_config:
    run_label: "all_WPs"
    prompt_version: "v2"
    content_validation: true
    min_cross_val_claims: 5
    recompute_cross_val: true
    figure_selection: "ranked"
    max_report_figures: 10
    rerun_issue_map_enabled: true
    protected_col_audit: true
    schema_profiling: "full"
    iteration_strategy: "convergent"
    visual_review_mode: "scientific"
    require_figure_references: true
    expert_library: "extended"
    critic_analytical_depth: true
    targeted_refinement: true
    critic_report: true
    numerical_accuracy_check: true
    min_quantitative_grounding: 0.70
```

**Test a single feature** (e.g., just enhanced prompts):
```yaml
pipeline:
  run_config:
    run_label: "v2_prompts_only"
    prompt_version: "v2"
    # Everything else stays at defaults
```

### Pre-Defined Run Configurations

| Run Label | What It Tests | Key Config |
|-----------|---------------|------------|
| `baseline` | Pre-improvement behaviour | `run_label: "baseline"` (all defaults) |
| `schema_only` | Schema intelligence (WP-1) | `schema_profiling: "roles_only"` |
| `convergent` | Quality-driven iteration (WP-2) | `iteration_strategy: "convergent"` |
| `scientific_review` | Scientific VLM review (WP-3) | `visual_review_mode: "scientific"`, `require_figure_references: true` |
| `extended_agents` | Pluggable agent library (WP-4) | `expert_library: "extended"` |
| `v2_prompts` | Enhanced prompts only | `prompt_version: "v2"` |
| `crossval` | Genuine cross-validation | `recompute_cross_val: true` |
| `depth_critic` | Analytical depth critic (WP-C3a) | `critic_analytical_depth: true` |
| `report_quality` | Report pipeline quality gate (WP-R) | `critic_report: true`, `numerical_accuracy_check: true`, `min_quantitative_grounding: 0.70` |
| `all_WPs` | Everything enabled | All toggles set to enhanced values |

### Comparing Runs

Use `compare_runs.py` to view results side-by-side:

```bash
# Compare specific runs
python Scripts/compare_runs.py Outputs/slurm_860423_* Outputs/slurm_860500_*

# Auto-discover all runs
python Scripts/compare_runs.py --glob "Outputs/slurm_*"

# Verbose mode: shows run_config and per-file breakdown
python Scripts/compare_runs.py --glob "Outputs/slurm_*" -v

# Save as CSV for external analysis
python Scripts/compare_runs.py --glob "Outputs/slurm_*" --output ablation_results.csv

# JSON output (for scripting)
python Scripts/compare_runs.py --glob "Outputs/slurm_*" --json
```

**Example output:**
```
run_dir                          | run_label  | pipeline_mode    | file_count | stages_all_complete | avg_plot_count | avg_verified_claims | overall_quality
---------------------------------+------------+------------------+------------+---------------------+----------------+---------------------+----------------
Outputs/slurm_860423_20260304... | baseline   | full_closed_loop | 2          | 2/2                 | 16.5           | 5.0                 | PASS
Outputs/slurm_860500_20260305... | all_WPs    | full_closed_loop | 2          | 2/2                 | 18.0           | 8.0                 | PASS
```

With `-v` (verbose), each run also shows:
- The full `run_config` JSON (exactly which toggles were active)
- Per-file breakdown (plots, claims, gaps per dataset)

### Evaluation with LLM-as-Judge

The ablation framework is designed to work with external LLM-as-Judge evaluation. Each run's manifest contains all the information needed for scoring:

**What to evaluate** (recommended dimensions):

| Dimension | What to Score | Affected By |
|-----------|---------------|-------------|
| Interpretation Depth | Do findings explain biological significance? | `prompt_version` |
| Statistical Rigor | Are p-values actual computed values? | `prompt_version` |
| Cross-Validation Validity | Are claims independently recomputed? | `recompute_cross_val` |
| Report Completeness | Are figures properly mapped and selected? | `figure_selection` |
| Quality Gate Effectiveness | Does the gated review loop catch real issues? | `iteration_strategy`, `convergence_target` |
| Cleaning Transparency | Are missingness warnings present? | `protected_col_audit` |
| Retry Effectiveness | Do retries succeed with targeted instructions? | `rerun_issue_map_enabled` |
| Schema Awareness | Does analysis use correct grouping and column roles? | `schema_profiling` |
| Figure Quality | Are plots scientifically appropriate (chart type, annotations)? | `visual_review_mode` |
| Agent Selection | Are the right experts activated for the data? | `expert_library` |
| Analytical Depth | Does the critic catch shallow analysis? | `critic_analytical_depth` |
| Report Narrative Quality | Is the report well-structured with grounded claims? | `critic_report`, `min_quantitative_grounding` |
| Numerical Accuracy | Do numbers in the report match the analysis artifacts? | `numerical_accuracy_check` |

**Where to find scoring inputs:**
- `manifest_batch_*.json` -> `run_config`, `files[].stages_completed`, `files[].verified_claims_count`
- `files/{dataset}/analysis/analysis_summary.json` -> `findings[]`, `per_run_per_stage`
- `files/{dataset}/cross_validation/` -> `verified_claims` with `match` and `recomputed` fields
- `reports/{dataset}__report.md` -> Full narrative report text
- `debug/*__content_eval.json` -> Content evaluator rubric scores and coverage ratio
- `debug/*__plot_eval.json` -> VLM plot quality check results
- `debug/*__quality_gate.json` -> Quality gate verdict with numeric score

---

## Evaluation Framework

The project includes a dedicated evaluation framework (`Scripts/evaluation/`) for automated, research-grade ablation studies. It acts as a **fully external wrapper** around the pipeline -- it reads only from `Outputs/` and never modifies pipeline behaviour.

The framework uses **LLM-as-judge** evaluation via [OpenRouter](https://openrouter.ai/), where configurable external models (e.g., GPT-4o, Claude Sonnet) score pipeline reports against a structured rubric. It supports replicated runs, pairwise head-to-head comparisons, statistical significance testing, and automated generation of dissertation-ready figures and LaTeX tables.

### Overview

```
Pipeline Runs (existing)          Evaluation Framework
-----                             -----
Outputs/                    --->  Scripts/evaluation/
  slurm_*/                          |-- Discover & index runs
    judge_input.json                |-- LLM-as-judge scoring (OpenRouter)
    manifest_batch_*.json           |-- Pairwise comparison
    files/*/analysis/               |-- Statistical analysis
    reports/*.md                    '-- Figures + LaTeX export
                                        |
                                  Evaluation/
                                    evaluation.db, figures/, exports/
```

**Design principles:**
- **Fully decoupled** -- zero imports from the pipeline code; reads only JSON/Markdown/PNG files
- **WP-agnostic** -- groups runs by `run_label` string, not by specific Work Package
- **Deterministic governance, non-deterministic analysis** -- Python code makes all gate decisions; LLMs handle qualitative evaluation

### Setup

1. **Copy the example config and add your OpenRouter API key:**

   ```bash
   cp evaluation_config.example.yaml evaluation_config.yaml
   ```

2. **Edit `evaluation_config.yaml`:**

   ```yaml
   openrouter_api_key: "sk-or-v1-YOUR-KEY-HERE"

   judge_models:
     - model_id: "openrouter/hunter-alpha"
       display_name: "hunter-alpha"
       temperature: 0.0
       max_tokens: 32768
       weight: 1.0
       supports_vision: true

     - model_id: "openrouter/healer-alpha"
       display_name: "healer-alpha"
       temperature: 0.0
       max_tokens: 32768
       weight: 1.0
       supports_vision: true
   ```

   All judge models are fully configurable. Any model on [OpenRouter](https://openrouter.ai/models) can be used. Models with `supports_vision: true` receive plot PNGs as base64-encoded images.

3. **No pipeline modifications required.**

### Running Evaluations

The recommended approach is to pass **explicit directories** rather than `--glob` when you only want to evaluate a subset of configurations. This prevents accidentally including incomplete or in-progress runs, and avoids mixing WP configurations you are not yet ready to compare.

```bash
# Evaluate specific runs only (recommended when comparing two configs)
python -m Scripts.evaluation evaluate \
    Outputs/slurm_16652393_20260404_123425 \
    Outputs/slurm_16652395_20260404_202806 \
    Outputs/slurm_16652396_20260405_012759 \
    Outputs/slurm_16654063_20260405_161550 \
    Outputs/slurm_16654064_20260405_205312 \
    Outputs/slurm_16654065_20260406_002128 \
    --skip-existing

# Full pipeline using explicit directories
python -m Scripts.evaluation full \
    Outputs/slurm_A Outputs/slurm_B Outputs/slurm_C \
    --baseline "Baseline" --skip-existing

# Alternatively, evaluate all discovered runs
python -m Scripts.evaluation evaluate --glob "Outputs/slurm_*" --skip-existing
```

> **Important:** Do **not** use `full` again on runs already evaluated — the pairwise step does not deduplicate and will corrupt BT/Elo rankings. Use `ablation` alone to regenerate the report from existing data after code changes.

```bash
# Regenerate report and figures only (no new LLM calls)
python -m Scripts.evaluation ablation \
    Outputs/slurm_A Outputs/slurm_B ... --baseline "Baseline"
```

### Replicated Runs

Because the pipeline produces **non-deterministic outputs**, the evaluation framework supports multiple runs per configuration for statistical robustness.

```bash
# Submit 5 replicate runs with the current context.md
bash Scripts/evaluation/launch_replicates.sh --replicates 5

# Submit 10 replicates with a specific context file
bash Scripts/evaluation/launch_replicates.sh --replicates 10 --context context_wp1.md

# Preview (no submission)
bash Scripts/evaluation/launch_replicates.sh --replicates 5 --dry-run
```

The launcher submits N copies of `Batch_Script.sh` via `sbatch`. The evaluation framework groups runs with matching `run_label` as replicates via `group_by_config()`.

**Recommended:** 5-10 replicates per configuration for reliable statistical comparisons.

### Evaluation Rubric

Reports are scored against a structured rubric with **7 criteria** on a 0-10 scale:

| Criterion | Weight | What It Measures |
|-----------|--------|------------------|
| `analytical_depth` | 0.20 | Per-group breakdown, multi-dimensional analysis, pattern identification |
| `statistical_reasoning` | 0.15 | Correct test selection, effect sizes, confidence intervals |
| `figure_quality` | 0.10 | Diversity, labelling, relevance, publication readiness |
| `claim_evidence_consistency` | 0.20 | Every claim traceable to computed values; cross-validation alignment |
| `report_clarity` | 0.10 | IMRAD structure, scientific prose quality, limitations section |
| `evidence_based_reasoning` | 0.15 | Regulatory citations (ICH, FDA), domain knowledge integration |
| `domain_correctness` | 0.10 | Biologics/chromatography/MS scientific accuracy |

Each criterion has **5 anchor points** (0, 3, 5, 7, 10) with concrete descriptions. Score 5 = adequate, 7 = good, 9+ = exceptional (rare).

**Noise reduction strategies:**
- Temperature = 0 for all judge calls
- Multi-model judging (weighted mean across models)
- Structured anchors reduce calibration drift
- Optional multi-pass averaging (`num_judge_passes` in config)
- Median aggregation option for robustness to outlier judges

### Pairwise Comparison

```bash
python -m Scripts.evaluation pairwise Outputs/slurm_A Outputs/slurm_B ...
```

For each pair, the judge receives both reports and answers: *"Which analysis is stronger and why?"*

**Position-bias mitigation:** Report order is randomised. With multiple judge models, each pair receives 4+ independent evaluations.

**Ranking methods:**
- **Bradley-Terry MLE** — log-linear strength model; returns log-scale strength scores (mean-normalised to 0)
- **Elo ratings** — simpler sequential alternative (K=32, initial rating=1000)

Both methods produce **run-level** scores internally, then the `ablation` command **aggregates them to config-level** by averaging across replications per `run_label`. This means the ablation report shows meaningful config comparisons rather than individual run IDs.

> **Interpretation note:** A single exceptional run in one configuration can dominate the run-level rankings even if the config's mean score is lower. Config-level aggregation and the LLM judge mean score together give a more complete picture — use both.

```bash
python -m Scripts.evaluation rankings
```

### Statistical Analysis

For replicated runs (N >= 2 per configuration):

| Method | Purpose |
|--------|---------|
| **Bootstrap CI** | 95% confidence intervals (10K resamples, percentile method) |
| **Mann-Whitney U** | Non-parametric significance test (appropriate for small N) |
| **Permutation test** | Two-sided, 10K permutations |
| **Cohen's d** | Effect size with Hedges' g correction for small samples; **95% bootstrap CIs reported in the forest plot** |
| **Holm-Bonferroni** | Multiple comparison correction across all pairs |
| **Coefficient of Variation (CoV)** | Repeatability: std/mean across replications (<0.15 stable, >0.25 high variance) |
| **Krippendorff's α** | Inter-judge agreement per criterion (≥0.8 good, 0.6–0.8 acceptable, <0.6 unreliable) |
| **Pearson r (cross-dataset)** | Correlation of per-dataset mean scores across configs; requires ≥3 shared datasets |

Effect size interpretation: |d| < 0.2 negligible, 0.2–0.5 small, 0.5–0.8 medium, >0.8 large.

**Inter-judge agreement (Krippendorff's α):** Negative α values indicate judges disagree more than chance — typically caused by calibration differences (systematic scoring offsets) or genuinely different internal rubrics. For ablation purposes, check that both judges agree on the **direction** of the config comparison even if absolute scores differ. Two weak judges with poor agreement is worse than one strong judge — if α is consistently below 0.4, consider replacing both models with a single stronger judge (GPT-4o, Claude Sonnet/Opus).

### Gate Quality vs Cross-Validation Quality

The evaluation framework distinguishes between two types of pipeline-internal quality signals:

**Gate quality scores** (from `debug/*__gate.json`) apply only to **cleaning and analysis** stages. These stages use a gated retry loop — the critic evaluates output quality, assigns a `quality_score` (0–1), and triggers retries until the score threshold is met or retries are exhausted. The gate quality table in the ablation report shows mean ± std across replications for these two stages only.

**Cross-validation** does not use a retry/gate loop. It calls the LLM once, then optionally applies a deterministic fallback to verify claims from the data files directly. Its quality is instead measured by two structural metrics (Section 4b of the ablation report):
- `verified_claims_count` — number of analysis claims independently recomputed from the parquet (higher = better)
- `cv_gaps_count` — number of gaps or inconsistencies identified (lower = better)

These structural metrics are zero-cost (no LLM calls) and are reported separately.

### Outputs and Exports

```bash
# Ablation analysis (requires baseline label)
python -m Scripts.evaluation ablation \
    Outputs/slurm_A Outputs/slurm_B ... --baseline "Baseline"
```

**Ablation report sections (`Evaluation/ablation_report.md`):**

| Section | Content |
|---------|---------|
| 1. Overview | LLM judge mean ± std per config per criterion |
| 2. Statistical Comparisons | Mean diff, Cohen's d with 95% CI, Mann-Whitney p, Holm-Bonferroni correction |
| 3. Rankings (Config-Level) | Bradley-Terry and Elo scores aggregated per config (averaged across replications) |
| 4. Gate Quality (Cleaning & Analysis) | Mean ± std pipeline gate scores (0–1); cleaning and analysis only |
| 4b. Cross-Validation Quality | verified_claims_count and cv_gaps_count per config |
| 5. Per-Dataset Breakdown | Overall judge score per config per dataset |
| 6. Per-Judge Breakdown | Overall score per config per judge model; reveals calibration offsets |
| 7. Inter-Judge Agreement | Krippendorff's α per criterion |
| 8. Repeatability (CoV) | Coefficient of variation per metric per config |
| 9. Cross-Dataset Correlation | Pearson r of per-dataset scores across configs (requires ≥3 datasets) |

**Figures (`Evaluation/figures/`):**

| Figure | Description |
|--------|-------------|
| `radar_chart.png` | Rubric criterion scores per configuration |
| `score_heatmap.png` | Configuration × Criterion matrix |
| `overall_bar_chart.png` | Mean + 95% CI; y-axis floored near data range to show differences |
| `overall_distributions.png` | Strip plot with mean line (N<5); violin suppressed below N=5 |
| `effect_size_forest.png` | Cohen's d point estimates with 95% bootstrap CIs; threshold guidelines on secondary x-axis |
| `rankings.png` | Config-level BT and Elo in **separate subplots** (different scales — sharing an axis hides BT values) |
| `stage_quality_progression.png` | Gate quality per stage; only stages with data are plotted (cross-validation excluded) |
| `repeatability_violin.png` | Score spread per config; violin suppressed for N<5, shows points + mean line |
| `judge_agreement.png` | Krippendorff's α heatmap per criterion |

**Exports (`Evaluation/exports/`):**

```bash
python -m Scripts.evaluation export --format latex   # LaTeX tables (booktabs)
python -m Scripts.evaluation export --format csv     # CSV
python -m Scripts.evaluation export --format json    # JSON
```

| Export | Description |
|--------|-------------|
| `results_table.tex` | Mean ± std per config × metric |
| `significance_table.tex` | Cohen's d, p-values with *** stars, Holm-Bonferroni correction |

### CLI Reference

```
python -m Scripts.evaluation <command> [options]

Commands:
  evaluate    Run LLM-as-judge evaluation on pipeline outputs
  pairwise    Run pairwise head-to-head comparisons
  ablation    Generate ablation analysis (report, figures, LaTeX)
  export      Export results in latex/csv/json format
  rankings    Show current Bradley-Terry and Elo rankings
  full        Run the complete evaluation pipeline (all of the above)

Global options:
  --config, -c PATH       Path to evaluation_config.yaml (default: evaluation_config.yaml)
  --log-level LEVEL       DEBUG, INFO, WARNING, ERROR (default: INFO)

Common options:
  --glob, -g PATTERN      Glob pattern for run directories (e.g. "Outputs/slurm_*")
  run_dirs                Explicit run directories (positional)
  --skip-existing         Skip runs already evaluated in the database
  --baseline LABEL        Run label for the baseline configuration (required for ablation/full)
```

### Example Workflow

```bash
# 1. Submit 5 baseline replicates
bash Scripts/evaluation/launch_replicates.sh --replicates 5

# 2. Edit context.md to enable WP1, change run_label to "wp1"
# 3. Submit 5 WP1 replicates
bash Scripts/evaluation/launch_replicates.sh --replicates 5

# 4. Run full evaluation
python -m Scripts.evaluation full \
    --glob "Outputs/slurm_*" --baseline "Baseline"

# 5. Results in Evaluation/ -- figures, LaTeX tables, SQLite DB
```

---

## Feature Toggle Reference

See the main [README](../README.md) for the complete list of all feature toggles including the WP-C critic architecture toggles (`critic_analytical_depth`, `critic_execution`, `targeted_refinement`, `refinement_cascade`) and the WP-R report quality controls (`critic_report`, `min_quantitative_grounding`, `numerical_accuracy_check`).

### WP-R: Report Quality Controls

These toggles control quality gates applied to the **report pipeline** output (Phase 3d in `report_pipeline.py`), not the captain pipeline fallback report. The quality gate logic lives in `report_quality.py` and orchestrates up to 2 revision rounds when checks are enabled.

| Toggle | Type | Default | Description |
|--------|------|---------|-------------|
| `critic_report` | bool | `true` | LLM narrative critic on report pipeline output. Uses a 6-criteria rubric (executive summary quality, numerical fidelity, conclusion support, data faithfulness, section completeness, synthesis vs enumeration) via the OpenRouter critic client (`CRITIC_OPENROUTER_API_KEY`). |
| `min_quantitative_grounding` | float | `0.0` | Minimum fraction of analytical paragraphs that must contain at least one number. If the report falls below this threshold, a revision round adds quantitative detail. Set to 0.0 to disable; recommended starting value is 0.70. |
| `numerical_accuracy_check` | bool | `true` | Pure-Python heuristic that flags numbers in report prose contradicting analysis or cleaning JSON artifacts. Produces MUST_FIX/SHOULD_FIX issues that feed into the revision round. |

When `critic_report` and `numerical_accuracy_check` are set to `false`, the quality gate is a no-op — the report passes through unchanged, preserving ablation baseline behaviour. Reports generated from degraded analyses automatically receive a quality caveat section. The captain pipeline's fallback report stage has no quality gates applied; it produces a basic LLM-generated narrative only.
