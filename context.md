# Analysis Context

Data: chromatography (IEX, SEC, HIC) and mass spectrometry runs from a biologics purification
campaign. Files may contain either or both data types; infer the domain from column names and
structure. Not all columns are guaranteed to be present across files — reason from what exists.

The `column` field identifies resin type (CM, DEAE, Q, SP) where present and should be used as
a primary experimental grouping factor alongside `run_no` and `chromatography_stage`.

Column descriptions are provided via `parameters.xlsx` (loaded as `metadata_context_text`).
Treat them as interpretive context, not a strict schema. Undocumented columns should be inferred
from naming conventions, units, and correlations with other columns.

## Pipeline Configuration

```yaml
pipeline:
  stages:
    - name: cleaning
      goals:
        - Remove fully-empty columns
        - Standardise column names
        - Flag anomalies without aggressive removal
      quality:
        min_findings: 2
      max_retries: 2

    - name: analysis
      goals:
        # Analytical PRINCIPLES — the planner should translate these into
        # specific methods and chart types based on the schema profiler's
        # dimensional structure and analysis contexts.  Do not treat these
        # as a fixed checklist — adapt to what the data actually contains.
        - Characterise variation across the detected dimensional hierarchy (e.g. experimental_unit × process_phase × condition)
        - Identify anomalies at every grouping level using domain-appropriate thresholds
        - Test for interactions between grouping dimensions (e.g. does column type effect vary by chromatography stage?)
        - Assess process stability and system suitability using domain-specific metrics and regulatory thresholds
      grouping_guidance: |
        Use the schema profiler's dimensional structure to determine grouping.
        If analysis_contexts are present, use them to select appropriate grouping
        for each analytical question. Do not default to a single flat grouping
        when richer structure is available.
        Each analytical question should use the most appropriate grouping —
        different questions may require different grouping strategies.
      quality:
        min_plots: 3
        min_findings: 3
        require_per_group: true
      agent_hints:
        require: [analysis_planner, ml_modeler]
        prefer: [statistical_analyst]
        max_agents: 5
        fallback_to_detection: true
      max_retries: 5
      expert_call_budget: 9

    - name: cross_validation
      goals:
        - Verify top-3 statistical claims against raw data
        - Re-compute reported outlier metrics
      agent_hints:
        require: [cross_validator]

    - name: report
      goals:
        - Executive summary with key findings
        - Detailed per-stage analysis section
        - Methodology and limitations

  constraints:
    preserve_columns: [run_no, run, chromatography_stage, Sample_Code, column, Fraction_number, charge_state, Spectrum_type]
    no_aggregation_across: [run, run_no]
    grouping_columns: []
    grouping_extend_when_present: [chromatography_stage, Sample_Code]
    parameters_path: "data/parameters.xlsx"

  run_config:
    run_label: "All WP"
    prompt_version: "v3"
    content_validation: true
    min_cross_val_claims: 5
    recompute_cross_val: true
    figure_selection: "ranked"
    max_report_figures: 10
    protected_col_audit: true
    knowledge_base: "chromadb_local"
    research_agent: "deep_research"
    max_groupchat_rounds: 15
    # WP 1
    schema_profiling: "full"
    # WP2
    iteration_strategy: "convergent"
    convergence_threshold: 0.02
    convergence_target: 0.85
    # WP 3
    visual_review_mode: "scientific"
    require_figure_references: true
    # WP 4
    expert_library: "extended"
    agent_definitions_dir: "agents/"
    # BS-4: Payload budgets (chars) — tuned for 262K context window.
    # Controls how much of each data source is embedded in report prompts.
    payload_budget_cleaning: 4000           # cleaning summary JSON
    payload_budget_analysis: 16000           # analysis summary JSON (findings, per_group, etc.)
    payload_budget_per_file: 10000           # per-file analysis in global captain reports
    payload_budget_global: 80000            # global report payload JSON (report_pipeline)
    payload_budget_report_excerpt: 10000    # individual report excerpt in cross-file reports
    should_fix_accumulation_threshold: 10  # non-plot SHOULD_FIX before retry (plot-quality excluded)
    # WP-C: Critic architecture toggles
    critic_structural: true             # structural gate (pure Python)
    critic_content: true                # LLM content evaluator
    critic_visual: true                 # VLM plot reviewer
    critic_analytical_depth: true       # WP-C3a: Python heuristic + LLM depth analysis
    critic_execution: true              # WP-C3b: execution correctness (pure Python)
    # WP-C2: Targeted refinement
    targeted_refinement: true           # enable PLOT_FIX / FINDING_FIX / GAP_FILL paths
    refinement_cascade: true           # escalation cascade (enable after validation)
    ml_backend: "tabpfn"                # "sklearn" | "tabpfn" | "both"
    # WP-R: Report quality controls — applied to report pipeline output (Phase 3d),
    # NOT to the captain pipeline fallback report. When all defaults (false/0),
    # the quality gate is a no-op and reports pass through unchanged.
    critic_report: true               # WP-R3: LLM narrative critic on report pipeline output (6 criteria rubric, via OpenRouter)
    min_quantitative_grounding: 0.7     # WP-R4: grounding revision threshold (0.0=off, try 0.70)
    numerical_accuracy_check: true     # WP-R6: check prose numbers against analysis/cleaning JSON artifacts
```

### ML Modelling Tasks

Define supervised learning tasks for the ml_modeler agent.  These give the agent
concrete prediction targets rather than relying on unsupervised exploration.
The agent will attempt tasks in order and report which were feasible.

```yaml
ml_tasks:
  chromatography:
    - task: "classification"
      target: "chromatography_stage"
      description: "Predict chromatography stage from UV/conductivity features"
      aggregate_by: ["run_no", "chromatography_stage", "column"]
      features: ["UV_1_280_ml", "Conductivity", "volume_ml"]
      note: "Aggregate raw rows to per-group summaries before modelling"
    - task: "regression"
      target: "peak_area_mean"
      description: "Predict mean peak area from process parameters per run"
      aggregate_by: ["run_no", "column"]
      features: "auto"  # use all numeric summary features
    - task: "classification"
      target: "outlier_flag"
      description: "Classify runs as outlier/normal based on CV > 15% threshold"
      aggregate_by: ["run_no", "column"]
      derive_target: "cv_threshold(UV_1_280_ml, 0.15)"

  mass_spectrometry:
    - task: "regression"
      target: "dominant_mass_kda_mean"
      description: "Predict dominant mass from charge state and response features"
      aggregate_by: ["run_no", "Sample_Code", "column"]
      features: "auto"
    - task: "classification"
      target: "column"
      description: "Classify resin type from mass spectrometry measurement profiles"
      aggregate_by: ["run_no", "column"]
      features: "auto"
    - task: "classification"
      target: "quality_flag"
      description: "Flag low-quality runs based on signal-to-noise or mass accuracy"
      aggregate_by: ["run_no", "column"]
      derive_target: "cv_threshold(Response, 0.20)"
```
