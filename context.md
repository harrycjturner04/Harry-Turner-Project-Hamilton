# Multi-Agent Biologics Data Analysis (AG2)

## Project Overview

This project implements an LLM-driven, multi-agent analysis pipeline using the AG2 framework.
The system is orchestrated by a CaptainAgent, which dynamically decomposes tasks, selects agents
from an agent library, and coordinates execution.

The primary goal is to analyse biologics experimental data in a flexible, inference-driven manner,
minimising hardcoded logic and maximising reasoning based on the data actually provided.

The pipeline is intended to:
- Clean raw experimental data
- Infer meaningful analytical approaches
- Execute quantitative analysis via tools/code
- Produce qualitative and quantitative scientific insights
- Validate outputs before final reporting

## Data Domain

The data processed by this system consists of:
- Chromatography data (e.g. HPLC-style outputs)
- Mass spectrometry (MS) data

These datasets may appear independently or together. The system must infer which analyses
are appropriate based on the presence, structure, and column headings of the data.

There is NO guarantee that:
- All expected columns are present
- All files conform to the same schema
- All experiments are directly comparable

Agents must reason from what they see, not what they expect.

## Data Semantics & Parameters

A `parameters.xlsx` file is provided alongside the data. This file:
- Explains the meaning of some (but not necessarily all) columns
- Provides domain context (units, biological interpretation, experimental intent)

Agents should:
- Use `parameters.xlsx` as supporting context
- NOT treat it as a strict schema
- Combine it with column names, value ranges, and data patterns to infer meaning

If a column is undocumented, agents should infer its role from:
- Naming conventions
- Units
- Correlation with other columns
- Typical chromatography / MS workflows

## Analysis Philosophy

Analysis MUST be inferred dynamically.

Agents should:
- Inspect column headings first
- Reason about what analyses are possible or useful
- Propose analysis plans before executing code
- Justify why each analysis step is being performed

Hardcoded analysis pipelines are explicitly discouraged.

## Data Cleaning Expectations

Cleaning should be conservative and explainable. Typical actions include:
- Removing columns that are entirely missing
- Standardising obvious format issues
- Flagging (not blindly removing) anomalies

## Validation & Trust

All analytical conclusions must be:
- Supported by computed results
- Scientifically plausible
- Consistent across chromatography and MS findings where applicable

## Design Intent

This system is intentionally:
- LLM-reasoning driven
- Minimally hardcoded
- Adaptive to new data formats
- Designed to "think on its feet"

If an agent cannot justify an analysis step from the data provided,
it should not perform that step.

## Current Baseline

The system has been stabilised through multiple iterations. The following reflects the
current operational state as of March 2026:

- **LLM backend**: vLLM 0.17.0 serving Qwen3.5-27B at BF16 on H200 NVL GPUs
  - `--max-model-len 131072` (128K context window)
  - TP=1 recommended; TP=2 functional but adds ~8 min startup overhead for negligible gain
- **CaptainAgent orchestration**: AG2 CaptainAgent with `_bad_tool_call_streak` guard (3 consecutive)
  - Explicit `timeout=300` on all direct LLM calls
  - Per-stage `chat_timeout`: cleaning/cross_val/report=900s, analysis=2400s (via `StageSpec.chat_timeout`)
  - `PIPELINE_CHAT_TIMEOUT_S` env var as fallback (default 600s)
- **Gated review loop**: Deterministic 4-phase validation per stage attempt:
  1. Structural gate (pure Python) — file existence, JSON validity, min plots/findings
  2. Content evaluator (LLM) — rubric-based scoring per stage type with coverage validation
  3. VLM plot quality review — aesthetic or scientific mode with graduated severity
  4. Quality gate (Python) — numeric quality scoring, convergence tracking
- **Schema profiler** (WP-1): Deterministic column role classification and grouping inference
  - Domain-agnostic: uses dtype, cardinality, naming patterns — not domain keywords
  - Replaces hardcoded `KNOWN_GROUP_COLUMNS` with data-driven grouping candidates
- **Convergence control** (WP-2): Quality-driven iteration strategies
  - Content evaluator hardened with coverage validation and retry logic
  - Numeric quality scoring: passed=1.0, should_fix=0.5, must_fix=0.0
  - Three strategies: `none` (single pass), `fixed` (baseline), `convergent` (quality-driven)
- **Scientific visual review** (WP-3): Evidence-grounded plot assessment
  - 7 criteria across 3 severity classes (scientific_validity, statistical_completeness, cosmetic)
  - Claim-figure traceability: findings must reference existing figures when enabled
  - Graduated severity: only scientific issues trigger retries; cosmetic issues are informational
- **Pluggable agent library** (WP-4): YAML-driven agent definitions
  - 8 baseline agents + 3 extended experts (bioprocess, visualisation, statistical modelling)
  - Schema-driven activation: agents selected by data profile match, not just keyword detection
  - New experts added by dropping a YAML file into `agents/` — no source code changes needed
- **Knowledge base**: Local ChromaDB RAG (in-memory) with sentence-transformers `all-MiniLM-L6-v2`
  - Indexes all `knowledge_base/*.md` files at startup (600-char chunks, 80-char overlap)
  - No OpenAI or external API calls required
- **Research agent**: DeepResearchAgent with browser-use + Playwright
  - Dependencies: langchain-google-genai 2.0.8, langchain-ollama 0.2.2
- **PDF rendering**: WeasyPrint with absolute figure paths
- **Environment**: protobuf 5.29.6, chardet 5.x, chromadb 1.3.7, browser-use telemetry disabled

## Known Limitations

Remaining items from baseline diagnosis (runs 866392 / 866394):

- **BS-1**: Global report figure references may point to per-file analysis figures instead of
  global figures — partially mitigated by WP-3 claim-figure traceability when enabled
- **BS-5**: Global VisualisationAgent code fails to properly read/merge parquet files due to
  differing column names between datasets — partially mitigated by dataset_columns metadata
  injection, but still relies on LLM following instructions
- Pipeline is fully sequential — no file-level parallelism, so multi-GPU provides no throughput gain

Previously diagnosed items now addressed:
- ~~BS-2: Figure path short-circuit~~ — `_embed_report_figures` now resolves broken `![`
  refs via stem matching, figure-number matching, and appends unreferenced figures
- ~~BS-3: GroupChat round exhaustion~~ — mitigated by convergence control (WP-2) and
  quality-driven stopping
- ~~BS-4: Payload truncation~~ — replaced hardcoded char slices with configurable
  `payload_budget_*` fields in run_config, tuned for 128K context window
- ~~Critic feedback loop non-functional~~ — replaced by deterministic gated review loop
  with content evaluator hardening (WP-2A) ensuring reliable rubric coverage

## Work Package Status

All WPs are toggle-gated and additive. When toggles are at their defaults, the pipeline
executes the same code paths as Baseline v2.

| WP | Name | Status | Toggle | Default |
|----|------|--------|--------|---------|
| WP-1A | Schema Intelligence (column roles, grouping) | implemented | `schema_profiling` | `"disabled"` |
| WP-1B | Distribution profiling + technique recommendations | not started | `schema_profiling: "full"` | — |
| WP-2A | Content evaluator hardening | implemented | always active | — |
| WP-2B | Numeric quality scoring | implemented | always active | — |
| WP-2C | Convergence control | implemented | `iteration_strategy` | `"fixed"` |
| WP-3A | Claim-figure traceability | implemented | `require_figure_references` | `false` |
| WP-3B | Scientific visual review prompt | implemented | `visual_review_mode` | `"basic"` |
| WP-3C | Graduated severity routing | implemented | `visual_review_mode` | `"basic"` |
| WP-4A | Agent YAML externalisation | implemented | `expert_library` | `"baseline"` |
| WP-4B | Schema-driven agent activation | implemented | `expert_library` | `"baseline"` |
| WP-4C | New domain experts (3) | implemented | `expert_library: "extended"` | — |

**Baseline absorption candidates** (should become defaults once validated):
- WP-1A (column roles + grouping) — strictly better than hardcoded KNOWN_GROUP_COLUMNS
- WP-2A (evaluator hardening) — fixes a reliability bug, no quality trade-off
- WP-3A (claim-figure traceability) — structured evidence mapping
- WP-4A (agent externalisation) — pure maintainability improvement

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
        # Data-structure-aware goals — the planner should adapt methods
        # based on the schema profiler's dimensional structure and
        # analysis contexts rather than applying fixed techniques.
        - Per-group chromatographic or spectral profile characterisation (adapt signal processing to data structure)
        - Cross-dimensional comparison using the detected hierarchy (e.g. experimental_unit × process_phase)
        - Peak or mass statistics per analytical group (use recommended grouping from data profile)
        - System suitability / quality assessment appropriate to the detected domain
        - Cross-run consistency analysis comparing experimental units
        - Process-phase trend analysis where stage-level grouping is available
        - Outlier detection within each analytical group (>2 SD from group mean)
        - Multi-dimensional heatmap or summary using the full grouping hierarchy
        - Extended grouping analysis where finer breakdown is available (e.g. per-sample or per-stage subset)
      grouping_guidance: |
        Use the schema profiler's dimensional structure to determine grouping.
        If analysis_contexts are present, use them to select appropriate grouping
        for each analytical question. Do not default to a single flat grouping
        when richer structure is available.
      quality:
        min_plots: 3
        min_findings: 3
        require_per_group: true
      agent_hints:
        require: [analysis_planner]
        prefer: [statistical_analyst, ml_modeler]
        max_agents: 5
        fallback_to_detection: true
      expert_call_budget: 4

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
    grouping_columns: [run, column]
    grouping_extend_when_present: [chromatography_stage, Sample_Code]

  run_config:
    run_label: "WP-1 FULL + CONVERGENT + Scientific"
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
    schema_profiling: "full"
    iteration_strategy: "convergent"
    convergence_threshold: 0.05
    convergence_target: 0.85
    visual_review_mode: "scientific"
    require_figure_references: true
    expert_library: "extended"
    agent_definitions_dir: "agents/"
    # BS-4: Payload budgets (chars) — tuned for 128K context window.
    # Controls how much of each data source is embedded in report prompts.
    payload_budget_cleaning: 4000           # cleaning summary JSON
    payload_budget_analysis: 8000           # analysis summary JSON (findings, per_group, etc.)
    payload_budget_per_file: 3000           # per-file analysis in global captain reports
    payload_budget_global: 40000            # global report payload JSON (report_pipeline)
    payload_budget_report_excerpt: 10000    # individual report excerpt in cross-file reports
    # WP-C: Critic architecture toggles
    critic_structural: true             # structural gate (pure Python)
    critic_content: true                # LLM content evaluator
    critic_visual: true                 # VLM plot reviewer
    critic_analytical_depth: true       # WP-C3a: Python heuristic + LLM depth analysis
    critic_execution: true              # WP-C3b: execution correctness (pure Python)
    # WP-C2: Targeted refinement
    targeted_refinement: true           # enable PLOT_FIX / FINDING_FIX / GAP_FILL paths
    refinement_cascade: false           # escalation cascade (enable after validation)
    ml_backend: "tabpfn"                # "sklearn" | "tabpfn" | "both"
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
      description: "Classify column type from mass spectrometry measurement profiles"
      aggregate_by: ["run_no", "column"]
      features: "auto"
    - task: "classification"
      target: "quality_flag"
      description: "Flag low-quality runs based on signal-to-noise or mass accuracy"
      aggregate_by: ["run_no", "column"]
      derive_target: "cv_threshold(Response, 0.20)"
```
