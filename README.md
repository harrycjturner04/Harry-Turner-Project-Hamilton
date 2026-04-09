# Multi-Agent Biologics Data Analysis Pipeline

A fully autonomous, LLM-driven system for analysing biologics experimental data using the [AG2 (AutoGen)](https://github.com/ag2ai/ag2) multi-agent framework. The pipeline dynamically assembles expert agent teams, executes quantitative analysis via code generation, validates results through a modular critic architecture, and produces publication-quality scientific reports.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Pipeline Stages](#pipeline-stages)
  - [Agent Orchestration](#agent-orchestration)
  - [Five-Pass Analysis Strategy](#five-pass-analysis-strategy)
  - [Gated Review Loop](#gated-review-loop)
  - [Critic Architecture](#critic-architecture)
  - [Targeted Refinement](#targeted-refinement)
  - [Report Pipeline](#report-pipeline)
  - [Agent Library](#agent-library)
  - [Monkey-Patches & Robustness Guards](#monkey-patches--robustness-guards)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Quick Start (SLURM)](#quick-start-slurm)
  - [Manual Execution](#manual-execution)
  - [CLI Reference](#cli-reference)
  - [Environment Variables](#environment-variables)
- [Configuration](#configuration)
  - [The context.md File](#the-contextmd-file)
  - [Pipeline Stages Configuration](#pipeline-stages-configuration)
  - [Data Constraints](#data-constraints)
  - [Quality Thresholds](#quality-thresholds)
  - [Run Configuration (Feature Toggles)](#run-configuration-feature-toggles)
  - [Agent Selection Hints](#agent-selection-hints)
  - [ML Modelling Tasks](#ml-modelling-tasks)
- [Output Structure](#output-structure)
- [Evaluation Framework](#evaluation-framework)
- [Adapting to New Data](#adapting-to-new-data)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [See Also](#see-also)

---

## Overview

This system analyses biologics experimental data (chromatography, mass spectrometry, or both) by orchestrating a team of specialised LLM-powered agents. Rather than following a fixed analysis script, each agent reasons from the data it observes -- inspecting column names, value distributions, and domain context to decide what analyses are appropriate.

The pipeline is governed by a single configuration file (`context.md`) that describes the project goals, data constraints, and quality expectations. This makes the system adaptable to different biologics datasets without code changes.

**Current model**: Qwen/Qwen3.5-27B served locally via [vLLM](https://github.com/vllm-project/vllm) on a single GPU (NVIDIA H200 NVL 144GB).

---

## Architecture

### Pipeline Stages

The system processes each input dataset through four sequential stages:

```
  Cleaning ──> Analysis ──> Cross-Validation ──> Reporting
     │            │               │                  │
     v            v               v                  v
  Cleaned     Findings +      Verified           Scientific
  Parquet     Plots + JSON    Claims JSON +      Report (PDF)
              + ML Models     claims.json
```

| Stage | Purpose | Default Timeout | Default Retries | Output |
|-------|---------|-----------------|-----------------|--------|
| **Cleaning** | Remove empty columns, standardise formats, flag anomalies | 900s | 2 | Cleaned parquet + `cleaning_summary.json` |
| **Analysis** | Generate statistical findings, plots, per-group metrics | 2400s | 5 | `analysis_summary.json` + PNG plots + `data_profile.json` |
| **Cross-Validation** | Re-compute top claims from raw data to verify accuracy | 1800s | 2 | `cross_validation.json` + `claims.json` with verified claims |
| **Reporting** | Synthesise narrative report with literature context | 900s | 0 | `report.md`, `report.html`, `report.pdf` |

Each stage has a configurable `expert_call_budget` limiting `seek_experts_help()` calls (defaults: cleaning=2, analysis=3, cross_validation=3, report=2; the `_ExpertCallBudget` hardcoded fallback uses analysis=9 to accommodate the five-pass strategy) and per-GroupChat max rounds (defaults: cleaning=15, analysis=30, cross_validation=15, report=15).

### Agent Orchestration

The system uses AG2's **CaptainAgent** as the top-level orchestrator:

```
User Proxy (captain_user)
    │
    v
CaptainAgent
    │── calls seek_experts_help() ──>  AutoBuild
    │        (budget: 9 analysis,        │
    │         2 no-code max)             v
    │                              Filtered Agent Library
    │                              (per-pass isolation:
    │                               PLAN = full library, no code
    │                               REVIEW = no ml_modeler, no code
    │                               EXECUTE/REFLECT = no ml_modeler
    │                               MODEL = ml_modeler only)
    │                                    │
    │                                    v
    │                              Expert GroupChat
    │                              [domain experts + Computer_terminal]
    │                                    │
    │<── structured JSON result ─────────┘
    │
    v
Pre-critic normalisation (figure_ref injection, deduplication)
    │
    v
Gated Review Loop (critic registry)
    │── pass ──────────────> Next stage
    │── retry (scoped) ───> CaptainAgent (top 3 issues if ≥5 MUST_FIX)
    │── refinement ───────> PLOT_FIX / FINDING_FIX / GAP_FILL
    └── degrade ──────────> Next stage (with warnings)
```

1. **captain_user** sends the task instruction to **CaptainAgent**
2. CaptainAgent decomposes the task and calls `seek_experts_help()` with a team specification
3. AG2's **AutoBuild** assembles an expert GroupChat from the agent library
4. Experts collaborate -- proposing code, executing it via `Computer_terminal`, and producing structured JSON
5. The result flows back to CaptainAgent and enters the gated review loop

**Budget enforcement**: `_ExpertCallBudget` tracks `seek_experts_help()` calls per stage and blocks further calls when the budget is exhausted, preventing redundant team spawns.

**Wall-clock enforcement**: `_ChatTimeout` wraps each CaptainAgent chat in a SIGALRM-based timeout with a daemon watchdog thread as backup escalation and a hard `os._exit(42)` as a nuclear fallback.

### Five-Pass Analysis Strategy

The analysis stage uses a structured five-pass strategy to ensure analytical depth, domain coverage, and predictive modelling:

```
Pass 1: PLAN       Pass 2: REVIEW       Pass 3: EXECUTE       Pass 4: MODEL         Pass 5: REFLECT
analysis_planner ──> domain experts ────> domain experts ────> ml_modeler ──────────> domain experts
  │                  (no code)            (with code)           (with code)            (with code)
  v                  v                    v                     v                      v
5-8 diverse        reviewed_items +     findings + plots +    ML models, feature     gap analysis,
  approaches       expert_additions     per_group + domain_   importance, per_group   cross-dimensional
                                        reasoning             performance             follow-ups
     ─── no-code budget (2) ───           ─── coding enforced ───────────────────────────────────
     ─── full agent library ───           ─ no ml_modeler ─    ─ ml_modeler only ─   ─ no ml_modeler ─
```

1. **PLAN pass**: `analysis_planner` recommends 5–8 diverse analytical approaches, each referencing the data's dimensional structure. Output is a structured JSON plan with ≥3 chart types.
2. **REVIEW pass**: Domain experts (chromatography, mass spec, statistical) critique the plan without executing code. They add `expert_additions` and flag gaps.
3. **EXECUTE pass**: Domain experts execute the reviewed plan via code. Output must include `domain_reasoning` with `hypotheses_tested`, `plan_modifications_applied`, and `unexpected_observations`.
4. **MODEL pass**: `ml_modeler` (only) builds supervised predictive models using the cleaned dataset and findings from Pass 3. Reports model metrics, feature importance, and per-group performance. Appends results to `analysis_summary.json`.
5. **REFLECT pass** (optional — skipped only if expert call budget is exhausted): Domain experts review execution and modelling results, identify gaps or surprising patterns, explore alternative grouping dimensions, and perform cross-dimensional follow-up analyses. Appends new plots and findings.

**Agent library isolation**: The pipeline dynamically filters the agent library per pass — PLAN passes get the full library (planner selected by text instruction), MODEL passes use an `ml_modeler`-only library, while REVIEW/EXECUTE/REFLECT passes exclude `ml_modeler` entirely, enforced via `_write_filtered_agent_library()`.

**No-code call budget**: A maximum of 2 no-code `seek_experts_help()` calls is allowed (for PLAN + REVIEW). After exhaustion, code execution is forced back on regardless of group name keywords, preventing the LLM from stalling in planning loops.

### Gated Review Loop

Each stage attempt passes through a **modular critic registry** that dispatches evaluators in a deterministic order. The registry is a pluggable system where each critic module independently decides whether to run and what severity to assign.

**Design principle**: Deterministic governance, non-deterministic analysis. Python code makes all gate/routing decisions; LLMs assess quality and produce structured signals.

The registry dispatches critics in order, aggregates their `CheckResult` outputs into a `StageVerdict`, and feeds the verdict into a deterministic `quality_gate()` function that decides pass/retry/degrade.

**Convergence control** (`iteration_strategy` toggle):
- `"none"` -- single pass, no validation
- `"fixed"` -- up to `max_retries` attempts (baseline)
- `"convergent"` -- quality-driven exit: stops early when score ≥ `convergence_target` with no `MUST_FIX` failures, or when improvement drops below `convergence_threshold` (diminishing returns)

Quality trajectory is tracked per stage: `(attempt, score, breakdown, critics_ran)`.

**Pre-critic normalisation**: Before each critic pass in the analysis stage, findings missing `figure_ref` keys are automatically matched to the most relevant PNG by token overlap. After the gated review loop exits, findings are deduplicated by normalised text across passes (Execute and Reflect can produce duplicates).

**Scoped retry**: When ≥5 MUST_FIX issues are present, retry instructions are scoped to address only the top 3 to prevent overcorrection. If findings increase by >50% between iterations, the agent is instructed to focus on quality over volume. When a retry causes net regression (more checks PASS→FAIL than FAIL→PASS), the previous artifacts are restored and the next retry receives explicit instructions to make surgical fixes only.

### Critic Architecture

The `Scripts/critics/` directory contains modular critic modules, each implementing the `CriticModule` ABC. The `CriticRegistry` instantiates all available critics at pipeline init and dispatches them per stage attempt. Each critic produces a `List[CheckResult]` that feeds into the `StageVerdict`.

```
Stage Output (JSON + PNGs)
    │
    v
┌─ Critic Registry (deterministic dispatch order) ────────────────────────┐
│                                                                         │
│  [10] StructuralCritic (Python)                                         │
│    │── FAIL (missing files, corrupt JSON) ──> short-circuit: skip rest  │
│    └── PASS ──v                                                         │
│                                                                         │
│  [20] ExecutionCritic (Python) ── 9 checks incl. ML accuracy/stability  │
│    └──v                                                                 │
│                                                                         │
│  [30] ContentCritic (LLM) ── rubric-based (7/4/3/6 criteria by stage)   │
│    └──v                                                                 │
│                                                                         │
│  [40] AnalyticalDepthCritic (Python+LLM) ── de-dup with ContentCritic   │
│    └──v                                                                 │
│                                                                         │
│  [50] PlotStructuralCritic (Python) ── PNG headers, aspect ratio tiers  │
│    │── marks exhausted plots (.png.unfixable)                           │
│    └──v                                                                 │
│                                                                         │
│  [60] VisualCritic (VLM) ── skips exhausted plots                       │
│    └── MUST_FIX? ── double-evaluation (2nd pass confirms/downgrades)    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
    │
    v
StageVerdict ──> quality_gate() ──> pass / retry / refinement / degrade
```

**Execution order** (lower number runs first):

| Order | Critic | Type | Toggle | Default | Stages | What it checks |
|-------|--------|------|--------|---------|--------|----------------|
| 10 | `StructuralCritic` | Python | `critic_structural` | `true` | cleaning, analysis, cross_val | File existence, JSON validity, min plots/findings, PNG size, domain reasoning field |
| 20 | `ExecutionCritic` | Python | `critic_execution` | `true` | analysis | NaN leakage, impossible values, uniform groups, finding-data mismatch, plot-finding alignment, duplicate p-values (MUST_FIX), feature importance completeness, ML accuracy floor, feature stability (small-N regression) |
| 30 | `ContentCritic` | LLM | `critic_content` | `true` | cleaning, analysis, cross_val, report | Rubric-based scoring (7 criteria for analysis, 4 for cleaning, 3 for cross-validation, 6 for report) with coverage validation; reference-anchored evaluation prevents regression across retries |
| 40 | `AnalyticalDepthCritic` | Python + LLM | `critic_analytical_depth` | `false` | analysis | Grouping adequacy, grouping compliance, context coverage, group comparison tests, chart diversity, interpretation depth, domain expectations (YAML-driven), outlier analysis, chart appropriateness, interaction depth |
| 50 | `PlotStructuralCritic` | Python | `critic_visual` | `true` | analysis | Corrupt PNG detection, resolution (<400px), two-tier aspect ratio (>4:1 SHOULD_FIX, >20:1 MUST_FIX), size outliers, duplicate plots |
| 60 | `VisualCritic` | VLM | `critic_visual` | `true` | analysis | Plot quality -- basic (aesthetic) or scientific (chart type, grouping, annotations, layout quality) via graduated severity; double-evaluation for MUST_FIX (second pass confirms or downgrades) |

**Short-circuit optimisation**: If the structural gate fails on artifact validity checks (missing files, corrupt JSON), LLM/VLM critics are skipped entirely. Plots previously retired as unfixable (`.png.unfixable`) are tracked as exhausted and excluded from VLM evaluation on subsequent passes.

**De-duplication**: If `ContentCritic` flags `domain_interpretation` as `MUST_FIX`, the `AnalyticalDepthCritic`'s overlapping `depth__bare_deviations` check is downgraded to `SHOULD_FIX` to avoid double-penalising the same gap. Similarly, `depth__missing_group_comparison` is downgraded when `per_group_depth` or `statistical_rigor` is already flagged. The retry instruction builder also suppresses redundant checks via a subsumption map, preventing contradictory guidance to the generator.

**Quality scoring**: Each `CheckResult` maps to passed=1.0, should_fix=0.5, must_fix=0.0. Content and plot checks receive 2× weight. Visual quality score excludes cosmetic-category checks to prevent score inflation. A coverage penalty of −0.05 applies per skipped evaluator. Domain-specific expectations are loaded from `critics/domain_biologics.yaml` (not hardcoded Python keywords).

**Gate decision logic** (`quality_gate()` in `tools.py`):
- No must_fix + evaluators ran → `passed`
- Only should_fix → `passed` with warnings (unless ≥ `should_fix_accumulation_threshold` accumulated → retry)
- must_fix + retry budget exhausted → `passed_degraded`
- must_fix + same failures for `issue_stall_max_consecutive` attempts → `passed_degraded` (stall detection)
- must_fix + budget remaining → route to refinement tier; net regression detection flags when a retry made things worse (more checks PASS→FAIL than FAIL→PASS)

**Regression firewall**: When the quality gate detects net regression after a full CaptainAgent retry, the previous attempt's `analysis_summary.json` is restored from backup, and subsequent retry instructions mandate surgical-only fixes to prevent wholesale replacement from destroying passing content.

**Reference-anchored evaluation**: The content evaluator receives the previous attempt's passed checks as an anchor. Previously-passing criteria can only be failed if the current output is genuinely worse on that specific dimension, creating monotonic evaluation pressure across retries.

**Critic logging**: Each critic run is appended to `{label}__critic_log.jsonl` with checks produced, must_fix count, should_fix count, and wall_time_ms.

### Targeted Refinement

When `targeted_refinement: true`, the system attempts surgical fixes before falling back to a full CaptainAgent retry. Refinement directives are built from the quality gate's analysis of must_fix failures. An `IssueTracker` tracks per-issue stall across attempts (downgrades after `issue_stall_max_consecutive` = 2 occurrences without resolution).

```
quality_gate() ── must_fix + budget remaining
    │
    v
Refinement Scope Selection (most targeted first when cascade=true)
    │
    ├── plot-only failures ──> PLOT_FIX
    │     │── attempt 1: LLM regenerates plot ──> VLM verify
    │     │── attempt 2: retry if still failing
    │     └── retire unfixable (.png.unfixable) ──> track as exhausted
    │
    ├── content failures ──> FINDING_FIX
    │     └── rewrite specific findings, preserve numeric values
    │
    ├── depth__ failures ──> GAP_FILL
    │     └── generate + execute code, APPEND results
    │
    └── fallback ──> FULL_RERUN (CaptainAgent retry with directives)
```

| Scope | When triggered | What happens |
|-------|---------------|-------------|
| `PLOT_FIX` | Plot-only must_fix failures | Direct LLM call (up to 2 attempts per plot) to regenerate failing plots, post-fix VLM verification, retirement pathway for unfixable plots (renamed to `.png.unfixable`, tracked as exhausted) |
| `FINDING_FIX` | Content must_fix with identifiable ref | Surgical merge: identifies specific failing findings by ref/text match, rewrites only those via LLM, merges back by index preserving untouched findings and metadata (figure_ref); falls back to full rewrite if no specific findings identified |
| `GAP_FILL` | Analytical depth must_fix (`depth__` prefix) | Generate and execute code for missing analytical dimensions (ANOVA, correlations, etc.), APPENDs results |
| `FULL_RERUN` | Always available as final escalation | Full CaptainAgent retry with retry instructions |

When `refinement_cascade: true`, the system escalates through scopes (most targeted first). When `false`, only the first matching scope is attempted.

### Report Pipeline

The report pipeline (`report_pipeline.py`) generates publication-quality scientific reports via a five-phase architecture. It runs after the captain pipeline completes all per-file stages. This is the primary report generation path — the captain pipeline's built-in report stage serves only as a fallback.

**Per-file report** (five phases):

```
Phase 1: Research      Phase 2: KB Queries     Phase 3: GroupChat        Phase 3d: WP-R        Phase 4: Assembly
DeepResearchAgent  --> ChromaDB RAG queries --> Vis + Interpretation  --> Quality Gate      --> MD → HTML → PDF
  |                      |                     agents (deterministic      |                      |
  v                      v                     speaker selection)         v                      v
background,          domain-specific           report.md + figures       critic review,         report.pdf
citations            reference ranges          (incl. Models section     numerical checks,
                                               when ML data present)     grounding revision
```

1. **Research**: DeepResearchAgent (Playwright browser-use) gathers literature background. Circuit breaker disables after 3 consecutive failures. Falls back to WebSurferAgent (crawl4ai) if unavailable.
2. **Knowledge base queries**: Local ChromaDB RAG with `all-MiniLM-L6-v2` embeddings indexes all `knowledge_base/*.md` files (600-char chunks, 80-char overlap). No external API required.
3. **GroupChat**: VisualisationAgent generates figures (max 5 turns), then InterpretationAgent writes the narrative report. Figure manifest injection ensures InterpretationAgent knows which exact files were generated. When the analysis summary contains an `ml_modeling` key, InterpretationAgent includes a **Predictive Modelling** section covering model performance, feature importance, and process predictability insights. Report terminates on `REPORT_COMPLETE` message.
4. **WP-R quality gate** (`report_quality.py`): The report undergoes up to 2 revision rounds. Checks include: LLM narrative critic (6-criteria rubric), numerical consistency (prose vs JSON artifacts), and quantitative grounding rate. Issues are aggregated and the top 10 by severity feed a single revision LLM call per round. Reports generated from degraded analyses automatically receive a quality caveat noting that the underlying analysis exited the quality gate in a degraded state. Quality gate checks can be individually disabled via `run_config` toggles for ablation baseline behaviour.
5. **Assembly**: `_assemble_report_with_figures()` embeds inline "Figure N" references via stem/prefix matching, then appends unreferenced figures grouped by analytical theme. PDF rendered via WeasyPrint (branded CSS, A4, page numbers) with reportlab fallback.

**Global cross-file report** (when 2+ files have findings):
- Separate GroupChat with `GlobalInterpretationAgent` for self-contained cross-file comparison
- Comparative boxplot and deviation heatmap figures
- Per-file analysis figures are NOT passed to the global report to prevent confusing figure references (BS-1 mitigation)
- Column metadata pre-read from cleaned parquet files to prevent KeyError failures (BS-5 mitigation)

**Payload budgets** (configurable via `run_config`, tuned for 128K context window):

| Budget | Default (chars) | Controls |
|--------|----------------|----------|
| `payload_budget_cleaning` | 4,000 | Cleaning summary JSON slice |
| `payload_budget_analysis` | 8,000 | Analysis summary JSON slice |
| `payload_budget_per_file` | 3,000 | Per-file analysis in global reports |
| `payload_budget_global` | 40,000 | Global report payload JSON |
| `payload_budget_report_excerpt` | 10,000 | Individual report excerpt in cross-file reports |

### Agent Library

Agents are selected dynamically per-task via schema-driven activation scoring (`_score_agent_activation()`). With `expert_library: "baseline"` (default), the 8 core agents are used. With `expert_library: "extended"`, 3 additional specialist agents are available. Agent definitions live in `agents/*.yaml` and are loaded at runtime -- new experts can be added by dropping a YAML file into `agents/` with no source code changes.

**Core agents** (always available):

| Agent | Role |
|-------|------|
| `data_cleaner` | Conservative data cleaning with protected column awareness |
| `chromatography_expert` | Baseline correction, peak detection, system suitability (SEC, IEX, domain thresholds) |
| `mass_spec_expert` | Mass accuracy, charge-state validation, glycoform analysis (reference masses, S/N, PTM detection) |
| `statistical_analyst` | Descriptive statistics, correlations, outlier detection (max 5 plots, group-level focus) |
| `analysis_planner` | Recommends 5--8 analysis strategies with dimensional structure awareness, grouping rationale per entry, and at least one supervised ML task (Mode A: strategy, Mode B: EDA) |
| `ml_modeler` | Machine learning (classification, regression, clustering) with TabPFN support, supervised task definitions, and small-N requirements (N<200: mandatory CV uncertainty, feature stability, regularisation comparison) |
| `cross_validator` | Independent re-computation of statistical claims from parquet with dynamic group-level and completeness checks |
| `report_writer` | Synthesises scientific narratives with pre-embedded data (no code execution step) and domain-conditional acceptance thresholds |

**Extended agents** (available when `expert_library: "extended"`):

| Agent | Role | Activation Criteria |
|-------|------|-------------------|
| `bioprocess_analyst` | Upstream/downstream process data, yield, step recovery | ordinal_stage + continuous_measurement columns |
| `visualisation_specialist` | Chart type selection, statistical annotations, plot design | Active when `visual_review_mode: "scientific"` |
| `statistical_modeler` | Mixed-effects models, DoE, multivariate techniques | ≥4 numeric columns, ≥50 rows |

**Prompt versions** (controlled by `prompt_version` toggle):
- `"v1"` -- baseline: CODE_GUIDANCE_TEMPLATE only
- `"v2"` -- enhanced: three-component interpretation (numeric + biological + actionable) + actual p-values
- `"v3"` -- graduated: flexible interpretation depth + mandatory data profile usage + dimensional structure awareness + 2+ secondary analyses + multi-column grouping exploration + experimental condition annotation from parameters file

### Monkey-Patches & Robustness Guards

The pipeline applies several monkey-patches to AG2 and Qwen3.5 to ensure stability:

| Patch | Problem | Solution |
|-------|---------|----------|
| **System message reordering** | Qwen3.5 requires system message as FIRST message; AG2 appends it last → 400 error | Patches `_reflection_with_llm` to prepend system message (opt-in `PIPELINE_USE_TRANSFORM_MESSAGES=1` for TransformMessages hook approach) |
| **Think-token stripping** | `<think>…</think>` reasoning tokens leak into agent messages | Three-layer strip: API response level → `receive()` → `send()` |
| **Malformed tool-call loop guard** | vLLM TP≥2 instability produces HTTP 200 with corrupt JSON → infinite retry | Sliding-window state machine (last 20 calls) with 3-tier abort: 3 consecutive bad, >60% bad ratio, or 15 session total. Returns synthetic TERMINATE on abort. |
| **Truncated code block repair** | LLM `max_tokens` truncation leaves unclosed `` ```python `` fences | Appends closing `` ``` `` when missing; filters out non-executable (JSON/YAML) code blocks |
| **Empty code block guard** | Empty `code_blocks` list causes crash in AG2 | Returns `(0, "")` for empty; also dedents uniformly-indented code (thinking artifact) |
| **AgentBuilder think-token strip** | AutoBuild's builder_model leaks `<think>` tokens | Patches `builder_model.create` to strip reasoning content |
| **DeepResearchTool max_turns** | Qwen3.5 causes infinite decomposition loops (450+ exchanges) | Limits decomposition chat to `max_turns=10` |

**Transient error recovery**: Retries on vLLM HTTP 503/429/502 with exponential backoff. Disk-artifact recovery fallback when GroupChat hits `max_round` (reads analysis_summary.json from disk, restores from backups).

---

## Project Structure

```
.
|-- agents/                        # WP-4: YAML agent definitions (plug-in system)
|   |-- data_cleaner.yaml
|   |-- chromatography_expert.yaml
|   |-- mass_spec_expert.yaml
|   |-- statistical_analyst.yaml
|   |-- analysis_planner.yaml
|   |-- ml_modeler.yaml
|   |-- cross_validator.yaml
|   |-- report_writer.yaml
|   |-- bioprocess_analyst.yaml     # Extended (WP-4C)
|   |-- visualisation_specialist.yaml
|   '-- statistical_modeler.yaml
|
|-- Scripts/
|   |-- main.py                 # CLI entry point (arg parsing, ServerManager, report-only mode)
|   |-- captain_pipeline.py     # Core orchestrator (CaptainAgent, stages, gated review, monkey-patches)
|   |-- schema_profiler.py      # WP-1: Column role classification, grouping inference, distribution profiling
|   |-- report_pipeline.py      # Five-phase report generation (Research + KB + GroupChat + WP-R Quality Gate + Assembly)
|   |-- report_quality.py       # WP-R: Report quality gate (narrative critic, numerical consistency, grounding revision)
|   |-- prompts.py              # All agent system prompts (v1/v2/v3) + code guidance template
|   |-- tools.py                # Quality gate, structural gate, CheckResult/StageVerdict/GateResult types
|   |-- context_parser.py       # Parses context.md into RunPlan/RunConfig/StageSpec
|   |-- pdf_writer.py           # Markdown -> HTML -> PDF (WeasyPrint primary, reportlab fallback)
|   |-- database.py             # Data discovery, Parquet conversion, metadata vectorisation (sentence-transformers)
|   |-- compare_runs.py         # Cross-run comparison utility for ablation studies
|   |-- critics/                # WP-C1: Modular critic architecture
|   |   |-- __init__.py
|   |   |-- base.py             # CriticModule ABC, CriticContext dataclass
|   |   |-- registry.py         # CriticRegistry: discovery, ordering, dispatch, de-duplication
|   |   |-- structural.py       # Pure Python file/JSON checks + domain reasoning
|   |   |-- content.py          # LLM rubric-based evaluation (delegates to pipeline._run_content_evaluator)
|   |   |-- visual.py           # VLM plot quality review (delegates to pipeline._run_plot_quality_evaluator)
|   |   |-- analytical_depth.py # WP-C3a: 10 heuristic checks + optional LLM depth assessment
|   |   |-- execution.py        # WP-C3b: NaN leakage, impossible values, finding-data mismatch
|   |   |-- plot_structural.py  # WP-C4: PNG header validation, resolution, aspect, size outliers, duplicates
|   |   '-- vlm_calibration.py  # Offline VLM calibration tool
|   '-- evaluation/             # Evaluation framework (fully decoupled from pipeline)
|       |-- __init__.py, __main__.py
|       |-- cli.py              # Evaluation CLI entry point
|       |-- config.py           # Evaluation configuration loader
|       |-- registry.py         # Model/rubric registry
|       |-- artifacts.py        # Artifact extraction from pipeline outputs
|       |-- rubrics.py          # Rubric definitions (versioned)
|       |-- judge.py            # LLM-as-Judge via OpenRouter
|       |-- scoring.py          # Score aggregation
|       |-- pairwise.py         # Pairwise comparison logic (position-swap consistency)
|       |-- statistics.py       # Statistical analysis (CoV, Krippendorff's alpha)
|       |-- ablation.py         # Ablation study analysis
|       |-- visualization.py    # Results visualisation
|       |-- storage.py          # SQLite evaluation database (schema v2)
|       |-- deterministic.py    # Deterministic metrics: gate scores, data coverage, grounding rate
|       '-- launch_replicates.sh  # Launch multiple evaluation runs
|
|-- critics/
|   '-- domain_biologics.yaml   # Domain-specific analytical expectations for AnalyticalDepthCritic
|
|-- knowledge_base/             # RAG source documents (indexed by ChromaDB at startup)
|   |-- analytical_method_reference_ranges.md
|   |-- biologics_characterisation_primer.md
|   |-- ce_sds_interpretation_guide.md
|   |-- formulation_stability_guide.md
|   |-- iex_interpretation_guide.md
|   |-- intact_mass_interpretation_guide.md
|   |-- regulatory_guidance_summary.md
|   '-- sec_interpretation_guide.md
|
|-- context.md                  # Project configuration (governs pipeline behaviour)
|-- evaluation_config.yaml      # LLM-as-Judge evaluation configuration (OpenRouter)
|-- Batch_Script.sh             # SLURM submission script
|-- Environments/
|   |-- Project_Env.yml         # Conda environment specification
|   |-- requirements.txt        # Pip requirements
|   '-- requirements-torch.txt  # PyTorch-specific requirements
|-- docs/
|   '-- evaluation_and_ablation_guide.md  # Evaluation & ablation reference
|-- Data/                       # Input data (CSV + XLSX files)
|   |-- chromatography_combined.csv
|   |-- MS_combined.csv
|   |-- metadata chromatography_combined.csv
|   '-- Parameters.xlsx         # Column descriptions & domain semantics
|
|-- Database/                   # Built at runtime (raw parquet, metadata vectors, artifacts)
|-- Evaluation/                 # Evaluation outputs (SQLite DB, raw judgments)
'-- Outputs/                    # Pipeline run outputs (timestamped directories)
```

---

## Installation

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA 12.x (H200 NVL 144GB recommended for 27B parameter models)
- Conda or Mamba package manager
- SLURM (for HPC submission) or a local GPU machine

### Environment Setup

```bash
# Clone the repository
git clone <repo-url>
cd biologics-multi-agent-analysis

# Create the conda environment
conda env create -f Environments/Project_Env.yml
conda activate harry_turner_project

# Install Playwright browsers (needed for web research agent)
playwright install chromium

# vLLM must be installed in an interactive GPU session (JIT compilation requires GPU):
srun --pty -p cuda --gres=gpu:h200_nvl:1 -t 01:30:00 bash
pip install 'vllm>=0.17.0'
```

**Key dependencies** (see `Environments/Project_Env.yml` for full list):

| Package | Purpose |
|---------|---------|
| `ag2[openai,captainagent,crawl4ai]>=0.11` | Multi-agent framework |
| `vllm>=0.17.0` | Local LLM inference server |
| `torch>=2.10.0` | GPU compute backend |
| `pandas>=2.1`, `numpy`, `scipy>=1.17` | Data processing |
| `matplotlib>=3.7`, `seaborn>=0.12` | Visualisation |
| `sentence-transformers>=2.2.0`, `chromadb>=0.4.6` | Local RAG (knowledge base + metadata vectorisation) |
| `weasyprint>=60.0` | PDF generation (primary) |
| `reportlab` | PDF generation (fallback) |
| `openai>=1.40` | LLM API client |
| `tabpfn` | TabPFN ML backend (optional, for classification/regression) |

---

## Usage

### Quick Start (SLURM)

```bash
# Place your data files in the Data/ directory
# Edit context.md to describe your project and constraints
sbatch Batch_Script.sh

# Override pipeline mode at submission time:
PIPELINE_MODE=single_pass sbatch Batch_Script.sh
```

The batch script handles: module loading, conda activation, CUDA workarounds (cuBLAS BF16 fix, FlashInfer cache management), vLLM server startup on a unique port derived from `SLURM_JOB_ID`, readiness polling, database build, and cleanup on exit.

### Manual Execution

```bash
# 1. Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-27B \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.95 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-prefix-caching \
    --limit-mm-per-prompt '{"image": 4}' \
    --override-generation-config '{"chat_template_kwargs": {"enable_thinking": false}}' &

# 2. Build the database
python Scripts/database.py

# 3. Run the pipeline
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="not-needed-for-local"

python Scripts/main.py Database/raw Outputs/my_run \
    --model "Qwen/Qwen3.5-27B" \
    --context-path context.md \
    --pipeline-mode full_closed_loop
```

### CLI Reference

```
python Scripts/main.py INPUT_DIR OUTPUT_DIR [OPTIONS]

Positional Arguments:
  INPUT_DIR              Path to raw data files (parquet, CSV, etc.)
  OUTPUT_DIR             Path to write outputs

Options:
  --model MODEL          LLM model name (default: $OPENAI_MODEL or "Qwen/Qwen3.5-27B")
  --temperature TEMP     Sampling temperature (default: 0.0)
  --api-key KEY          OpenAI API key (or set OPENAI_API_KEY)
  --base-url URL         LLM endpoint URL (or set OPENAI_BASE_URL)
  --metadata-db PATH     Path to metadata database directory (or set $METADATA_DB_DIR)
  --context-path PATH    Path to context.md (auto-discovered if absent, or set $CONTEXT_PATH)
  --log-level LEVEL      Logging level: DEBUG, INFO, WARNING (default: $LOG_LEVEL or "INFO")
  --pipeline-mode MODE   single_pass | stage_validation | full_closed_loop
  --skip-report          Skip report generation pipeline
  --report-only PATH     Run only the report pipeline on an existing manifest JSON
```

**Pipeline modes**:
- `single_pass` -- No validation, one-shot execution. Ablation baseline.
- `stage_validation` -- Gated review with critic registry. VLM review if multimodal server detected.
- `full_closed_loop` -- Alias for `stage_validation` (backward compatibility).

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENAI_API_KEY` | API key for LLM endpoint | Required |
| `OPENAI_BASE_URL` | LLM server URL (e.g. vLLM) | Required |
| `OPENAI_MODEL` | Model name override | CLI `--model` |
| `PIPELINE_MODE` | Execution mode | `full_closed_loop` |
| `PIPELINE_CHAT_TIMEOUT_S` | Fallback wall-clock timeout per chat session (seconds) | `600` |
| `PIPELINE_MAX_TOKENS` | Maximum generation tokens per LLM request | `32768` |
| `PIPELINE_LLM_REQUEST_TIMEOUT_S` | Per-request httpx timeout for LLM calls (seconds) | `900` |
| `PIPELINE_MAX_WALL_S` | Hard wall-clock limit for entire pipeline run (0 = disabled) | `0` |
| `PIPELINE_MIN_PLOTS_ANALYSIS` | Override min plots for analysis | `3` |
| `PIPELINE_USE_TRANSFORM_MESSAGES` | `"1"` to use TransformMessages hook instead of monkey-patch | `"0"` |
| `VLLM_PID` | Externally-started vLLM server PID (for ServerManager adoption) | (none) |
| `VLLM_PORT` | vLLM server port | (none) |
| `VLLM_READY_TIMEOUT_S` | vLLM readiness probe timeout | `600` |
| `HUGGINGFACE_API_KEY` | HF token for gated model downloads | (none) |
| `CRITIC_OPENROUTER_API_KEY` | OpenRouter API key for content/plot evaluators | (none) |
| `OPENROUTER_API_KEY` | OpenRouter API key for evaluation framework (falls back to `CRITIC_OPENROUTER_API_KEY`, then YAML config) | (none) |
| `CRITIC_BASE_URL` | OpenRouter endpoint | `https://openrouter.ai/api/v1` |
| `CRITIC_MODEL` | Content evaluator model (via OpenRouter) | `x-ai/grok-4.1-fast` |
| `CRITIC_VISION_MODEL` | Plot quality evaluator model (via OpenRouter) | `x-ai/grok-4.1-fast` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

---

## Configuration

### The context.md File

`context.md` is the single configuration file that governs pipeline behaviour. It has two parts:

1. **Prose sections** (Markdown) -- injected into agent prompts as domain context. Describe your project, data, and analysis expectations in natural language.
2. **YAML block** (fenced with `` ```yaml ``) -- structured configuration for pipeline stages, quality thresholds, data constraints, ML tasks, and feature toggles.

If no `context.md` is provided, the pipeline uses sensible defaults for all settings. Multiple YAML blocks are merged (later blocks override earlier ones).

### Pipeline Stages Configuration

```yaml
pipeline:
  domain: "chromatography"  # Optional: override auto-detection (chromatography|mass_spectrometry|both)
  stages:
    - name: cleaning
      goals:
        - Remove fully-empty columns
        - Standardise column names
      quality:
        min_findings: 2
      max_retries: 2

    - name: analysis
      goals:
        - Per-run chromatographic overlays
        - Outlier detection by metric deviation
      quality:
        min_plots: 3
        min_findings: 3
        require_per_group: true
      expert_call_budget: 3
      grouping_guidance: |
        Use the schema profiler's dimensional structure to determine grouping.

    - name: cross_validation
      goals:
        - Verify top-3 statistical claims
      agent_hints:
        require: [cross_validator]

    - name: report
      goals:
        - Executive summary with key findings
```

### Data Constraints

```yaml
pipeline:
  constraints:
    preserve_columns: [run_no, run, chromatography_stage, Sample_Code, column, Fraction_number, charge_state, Spectrum_type]
    no_aggregation_across: [run_no]
    grouping_columns: [run_no, chromatography_stage, Sample_Code]
    grouping_extend_when_present: [chromatography_stage, Sample_Code]  # finer grouping when columns exist
    parameters_path: "Data/Parameters.xlsx"  # Optional: supplementary metadata file for experimental conditions
```

When `parameters_path` is set, the pipeline loads the file, summarises its column structure, and builds an experimental condition map (first column = run/experiment ID, remaining columns = conditions). This injects domain context into agent prompts and enables experimental condition annotation in findings.

### Quality Thresholds

| Parameter | Stage | Default | Description |
|-----------|-------|---------|-------------|
| `min_plots` | analysis | 3 | Minimum number of PNG plots |
| `min_findings` | all | 2-3 | Minimum analytical findings |
| `require_per_group` | analysis | true | Must produce per-group structure |
| `max_retries` | cleaning=2, analysis=5, cross_val=2, report=0 | varies | Maximum retry attempts |
| `expert_call_budget` | cleaning=2, analysis=3, cross_val=3, report=2 | varies | Maximum `seek_experts_help` calls (context.md defaults; pipeline fallback uses analysis=9 to accommodate the five-pass strategy) |
| `chat_timeout` | cleaning=900, analysis=2400, cross_val=1800, report=900 | varies | Per-stage wall-clock timeout (seconds) |
| `png_min_bytes` | analysis | 5000 | Minimum PNG file size |
| `summary_size_bounds` | all | (1024, 500000) | Acceptable JSON summary size range |

### Run Configuration (Feature Toggles)

The `run_config` block controls experimental features via independently togglable parameters. This is the mechanism for ablation studies.

#### General Toggles

| Toggle | Type | Default | Description |
|--------|------|---------|-------------|
| `run_label` | string | `""` | Human-readable name for this run |
| `prompt_version` | string | `"v1"` | `"v1"` = baseline, `"v2"` = enhanced (interpretation + p-value), `"v3"` = graduated guidance (dimensional structure awareness) |
| `content_validation` | bool | `true` | Validate JSON content, not just file existence |
| `min_cross_val_claims` | int | `5` | Minimum claims to verify in cross-validation |
| `recompute_cross_val` | bool | `false` | Actually recompute metrics from parquet (`false` = tautological) |
| `figure_selection` | string | `"all"` | `"all"` / `"ranked"` / `"top_n"` for report figures |
| `max_report_figures` | int | `10` | Cap on figures when using ranked/top_n |
| `protected_col_audit` | bool | `false` | Warn on >95% missing protected columns |
| `max_input_tokens` | int | `55000` | Token budget for payload trimming |

#### WP-1: Schema Intelligence

| Toggle | Type | Default | Description |
|--------|------|---------|-------------|
| `schema_profiling` | string | `"disabled"` | `"disabled"` / `"roles_only"` (Phase A: column roles + grouping) / `"full"` (Phase A+B: + distributions + technique recommendations) |

When enabled, the schema profiler classifies every column by role (identifier, categorical_group, ordinal_stage, continuous_measurement, metadata_text, datetime, constant) and infers semantic purposes (experimental_unit, experimental_condition, process_phase, technical_replicate). It builds a dimensional structure with nesting/crossing relationships and recommends grouping candidates scored by quality (group count, evenness, min size, coverage) with a dimensionality bonus (+0.10 per column beyond first, capped at +0.30). In `"full"` mode, it also profiles distributions (normality, modality, skewness), recommends statistical techniques, generates two additional analysis contexts (`interaction_analysis` for 3+ dimension crossings and `deep_process_trend` for full hierarchy with replicates), and includes trajectory plot guidance for ordinal stages.

#### WP-2: Adaptive Convergence

| Toggle | Type | Default | Description |
|--------|------|---------|-------------|
| `iteration_strategy` | string | `"fixed"` | `"none"` (single pass) / `"fixed"` (baseline retries) / `"convergent"` (quality-driven) |
| `convergence_threshold` | float | `0.03` | Min quality score improvement to continue iterating |
| `convergence_target` | float | `0.92` | Quality score at which to stop early (if no MUST_FIX) |
| `should_fix_accumulation_threshold` | int | `4` | Retry if ≥ this many SHOULD_FIX issues accumulated |
| `issue_stall_max_consecutive` | int | `2` | Downgrade persistent issue to SHOULD_FIX after N consecutive attempts |
| `min_iterations` | int | `0` | Minimum attempts before accepting any degraded exit |

#### WP-3: Scientific Visual Review

| Toggle | Type | Default | Description |
|--------|------|---------|-------------|
| `visual_review_mode` | string | `"basic"` | `"basic"` (aesthetic) / `"scientific"` (graduated severity: scientific_validity → MUST_FIX, statistical_completeness → MUST_FIX, layout_quality → SHOULD_FIX, cosmetic → informational; double-evaluation confirms MUST_FIX) |
| `require_figure_references` | bool | `false` | Enforce claim-figure mapping in findings (`findings["figure_ref"]` → disk PNG) |

#### WP-4: Pluggable Agent Library

| Toggle | Type | Default | Description |
|--------|------|---------|-------------|
| `expert_library` | string | `"baseline"` | `"baseline"` (8 core agents) / `"extended"` (+ 3 specialists from YAML) |
| `agent_definitions_dir` | string | `"agents/"` | Path to agent YAML definition files |

#### WP-C: Critic Architecture

| Toggle | Type | Default | Description |
|--------|------|---------|-------------|
| `critic_structural` | bool | `true` | Structural gate (pure Python file/JSON checks) |
| `critic_content` | bool | `true` | LLM content evaluator (rubric-based, via OpenRouter or local vLLM) |
| `critic_visual` | bool | `true` | VLM plot reviewer + plot structural checks (PNG validation pre-filter) |
| `critic_analytical_depth` | bool | `false` | Python heuristic + LLM depth analysis -- 10 checks including domain expectations from YAML (WP-C3a) |
| `critic_execution` | bool | `true` | Execution correctness checks: NaN leakage, impossible values, data-finding mismatch (WP-C3b) |
| `targeted_refinement` | bool | `false` | Enable PLOT_FIX / FINDING_FIX / GAP_FILL paths (WP-C2) |
| `refinement_cascade` | bool | `false` | Escalate through refinement scopes before full rerun |

#### WP-R: Report Quality Controls

These toggles control quality gates applied to the **report pipeline** output (Phase 3d), not the captain pipeline fallback report. When all toggles are at defaults, the quality gate is a no-op and reports pass through unchanged.

| Toggle | Type | Default | Description |
|--------|------|---------|-------------|
| `critic_report` | bool | `true` | LLM rubric check on report pipeline output (6 criteria: executive summary quality, numerical fidelity, conclusion support, data faithfulness, section completeness, synthesis vs enumeration). Uses OpenRouter critic client (`CRITIC_OPENROUTER_API_KEY`). |
| `min_quantitative_grounding` | float | `0.0` | Trigger grounding revision if fraction of quantitative paragraphs is below threshold (0.0 = disabled). Applied after GroupChat produces the report, before figure assembly. |
| `numerical_accuracy_check` | bool | `true` | Pure-Python heuristic: flag numbers in report prose that contradict analysis/cleaning JSON artifacts |

#### ML Modelling

| Toggle | Type | Default | Description |
|--------|------|---------|-------------|
| `ml_backend` | string | `"sklearn"` | `"sklearn"` (RandomForest etc.) / `"tabpfn"` (TabPFN for ≤10K rows, ≤500 features) / `"both"` (TabPFN with sklearn fallback) |

#### Payload Budgets (BS-4)

Configurable character limits for report prompt assembly, tuned for the 128K context window:

| Toggle | Type | Default | Description |
|--------|------|---------|-------------|
| `payload_budget_cleaning` | int | `4000` | Cleaning summary JSON slice |
| `payload_budget_analysis` | int | `8000` | Analysis summary JSON slice |
| `payload_budget_per_file` | int | `3000` | Per-file analysis in global reports |
| `payload_budget_global` | int | `40000` | Global report payload JSON |
| `payload_budget_report_excerpt` | int | `10000` | Individual report excerpt in cross-file reports |

### Agent Selection Hints

```yaml
stages:
  - name: analysis
    agent_hints:
      require: [chromatography_expert, statistical_analyst]
      prefer: [ml_modeler]
      exclude: [mass_spec_expert]
      max_agents: 4
      fallback_to_detection: true  # fall back to domain-based detection if hints insufficient
```

### ML Modelling Tasks

Define supervised learning tasks for the `ml_modeler` agent in context.md. These give the agent concrete prediction targets rather than relying on unsupervised exploration:

```yaml
ml_tasks:
  chromatography:
    - task: "classification"
      target: "chromatography_stage"
      description: "Predict chromatography stage from UV/conductivity features"
      aggregate_by: ["run_no", "chromatography_stage", "column"]
      features: ["UV_1_280_ml", "Conductivity", "volume_ml"]
    - task: "regression"
      target: "peak_area_mean"
      description: "Predict mean peak area from process parameters"
      aggregate_by: ["run_no", "column"]
      features: "auto"
    - task: "classification"
      target: "outlier_flag"
      description: "Classify runs as outlier/normal based on CV threshold"
      derive_target: "cv_threshold(UV_1_280_ml, 0.15)"

  mass_spectrometry:
    - task: "regression"
      target: "dominant_mass_kda_mean"
      features: "auto"
    - task: "classification"
      target: "column"
      features: "auto"
```

---

## Output Structure

Each pipeline run produces a timestamped output directory:

```
Outputs/slurm_{JOB_ID}_{TIMESTAMP}/
|-- manifest_batch_*.json          # Complete run manifest with run_config and quality trajectory
|-- files/{dataset}/
|   |-- cleaning/
|   |   |-- {dataset}__cleaned.parquet
|   |   '-- cleaning_summary.json
|   |-- analysis/
|   |   |-- analysis_summary.json   # findings, per_group, artifacts, domain_reasoning
|   |   |-- data_profile.json       # Schema profiler output (when enabled)
|   |   '-- artifacts/*.png
|   '-- cross_validation/
|       |-- cross_validation.json    # verified_claims, conflicts, gaps, recommendations
|       '-- claims.json              # Verified claims (written unconditionally)
|-- reports/{dataset}/
|   |-- report.md
|   |-- report.html
|   '-- report.pdf
'-- debug/
    |-- *__request.json            # Payload sent to CaptainAgent
    |-- *__response.json           # Structured JSON returned
    |-- *__critic_log.jsonl        # Per-critic dispatch log (checks, severity, wall_time_ms)
    |-- *__structural_gate.json    # Structural gate results
    |-- *__content_eval.json       # Content evaluator rubric scores
    |-- *__plot_eval.json          # VLM plot quality results
    '-- *__quality_gate.json       # Quality gate verdict + score + quality_breakdown
```

---

## Evaluation Framework

An LLM-as-Judge evaluation framework (`Scripts/evaluation/`) enables systematic comparison of pipeline outputs across ablation studies. Configuration is in `evaluation_config.yaml`.

**Architecture**:
- **Deterministic metrics layer** (`deterministic.py`): Ingests pipeline-internal gate quality scores from `debug/*__gate.json` files and computes three report-level metrics without LLM calls: `data_coverage_rate` (fraction of dataset columns referenced in report), `figure_reference_rate` (fraction of generated figures cited), and `quantitative_grounding_rate` (fraction of analytical paragraphs containing numbers — same metric used by the WP-R quality gate in the report pipeline)
- Multi-model judging via OpenRouter (configurable in `evaluation_config.yaml`; vision-capable models recommended)
- Versioned rubrics (`rubrics.py`) with per-dimension scoring
- Pairwise comparison with position-swap consistency checking: each comparison runs twice (A-first, B-first) and the verdict is accepted only when both orderings agree, mitigating position bias
- Configurable artifact truncation (report: 15K chars, analysis: 10K chars, max 10 plots per eval)
- SQLite storage (`Evaluation/evaluation.db`, schema v2) with per-stage gate quality metrics and auto-migration from v1
- **Gate quality vs CV quality split**: gated stages (cleaning, analysis) report `gate.json` quality scores; cross-validation uses a direct call path with no gate loop and is reported separately via structural metrics (`verified_claims_count`, `cv_gaps_count`)
- **Config-level rankings**: Bradley-Terry and Elo scores computed at run level, then aggregated per configuration label by averaging across replications — prevents individual outlier runs from dominating rankings
- **Per-dataset and per-judge score breakdowns**: overall scores disaggregated by dataset and by judge model, surfacing dataset difficulty effects and judge calibration bias
- **Inter-judge agreement**: Krippendorff's alpha per criterion (≥0.8 good, 0.6–0.8 acceptable)
- **Repeatability**: Coefficient of variation (CoV) across replicated runs (<0.15 stable, >0.25 high variance)

**CLI subcommands**:
```bash
python -m Scripts.evaluation deterministic Outputs/slurm_*         # Ingest gate scores + report metrics (no LLM)
python -m Scripts.evaluation evaluate Outputs/slurm_*              # LLM-as-Judge scoring
python -m Scripts.evaluation pairwise Outputs/run_A Outputs/run_B  # Head-to-head comparison
python -m Scripts.evaluation ablation                               # Full ablation report (all configs in DB)
python -m Scripts.evaluation ablation --configs Baseline WP1        # Restrict to specific configs only
python -m Scripts.evaluation repeatability                          # CoV repeatability table
python -m Scripts.evaluation full Outputs/slurm_*                  # Full pipeline (deterministic → evaluate → ablation)
```

**Cross-run comparison** (`compare_runs.py`):
```bash
# Compare specific runs
python Scripts/compare_runs.py Outputs/slurm_123_* Outputs/slurm_456_*

# Auto-discover and compare all runs
python Scripts/compare_runs.py --glob "Outputs/slurm_*" --output comparison.csv
```

See [Evaluation & Ablation Guide](docs/evaluation_and_ablation_guide.md) for full workflow.

---

## Adapting to New Data

1. **Place your data** in a directory (CSV, Excel, or Parquet files)
2. **Write a `context.md`** describing:
   - What the data represents (project overview in prose)
   - Which columns are important (`preserve_columns`)
   - How data should be grouped (`grouping_columns`, `grouping_extend_when_present`)
   - What quality you expect (`min_plots`, `min_findings`)
   - ML modelling tasks if applicable (`ml_tasks` block)
3. **Run the pipeline** pointing at your data directory

With `schema_profiling: "full"` enabled, the system classifies columns by role, infers semantic purposes, detects dimensional structure (nesting/crossing relationships), profiles distributions, and recommends statistical techniques -- all without domain-specific keywords.

**No code changes needed** -- `context.md` is the only thing that changes between projects. New agent experts can be added by dropping a YAML file into `agents/`.

---

## Known Limitations

- **BS-1**: Global report figure references may point to per-file analysis figures instead of global figures -- partially mitigated by not passing per-file figures to the global report GroupChat
- **BS-5**: Global VisualisationAgent code fails to properly read/merge parquet files with differing column names -- partially mitigated by pre-reading column metadata from cleaned parquets, but still relies on LLM following instructions
- Pipeline is fully sequential -- no file-level parallelism, so multi-GPU provides no throughput gain
- vLLM TP≥2 causes instability (malformed tool calls); TP=1 recommended
- DeepResearchAgent with Qwen3.5 can enter infinite decomposition loops (mitigated by `max_turns=10` cap, circuit breaker after 3 failures)

**Previously resolved**:
- ~~BS-2: Figure path short-circuit~~ -- `_assemble_report_with_figures` resolves broken `![` refs via stem matching, figure-number matching, and appends unreferenced figures with themed grouping
- ~~BS-3: GroupChat round exhaustion~~ -- mitigated by convergence control (WP-2), disk-artifact recovery, and quality-driven stopping
- ~~BS-4: Payload truncation~~ -- replaced hardcoded char slices with configurable `payload_budget_*` fields in run_config, tuned for 128K context window
- ~~Critic feedback loop non-functional~~ -- replaced by modular critic registry (WP-C1) with deterministic dispatch and per-critic toggling
- ~~Content evaluator oscillation~~ -- stabilised via criteria merge (8→7), few-shot examples, reference-anchored evaluation, and check subsumption de-duplication
- ~~Wholesale replacement on retry~~ -- mitigated by regression firewall (artifact rollback on net regression), surgical finding merge, and explicit preservation instructions
- ~~Contradictory check pressure~~ -- resolved by check dependency ordering (severity downgrade cascades) and retry instruction deduplication via subsumption map
- ~~Stall threshold unreachable~~ -- `issue_stall_max_consecutive` reduced from 4 to 2, aligned with typical retry budgets
- ~~Report pipeline ignores degraded flag~~ -- degraded analyses now receive quality caveat in reports; report quality gate defaults to enabled
- ~~Duplicate grouping checks~~ -- removed from StructuralCritic; sole ownership moved to AnalyticalDepthCritic

---

## Troubleshooting

| Issue | Likely Cause | Solution |
|-------|-------------|----------|
| `'dict' object has no attribute 'config_list'` | AG2 >= 0.11 LLMConfig API change | Ensure `LLMConfig(*config_entries, **kwargs)` |
| `System message must be at the beginning` | Qwen3.5 chat template requirement | Handled automatically by monkey-patch |
| `No user query found in messages` | Qwen3.5 requires at least one user message | Handled by `_patched_oai_create` |
| Expert teams loop indefinitely | Budget exhaustion or malformed tool calls | Check `expert_call_budget` in context.md; malformed tool-call guard auto-aborts after 3 consecutive bad calls |
| vLLM server not starting | CUDA version mismatch or OOM | Verify CUDA 12.x and GPU memory (~55GB for 27B); clear stale caches: `rm -rf ~/.cache/flashinfer/ ~/.cache/vllm/torch_compile_cache/` |
| `CUBLAS_STATUS_INVALID_VALUE` | cuBLAS 12.9 module loaded, shadowing torch's bundled cuBLAS 12.8 | Set CUDA_HOME manually instead of `module load cuda`; set `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| Empty analysis output | Insufficient max_round for GroupChat or disk recovery failed | Increase `_STAGE_MAX_ROUNDS`; check backup analysis_summary.json |
| Pipeline timeout on SLURM | Run exceeded wall time | Increase `-t` in `Batch_Script.sh` (default: 12h) |
| Content evaluator returns no criteria | OpenRouter API key missing or model unavailable | Set `CRITIC_OPENROUTER_API_KEY`; falls back to local vLLM if unavailable |
| Visual critic skipped | VLM not detected (non-multimodal model) | Ensure `--limit-mm-per-prompt '{"image": 4}'` in vLLM args |
| `PermissionError: /local/slurm.<old_job>` | Stale torch inductor cache referencing previous TMPDIR | Set `TORCHINDUCTOR_CACHE_DIR` to persistent path; clear `~/.cache/vllm/torch_compile_cache/` |

**Debug logs** are saved to `Outputs/{RUN_ID}/debug/` and contain full request/response payloads, critic dispatch logs, and quality gate verdicts. SLURM stderr/stdout files contain the complete execution trace.

---

## See Also

- **[Evaluation & Ablation Guide](docs/evaluation_and_ablation_guide.md)** -- Ablation study workflows, run comparison, LLM-as-Judge evaluation framework, replicated runs, statistical analysis, and LaTeX export
- **[context.md](context.md)** -- Current pipeline configuration with all active toggles and WP status
- **[evaluation_config.yaml](evaluation_config.yaml)** -- LLM-as-Judge configuration (OpenRouter models, rubric version, pairwise settings)
