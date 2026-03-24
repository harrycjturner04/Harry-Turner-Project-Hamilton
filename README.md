# Multi-Agent Biologics Data Analysis Pipeline

A fully autonomous, LLM-driven system for analysing biologics experimental data using the [AG2 (AutoGen)](https://github.com/ag2ai/ag2) multi-agent framework. The pipeline dynamically assembles expert agent teams, executes quantitative analysis via code generation, validates results through a modular critic architecture, and produces publication-quality scientific reports.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Pipeline Stages](#pipeline-stages)
  - [Agent Orchestration](#agent-orchestration)
  - [Gated Review Loop](#gated-review-loop)
  - [Critic Architecture](#critic-architecture)
  - [Targeted Refinement](#targeted-refinement)
  - [Agent Library](#agent-library)
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
- [Output Structure](#output-structure)
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
  Cleaning --> Analysis --> Cross-Validation --> Reporting
     |            |               |                  |
     v            v               v                  v
  Cleaned     Findings +      Verified          Scientific
  Parquet     Plots + JSON    Claims JSON       Report (PDF)
```

| Stage | Purpose | Output |
|-------|---------|--------|
| **Cleaning** | Remove empty columns, standardise formats, flag anomalies | Cleaned parquet + `cleaning_summary.json` |
| **Analysis** | Generate statistical findings, plots, per-group metrics | `analysis_summary.json` + PNG plots |
| **Cross-Validation** | Re-compute top claims from raw data to verify accuracy | `cross_validation.json` with verified claims |
| **Reporting** | Synthesise narrative report with literature context | `report.md`, `report.html`, `report.pdf` |

### Agent Orchestration

The system uses AG2's **CaptainAgent** as the top-level orchestrator:

```
User Proxy (captain_user)
    |
    v
CaptainAgent
    |-- calls seek_experts_help() -->  AutoBuild
    |                                     |
    |                                     v
    |                               Expert GroupChat
    |                               [data_cleaner, chromatography_expert,
    |                                Computer_terminal, ...]
    |                                     |
    |<-- structured JSON result ----------+
    |
    v
Next stage...
```

1. **captain_user** sends the task instruction to **CaptainAgent**
2. CaptainAgent decomposes the task and calls `seek_experts_help()` with a team specification
3. AG2's **AutoBuild** assembles an expert GroupChat from the agent library
4. Experts collaborate -- proposing code, executing it via `Computer_terminal`, and producing structured JSON
5. The result flows back to CaptainAgent for the next stage

### Gated Review Loop

Each stage attempt passes through a **modular critic registry** that dispatches evaluators in a deterministic order. The registry replaces the earlier "4-phase" description -- it is a pluggable system where each critic module independently decides whether to run and what severity to assign.

**Design principle**: Deterministic governance, non-deterministic analysis. Python code makes all gate/routing decisions; LLMs assess quality and produce structured signals.

The registry dispatches critics in order, aggregates their `CheckResult` outputs into a `StageVerdict`, and feeds the verdict into a deterministic `quality_gate()` function that decides pass/retry/degrade.

### Critic Architecture

The `critics/` directory contains modular critic modules, each implementing the `CriticModule` ABC. The `CriticRegistry` instantiates all available critics at pipeline init and dispatches them per stage attempt.

**Execution order** (lower number runs first):

| Order | Critic | Type | Toggle | Default | What it checks |
|-------|--------|------|--------|---------|----------------|
| 10 | `StructuralCritic` | Python | `critic_structural` | `true` | File existence, JSON validity, min plots/findings, PNG size |
| 20 | `ExecutionCritic` | Python | `critic_execution` | `false` | NaN leakage, impossible values, uniform groups, duplicate p-values |
| 30 | `ContentCritic` | LLM | `critic_content` | `true` | Rubric-based scoring (6 criteria for analysis, 4 for cleaning, 3 for cross-validation) |
| 40 | `AnalyticalDepthCritic` | Python + LLM | `critic_analytical_depth` | `false` | Grouping adequacy, group comparison tests, chart diversity, interpretation depth, domain expectations, outlier analysis |
| 50 | `PlotStructuralCritic` | Python | `critic_visual` | `true` | Plot resolution, aspect ratio, size outliers |
| 60 | `VisualCritic` | VLM | `critic_visual` | `true` | Plot quality -- basic (aesthetic) or scientific (chart type, grouping, annotations) |

**Short-circuit optimisation**: If the structural gate fails, LLM/VLM critics are skipped (they need valid artifacts to evaluate).

**Quality scoring**: Each `CheckResult` maps to passed=1.0, should_fix=0.5, must_fix=0.0. Content and plot checks receive 2x weight. A hard cap ensures content/plot `MUST_FIX` issues cap the score at 0.80, preventing convergent early-stop from silently accepting substantive failures.

**Gate decision logic** (`quality_gate()` in `tools.py`):
- No must_fix + evaluators ran -> `passed`
- Only should_fix -> `passed` with warnings
- must_fix + retry budget exhausted -> `passed_degraded`
- must_fix + same failures as previous attempt -> `passed_degraded` (stall detection)
- must_fix + budget remaining -> route to retry tier

### Targeted Refinement

When `targeted_refinement: true`, the system attempts surgical fixes before falling back to a full CaptainAgent retry. Refinement directives are built from the quality gate's analysis of must_fix failures:

| Scope | When triggered | What happens |
|-------|---------------|-------------|
| `PLOT_FIX` | Plot-only must_fix failures | Regenerate failing plots without re-running full analysis |
| `FINDING_FIX` | Content must_fix with identifiable ref | Fix specific finding without re-running all analysis |
| `GAP_FILL` | Analytical depth must_fix (`depth__` prefix) | Fill specific analytical gap |
| `FULL_RERUN` | Always available as final escalation | Full CaptainAgent retry with retry instructions |

When `refinement_cascade: true`, the system escalates through scopes (most targeted first). When `false`, only the first matching scope is attempted.

### Agent Library

Agents are selected dynamically per-task. With `expert_library: "baseline"` (default), the 8 core agents are used. With `expert_library: "extended"`, 3 additional specialist agents are available. Agent definitions live in `agents/*.yaml`.

**Core agents** (always available):

| Agent | Role |
|-------|------|
| `data_cleaner` | Conservative data cleaning with protected column awareness |
| `chromatography_expert` | Baseline correction, peak detection, system suitability |
| `mass_spec_expert` | Mass accuracy, charge-state validation, glycoform analysis |
| `statistical_analyst` | Descriptive statistics, correlations, outlier detection |
| `analysis_planner` | Recommends analysis strategies before execution |
| `ml_modeler` | Machine learning approaches (classification, clustering) |
| `cross_validator` | Independent re-computation of statistical claims |
| `report_writer` | Synthesises scientific narratives from findings |

**Extended agents** (available when `expert_library: "extended"`):

| Agent | Role | Activation Criteria |
|-------|------|-------------------|
| `bioprocess_analyst` | Upstream/downstream process data, yield, step recovery | ordinal_stage + continuous_measurement columns |
| `visualisation_specialist` | Chart type selection, statistical annotations, plot design | Active when `visual_review_mode: "scientific"` |
| `statistical_modeler` | Mixed-effects models, DoE, multivariate techniques | >=4 numeric columns, >=50 rows |

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
|   |-- main.py                 # CLI entry point
|   |-- captain_pipeline.py     # Core orchestrator (CaptainAgent, stages, gated review)
|   |-- schema_profiler.py      # WP-1: Column role classification and grouping inference
|   |-- report_pipeline.py      # Report generation (DeepResearch + GroupChat)
|   |-- prompts.py              # All agent system prompts (core + extended)
|   |-- tools.py                # Utility functions, structural gate, quality gate
|   |-- context_parser.py       # Parses context.md into RunPlan
|   |-- pdf_writer.py           # Markdown -> HTML -> PDF conversion
|   |-- database.py             # Data discovery, metadata vectorisation
|   |-- compare_runs.py         # Cross-run comparison utility
|   |-- critics/                # WP-C1: Modular critic architecture
|   |   |-- __init__.py
|   |   |-- base.py             # CriticModule ABC, CriticContext dataclass
|   |   |-- registry.py         # CriticRegistry: discovery, ordering, dispatch
|   |   |-- structural.py       # Pure Python file/JSON checks
|   |   |-- content.py          # LLM rubric-based evaluation
|   |   |-- visual.py           # VLM plot quality review
|   |   |-- analytical_depth.py # WP-C3a: Python heuristic + LLM depth analysis
|   |   |-- execution.py        # WP-C3b: Execution correctness checks
|   |   |-- plot_structural.py  # WP-C4: PNG resolution/aspect validation
|   |   '-- vlm_calibration.py  # Offline VLM calibration tool
|   '-- evaluation/             # Evaluation framework (fully decoupled)
|       |-- cli.py, config.py, registry.py, artifacts.py
|       |-- rubrics.py, judge.py, scoring.py, pairwise.py
|       |-- statistics.py, ablation.py, visualization.py, storage.py
|       '-- launch_replicates.sh
|
|-- context.md                  # Project configuration (governs pipeline behaviour)
|-- Batch_Script.sh             # SLURM submission script
|-- Environments/
|   '-- Project_Env.yml         # Conda environment specification
|-- docs/
|   '-- evaluation_and_ablation_guide.md  # Evaluation & ablation reference
|
|-- Database/                   # Built at runtime
|-- Evaluation/                 # Evaluation outputs (created at runtime)
'-- Outputs/                    # Pipeline run outputs
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
```

**Key dependencies** (see `Environments/Project_Env.yml` for full list):

| Package | Purpose |
|---------|---------|
| `ag2[openai,captainagent,browser-use,crawl4ai]>=0.11` | Multi-agent framework |
| `vllm>=0.17.0` | Local LLM inference server |
| `torch>=2.10.0` | GPU compute backend |
| `pandas>=2.1`, `numpy`, `scipy>=1.17` | Data processing |
| `matplotlib>=3.7`, `seaborn>=0.12` | Visualisation |
| `sentence-transformers>=2.2.0`, `chromadb>=0.4.6` | Metadata vectorisation |
| `weasyprint>=60.0` | PDF generation |
| `openai>=1.40` | LLM API client |

---

## Usage

### Quick Start (SLURM)

```bash
# Place your data files in the Data/ directory
# Edit context.md to describe your project and constraints
sbatch Batch_Script.sh
```

### Manual Execution

```bash
# 1. Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-27B \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --enable-prefix-caching &

# 2. Build the database
python Scripts/database.py

# 3. Run the pipeline
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="not-needed-for-local"

python Scripts/main.py Database/raw Outputs/my_run \
    --model "Qwen/Qwen3.5-27B" \
    --context-path context.md \
    --pipeline-mode stage_validation
```

### CLI Reference

```
python Scripts/main.py INPUT_DIR OUTPUT_DIR [OPTIONS]

Positional Arguments:
  INPUT_DIR              Path to raw data files (parquet, CSV, etc.)
  OUTPUT_DIR             Path to write outputs

Options:
  --model MODEL          LLM model name (default: Qwen/Qwen3.5-27B)
  --temperature TEMP     Sampling temperature (default: 0.0)
  --api-key KEY          OpenAI API key (or set OPENAI_API_KEY)
  --base-url URL         LLM endpoint URL (or set OPENAI_BASE_URL)
  --metadata-db PATH     Path to metadata database directory
  --context-path PATH    Path to context.md (auto-discovered if absent)
  --log-level LEVEL      Logging level: DEBUG, INFO, WARNING (default: INFO)
  --pipeline-mode MODE   single_pass | stage_validation | full_closed_loop
  --skip-report          Skip report generation
  --report-only          Run only the report pipeline on an existing manifest
```

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENAI_API_KEY` | API key for LLM endpoint | Required |
| `OPENAI_BASE_URL` | LLM server URL (e.g. vLLM) | Required |
| `OPENAI_MODEL` | Model name override | CLI `--model` |
| `PIPELINE_MODE` | Execution mode | `stage_validation` |
| `PIPELINE_CHAT_TIMEOUT_S` | Wall-clock timeout per chat session (seconds) | 600 |
| `PIPELINE_MIN_PLOTS_ANALYSIS` | Override min plots for analysis | 3 |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

---

## Configuration

### The context.md File

`context.md` is the single configuration file that governs pipeline behaviour. It has two parts:

1. **Prose sections** (Markdown) -- injected into agent prompts as domain context. Describe your project, data, and analysis expectations in natural language.
2. **YAML block** (fenced with `` ```yaml ``) -- structured configuration for pipeline stages, quality thresholds, and data constraints.

If no `context.md` is provided, the pipeline uses sensible defaults for all settings.

### Pipeline Stages Configuration

```yaml
pipeline:
  domain: "chromatography"  # Optional: override auto-detection
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
      expert_call_budget: 4

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
    preserve_columns: [run_no, run, chromatography_stage, Sample_Code, column]
    no_aggregation_across: [run_no]
    grouping_columns: [run_no, chromatography_stage, Sample_Code]
```

### Quality Thresholds

| Parameter | Stage | Default | Description |
|-----------|-------|---------|-------------|
| `min_plots` | analysis | 3 | Minimum number of PNG plots |
| `min_findings` | all | 2-3 | Minimum analytical findings |
| `require_per_group` | analysis | true | Must produce per-group structure |
| `max_retries` | all | 0-2 | Maximum retry attempts |
| `expert_call_budget` | all | 2-4 | Maximum `seek_experts_help` calls |

### Run Configuration (Feature Toggles)

The `run_config` block controls experimental features via independently togglable parameters. This is the mechanism for ablation studies.

#### General Toggles

| Toggle | Type | Default | Description |
|--------|------|---------|-------------|
| `run_label` | string | `""` | Human-readable name for this run |
| `prompt_version` | string | `"v1"` | `"v1"` = baseline prompts, `"v2"` = enhanced (interpretation + p-value requirements) |
| `content_validation` | bool | `true` | Validate JSON content, not just file existence |
| `min_cross_val_claims` | int | `5` | Minimum claims to verify in cross-validation |
| `recompute_cross_val` | bool | `false` | Actually recompute metrics from parquet (`false` = tautological) |
| `figure_selection` | string | `"all"` | `"all"` / `"ranked"` / `"top_n"` for report figures |
| `max_report_figures` | int | `10` | Cap on figures when using ranked/top_n |
| `rerun_issue_map_enabled` | bool | `false` | Inject targeted corrective instructions on retry |
| `protected_col_audit` | bool | `false` | Warn on >95% missing protected columns |

#### WP-1: Schema Intelligence

| Toggle | Type | Default | Description |
|--------|------|---------|-------------|
| `schema_profiling` | string | `"disabled"` | `"disabled"` / `"roles_only"` / `"full"` -- column role classification and grouping inference |

#### WP-2: Adaptive Convergence

| Toggle | Type | Default | Description |
|--------|------|---------|-------------|
| `iteration_strategy` | string | `"fixed"` | `"none"` (single pass) / `"fixed"` (baseline retries) / `"convergent"` (quality-driven) |
| `convergence_threshold` | float | `0.05` | Min improvement to continue iterating |
| `convergence_target` | float | `0.85` | Quality score to stop early |

#### WP-3: Scientific Visual Review

| Toggle | Type | Default | Description |
|--------|------|---------|-------------|
| `visual_review_mode` | string | `"basic"` | `"basic"` (aesthetic) / `"scientific"` (graduated severity) |
| `require_figure_references` | bool | `false` | Enforce claim-figure mapping in findings |

#### WP-4: Pluggable Agent Library

| Toggle | Type | Default | Description |
|--------|------|---------|-------------|
| `expert_library` | string | `"baseline"` | `"baseline"` (8 core agents) / `"extended"` (+ 3 specialists from YAML) |
| `agent_definitions_dir` | string | `"agents/"` | Path to agent YAML definition files |

#### WP-C: Critic Architecture

| Toggle | Type | Default | Description |
|--------|------|---------|-------------|
| `critic_structural` | bool | `true` | Structural gate (pure Python file/JSON checks) |
| `critic_content` | bool | `true` | LLM content evaluator (rubric-based) |
| `critic_visual` | bool | `true` | VLM plot reviewer + plot structural checks |
| `critic_analytical_depth` | bool | `false` | Python heuristic + LLM depth analysis (WP-C3a) |
| `critic_execution` | bool | `false` | Execution correctness checks (WP-C3b) |
| `targeted_refinement` | bool | `false` | Enable PLOT_FIX / FINDING_FIX / GAP_FILL paths (WP-C2) |
| `refinement_cascade` | bool | `false` | Escalate through refinement scopes before full rerun |

### Agent Selection Hints

```yaml
stages:
  - name: analysis
    agent_hints:
      require: [chromatography_expert, statistical_analyst]
      prefer: [ml_modeler]
      exclude: [mass_spec_expert]
      max_agents: 4
```

---

## Output Structure

Each pipeline run produces a timestamped output directory:

```
Outputs/slurm_{JOB_ID}_{TIMESTAMP}/
|-- manifest_batch_*.json          # Complete run manifest with run_config
|-- files/{dataset}/
|   |-- cleaning/
|   |   |-- {dataset}__cleaned.parquet
|   |   '-- cleaning_summary.json
|   |-- analysis/
|   |   |-- analysis_summary.json
|   |   '-- artifacts/*.png
|   '-- cross_validation/
|       '-- cross_validation.json
|-- reports/{dataset}/
|   |-- report.md
|   |-- report.html
|   '-- report.pdf
'-- debug/
    |-- *__request.json            # Payload sent to CaptainAgent
    |-- *__response.json           # Structured JSON returned
    |-- *__structural_gate.json    # Structural gate results
    |-- *__content_eval.json       # Content evaluator rubric scores
    |-- *__plot_eval.json          # VLM plot quality results
    '-- *__quality_gate.json       # Quality gate verdict + score
```

---

## Adapting to New Data

1. **Place your data** in a directory (CSV, Excel, or Parquet files)
2. **Write a `context.md`** describing:
   - What the data represents (project overview in prose)
   - Which columns are important (`preserve_columns`)
   - How data should be grouped (`grouping_columns`)
   - What quality you expect (`min_plots`, `min_findings`)
3. **Run the pipeline** pointing at your data directory

With `schema_profiling` enabled, the system classifies columns by role and infers grouping structures without domain-specific keywords.

**No code changes needed** -- `context.md` is the only thing that changes between projects.

---

## Known Limitations

- **BS-1**: Global report figure references may point to per-file analysis figures instead of global figures
- **BS-2**: `_assemble_report_with_figures` short-circuits when report already contains `![` references
- **BS-4**: Payload truncation at 12K chars (per-file) and 15K chars (global) loses critical analysis findings
- **BS-5**: Global VisualisationAgent code fails to properly read/merge parquet files with differing column names
- Pipeline is fully sequential -- no file-level parallelism

---

## Troubleshooting

| Issue | Likely Cause | Solution |
|-------|-------------|----------|
| `'dict' object has no attribute 'config_list'` | AG2 >= 0.11 LLMConfig API change | Ensure `LLMConfig(*config_entries, **kwargs)` |
| `System message must be at the beginning` | Qwen3.5 chat template requirement | Handled automatically by monkey-patch |
| `No user query found in messages` | Qwen3.5 requires at least one user message | Handled by `_patched_oai_create` |
| Expert teams loop indefinitely | Budget exhaustion or missing termination | Check `expert_call_budget` in context.md |
| vLLM server not starting | CUDA version mismatch or OOM | Verify CUDA 12.x and GPU memory (~55GB for 27B) |
| Empty analysis output | Insufficient `max_round` for GroupChat | Increase `_STAGE_MAX_ROUNDS` |
| Pipeline timeout on SLURM | Run exceeded wall time | Increase `-t` in `Batch_Script.sh` |

**Debug logs** are saved to `Outputs/{RUN_ID}/debug/` and contain full request/response payloads. SLURM stderr/stdout files contain the complete execution trace.

---

## See Also

- **[Evaluation & Ablation Guide](docs/evaluation_and_ablation_guide.md)** -- Ablation study workflows, run comparison, LLM-as-Judge evaluation framework, replicated runs, statistical analysis, and LaTeX export
- **[context.md](context.md)** -- Current pipeline configuration with all active toggles and WP status
