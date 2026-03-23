# Multi-Agent Biologics Data Analysis Pipeline

A fully autonomous, LLM-driven system for analysing biologics experimental data using the [AG2 (AutoGen)](https://github.com/ag2ai/ag2) multi-agent framework. The pipeline dynamically assembles expert agent teams, executes quantitative analysis via code generation, validates results, and produces publication-quality scientific reports -- all without hardcoded analysis logic.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
  - [Pipeline Stages](#pipeline-stages)
  - [Agent Orchestration](#agent-orchestration)
  - [Agent Library](#agent-library)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
  - [Data Flow](#data-flow)
  - [Domain Detection](#domain-detection)
  - [Context-Driven Configuration](#context-driven-configuration)
  - [Execution Modes](#execution-modes)
  - [Validation and Quality Assurance](#validation-and-quality-assurance)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Environment Setup](#environment-setup)
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
- [Ablation Studies & Run Comparison](#ablation-studies--run-comparison)
  - [How It Works](#how-ablation-works)
  - [Configuring Ablation Runs](#configuring-ablation-runs)
  - [Feature Toggle Reference](#feature-toggle-reference)
  - [Pre-Defined Run Configurations](#pre-defined-run-configurations)
  - [Comparing Runs](#comparing-runs)
  - [Evaluation with LLM-as-Judge](#evaluation-with-llm-as-judge)
- [Evaluation Framework](#evaluation-framework)
  - [Overview](#evaluation-overview)
  - [Setup](#evaluation-setup)
  - [Running Evaluations](#running-evaluations)
  - [Replicated Runs](#replicated-runs)
  - [Evaluation Rubric](#evaluation-rubric)
  - [Pairwise Comparison](#pairwise-comparison)
  - [Statistical Analysis](#statistical-analysis)
  - [Outputs and Exports](#outputs-and-exports)
  - [CLI Reference](#evaluation-cli-reference)
- [Output Structure](#output-structure)
  - [Directory Layout](#directory-layout)
  - [Manifest File](#manifest-file)
  - [Debug Artifacts](#debug-artifacts)
- [Adapting to New Data](#adapting-to-new-data)
- [Design Philosophy](#design-philosophy)
- [Recent Improvements](#recent-improvements)
- [Troubleshooting](#troubleshooting)

---

## Overview

This system analyses biologics experimental data (chromatography, mass spectrometry, or both) by orchestrating a team of specialised LLM-powered agents. Rather than following a fixed analysis script, each agent reasons from the data it observes -- inspecting column names, value distributions, and domain context to decide what analyses are appropriate.

The pipeline is governed by a single configuration file (`context.md`) that describes the project goals, data constraints, and quality expectations. This makes the system adaptable to different biologics datasets without code changes.

**Current model**: Qwen/Qwen3.5-27B served locally via [vLLM](https://github.com/vllm-project/vllm) on a single GPU (NVIDIA H200 NVL 144GB).

---

## Key Features

- **Inference-driven analysis** -- agents inspect the data and decide what to do; no hardcoded pipelines
- **Dynamic agent assembly** -- CaptainAgent selects the right experts for each task from an agent library
- **Four-stage pipeline** -- cleaning, analysis, cross-validation, and reporting with validation loops
- **Multi-domain support** -- chromatography, mass spectrometry, or both, auto-detected or overridden
- **Context-governed** -- a single `context.md` file controls pipeline behaviour, quality thresholds, and constraints
- **Code execution sandbox** -- agents write and execute Python code in isolated processes with timeout protection
- **Cross-validation** -- independent re-computation of statistical claims against raw data
- **Publication-quality reports** -- Markdown, HTML, and PDF output with embedded figures and literature context
- **Semantic metadata search** -- vectorised metadata (via ChromaDB + SentenceTransformers) for context injection
- **Robust error handling** -- expert call budgets, disk-artifact recovery, deterministic quality guardrails, auto-retry
- **Two-pass analysis strategy** -- analysis planner recommends diverse approaches before domain experts execute, separating strategy from implementation
- **Gated review loop** -- 4-phase deterministic validation per stage (structural gate, content evaluator, VLM plot review, quality gate) with domain-specific rubrics
- **Structured context parser** -- `context.md` with embedded YAML is parsed into a typed `RunPlan` with per-stage quality specs, agent hints, and data constraints
- **Dynamic code guidance** -- shared code execution rules injected into all expert prompts with runtime-detected library versions (pandas, scipy, numpy)
- **Truncation loop detection** -- detects and breaks infinite loops from truncated code blocks; auto-repairs unclosed code fences
- **Wall-clock timeout protection** -- SIGALRM-based timeout prevents runaway agent chats from blocking the pipeline
- **Transient error retry** -- exponential backoff for vLLM 503/429/502 errors with automatic recovery
- **Scientific visual review** (WP-3) -- VLM-based reviewer with graduated severity: scientific validity issues (chart type, grouping, axis scaling) trigger retries, cosmetic issues are informational; claim-figure traceability validates that findings reference existing plots
- **Row removal safeguards** -- data cleaner enforces a 10% row removal ceiling and distinguishes domain-valid values from true anomalies
- **Schema intelligence** (WP-1) -- deterministic column role classification and grouping inference replaces hardcoded column lists; domain-agnostic profiling adapts to any data format
- **Adaptive convergence** (WP-2) -- quality-driven iteration control with numeric scoring, content evaluator hardening, and configurable convergence strategies
- **Pluggable agent library** (WP-4) -- YAML-driven agent definitions with schema-based activation scoring; new experts added by dropping a file into `agents/` with no source changes

---

## Architecture

### Pipeline Stages

The system processes each input dataset through four sequential stages:

```
  Cleaning ──> Analysis ──> Cross-Validation ──> Reporting
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

### Agent Library

Agents are selected dynamically per-task. With `expert_library: "baseline"` (default), the 8 core agents are used. With `expert_library: "extended"`, 3 additional specialist agents are available. Agent definitions live in `agents/*.yaml` and can be added without modifying source code.

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
| `bioprocess_analyst` | Upstream/downstream process data, yield, step recovery | ordinal_stage + continuous_measurement columns; process keywords |
| `visualisation_specialist` | Chart type selection, statistical annotations, plot design | Active when `visual_review_mode: "scientific"` |
| `statistical_modeler` | Mixed-effects models, DoE, multivariate techniques | ≥4 numeric columns, ≥50 rows |

**Infrastructure agent** (managed by AG2):

| Agent | Role |
|-------|------|
| `Computer_terminal` | Executes Python code in a sandboxed environment |

---

## Project Structure

```
.
├── agents/                        # WP-4: YAML agent definitions (plug-in system)
│   ├── data_cleaner.yaml
│   ├── chromatography_expert.yaml
│   ├── mass_spec_expert.yaml
│   ├── statistical_analyst.yaml
│   ├── analysis_planner.yaml
│   ├── ml_modeler.yaml
│   ├── cross_validator.yaml
│   ├── report_writer.yaml
│   ├── bioprocess_analyst.yaml     # Extended (WP-4C)
│   ├── visualisation_specialist.yaml  # Extended (WP-4C)
│   └── statistical_modeler.yaml    # Extended (WP-4C)
│
├── Scripts/
│   ├── main.py                 # CLI entry point
│   ├── captain_pipeline.py     # Core orchestrator (CaptainAgent, stages, monkey-patches)
│   ├── schema_profiler.py      # WP-1: Column role classification and grouping inference
│   ├── report_pipeline.py      # Report generation (DeepResearch + GroupChat)
│   ├── prompts.py              # All agent system prompts (core + extended)
│   ├── tools.py                # Utility functions, structural gate, quality gate
│   ├── context_parser.py       # Parses context.md into RunPlan (all WP config fields)
│   ├── pdf_writer.py           # Markdown -> HTML -> PDF conversion
│   ├── database.py             # Data discovery, metadata vectorisation
│   ├── compare_runs.py         # Cross-run comparison utility
│   └── evaluation/             # Evaluation framework (fully decoupled)
│       ├── cli.py              # CLI entry point (python -m Scripts.evaluation)
│       ├── config.py           # EvalConfig, JudgeModelSpec, OpenRouter client
│       ├── registry.py         # Run discovery and grouping
│       ├── artifacts.py        # Artifact indexing (no copying)
│       ├── rubrics.py          # 7-criterion evaluation rubric + prompt templates
│       ├── judge.py            # LLM-as-judge via OpenRouter (text + vision)
│       ├── scoring.py          # Multi-judge aggregation
│       ├── pairwise.py         # Pairwise comparison, Bradley-Terry, Elo
│       ├── statistics.py       # Bootstrap CI, Mann-Whitney, Cohen's d
│       ├── ablation.py         # Automated ablation analysis + reporting
│       ├── visualization.py    # Figures (radar, heatmap, forest, violin plots)
│       ├── storage.py          # SQLite database
│       └── launch_replicates.sh # SLURM launcher for replicate runs
│
├── context.md                  # Project configuration (governs pipeline behaviour)
├── Batch_Script.sh             # SLURM submission script
├── evaluation_config.example.yaml  # Evaluation config template
├── Environments/
│   └── Project_Env.yml         # Conda environment specification
│
├── Database/                   # Built at runtime
│   ├── raw/                    # Input data as parquet
│   └── metadata/               # Vectorised metadata (ChromaDB)
│
├── Evaluation/                 # Evaluation outputs (created at runtime)
│   ├── evaluation.db           # SQLite database
│   ├── figures/                # Comparison plots
│   ├── reports/                # Ablation analysis reports
│   ├── raw_judgments/          # Full judge responses (provenance)
│   └── exports/                # LaTeX tables
│
└── Outputs/                    # Pipeline run outputs
    └── slurm_{JOB_ID}_{TIMESTAMP}/
        ├── manifest_batch_*.json
        ├── files/{dataset}/
        │   ├── cleaning/
        │   ├── analysis/
        │   └── cross_validation/
        ├── reports/{dataset}/
        │   ├── report.md
        │   ├── report.html
        │   └── report.pdf
        └── debug/
```

---

## How It Works

### Data Flow

```
Raw Data Files (.csv, .xlsx, .parquet)
        |
        v
  [database.py] ── Discover files, classify metadata vs raw,
        |           convert to parquet, vectorise metadata
        v
  Database/raw/*.parquet    Database/metadata/ (ChromaDB)
        |                          |
        v                          v
  [main.py] ──────> [captain_pipeline.py]
                          |
                    For each dataset:
                          |
                    1. Inspect table (columns, dtypes, missingness, preview)
                    2. Detect domain (chromatography / MS / both)
                    3. Load context.md constraints
                    4. Run Cleaning stage
                    5. Run Analysis stage (two-pass: plan then execute)
                    6. Run Cross-Validation stage
                    7. Collect into manifest
                          |
                          v
                    [report_pipeline.py]
                          |
                    1. Web research (DeepResearchAgent)
                    2. Multi-agent GroupChat for interpretation
                    3. Generate Markdown -> HTML -> PDF
                          |
                          v
                    Final Reports + Manifest
```

### Domain Detection

The system uses a three-tier approach to determine what kind of data it is processing:

1. **Tier 1 -- Context override** (highest priority): If `context.md` specifies `pipeline.domain: chromatography`, that is used directly
2. **Tier 2 -- Keyword detection**: Column names are scanned against known keyword sets:
   - *Chromatography*: `retention_time`, `rt`, `peak_area`, `mau`, `au`, `elution`, `gradient`, `uv_280`, etc.
   - *Mass spectrometry*: `m/z`, `mz`, `charge_state`, `molecular_weight`, `tic`, `bpc`, `dalton`, etc.
3. **Tier 3 -- LLM classification** (fallback): If keyword detection is ambiguous, the LLM is asked to classify the data based on column names and sample values

Domain detection drives agent selection -- chromatography data gets a `chromatography_expert`, MS data gets a `mass_spec_expert`, and mixed datasets get both.

### Context-Driven Configuration

The entire pipeline is governed by `context.md`. This file combines natural language project description with structured YAML configuration:

```markdown
# My Biologics Project

## Project Overview
This project analyses IEX chromatography data from a DoE study...

## Data Domain
The data consists of chromatography runs across CM and Q columns...

## Analysis Philosophy
Focus on run-to-run consistency and purity assessment...

    ```yaml
    pipeline:
      stages:
        - name: cleaning
          goals:
            - Remove fully-empty columns
            - Flag anomalies without aggressive removal
          quality:
            min_findings: 2

        - name: analysis
          goals:
            - Per-run chromatographic overlays
            - Outlier detection by key metric deviation
          quality:
            min_plots: 3
            min_findings: 3
            require_per_group: true
          expert_call_budget: 4

      constraints:
        preserve_columns: [run_no, run, Sample_Code]
        grouping_columns: [run_no, chromatography_stage]
    ```
```

The prose sections are injected into agent prompts as domain context. The YAML block configures stage behaviour, quality thresholds, and data constraints. See [Configuration](#configuration) for full details.

### Execution Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `single_pass` | One-shot execution, no validation loops | Fast ablation baseline |
| `stage_validation` | Gated review loop (structural + content + quality gate) with convergence-driven retry | Default for most runs |
| `full_closed_loop` | `stage_validation` + VLM plot quality review (basic or scientific) | Highest quality output |

Set via `--pipeline-mode` CLI flag or `PIPELINE_MODE` environment variable.

### Validation and Quality Assurance

Each stage attempt passes through a **deterministic gated review loop** with four phases:

| Phase | Type | What it checks |
|-------|------|----------------|
| 1. Structural gate | Pure Python | File existence, JSON validity, min plots/findings, PNG size, claim-figure references |
| 2. Content evaluator | LLM | Rubric-based scoring (6 criteria for analysis, 4 for cleaning, 3 for cross-validation) |
| 3. VLM plot review | VLM | Plot quality — basic (aesthetic) or scientific (chart type, grouping, annotations) |
| 4. Quality gate | Pure Python | Aggregates all checks into numeric quality score, applies convergence logic |

**Design principle**: Deterministic governance, non-deterministic analysis. Python code makes all gate/routing decisions; LLMs assess quality and produce structured signals.

Additional configurable thresholds:

| Check | Default | Configurable Via |
|-------|---------|-----------------|
| Minimum plots (analysis) | 3 | `quality.min_plots` in context.md |
| Minimum findings | 3 | `quality.min_findings` in context.md |
| Per-group structure required | true | `quality.require_per_group` in context.md |
| Expert call budget | Stage-dependent | `expert_call_budget` in context.md |
| Max retries per stage | 2 (cleaning), 1 (others) | `max_retries` in context.md |
| Iteration strategy | `"fixed"` | `iteration_strategy` in run_config |
| Convergence target | 0.85 | `convergence_target` in run_config |

**Quality scoring**: Each check contributes to a numeric score (passed=1.0, should_fix=0.5, must_fix=0.0). Content and plot checks receive 2x weight. The convergent iteration strategy stops when quality reaches the target or improvement plateaus.

**Disk-artifact recovery**: If a GroupChat hits its `max_round` limit and AG2 produces a narrative summary instead of structured JSON, the pipeline checks disk for files the experts may have already written (e.g., cleaned parquet, analysis_summary.json) and recovers from those.

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
git clone https://github.com/<your-username>/biologics-multi-agent-analysis.git
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
| `weasyprint>=60.0`, `reportlab>=4.0` | PDF generation |
| `openai>=1.40` | LLM API client |

---

## Usage

### Quick Start (SLURM)

The simplest way to run the pipeline on an HPC cluster:

```bash
# Place your data files in the Data/ directory
# Edit context.md to describe your project and constraints

# Submit the job
sbatch Batch_Script.sh
```

The batch script handles:
1. Building the metadata database from raw files
2. Starting the vLLM inference server
3. Running the captain pipeline (cleaning -> analysis -> cross-validation)
4. Running the report pipeline (literature research -> report generation)
5. Cleaning up the vLLM server on exit

### Manual Execution

For local development or non-SLURM environments:

```bash
# 1. Start vLLM server manually
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
| `METADATA_DB_DIR` | Metadata database directory | `Database/metadata` |
| `CONTEXT_PATH` | Path to context.md | Auto-discovered |
| `PIPELINE_MODE` | Execution mode | `stage_validation` |
| `VLLM_PID` | vLLM process ID (for managed lifecycle) | -- |
| `VLLM_PORT` | vLLM server port | -- |
| `PIPELINE_MIN_PLOTS_ANALYSIS` | Override min plots for analysis | 3 |
| `PIPELINE_CHAT_TIMEOUT_S` | Wall-clock timeout per chat session (seconds) | 600 |
| `PIPELINE_USE_TRANSFORM_MESSAGES` | Use native AG2 TransformMessages hook (A/B test) | `0` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

---

## Configuration

### The context.md File

`context.md` is the single configuration file that governs pipeline behaviour. It has two parts:

1. **Prose sections** (Markdown) -- injected into agent prompts as domain context. Describe your project, data, and analysis expectations in natural language. Agents use this to make informed decisions.

2. **YAML block** (fenced with ` ```yaml `) -- structured configuration for pipeline stages, quality thresholds, and data constraints.

If no `context.md` is provided, the pipeline uses sensible defaults for all settings.

### Pipeline Stages Configuration

```yaml
pipeline:
  domain: "chromatography"  # Optional: override auto-detection
                             # Values: chromatography | mass_spectrometry | both
  stages:
    - name: cleaning
      goals:                 # Natural language objectives for this stage
        - Remove fully-empty columns
        - Standardise column names
      quality:
        min_findings: 2      # Minimum findings to pass validation
      max_retries: 2         # Retry limit on validation failure

    - name: analysis
      goals:
        - Per-run chromatographic overlays
        - Outlier detection by metric deviation
      quality:
        min_plots: 3         # Minimum PNG plots to produce
        min_findings: 3      # Minimum analytical findings
        require_per_group: true  # Must produce per-group statistics
      expert_call_budget: 4  # Max seek_experts_help calls

    - name: cross_validation
      goals:
        - Verify top-3 statistical claims
      agent_hints:
        require: [cross_validator]  # Force specific agents

    - name: report
      goals:
        - Executive summary with key findings
```

### Data Constraints

```yaml
pipeline:
  constraints:
    preserve_columns:        # Never drop these columns
      - run_no
      - run
      - chromatography_stage
      - Sample_Code
      - column
      - Fraction_number
    no_aggregation_across:   # Don't merge data across these grouping levels
      - run_no
    grouping_columns:        # Primary dimensions for per-group analysis
      - run_no
      - chromatography_stage
      - Sample_Code
```

### Quality Thresholds

These control the validation loop. If a stage's output doesn't meet the threshold, it is retried (up to `max_retries`).

| Parameter | Stage | Default | Description |
|-----------|-------|---------|-------------|
| `min_plots` | analysis | 3 | Minimum number of PNG plots |
| `min_findings` | all | 2-3 | Minimum analytical findings |
| `require_per_group` | analysis | true | Must produce per-group structure |
| `max_retries` | all | 0-2 | Maximum retry attempts |
| `expert_call_budget` | all | 2-4 | Maximum `seek_experts_help` calls |

### Run Configuration (Feature Toggles)

The `run_config` block controls experimental features and quality enhancements that can be independently toggled on and off. This is the mechanism for running ablation studies — each configuration produces a labelled run that can be compared against others.

```yaml
pipeline:
  run_config:
    run_label: "all_WPs"              # Human-readable name for this run
    prompt_version: "v2"              # "v1" = baseline prompts, "v2" = enhanced
    content_validation: true          # Validate JSON content, not just file existence
    min_cross_val_claims: 5           # Minimum claims to verify
    recompute_cross_val: true         # Actually recompute from parquet
    figure_selection: "ranked"        # How to select report figures
    max_report_figures: 10            # Cap on figures in reports
    rerun_issue_map_enabled: true     # Targeted retry instructions
    protected_col_audit: true         # Warn on near-empty protected columns
    # WP-1: Schema Intelligence
    schema_profiling: "roles_only"    # "disabled"|"roles_only"|"full"
    # WP-2: Adaptive Convergence
    iteration_strategy: "convergent"  # "none"|"fixed"|"convergent"
    convergence_threshold: 0.05       # Min improvement to continue iterating
    convergence_target: 0.85          # Quality score to stop early
    # WP-3: Scientific Visual Review
    visual_review_mode: "scientific"  # "basic"|"scientific"
    require_figure_references: true   # Enforce claim-figure mapping
    # WP-4: Pluggable Agent Library
    expert_library: "extended"        # "baseline"|"extended"
    agent_definitions_dir: "agents/"  # Path to agent YAML files
```

Every run_config is:
- **Logged at pipeline start** — a formatted block in the logs shows exactly what's enabled
- **Written into the manifest JSON** — every output folder is self-documenting under the `"run_config"` key
- **Displayed by `compare_runs.py`** — side-by-side comparison shows which features differed

To revert to baseline behaviour, simply remove the `run_config` block or set `run_label: "baseline"` with no other fields (all defaults are safe baseline values).

See [Ablation Studies & Run Comparison](#ablation-studies--run-comparison) for full details on each toggle and how to design comparison experiments.

### Agent Selection Hints

Control which agents are included in each stage's expert team:

```yaml
stages:
  - name: analysis
    agent_hints:
      require: [chromatography_expert, statistical_analyst]  # Must include
      prefer: [ml_modeler]                                   # Include if available
      exclude: [mass_spec_expert]                            # Never include
      max_agents: 4                                          # Team size limit
```

---

## Ablation Studies & Run Comparison

The pipeline includes a built-in system for running controlled experiments. You can toggle individual improvements on or off via `context.md`, run the pipeline, and then compare results across runs using `compare_runs.py`. Every run self-documents its configuration in the manifest, so you always know exactly what was enabled.

### How Ablation Works

```
1. Edit context.md         →  Set run_config toggles
2. Run the pipeline        →  sbatch Batch_Script.sh  (or manual)
3. Output folder created   →  Outputs/slurm_XXXX/manifest_*.json contains run_config
4. Repeat with different   →  Change run_config, run again
   toggles
5. Compare results         →  python Scripts/compare_runs.py --glob "Outputs/slurm_*"
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
    schema_profiling: "roles_only"
    iteration_strategy: "convergent"
    visual_review_mode: "scientific"
    require_figure_references: true
    expert_library: "extended"
```

**Test a single feature** (e.g., just enhanced prompts):
```yaml
pipeline:
  run_config:
    run_label: "v2_prompts_only"
    prompt_version: "v2"
    # Everything else stays at defaults
```

### Feature Toggle Reference

Each toggle controls a specific improvement. Here's what they do and when to use them:

#### `run_label` (string, default: `""`)
A human-readable name for this run. Appears in logs, the manifest, and `compare_runs.py` output. Use descriptive names like `"baseline"`, `"WP1-3_gates"`, `"all_WPs"`.

#### `prompt_version` (string, default: `"v1"`)
Controls which prompt enhancements are active.

| Value | Behaviour |
|-------|-----------|
| `"v1"` | **Baseline prompts.** Findings are bare numeric deviations (e.g., "359% deviation"). P-values may be blanket "p<0.05". |
| `"v2"` | **Enhanced prompts.** Each finding must include biological interpretation and a possible root cause. P-values must be actual computed values. The content evaluator will flag bare deviations as "insufficient". |

**What changes with v2:**
- Expert prompts (chromatography, mass spec, statistical) get an `INTERPRETATION REQUIREMENT` block with biologics reference ranges (UV280 CV >15% = loading inconsistency, mass accuracy >50 ppm = PTM/calibration drift, etc.)
- Expert prompts get a `STATISTICAL RIGOR REQUIREMENT` block requiring actual p-values from `scipy.stats.f_oneway` or `ttest_ind`
- The `ANALYSIS_RUBRIC` (used by the content evaluator) gains two new criteria: interpretation depth and statistical rigor

**Example impact:**
- v1 finding: `"Run R17 shows 359.4% deviation vs group mean of 37.06 in metric UV_1_280_mAU (p<0.05)"`
- v2 finding: `"Run R17 shows 359.4% deviation vs group mean of 37.06 mAU (p=0.0000), suggesting potential protein aggregation or column overloading — verify column binding capacity and check sample concentration"`

#### `content_validation` (bool, default: `true`)
Controls whether the pipeline checks the *content* of artifact files, not just their existence.

| Value | Behaviour |
|-------|-----------|
| `true` | **Active.** After confirming files exist on disk, parses JSON and checks required keys are non-empty (e.g., `analysis_summary.json` must have non-empty `findings[]` and `per_group`/`per_run_per_stage`). |
| `false` | Files are accepted if they exist and are >100 bytes. A malformed JSON with empty findings would pass. |

#### `min_cross_val_claims` (int, default: `5`)
Minimum number of claims the deterministic cross-validation fallback should verify. Increasing this gives broader verification coverage.

#### `recompute_cross_val` (bool, default: `false`)
Controls whether cross-validation actually recomputes metrics from the parquet file.

| Value | Behaviour |
|-------|-----------|
| `false` | **Baseline (tautological).** The "verified" claims compare a value from `analysis_summary.json` against itself. Every claim always matches. This confirms the JSON is self-consistent but doesn't verify computational correctness. |
| `true` | **Genuine verification.** Loads the cleaned parquet, groups by the appropriate columns, recomputes means, and compares against the claimed values with 1% tolerance. Claims that don't match are flagged with `"match": false`. Each claim includes a `"recomputed": true/false` field. |

**What to look for:** With `recompute_cross_val: true`, check the manifest's `cross_validation.verified_claims` — any `"match": false` entries indicate the analysis stage computed something incorrectly.

#### `figure_selection` (string, default: `"all"`)
Controls how figures are selected for inclusion in reports.

| Value | Behaviour |
|-------|-----------|
| `"all"` | **Baseline.** Every plot is included in the report's Figures section. Can result in 30+ figures. |
| `"ranked"` | Figures are ranked by file size (larger = more content), deduplicated by stem name, filtered to remove tiny (<5KB) images, and capped at `max_report_figures`. Produces focused, high-quality figure sections. |
| `"top_n"` | Takes the first `max_report_figures` figures in directory order. Simple but not quality-filtered. |

#### `max_report_figures` (int, default: `10`)
Maximum number of figures when using `"ranked"` or `"top_n"` selection. Ignored when `figure_selection` is `"all"`.

#### `rerun_issue_map_enabled` (bool, default: `false`)
Controls whether the retry loop injects targeted corrective instructions.

| Value | Behaviour |
|-------|-----------|
| `false` | **Baseline.** Retries use only the generic validator feedback ("Missing files: ..."). |
| `true` | The pipeline checks validator issues against a map of known problems and injects specific corrective instructions (e.g., "Your `per_run_per_stage` key was missing — add it with one dict entry per run×stage combination"). |

**When to enable:** If you see retries that fail repeatedly on the same structural issue, this can help the LLM fix the specific problem rather than guessing.

#### `protected_col_audit` (bool, default: `false`)
Controls whether the pipeline audits protected columns for near-total missingness after cleaning.

| Value | Behaviour |
|-------|-----------|
| `false` | **Baseline.** Protected columns are preserved regardless of missingness. No warnings. |
| `true` | After cleaning, checks each protected column. If any has >95% missing values, a warning is logged and written to `cleaning_summary.json` under `"protected_column_warnings"`. |

**Why this matters:** The baseline run showed `Observed_m/z` (a protected column) at 99.99% missing. With this toggle on, that gets flagged explicitly so you know the column is preserved but essentially empty.

#### `schema_profiling` (string, default: `"disabled"`)
Controls the WP-1 schema intelligence layer that classifies columns and infers grouping structures.

| Value | Behaviour |
|-------|-----------|
| `"disabled"` | **Baseline.** Uses `inspect_table()` + hardcoded `KNOWN_GROUP_COLUMNS` for grouping. |
| `"roles_only"` | Runs column role classification (identifier, categorical_group, ordinal_stage, continuous_measurement, etc.) and infers multi-column grouping candidates. Replaces hardcoded grouping with data-driven inference. |
| `"full"` | Adds distribution profiling (normality tests, modality) and technique recommendations on top of `roles_only`. Increases payload size. |

**Impact:** Addresses the most fundamental limitation — the system's understanding of data structure. Multi-column grouping prevents misleading overlays of heterogeneous data. Column role classification prevents treating identifiers as measurements.

#### `iteration_strategy` (string, default: `"fixed"`)
Controls WP-2 convergence behaviour for the gated review retry loop.

| Value | Behaviour |
|-------|-----------|
| `"none"` | **Single pass.** No retries regardless of quality. Useful as ablation baseline to measure raw agent quality. |
| `"fixed"` | **Baseline.** Retries up to `max_retries` regardless of quality trajectory. |
| `"convergent"` | **Quality-driven.** Stops if quality_score >= `convergence_target` OR improvement < `convergence_threshold`. Continues only when quality is below target AND still improving. |

#### `convergence_threshold` (float, default: `0.05`)
Minimum quality score improvement between attempts to justify continuing iteration. Only applies when `iteration_strategy: "convergent"`.

#### `convergence_target` (float, default: `0.85`)
Quality score at which to stop iterating early. Only applies when `iteration_strategy: "convergent"`.

#### `visual_review_mode` (string, default: `"basic"`)
Controls WP-3 VLM plot quality assessment mode.

| Value | Behaviour |
|-------|-----------|
| `"basic"` | **Baseline.** VLM evaluates readability, data quality, information value, overcrowding. Returns overall score (good/acceptable/poor). |
| `"scientific"` | **Enhanced.** VLM evaluates 7 criteria across 3 severity classes. Scientific validity issues (wrong chart type, misleading axes, incorrect grouping) → MUST_FIX. Statistical completeness issues (missing annotations) → SHOULD_FIX. Cosmetic issues → informational only (no retry triggered). |

**Impact:** Prevents weak plots from passing review due to aesthetic-only checking. Graduated severity avoids wasting retries on cosmetic issues.

#### `require_figure_references` (bool, default: `false`)
Controls WP-3A claim-figure traceability.

| Value | Behaviour |
|-------|-----------|
| `false` | **Baseline.** Findings in `analysis_summary.json` can be plain strings or dicts. No figure reference validation. |
| `true` | Each finding must be a dict with a `"figure"` key pointing to an existing PNG filename. Orphaned references are flagged as SHOULD_FIX. |

#### `expert_library` (string, default: `"baseline"`)
Controls WP-4 agent library mode.

| Value | Behaviour |
|-------|-----------|
| `"baseline"` | **Baseline.** Uses the 8 core agents with keyword-based domain selection. Agent definitions are inline in Python. |
| `"extended"` | Loads agent definitions from YAML files in `agent_definitions_dir`. Includes 3 additional specialist agents (bioprocess_analyst, visualisation_specialist, statistical_modeler). Uses schema-driven activation scoring to select agents based on data profile match. |

#### `agent_definitions_dir` (string, default: `"agents/"`)
Path to the directory containing agent YAML definition files. Relative paths are resolved from the project root. Only used when `expert_library` is not `"baseline"`.

### Pre-Defined Run Configurations

Here are ready-to-use configurations for common ablation experiments:

| Run Label | What It Tests | Key Config |
|-----------|---------------|------------|
| `baseline` | Pre-improvement behaviour | `run_label: "baseline"` (all defaults) |
| `schema_only` | Schema intelligence (WP-1) | `schema_profiling: "roles_only"` |
| `convergent` | Quality-driven iteration (WP-2) | `iteration_strategy: "convergent"` |
| `scientific_review` | Scientific VLM review (WP-3) | `visual_review_mode: "scientific"`, `require_figure_references: true` |
| `extended_agents` | Pluggable agent library (WP-4) | `expert_library: "extended"` |
| `v2_prompts` | Enhanced prompts only | `prompt_version: "v2"` |
| `crossval` | Genuine cross-validation | `recompute_cross_val: true` |
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

**Where to find scoring inputs:**
- `manifest_batch_*.json` → `run_config`, `files[].stages_completed`, `files[].verified_claims_count`
- `files/{dataset}/analysis/analysis_summary.json` → `findings[]`, `per_run_per_stage`
- `files/{dataset}/cross_validation/` → `verified_claims` with `match` and `recomputed` fields
- `reports/{dataset}__report.md` → Full narrative report text
- `debug/*__content_eval.json` → Content evaluator rubric scores and coverage ratio
- `debug/*__plot_eval.json` → VLM plot quality check results
- `debug/*__quality_gate.json` → Quality gate verdict with numeric score

---

## Evaluation Framework

The project includes a dedicated evaluation framework (`Scripts/evaluation/`) for automated, research-grade ablation studies. It acts as a **fully external wrapper** around the pipeline — it reads only from `Outputs/` and never modifies pipeline behaviour.

The framework uses **LLM-as-judge** evaluation via [OpenRouter](https://openrouter.ai/), where configurable external models (e.g., GPT-4o, Claude Sonnet) score pipeline reports against a structured rubric. It supports replicated runs, pairwise head-to-head comparisons, statistical significance testing, and automated generation of dissertation-ready figures and LaTeX tables.

### Evaluation Overview

```
Pipeline Runs (existing)          Evaluation Framework
─────────────────────────         ─────────────────────────
Outputs/                    ───►  Scripts/evaluation/
  slurm_*/                          ├── Discover & index runs
    judge_input.json                ├── LLM-as-judge scoring (OpenRouter)
    manifest_batch_*.json           ├── Pairwise comparison
    files/*/analysis/               ├── Statistical analysis
    reports/*.md                    └── Figures + LaTeX export
                                        ↓
                                  Evaluation/
                                    evaluation.db, figures/, exports/
```

**Design principles:**
- **Fully decoupled** — zero imports from the pipeline code; reads only JSON/Markdown/PNG files
- **WP-agnostic** — groups runs by `run_label` string, not by specific Work Package. Works for any current or future configuration
- **Deterministic governance, non-deterministic analysis** — Python code makes all gate decisions (statistical significance, ranking computation); LLMs handle qualitative evaluation (rubric scoring, pairwise preference)

### Evaluation Setup

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
       supports_vision: true      # Send plot PNGs for visual evaluation

     - model_id: "openrouter/healer-alpha"
       display_name: "healer-alpha"
       temperature: 0.0
       max_tokens: 32768
       weight: 1.0
       supports_vision: true
   ```

   All judge models are fully configurable — there are no hardcoded defaults. You must specify at least one model. Any model available on [OpenRouter](https://openrouter.ai/models) can be used (e.g., `openai/gpt-4o`, `anthropic/claude-sonnet-4`, `openrouter/hunter-alpha`). Models with `supports_vision: true` receive plot PNGs as base64-encoded images for visual figure quality evaluation.

3. **No pipeline modifications required.** The evaluation framework is a separate package with no dependencies on the pipeline code.

### Running Evaluations

**Evaluate all runs:**
```bash
python -m Scripts.evaluation evaluate --glob "Outputs/slurm_*"
```

**Run the full pipeline** (evaluate + pairwise comparisons + ablation analysis + export):
```bash
python -m Scripts.evaluation full \
    --glob "Outputs/slurm_*" \
    --baseline "baseline-v2"
```

**Skip already-evaluated runs** (useful when adding new runs incrementally):
```bash
python -m Scripts.evaluation evaluate \
    --glob "Outputs/slurm_*" --skip-existing
```

### Replicated Runs

Because the pipeline produces **non-deterministic outputs**, the evaluation framework supports multiple runs per configuration for statistical robustness.

**Submit replicate runs using the launcher script:**
```bash
# Submit 5 replicate runs with the current context.md
bash Scripts/evaluation/launch_replicates.sh --replicates 5

# Submit 10 replicates with a specific context file
bash Scripts/evaluation/launch_replicates.sh --replicates 10 --context context_wp1.md

# Preview what would be submitted (no actual submission)
bash Scripts/evaluation/launch_replicates.sh --replicates 5 --dry-run
```

The launcher submits N copies of `Batch_Script.sh` via `sbatch`. Each job gets a unique output directory (via SLURM job ID), but uses the same `run_label` in `context.md`. The evaluation framework automatically groups runs with matching `run_label` as replicates via `group_by_config()`.

**Recommended:** 5–10 replicates per configuration for reliable statistical comparisons.

### Evaluation Rubric

Reports are scored against a structured rubric with **7 criteria** on a 0–10 scale:

| Criterion | Weight | What It Measures |
|-----------|--------|------------------|
| `analytical_depth` | 0.20 | Per-group breakdown, multi-dimensional analysis, pattern identification |
| `statistical_reasoning` | 0.15 | Correct test selection, effect sizes, confidence intervals |
| `figure_quality` | 0.10 | Diversity, labelling, relevance, publication readiness |
| `claim_evidence_consistency` | 0.20 | Every claim traceable to computed values; cross-validation alignment |
| `report_clarity` | 0.10 | IMRAD structure, scientific prose quality, limitations section |
| `evidence_based_reasoning` | 0.15 | Regulatory citations (ICH, FDA), domain knowledge integration |
| `domain_correctness` | 0.10 | Biologics/chromatography/MS scientific accuracy |

Each criterion has **5 anchor points** (0, 3, 5, 7, 10) with concrete descriptions to calibrate judges. Score 5 = adequate, 7 = good, 9+ = exceptional (rare).

The **weighted overall score** is computed as `sum(weight_i × score_i)` on a 0–10 scale.

**Noise reduction strategies:**
- Temperature = 0 for all judge calls
- Multi-model judging (weighted mean across models)
- Structured anchors reduce calibration drift across models
- Optional multi-pass averaging (`num_judge_passes` in config)
- Median aggregation option for robustness to outlier judges

### Pairwise Comparison

In addition to rubric scoring, the framework supports **head-to-head comparison** between pairs of reports:

```bash
python -m Scripts.evaluation pairwise --glob "Outputs/slurm_*"
```

For each pair, the judge receives both reports and answers: *"Which analysis is stronger and why?"* — producing per-criterion preferences and an overall winner with confidence.

**Position-bias mitigation:** Report presentation order (A/B vs B/A) is randomised. With multiple judge models, each pair receives 4+ independent evaluations.

**Ranking from pairwise results:**
- **Bradley-Terry MLE** — log-linear strength model via iterative maximum likelihood
- **Elo ratings** — simpler alternative (K=32)

Both produce a ranked ordering of configurations by strength:
```bash
python -m Scripts.evaluation rankings
```

### Statistical Analysis

For replicated runs (N ≥ 2 per configuration), the framework computes:

| Method | Purpose |
|--------|---------|
| **Bootstrap CI** | 95% confidence intervals (10K resamples) |
| **Mann-Whitney U** | Non-parametric significance test (appropriate for small N) |
| **Permutation test** | Two-sided, 10K permutations |
| **Cohen's d** | Effect size with Hedges' g correction for small samples |
| **Holm-Bonferroni** | Multiple comparison correction across all pairs |

Effect size interpretation: |d| < 0.2 negligible, 0.2–0.5 small, 0.5–0.8 medium, > 0.8 large.

### Outputs and Exports

**Ablation analysis** (requires a baseline label):
```bash
python -m Scripts.evaluation ablation \
    --baseline "baseline-v2" --glob "Outputs/slurm_*"
```

This generates:

| Output | Location | Description |
|--------|----------|-------------|
| **SQLite database** | `Evaluation/evaluation.db` | All scores, judgments, comparisons, rankings |
| **Ablation report** | `Evaluation/reports/ablation_report.md` | Markdown summary with tables and figure references |
| **Radar chart** | `Evaluation/figures/radar_chart.png` | Rubric criterion scores per configuration |
| **Score heatmap** | `Evaluation/figures/score_heatmap.png` | Configuration × Criterion matrix |
| **Bar chart** | `Evaluation/figures/overall_bar_chart.png` | Mean ± 95% CI per configuration |
| **Violin plots** | `Evaluation/figures/overall_distributions.png` | Score distributions (essential for N=5–10) |
| **Forest plot** | `Evaluation/figures/effect_size_forest.png` | Effect sizes with significance |
| **Win matrix** | `Evaluation/figures/pairwise_win_matrix.png` | NxN pairwise win rates |
| **LaTeX results table** | `Evaluation/exports/results_table.tex` | Mean ± std, booktabs format |
| **LaTeX significance table** | `Evaluation/exports/significance_table.tex` | p-values with stars, effect sizes |
| **Raw judgments** | `Evaluation/raw_judgments/*.json` | Full judge responses for provenance |

**Export formats:**
```bash
# LaTeX tables (default)
python -m Scripts.evaluation export --format latex

# CSV
python -m Scripts.evaluation export --format csv

# JSON
python -m Scripts.evaluation export --format json
```

### Evaluation CLI Reference

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

**Example workflow:**
```bash
# 1. Submit 5 baseline replicates
bash Scripts/evaluation/launch_replicates.sh --replicates 5

# 2. Edit context.md to enable WP1, change run_label to "wp1"
# 3. Submit 5 WP1 replicates
bash Scripts/evaluation/launch_replicates.sh --replicates 5

# 4. Run full evaluation
python -m Scripts.evaluation full \
    --glob "Outputs/slurm_*" --baseline "baseline"

# 5. Results in Evaluation/ — figures, LaTeX tables, SQLite DB
```

---

## Output Structure

### Directory Layout

Each pipeline run produces a timestamped output directory:

```
Outputs/slurm_859801_20260302_222508/
├── manifest_batch_859801.json          # Complete run manifest
├── files/
│   ├── chromatography_combined/
│   │   ├── cleaning/
│   │   │   ├── chromatography_combined__cleaned.parquet
│   │   │   ├── cleaning_summary.json
│   │   │   └── artifacts/
│   │   ├── analysis/
│   │   │   ├── analysis_summary.json
│   │   │   ├── artifacts/
│   │   │   │   ├── uv_overlay_by_run.png
│   │   │   │   ├── peak_metrics_heatmap.png
│   │   │   │   └── ...
│   │   │   └── plots/
│   │   └── cross_validation/
│   │       └── cross_validation.json
│   └── ms_combined/
│       └── ...  (same structure)
├── reports/
│   ├── chromatography_combined/
│   │   ├── report.md
│   │   ├── report.html
│   │   └── report.pdf
│   └── ms_combined/
│       └── ...
└── debug/
    ├── chromatography_combined__cleaning__attempt1__request.json
    ├── chromatography_combined__cleaning__attempt1__response.json
    └── ...
```

### Manifest File

The manifest is the complete record of a pipeline run:

```json
{
  "batch_id": "batch_20260304_111147",
  "pipeline_mode": "full_closed_loop",
  "model": "Qwen/Qwen3.5-27B",
  "timestamp": "2026-03-04T11:47:23",
  "run_config": {
    "run_label": "all_WPs",
    "prompt_version": "v2",
    "content_validation": true,
    "recompute_cross_val": true,
    "figure_selection": "ranked",
    "rerun_issue_map_enabled": true,
    "protected_col_audit": true,
    "schema_profiling": "roles_only",
    "iteration_strategy": "convergent",
    "visual_review_mode": "scientific",
    "expert_library": "extended"
  },
  "file_count": 2,
  "files": [
    {
      "name": "chromatography_combined.parquet",
      "stages_completed": {
        "cleaning": true, "analysis": true,
        "cross_validation": true, "report": true
      },
      "plot_count": 20,
      "verified_claims_count": 5,
      "cv_gaps": []
    }
  ],
  "quality_review_overall": "PASS"
}
```

### Debug Artifacts

Every captain-level interaction is logged to `debug/`:

- `*__request.json` -- the payload sent to CaptainAgent (truncated to 14K tokens)
- `*__response.json` -- the structured JSON returned
- `*__full_response.txt` -- full reply text (if > 5KB)
- `*__structural_gate.json` -- structural gate check results (file existence, JSON validity, plot counts)
- `*__content_eval.json` -- content evaluator rubric scores and coverage ratio
- `*__plot_eval.json` -- VLM plot quality evaluator results (basic or scientific mode)
- `*__quality_gate.json` -- final quality gate verdict with numeric quality score

---

## Adapting to New Data

The system is designed to handle any biologics dataset without code changes. To analyse a new dataset:

1. **Place your data** in a directory (CSV, Excel, or Parquet files)

2. **Write a `context.md`** describing:
   - What the data represents (project overview in prose)
   - Which columns are important (preserve_columns)
   - How data should be grouped (grouping_columns)
   - What quality you expect (min_plots, min_findings)
   - Optionally, which domain to force (chromatography, mass_spectrometry, or both)

3. **Run the pipeline** pointing at your data directory

The system will:
- Auto-detect whether it's chromatography, mass spectrometry, or mixed data
- With `schema_profiling` enabled: classify columns by role, infer multi-column grouping structures, and recommend appropriate analytical techniques — all without domain-specific keywords
- Select appropriate expert agents — via keyword detection (baseline) or schema-driven activation scoring (extended)
- Infer which analyses are meaningful based on the columns present
- Respect your constraints (protected columns, grouping rules)
- Produce a complete analysis with validated findings and a scientific report

**No code changes needed** -- the context.md file is the only thing that changes between projects.

### Extending with New Agent Types

To add a new specialist agent (e.g., for a new data domain):

1. **Add a system prompt** to `Scripts/prompts.py` with a descriptive constant name (e.g., `MY_EXPERT_PROMPT`)
2. **Create a YAML definition** in `agents/` (see existing files for the format):
   ```yaml
   name: my_expert
   description: "One-line description for AutoBuild agent selection."
   system_prompt_key: MY_EXPERT_PROMPT
   stages: [analysis]
   prompt_modifiers:
     grouping_instructions: true   # inject grouping context into {grouping_instructions}
     v2_suffix: true               # append interpretation + p-value blocks (v2 prompts)
   activation:
     domain_keywords: [keyword1, keyword2]  # match against column names
     column_roles: [continuous_measurement]  # match against WP-1 profile roles
     min_numeric_columns: 3                 # minimum numeric columns required
   extended: true  # only loaded when expert_library: "extended"
   ```
3. **Register the prompt** in the `_PROMPT_REGISTRY` dict in `Scripts/captain_pipeline.py`
4. **Add the agent name** to `KNOWN_AGENT_NAMES` in `Scripts/context_parser.py` (for validation)
5. Optionally reference the agent in your `context.md` via `agent_hints.require` to force its inclusion

With `expert_library: "extended"`, the new agent will be automatically activated when its criteria match the data profile. No other source code changes are needed.

---

## Design Philosophy

### Core Principles

1. **Reason from what you see, not what you expect** -- agents inspect data before deciding on analysis. No hardcoded column assumptions.

2. **Conservative cleaning** -- preserve data wherever possible. Flag anomalies rather than deleting them. Only remove columns that are entirely empty.

3. **Two-pass analysis** -- first an analysis planner recommends strategies, then domain experts execute code. This separates planning from implementation.

4. **Per-group granularity** -- results are computed per run, per stage, per sample. Cross-run aggregation is avoided unless explicitly requested.

5. **Validate everything** -- cross-validation re-computes statistical claims from raw data. Findings that can't be reproduced are flagged.

6. **Fail gracefully** -- expert call budgets prevent infinite loops, disk-artifact recovery handles truncated chats, and the pipeline continues through partial failures.

7. **Deterministic governance, non-deterministic analysis** -- Python code makes all gate/routing decisions; LLMs assess quality and produce structured signals. Toggle-gated features ensure the baseline is never broken by new capabilities.

### Why AG2 / CaptainAgent?

- **Dynamic team assembly**: CaptainAgent selects the right experts per task rather than using a fixed agent pipeline
- **Code execution**: Agents write and execute Python code in sandboxed environments
- **Structured tool calling**: `seek_experts_help` provides a clean interface for task delegation
- **GroupChat**: Expert teams collaborate naturally, building on each other's code and findings

### Why Local LLM (vLLM)?

- **Data privacy**: Biologics data stays on-premise; no external API calls for inference
- **Cost**: No per-token charges for analysis of million-row datasets
- **Latency**: Local GPU inference with KV cache prefix sharing
- **Reproducibility**: Fixed model version per experiment

---

## Recent Improvements

### Qwen3.5-27B Model Upgrade

The pipeline migrated from Qwen2.5 to **Qwen3.5-27B**, which provides improved reasoning and code generation. This required several compatibility patches:

- **System message reordering**: Qwen3.5 requires exactly one system message at the start. A monkey-patch on `_reflection_with_llm` prepends (rather than appends) system messages, with an A/B test option for AG2's native `TransformMessages` hook.
- **User message injection**: AG2 GroupChat can produce message lists without user messages (e.g., during speaker selection). The `_patched_oai_create` hook detects this and injects a minimal user message to prevent 400 errors.
- **Think-token stripping**: Qwen3.5's `<think>` tokens are stripped at three layers (OpenAIWrapper.create, ConversableAgent.receive, ConversableAgent.send) to prevent reasoning traces from leaking into agent messages and log files.

### Two-Pass Analysis Strategy

Analysis now runs in two sequential passes for improved quality:

1. **Pass 1 (Strategy)**: An `analysis_planner` agent inspects the data evidence and recommends 5-8 diverse analytical approaches and plot types, specifying which columns each requires.
2. **Pass 2 (Execution)**: Domain expert(s) receive the planner's strategy and execute the actual analysis, generating plots and `analysis_summary.json`.

This separation ensures analytical diversity (at least 3-4 different chart types) and prevents experts from defaulting to a narrow set of familiar analyses.

### Context Parser and RunPlan

A new `context_parser.py` module parses `context.md` (Markdown with embedded YAML blocks) into a typed `RunPlan` dataclass hierarchy:

- **`RunPlan`**: Top-level execution plan with project description, stage list, global constraints, and domain override.
- **`StageSpec`**: Per-stage configuration including goals, quality thresholds, agent hints, max retries, and expert call budgets.
- **`QualitySpec`**: Validation thresholds (min_plots, min_findings, require_per_group, png_min_bytes, summary_size_bounds).
- **`ConstraintSpec`**: Data constraints (preserve_columns, grouping_columns, no_aggregation_across).

The parser validates YAML against a known schema, warns about unknown keys or agent names, and falls back gracefully for plain-text context files.

### Gated Review Loop (Deterministic Validation)

The earlier fire-and-forget critic was replaced with a 4-phase deterministic review loop governed by Python gates. See [Validation and Quality Assurance](#validation-and-quality-assurance) for the full phase breakdown. The loop incorporates content evaluator hardening (WP-2A), numeric quality scoring (WP-2B), convergence control (WP-2C), and scientific severity-based plot review (WP-3B/C).

### Dynamic Code Guidance and Grouping

- **Code Guidance Template**: A shared set of 9 code execution rules is appended to all code-generating expert prompts, with runtime-injected library versions (e.g., `pandas 2.2.3, scipy 1.17.0`). Rules cover standalone scripts, imports, column safety, JSON serialization, NaN handling, and concise code blocks.
- **Dynamic Grouping Instructions**: `build_grouping_instructions()` generates context-driven grouping blocks from `context.md` constraints, replacing hardcoded grouping rules in expert prompts.

### Robustness Improvements

- **Truncated code block repair**: If an LLM hits max_tokens mid-code-block, the pipeline detects the unclosed fence and appends a closing ``` so AG2's code extractor can still find and execute the partial code.
- **Truncation loop detection**: If `Computer_terminal` reports "no code found" 2+ times consecutively, the GroupChat terminates early instead of looping to max_round.
- **Code block dedenting**: LLMs sometimes generate uniformly-indented code (as if inside a function). `execute_code_blocks` now auto-dedents all code via `textwrap.dedent`.
- **Wall-clock timeout**: A `_ChatTimeout` context manager using SIGALRM prevents any single `initiate_chat` call from running indefinitely (default: 600s, configurable via `PIPELINE_CHAT_TIMEOUT_S`).
- **Transient vLLM error retry**: The OpenAIWrapper patch retries on 503/429/502 errors with exponential backoff (5s, 10s, 20s).
- **GroupChat underpopulation handling**: If AutoBuild assembles too few agents, the pipeline returns a graceful JSON error instead of crashing.

### Mandatory Findings Format and Self-Validation

Expert prompts now enforce structured findings with explicit examples:
- Each finding must be a string (not a dict) with a run reference, numeric value, and comparison.
- Required format: `"Run R{X} shows {N}% {direction} vs group mean of {M} {units} in metric {K}"`.
- Code blocks include inline structural assertions that verify `per_group` exists, findings count >= 3, and all findings are strings -- catching issues before the validator runs.

### WP-1: Schema Intelligence Layer

A deterministic data profiler (`Scripts/schema_profiler.py`) that classifies columns and infers grouping structures, replacing the hardcoded `KNOWN_GROUP_COLUMNS` list and keyword-based domain detection:

- **Column role classification**: identifier, categorical_group, ordinal_stage, continuous_measurement, metadata_text, datetime, constant. Uses dtype, cardinality ratios, and naming patterns — not domain-specific keywords.
- **Grouping inference**: Enumerates 1/2/3-column combinations of categorical + ordinal columns. Scores by group count (prefer 5-50), evenness (low CV of group sizes), and minimum group size (≥5 rows).
- **Profile injection**: Structured profile replaces flat evidence dict in analysis payloads. Tiered — Phase B content (distributions, technique recommendations) is trimmed first when payload exceeds budget.

Toggle: `schema_profiling: "disabled" | "roles_only" | "full"`

### WP-2: Adaptive Convergence Control

Quality-driven iteration replacing fixed retry counts:

- **Content evaluator hardening** (WP-2A): Validates LLM response covers all expected rubric criteria. Retries once if incomplete. Fills missing criteria as "not_evaluated". Logs `coverage_ratio` for monitoring.
- **Numeric quality scoring** (WP-2B): Each CheckResult maps to passed=1.0, should_fix=0.5, must_fix=0.0. Content and plot checks get 2x weight. Stage quality score is the weighted mean.
- **Convergence strategy** (WP-2C): Three modes — `none` (single pass, ablation baseline), `fixed` (baseline retries), `convergent` (stop when quality >= target or improvement < threshold). Quality trajectory tracked across attempts.

Toggle: `iteration_strategy: "none" | "fixed" | "convergent"`

### WP-3: Evidence-Grounded Scientific Review

Transform VLM plot review from aesthetic checking to scientific quality assessment:

- **Claim-figure traceability** (WP-3A): When enabled, findings must reference existing PNG files. Structural gate validates references exist on disk.
- **Scientific plot rubric** (WP-3B): 7 criteria across 3 severity classes — scientific_validity (chart type, axis scaling, grouping), statistical_completeness (annotations, data sufficiency), cosmetic (readability, overcrowding).
- **Graduated severity** (WP-3C): Only scientific_validity failures trigger MUST_FIX retries. Statistical_completeness → SHOULD_FIX. Cosmetic → informational (no retry). Prevents wasting cycles on font size issues.

Toggle: `visual_review_mode: "basic" | "scientific"`, `require_figure_references: true | false`

### WP-4: Pluggable Domain Expert Library

File-driven agent plug-in system replacing inline Python definitions:

- **YAML agent definitions** (WP-4A): Each agent defined in `agents/*.yaml` with name, prompt key, stages, activation criteria, and prompt modifiers. Pipeline scans directory at startup.
- **Schema-driven activation** (WP-4B): Agents scored against current data using column role matching (from WP-1 profile), keyword matching, numeric column counts, and row thresholds. Highest-scoring agents selected. Context.md `agent_hints` override still takes precedence.
- **New domain experts** (WP-4C): bioprocess_analyst (upstream/downstream process data), visualisation_specialist (plot design guidance), statistical_modeler (mixed-effects, DoE, multivariate). Each activated by data profile match.

Toggle: `expert_library: "baseline" | "extended"`

---

## Troubleshooting

| Issue | Likely Cause | Solution |
|-------|-------------|----------|
| `'dict' object has no attribute 'config_list'` | AG2 >= 0.11 LLMConfig API change | Ensure `LLMConfig(*config_entries, **kwargs)` (positional args, not `config_list=`) |
| `System message must be at the beginning` | Qwen3.5 chat template requirement | The monkey-patch in `captain_pipeline.py` handles this automatically |
| `No user query found in messages` | Qwen3.5 requires at least one user message | The `_patched_oai_create` monkey-patch injects a minimal user message |
| `TypeError: Object of type bool_ is not JSON serializable` | numpy types in `json.dump()` | Use `json.dump(..., default=str)` in all serialization calls |
| Expert teams loop indefinitely | Budget exhaustion or missing termination | Check `expert_call_budget` in context.md; budgets enforce limits |
| vLLM server not starting | CUDA version mismatch or OOM | Check GPU memory (27B model needs ~55GB); verify CUDA 12.x |
| Empty analysis output | Insufficient `max_round` for GroupChat | Increase `_STAGE_MAX_ROUNDS` in `captain_pipeline.py` |
| Report pipeline hangs | DeepResearchAgent decomposition loop | The monkey-patch in `report_pipeline.py` limits to 10 turns |
| Pipeline timeout on SLURM | Run exceeded wall time | Increase `-t` in `Batch_Script.sh` or use `single_pass` mode |

**Debug logs** are saved to `Outputs/{RUN_ID}/debug/` and contain the full request/response payloads for every captain-level interaction. SLURM stderr/stdout files (`Harry_Turner_Project_{JOB_ID}.err/.out`) contain the complete execution trace.
