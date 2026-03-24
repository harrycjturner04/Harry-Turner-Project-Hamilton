"""Context parser for the biologics analysis pipeline.

Parses structured context.md files (Markdown with embedded YAML blocks)
into a deterministic RunPlan that drives pipeline stage ordering, agent
selection, validation thresholds, and data constraints.

Backward-compatible: plain .txt files or Markdown without YAML blocks
produce an all-default RunPlan with the file contents as prose context.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dataclasses import asdict as _asdict

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants — known stage names and agent names
# ──────────────────────────────────────────────────────────────────────

VALID_STAGE_NAMES = frozenset({"cleaning", "analysis", "cross_validation", "report"})

KNOWN_AGENT_NAMES = frozenset({
    "data_cleaner", "chromatography_expert", "mass_spec_expert",
    "statistical_analyst", "ml_modeler", "analysis_planner",
    "cross_validator", "validator", "report_writer",
    # WP-4C extended agents
    "bioprocess_analyst", "visualisation_specialist", "statistical_modeler",
})

VALID_DOMAINS = frozenset({"chromatography", "mass_spectrometry", "both"})

# Default values migrated from captain_pipeline.py hardcoded logic
DEFAULT_STAGE_ORDER = ["cleaning", "analysis", "cross_validation", "report"]

DEFAULT_MAX_RETRIES = {
    "cleaning": 2,
    "analysis": 2,
    "cross_validation": 1,
    "report": 0,
}

DEFAULT_EXPERT_CALL_BUDGETS = {
    "cleaning": 2,
    "analysis": 3,
    "cross_validation": 3,
    "report": 2,
}

DEFAULT_CHAT_TIMEOUTS = {
    "cleaning": 900,
    "analysis": 2400,       # 4 expert calls × ~7 min each; needs ~30 min
    "cross_validation": 900,
    "report": 900,
}

DEFAULT_PRESERVE_COLUMNS = [
    "run_no", "run", "chromatography_stage", "Sample_Code",
    "column", "Fraction_number", "charge_state", "Spectrum_type",
    "Observed_m/z", "Expected_mass_(Da)",
]

# ──────────────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────────────


@dataclass
class RunConfig:
    """Per-run feature toggles for ablation studies.

    Parsed from context.md YAML under a ``run_config:`` key.
    Every field has a safe default that reproduces baseline behaviour.
    """
    run_label: str = ""                     # human-readable label for this run
    prompt_version: str = "v1"              # "v1" = baseline, "v2" = enhanced (WP4/WP8)
    # NOTE: critic_fail_safe and critic_max_attempts removed — the gated review
    # system now runs content evaluation on every attempt and uses structural_gate()
    # as the safety net when the evaluator LLM fails.
    content_validation: bool = True         # WP3: validate artifact JSON content, not just existence
    min_cross_val_claims: int = 5           # WP5: minimum verified claims in cross-validation
    recompute_cross_val: bool = False       # WP5: recompute metrics from parquet (fix tautological)
    figure_selection: str = "all"           # WP6: "all"|"ranked"|"top_n"
    max_report_figures: int = 10            # WP6: max figures when using ranked/top_n
    rerun_issue_map_enabled: bool = False   # WP9: inject targeted retry instructions
    protected_col_audit: bool = False       # WP10: warn on >95% missing protected columns
    schema_profiling: str = "disabled"      # WP-1A: "disabled"|"roles_only"|"full"
    visual_review_mode: str = "basic"       # WP-3: "basic"|"scientific"
    require_figure_references: bool = False  # WP-3: enforce claim-figure mapping
    iteration_strategy: str = "fixed"       # WP-2: "none"|"fixed"|"convergent"
    convergence_threshold: float = 0.05     # WP-2: min improvement to continue iterating
    convergence_target: float = 0.85        # WP-2: quality score at which to stop early
    expert_library: str = "baseline"        # WP-4: "baseline"|"extended"
    agent_definitions_dir: str = "agents/"  # WP-4: path to agent YAML files
    # WP-C1: Per-critic module toggles (backward-compatible defaults)
    critic_structural: bool = True          # structural gate (always-on by default)
    critic_content: bool = True             # LLM content evaluator
    critic_visual: bool = True              # VLM plot reviewer
    critic_analytical_depth: bool = False   # WP-C3a: analytical depth (off until validated)
    critic_execution: bool = False          # WP-C3b: execution correctness (off until validated)
    # WP-C2: Targeted refinement toggles
    targeted_refinement: bool = False       # enable finding_fix / gap_fill paths
    refinement_cascade: bool = False        # enable escalation cascade

    def to_dict(self) -> Dict[str, Any]:
        return _asdict(self)


@dataclass
class QualitySpec:
    """Validation thresholds for a pipeline stage."""
    min_plots: int = 3
    min_findings: int = 3
    require_per_group: bool = True
    summary_size_bounds: Tuple[int, int] = (1024, 500_000)
    png_min_bytes: int = 5000
    critic_enabled: bool = True


@dataclass
class AgentSelectionSpec:
    """Agent selection policy for a stage."""
    require: List[str] = field(default_factory=list)
    prefer: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)
    max_agents: int = 4
    fallback_to_detection: bool = True


@dataclass
class StageSpec:
    """Per-stage configuration."""
    name: str
    goals: List[str] = field(default_factory=list)
    required_outputs: List[str] = field(default_factory=list)
    agent_hints: AgentSelectionSpec = field(default_factory=AgentSelectionSpec)
    quality: QualitySpec = field(default_factory=QualitySpec)
    max_retries: int = 1
    expert_call_budget: int = 2
    max_rounds: int = 0  # 0 = use pipeline default (_STAGE_MAX_ROUNDS)
    chat_timeout: int = 0  # 0 = use global PIPELINE_CHAT_TIMEOUT_S / default
    custom_instructions: Optional[str] = None


@dataclass
class ConstraintSpec:
    """Data handling constraints."""
    preserve_columns: List[str] = field(default_factory=lambda: list(DEFAULT_PRESERVE_COLUMNS))
    no_aggregation_across: List[str] = field(default_factory=list)
    grouping_columns: List[str] = field(default_factory=list)
    grouping_extend_when_present: List[str] = field(default_factory=list)


@dataclass
class RunPlan:
    """Compiled execution plan from context file + defaults."""
    project_description: str = ""
    stages: List[StageSpec] = field(default_factory=list)
    global_constraints: ConstraintSpec = field(default_factory=ConstraintSpec)
    domain_override: Optional[str] = None
    run_config: RunConfig = field(default_factory=RunConfig)
    raw_yaml: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────────────────

# Regex to find ```yaml ... ``` fenced blocks in Markdown
_YAML_FENCE_RE = re.compile(
    r"```yaml\s*\n(.*?)```",
    re.DOTALL,
)


def _extract_yaml_blocks(text: str) -> Tuple[List[Dict[str, Any]], str]:
    """Extract YAML fenced blocks from Markdown text.

    Returns:
        (yaml_dicts, prose_text) where prose_text is the Markdown with
        YAML blocks removed.
    """
    yaml_dicts: List[Dict[str, Any]] = []
    prose_parts: List[str] = []
    last_end = 0

    for match in _YAML_FENCE_RE.finditer(text):
        prose_parts.append(text[last_end:match.start()])
        last_end = match.end()
        raw = match.group(1)
        try:
            parsed = yaml.safe_load(raw)
            if isinstance(parsed, dict):
                yaml_dicts.append(parsed)
            else:
                logger.warning("YAML block parsed to non-dict type (%s), skipping", type(parsed).__name__)
        except yaml.YAMLError as exc:
            logger.warning("Failed to parse YAML block: %s", exc)

    prose_parts.append(text[last_end:])
    prose_text = "\n".join(prose_parts).strip()
    return yaml_dicts, prose_text


def _merge_yaml_blocks(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple YAML blocks into a single dict (later blocks override)."""
    merged: Dict[str, Any] = {}
    for block in blocks:
        merged.update(block)
    return merged


# ──────────────────────────────────────────────────────────────────────
# Schema validation
# ──────────────────────────────────────────────────────────────────────

def _validate_schema(raw: Dict[str, Any]) -> List[str]:
    """Validate the parsed YAML against expected schema.

    Returns a list of warning messages (empty = valid).
    Does NOT raise — the pipeline should still work with partial/invalid context.
    """
    warnings: List[str] = []
    pipeline = raw.get("pipeline", {})
    if not isinstance(pipeline, dict):
        warnings.append(f"'pipeline' should be a dict, got {type(pipeline).__name__}")
        return warnings

    # Validate domain
    domain = pipeline.get("domain")
    if domain is not None and domain not in VALID_DOMAINS:
        warnings.append(
            f"Unknown domain '{domain}'. Valid: {sorted(VALID_DOMAINS)}"
        )

    # Validate stages
    stages = pipeline.get("stages", [])
    if not isinstance(stages, list):
        warnings.append(f"'pipeline.stages' should be a list, got {type(stages).__name__}")
    else:
        for i, stage in enumerate(stages):
            if not isinstance(stage, dict):
                warnings.append(f"Stage {i} should be a dict, got {type(stage).__name__}")
                continue
            name = stage.get("name")
            if name not in VALID_STAGE_NAMES:
                warnings.append(
                    f"Stage {i}: unknown name '{name}'. "
                    f"Valid: {sorted(VALID_STAGE_NAMES)}"
                )
            # Validate agent names
            hints = stage.get("agent_hints", {})
            if isinstance(hints, dict):
                for key in ("require", "prefer", "exclude"):
                    agents = hints.get(key, [])
                    if isinstance(agents, list):
                        for agent in agents:
                            if agent not in KNOWN_AGENT_NAMES:
                                warnings.append(
                                    f"Stage '{name}': unknown agent '{agent}' in "
                                    f"agent_hints.{key}. "
                                    f"Valid: {sorted(KNOWN_AGENT_NAMES)}"
                                )
            # Validate quality thresholds
            quality = stage.get("quality", {})
            if isinstance(quality, dict):
                for key in ("min_plots", "min_findings", "png_min_bytes"):
                    val = quality.get(key)
                    if val is not None:
                        if not isinstance(val, int) or val <= 0:
                            warnings.append(
                                f"Stage '{name}': quality.{key} must be a "
                                f"positive integer, got {val!r}"
                            )

    # Warn about unknown top-level keys
    known_top = {"pipeline", "constraints", "quality", "run_config"}
    unknown = set(raw.keys()) - known_top
    if unknown:
        warnings.append(f"Unknown top-level keys (ignored): {sorted(unknown)}")

    return warnings


# ──────────────────────────────────────────────────────────────────────
# RunPlan compilation
# ──────────────────────────────────────────────────────────────────────

def _build_quality_spec(raw: Dict[str, Any], stage_name: str) -> QualitySpec:
    """Build a QualitySpec from raw YAML dict, with defaults."""
    defaults = QualitySpec()
    spec = QualitySpec(
        min_plots=raw.get("min_plots", defaults.min_plots),
        min_findings=raw.get("min_findings", defaults.min_findings),
        require_per_group=raw.get("require_per_group", defaults.require_per_group),
        png_min_bytes=raw.get("png_min_bytes", defaults.png_min_bytes),
    )
    # Handle summary_size_bounds
    bounds = raw.get("summary_size_bounds")
    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
        spec.summary_size_bounds = (int(bounds[0]), int(bounds[1]))

    # Safety floors: reject zero/negative but allow lowering with warning
    for attr in ("min_plots", "min_findings", "png_min_bytes"):
        val = getattr(spec, attr)
        if val <= 0:
            default_val = getattr(defaults, attr)
            logger.warning(
                "Stage '%s': quality.%s=%d is invalid (must be > 0), "
                "using default %d",
                stage_name, attr, val, default_val,
            )
            setattr(spec, attr, default_val)
        elif val < getattr(defaults, attr):
            logger.warning(
                "Stage '%s': quality.%s=%d is below default (%d). "
                "Proceeding with reduced threshold.",
                stage_name, attr, val, getattr(defaults, attr),
            )

    return spec


def _build_agent_selection_spec(raw: Dict[str, Any]) -> AgentSelectionSpec:
    """Build an AgentSelectionSpec from raw YAML dict."""
    return AgentSelectionSpec(
        require=list(raw.get("require", [])),
        prefer=list(raw.get("prefer", [])),
        exclude=list(raw.get("exclude", [])),
        max_agents=int(raw.get("max_agents", 4)),
        fallback_to_detection=bool(raw.get("fallback_to_detection", True)),
    )


def _build_stage_spec(raw: Dict[str, Any]) -> Optional[StageSpec]:
    """Build a StageSpec from a raw YAML stage dict."""
    name = raw.get("name")
    if not isinstance(name, str) or name not in VALID_STAGE_NAMES:
        return None

    quality_raw = raw.get("quality", {})
    if not isinstance(quality_raw, dict):
        quality_raw = {}

    hints_raw = raw.get("agent_hints", {})
    if not isinstance(hints_raw, dict):
        hints_raw = {}

    goals = raw.get("goals", [])
    if not isinstance(goals, list):
        goals = [str(goals)]

    return StageSpec(
        name=name,
        goals=[str(g) for g in goals],
        required_outputs=list(raw.get("required_outputs", [])),
        agent_hints=_build_agent_selection_spec(hints_raw),
        quality=_build_quality_spec(quality_raw, name),
        max_retries=int(raw.get("max_retries", DEFAULT_MAX_RETRIES.get(name, 1))),
        expert_call_budget=int(raw.get(
            "expert_call_budget",
            DEFAULT_EXPERT_CALL_BUDGETS.get(name, 2),
        )),
        max_rounds=int(raw.get("max_rounds", 0)),
        chat_timeout=int(raw.get(
            "chat_timeout",
            DEFAULT_CHAT_TIMEOUTS.get(name, 0),
        )),
        custom_instructions=raw.get("custom_instructions"),
    )


def _build_run_config(raw: Dict[str, Any]) -> RunConfig:
    """Build a RunConfig from raw YAML dict (under ``run_config:`` key)."""
    if not isinstance(raw, dict):
        return RunConfig()
    defaults = RunConfig()
    # Warn on deprecated fields that old context.md files may still reference
    for _deprecated_key in ("critic_fail_safe", "critic_max_attempts"):
        if _deprecated_key in raw:
            logger.warning(
                "run_config.%s is deprecated and ignored "
                "(replaced by gated review system)", _deprecated_key,
            )
    cfg = RunConfig(
        run_label=str(raw.get("run_label", defaults.run_label)),
        prompt_version=str(raw.get("prompt_version", defaults.prompt_version)),
        content_validation=bool(raw.get("content_validation", defaults.content_validation)),
        min_cross_val_claims=int(raw.get("min_cross_val_claims", defaults.min_cross_val_claims)),
        recompute_cross_val=bool(raw.get("recompute_cross_val", defaults.recompute_cross_val)),
        figure_selection=str(raw.get("figure_selection", defaults.figure_selection)),
        max_report_figures=int(raw.get("max_report_figures", defaults.max_report_figures)),
        rerun_issue_map_enabled=bool(raw.get("rerun_issue_map_enabled", defaults.rerun_issue_map_enabled)),
        protected_col_audit=bool(raw.get("protected_col_audit", defaults.protected_col_audit)),
        schema_profiling=str(raw.get("schema_profiling", defaults.schema_profiling)),
        visual_review_mode=str(raw.get("visual_review_mode", defaults.visual_review_mode)),
        require_figure_references=bool(raw.get("require_figure_references", defaults.require_figure_references)),
        iteration_strategy=str(raw.get("iteration_strategy", defaults.iteration_strategy)),
        convergence_threshold=float(raw.get("convergence_threshold", defaults.convergence_threshold)),
        convergence_target=float(raw.get("convergence_target", defaults.convergence_target)),
        expert_library=str(raw.get("expert_library", defaults.expert_library)),
        agent_definitions_dir=str(raw.get("agent_definitions_dir", defaults.agent_definitions_dir)),
        # WP-C1: Per-critic module toggles
        critic_structural=bool(raw.get("critic_structural", defaults.critic_structural)),
        critic_content=bool(raw.get("critic_content", defaults.critic_content)),
        critic_visual=bool(raw.get("critic_visual", defaults.critic_visual)),
        critic_analytical_depth=bool(raw.get("critic_analytical_depth", defaults.critic_analytical_depth)),
        critic_execution=bool(raw.get("critic_execution", defaults.critic_execution)),
        # WP-C2: Targeted refinement toggles
        targeted_refinement=bool(raw.get("targeted_refinement", defaults.targeted_refinement)),
        refinement_cascade=bool(raw.get("refinement_cascade", defaults.refinement_cascade)),
    )
    if cfg.visual_review_mode not in ("basic", "scientific"):
        logger.warning(
            "run_config.visual_review_mode='%s' invalid, defaulting to 'basic'",
            cfg.visual_review_mode,
        )
        cfg.visual_review_mode = "basic"
    if cfg.iteration_strategy not in ("none", "fixed", "convergent"):
        logger.warning(
            "run_config.iteration_strategy='%s' invalid, defaulting to 'fixed'",
            cfg.iteration_strategy,
        )
        cfg.iteration_strategy = "fixed"
    if cfg.schema_profiling not in ("disabled", "roles_only", "full"):
        logger.warning(
            "run_config.schema_profiling='%s' invalid, defaulting to 'disabled'",
            cfg.schema_profiling,
        )
        cfg.schema_profiling = "disabled"
    if cfg.figure_selection not in ("all", "ranked", "top_n"):
        logger.warning(
            "run_config.figure_selection='%s' invalid, defaulting to 'all'",
            cfg.figure_selection,
        )
        cfg.figure_selection = "all"
    if cfg.prompt_version not in ("v1", "v2"):
        logger.warning(
            "run_config.prompt_version='%s' invalid, defaulting to 'v1'",
            cfg.prompt_version,
        )
        cfg.prompt_version = "v1"
    if cfg.expert_library not in ("baseline", "extended"):
        logger.warning(
            "run_config.expert_library='%s' invalid, defaulting to 'baseline'",
            cfg.expert_library,
        )
        cfg.expert_library = "baseline"
    return cfg


def _build_constraint_spec(raw: Dict[str, Any]) -> ConstraintSpec:
    """Build a ConstraintSpec from raw YAML."""
    return ConstraintSpec(
        preserve_columns=list(raw.get("preserve_columns", DEFAULT_PRESERVE_COLUMNS)),
        no_aggregation_across=list(raw.get("no_aggregation_across", [])),
        grouping_columns=list(raw.get("grouping_columns", [])),
        grouping_extend_when_present=list(raw.get("grouping_extend_when_present", [])),
    )


def _default_stages() -> List[StageSpec]:
    """Build the default four-stage pipeline (matches current hardcoded behaviour)."""
    return [
        StageSpec(
            name=name,
            max_retries=DEFAULT_MAX_RETRIES[name],
            expert_call_budget=DEFAULT_EXPERT_CALL_BUDGETS[name],
            chat_timeout=DEFAULT_CHAT_TIMEOUTS.get(name, 0),
        )
        for name in DEFAULT_STAGE_ORDER
    ]


def compile_run_plan(
    yaml_data: Dict[str, Any],
    prose_text: str,
) -> RunPlan:
    """Compile a RunPlan from parsed YAML + prose text.

    Merges YAML overrides with defaults. If no YAML is provided,
    returns an all-default RunPlan.
    """
    pipeline = yaml_data.get("pipeline", {})
    if not isinstance(pipeline, dict):
        pipeline = {}

    # Domain override
    domain_override = pipeline.get("domain")
    if domain_override not in VALID_DOMAINS:
        if domain_override is not None:
            logger.warning("Invalid domain '%s', ignoring override", domain_override)
        domain_override = None

    # Stages
    stages_raw = pipeline.get("stages")
    if isinstance(stages_raw, list) and stages_raw:
        stages: List[StageSpec] = []
        for raw_stage in stages_raw:
            if isinstance(raw_stage, dict):
                spec = _build_stage_spec(raw_stage)
                if spec is not None:
                    stages.append(spec)
        if not stages:
            logger.warning("No valid stages parsed from context, using defaults")
            stages = _default_stages()
        # Validate ordering constraints
        stage_names = [s.name for s in stages]
        if "cleaning" in stage_names and stage_names[0] != "cleaning":
            logger.warning(
                "Stage 'cleaning' must be first when present. "
                "Reordering to move cleaning to position 0."
            )
            cleaning = [s for s in stages if s.name == "cleaning"]
            others = [s for s in stages if s.name != "cleaning"]
            stages = cleaning + others
        if "report" in stage_names and stage_names[-1] != "report":
            logger.warning(
                "Stage 'report' must be last when present. "
                "Reordering to move report to final position."
            )
            report = [s for s in stages if s.name == "report"]
            others = [s for s in stages if s.name != "report"]
            stages = others + report
    else:
        stages = _default_stages()

    # Constraints — from pipeline.constraints or top-level constraints
    constraints_raw = pipeline.get("constraints", yaml_data.get("constraints", {}))
    if not isinstance(constraints_raw, dict):
        constraints_raw = {}

    # Run config — from pipeline.run_config or top-level run_config
    run_config_raw = pipeline.get("run_config", yaml_data.get("run_config", {}))
    if not isinstance(run_config_raw, dict):
        run_config_raw = {}

    return RunPlan(
        project_description=prose_text,
        stages=stages,
        global_constraints=_build_constraint_spec(constraints_raw),
        domain_override=domain_override,
        run_config=_build_run_config(run_config_raw),
        raw_yaml=yaml_data,
    )


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def parse_context(path: Optional[Path]) -> RunPlan:
    """Parse a context file and compile a RunPlan.

    Supports:
      - .md files with embedded ```yaml blocks
      - .txt files (treated as prose-only, all-default RunPlan)
      - None (no context file, all-default RunPlan)

    Always returns a valid RunPlan — never raises on malformed input.
    """
    if path is None:
        logger.info("No context file provided, using all-default RunPlan")
        return compile_run_plan({}, "")

    if not path.exists():
        logger.warning("Context file not found: %s, using defaults", path)
        return compile_run_plan({}, "")

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        logger.error("Failed to read context file %s: %s", path, exc)
        return compile_run_plan({}, "")

    # For .txt files or files with no YAML blocks, treat as prose-only
    if path.suffix.lower() == ".txt":
        logger.info("Context file is .txt, treating as prose-only")
        return compile_run_plan({}, text.strip())

    # Extract YAML blocks from Markdown
    yaml_blocks, prose_text = _extract_yaml_blocks(text)

    if not yaml_blocks:
        logger.info("No YAML blocks found in %s, treating as prose-only", path.name)
        return compile_run_plan({}, text.strip())

    merged = _merge_yaml_blocks(yaml_blocks)

    # Validate
    warnings = _validate_schema(merged)
    for w in warnings:
        logger.warning("Context schema: %s", w)

    return compile_run_plan(merged, prose_text)
