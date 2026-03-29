"""StructuralCritic — wraps the existing ``structural_gate()`` from tools.py.

WP-C1: Zero behaviour change for existing checks.  Delegates to the
pure-Python structural gate and returns its CheckResult list through
the CriticModule interface.

Grouping adequacy check (WP-1A): On the analysis stage, compares the
grouping dimensions used in the analysis_summary against the dimensional
structure detected by the schema profiler.  Flags as should_fix if
analytically meaningful dimensions were detected but not used.
"""
from __future__ import annotations

import logging
from typing import List, Set

from tools import CheckCategory, CheckResult, Severity, structural_gate
from critics.base import CriticContext, CriticModule

logger = logging.getLogger("captain_pipeline")

# Semantic purposes considered analytically meaningful for grouping.
_MEANINGFUL_PURPOSES = {"process_phase", "experimental_condition", "experimental_unit"}


class StructuralCritic(CriticModule):
    """Pure-Python structural validation (file existence, JSON validity, thresholds)."""

    name = "structural"
    category = CheckCategory.STRUCTURAL
    stage_applicability: Set[str] = {"cleaning", "analysis", "cross_validation"}
    requires_llm = False
    requires_vlm = False

    def can_run(self, ctx: CriticContext) -> bool:
        if ctx.stage_name not in self.stage_applicability:
            return False
        return getattr(ctx.run_config, "critic_structural", True)

    def evaluate(self, ctx: CriticContext) -> List[CheckResult]:
        checks = structural_gate(
            stage_name=ctx.stage_name,
            expected_paths=ctx.expected_paths,
            output_dir_listing=ctx.listing,
            quality_min_plots=getattr(ctx.quality_spec, "min_plots", 3),
            quality_min_findings=getattr(ctx.quality_spec, "min_findings", 3),
            quality_require_per_group=getattr(ctx.quality_spec, "require_per_group", True),
            quality_png_min_bytes=getattr(ctx.quality_spec, "png_min_bytes", 5000),
            require_figure_references=getattr(
                ctx.run_config, "require_figure_references", False,
            ),
        )

        # Grouping adequacy check — analysis stage only
        if ctx.stage_name == "analysis":
            checks.extend(self._check_grouping_adequacy(ctx))

        return checks

    # ── Grouping adequacy (WP-1A) ────────────────────────────────────

    @staticmethod
    def _check_grouping_adequacy(ctx: CriticContext) -> List[CheckResult]:
        """Compare used grouping against detected dimensional structure.

        Flags should_fix if the profiler detected analytically meaningful
        dimensions (process_phase, experimental_condition) that do not
        appear in the analysis grouping.
        """
        results: List[CheckResult] = []

        # Extract data profile from payload (set by captain_pipeline)
        profile = (ctx.payload or {}).get("data_profile")
        if not profile or not isinstance(profile, dict):
            return results

        ds = profile.get("dimensional_structure")
        if not ds:
            return results

        rec = profile.get("recommended_grouping")
        if not rec:
            return results

        used_cols = set(rec.get("columns", []))
        dimensions = ds.get("dimensions", [])

        # Collect meaningful dimensions not covered by the grouping
        missing_dims = []
        for dim in dimensions:
            purpose = dim.get("purpose", "")
            dim_cols = set(dim.get("columns", []))
            if purpose in _MEANINGFUL_PURPOSES and not dim_cols & used_cols:
                missing_dims.append(
                    f"{dim.get('name', '?')} [{purpose}, "
                    f"{dim.get('cardinality', '?')} levels]"
                )

        if len(missing_dims) >= 2:
            # 2+ missing important dimensions → MUST_FIX to force secondary
            # analysis context.  The primary grouping key stays as-is; agents
            # must add a secondary analysis covering the missing dimensions.
            results.append(CheckResult(
                name="grouping_adequacy",
                passed=False,
                severity=Severity.MUST_FIX,
                category=CheckCategory.STRUCTURAL,
                detail=(
                    f"Grouping uses {sorted(used_cols)} but the profiler "
                    f"detected {len(missing_dims)} additional meaningful "
                    f"dimensions not covered: {', '.join(missing_dims)}. "
                    f"A secondary analysis context using these dimensions "
                    f"is required."
                ),
                fix_instruction=(
                    "Keep the current primary grouping key. Add a SECONDARY "
                    "analysis using the missing dimensions (e.g. per-stage "
                    "trends, per-condition comparisons). Include results in "
                    "a 'secondary_analysis' key in analysis_summary.json or "
                    "as additional findings that explicitly reference the "
                    "missing dimensions. Use the ANALYSIS CONTEXTS from the "
                    "data profile for guidance."
                ),
            ))
        elif len(missing_dims) == 1:
            # 1 missing dimension → advisory SHOULD_FIX
            results.append(CheckResult(
                name="grouping_adequacy",
                passed=False,
                severity=Severity.SHOULD_FIX,
                category=CheckCategory.STRUCTURAL,
                detail=(
                    f"Grouping uses {sorted(used_cols)} but the profiler "
                    f"detected 1 additional meaningful dimension not "
                    f"covered: {missing_dims[0]}. Consider including it "
                    f"in a secondary analysis."
                ),
                fix_instruction=(
                    "Consider adding a secondary analysis using the missing "
                    "dimension for finer-grained insights."
                ),
            ))
        else:
            results.append(CheckResult(
                name="grouping_adequacy",
                passed=True,
                category=CheckCategory.STRUCTURAL,
                detail=(
                    f"Grouping {sorted(used_cols)} covers the major "
                    f"analytical dimensions."
                ),
            ))

        return results
