"""StructuralCritic — wraps the existing ``structural_gate()`` from tools.py.

WP-C1: Zero behaviour change.  Delegates to the pure-Python structural gate
and returns its CheckResult list through the CriticModule interface.
"""
from __future__ import annotations

from typing import List, Set

from tools import CheckCategory, CheckResult, structural_gate
from critics.base import CriticContext, CriticModule


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
        return structural_gate(
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
