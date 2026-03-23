"""VisualCritic — wraps the existing ``_run_plot_quality_evaluator()`` pipeline method.

WP-C1: Zero behaviour change.  Delegates to the CaptainPipeline instance
method which handles VLM calls, scientific/basic mode switching, and
per-plot CheckResult generation.
"""
from __future__ import annotations

from typing import Any, List, Set

from tools import CheckCategory, CheckResult
from critics.base import CriticContext, CriticModule


class VisualCritic(CriticModule):
    """VLM-based per-plot quality evaluation (analysis stage only)."""

    name = "plot_quality"
    category = CheckCategory.PLOT_QUALITY
    stage_applicability: Set[str] = {"analysis"}
    requires_llm = False
    requires_vlm = True

    def __init__(self, pipeline: Any) -> None:
        """Accept a CaptainPipeline reference for delegation.

        Parameters
        ----------
        pipeline : CaptainPipeline
            The pipeline instance that owns ``_run_plot_quality_evaluator()``
            and ``_vlm_available()``.
        """
        self._pipeline = pipeline

    def can_run(self, ctx: CriticContext) -> bool:
        if ctx.stage_name not in self.stage_applicability:
            return False
        if not getattr(ctx.run_config, "critic_visual", True):
            return False
        return self._pipeline._vlm_available()

    def evaluate(self, ctx: CriticContext) -> List[CheckResult]:
        return self._pipeline._run_plot_quality_evaluator(
            ctx.listing, ctx.label,
        )
