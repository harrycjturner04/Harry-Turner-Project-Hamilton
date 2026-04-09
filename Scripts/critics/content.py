"""ContentCritic — wraps the existing ``_run_content_evaluator()`` pipeline method.

WP-C1: Zero behaviour change.  Delegates to the CaptainPipeline instance
method which handles LLM calls, rubric selection, coverage validation,
and CheckResult parsing.
"""
from __future__ import annotations

from typing import Any, List, Set

from tools import CheckCategory, CheckResult
from critics.base import CriticContext, CriticModule


class ContentCritic(CriticModule):
    """LLM-based content quality evaluation using per-stage rubrics."""

    name = "content"
    category = CheckCategory.CONTENT_QUALITY
    stage_applicability: Set[str] = {"cleaning", "analysis", "cross_validation"}
    requires_llm = True
    requires_vlm = False

    def __init__(self, pipeline: Any) -> None:
        """Accept a CaptainPipeline reference for delegation.

        Parameters
        ----------
        pipeline : CaptainPipeline
            The pipeline instance that owns ``_run_content_evaluator()``
            and ``_build_critic_input()``.
        """
        self._pipeline = pipeline

    def can_run(self, ctx: CriticContext) -> bool:
        if ctx.stage_name not in self.stage_applicability:
            return False
        # Honour both the new per-critic toggle and the legacy QualitySpec toggle
        if not getattr(ctx.run_config, "critic_content", True):
            return False
        if not getattr(ctx.quality_spec, "critic_enabled", True):
            return False
        return True

    def evaluate(self, ctx: CriticContext) -> List[CheckResult]:
        # Build the artifact summary using the existing pipeline method
        artifacts_summary = self._pipeline._build_critic_input(
            ctx.stage_name, ctx.payload, ctx.listing, ctx.last_reply,
        )
        return self._pipeline._run_content_evaluator(
            ctx.stage_name,
            artifacts_summary,
            f"{ctx.label}__content_eval",
            previous_content_checks=ctx.previous_content_checks,
        )
