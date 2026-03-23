"""CriticRegistry — discovers, orders, and dispatches critic modules.

WP-C1: The registry is instantiated once at pipeline init.  The dispatch
loop calls ``get_active_critics()`` per stage to get the ordered list of
modules whose ``can_run()`` check should be evaluated.

Execution order mirrors the existing pipeline logic:
  1. StructuralCritic    (pure Python, always first)
  2. ExecutionCritic     (pure Python, WP-C3b — before LLM critics)
  3. ContentCritic       (LLM)
  4. AnalyticalDepthCritic (LLM, WP-C3a)
  5. PlotStructuralCritic (pure Python, WP-C4 — before VLM)
  6. VisualCritic        (VLM)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from critics.base import CriticContext, CriticModule
from tools import CheckCategory, CheckResult, StageVerdict

logger = logging.getLogger("captain_pipeline")

# Explicit execution order — lower number runs first
_CRITIC_ORDER: Dict[str, int] = {
    "structural": 10,
    "execution": 20,
    "content": 30,
    "analytical_depth": 40,
    "plot_structural": 50,
    "plot_quality": 60,
}


class CriticRegistry:
    """Registry of critic modules for the gated review loop.

    Parameters
    ----------
    pipeline : CaptainPipeline
        The pipeline instance (passed to critics that need delegation).
    """

    def __init__(self, pipeline: Any) -> None:
        self._critics: List[CriticModule] = []
        self._build_registry(pipeline)

    def _build_registry(self, pipeline: Any) -> None:
        """Instantiate all available critic modules."""
        # WP-C1: core critics (wrapping existing logic)
        from critics.structural import StructuralCritic
        from critics.content import ContentCritic
        from critics.visual import VisualCritic

        self._critics.append(StructuralCritic())
        self._critics.append(ContentCritic(pipeline))
        self._critics.append(VisualCritic(pipeline))

        # WP-C3b: Execution critic (pure Python)
        try:
            from critics.execution import ExecutionCritic
            self._critics.append(ExecutionCritic())
        except ImportError:
            pass

        # WP-C3a: Analytical depth critic
        try:
            from critics.analytical_depth import AnalyticalDepthCritic
            self._critics.append(AnalyticalDepthCritic(pipeline))
        except ImportError:
            pass

        # WP-C4: Plot structural critic (pure Python)
        try:
            from critics.plot_structural import PlotStructuralCritic
            self._critics.append(PlotStructuralCritic())
        except ImportError:
            pass

        # Sort by execution order
        self._critics.sort(
            key=lambda c: _CRITIC_ORDER.get(c.name, 100)
        )

        logger.info(
            "CriticRegistry built with %d modules: %s",
            len(self._critics),
            [c.name for c in self._critics],
        )

    def get_active_critics(
        self,
        stage_name: str,
        run_config: Any,
    ) -> List[CriticModule]:
        """Return ordered list of critics that apply to this stage and are enabled."""
        return [
            c for c in self._critics
            if stage_name in c.stage_applicability
        ]

    def dispatch(
        self,
        ctx: CriticContext,
    ) -> StageVerdict:
        """Run all active critics and aggregate results into a StageVerdict.

        This is the main entry point called by ``run_stage_gated()``.
        Each critic's ``can_run()`` is checked; if True, ``safe_evaluate()``
        is called (which handles exceptions and logging).  Results are
        routed into the correct StageVerdict slot by ``critic.category``.
        """
        all_checks: Dict[CheckCategory, List[CheckResult]] = {}
        active = self.get_active_critics(ctx.stage_name, ctx.run_config)

        # Track whether structural gate found failures (used for early-exit
        # optimisation: skip LLM critics if structural gate failed)
        structural_failed = False

        for critic in active:
            if not critic.can_run(ctx):
                ctx.evaluators_ran[critic.name] = False
                continue

            # Optimisation: if structural gate failed, skip LLM/VLM critics
            # (they need valid artifacts to evaluate). This preserves the
            # existing short-circuit behaviour from captain_pipeline.py.
            if structural_failed and (critic.requires_llm or critic.requires_vlm):
                ctx.evaluators_ran[critic.name] = False
                continue

            checks = critic.safe_evaluate(ctx)
            all_checks.setdefault(critic.category, []).extend(checks)
            ctx.evaluators_ran[critic.name] = bool(checks)

            # Track structural failures for the short-circuit
            if critic.name == "structural":
                structural_failed = any(not c.passed for c in checks)

        return StageVerdict(
            stage=ctx.stage_name,
            attempt=ctx.attempt,
            structural_checks=all_checks.get(CheckCategory.STRUCTURAL, []),
            content_checks=all_checks.get(CheckCategory.CONTENT_QUALITY, []),
            plot_checks=all_checks.get(CheckCategory.PLOT_QUALITY, []),
            claim_checks=all_checks.get(CheckCategory.CLAIM_EVIDENCE, []),
            evaluators_ran=dict(ctx.evaluators_ran),
        )
