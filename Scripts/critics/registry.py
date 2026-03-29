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
from tools import CheckCategory, CheckResult, Severity, StageVerdict

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

        # Track whether structural gate found *artifact validity* failures
        # (missing files, corrupt JSON).  Only these justify skipping LLM/VLM
        # critics — analytical coverage gaps (e.g. grouping_adequacy) should
        # NOT prevent content/visual evaluation.
        _ARTIFACT_VALIDITY_CHECKS = {
            "file_exists__analysis_summary_path",
            "file_exists__cleaned_path",
            "file_exists__summary_path",
            "json_valid__analysis_summary",
            "cleaning_row_counts",
        }
        structural_failed = False

        for critic in active:
            if not critic.can_run(ctx):
                ctx.evaluators_ran[critic.name] = False
                continue

            # Optimisation: if structural gate found artifact-level failures,
            # skip LLM/VLM critics (they need valid artifacts to evaluate).
            if structural_failed and (critic.requires_llm or critic.requires_vlm):
                ctx.evaluators_ran[critic.name] = False
                continue

            checks = critic.safe_evaluate(ctx)

            # De-duplicate interpretation depth: if the content evaluator
            # already flagged interpretation_depth as MUST_FIX, downgrade
            # the analytical depth critic's bare_deviations check to
            # SHOULD_FIX to avoid double-penalising the same gap.
            if critic.name == "analytical_depth":
                _content_flagged_interp = any(
                    c.name == "interpretation_depth"
                    and c.severity == Severity.MUST_FIX
                    for c in all_checks.get(CheckCategory.CONTENT_QUALITY, [])
                )
                if _content_flagged_interp:
                    for c in checks:
                        if (
                            c.name == "depth__bare_deviations"
                            and c.severity == Severity.MUST_FIX
                        ):
                            c.severity = Severity.SHOULD_FIX
                            logger.debug(
                                "Downgraded depth__bare_deviations to SHOULD_FIX "
                                "(interpretation_depth already flagged by content critic)"
                            )

            all_checks.setdefault(critic.category, []).extend(checks)
            ctx.evaluators_ran[critic.name] = bool(checks)

            # Only short-circuit on artifact validity failures (missing files,
            # corrupt JSON).  Analytical gaps like grouping_adequacy are
            # MUST_FIX but the artifacts are valid — LLM/VLM critics can
            # still evaluate them.
            if critic.name == "structural":
                structural_failed = any(
                    not c.passed and c.severity == Severity.MUST_FIX
                    and c.name in _ARTIFACT_VALIDITY_CHECKS
                    for c in checks
                )

        return StageVerdict(
            stage=ctx.stage_name,
            attempt=ctx.attempt,
            structural_checks=all_checks.get(CheckCategory.STRUCTURAL, []),
            content_checks=all_checks.get(CheckCategory.CONTENT_QUALITY, []),
            plot_checks=all_checks.get(CheckCategory.PLOT_QUALITY, []),
            claim_checks=all_checks.get(CheckCategory.CLAIM_EVIDENCE, []),
            evaluators_ran=dict(ctx.evaluators_ran),
        )
