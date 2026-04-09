"""StructuralCritic — wraps the existing ``structural_gate()`` from tools.py.

WP-C1: Zero behaviour change for existing checks.  Delegates to the
pure-Python structural gate and returns its CheckResult list through
the CriticModule interface.

Domain reasoning structural check: verifies that analysis_summary.json
contains a domain_reasoning field with hypotheses or observations.

Note: Grouping adequacy is handled solely by AnalyticalDepthCritic
(depth__grouping_adequacy / depth__grouping_inadequate) to avoid
duplicate checks with divergent thresholds.
"""
from __future__ import annotations

import logging
from typing import List, Set

from tools import CheckCategory, CheckResult, Severity, structural_gate
from critics.base import CriticContext, CriticModule

logger = logging.getLogger("captain_pipeline")


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

        # Domain reasoning structural check — analysis stage only.
        # NOTE: Grouping adequacy is handled solely by AnalyticalDepthCritic
        # (depth__grouping_adequacy / depth__grouping_inadequate) to avoid
        # duplicate checks with divergent thresholds.
        if ctx.stage_name == "analysis":
            checks.extend(self._check_domain_reasoning(ctx))

        return checks

    # ── Domain reasoning structural check ─────────────────────────────

    @staticmethod
    def _check_domain_reasoning(ctx: CriticContext) -> List[CheckResult]:
        """Verify analysis_summary.json contains a domain_reasoning field.

        Checks for at least 1 entry in hypotheses_tested or
        plan_modifications_applied. Lightweight Python check (no LLM).
        """
        import json as _json

        results: List[CheckResult] = []

        # Find analysis_summary.json in the payload
        asp = (ctx.payload or {}).get("analysis_summary_path", "")
        if not asp:
            return results

        from pathlib import Path
        asp_path = Path(asp)
        if not asp_path.exists():
            return results

        try:
            data = _json.loads(asp_path.read_text("utf-8"))
        except Exception:
            return results

        dr = data.get("domain_reasoning")
        if not isinstance(dr, dict):
            results.append(CheckResult(
                name="domain_reasoning",
                passed=False,
                severity=Severity.MUST_FIX,
                category=CheckCategory.STRUCTURAL,
                detail=(
                    "analysis_summary.json is missing the 'domain_reasoning' "
                    "field. Expert output must include hypotheses tested, plan "
                    "modifications, or unexpected observations."
                ),
                fix_instruction=(
                    "Add a 'domain_reasoning' field to analysis_summary.json "
                    "with at least one entry in 'hypotheses_tested' or "
                    "'plan_modifications_applied'. Format: "
                    '{"domain_reasoning": {"hypotheses_tested": [{"hypothesis": '
                    '"...", "verdict": "supported|refuted|inconclusive", '
                    '"evidence": "..."}], "plan_modifications_applied": [...], '
                    '"unexpected_observations": [...]}}'
                ),
            ))
            return results

        hypotheses = dr.get("hypotheses_tested", [])
        modifications = dr.get("plan_modifications_applied", [])
        observations = dr.get("unexpected_observations", [])
        total_entries = len(hypotheses) + len(modifications) + len(observations)

        if total_entries == 0:
            results.append(CheckResult(
                name="domain_reasoning",
                passed=False,
                severity=Severity.SHOULD_FIX,
                category=CheckCategory.STRUCTURAL,
                detail=(
                    "domain_reasoning field exists but is empty — no hypotheses "
                    "tested, no plan modifications, and no unexpected observations "
                    "documented."
                ),
                fix_instruction=(
                    "Populate domain_reasoning with at least one hypothesis "
                    "tested or one plan modification rationale."
                ),
            ))
        else:
            # Check whether the two most diagnostic sub-fields are populated.
            # hypotheses_tested alone is sufficient to pass the existence gate,
            # but plan_modifications_applied and unexpected_observations add
            # important provenance — flag as should_fix when both are absent.
            incomplete: list = []
            if not modifications:
                incomplete.append("plan_modifications_applied is empty")
            if not observations:
                incomplete.append("unexpected_observations is empty")

            if incomplete:
                results.append(CheckResult(
                    name="domain_reasoning",
                    passed=False,
                    severity=Severity.SHOULD_FIX,
                    category=CheckCategory.STRUCTURAL,
                    detail=(
                        f"domain_reasoning present: {len(hypotheses)} hypotheses, "
                        f"{len(modifications)} modifications, "
                        f"{len(observations)} observations — incomplete: "
                        + "; ".join(incomplete)
                    ),
                    fix_instruction=(
                        "Populate 'domain_reasoning' in analysis_summary.json. "
                        "'plan_modifications_applied' should list at least one "
                        "adaptation made during analysis. "
                        "'unexpected_observations' should list at least one "
                        "observation that was not anticipated."
                    ),
                ))
            else:
                results.append(CheckResult(
                    name="domain_reasoning",
                    passed=True,
                    category=CheckCategory.STRUCTURAL,
                    detail=(
                        f"domain_reasoning present: {len(hypotheses)} hypotheses, "
                        f"{len(modifications)} modifications, "
                        f"{len(observations)} observations."
                    ),
                ))

        return results
