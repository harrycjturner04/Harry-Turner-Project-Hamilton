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
            checks.extend(self._check_domain_reasoning(ctx))

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

        # ── Check whether secondary_analysis already covers the missing dims ──
        # If the agent has already added a 'secondary_analysis' key that covers
        # the missing dimensions, downgrade from MUST_FIX to SHOULD_FIX.
        secondary_covers = False
        if missing_dims:
            import json as _json
            from pathlib import Path as _Path
            asp = (ctx.payload or {}).get("analysis_summary_path", "")
            if asp:
                try:
                    _data = _json.loads(_Path(asp).read_text("utf-8"))
                    _sa = _data.get("secondary_analysis", {})
                    if isinstance(_sa, dict) and _sa:
                        _sa_keys = {k.lower() for k in _sa.keys()}
                        # missing_dims entries: "run [experimental_unit, 11 levels]"
                        _dim_names = {d.split(" [")[0].lower() for d in missing_dims}
                        if _dim_names & _sa_keys:
                            secondary_covers = True
                except Exception:
                    pass

        if len(missing_dims) >= 2 and not secondary_covers:
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
        elif len(missing_dims) >= 2 and secondary_covers:
            # secondary_analysis covers the missing dimensions — advisory only
            results.append(CheckResult(
                name="grouping_adequacy",
                passed=False,
                severity=Severity.SHOULD_FIX,
                category=CheckCategory.STRUCTURAL,
                detail=(
                    f"Grouping uses {sorted(used_cols)} but {len(missing_dims)} "
                    f"meaningful dimensions are not in the primary grouping: "
                    f"{', '.join(missing_dims)}. Secondary analysis is present "
                    f"and covers the missing dimensions."
                ),
                fix_instruction=(
                    "The secondary_analysis key covers the missing dimensions. "
                    "Consider whether the primary grouping should also include "
                    "these dimensions for completeness."
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
