"""AnalyticalDepthCritic — detects shallow or incomplete analysis.

WP-C3a: Identifies outputs that pass structural and content checks but
lack scientific substance.  The existing content evaluator checks whether
findings are well-formed; this critic checks whether the right analyses
were performed at all.

Combines pure-Python heuristic checks (group comparison, chart diversity,
domain-specific expectations) with an optional LLM-enhanced assessment
for statistical test selection and missed dimensions.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tools import (
    CheckCategory,
    CheckResult,
    Severity,
    detect_data_domains,
    finding_text,
    parse_json_tolerant,
    safe_write_json,
    strip_think_tokens,
)
from critics.base import CriticContext, CriticModule

logger = logging.getLogger("captain_pipeline")

# Keywords that indicate interpretive depth (not just bare numbers)
_INTERPRETATION_VOCABULARY = {
    "degradation", "trend", "deviation", "anomaly", "outlier",
    "correlation", "interaction", "root cause", "hypothesis",
    "suggests", "indicates", "consistent with", "driven by",
    "attributed to", "mechanism", "process", "impact",
    "significance", "implication", "recommendation",
}

# Chart type keywords detected from filenames
_CHART_TYPES = {
    "overlay", "line", "trend", "timeseries",
    "boxplot", "box", "violin",
    "heatmap", "heat",
    "scatter", "correlation",
    "bar", "histogram", "dist",
    "pca", "cluster",
    "residual", "qq",
}


def _load_all_domain_expectations() -> Dict[str, Dict[str, Any]]:
    """Load all domain_*.yaml files and return {domain: expectations_dict}."""
    import yaml

    critics_dir = Path(__file__).parent.parent.parent / "critics"
    all_expectations: Dict[str, Dict[str, Any]] = {}
    for yp in sorted(critics_dir.glob("domain_*.yaml")):
        try:
            with open(yp) as fh:
                data = yaml.safe_load(fh)
            for domain, spec in (data.get("expectations") or {}).items():
                all_expectations[domain] = spec
        except Exception:
            pass
    return all_expectations


class AnalyticalDepthCritic(CriticModule):
    """Detects missing analytical dimensions and shallow findings."""

    name = "analytical_depth"
    category = CheckCategory.CONTENT_QUALITY
    stage_applicability: Set[str] = {"analysis"}
    requires_llm = True  # has optional LLM component
    requires_vlm = False

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline

    def can_run(self, ctx: CriticContext) -> bool:
        if ctx.stage_name not in self.stage_applicability:
            return False
        return getattr(ctx.run_config, "critic_analytical_depth", False)

    def evaluate(self, ctx: CriticContext) -> List[CheckResult]:
        checks: List[CheckResult] = []

        # Load analysis_summary.json
        asp = ctx.payload.get("analysis_summary_path", "")
        if not asp or not Path(asp).exists():
            return checks

        try:
            summary = json.loads(Path(asp).read_text("utf-8"))
        except Exception:
            return checks

        findings = summary.get("findings", [])
        per_group = summary.get("per_group", summary.get("per_run_per_stage", {}))

        # ── Pure-Python heuristic checks ──

        # 0. Grouping adequacy (are the key dimensions covered?)
        checks.extend(self._check_grouping_adequacy(summary, ctx))

        # 0b. Grouping compliance (do per_group keys match recommended grouping?)
        checks.extend(self._check_grouping_compliance(summary, ctx))

        # 0c. Multi-context coverage (were provided analysis contexts used?)
        checks.extend(self._check_context_coverage(summary, ctx))

        # 1. Missing group comparison
        checks.extend(self._check_group_comparison(findings, per_group))

        # 2. Homogeneous chart types
        checks.extend(self._check_chart_diversity(ctx.listing))

        # 3. Bare deviations without interpretation
        checks.extend(self._check_interpretation_depth(findings))

        # 4. Domain-specific expectations
        checks.extend(self._check_domain_expectations(summary, ctx))

        # 5. Missing outlier analysis
        checks.extend(self._check_outlier_analysis(findings, per_group))

        # 6. Chart appropriateness (are chart types suited to the data?)
        checks.extend(self._check_chart_appropriateness(
            ctx.listing, per_group, summary,
        ))

        # 7. Cross-group interaction depth
        checks.extend(self._check_interaction_depth(per_group, findings))

        # ── Optional LLM-enhanced check ──
        if ctx.llm_client or self._pipeline:
            llm_checks = self._run_llm_depth_check(summary, ctx)
            checks.extend(llm_checks)

        return checks

    def _check_grouping_adequacy(
        self, summary: Dict[str, Any], ctx: CriticContext,
    ) -> List[CheckResult]:
        """Check whether the analysis grouping covers the dataset's key dimensions.

        Loads the data_profile.json produced by the schema profiler and compares
        the per_group keys against the dimensional structure.  Flags when major
        analytical dimensions (process_phase, experimental_condition) are absent
        from the grouping, which usually means the analysis is too coarse.
        """
        # Locate data_profile.json (sibling of analysis_summary.json)
        asp = ctx.payload.get("analysis_summary_path", "")
        if not asp:
            return []
        profile_path = Path(asp).parent / "data_profile.json"
        if not profile_path.exists():
            return []

        try:
            profile_data = json.loads(profile_path.read_text("utf-8"))
        except Exception:
            return []

        dim_struct = profile_data.get("dimensional_structure", {})
        if not dim_struct:
            return []

        # Important purposes that should appear in the grouping
        important_purposes = {"process_phase", "experimental_condition", "experimental_unit"}

        # Collect available dimensions and their purposes
        available_dims: Dict[str, Dict[str, Any]] = {}
        for dim in dim_struct.get("dimensions", []):
            purpose = dim.get("semantic_purpose", "uncategorised")
            if purpose in important_purposes:
                available_dims[purpose] = {
                    "name": dim.get("name", ""),
                    "cardinality": dim.get("cardinality", 0),
                    "columns": dim.get("columns", [dim.get("name", "")]),
                }

        if not available_dims:
            return []  # no classified dimensions → can't assess

        # Determine which columns the analysis actually grouped by
        per_group = summary.get("per_group", summary.get("per_run_per_stage", {}))
        grouping_cols = summary.get("descriptive_stats", {}).get("grouping_columns", [])

        # If grouping_columns not in summary, infer from per_group key structure
        if not grouping_cols and isinstance(per_group, dict) and per_group:
            # Heuristic: look at which column names appear in the profile
            # and map them to per_group key structure
            all_col_names = {
                c.get("name", "") for c in profile_data.get("columns", [])
                if c.get("role") in ("categorical_group", "ordinal_stage")
            }
            # Check recommended grouping from profile
            rec_grp = profile_data.get("recommended_grouping")
            if isinstance(rec_grp, dict):
                grouping_cols = rec_grp.get("columns", [])

        # Which purposes are covered by the grouping columns?
        purpose_map = {}
        for c_info in profile_data.get("columns", []):
            purpose_map[c_info.get("name", "")] = c_info.get("semantic_purpose", "uncategorised")

        covered_purposes: set = set()
        for col in grouping_cols:
            p = purpose_map.get(col, "uncategorised")
            if p in important_purposes:
                covered_purposes.add(p)

        # Which important purposes are available but NOT covered?
        missing = set(available_dims.keys()) - covered_purposes

        if not missing:
            return [CheckResult(
                name="depth__grouping_adequacy",
                passed=True,
                category=CheckCategory.CONTENT_QUALITY,
                detail=(
                    f"Grouping covers key dimensions: "
                    f"{', '.join(sorted(covered_purposes))}"
                ),
            )]

        # Build a descriptive failure message
        missing_details = []
        for purpose in sorted(missing):
            dim = available_dims[purpose]
            missing_details.append(
                f"{purpose} ({dim['name']}, {dim['cardinality']} levels)"
            )

        covered_str = ", ".join(sorted(covered_purposes)) if covered_purposes else "none"
        missing_str = ", ".join(missing_details)

        # Graduated severity: 3+ missing → MUST_FIX, 1-2 → SHOULD_FIX
        # Requiring a full secondary analysis across all missing dimensions
        # is beyond what a single retry can accomplish; only hard-block when
        # the analysis omits the majority of the dimensional structure.
        severity = Severity.MUST_FIX if len(missing) >= 3 else Severity.SHOULD_FIX

        return [CheckResult(
            name="depth__grouping_inadequate",
            passed=False,
            severity=severity,
            category=CheckCategory.CONTENT_QUALITY,
            detail=(
                f"Analysis grouped by {grouping_cols or '(unknown)'} "
                f"(covers: {covered_str}) but dataset has important ungrouped "
                f"dimensions: {missing_str}. "
                f"{'A secondary analysis context is required.' if len(missing) >= 2 else 'Consider a secondary analysis.'}"
            ),
            fix_instruction=(
                f"Keep the current primary grouping key. Add a SECONDARY "
                f"analysis using the missing dimensions: {missing_str}. "
                f"Include results in a 'secondary_analysis' key or as "
                f"additional findings referencing these dimensions. "
                f"Use analysis_contexts from the data profile for guidance."
            ),
        )]

    def _check_grouping_compliance(
        self, summary: Dict[str, Any], ctx: CriticContext,
    ) -> List[CheckResult]:
        """Verify per_group keys match (or are finer-grained than) recommended grouping.

        Loads the data_profile.json and checks whether the per_group key
        structure in the analysis output contains the recommended grouping
        column values.  A mismatch (e.g. agent grouped by a different
        column entirely) is MUST_FIX; a coarser grouping (missing one
        recommended column) is SHOULD_FIX.
        """
        asp = ctx.payload.get("analysis_summary_path", "")
        if not asp:
            return []
        profile_path = Path(asp).parent / "data_profile.json"
        if not profile_path.exists():
            return []

        try:
            profile_data = json.loads(profile_path.read_text("utf-8"))
        except Exception:
            return []

        rec_grp = profile_data.get("recommended_grouping")
        if not isinstance(rec_grp, dict) or not rec_grp.get("columns"):
            return []
        rec_cols = rec_grp["columns"]

        per_group = summary.get("per_group", summary.get("per_run_per_stage", {}))
        if not isinstance(per_group, dict) or not per_group:
            return []

        # Heuristic: check if the per_group keys contain substrings matching
        # the known group values for each recommended column.
        # Look up sample values from the profile's dimensional structure.
        dim_struct = profile_data.get("dimensional_structure", {})
        dims_by_col: Dict[str, List[str]] = {}
        for dim in dim_struct.get("dimensions", []):
            for col in dim.get("columns", []):
                vals = [str(v) for v in dim.get("sample_values", [])]
                if vals:
                    dims_by_col[col] = vals

        # Check a sample of per_group keys for presence of recommended column values
        sample_keys = list(per_group.keys())[:20]
        keys_text = " ".join(sample_keys).lower()

        covered_cols = []
        missing_cols = []
        for col in rec_cols:
            sample_vals = dims_by_col.get(col, [])
            if sample_vals:
                # Check if any sample value appears in the per_group keys
                found = any(str(v).lower() in keys_text for v in sample_vals[:5])
                if found:
                    covered_cols.append(col)
                else:
                    missing_cols.append(col)
            else:
                # No sample values to check — assume covered
                covered_cols.append(col)

        if not missing_cols:
            return [CheckResult(
                name="depth__grouping_compliance",
                passed=True,
                category=CheckCategory.CONTENT_QUALITY,
                detail=(
                    f"per_group keys match recommended grouping: "
                    f"{', '.join(rec_cols)}"
                ),
            )]

        # All recommended columns missing → agent used completely wrong grouping
        if len(missing_cols) == len(rec_cols):
            return [CheckResult(
                name="depth__grouping_mismatch",
                passed=False,
                severity=Severity.MUST_FIX,
                category=CheckCategory.CONTENT_QUALITY,
                detail=(
                    f"per_group keys do not contain values from recommended "
                    f"grouping columns {rec_cols}. The agent appears to have "
                    f"used a different grouping than recommended by the profiler."
                ),
                fix_instruction=(
                    f"Regroup analysis using the compound key: "
                    f"({', '.join(rec_cols)}). Use '__' to join values in "
                    f"per_group keys. The data profile specifies this grouping."
                ),
            )]

        # Partial mismatch — coarser than recommended
        return [CheckResult(
            name="depth__grouping_partial_mismatch",
            passed=False,
            severity=Severity.SHOULD_FIX,
            category=CheckCategory.CONTENT_QUALITY,
            detail=(
                f"per_group keys cover {covered_cols} but miss "
                f"{missing_cols} from the recommended grouping."
            ),
            fix_instruction=(
                f"The recommended grouping is ({', '.join(rec_cols)}). "
                f"Include {', '.join(missing_cols)} in your compound grouping "
                f"key for finer-grained analysis."
            ),
        )]

    def _check_context_coverage(
        self, summary: Dict[str, Any], ctx: CriticContext,
    ) -> List[CheckResult]:
        """Flag when multiple analysis contexts were provided but only one used.

        Checks findings + secondary_analysis for references to context-specific
        column names. When N contexts exist but fewer than max(2, N-1) are
        referenced, flags as SHOULD_FIX.
        """
        asp = ctx.payload.get("analysis_summary_path", "")
        if not asp:
            return []
        profile_path = Path(asp).parent / "data_profile.json"
        if not profile_path.exists():
            return []

        try:
            profile_data = json.loads(profile_path.read_text("utf-8"))
        except Exception:
            return []

        dim_struct = profile_data.get("dimensional_structure", {})
        contexts = dim_struct.get("analysis_contexts", [])
        if len(contexts) < 2:
            return []  # nothing to check with 0-1 contexts

        # Split into meaningful vs hierarchy drill-down contexts.
        # Hierarchy contexts (name starts with "hierarchy_") are auto-generated
        # adjacency pairs from the schema profiler and represent data-structural
        # relationships rather than analytical questions — they are not enforced.
        meaningful_contexts = [
            c for c in contexts if not c.get("name", "").startswith("hierarchy_")
        ]
        if len(meaningful_contexts) < 2:
            return []  # not enough meaningful contexts to enforce coverage

        # Build text corpus from findings + secondary_analysis
        findings = summary.get("findings", [])
        findings_text = " ".join(finding_text(f) for f in findings).lower()
        sa = summary.get("secondary_analysis", {})
        if isinstance(sa, dict):
            sa_text = json.dumps(sa, default=str).lower()
        else:
            sa_text = ""
        corpus = findings_text + " " + sa_text

        # Check which meaningful contexts are referenced
        used_contexts = []
        unused_contexts = []
        for ac in meaningful_contexts:
            compare_col = ac.get("compare_across", "")
            group_cols = ac.get("group_by", [])
            name = ac.get("name", "")
            found = (
                compare_col.lower() in corpus
                or name.lower().replace("_", " ") in corpus
                or any(c.lower() in corpus for c in group_cols)
            )
            if found:
                used_contexts.append(name)
            else:
                unused_contexts.append(name)

        min_expected = max(2, len(meaningful_contexts) - 1)
        if len(used_contexts) >= min_expected:
            return [CheckResult(
                name="depth__context_coverage",
                passed=True,
                category=CheckCategory.CONTENT_QUALITY,
                detail=(
                    f"{len(used_contexts)}/{len(meaningful_contexts)} meaningful "
                    f"analysis contexts addressed: {', '.join(used_contexts)}"
                ),
            )]

        return [CheckResult(
            name="depth__low_context_coverage",
            passed=False,
            severity=Severity.SHOULD_FIX,
            category=CheckCategory.CONTENT_QUALITY,
            detail=(
                f"Only {len(used_contexts)}/{len(meaningful_contexts)} meaningful "
                f"analysis contexts addressed. Unused: {', '.join(unused_contexts)}."
            ),
            fix_instruction=(
                f"The data profile provides {len(meaningful_contexts)} analytical "
                f"contexts: {', '.join(ac.get('name', '?') for ac in meaningful_contexts)}. "
                f"Address at least {min_expected} — add findings or secondary "
                f"analysis for: {', '.join(unused_contexts)}."
            ),
        )]

    def _check_group_comparison(
        self, findings: List[str], per_group: Dict[str, Any],
    ) -> List[CheckResult]:
        """Flag if ≥2 groups but no statistical comparison in findings."""
        n_groups = len(per_group)
        if n_groups < 2:
            return []

        findings_text = " ".join(finding_text(f) for f in findings).lower()

        if n_groups == 2:
            # Binary design: expect t-test or pairwise comparison
            has_comparison = any(
                kw in findings_text
                for kw in ("t-test", "t_test", "ttest", "mann-whitney", "wilcoxon",
                            "cohen", "effect size", "p-value", "p_value",
                            "pairwise", "between groups", "compared to")
            )
            if has_comparison:
                return [CheckResult(
                    name="depth__group_comparison",
                    passed=True,
                    category=CheckCategory.CONTENT_QUALITY,
                    detail=f"Pairwise comparison present for {n_groups} groups",
                )]
            return [CheckResult(
                name="depth__missing_group_comparison",
                passed=False,
                severity=Severity.MUST_FIX,
                category=CheckCategory.CONTENT_QUALITY,
                detail=f"{n_groups} groups present but no t-test or pairwise comparison performed",
                fix_instruction=(
                    "Add a pairwise comparison (scipy.stats.ttest_ind for normal data, "
                    "scipy.stats.mannwhitneyu otherwise). Report p-value and effect size "
                    "(Cohen's d). Store results in analysis_summary.json."
                ),
            )]

        # 3+ groups: expect ANOVA or Kruskal-Wallis
        has_comparison = any(
            kw in findings_text
            for kw in ("anova", "kruskal", "f-test", "f_oneway", "p-value", "p_value",
                        "group comparison", "between groups", "across groups")
        )
        if has_comparison:
            return [CheckResult(
                name="depth__group_comparison",
                passed=True,
                category=CheckCategory.CONTENT_QUALITY,
                detail=f"Group comparison present for {n_groups} groups",
            )]

        return [CheckResult(
            name="depth__missing_group_comparison",
            passed=False,
            severity=Severity.MUST_FIX,
            category=CheckCategory.CONTENT_QUALITY,
            detail=f"{n_groups} groups present but no ANOVA/Kruskal-Wallis comparison performed",
            fix_instruction=(
                "Add a group comparison test (scipy.stats.f_oneway for normal data, "
                "scipy.stats.kruskal otherwise) across the groups. Report actual p-values. "
                "Store results in anova_p_values dict in analysis_summary.json."
            ),
        )]

    def _check_chart_diversity(
        self, listing: List[Dict[str, Any]],
    ) -> List[CheckResult]:
        """Flag if all plots are the same chart type."""
        png_names = [
            Path(item["path"]).stem.lower()
            for item in listing
            if isinstance(item, dict) and str(item.get("path", "")).endswith(".png")
        ]
        if len(png_names) < 3:
            return []

        # Detect chart types from filenames
        detected_types: Set[str] = set()
        for name in png_names:
            for ct in _CHART_TYPES:
                if ct in name:
                    detected_types.add(ct)
                    break

        if len(detected_types) <= 1 and len(png_names) >= 3:
            # Graduated: many homogeneous plots → MUST_FIX, few → SHOULD_FIX
            _chart_severity = (
                Severity.MUST_FIX if len(png_names) >= 5
                else Severity.SHOULD_FIX
            )
            return [CheckResult(
                name="depth__homogeneous_charts",
                passed=False,
                severity=_chart_severity,
                category=CheckCategory.CONTENT_QUALITY,
                detail=(
                    f"All {len(png_names)} plots appear to be the same chart type. "
                    f"Detected types: {detected_types or {'unknown'}}"
                ),
                fix_instruction=(
                    "Generate diverse plot types: overlay/line, boxplot/violin, "
                    "heatmap, scatter/correlation. Each should answer a distinct "
                    "analytical question."
                ),
            )]

        return [CheckResult(
            name="depth__chart_diversity",
            passed=True,
            category=CheckCategory.CONTENT_QUALITY,
            detail=f"Chart diversity: {len(detected_types)} distinct types across {len(png_names)} plots",
        )]

    def _check_interpretation_depth(
        self, findings: List[str],
    ) -> List[CheckResult]:
        """Flag findings with numeric deviations but no genuine interpretation.

        Uses LLM semantic evaluation when a critic client is available,
        falling back to a lightweight heuristic otherwise.
        """
        if not findings:
            return []

        # ── Try LLM-based semantic evaluation first ──
        if hasattr(self, '_pipeline') and self._pipeline._critic_client is not None:
            return self._llm_interpretation_check(findings)

        # ── Fallback: lightweight heuristic (less strict than keyword matching) ──
        bare_count = 0
        for f in findings:
            ft = finding_text(f)
            f_lower = ft.lower()
            has_number = bool(re.search(r'\d+\.?\d*%', ft))
            # Check for ANY explanatory language, not just vocabulary tokens
            has_interpretation = (
                any(word in f_lower for word in _INTERPRETATION_VOCABULARY)
                or len(f_lower.split()) > 15  # longer findings likely contain explanation
                or any(phrase in f_lower for phrase in [
                    " because ", " due to ", " which ", " this ",
                    " may ", " could ", " likely ", " potential",
                    " check ", " verify ", " investigate",
                ])
            )
            if has_number and not has_interpretation:
                bare_count += 1

        if bare_count == 0:
            return [CheckResult(
                name="depth__interpretation_depth",
                passed=True,
                category=CheckCategory.CONTENT_QUALITY,
                detail="All findings with numeric values include interpretive context",
            )]

        # Graduated: majority bare → MUST_FIX, minority → SHOULD_FIX
        _interp_severity = (
            Severity.MUST_FIX if bare_count / len(findings) > 0.5
            else Severity.SHOULD_FIX
        )
        return [CheckResult(
            name="depth__bare_deviations",
            passed=False,
            severity=_interp_severity,
            category=CheckCategory.CONTENT_QUALITY,
            detail=(
                f"{bare_count}/{len(findings)} findings state numeric deviations "
                "without biological significance or root cause interpretation"
            ),
            fix_instruction=(
                "Each finding with a numeric deviation should explain: "
                "(1) what the deviation means in domain context, "
                "(2) a possible root cause or process explanation, "
                "(3) a recommended action or investigation."
            ),
        )]

    def _llm_interpretation_check(
        self, findings: List[str],
    ) -> List[CheckResult]:
        """LLM-based semantic interpretation depth check."""
        prompt = (
            "You are evaluating findings from a biologics data analysis pipeline.\n"
            "The agent model is a 27B parameter model (Qwen3.5-27B).\n\n"
            "For each finding below, assess whether it includes genuine domain\n"
            "interpretation — not just a numeric deviation, but an explanation of\n"
            "what it means for product quality or the process.\n\n"
            "A finding PASSES only if it contains ALL of:\n"
            "  (1) a quantitative observation with specific values AND\n"
            "  (2) domain-specific interpretation explaining what the observation\n"
            "      means for product quality, process performance, or scientific\n"
            "      understanding (not just restating the number in words)\n"
            "A finding FAILS if:\n"
            "  - It states a deviation without explaining significance\n"
            "  - It restates a number in words without domain reasoning\n"
            "  - It lacks specific group/run references\n\n"
            "Be strict — superficial restatements of numbers do not constitute\n"
            "interpretation. Require genuine domain reasoning.\n\n"
            "Return JSON: {\"bare_count\": N, \"total\": N, \"detail\": \"...\"}\n"
            "No code fences."
        )
        findings_text = "\n".join(f"  {i+1}. {finding_text(f)}" for i, f in enumerate(findings[:8]))
        try:
            response = self._pipeline._critic_client.chat.completions.create(
                model=self._pipeline._critic_model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Findings:\n{findings_text}"},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            reply = strip_think_tokens(response.choices[0].message.content or "")
            parsed = parse_json_tolerant(reply)
            if not isinstance(parsed, dict):
                return []

            bare_count = int(parsed.get("bare_count", 0))
            if bare_count == 0:
                return [CheckResult(
                    name="depth__interpretation_depth",
                    passed=True,
                    category=CheckCategory.CONTENT_QUALITY,
                    detail=parsed.get("detail", "All findings include interpretive context"),
                )]
            return [CheckResult(
                name="depth__bare_deviations",
                passed=False,
                severity=Severity.SHOULD_FIX,
                category=CheckCategory.CONTENT_QUALITY,
                detail=parsed.get("detail", f"{bare_count}/{len(findings)} findings lack interpretation"),
                fix_instruction=(
                    "Each finding with a numeric deviation should explain: "
                    "(1) what the deviation means in domain context, "
                    "(2) a possible root cause or process explanation, "
                    "(3) a recommended action or investigation."
                ),
            )]
        except Exception as exc:
            logger.warning("LLM interpretation check failed, skipping: %s", exc)
            return []

    def _check_domain_expectations(
        self, summary: Dict[str, Any], ctx: CriticContext,
    ) -> List[CheckResult]:
        """Check domain-specific required analyses.

        Domain detection uses three cascading signals:
        1. ``domain_hints`` already present in the analysis summary.
        2. YAML-driven ``detection_keywords`` matched against column names
           in the cleaned dataset (no hardcoded keywords in Python).
        3. Falls back gracefully — if neither signal fires, no domain
           checks are emitted.
        """
        checks: List[CheckResult] = []
        all_expectations = _load_all_domain_expectations()
        if not all_expectations:
            return checks

        # --- Resolve domain flags ---
        domain_flags: Dict[str, bool] = dict(summary.get("domain_hints", {}))

        if not domain_flags:
            # Fallback: match YAML detection_keywords against column names
            cleaned_path = ctx.payload.get("cleaned_path", "")
            if cleaned_path and Path(cleaned_path).exists():
                try:
                    import pandas as pd
                    df = pd.read_parquet(cleaned_path, columns=None)
                    cols_lower = [c.lower() for c in df.columns]
                    col_text = " ".join(cols_lower)
                    for domain, spec in all_expectations.items():
                        keywords = spec.get("detection_keywords", [])
                        if any(kw.lower() in col_text for kw in keywords):
                            domain_flags[domain] = True
                except Exception:
                    pass

        # --- Evaluate required analyses per detected domain ---
        findings_text = " ".join(str(f) for f in summary.get("findings", [])).lower()

        for domain, is_present in domain_flags.items():
            if not is_present:
                continue
            spec = all_expectations.get(domain, {})
            required = spec.get("required_analyses", [])

            for req in required:
                analysis_name = req.get("analysis", "")
                if analysis_name.lower().replace("_", " ") in findings_text:
                    checks.append(CheckResult(
                        name=f"depth__domain_{analysis_name}",
                        passed=True,
                        category=CheckCategory.CONTENT_QUALITY,
                        detail=f"Domain analysis present: {analysis_name}",
                    ))
                else:
                    checks.append(CheckResult(
                        name=f"depth__missing_domain_analysis",
                        passed=False,
                        severity=Severity.MUST_FIX,
                        category=CheckCategory.CONTENT_QUALITY,
                        detail=f"{domain} data detected but missing required analysis: {analysis_name}",
                        fix_instruction=(
                            f"Add {analysis_name} analysis for the {domain} data. "
                            f"This is a standard analytical requirement for this data type."
                        ),
                        ref=analysis_name,
                    ))

        return checks

    def _check_outlier_analysis(
        self, findings: List[str], per_group: Dict[str, Any],
    ) -> List[CheckResult]:
        """Flag if per-group metrics exist but no outlier detection."""
        if len(per_group) < 3:
            return []

        findings_text = " ".join(finding_text(f) for f in findings).lower()
        has_outlier = any(
            kw in findings_text
            for kw in ("outlier", "z-score", "iqr", "interquartile", "extreme",
                        "deviation beyond", "outside range")
        )
        if has_outlier:
            return [CheckResult(
                name="depth__outlier_analysis",
                passed=True,
                category=CheckCategory.CONTENT_QUALITY,
                detail="Outlier analysis present in findings",
            )]

        return [CheckResult(
            name="depth__no_outlier_analysis",
            passed=False,
            severity=Severity.SHOULD_FIX,
            category=CheckCategory.CONTENT_QUALITY,
            detail=(
                f"Per-group metrics for {len(per_group)} groups but no outlier "
                "detection (Z-score, IQR, etc.) in findings"
            ),
            fix_instruction=(
                "Add outlier detection across groups. Flag any group whose key "
                "metrics fall outside 2 standard deviations or 1.5×IQR from the "
                "group median. Report which groups are outliers and why."
            ),
        )]

    def _check_chart_appropriateness(
        self,
        listing: List[Dict[str, Any]],
        per_group: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> List[CheckResult]:
        """Flag when more informative chart types are available but unused."""
        png_names = [
            Path(item["path"]).stem.lower()
            for item in listing
            if isinstance(item, dict) and str(item.get("path", "")).endswith(".png")
        ]
        if not png_names:
            return []

        detected_types: Set[str] = set()
        for name in png_names:
            for ct in _CHART_TYPES:
                if ct in name:
                    detected_types.add(ct)
                    break

        checks: List[CheckResult] = []
        n_groups = len(per_group)

        # Multi-group comparison benefits from heatmap/box/violin
        if n_groups >= 5 and not (detected_types & {"heatmap", "box", "violin"}):
            checks.append(CheckResult(
                name="depth__missing_multigroup_chart",
                passed=False,
                severity=Severity.SHOULD_FIX,
                category=CheckCategory.CONTENT_QUALITY,
                detail=(
                    f"{n_groups} groups but no heatmap, box, or violin plot "
                    "for multi-group comparison"
                ),
                fix_instruction=(
                    "Add a heatmap or box/violin plot for multi-group "
                    "comparison of key metrics."
                ),
            ))

        # High-dimensional data benefits from correlation/PCA
        dp = summary.get("data_profile") or {}
        col_roles = dp.get("column_roles", {})
        n_numeric = len(col_roles.get("continuous_measurement", []))
        if n_numeric > 10 and not (
            detected_types & {"scatter", "correlation", "pca", "cluster"}
        ):
            checks.append(CheckResult(
                name="depth__missing_dimensionality_chart",
                passed=False,
                severity=Severity.SHOULD_FIX,
                category=CheckCategory.CONTENT_QUALITY,
                detail=(
                    f"{n_numeric} numeric columns but no scatter, correlation "
                    "matrix, or PCA plot for dimensionality exploration"
                ),
                fix_instruction=(
                    "Add a scatter/correlation matrix or PCA plot to explore "
                    "relationships across numeric columns."
                ),
            ))

        return checks

    def _check_interaction_depth(
        self,
        per_group: Dict[str, Any],
        findings: List[str],
    ) -> List[CheckResult]:
        """Flag when multi-group data lacks cross-group interaction analysis."""
        if len(per_group) < 4:
            return []

        findings_text = " ".join(finding_text(f) for f in findings).lower()
        _interaction_keywords = (
            "interaction", "cross-group", "between groups",
            "group x", "two-way", "factorial", "moderation",
            "depends on", "varies across", "differs by",
        )
        has_interaction = any(kw in findings_text for kw in _interaction_keywords)
        if has_interaction:
            return [CheckResult(
                name="depth__interaction_analysis",
                passed=True,
                category=CheckCategory.CONTENT_QUALITY,
                detail="Cross-group interaction analysis present",
            )]

        return [CheckResult(
            name="depth__no_interaction_analysis",
            passed=False,
            severity=Severity.SHOULD_FIX,
            category=CheckCategory.CONTENT_QUALITY,
            detail=(
                f"{len(per_group)} groups but no cross-group interaction "
                "analysis in findings"
            ),
            fix_instruction=(
                "Explore whether the effect of one grouping variable varies "
                "across levels of another (e.g. does column type performance "
                "differ between runs?). Include interaction findings."
            ),
        )]

    def _run_llm_depth_check(
        self, summary: Dict[str, Any], ctx: CriticContext,
    ) -> List[CheckResult]:
        """Optional LLM-enhanced depth assessment."""
        try:
            from prompts import ANALYTICAL_DEPTH_RUBRIC
        except ImportError:
            return []

        findings = summary.get("findings", [])
        per_group = summary.get("per_group", summary.get("per_run_per_stage", {}))
        if not findings:
            return []

        from prompts import STAGE_CRITIC_PROMPT
        prompt = STAGE_CRITIC_PROMPT.format(
            stage_name="analysis_depth",
            stage_rubric=ANALYTICAL_DEPTH_RUBRIC,
        )

        input_text = (
            f"Findings ({len(findings)} total):\n"
            + "\n".join(f"  - {finding_text(f)}" for f in findings[:8])
            + f"\n\nGroups: {len(per_group)}"
            + f"\nGroup names (sample): {list(per_group.keys())[:5]}"
        )

        try:
            if self._pipeline._critic_client is not None:
                # ── OpenRouter path (no local GPU contention) ──
                response = self._pipeline._critic_client.chat.completions.create(
                    model=self._pipeline._critic_model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": input_text[:4000]},
                    ],
                    temperature=0.0,
                    max_tokens=2048,
                )
                reply = response.choices[0].message.content or ""
            else:
                # ── Fallback: local vLLM via OpenAIWrapper ──
                from autogen.oai import OpenAIWrapper

                cfg = self._pipeline._base_config_dict()
                cfg["temperature"] = 0.0
                for _entry in cfg.get("config_list", []):
                    _entry["timeout"] = 600  # vLLM fallback: must wait out GroupChat queue
                client = OpenAIWrapper(**cfg)
                response = client.create(messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": input_text[:4000]},
                ])
                reply = strip_think_tokens(
                    response.choices[0].message.content or ""
                )
            parsed = parse_json_tolerant(reply)
            if not isinstance(parsed, list):
                return []

            checks: List[CheckResult] = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                name = f"depth__{item.get('criterion', 'unknown')}"
                passed = bool(item.get("passed", True))
                sev_str = item.get("severity")
                severity = None
                if not passed and sev_str:
                    severity = (
                        Severity.MUST_FIX if sev_str == "must_fix"
                        else Severity.SHOULD_FIX
                    )
                checks.append(CheckResult(
                    name=name,
                    passed=passed,
                    severity=severity,
                    category=CheckCategory.CONTENT_QUALITY,
                    detail=str(item.get("detail", "")),
                    fix_instruction=str(item.get("fix_instruction", "")),
                ))
            return checks

        except Exception as exc:
            logger.warning("Analytical depth LLM check failed: %s", exc)
            return []
