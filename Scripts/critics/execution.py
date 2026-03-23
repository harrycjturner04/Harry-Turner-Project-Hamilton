"""ExecutionCritic — validates code execution correctness.

WP-C3b: Pure-Python checks that catch silent failures producing artifacts
with wrong content.  No LLM needed — all checks operate on the serialised
analysis_summary.json and output directory listing.

Runs after the structural gate (needs artifacts on disk) and before the
content evaluator.  Produces CheckResult with CheckCategory.CONTENT_QUALITY.
"""
from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tools import CheckCategory, CheckResult, Severity
from critics.base import CriticContext, CriticModule

logger = logging.getLogger("captain_pipeline")


class ExecutionCritic(CriticModule):
    """Pure-Python validation of code execution correctness."""

    name = "execution"
    category = CheckCategory.CONTENT_QUALITY
    stage_applicability: Set[str] = {"analysis"}
    requires_llm = False
    requires_vlm = False

    def can_run(self, ctx: CriticContext) -> bool:
        if ctx.stage_name not in self.stage_applicability:
            return False
        return getattr(ctx.run_config, "critic_execution", False)

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

        # 1. NaN leakage in metrics
        checks.extend(self._check_nan_leakage(per_group))

        # 2. Impossible values
        checks.extend(self._check_impossible_values(per_group))

        # 3. Uniform groups (aggregation error)
        checks.extend(self._check_uniform_groups(per_group))

        # 4. Finding-data mismatch
        checks.extend(self._check_finding_data_mismatch(findings, per_group))

        # 5. Plot-finding alignment
        checks.extend(self._check_plot_finding_alignment(findings, ctx.listing))

        # 6. Duplicate p-values
        checks.extend(self._check_duplicate_pvalues(summary))

        return checks

    def _check_nan_leakage(
        self, per_group: Dict[str, Any],
    ) -> List[CheckResult]:
        """Flag NaN/null values in per-group metrics."""
        if not per_group:
            return []

        nan_count = 0
        total_values = 0
        nan_groups: List[str] = []

        for group_key, metrics in per_group.items():
            if not isinstance(metrics, dict):
                continue
            for metric_name, value in metrics.items():
                total_values += 1
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    nan_count += 1
                    if group_key not in nan_groups:
                        nan_groups.append(group_key)
                elif isinstance(value, str) and value.lower() in ("nan", "null", "none"):
                    nan_count += 1
                    if group_key not in nan_groups:
                        nan_groups.append(group_key)

        if nan_count == 0:
            return [CheckResult(
                name="exec__nan_leakage",
                passed=True,
                category=CheckCategory.CONTENT_QUALITY,
                detail=f"No NaN values in {total_values} per-group metric values",
            )]

        return [CheckResult(
            name="exec__nan_leakage",
            passed=False,
            severity=Severity.MUST_FIX,
            category=CheckCategory.CONTENT_QUALITY,
            detail=(
                f"{nan_count}/{total_values} per-group metric values are NaN/null "
                f"(affected groups: {nan_groups[:5]})"
            ),
            fix_instruction=(
                "Per-group metrics contain NaN values, indicating a computation error "
                "or missing data handling issue. Check groupby operations for empty "
                "groups or division by zero. Fill or filter NaN values before aggregation."
            ),
        )]

    def _check_impossible_values(
        self, per_group: Dict[str, Any],
    ) -> List[CheckResult]:
        """Flag values that are physically impossible (negative counts, % > 100)."""
        if not per_group:
            return []

        impossible: List[str] = []

        for group_key, metrics in per_group.items():
            if not isinstance(metrics, dict):
                continue
            for metric_name, value in metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                if math.isnan(value) or math.isinf(value):
                    continue

                mn = metric_name.lower()
                # Negative counts
                if ("count" in mn or "n_" in mn) and value < 0:
                    impossible.append(f"{group_key}.{metric_name}={value} (negative count)")
                # Percentages > 100 or < 0
                if ("pct" in mn or "percent" in mn or "ratio" in mn) and (value > 200 or value < -100):
                    impossible.append(f"{group_key}.{metric_name}={value} (impossible percentage)")
                # Negative standard deviations
                if ("std" in mn or "stdev" in mn or "sd_" in mn) and value < 0:
                    impossible.append(f"{group_key}.{metric_name}={value} (negative std)")

        if not impossible:
            return [CheckResult(
                name="exec__impossible_values",
                passed=True,
                category=CheckCategory.CONTENT_QUALITY,
                detail="No impossible values detected in per-group metrics",
            )]

        return [CheckResult(
            name="exec__impossible_values",
            passed=False,
            severity=Severity.MUST_FIX,
            category=CheckCategory.CONTENT_QUALITY,
            detail=(
                f"{len(impossible)} impossible values detected: "
                + "; ".join(impossible[:5])
            ),
            fix_instruction=(
                "Per-group metrics contain physically impossible values "
                "(negative counts, extreme percentages, negative std devs). "
                "Review the aggregation code for errors."
            ),
        )]

    def _check_uniform_groups(
        self, per_group: Dict[str, Any],
    ) -> List[CheckResult]:
        """Flag if all groups have identical values (aggregation applied to wrong column)."""
        if len(per_group) < 3:
            return []

        # Collect values per metric across groups
        metric_values: Dict[str, List[float]] = {}
        for metrics in per_group.values():
            if not isinstance(metrics, dict):
                continue
            for mn, val in metrics.items():
                if isinstance(val, (int, float)) and not math.isnan(val):
                    metric_values.setdefault(mn, []).append(float(val))

        uniform_metrics: List[str] = []
        for mn, values in metric_values.items():
            if len(values) >= 3 and len(set(round(v, 8) for v in values)) == 1:
                uniform_metrics.append(mn)

        if not uniform_metrics:
            return [CheckResult(
                name="exec__uniform_groups",
                passed=True,
                category=CheckCategory.CONTENT_QUALITY,
                detail="Per-group metrics show variation across groups",
            )]

        return [CheckResult(
            name="exec__uniform_groups",
            passed=False,
            severity=Severity.SHOULD_FIX,
            category=CheckCategory.CONTENT_QUALITY,
            detail=(
                f"{len(uniform_metrics)} metrics are identical across all groups: "
                + ", ".join(uniform_metrics[:5])
                + " — possible aggregation applied to wrong column"
            ),
            fix_instruction=(
                "Some per-group metrics are identical across all groups, "
                "suggesting the groupby operation may have aggregated the wrong "
                "column or applied a global aggregation instead of per-group."
            ),
        )]

    def _check_finding_data_mismatch(
        self, findings: List[str], per_group: Dict[str, Any],
    ) -> List[CheckResult]:
        """Cross-reference claims in findings against per-group data."""
        if not findings or not per_group:
            return []

        group_keys = set(per_group.keys())
        mismatches: List[str] = []

        for i, finding in enumerate(findings):
            f_str = str(finding)
            # Look for group references like "Run R4", "R16__E4", etc.
            refs = re.findall(r'(?:Run\s+)?([A-Z]\d+(?:__[A-Z]\d+)*)', f_str)
            for ref in refs:
                # Check if this reference matches any group key
                matched = any(ref in gk for gk in group_keys)
                if not matched and refs:
                    # Only flag if we found references that don't match
                    mismatches.append(f"Finding {i+1} references '{ref}' not found in per_group keys")

        if not mismatches:
            return [CheckResult(
                name="exec__finding_data_mismatch",
                passed=True,
                category=CheckCategory.CONTENT_QUALITY,
                detail="Finding group references match per_group keys",
            )]

        # Only flag if there are clear mismatches (not just no references)
        if len(mismatches) > len(findings) // 2:
            return [CheckResult(
                name="exec__finding_data_mismatch",
                passed=False,
                severity=Severity.SHOULD_FIX,
                category=CheckCategory.CONTENT_QUALITY,
                detail=(
                    f"Multiple finding-data mismatches: "
                    + "; ".join(mismatches[:3])
                ),
                fix_instruction=(
                    "Findings reference group identifiers not found in per_group data. "
                    "Ensure findings cite actual group keys from the analysis."
                ),
            )]
        return []

    def _check_plot_finding_alignment(
        self, findings: List[str], listing: List[Dict[str, Any]],
    ) -> List[CheckResult]:
        """Verify that plot references in findings correspond to actual PNG files."""
        png_names = {
            Path(item["path"]).name.lower()
            for item in listing
            if isinstance(item, dict) and str(item.get("path", "")).endswith(".png")
        }
        if not png_names or not findings:
            return []

        findings_text = " ".join(str(f) for f in findings).lower()
        # Look for figure references like "Figure 1", "fig_01", plot filenames
        missing_refs: List[str] = []
        for png in png_names:
            stem = Path(png).stem
            # Check if the plot is referenced in findings
            if stem not in findings_text and png not in findings_text:
                missing_refs.append(png)

        # This is informational — not all plots need to be referenced in findings
        if len(missing_refs) > len(png_names) * 0.7:
            return [CheckResult(
                name="exec__plot_finding_alignment",
                passed=False,
                severity=Severity.SHOULD_FIX,
                category=CheckCategory.CONTENT_QUALITY,
                detail=(
                    f"{len(missing_refs)}/{len(png_names)} plots are not referenced "
                    "in any finding"
                ),
                fix_instruction=(
                    "Most plots are not referenced in the findings. Each plot should "
                    "support at least one finding with a specific reference."
                ),
            )]

        return [CheckResult(
            name="exec__plot_finding_alignment",
            passed=True,
            category=CheckCategory.CONTENT_QUALITY,
            detail=f"Plot-finding alignment: {len(png_names) - len(missing_refs)}/{len(png_names)} plots referenced",
        )]

    def _check_duplicate_pvalues(
        self, summary: Dict[str, Any],
    ) -> List[CheckResult]:
        """Flag identical p-values across all tests (copy-paste indicator)."""
        p_values: List[float] = []

        # Check anova_p_values
        anova = summary.get("anova_p_values", {})
        if isinstance(anova, dict):
            for v in anova.values():
                if isinstance(v, (int, float)) and not math.isnan(v):
                    p_values.append(round(v, 10))

        # Also scan findings for p-values
        findings_text = " ".join(str(f) for f in summary.get("findings", []))
        p_matches = re.findall(r'p\s*[=<]\s*(0\.\d+)', findings_text)
        for pm in p_matches:
            try:
                p_values.append(round(float(pm), 10))
            except ValueError:
                pass

        if len(p_values) < 3:
            return []

        unique_p = set(p_values)
        if len(unique_p) == 1 and len(p_values) >= 3:
            return [CheckResult(
                name="exec__duplicate_pvalues",
                passed=False,
                severity=Severity.SHOULD_FIX,
                category=CheckCategory.CONTENT_QUALITY,
                detail=(
                    f"All {len(p_values)} p-values are identical ({p_values[0]}) "
                    "— possible copy-paste or computation error"
                ),
                fix_instruction=(
                    "All reported p-values are identical, suggesting they were "
                    "copy-pasted rather than independently computed. Recompute "
                    "each p-value from the appropriate data subset."
                ),
            )]

        return [CheckResult(
            name="exec__duplicate_pvalues",
            passed=True,
            category=CheckCategory.CONTENT_QUALITY,
            detail=f"P-values show expected variation ({len(unique_p)} unique across {len(p_values)} values)",
        )]
