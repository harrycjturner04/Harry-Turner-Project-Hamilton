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

from tools import CheckCategory, CheckResult, Severity, finding_text
from critics.base import CriticContext, CriticModule

logger = logging.getLogger("captain_pipeline")

# Single-letter ML metric abbreviations that match the group-ref regex but are
# never used as per-group identifiers.  'F1' appears in findings as the F1-score
# (precision-recall harmonic mean); excluding it prevents false positives.
_ML_METRIC_TOKENS: Set[str] = {"F1"}


def _collect_group_keys(d: Any, depth: int = 0) -> Set[str]:
    """Recursively collect all dict keys from a nested per_group structure.

    Handles both flat structures ({group_key: metrics_dict}) and nested ones
    ({task_name: {group_key: metrics_dict}}) up to three levels deep.
    """
    if not isinstance(d, dict) or depth > 3:
        return set()
    keys: Set[str] = set(d.keys())
    for v in d.values():
        if isinstance(v, dict):
            keys.update(_collect_group_keys(v, depth + 1))
    return keys


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

        # 7. Feature importance completeness
        checks.extend(self._check_feature_importance_present(summary))

        # 8. ML accuracy floor
        checks.extend(self._check_ml_accuracy_floor(summary))

        # 9. Feature stability for small-N regression
        checks.extend(self._check_feature_stability_regression(summary, ctx))

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
        """Cross-reference claims in findings against per-group data.

        Handles both flat per_group structures ({group_key: metrics}) and nested
        ones ({task_name: {group_key: metrics}}) by collecting all keys at every
        level.  Known ML metric abbreviations (e.g. 'F1' for F1-score) are
        excluded to prevent false positives.
        """
        if not findings or not per_group:
            return []

        # Collect keys at all nesting levels so both flat and nested per_group
        # structures are handled correctly.
        group_keys = _collect_group_keys(per_group)

        # Track unique (finding_index, ref) pairs to avoid double-counting when
        # the same token appears multiple times within one finding.
        seen: Set[tuple] = set()
        mismatches: List[str] = []

        for i, finding in enumerate(findings):
            f_str = finding_text(finding)
            # Look for group references like "Run R4", "R16__E4", etc.
            refs = set(re.findall(r'(?:Run\s+)?([A-Z]\d+(?:__[A-Z]\d+)*)', f_str))
            for ref in refs:
                # Skip known ML metric abbreviations (F1-score, etc.)
                if ref in _ML_METRIC_TOKENS:
                    continue
                pair = (i, ref)
                if pair in seen:
                    continue
                seen.add(pair)
                # Check if this reference matches any group key (substring match)
                matched = any(ref in gk for gk in group_keys)
                if not matched:
                    mismatches.append(f"Finding {i+1} references '{ref}' not found in per_group keys")

        if not mismatches:
            return [CheckResult(
                name="exec__finding_data_mismatch",
                passed=True,
                category=CheckCategory.CONTENT_QUALITY,
                detail="Finding group references match per_group keys",
            )]

        # Only flag when mismatches exceed half the findings — avoids noise from
        # analyses that use a mix of group and non-group identifiers.
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
        return [CheckResult(
            name="exec__finding_data_mismatch",
            passed=True,
            category=CheckCategory.CONTENT_QUALITY,
            detail=(
                f"Finding-data mismatches below noise threshold "
                f"({len(mismatches)}/{len(findings)})"
            ),
        )]

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

        # Build search corpus: finding text + figure_ref/figure fields from dict findings
        _parts = [finding_text(f) for f in findings]
        for f in findings:
            if isinstance(f, dict):
                _parts.append(str(f.get("figure_ref") or ""))
                _parts.append(str(f.get("figure") or ""))
        findings_text = " ".join(_parts).lower()
        # Look for figure references like "Figure 1", "fig_01", plot filenames
        missing_refs: List[str] = []
        for png in png_names:
            stem = Path(png).stem
            # Check if the plot is referenced in findings
            if stem not in findings_text and png not in findings_text:
                missing_refs.append(png)

        # Require ≥90% of plots to be referenced in findings.
        if len(missing_refs) > len(png_names) * 0.1:
            _all_missing = ", ".join(sorted(missing_refs))
            return [CheckResult(
                name="exec__plot_finding_alignment",
                passed=False,
                severity=Severity.MUST_FIX,
                category=CheckCategory.CONTENT_QUALITY,
                detail=(
                    f"{len(missing_refs)}/{len(png_names)} plots are not referenced "
                    "in any finding"
                ),
                fix_instruction=(
                    f"{len(missing_refs)} of {len(png_names)} plots are not referenced "
                    f"in any finding: {_all_missing}. "
                    "For EACH plot, add a 'figure_ref' key to the most relevant existing "
                    "finding dict (e.g. update {\"text\": \"...\", \"figure_ref\": \"plot.png\"}). "
                    "You do NOT need to add new findings — updating figure_ref on existing "
                    "findings is sufficient. The model is responsible for assigning each "
                    "plot to the finding it supports."
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
            # Large-sample guard: p=0.0 is legitimate float64 underflow when
            # group sizes exceed ~100K rows.  Don't flag as copy-paste in that case.
            if p_values[0] == 0.0:
                per_group = summary.get("per_group", summary.get("per_run_per_stage", {}))
                if isinstance(per_group, dict) and per_group:
                    _sizes = [
                        v.get("row_count", 0)
                        for v in per_group.values()
                        if isinstance(v, dict) and v.get("row_count", 0) > 0
                    ]
                    if _sizes:
                        _median_n = sorted(_sizes)[len(_sizes) // 2]
                        if _median_n > 10_000:
                            return []  # p=0.0 expected for large N

            return [CheckResult(
                name="exec__duplicate_pvalues",
                passed=False,
                severity=Severity.MUST_FIX,
                category=CheckCategory.CONTENT_QUALITY,
                detail=(
                    f"All {len(p_values)} p-values are identical ({p_values[0]}) "
                    "— copy-paste or computation error"
                ),
                fix_instruction=(
                    "All reported p-values are identical, indicating they were "
                    "copy-pasted rather than independently computed.  For each "
                    "test, recompute the p-value from its specific data subset "
                    "(e.g. scipy.stats.kruskal(*[group_vals...]) or "
                    "scipy.stats.f_oneway(*[group_vals...])) and report the "
                    "actual returned p-value alongside the sample size (n) for "
                    "each subset.  Do NOT reuse a single p-value across findings."
                ),
            )]

        return [CheckResult(
            name="exec__duplicate_pvalues",
            passed=True,
            category=CheckCategory.CONTENT_QUALITY,
            detail=f"P-values show expected variation ({len(unique_p)} unique across {len(p_values)} values)",
        )]

    def _check_feature_importance_present(
        self, summary: Dict[str, Any],
    ) -> List[CheckResult]:
        """Flag empty or absent feature_importance when ML modelling was performed."""
        ml = summary.get("ml_modeling")
        if not isinstance(ml, dict):
            return []  # No ml_modeling block — nothing to check

        # Check whether any task keys exist (task_1_*, task_2_*, ...)
        task_keys = [k for k in ml if k.startswith("task_")]
        if not task_keys:
            return []  # No tasks run

        fi = ml.get("feature_importance")
        if fi is None or (isinstance(fi, (dict, list)) and len(fi) == 0):
            return [CheckResult(
                name="exec__feature_importance_empty",
                passed=False,
                severity=Severity.MUST_FIX,
                category=CheckCategory.CONTENT_QUALITY,
                detail=(
                    f"ml_modeling has {len(task_keys)} task(s) but feature_importance "
                    f"is {'absent' if fi is None else 'empty'}"
                ),
                fix_instruction=(
                    "The ml_modeler ran tasks but produced no aggregate feature_importance. "
                    "After fitting each model, compute permutation_importance (sklearn) and "
                    "store a {feature_name: score} dict under ml_modeling['feature_importance'] "
                    "in analysis_summary.json."
                ),
            )]

        return [CheckResult(
            name="exec__feature_importance_present",
            passed=True,
            category=CheckCategory.CONTENT_QUALITY,
            detail=f"feature_importance present ({len(fi)} features)",
        )]

    def _check_feature_stability_regression(
        self, summary: Dict[str, Any], ctx: CriticContext,
    ) -> List[CheckResult]:
        """For small-N regression tasks, check that feature_stability is reported.

        Small datasets (N<200) produce unreliable single-split R² estimates.
        Feature stability (% of CV folds a feature was selected) is the most
        actionable output for these settings.  This is a SHOULD_FIX only —
        the model may still be valid but the output is less informative.
        """
        ml = summary.get("ml_modeling")
        if not isinstance(ml, dict):
            return []

        # Check if any regression tasks were run
        regression_tasks = [
            k for k, v in ml.items()
            if k.startswith("task_") and isinstance(v, dict)
            and v.get("task_type") in ("regression", "supervised_regression")
        ]
        if not regression_tasks:
            return []

        # Estimate dataset size from per_group (use row counts if available)
        per_group = summary.get("per_group", summary.get("per_run_per_stage", {}))
        total_rows = 0
        if isinstance(per_group, dict):
            for v in per_group.values():
                if isinstance(v, dict):
                    total_rows += v.get("row_count", 0)

        # Only apply check when N is small (< 200) or unknown (0 = treat as small)
        if total_rows >= 200:
            return []

        # Check for feature_stability in ml_modeling
        stability = ml.get("feature_stability")
        if stability and isinstance(stability, dict) and len(stability) > 0:
            return [CheckResult(
                name="exec__feature_stability_present",
                passed=True,
                category=CheckCategory.CONTENT_QUALITY,
                detail=f"feature_stability reported for {len(stability)} features (small-N regression)",
            )]

        return [CheckResult(
            name="exec__feature_stability_missing",
            passed=False,
            severity=Severity.SHOULD_FIX,
            category=CheckCategory.CONTENT_QUALITY,
            detail=(
                f"Regression task(s) run on small dataset (N≈{total_rows or 'unknown'}) "
                f"but ml_modeling['feature_stability'] is absent or empty"
            ),
            fix_instruction=(
                "For small datasets (N<200), report feature_stability: the fraction of "
                "CV folds in which each feature had non-negligible importance. "
                "Implement cross-validated feature selection (e.g. k-fold with Ridge "
                "regularisation) and record for each feature: "
                "ml_modeling['feature_stability'] = {feature_name: fraction_of_folds_selected}. "
                "Also report uncertainty on R²/RMSE using CV fold std or bootstrap resampling."
            ),
        )]

    def _check_ml_accuracy_floor(
        self, summary: Dict[str, Any],
    ) -> List[CheckResult]:
        """Flag classification tasks at or near the random-baseline accuracy."""
        ml = summary.get("ml_modeling")
        if not isinstance(ml, dict):
            return []

        low_accuracy: List[str] = []
        for key, metrics in ml.items():
            if not key.startswith("task_") or not isinstance(metrics, dict):
                continue
            acc = metrics.get("accuracy")
            n_classes = metrics.get("n_classes")
            if acc is None or not isinstance(acc, (int, float)):
                continue
            if not isinstance(n_classes, int) or n_classes < 2:
                continue
            baseline = 1.0 / n_classes
            if acc <= baseline + 0.05:  # within 5pp of random
                low_accuracy.append(
                    f"{key}: accuracy={acc:.1%} "
                    f"(random≈{baseline:.1%} for {n_classes} classes)"
                )

        if not low_accuracy:
            return []

        return [CheckResult(
            name="exec__ml_accuracy_floor",
            passed=False,
            severity=Severity.SHOULD_FIX,
            category=CheckCategory.CONTENT_QUALITY,
            detail=(
                f"{len(low_accuracy)} classification task(s) at or near random baseline: "
                + "; ".join(low_accuracy)
            ),
            fix_instruction=(
                "Classification accuracy is near the random baseline — the model is not "
                "generalising. Consider: (1) verifying class balance in the target column, "
                "(2) using different features or grouping, (3) increasing training data via "
                "a coarser aggregation, or (4) reporting that this target is not predictable "
                "from the available features and explaining why."
            ),
        )]
