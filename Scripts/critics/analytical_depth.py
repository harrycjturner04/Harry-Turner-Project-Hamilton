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


def _load_domain_expectations(domain: str) -> Dict[str, Any]:
    """Load domain-specific analytical expectations from YAML."""
    import yaml

    yaml_paths = [
        Path(__file__).parent.parent.parent / "critics" / "domain_biologics.yaml",
        Path(__file__).parent.parent.parent / "critics" / f"domain_{domain}.yaml",
    ]
    for yp in yaml_paths:
        if yp.exists():
            try:
                with open(yp) as fh:
                    data = yaml.safe_load(fh)
                return data.get("expectations", {}).get(domain, {})
            except Exception:
                pass
    return {}


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

        # ── Optional LLM-enhanced check ──
        if ctx.llm_client or self._pipeline:
            llm_checks = self._run_llm_depth_check(summary, ctx)
            checks.extend(llm_checks)

        return checks

    def _check_group_comparison(
        self, findings: List[str], per_group: Dict[str, Any],
    ) -> List[CheckResult]:
        """Flag if ≥3 groups but no ANOVA/Kruskal-Wallis in findings."""
        n_groups = len(per_group)
        if n_groups < 3:
            return []

        findings_text = " ".join(str(f) for f in findings).lower()
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
            severity=Severity.SHOULD_FIX,
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
            return [CheckResult(
                name="depth__homogeneous_charts",
                passed=False,
                severity=Severity.SHOULD_FIX,
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
        """Flag findings with numeric deviations but no interpretive vocabulary."""
        if not findings:
            return []

        bare_count = 0
        for f in findings:
            f_lower = str(f).lower()
            has_number = bool(re.search(r'\d+\.?\d*%', str(f)))
            has_interpretation = any(word in f_lower for word in _INTERPRETATION_VOCABULARY)
            if has_number and not has_interpretation:
                bare_count += 1

        if bare_count == 0:
            return [CheckResult(
                name="depth__interpretation_depth",
                passed=True,
                category=CheckCategory.CONTENT_QUALITY,
                detail="All findings with numeric values include interpretive context",
            )]

        return [CheckResult(
            name="depth__bare_deviations",
            passed=False,
            severity=Severity.SHOULD_FIX,
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

    def _check_domain_expectations(
        self, summary: Dict[str, Any], ctx: CriticContext,
    ) -> List[CheckResult]:
        """Check domain-specific required analyses."""
        checks: List[CheckResult] = []

        # Detect domain from data
        domain_flags = summary.get("domain_hints", {})
        if not domain_flags:
            # Try to detect from context
            cleaned_path = ctx.payload.get("cleaned_path", "")
            if cleaned_path and Path(cleaned_path).exists():
                try:
                    import pandas as pd
                    df = pd.read_parquet(cleaned_path, columns=None)
                    cols = [c.lower() for c in df.columns]
                    if any("retention" in c or "uv" in c or "mau" in c for c in cols):
                        domain_flags["chromatography"] = True
                    if any("m/z" in c or "mass" in c or "charge" in c for c in cols):
                        domain_flags["mass_spectrometry"] = True
                except Exception:
                    pass

        findings_text = " ".join(str(f) for f in summary.get("findings", [])).lower()

        for domain, is_present in domain_flags.items():
            if not is_present:
                continue
            expectations = _load_domain_expectations(domain)
            required = expectations.get("required_analyses", [])

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

        findings_text = " ".join(str(f) for f in findings).lower()
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
            + "\n".join(f"  - {f}" for f in findings[:8])
            + f"\n\nGroups: {len(per_group)}"
            + f"\nGroup names (sample): {list(per_group.keys())[:5]}"
        )

        try:
            from autogen.oai import OpenAIWrapper

            cfg = self._pipeline._base_config_dict()
            cfg["temperature"] = 0.0
            for _entry in cfg.get("config_list", []):
                _entry["timeout"] = 300
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
