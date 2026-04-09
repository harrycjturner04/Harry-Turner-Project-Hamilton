#!/usr/bin/env python3
"""WP-R: Report quality gate module.

Provides blocking critique → revision loop for the report pipeline.
All checks are gated behind context.md run_config toggles so they are
disabled by default (preserving ablation baseline behaviour).

Extracted from captain_pipeline.py and adapted as parameterised free
functions with no class coupling.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from prompts import REPORT_RUBRIC, REPORT_WRITER_PROMPT, STAGE_CRITIC_PROMPT
from tools import (
    CheckCategory,
    CheckResult,
    Severity,
    _EXPECTED_CRITERIA_COUNT,
    check_to_dict,
    evaluator_coverage_ratio,
    parse_json_tolerant,
    safe_write_json,
    strip_think_tokens,
)

logger = logging.getLogger("report_quality")

# Module-level failure tracking (mirrors CaptainPipeline._critic_failure_counts)
_critic_failure_counts: Dict[str, int] = {}
_MAX_CRITIC_FAILURES = 3

# Rubric lookup for the content evaluator
_CRITIC_RUBRICS: Dict[str, str] = {
    "report": REPORT_RUBRIC,
}


# ──────────────────────────────────────────────────────────────────────
# Tier 1: Pure-Python checks (extracted from CaptainPipeline statics)
# ──────────────────────────────────────────────────────────────────────

def compute_grounding_rate(report_md: str) -> Optional[float]:
    """Fraction of analytical paragraphs containing at least one number.

    WP-R4: Mirrors evaluation/deterministic.compute_quantitative_grounding_rate().
    """
    if not report_md:
        return None
    lines = report_md.split("\n")
    analytical_start = 0
    for i, line in enumerate(lines):
        if line.startswith("## ") and any(
            kw in line.lower()
            for kw in ("result", "finding", "data", "analysis",
                       "discussion", "quality", "material", "method")
        ):
            analytical_start = i
            break
    analytical_text = "\n".join(lines[analytical_start:])
    paragraphs = [
        p.strip() for p in analytical_text.split("\n\n")
        if len(p.strip()) > 50
        and not p.strip().startswith("#")
        and not p.strip().startswith("|")
    ]
    if not paragraphs:
        return None
    _num_re = re.compile(r"\b\d+[\.,]?\d*\b")
    grounded = sum(1 for p in paragraphs if _num_re.search(p))
    return round(grounded / len(paragraphs), 4)


def check_numerical_consistency(
    report_md: str,
    analysis_data: Dict[str, Any],
    cleaning_data: Dict[str, Any],
    verified_claims: Optional[List[Dict[str, Any]]] = None,
) -> List[CheckResult]:
    """Flag numbers in report that contradict JSON artifacts.

    WP-R6: Extracts metric=value patterns from report prose and compares
    against per_group and findings in analysis_data.  Returns SHOULD_FIX
    CheckResults for explicit contradictions.

    WP-R6b: Also checks report prose against cross-validation verified
    claims. Discrepancies against verified claims are MUST_FIX since
    cross-validation has independently confirmed the correct values.
    """
    issues: List[CheckResult] = []
    if not report_md or not analysis_data:
        return issues

    # Build a flat lookup of metric → value from per_group means
    per_group = analysis_data.get(
        "per_group", analysis_data.get("per_run_per_stage", {})
    )
    metric_means: Dict[str, float] = {}
    if isinstance(per_group, dict):
        metric_sums: Dict[str, List[float]] = {}
        for gv in per_group.values():
            if isinstance(gv, dict):
                for mk, mv in gv.items():
                    if isinstance(mv, (int, float)):
                        metric_sums.setdefault(mk, []).append(float(mv))
        for mk, vals in metric_sums.items():
            metric_means[mk.lower()] = sum(vals) / len(vals)

    # Simple cleaning stats lookup
    rows_before = cleaning_data.get("rows_before")
    rows_after = cleaning_data.get("rows_after")
    if isinstance(rows_before, (int, float)):
        metric_means["rows_before"] = float(rows_before)
    if isinstance(rows_after, (int, float)):
        metric_means["rows_after"] = float(rows_after)

    if not metric_means:
        return issues

    _kv_re = re.compile(
        r"(?:([a-z_][a-z0-9_]*)\s*[=:]\s*)(\d+[\.,]?\d*(?:\s*%)?)",
        re.IGNORECASE,
    )
    tolerance = 0.05  # ±5%

    for match in _kv_re.finditer(report_md):
        key = match.group(1).lower().strip()
        raw_val = match.group(2).replace(",", ".").replace("%", "").strip()
        try:
            reported_val = float(raw_val)
        except ValueError:
            continue

        exact = metric_means.get(key)
        if exact is None:
            for mk in metric_means:
                if key in mk or mk in key:
                    exact = metric_means[mk]
                    break

        if exact is None or exact == 0:
            continue

        if abs(reported_val - exact) / abs(exact) > tolerance:
            issues.append(CheckResult(
                name=f"numerical_consistency_{key}",
                passed=False,
                severity=Severity.SHOULD_FIX,
                category=CheckCategory.CONTENT_QUALITY,
                detail=(
                    f"Report states '{key}={reported_val}' but data shows "
                    f"{key}≈{exact:.4g} (difference > {tolerance*100:.0f}% tolerance)."
                ),
                fix_instruction=(
                    f"Correct the value of '{key}' to match the data: "
                    f"use {exact:.4g} instead of {reported_val}."
                ),
            ))

    # ── WP-R6b: Cross-validation verified claims ──
    # Discrepancies against verified claims are MUST_FIX (independently confirmed).
    if verified_claims:
        _number_re = re.compile(r'\b(\d+[\.,]?\d*)\s*%?')
        for claim in verified_claims:
            if not isinstance(claim, dict):
                continue
            actual_val = claim.get("actual_value") or claim.get("recomputed_value")
            claimed_val = claim.get("claimed_value") or claim.get("original_value")
            metric_name = claim.get("metric", claim.get("name", ""))
            discrepancy = claim.get("discrepancy", False)
            if not discrepancy or actual_val is None:
                continue
            try:
                actual_float = float(str(actual_val).replace(",", ".").replace("%", ""))
            except (ValueError, TypeError):
                continue
            # Search report for the incorrect claimed value
            if claimed_val is not None:
                try:
                    claimed_float = float(str(claimed_val).replace(",", ".").replace("%", ""))
                except (ValueError, TypeError):
                    continue
                claimed_str = f"{claimed_float:g}"
                if claimed_str in report_md or str(claimed_val) in report_md:
                    issues.append(CheckResult(
                        name=f"crossval_discrepancy_{metric_name}".replace(" ", "_")[:60],
                        passed=False,
                        severity=Severity.MUST_FIX,
                        category=CheckCategory.CONTENT_QUALITY,
                        detail=(
                            f"Cross-validation found: '{metric_name}' claimed "
                            f"{claimed_val} but verified value is {actual_val}."
                        ),
                        fix_instruction=(
                            f"Replace the value {claimed_val} for '{metric_name}' "
                            f"with the cross-validation verified value: {actual_val}."
                        ),
                    ))

    return issues


def check_statistical_plausibility(report_md: str) -> List[CheckResult]:
    """Flag statistically impossible claims in report prose.

    WP-R7: Catches common LLM fabrication patterns like:
    - F-statistic ≈ 1 with p < 0.01 (impossible — F≈1 means p≈0.5)
    - R² outside [0, 1] or negative (unless explicitly labelled adjusted R²)
    - Cohen's d near 0 described as a large effect
    - p-values outside [0, 1]
    """
    issues: List[CheckResult] = []
    if not report_md:
        return issues

    # Pattern: F = <number> ... p < <number> on the same sentence/line
    _f_p_re = re.compile(
        r'F\s*[=≈]\s*(\d+[\.,]?\d*)\s*.*?p\s*[<≤]\s*(\d+[\.,]?\d*)',
        re.IGNORECASE,
    )
    for m in _f_p_re.finditer(report_md):
        try:
            f_val = float(m.group(1).replace(",", "."))
            p_val = float(m.group(2).replace(",", "."))
        except ValueError:
            continue
        # F close to 1 cannot produce a tiny p-value
        if 0.5 <= f_val <= 1.5 and p_val < 0.01:
            issues.append(CheckResult(
                name="stat_plausibility_f_test",
                passed=False,
                severity=Severity.MUST_FIX,
                category=CheckCategory.CONTENT_QUALITY,
                detail=(
                    f"Implausible F-test: F={f_val} with p<{p_val}. "
                    f"An F-statistic near 1 indicates no group difference "
                    f"and cannot produce p<0.01."
                ),
                fix_instruction=(
                    "Remove or correct this F-test result. Recompute from "
                    "the data or remove the statistical claim entirely."
                ),
            ))

    # Pattern: R² or R-squared = <number>
    _r2_re = re.compile(
        r'R[²2]\s*[=≈]\s*(-?\d+[\.,]?\d*)',
        re.IGNORECASE,
    )
    for m in _r2_re.finditer(report_md):
        try:
            r2_val = float(m.group(1).replace(",", "."))
        except ValueError:
            continue
        if r2_val > 1.0 or r2_val < -0.5:
            issues.append(CheckResult(
                name="stat_plausibility_r_squared",
                passed=False,
                severity=Severity.MUST_FIX,
                category=CheckCategory.CONTENT_QUALITY,
                detail=f"Implausible R²={r2_val} (valid range: 0–1 for standard R²).",
                fix_instruction="Correct the R² value to match the actual model output.",
            ))

    # Pattern: p = <number> or p-value = <number>
    _p_re = re.compile(
        r'p[\-\s]*(?:value)?\s*[=≈]\s*(-?\d+[\.,]?\d*)',
        re.IGNORECASE,
    )
    for m in _p_re.finditer(report_md):
        try:
            p_val = float(m.group(1).replace(",", "."))
        except ValueError:
            continue
        if p_val < 0 or p_val > 1:
            issues.append(CheckResult(
                name="stat_plausibility_p_value",
                passed=False,
                severity=Severity.MUST_FIX,
                category=CheckCategory.CONTENT_QUALITY,
                detail=f"Implausible p-value={p_val} (must be between 0 and 1).",
                fix_instruction="Correct the p-value to a valid value between 0 and 1.",
            ))

    return issues


def parse_evaluator_items(items: List[Dict[str, Any]]) -> List[CheckResult]:
    """Parse content evaluator JSON items into CheckResult objects."""
    checks: List[CheckResult] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        severity_str = item.get("severity")
        severity = None
        if severity_str == "must_fix":
            severity = Severity.MUST_FIX
        elif severity_str == "should_fix":
            severity = Severity.SHOULD_FIX
        checks.append(CheckResult(
            name=str(item.get("criterion", "unknown")),
            passed=bool(item.get("passed", True)),
            severity=severity,
            category=CheckCategory.CONTENT_QUALITY,
            detail=str(item.get("detail", "")),
            fix_instruction=str(item.get("fix_instruction", "")),
        ))
    return checks


# ──────────────────────────────────────────────────────────────────────
# Tier 2: Refactored content evaluator (decoupled from CaptainPipeline)
# ──────────────────────────────────────────────────────────────────────

def run_content_evaluator(
    stage_name: str,
    artifacts_summary: str,
    label: str,
    critic_client: Any,
    critic_model: str,
    llm_config: Dict[str, Any],
    debug_root: Optional[Path] = None,
) -> List[CheckResult]:
    """Evaluate report quality via LLM call with REPORT_RUBRIC.

    Uses OpenRouter critic client if available, falls back to local vLLM
    via autogen OpenAIWrapper.
    """
    global _critic_failure_counts

    rubric = _CRITIC_RUBRICS.get(stage_name)
    if not rubric:
        return []

    if _critic_failure_counts.get(stage_name, 0) >= _MAX_CRITIC_FAILURES:
        logger.warning(
            "Content evaluator disabled for stage '%s' after %d cumulative "
            "failures — skipping for %s",
            stage_name, _critic_failure_counts[stage_name], label,
        )
        return []

    prompt = STAGE_CRITIC_PROMPT.format(
        stage_name=stage_name,
        stage_rubric=rubric,
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": artifacts_summary[:10000]},
    ]
    try:
        if critic_client is not None:
            response = critic_client.chat.completions.create(
                model=critic_model,
                messages=messages,
                temperature=0.0,
                max_tokens=4096,
            )
            reply = response.choices[0].message.content or ""
        else:
            from autogen.oai import OpenAIWrapper

            cfg = dict(llm_config)
            cfg["temperature"] = 0.0
            for _entry in cfg.get("config_list", []):
                _entry = dict(_entry)
                _entry["timeout"] = 600
            client = OpenAIWrapper(**cfg)
            response = client.create(messages=messages)
            reply = strip_think_tokens(
                response.choices[0].message.content or ""
            )
    except Exception as exc:
        _critic_failure_counts[stage_name] = (
            _critic_failure_counts.get(stage_name, 0) + 1
        )
        logger.warning(
            "Content evaluator failed for %s (failure #%d for stage '%s'): %s",
            label, _critic_failure_counts[stage_name], stage_name, exc,
        )
        if debug_root:
            safe_write_json(
                debug_root / f"{label}__report_eval_FAILED.json",
                {"label": label,
                 "failure_count": _critic_failure_counts[stage_name],
                 "error": str(exc),
                 "error_type": type(exc).__name__},
            )
        return []

    if debug_root:
        safe_write_json(
            debug_root / f"{label}__report_eval.json",
            {"label": label, "reply": reply[:3000]},
        )

    # Parse JSON array of per-criterion checks
    parsed = parse_json_tolerant(reply)

    items: List[Dict[str, Any]] = []
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        if "pass" in parsed:
            # Legacy format
            items = _legacy_critic_to_checks(parsed, stage_name)
        else:
            items = [parsed]

    if not items:
        _critic_failure_counts[stage_name] = (
            _critic_failure_counts.get(stage_name, 0) + 1
        )
        logger.warning(
            "Content evaluator returned unparseable response for %s "
            "(failure #%d for stage '%s')",
            label, _critic_failure_counts[stage_name], stage_name,
        )
        return []

    checks: List[CheckResult] = parse_evaluator_items(items)

    # Coverage validation: fill missing criteria
    expected_count = _EXPECTED_CRITERIA_COUNT.get(stage_name, 0)
    if expected_count > 0 and len(checks) < expected_count:
        _expected_names = []
        for line in rubric.split("\n"):
            line = line.strip()
            if line and line[0].isdigit() and "." in line:
                parts = line.split(".", 1)
                if len(parts) == 2:
                    name_part = parts[1].strip().split(":")[0].strip()
                    _expected_names.append(name_part)
        returned_names = {c.name for c in checks}
        for name in _expected_names:
            if name not in returned_names:
                checks.append(CheckResult(
                    name=name,
                    passed=False,
                    severity=Severity.SHOULD_FIX,
                    category=CheckCategory.CONTENT_QUALITY,
                    detail="Not evaluated by content evaluator (coverage gap)",
                    fix_instruction=(
                        "Content evaluator did not assess this criterion. "
                        "Ensure output quality meets this rubric requirement."
                    ),
                ))

    coverage = evaluator_coverage_ratio(checks, stage_name)
    logger.info(
        "Content evaluator coverage for %s: %.0f%% (%d/%d criteria)",
        label, coverage * 100,
        min(len(checks), expected_count) if expected_count else len(checks),
        expected_count or len(checks),
    )

    if debug_root:
        safe_write_json(
            debug_root / f"{label}__report_eval_result.json",
            {"coverage_ratio": coverage,
             "checks": [check_to_dict(c) for c in checks]},
        )

    return checks


def _legacy_critic_to_checks(
    parsed: Dict[str, Any], stage_name: str,
) -> List[Dict[str, Any]]:
    """Convert old-format critic response to per-criterion check items."""
    items: List[Dict[str, Any]] = []
    passed = bool(parsed.get("pass", True))
    score = str(parsed.get("score", "adequate"))
    feedback_list = parsed.get("feedback", [])
    priority = parsed.get("priority_fix", "")

    if passed:
        items.append({
            "criterion": f"{stage_name}_overall",
            "passed": True,
            "severity": None,
            "detail": f"Overall score: {score}",
            "fix_instruction": "",
        })
    else:
        for i, fb in enumerate(feedback_list[:6]):
            items.append({
                "criterion": f"{stage_name}_feedback_{i+1}",
                "passed": False,
                "severity": "must_fix" if i == 0 and priority else "should_fix",
                "detail": str(fb),
                "fix_instruction": str(priority) if i == 0 else "",
            })
    return items


# ──────────────────────────────────────────────────────────────────────
# Tier 3: Quality gate orchestrator
# ──────────────────────────────────────────────────────────────────────

def run_report_quality_gate(
    report_md: str,
    analysis_summary: Dict[str, Any],
    cleaning_summary: Dict[str, Any],
    run_config: Any,
    critic_client: Any,
    critic_model: str,
    llm_config: Dict[str, Any],
    label: str = "report",
    debug_root: Optional[Path] = None,
    max_revision_rounds: int = 3,
    verified_claims: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Orchestrate all WP-R quality checks and revision rounds.

    Runs enabled checks, aggregates issues, and performs up to
    ``max_revision_rounds`` revision LLM calls when MUST_FIX issues
    are found.

    Returns ``(revised_report_md, quality_metadata)``.
    When all toggles are at defaults (off), returns the report unchanged.
    """
    meta: Dict[str, Any] = {
        "revisions_applied": 0,
        "checks_per_round": [],
    }

    # Early exit: no run_config or all toggles off
    if run_config is None:
        return report_md, meta

    _critic_report = getattr(run_config, "critic_report", True)
    _num_check = getattr(run_config, "numerical_accuracy_check", True)
    _min_grounding = getattr(run_config, "min_quantitative_grounding", 0.0)

    if not (_critic_report or _num_check or _min_grounding > 0.0):
        return report_md, meta

    for round_num in range(max_revision_rounds):
        all_checks: List[CheckResult] = []

        # ── WP-R3: Narrative critic ──
        if _critic_report and critic_client is not None:
            artifacts_summary = _build_artifacts_summary(
                report_md, analysis_summary, cleaning_summary,
            )
            critic_checks = run_content_evaluator(
                "report", artifacts_summary, f"{label}_r{round_num}",
                critic_client, critic_model, llm_config, debug_root,
            )
            all_checks.extend(critic_checks)

        # ── WP-R6: Numerical consistency ──
        if _num_check:
            num_checks = check_numerical_consistency(
                report_md, analysis_summary, cleaning_summary,
                verified_claims=verified_claims,
            )
            all_checks.extend(num_checks)

        # ── WP-R7: Statistical plausibility ──
        stat_checks = check_statistical_plausibility(report_md)
        all_checks.extend(stat_checks)

        # ── WP-R4: Quantitative grounding ──
        if _min_grounding > 0.0:
            rate = compute_grounding_rate(report_md)
            if rate is not None and rate < _min_grounding:
                all_checks.append(CheckResult(
                    name="quantitative_grounding",
                    passed=False,
                    severity=Severity.MUST_FIX,
                    category=CheckCategory.CONTENT_QUALITY,
                    detail=(
                        f"Quantitative grounding rate is {rate:.2%}, below "
                        f"the {_min_grounding:.2%} threshold."
                    ),
                    fix_instruction=(
                        "Add specific numeric values (means, CVs, p-values, "
                        "counts) from the analysis data to each analytical "
                        "paragraph. Every prose paragraph in the Results and "
                        "Analysis sections should cite at least one concrete "
                        "number from the data."
                    ),
                ))

        meta["checks_per_round"].append(
            [check_to_dict(c) for c in all_checks]
        )

        # Determine if revision is needed
        must_fix = [
            c for c in all_checks
            if not c.passed and c.severity == Severity.MUST_FIX
        ]
        should_fix = [
            c for c in all_checks
            if not c.passed and c.severity == Severity.SHOULD_FIX
        ]

        if not must_fix:
            logger.info(
                "WP-R quality gate passed for %s (round %d): "
                "%d should_fix, 0 must_fix",
                label, round_num + 1, len(should_fix),
            )
            break

        # Build aggregated revision instructions (top 10 by severity)
        revision_parts = [
            f"REVISION REQUIRED (round {round_num + 1}). "
            f"Address these {len(must_fix)} critical issues:"
        ]
        for i, c in enumerate(must_fix[:10], 1):
            revision_parts.append(
                f"{i}. [{c.name}] {c.fix_instruction or c.detail}"
            )
        if should_fix:
            revision_parts.append(
                f"\nAlso consider these {len(should_fix)} improvements:"
            )
            for c in should_fix[:5]:
                revision_parts.append(f"- [{c.name}] {c.fix_instruction or c.detail}")

        revision_text = "\n".join(revision_parts)

        logger.info(
            "WP-R quality gate: %d MUST_FIX, %d SHOULD_FIX for %s — "
            "issuing revision (round %d)",
            len(must_fix), len(should_fix), label, round_num + 1,
        )

        # ── Revision LLM call ──
        revised = _revise_report(
            report_md, revision_text, analysis_summary, cleaning_summary,
            critic_client, critic_model, llm_config,
        )

        # ── Strengthened revision validation (WP-R P2-9) ──
        _accept = True
        if not revised or not revised.strip().startswith("#") or len(revised.strip()) < 200:
            _accept = False
            logger.warning(
                "Revision produced degenerate output for %s — keeping "
                "previous version (basic validation failed)",
                label,
            )
        elif len(revised.strip()) < len(report_md.strip()) * 0.4:
            _accept = False
            logger.warning(
                "Revision lost >60%% of content for %s (%d→%d chars) — "
                "keeping previous version",
                label, len(report_md.strip()), len(revised.strip()),
            )
        else:
            # Check that the revision didn't drop major section headings
            orig_h2 = set(re.findall(r'^##\s+(.+)', report_md, re.MULTILINE))
            new_h2 = set(re.findall(r'^##\s+(.+)', revised, re.MULTILINE))
            dropped = orig_h2 - new_h2
            if len(dropped) > len(orig_h2) * 0.5 and len(orig_h2) >= 3:
                _accept = False
                logger.warning(
                    "Revision dropped %d/%d sections for %s — keeping "
                    "previous version (dropped: %s)",
                    len(dropped), len(orig_h2), label,
                    ", ".join(sorted(dropped)[:5]),
                )

        if _accept:
            report_md = revised
            meta["revisions_applied"] += 1
        else:
            break

    if debug_root:
        safe_write_json(
            debug_root / f"{label}__quality_gate.json", meta,
        )

    return report_md, meta


# ──────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────

def _build_artifacts_summary(
    report_md: str,
    analysis_summary: Dict[str, Any],
    cleaning_summary: Dict[str, Any],
) -> str:
    """Build an artifact summary string for the content evaluator."""
    parts = []
    if cleaning_summary:
        parts.append(
            "## Cleaning Summary\n"
            + json.dumps(cleaning_summary, indent=2, default=str)[:2000]
        )
    if analysis_summary:
        # Include key fields only to stay within evaluator context window
        compact = {
            k: v for k, v in analysis_summary.items()
            if k in (
                "findings", "per_group", "ml_modeling",
                "domain_reasoning", "anova_p_values",
            )
        }
        parts.append(
            "## Analysis Summary\n"
            + json.dumps(compact, indent=2, default=str)[:6000]
        )
    parts.append(
        "## Generated Report\n" + report_md[:8000]
    )
    return "\n\n".join(parts)


def _revise_report(
    report_md: str,
    revision_instructions: str,
    analysis_summary: Dict[str, Any],
    cleaning_summary: Dict[str, Any],
    critic_client: Any,
    critic_model: str,
    llm_config: Dict[str, Any],
) -> Optional[str]:
    """Issue a single revision LLM call to fix identified issues."""
    # Build a compact context of the original data for grounding
    context_parts = []
    if cleaning_summary:
        context_parts.append(
            "## Cleaning Summary\n```json\n"
            + json.dumps(cleaning_summary, indent=2, default=str)[:2000]
            + "\n```"
        )
    if analysis_summary:
        compact = {
            k: v for k, v in analysis_summary.items()
            if k in (
                "findings", "per_group", "ml_modeling",
                "domain_reasoning", "anova_p_values", "notes",
            )
        }
        context_parts.append(
            "## Analysis Summary\n```json\n"
            + json.dumps(compact, indent=2, default=str)[:6000]
            + "\n```"
        )
    data_context = "\n\n".join(context_parts) if context_parts else ""

    messages = [
        {"role": "system", "content": REPORT_WRITER_PROMPT},
        {"role": "user", "content": data_context},
        {"role": "assistant", "content": report_md},
        {"role": "user", "content": revision_instructions},
    ]

    try:
        if critic_client is not None:
            response = critic_client.chat.completions.create(
                model=critic_model,
                messages=messages,
                temperature=0.2,
                max_tokens=16384,
            )
            raw = response.choices[0].message.content or ""
        else:
            from autogen.oai import OpenAIWrapper

            cfg = dict(llm_config)
            cfg["temperature"] = 0.2
            client = OpenAIWrapper(**cfg)
            response = client.create(messages=messages)
            raw = strip_think_tokens(
                response.choices[0].message.content or ""
            )

        # Try to extract from JSON wrapper if present
        parsed = parse_json_tolerant(raw)
        if isinstance(parsed, dict) and parsed.get("report_markdown"):
            return parsed["report_markdown"]
        return raw
    except Exception as exc:
        logger.warning("Report revision LLM call failed: %s", exc)
        return None
