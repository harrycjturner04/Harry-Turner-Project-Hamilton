"""LLM-as-judge evaluation via OpenRouter."""

import base64
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .artifacts import (
    DatasetArtifacts,
    RunArtifactIndex,
    load_json_artifact,
    load_text_artifact,
)
from .config import EvalConfig, JudgeModelSpec, build_openrouter_client
from .rubrics import (
    GLOBAL_JUDGE_USER_PROMPT,
    JUDGE_USER_PROMPT,
    JUDGE_USER_PROMPT_WITH_VISION,
    PAIRWISE_SYSTEM_PROMPT,
    PAIRWISE_USER_PROMPT,
    build_judge_system_prompt,
    get_criterion_names,
    get_criterion_weights,
)

logger = logging.getLogger(__name__)


@dataclass
class JudgmentResult:
    """Result of one judge evaluating one artifact."""

    judge_model: str
    judge_pass: int
    artifact_type: str
    dataset_name: str
    run_id: str
    rubric_version: str
    criterion_scores: Dict[str, float]
    criterion_explanations: Dict[str, str]
    overall_score: float
    raw_response: str
    latency_ms: int
    tokens_used: int


def _encode_image_base64(path: Path) -> Optional[str]:
    """Read a PNG file and return base64-encoded string."""
    try:
        data = path.read_bytes()
        return base64.b64encode(data).decode("utf-8")
    except Exception as exc:
        logger.warning("Could not read image %s: %s", path, exc)
        return None


def _build_vision_content(
    user_text: str,
    plot_paths: List[Path],
    max_plots: int = 10,
) -> List[Dict[str, Any]]:
    """Build OpenAI-compatible multimodal content array."""
    content = [{"type": "text", "text": user_text}]
    for p in plot_paths[:max_plots]:
        b64 = _encode_image_base64(p)
        if b64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                },
            })
    return content


def _compute_weighted_score(
    criterion_scores: Dict[str, float],
    rubric_version: str,
) -> float:
    """Compute weighted overall score from criterion scores."""
    weights = get_criterion_weights(rubric_version)
    total = 0.0
    for name, weight in weights.items():
        total += weight * criterion_scores.get(name, 0.0)
    return round(total, 2)


def _parse_judge_response(
    raw_text: str,
    rubric_version: str,
) -> Dict[str, Any]:
    """Parse JSON response from judge, handling markdown fences."""
    text = raw_text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    parsed = json.loads(text)

    scores = parsed.get("scores", {})
    explanations = parsed.get("explanations", {})

    # Validate all expected criteria are present
    expected = get_criterion_names(rubric_version)
    for name in expected:
        if name not in scores:
            scores[name] = 0
            logger.warning("Missing criterion %r in judge response", name)
        else:
            # Clamp to 0-10
            scores[name] = max(0, min(10, int(scores[name])))
        if name not in explanations:
            explanations[name] = ""

    return {
        "scores": scores,
        "explanations": explanations,
        "overall_assessment": parsed.get("overall_assessment", ""),
    }


def evaluate_report(
    run_id: str,
    dataset_name: str,
    report_content: str,
    analysis_summary: Optional[Dict[str, Any]],
    cross_validation: Optional[Dict[str, Any]],
    plot_paths: List[Path],
    judge_model: JudgeModelSpec,
    client: Any,
    rubric_version: str = "v1",
    pass_number: int = 0,
    max_report_chars: int = 15_000,
    max_analysis_chars: int = 10_000,
    max_plots: int = 10,
) -> JudgmentResult:
    """Evaluate a pipeline report using a single judge model.

    Args:
        run_id: Identifier for the pipeline run.
        dataset_name: Name of the dataset being evaluated.
        report_content: Full markdown report text.
        analysis_summary: Parsed analysis_summary.json (context only).
        cross_validation: Parsed cross_validation.json (context only).
        plot_paths: Paths to PNG plot files.
        judge_model: Judge model specification.
        client: OpenAI-compatible client.
        rubric_version: Rubric version to use.
        pass_number: Pass index (for multi-pass evaluation).
        max_report_chars: Max chars of report to send.
        max_analysis_chars: Max chars of analysis summary to send.
        max_plots: Max number of plots to send for vision evaluation.

    Returns:
        JudgmentResult with scores and explanations.
    """
    # Truncate report
    if len(report_content) > max_report_chars:
        report_content = (
            report_content[:max_report_chars]
            + "\n\n[... truncated ...]"
        )

    # Format analysis summary excerpt
    analysis_excerpt = ""
    if analysis_summary:
        analysis_text = json.dumps(analysis_summary, indent=2, default=str)
        if len(analysis_text) > max_analysis_chars:
            analysis_text = (
                analysis_text[:max_analysis_chars]
                + "\n... truncated ..."
            )
        analysis_excerpt = analysis_text

    # Format cross-validation excerpt
    cv_excerpt = ""
    if cross_validation:
        cv_excerpt = json.dumps(cross_validation, indent=2, default=str)

    # Build config summary
    run_config_summary = f"Run: {run_id}, Dataset: {dataset_name}"

    # Build system prompt
    system_prompt = build_judge_system_prompt(rubric_version)

    # Build user prompt
    use_vision = judge_model.supports_vision and plot_paths
    prompt_template = (
        JUDGE_USER_PROMPT_WITH_VISION if use_vision
        else JUDGE_USER_PROMPT
    )
    user_text = prompt_template.format(
        dataset_name=dataset_name,
        run_config_summary=run_config_summary,
        report_content=report_content,
        analysis_summary_excerpt=analysis_excerpt,
        cross_validation_excerpt=cv_excerpt,
    )

    # Build messages
    messages = [{"role": "system", "content": system_prompt}]

    if use_vision:
        content = _build_vision_content(user_text, plot_paths, max_plots)
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": user_text})

    # Call OpenRouter
    start_ms = int(time.time() * 1000)
    try:
        response = client.chat.completions.create(
            model=judge_model.model_id,
            messages=messages,
            temperature=judge_model.temperature,
            max_tokens=judge_model.max_tokens,
        )
    except Exception as exc:
        logger.error(
            "Judge API call failed for %s/%s with %s: %s",
            run_id, dataset_name, judge_model.model_id, exc,
        )
        raise
    elapsed_ms = int(time.time() * 1000) - start_ms

    raw_text = response.choices[0].message.content or ""
    tokens_used = 0
    if response.usage:
        tokens_used = response.usage.total_tokens

    # Parse response
    try:
        parsed = _parse_judge_response(raw_text, rubric_version)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.error(
            "Failed to parse judge response from %s: %s\nRaw: %s",
            judge_model.model_id, exc, raw_text[:500],
        )
        raise ValueError(
            f"Judge {judge_model.model_id} returned unparseable response"
        ) from exc

    overall = _compute_weighted_score(parsed["scores"], rubric_version)

    return JudgmentResult(
        judge_model=judge_model.model_id,
        judge_pass=pass_number,
        artifact_type="report",
        dataset_name=dataset_name,
        run_id=run_id,
        rubric_version=rubric_version,
        criterion_scores=parsed["scores"],
        criterion_explanations=parsed["explanations"],
        overall_score=overall,
        raw_response=raw_text,
        latency_ms=elapsed_ms,
        tokens_used=tokens_used,
    )


def multi_judge_evaluate(
    run_id: str,
    dataset_name: str,
    artifacts: DatasetArtifacts,
    config: EvalConfig,
    client: Any,
) -> List[JudgmentResult]:
    """Run evaluation across all configured judge models and passes.

    Returns list of JudgmentResults (len = num_models * num_passes).
    """
    # Load artifacts
    report_content = load_text_artifact(
        artifacts.report_path, config.max_report_chars
    )
    if not report_content:
        logger.warning(
            "No report found for %s/%s, skipping evaluation",
            run_id, dataset_name,
        )
        return []

    analysis_summary = load_json_artifact(artifacts.analysis_summary_path)
    cross_validation = load_json_artifact(artifacts.cross_validation_path)

    results = []
    for judge_model in config.judge_models:
        for pass_num in range(config.num_judge_passes):
            try:
                result = evaluate_report(
                    run_id=run_id,
                    dataset_name=dataset_name,
                    report_content=report_content,
                    analysis_summary=analysis_summary,
                    cross_validation=cross_validation,
                    plot_paths=artifacts.plot_paths,
                    judge_model=judge_model,
                    client=client,
                    rubric_version=config.rubric_version,
                    pass_number=pass_num,
                    max_report_chars=config.max_report_chars,
                    max_analysis_chars=config.max_analysis_chars,
                    max_plots=config.max_plots_per_eval,
                )
                results.append(result)
                logger.info(
                    "Judgment: %s/%s by %s pass %d -> %.2f",
                    run_id, dataset_name,
                    judge_model.display_name, pass_num,
                    result.overall_score,
                )
            except Exception as exc:
                logger.error(
                    "Evaluation failed: %s/%s by %s pass %d: %s",
                    run_id, dataset_name,
                    judge_model.display_name, pass_num, exc,
                )
    return results


def evaluate_global_report(
    run_id: str,
    report_content: str,
    dataset_names: List[str],
    per_dataset_summaries: str,
    judge_model: JudgeModelSpec,
    client: Any,
    rubric_version: str = "global_v1",
    pass_number: int = 0,
    max_report_chars: int = 15_000,
) -> JudgmentResult:
    """Evaluate the global synthesis report using a single judge model.

    Args:
        run_id: Identifier for the pipeline run.
        report_content: Full markdown global report text.
        dataset_names: Names of datasets covered by the report.
        per_dataset_summaries: Concatenated summaries of per-dataset reports.
        judge_model: Judge model specification.
        client: OpenAI-compatible client.
        rubric_version: Rubric version (should be "global_v1").
        pass_number: Pass index (for multi-pass evaluation).
        max_report_chars: Max chars of report to send.

    Returns:
        JudgmentResult with scores and explanations.
    """
    if len(report_content) > max_report_chars:
        report_content = (
            report_content[:max_report_chars]
            + "\n\n[... truncated ...]"
        )

    run_config_summary = f"Run: {run_id}"

    system_prompt = build_judge_system_prompt(rubric_version)

    user_text = GLOBAL_JUDGE_USER_PROMPT.format(
        run_config_summary=run_config_summary,
        dataset_names=", ".join(dataset_names),
        report_content=report_content,
        per_dataset_summaries=per_dataset_summaries,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    start_ms = int(time.time() * 1000)
    try:
        response = client.chat.completions.create(
            model=judge_model.model_id,
            messages=messages,
            temperature=judge_model.temperature,
            max_tokens=judge_model.max_tokens,
        )
    except Exception as exc:
        logger.error(
            "Judge API call failed for %s/global with %s: %s",
            run_id, judge_model.model_id, exc,
        )
        raise
    elapsed_ms = int(time.time() * 1000) - start_ms

    raw_text = response.choices[0].message.content or ""
    tokens_used = 0
    if response.usage:
        tokens_used = response.usage.total_tokens

    try:
        parsed = _parse_judge_response(raw_text, rubric_version)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.error(
            "Failed to parse global judge response from %s: %s\nRaw: %s",
            judge_model.model_id, exc, raw_text[:500],
        )
        raise ValueError(
            f"Judge {judge_model.model_id} returned unparseable response"
        ) from exc

    overall = _compute_weighted_score(parsed["scores"], rubric_version)

    return JudgmentResult(
        judge_model=judge_model.model_id,
        judge_pass=pass_number,
        artifact_type="global_report",
        dataset_name="__global__",
        run_id=run_id,
        rubric_version=rubric_version,
        criterion_scores=parsed["scores"],
        criterion_explanations=parsed["explanations"],
        overall_score=overall,
        raw_response=raw_text,
        latency_ms=elapsed_ms,
        tokens_used=tokens_used,
    )


def multi_judge_evaluate_global(
    run_id: str,
    index: RunArtifactIndex,
    config: EvalConfig,
    client: Any,
    max_summary_chars_per_dataset: int = 2_000,
) -> List[JudgmentResult]:
    """Run global report evaluation across all configured judge models.

    Loads the global report and builds per-dataset summaries from the
    first portion of each dataset's report to provide reference context
    for the judge.

    Returns list of JudgmentResults (len = num_models * num_passes).
    """
    global_content = load_text_artifact(
        index.global_report_path, config.max_report_chars
    )
    if not global_content:
        logger.warning(
            "No global report found for %s, skipping global evaluation",
            run_id,
        )
        return []

    # Build per-dataset summaries as reference context
    summary_parts = []
    dataset_names = []
    for ds_name, arts in sorted(index.datasets.items()):
        dataset_names.append(ds_name)
        ds_report = load_text_artifact(
            arts.report_path, max_summary_chars_per_dataset
        )
        if ds_report:
            summary_parts.append(
                f"### {ds_name}\n{ds_report}"
            )
        else:
            summary_parts.append(f"### {ds_name}\n[No report available]")

    per_dataset_summaries = "\n\n".join(summary_parts)

    results = []
    for judge_model in config.judge_models:
        for pass_num in range(config.num_judge_passes):
            try:
                result = evaluate_global_report(
                    run_id=run_id,
                    report_content=global_content,
                    dataset_names=dataset_names,
                    per_dataset_summaries=per_dataset_summaries,
                    judge_model=judge_model,
                    client=client,
                    rubric_version="global_v1",
                    pass_number=pass_num,
                    max_report_chars=config.max_report_chars,
                )
                results.append(result)
                logger.info(
                    "Global judgment: %s by %s pass %d -> %.2f",
                    run_id, judge_model.display_name,
                    pass_num, result.overall_score,
                )
            except Exception as exc:
                logger.error(
                    "Global evaluation failed: %s by %s pass %d: %s",
                    run_id, judge_model.display_name, pass_num, exc,
                )
    return results


def save_raw_judgment(
    result: JudgmentResult,
    output_dir: Union[str, Path],
) -> Path:
    """Save raw judge response as JSON file for provenance."""
    out_dir = Path(output_dir) / "raw_judgments"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_safe = result.judge_model.replace("/", "_")
    filename = (
        f"{result.run_id}__{result.dataset_name}"
        f"__{result.artifact_type}__{model_safe}"
        f"__pass{result.judge_pass}.json"
    )
    path = out_dir / filename

    data = {
        "meta": {
            "run_id": result.run_id,
            "dataset_name": result.dataset_name,
            "artifact_type": result.artifact_type,
            "judge_model": result.judge_model,
            "judge_pass": result.judge_pass,
            "rubric_version": result.rubric_version,
            "latency_ms": result.latency_ms,
            "tokens_used": result.tokens_used,
        },
        "scores": result.criterion_scores,
        "explanations": result.criterion_explanations,
        "overall_score": result.overall_score,
        "raw_response": result.raw_response,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def response_hash(raw_response: str) -> str:
    """SHA-256 hash of the raw response for provenance tracking."""
    return hashlib.sha256(raw_response.encode("utf-8")).hexdigest()[:16]
