"""Evaluation configuration and OpenRouter client setup."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


@dataclass
class JudgeModelSpec:
    """Specification for one judge model via OpenRouter."""

    model_id: str               # e.g. "openai/gpt-4o"
    display_name: str           # e.g. "GPT-4o"
    temperature: float = 0.0
    max_tokens: int = 4096
    weight: float = 1.0         # Relative weight in multi-judge aggregation
    supports_vision: bool = False  # Send plot PNGs to this judge


@dataclass
class EvalConfig:
    """Top-level evaluation configuration."""

    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    judge_models: List[JudgeModelSpec] = field(default_factory=list)
    rubric_version: str = "v1"
    pairwise_enabled: bool = True
    num_judge_passes: int = 1
    evaluation_db_path: str = "Evaluation/evaluation.db"
    output_dir: str = "Evaluation"
    max_report_chars: int = 15_000
    max_analysis_chars: int = 10_000
    max_plots_per_eval: int = 10

    def __post_init__(self) -> None:
        if not self.judge_models:
            raise ValueError(
                "No judge models configured. "
                "Specify at least one model in evaluation_config.yaml"
            )


def load_config(config_path: Union[str, Path]) -> EvalConfig:
    """Load evaluation config from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation config not found: {path}\n"
            f"Copy evaluation_config.example.yaml and fill in your API key."
        )
    raw = yaml.safe_load(path.read_text("utf-8"))

    models = []
    for m in raw.get("judge_models", []):
        models.append(JudgeModelSpec(
            model_id=m["model_id"],
            display_name=m.get("display_name", m["model_id"]),
            temperature=m.get("temperature", 0.0),
            max_tokens=m.get("max_tokens", 4096),
            weight=m.get("weight", 1.0),
            supports_vision=m.get("supports_vision", False),
        ))

    return EvalConfig(
        openrouter_api_key=raw["openrouter_api_key"],
        openrouter_base_url=raw.get(
            "openrouter_base_url", "https://openrouter.ai/api/v1"
        ),
        judge_models=models,
        rubric_version=raw.get("rubric_version", "v1"),
        pairwise_enabled=raw.get("pairwise_enabled", True),
        num_judge_passes=raw.get("num_judge_passes", 1),
        evaluation_db_path=raw.get(
            "evaluation_db_path", "Evaluation/evaluation.db"
        ),
        output_dir=raw.get("output_dir", "Evaluation"),
        max_report_chars=raw.get("max_report_chars", 15_000),
        max_analysis_chars=raw.get("max_analysis_chars", 10_000),
        max_plots_per_eval=raw.get("max_plots_per_eval", 10),
    )


def build_openrouter_client(config: EvalConfig) -> Any:
    """Build OpenAI-compatible client pointing at OpenRouter."""
    from openai import OpenAI

    return OpenAI(
        api_key=config.openrouter_api_key,
        base_url=config.openrouter_base_url,
    )
