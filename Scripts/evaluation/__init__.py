"""Evaluation framework for automated ablation studies.

Fully decoupled from the pipeline — reads only from Outputs/ and writes
to Evaluation/.  Uses OpenRouter for LLM-as-judge evaluation.

Quick start::

    # 1. Copy and configure
    cp evaluation_config.example.yaml evaluation_config.yaml
    # Edit evaluation_config.yaml — add your OpenRouter API key

    # 2. Run full evaluation
    python -m Scripts.evaluation full \\
        --glob "Outputs/slurm_*" --baseline "baseline-v2"

Subcommands: evaluate, pairwise, ablation, export, rankings, full.
Run ``python -m Scripts.evaluation --help`` for details.
"""

from .config import EvalConfig, JudgeModelSpec, load_config
from .registry import RunRecord, discover_runs, group_by_config
from .storage import EvalDB

__all__ = [
    "EvalConfig",
    "JudgeModelSpec",
    "load_config",
    "RunRecord",
    "discover_runs",
    "group_by_config",
    "EvalDB",
]
