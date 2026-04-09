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

Subcommands: deterministic, evaluate, pairwise, ablation, repeatability,
             export, rankings, full.
Run ``python -m Scripts.evaluation --help`` for details.
"""

from .config import EvalConfig, JudgeModelSpec, load_config
from .deterministic import ingest_run_deterministic
from .registry import RunRecord, discover_runs, group_by_config
from .statistics import (
    coefficient_of_variation,
    compute_krippendorff_per_criterion,
    krippendorff_alpha,
)
from .storage import EvalDB

__all__ = [
    "EvalConfig",
    "JudgeModelSpec",
    "load_config",
    "RunRecord",
    "discover_runs",
    "group_by_config",
    "EvalDB",
    "ingest_run_deterministic",
    "coefficient_of_variation",
    "krippendorff_alpha",
    "compute_krippendorff_per_criterion",
]
