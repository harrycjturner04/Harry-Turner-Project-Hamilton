"""Modular critic framework for the biologics analysis pipeline.

WP-C1: Composable critic modules that replace inline evaluation logic
in captain_pipeline.py.  Each module implements the CriticModule protocol,
produces List[CheckResult], and slots into the existing StageVerdict /
quality_gate() deterministic decision layer without modification.
"""

from critics.base import CriticContext, CriticModule  # noqa: F401
from critics.registry import CriticRegistry  # noqa: F401
