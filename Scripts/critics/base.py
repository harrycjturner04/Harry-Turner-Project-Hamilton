"""CriticModule abstract base class and CriticContext dataclass.

Every critic module implements ``CriticModule``.  The pipeline constructs a
single ``CriticContext`` per evaluation pass and feeds it to each active
critic in order.  Critics produce ``List[CheckResult]`` that slot directly
into the existing ``StageVerdict`` — the deterministic ``quality_gate()``
remains unchanged.
"""
from __future__ import annotations

import abc
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tools import CheckCategory, CheckResult, Severity  # noqa: F401 (re-export for convenience)

logger = logging.getLogger("captain_pipeline")


# ──────────────────────────────────────────────────────────────────────
# CriticContext — everything a critic module needs to evaluate
# ──────────────────────────────────────────────────────────────────────

@dataclass
class CriticContext:
    """Immutable evaluation context passed to every active critic."""

    # Stage identity
    stage_name: str
    attempt: int
    label: str                                       # e.g. "file1__analysis__retry1"

    # Stage I/O
    payload: Dict[str, Any]                          # the task payload sent to CaptainAgent
    listing: List[Dict[str, Any]]                    # output dir file listing
    last_reply: str                                  # agent's text reply (may be truncated)
    expected_paths: Dict[str, str]                   # e.g. {"cleaned_path": "/abs/path"}

    # Configuration
    quality_spec: Any                                # QualitySpec instance
    run_config: Any                                  # RunConfig instance

    # Pre-parsed artifacts (populated lazily by the pipeline before critic dispatch)
    analysis_summary: Optional[Dict[str, Any]] = None

    # Infrastructure (optional — not all critics need these)
    debug_root: Optional[Path] = None
    llm_client: Optional[Any] = None                 # DirectOpenAIWrapper or similar
    vlm_client: Optional[Any] = None                 # VLM endpoint wrapper

    # Tracking — mutated by the dispatch loop to record which critics ran
    evaluators_ran: Dict[str, bool] = field(default_factory=dict)

    # Plots that have exhausted fix attempts and must not be re-evaluated.
    # Populated by _fix_plots() and carried across loop iterations so the
    # VLM does not re-flag unfixable plots on every subsequent pass.
    exhausted_plots: Set[str] = field(default_factory=set)

    # Previous content evaluation results — used by the content evaluator
    # for reference-anchored evaluation (monotonic evaluation pressure).
    # When non-empty, the evaluator is instructed to only fail criteria
    # that previously passed if the current output is genuinely worse.
    previous_content_checks: List[Any] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# CriticModule ABC
# ──────────────────────────────────────────────────────────────────────

class CriticModule(abc.ABC):
    """Abstract base for all critic modules.

    Subclasses must set the class-level attributes and implement
    ``evaluate()`` and ``can_run()``.

    The dispatch loop calls ``can_run()`` first; if it returns False the
    critic is silently skipped and ``evaluators_ran[name]`` is set to False.
    """

    # ── Subclass must set these ──────────────────────────────────────
    name: str = ""                                   # unique identifier
    category: CheckCategory = CheckCategory.STRUCTURAL
    stage_applicability: Set[str] = set()            # stages this critic applies to
    requires_llm: bool = False
    requires_vlm: bool = False

    # ── Interface ────────────────────────────────────────────────────

    @abc.abstractmethod
    def evaluate(self, ctx: CriticContext) -> List[CheckResult]:
        """Run evaluation and return check results.

        Must NOT raise — catch exceptions internally and return an empty
        list on failure so the pipeline can continue with remaining critics.
        """
        ...

    @abc.abstractmethod
    def can_run(self, ctx: CriticContext) -> bool:
        """Return True if this critic should execute for the given context.

        Typical checks: stage applicability, feature flag, LLM/VLM availability.
        """
        ...

    # ── Convenience helpers ──────────────────────────────────────────

    def safe_evaluate(self, ctx: CriticContext) -> List[CheckResult]:
        """Wrapper that catches exceptions and logs them, returning [] on failure."""
        start = time.monotonic()
        try:
            checks = self.evaluate(ctx)
        except Exception:
            logger.exception("Critic %s raised during evaluate()", self.name)
            checks = []
        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Build standardised log entry
        must_fix = sum(1 for c in checks if not c.passed and c.severity == Severity.MUST_FIX)
        should_fix = sum(1 for c in checks if not c.passed and c.severity == Severity.SHOULD_FIX)
        passed = sum(1 for c in checks if c.passed)

        log_entry = {
            "critic_name": self.name,
            "stage": ctx.stage_name,
            "attempt": ctx.attempt,
            "checks_produced": len(checks),
            "must_fix_count": must_fix,
            "should_fix_count": should_fix,
            "passed_count": passed,
            "wall_time_ms": elapsed_ms,
        }
        logger.info(
            "Critic %-25s  checks=%d  must_fix=%d  should_fix=%d  passed=%d  (%dms)",
            self.name, len(checks), must_fix, should_fix, passed, elapsed_ms,
        )
        # Persist detailed log if debug_root is available
        if ctx.debug_root:
            from tools import check_to_dict, safe_write_json
            log_entry["check_details"] = [check_to_dict(c) for c in checks]
            # Append to cumulative critic log for this label
            log_path = ctx.debug_root / f"{ctx.label}__critic_log.jsonl"
            try:
                import json
                with open(log_path, "a") as fh:
                    fh.write(json.dumps(log_entry, default=str) + "\n")
            except Exception:
                logger.debug("Failed to write critic log to %s", log_path)

        return checks
