"""Pairwise comparison and ranking (Bradley-Terry, Elo)."""

import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .artifacts import load_text_artifact
from .config import EvalConfig, JudgeModelSpec
from .rubrics import PAIRWISE_SYSTEM_PROMPT, PAIRWISE_USER_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class PairwiseResult:
    """Result of one pairwise comparison between two runs."""

    run_a_id: str
    run_b_id: str
    dataset_name: str
    judge_model: str
    winner: str              # run_a_id | run_b_id | "tie"
    confidence: float        # 0-1
    explanation: str
    criterion_preferences: Optional[Dict[str, Any]]
    raw_response: str
    presentation_order: str  # "a_first" | "b_first"


def _parse_pairwise_response(raw_text: str) -> Dict[str, Any]:
    """Parse JSON response from pairwise judge."""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)


def pairwise_compare(
    run_a_id: str,
    run_b_id: str,
    report_a: str,
    report_b: str,
    dataset_name: str,
    judge_model: JudgeModelSpec,
    client: Any,
    max_chars: int = 15_000,
    randomize_order: bool = True,
) -> PairwiseResult:
    """Compare two reports head-to-head.

    Randomizes presentation order (A/B) to reduce position bias.
    """
    # Truncate
    if len(report_a) > max_chars:
        report_a = report_a[:max_chars] + "\n\n[... truncated ...]"
    if len(report_b) > max_chars:
        report_b = report_b[:max_chars] + "\n\n[... truncated ...]"

    # Randomize order
    if randomize_order and random.random() < 0.5:
        label_a, label_b = "B", "A"
        content_a, content_b = report_b, report_a
        actual_a_id, actual_b_id = run_b_id, run_a_id
        order = "b_first"
    else:
        label_a, label_b = "A", "B"
        content_a, content_b = report_a, report_b
        actual_a_id, actual_b_id = run_a_id, run_b_id
        order = "a_first"

    user_prompt = PAIRWISE_USER_PROMPT.format(
        dataset_name=dataset_name,
        label_a=label_a,
        label_b=label_b,
        report_a_content=content_a,
        report_b_content=content_b,
    )

    messages = [
        {"role": "system", "content": PAIRWISE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    response = client.chat.completions.create(
        model=judge_model.model_id,
        messages=messages,
        temperature=judge_model.temperature,
        max_tokens=judge_model.max_tokens,
    )

    raw_text = response.choices[0].message.content or ""

    try:
        parsed = _parse_pairwise_response(raw_text)
    except (json.JSONDecodeError, KeyError) as exc:
        logger.error(
            "Failed to parse pairwise response from %s: %s",
            judge_model.model_id, exc,
        )
        raise ValueError(
            f"Judge {judge_model.model_id} returned unparseable "
            f"pairwise response"
        ) from exc

    # Map the winner label back to actual run IDs
    raw_winner = parsed.get("overall_winner", "tie")
    if raw_winner == label_a:
        winner = actual_a_id
    elif raw_winner == label_b:
        winner = actual_b_id
    else:
        winner = "tie"

    # Map criterion preferences back
    criterion_prefs = parsed.get("criterion_preferences", {})
    mapped_prefs = {}
    for crit, pref in criterion_prefs.items():
        if isinstance(pref, dict):
            pref_winner = pref.get("winner", "tie")
            if pref_winner == label_a:
                pref["winner"] = actual_a_id
            elif pref_winner == label_b:
                pref["winner"] = actual_b_id
            else:
                pref["winner"] = "tie"
            mapped_prefs[crit] = pref

    return PairwiseResult(
        run_a_id=run_a_id,
        run_b_id=run_b_id,
        dataset_name=dataset_name,
        judge_model=judge_model.model_id,
        winner=winner,
        confidence=float(parsed.get("confidence", 0.5)),
        explanation=parsed.get("explanation", ""),
        criterion_preferences=mapped_prefs,
        raw_response=raw_text,
        presentation_order=order,
    )


# ── Ranking Algorithms ───────────────────────────────────────────────────────


def compute_bradley_terry(
    pairwise_results: List[PairwiseResult],
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """Compute Bradley-Terry model parameters from pairwise comparisons.

    Returns dict mapping run_label -> strength parameter (log-scale).
    Higher = better. Normalized so mean = 0.
    """
    # Collect win counts
    wins = {}  # type: Dict[Tuple[str, str], int]
    participants = set()  # type: set

    for r in pairwise_results:
        participants.add(r.run_a_id)
        participants.add(r.run_b_id)
        if r.winner == r.run_a_id:
            key = (r.run_a_id, r.run_b_id)
            wins[key] = wins.get(key, 0) + 1
        elif r.winner == r.run_b_id:
            key = (r.run_b_id, r.run_a_id)
            wins[key] = wins.get(key, 0) + 1
        else:
            # Tie: half win each
            key_a = (r.run_a_id, r.run_b_id)
            key_b = (r.run_b_id, r.run_a_id)
            wins[key_a] = wins.get(key_a, 0) + 0.5
            wins[key_b] = wins.get(key_b, 0) + 0.5

    if not participants:
        return {}

    labels = sorted(participants)
    n = len(labels)
    idx = {label: i for i, label in enumerate(labels)}

    # Initialize strengths
    p = [1.0] * n

    for iteration in range(max_iter):
        p_old = list(p)
        for i in range(n):
            w_i = sum(
                wins.get((labels[i], labels[j]), 0)
                for j in range(n) if j != i
            )
            denom = sum(
                (wins.get((labels[i], labels[j]), 0)
                 + wins.get((labels[j], labels[i]), 0))
                / (p[i] + p[j])
                for j in range(n) if j != i
                if (p[i] + p[j]) > 0
            )
            if denom > 0:
                p[i] = w_i / denom

        # Normalize so product = 1
        geom_mean = 1.0
        for pi in p:
            geom_mean *= pi
        geom_mean = geom_mean ** (1.0 / n) if geom_mean > 0 else 1.0
        if geom_mean > 0:
            p = [pi / geom_mean for pi in p]

        # Check convergence
        max_change = max(abs(p[i] - p_old[i]) for i in range(n))
        if max_change < tol:
            break

    # Convert to log-scale, normalize mean=0
    import math
    log_p = [math.log(pi) if pi > 0 else -10.0 for pi in p]
    mean_log = sum(log_p) / n
    log_p = [lp - mean_log for lp in log_p]

    return {labels[i]: round(log_p[i], 4) for i in range(n)}


def compute_elo_ratings(
    pairwise_results: List[PairwiseResult],
    initial_rating: float = 1000.0,
    k_factor: float = 32.0,
) -> Dict[str, float]:
    """Compute Elo ratings from pairwise comparisons.

    Processes results in order. Returns dict of run_label -> rating.
    """
    ratings = {}  # type: Dict[str, float]

    for r in pairwise_results:
        if r.run_a_id not in ratings:
            ratings[r.run_a_id] = initial_rating
        if r.run_b_id not in ratings:
            ratings[r.run_b_id] = initial_rating

        ra = ratings[r.run_a_id]
        rb = ratings[r.run_b_id]

        # Expected scores
        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        eb = 1.0 / (1.0 + 10.0 ** ((ra - rb) / 400.0))

        # Actual scores
        if r.winner == r.run_a_id:
            sa, sb = 1.0, 0.0
        elif r.winner == r.run_b_id:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        ratings[r.run_a_id] = ra + k_factor * (sa - ea)
        ratings[r.run_b_id] = rb + k_factor * (sb - eb)

    return {k: round(v, 1) for k, v in ratings.items()}


def response_hash(raw_response: str) -> str:
    """SHA-256 hash for provenance."""
    return hashlib.sha256(raw_response.encode("utf-8")).hexdigest()[:16]


@dataclass
class PairwiseResultWithConsistency:
    """Pairwise comparison result enriched with position-swap consistency."""

    primary: PairwiseResult
    swapped: Optional[PairwiseResult]
    consistent: bool         # True if both orderings agree on winner
    winner: str              # Consensus winner; 'tie' if inconsistent
    inconsistency_note: str  # Empty string if consistent


def pairwise_compare_with_swap(
    run_a_id: str,
    run_b_id: str,
    report_a: str,
    report_b: str,
    dataset_name: str,
    judge_model: "JudgeModelSpec",
    client: Any,
    max_chars: int = 15_000,
) -> PairwiseResultWithConsistency:
    """Compare two reports with position-swap consistency check.

    Runs the comparison twice: once with A first, once with B first.
    Only accepts the verdict if both orderings agree on the winner,
    mitigating position bias (documented at 60-75% without this check).

    Args:
        Same as pairwise_compare(), but randomize_order is controlled
        internally to ensure exactly one A-first and one B-first run.

    Returns:
        PairwiseResultWithConsistency with consistent=True only when
        both orderings produce the same winner.
    """
    # First pass: A presented first
    primary = pairwise_compare(
        run_a_id=run_a_id,
        run_b_id=run_b_id,
        report_a=report_a,
        report_b=report_b,
        dataset_name=dataset_name,
        judge_model=judge_model,
        client=client,
        max_chars=max_chars,
        randomize_order=False,  # force A first
    )

    try:
        # Second pass: swap A and B positions
        swapped_raw = pairwise_compare(
            run_a_id=run_b_id,  # swap IDs
            run_b_id=run_a_id,
            report_a=report_b,  # swap reports
            report_b=report_a,
            dataset_name=dataset_name,
            judge_model=judge_model,
            client=client,
            max_chars=max_chars,
            randomize_order=False,  # force swapped-A (original B) first
        )
    except Exception as exc:
        logger.warning(
            "Position-swap comparison failed for %s/%s: %s",
            run_a_id, run_b_id, exc,
        )
        return PairwiseResultWithConsistency(
            primary=primary,
            swapped=None,
            consistent=False,
            winner="tie",
            inconsistency_note=f"Swap call failed: {exc}",
        )

    # Map swapped result back to original ID space
    # swapped.winner is in terms of run_b_id/run_a_id (swapped roles)
    if swapped_raw.winner == run_b_id:
        # Swapped comparison had "run_b_id" in position A, and it won
        # → original run_a_id would be the winner in standard framing
        swapped_winner_in_original = run_a_id
    elif swapped_raw.winner == run_a_id:
        swapped_winner_in_original = run_b_id
    else:
        swapped_winner_in_original = "tie"

    consistent = primary.winner == swapped_winner_in_original
    consensus_winner = primary.winner if consistent else "tie"
    note = (
        ""
        if consistent
        else (
            f"Inconsistent: A-first={primary.winner}, "
            f"B-first={swapped_winner_in_original}"
        )
    )

    return PairwiseResultWithConsistency(
        primary=primary,
        swapped=swapped_raw,
        consistent=consistent,
        winner=consensus_winner,
        inconsistency_note=note,
    )
