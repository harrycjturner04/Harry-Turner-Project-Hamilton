"""Run registry — discover, index, and group pipeline runs."""

import glob as _glob
import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    """Complete metadata record for one pipeline execution."""

    run_id: str                       # Directory name
    run_dir: Path                     # Absolute path
    run_label: str                    # From run_config.run_label
    pipeline_mode: str
    model: str
    timestamp: str
    batch_id: str
    run_config: Dict[str, Any]
    context_md_hash: Optional[str]
    git_commit_hash: Optional[str]
    datasets: List[str]
    file_count: int
    stages_all_complete: bool
    judge_input: Optional[Dict[str, Any]] = field(default=None, repr=False)
    manifest: Optional[Dict[str, Any]] = field(default=None, repr=False)


def _load_judge_input(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load judge_input.json, falling back to manifest.

    Mirrors the pattern from compare_runs.py:load_judge_input().
    """
    ji_path = run_dir / "judge_input.json"
    if ji_path.exists():
        try:
            return json.loads(ji_path.read_text("utf-8"))
        except Exception as exc:
            logger.warning("Could not parse %s: %s", ji_path, exc)

    # Fallback: extract from manifest
    for m in sorted(run_dir.glob("manifest_*.json")):
        try:
            manifest = json.loads(m.read_text("utf-8"))
            items = manifest.get("items", [])
            file_summaries = []
            for mi in items:
                analysis = mi.get("analysis", {})
                xval = mi.get("cross_validation", {})
                file_summaries.append({
                    "name": Path(mi.get("file", "")).name,
                    "stages_completed": {
                        "cleaning": mi.get("cleaning", {}).get("ok"),
                        "analysis": analysis.get("ok"),
                        "cross_validation": xval.get("consistent"),
                        "report": mi.get("report_path") is not None,
                    },
                    "plot_count": sum(
                        1 for a in analysis.get("artifacts", [])
                        if isinstance(a, str) and a.endswith(".png")
                    ),
                    "verified_claims_count": len(
                        xval.get("verified_claims", [])
                    ),
                    "cv_gaps": xval.get("gaps", []),
                })
            return {
                "batch_id": manifest.get("batch_id", "unknown"),
                "pipeline_mode": manifest.get("pipeline_mode", "unknown"),
                "model": "unknown",
                "timestamp": "",
                "file_count": len(items),
                "files": file_summaries,
                "quality_review_overall": "unknown",
                "visual_review_count": 0,
                "visual_discrepancies_count": 0,
                "_from_manifest": str(m),
            }
        except Exception as exc:
            logger.warning("Could not parse %s: %s", m, exc)

    return None


def _load_manifest(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load the first manifest_batch_*.json found."""
    for m in sorted(run_dir.glob("manifest_*.json")):
        try:
            return json.loads(m.read_text("utf-8"))
        except Exception:
            continue
    return None


def _all_stages_complete(files: List[Dict[str, Any]]) -> bool:
    """Check if all files completed all stages."""
    if not files:
        return False
    return all(
        all(f.get("stages_completed", {}).values())
        for f in files
    )


def _compute_context_hash(run_dir: Path) -> Optional[str]:
    """Try to find and hash the context.md used for this run."""
    # Check if context.md was copied into the run directory
    for candidate in [run_dir / "context.md", run_dir / "context_snapshot.md"]:
        if candidate.exists():
            content = candidate.read_bytes()
            return hashlib.sha256(content).hexdigest()[:16]
    return None


def _get_git_hash() -> Optional[str]:
    """Get current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def _build_run_record(run_dir: Path) -> Optional[RunRecord]:
    """Build a RunRecord from a run directory."""
    ji = _load_judge_input(run_dir)
    if ji is None:
        logger.warning("No judge_input.json or manifest in %s", run_dir)
        return None

    manifest = _load_manifest(run_dir)
    files = ji.get("files", [])
    rc = ji.get("run_config", {})
    run_label = rc.get("run_label", "") if isinstance(rc, dict) else ""

    datasets = [f.get("name", "unknown") for f in files]

    return RunRecord(
        run_id=run_dir.name,
        run_dir=run_dir.resolve(),
        run_label=run_label,
        pipeline_mode=ji.get("pipeline_mode", "unknown"),
        model=ji.get("model", "unknown"),
        timestamp=ji.get("timestamp", ""),
        batch_id=ji.get("batch_id", "unknown"),
        run_config=rc if isinstance(rc, dict) else {},
        context_md_hash=_compute_context_hash(run_dir),
        git_commit_hash=_get_git_hash(),
        datasets=datasets,
        file_count=ji.get("file_count", len(files)),
        stages_all_complete=_all_stages_complete(files),
        judge_input=ji,
        manifest=manifest,
    )


def discover_runs(
    outputs_dir: Union[str, Path],
    glob_pattern: str = "slurm_*",
    run_dirs: Optional[List[Union[str, Path]]] = None,
) -> List[RunRecord]:
    """Discover and index all pipeline runs.

    Args:
        outputs_dir: Base directory containing run directories.
        glob_pattern: Glob pattern to match run directories.
        run_dirs: Explicit list of run directories (overrides glob).

    Returns:
        List of RunRecord sorted by timestamp.
    """
    dirs: List[Path] = []

    if run_dirs:
        dirs.extend(Path(d) for d in run_dirs)
    else:
        pattern = str(Path(outputs_dir) / glob_pattern)
        dirs.extend(
            Path(d) for d in sorted(_glob.glob(pattern))
            if Path(d).is_dir()
        )

    records: List[RunRecord] = []
    for d in dirs:
        if not d.is_dir():
            logger.warning("Not a directory: %s", d)
            continue
        record = _build_run_record(d)
        if record is not None:
            records.append(record)

    records.sort(key=lambda r: r.timestamp)
    logger.info("Discovered %d runs", len(records))
    return records


def group_by_config(runs: List[RunRecord]) -> Dict[str, List[RunRecord]]:
    """Group runs by their run_label (experimental condition).

    WP-agnostic: each unique run_label is treated as a condition.
    """
    groups: Dict[str, List[RunRecord]] = {}
    for run in runs:
        label = run.run_label or run.run_id
        groups.setdefault(label, []).append(run)
    return groups
