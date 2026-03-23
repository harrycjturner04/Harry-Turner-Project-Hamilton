"""Artifact indexer — catalog pipeline output files without copying them."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class DatasetArtifacts:
    """Index of all artifacts for one dataset within one run."""

    dataset_name: str
    cleaning_summary_path: Optional[Path] = None
    cleaned_parquet_path: Optional[Path] = None
    analysis_summary_path: Optional[Path] = None
    plot_paths: List[Path] = field(default_factory=list)
    cross_validation_path: Optional[Path] = None
    report_path: Optional[Path] = None
    debug_artifacts: Dict[str, Path] = field(default_factory=dict)


@dataclass
class RunArtifactIndex:
    """Complete artifact index for one run, referencing files in-place."""

    run_id: str
    run_dir: Path
    judge_input_path: Optional[Path] = None
    manifest_path: Optional[Path] = None
    global_report_path: Optional[Path] = None
    datasets: Dict[str, DatasetArtifacts] = field(default_factory=dict)


def _find_dataset_dirs(files_dir: Path) -> List[Path]:
    """Find dataset subdirectories under files/."""
    if not files_dir.is_dir():
        return []
    return [d for d in sorted(files_dir.iterdir()) if d.is_dir()]


def _index_dataset(dataset_dir: Path) -> DatasetArtifacts:
    """Index artifacts for a single dataset directory."""
    name = dataset_dir.name
    arts = DatasetArtifacts(dataset_name=name)

    # Cleaning
    cleaning_dir = dataset_dir / "cleaning"
    if cleaning_dir.is_dir():
        for f in cleaning_dir.iterdir():
            if f.name.endswith("_summary.json") or f.name == "cleaning_summary.json":
                arts.cleaning_summary_path = f
            elif f.suffix == ".parquet":
                arts.cleaned_parquet_path = f

    # Analysis
    analysis_dir = dataset_dir / "analysis"
    if analysis_dir.is_dir():
        for f in sorted(analysis_dir.iterdir()):
            if f.name == "analysis_summary.json":
                arts.analysis_summary_path = f
            elif f.suffix == ".png":
                arts.plot_paths.append(f)

    # Cross-validation
    cv_dir = dataset_dir / "cross_validation"
    if cv_dir.is_dir():
        for f in cv_dir.iterdir():
            if f.suffix == ".json":
                arts.cross_validation_path = f
                break
    # Also check flat file
    cv_flat = dataset_dir / "cross_validation.json"
    if cv_flat.exists() and arts.cross_validation_path is None:
        arts.cross_validation_path = cv_flat

    return arts


def _find_reports(run_dir: Path, dataset_names: List[str]) -> Dict[str, Path]:
    """Map dataset names to their report files."""
    reports_dir = run_dir / "reports"
    mapping: Dict[str, Path] = {}
    if not reports_dir.is_dir():
        return mapping

    for f in reports_dir.iterdir():
        if not f.suffix == ".md":
            continue
        for ds in dataset_names:
            # Match patterns like chromatography_combined__report.md
            stem_no_ext = ds.replace(".parquet", "")
            if stem_no_ext in f.name and f.name != "global_report.md":
                mapping[ds] = f
                break
    return mapping


def _find_global_report(run_dir: Path) -> Optional[Path]:
    """Find the global report."""
    reports_dir = run_dir / "reports"
    if reports_dir.is_dir():
        gr = reports_dir / "global_report.md"
        if gr.exists():
            return gr
    return None


def _index_debug(run_dir: Path, dataset_name: str) -> Dict[str, Path]:
    """Index debug artifacts for a dataset."""
    debug_dir = run_dir / "debug"
    result: Dict[str, Path] = {}
    if not debug_dir.is_dir():
        return result

    stem = dataset_name.replace(".parquet", "")
    for f in debug_dir.iterdir():
        if stem in f.name and f.suffix == ".json":
            # Extract the artifact type from filename
            # e.g. chromatography_combined__analysis__verdict.json
            parts = f.stem.split("__")
            if len(parts) >= 2:
                key = "__".join(parts[1:])
                result[key] = f
    return result


def index_run_artifacts(run_dir: Path) -> RunArtifactIndex:
    """Walk the run directory and index all artifacts by type.

    Does NOT copy any files. Only records paths.
    """
    run_dir = Path(run_dir)
    index = RunArtifactIndex(
        run_id=run_dir.name,
        run_dir=run_dir.resolve(),
    )

    # Judge input
    ji = run_dir / "judge_input.json"
    if ji.exists():
        index.judge_input_path = ji

    # Manifest
    for m in sorted(run_dir.glob("manifest_*.json")):
        index.manifest_path = m
        break

    # Global report
    index.global_report_path = _find_global_report(run_dir)

    # Per-dataset artifacts
    files_dir = run_dir / "files"
    dataset_dirs = _find_dataset_dirs(files_dir)
    dataset_names = [d.name for d in dataset_dirs]

    report_map = _find_reports(run_dir, dataset_names)

    for dd in dataset_dirs:
        arts = _index_dataset(dd)
        arts.report_path = report_map.get(dd.name)
        arts.debug_artifacts = _index_debug(run_dir, dd.name)
        index.datasets[dd.name] = arts

    logger.info(
        "Indexed run %s: %d datasets, %d total plots",
        run_dir.name,
        len(index.datasets),
        sum(len(a.plot_paths) for a in index.datasets.values()),
    )
    return index


def load_json_artifact(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Safely load a JSON artifact file."""
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception as exc:
        logger.warning("Could not parse %s: %s", path, exc)
        return None


def load_text_artifact(
    path: Optional[Path],
    max_chars: Optional[int] = None,
) -> Optional[str]:
    """Safely load a text artifact, optionally truncating."""
    if path is None or not path.exists():
        return None
    try:
        text = path.read_text("utf-8")
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "\n\n[... truncated ...]"
        return text
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None
