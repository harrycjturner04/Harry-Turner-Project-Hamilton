#!/usr/bin/env python3
"""compare_runs.py — Compare pipeline outputs across ablation study runs.

Reads judge_input.json (or falls back to manifest_*.json) from each run
directory and produces a side-by-side comparison table.  No AG2 dependency —
uses stdlib only.

Usage::
    # Compare named run directories
    python Scripts/compare_runs.py Outputs/run_a Outputs/run_b Outputs/run_c

    # Auto-discover all SLURM output directories
    python Scripts/compare_runs.py --glob "Outputs/slurm_*"

    # Save as CSV for further analysis
    python Scripts/compare_runs.py --glob "Outputs/slurm_*" --output ablation.csv

    # JSON output
    python Scripts/compare_runs.py Outputs/run_a Outputs/run_b --json
"""
from __future__ import annotations

import argparse
import csv
import glob as _glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Data loading ──────────────────────────────────────────────────────────────

def load_judge_input(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load judge_input.json from a run directory.

    Falls back to extracting from manifest if judge_input.json is absent
    (e.g. runs produced before this feature was added).
    """
    # Primary: judge_input.json
    ji_path = run_dir / "judge_input.json"
    if ji_path.exists():
        try:
            return json.loads(ji_path.read_text("utf-8"))
        except Exception as exc:
            print(f"WARN: Could not parse {ji_path}: {exc}", file=sys.stderr)

    # Fallback: extract from manifest
    manifests = sorted(run_dir.glob("manifest_*.json"))
    for m in manifests:
        try:
            manifest = json.loads(m.read_text("utf-8"))
            # Re-derive a compatible structure from the manifest
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
                    "verified_claims_count": len(xval.get("verified_claims", [])),
                    "cv_gaps": xval.get("gaps", []),
                })
            qr = manifest.get("quality_review", {})
            return {
                "batch_id": manifest.get("batch_id", "unknown"),
                "pipeline_mode": manifest.get("pipeline_mode", "unknown"),
                "model": "unknown",
                "timestamp": "",
                "file_count": len(items),
                "files": file_summaries,
                "quality_review_overall": qr.get("overall_quality", "unknown"),
                "quality_review_summary": qr.get("summary", ""),
                "visual_review_count": manifest.get("visual_review_count", 0),
                "visual_discrepancies_count": 0,
                "_from_manifest": str(m),
            }
        except Exception as exc:
            print(f"WARN: Could not parse {m}: {exc}", file=sys.stderr)

    return None


# ── Metric derivation ─────────────────────────────────────────────────────────

def _stages_complete(files: List[Dict[str, Any]]) -> str:
    """Return 'N/M' string for files where all 4 stages completed."""
    if not files:
        return "0/0"
    complete = sum(
        1 for f in files
        if all(f.get("stages_completed", {}).values())
    )
    return f"{complete}/{len(files)}"


def _avg(values: List[float]) -> str:
    if not values:
        return "N/A"
    return f"{sum(values) / len(values):.1f}"


def derive_metrics(ji: Dict[str, Any]) -> Dict[str, Any]:
    """Derive a flat metrics dict from a judge_input record."""
    files = ji.get("files", [])
    plot_counts   = [f.get("plot_count", 0) for f in files]
    claim_counts  = [f.get("verified_claims_count", 0) for f in files]
    gap_counts    = [len(f.get("cv_gaps", [])) for f in files]

    visual_disc = ji.get("visual_discrepancies_count", 0)
    visual_count = ji.get("visual_review_count", 0)
    visual_str = f"{visual_disc}" if visual_count > 0 else "N/A (not run)"

    # Extract run_config label (WP0)
    run_config = ji.get("run_config", {})
    run_label = run_config.get("run_label", "") if isinstance(run_config, dict) else ""

    return {
        "run_dir":             "",          # filled by caller
        "run_label":           run_label,
        "batch_id":            ji.get("batch_id", "unknown"),
        "pipeline_mode":       ji.get("pipeline_mode", "unknown"),
        "model":               ji.get("model", "unknown"),
        "timestamp":           ji.get("timestamp", ""),
        "file_count":          ji.get("file_count", len(files)),
        "stages_all_complete": _stages_complete(files),
        "avg_plot_count":      _avg(plot_counts),
        "avg_verified_claims": _avg(claim_counts),
        "avg_cv_gaps":         _avg(gap_counts),
        "visual_discrepancies": visual_str,
        "overall_quality":     ji.get("quality_review_overall", "not_run"),
        "quality_summary":     (ji.get("quality_review_summary") or "")[:80],
    }


# ── Display ───────────────────────────────────────────────────────────────────

_DISPLAY_COLUMNS = [
    "run_dir",
    "run_label",
    "pipeline_mode",
    "file_count",
    "stages_all_complete",
    "avg_plot_count",
    "avg_verified_claims",
    "avg_cv_gaps",
    "visual_discrepancies",
    "overall_quality",
]


def print_table(rows: List[Dict[str, Any]]) -> None:
    """Print an ASCII comparison table."""
    if not rows:
        print("No runs to compare.")
        return

    # Shorten run_dir to last 2 path components
    display_rows = []
    for row in rows:
        d = dict(row)
        rdir = Path(row["run_dir"])
        d["run_dir"] = "/".join(rdir.parts[-2:]) if len(rdir.parts) >= 2 else str(rdir)
        display_rows.append(d)

    cols = _DISPLAY_COLUMNS
    widths = {
        col: max(len(col), max(len(str(r.get(col, ""))) for r in display_rows))
        for col in cols
    }

    header = " | ".join(col.ljust(widths[col]) for col in cols)
    sep    = "-+-".join("-" * widths[col] for col in cols)
    print(header)
    print(sep)
    for row in display_rows:
        print(" | ".join(str(row.get(col, "")).ljust(widths[col]) for col in cols))


def save_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved comparison to {output_path}")


def print_json(rows: List[Dict[str, Any]]) -> None:
    print(json.dumps(rows, indent=2))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare pipeline outputs across ablation study runs.\n"
            "Reads judge_input.json (or manifest) from each run directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_dirs",
        nargs="*",
        help="One or more output directories to compare",
    )
    parser.add_argument(
        "--glob", "-g",
        metavar="PATTERN",
        help="Glob pattern to auto-discover run directories (e.g. 'Outputs/slurm_*')",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="PATH",
        help="Save comparison as CSV (optional)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print output as JSON instead of ASCII table",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-file breakdown for each run",
    )
    args = parser.parse_args()

    # ── Collect directories ──
    dirs: List[Path] = []
    if args.run_dirs:
        dirs.extend(Path(d) for d in args.run_dirs)
    if args.glob:
        dirs.extend(
            Path(d) for d in sorted(_glob.glob(args.glob))
            if Path(d).is_dir()
        )
    if not dirs:
        print("ERROR: Provide at least one run directory or --glob pattern", file=sys.stderr)
        return 1

    # ── Load and derive metrics ──
    rows: List[Dict[str, Any]] = []
    for run_dir in dirs:
        if not run_dir.is_dir():
            print(f"WARN: Not a directory: {run_dir}", file=sys.stderr)
            continue
        ji = load_judge_input(run_dir)
        if ji is None:
            print(f"WARN: No judge_input.json or manifest found in {run_dir}", file=sys.stderr)
            row = {
                "run_dir": str(run_dir),
                "batch_id": "N/A",
                "pipeline_mode": "unknown",
                "model": "unknown",
                "timestamp": "",
                "file_count": "N/A",
                "stages_all_complete": "N/A",
                "avg_plot_count": "N/A",
                "avg_verified_claims": "N/A",
                "avg_cv_gaps": "N/A",
                "visual_discrepancies": "N/A",
                "overall_quality": "N/A",
                "quality_summary": "",
            }
        else:
            row = derive_metrics(ji)
            row["run_dir"] = str(run_dir)
        rows.append(row)

    if not rows:
        print("No valid runs found.")
        return 1

    # ── Output ──
    if args.json:
        print_json(rows)
    else:
        print_table(rows)
        if args.verbose:
            print()
            for run_dir in dirs:
                ji = load_judge_input(run_dir)
                if ji and ji.get("files"):
                    rc = ji.get("run_config", {})
                    label = rc.get("run_label", "") if isinstance(rc, dict) else ""
                    print(f"\n{'='*60}")
                    print(f"Per-file breakdown: {run_dir.name} ({ji.get('pipeline_mode')})")
                    if label:
                        print(f"  Run label: {label}")
                    if isinstance(rc, dict) and rc:
                        print(f"  Run config: {json.dumps(rc, default=str)}")
                    print(f"{'='*60}")
                    for f in ji["files"]:
                        stages = f.get("stages_completed", {})
                        print(
                            f"  {f['name']}: "
                            f"plots={f.get('plot_count', 0)}, "
                            f"claims={f.get('verified_claims_count', 0)}, "
                            f"gaps={len(f.get('cv_gaps', []))}, "
                            f"stages={stages}"
                        )

    if args.output:
        save_csv(rows, Path(args.output))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
