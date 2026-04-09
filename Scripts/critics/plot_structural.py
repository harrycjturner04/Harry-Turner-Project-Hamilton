"""PlotStructuralCritic — pure-Python PNG validation checks.

WP-C4: Catches obvious plot failures without consuming VLM inference budget.
Runs before the VLM VisualCritic: corrupt PNGs, resolution issues, duplicate
plots, and size outliers are detected and flagged.

All checks operate on raw file bytes — no image library required.
"""
from __future__ import annotations

import hashlib
import logging
import struct
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Set

from tools import CheckCategory, CheckResult, Severity
from critics.base import CriticContext, CriticModule

logger = logging.getLogger("captain_pipeline")

# PNG magic bytes (first 8 bytes of every valid PNG)
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _read_png_dimensions(path: Path) -> tuple[int, int] | None:
    """Read width and height from a PNG IHDR chunk.

    Returns (width, height) or None on failure.
    """
    try:
        with open(path, "rb") as fh:
            header = fh.read(8)
            if header != _PNG_MAGIC:
                return None
            # IHDR is always the first chunk after the 8-byte signature
            chunk_len = struct.unpack(">I", fh.read(4))[0]
            chunk_type = fh.read(4)
            if chunk_type != b"IHDR" or chunk_len < 13:
                return None
            width, height = struct.unpack(">II", fh.read(8))
            return (width, height)
    except Exception:
        return None


class PlotStructuralCritic(CriticModule):
    """Pure-Python structural validation of PNG plot files."""

    name = "plot_structural"
    category = CheckCategory.PLOT_QUALITY
    stage_applicability: Set[str] = {"analysis"}
    requires_llm = False
    requires_vlm = False

    def can_run(self, ctx: CriticContext) -> bool:
        if ctx.stage_name not in self.stage_applicability:
            return False
        # Runs whenever visual critic is enabled (acts as pre-filter)
        return getattr(ctx.run_config, "critic_visual", True)

    def evaluate(self, ctx: CriticContext) -> List[CheckResult]:
        checks: List[CheckResult] = []

        png_items = [
            item for item in ctx.listing
            if isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and item["path"].lower().endswith(".png")
        ]

        if not png_items:
            return checks

        # Collect sizes and hashes for cross-plot analysis
        sizes: List[int] = []
        hashes: Dict[str, List[str]] = {}  # hash -> [filename, ...]

        for item in png_items:
            plot_path = Path(item["path"])
            fname = plot_path.name

            if not plot_path.exists():
                continue

            file_size = plot_path.stat().st_size
            sizes.append(file_size)

            # 1. Corrupt PNG detection
            try:
                with open(plot_path, "rb") as fh:
                    magic = fh.read(8)
                if magic != _PNG_MAGIC:
                    checks.append(CheckResult(
                        name=f"plot_struct__corrupt_png",
                        passed=False,
                        severity=Severity.MUST_FIX,
                        category=CheckCategory.PLOT_QUALITY,
                        detail=f"'{fname}' has invalid PNG header (corrupt file)",
                        fix_instruction=f"Regenerate '{fname}' — the file is corrupt.",
                        ref=fname,
                    ))
                    continue  # Skip further checks on corrupt files
            except Exception:
                continue

            # 2. Resolution check
            dims = _read_png_dimensions(plot_path)
            if dims:
                width, height = dims
                if width < 400 or height < 400:
                    checks.append(CheckResult(
                        name=f"plot_struct__low_resolution",
                        passed=False,
                        severity=Severity.SHOULD_FIX,
                        category=CheckCategory.PLOT_QUALITY,
                        detail=f"'{fname}' resolution {width}x{height} is below 400px minimum",
                        fix_instruction=(
                            f"Regenerate '{fname}' with higher resolution "
                            "(use dpi=300 and figsize=(10,6) or larger)."
                        ),
                        ref=fname,
                    ))

                # 3. Extreme aspect ratio
                if dims[0] > 0 and dims[1] > 0:
                    ratio = max(dims) / min(dims)
                    if ratio > 20.0:
                        # Physically unusable — MUST_FIX
                        checks.append(CheckResult(
                            name=f"plot_struct__extreme_aspect",
                            passed=False,
                            severity=Severity.MUST_FIX,
                            category=CheckCategory.PLOT_QUALITY,
                            detail=(
                                f"'{fname}' has extreme aspect ratio {ratio:.1f}:1 "
                                f"({width}x{height}) — physically unreadable"
                            ),
                            fix_instruction=(
                                f"Regenerate '{fname}' with a balanced aspect ratio. "
                                "Use figsize=(12, 6) or similar. If many subplots are "
                                "needed, split into multiple figures rather than stacking "
                                "all in one tall figure. Aim for 1:1 to 2:1 (width:height)."
                            ),
                            ref=fname,
                        ))
                    elif ratio > 4.0:
                        checks.append(CheckResult(
                            name=f"plot_struct__extreme_aspect",
                            passed=False,
                            severity=Severity.SHOULD_FIX,
                            category=CheckCategory.PLOT_QUALITY,
                            detail=(
                                f"'{fname}' has extreme aspect ratio {ratio:.1f}:1 "
                                f"({width}x{height})"
                            ),
                            fix_instruction=(
                                f"Regenerate '{fname}' with a more balanced aspect ratio "
                                "(aim for 1:1 to 2:1)."
                            ),
                            ref=fname,
                        ))

            # Collect hash for duplicate detection (first 1024 bytes)
            try:
                with open(plot_path, "rb") as fh:
                    content_hash = hashlib.md5(fh.read(1024)).hexdigest()
                hashes.setdefault(content_hash, []).append(fname)
            except Exception:
                pass

        # 4. Size outlier detection (plot 10x smaller than median)
        if len(sizes) >= 3:
            med_size = median(sizes)
            if med_size > 0:
                for item in png_items:
                    pp = Path(item["path"])
                    if not pp.exists():
                        continue
                    sz = pp.stat().st_size
                    if sz < med_size / 10 and sz < 5000:
                        checks.append(CheckResult(
                            name=f"plot_struct__size_outlier",
                            passed=False,
                            severity=Severity.MUST_FIX,
                            category=CheckCategory.PLOT_QUALITY,
                            detail=(
                                f"'{pp.name}' is {sz} bytes — 10x smaller than "
                                f"median ({med_size:.0f} bytes), likely empty or broken"
                            ),
                            fix_instruction=f"Regenerate '{pp.name}' — it appears empty or broken.",
                            ref=pp.name,
                        ))

        # 5. Duplicate plot detection
        for hash_val, fnames in hashes.items():
            if len(fnames) > 1:
                checks.append(CheckResult(
                    name=f"plot_struct__duplicate_plot",
                    passed=False,
                    severity=Severity.SHOULD_FIX,
                    category=CheckCategory.PLOT_QUALITY,
                    detail=(
                        f"Duplicate plots detected (identical first 1024 bytes): "
                        + ", ".join(fnames)
                    ),
                    fix_instruction=(
                        "Multiple plots have identical content. Each plot should "
                        "answer a distinct analytical question. Remove duplicates "
                        "or regenerate with different data/chart types."
                    ),
                    ref=fnames[0],
                ))

        # If no issues found, add a passing check
        if not checks:
            checks.append(CheckResult(
                name="plot_struct__all_valid",
                passed=True,
                category=CheckCategory.PLOT_QUALITY,
                detail=f"All {len(png_items)} plots pass structural validation",
            ))

        return checks
