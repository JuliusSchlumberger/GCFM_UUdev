"""Shared paths and helpers for the stand-alone analysis scripts in analysis/."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASIN_ID = "2433835"


def site_path(key: str) -> Path:
    """Return a site setting (see config/site.example.yml) as a Path."""
    workflow_dir = str(REPO_ROOT / "workflow")
    if workflow_dir not in sys.path:
        sys.path.insert(0, workflow_dir)
    from src.io import site_setting

    return Path(site_setting(key))


def analysis_dir(basin_id: str | int | None = None) -> Path:
    """Return (and create) <results_dir>/<basin_id>/figures/analysis/, or the cross-basin one.

    basin_id None: <results_dir>/figures/analysis/, for outputs spanning
    several basins (summaries, global overviews).
    """
    root = site_path("results_dir")
    out = (
        (root / str(basin_id) if basin_id is not None else root)
        / "figures"
        / "analysis"
    )
    out.mkdir(parents=True, exist_ok=True)
    return out


def analysis_path(basin_id: str | int | None, script: str, quantity: str) -> Path:
    """Return the path of one output of an analysis script.

    Per basin: 90_<script stem>_<quantity> in the basin's analysis folder, so
    these sort after the pipeline's own diagnostics; cross-basin (basin_id
    None): <script stem>_<quantity>. script is the caller's __file__.
    """
    prefix = "90_" if basin_id is not None else ""
    return analysis_dir(basin_id) / f"{prefix}{Path(script).stem}_{quantity}"


def script_parser(doc: str | None) -> argparse.ArgumentParser:
    """Return an argument parser whose help text is the script's module docstring."""
    return argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )


def add_basin_id_argument(parser: argparse.ArgumentParser) -> None:
    """Add the optional positional basin_id (default DEFAULT_BASIN_ID)."""
    parser.add_argument(
        "basin_id",
        nargs="?",
        default=DEFAULT_BASIN_ID,
        help=f"basin id (default: {DEFAULT_BASIN_ID})",
    )


def first_existing(candidates: list[Path]) -> Path:
    """Return the first existing path; raise FileNotFoundError listing all if none."""
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "None of the candidate paths exist:\n" + "\n".join(f"  {p}" for p in candidates)
    )


def label(s: str) -> str:
    """Return a name as a title-case label."""
    return s.replace("_", " ").title()
