"""Shared paths and helpers for the stand-alone analysis scripts in analysis/."""

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


def basin_id_from_argv() -> str:
    """Return the basin id given as the first CLI argument, else DEFAULT_BASIN_ID."""
    return sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASIN_ID


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
