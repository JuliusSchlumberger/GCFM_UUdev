"""List figures left in the locations the pipeline no longer writes to; delete them on request.

The pipeline writes every figure to <results_dir>/{basin_id}/figures/ (and
figures/analysis/). Older results keep image files under
preprocessing_inputs/visuals/, spin_up/, runs/ and adaptation_lines/, and
analysis/ scripts kept theirs in <repo>/figs/. Only .png, .mp4 and .gif files
are matched; data files (.tif, .csv, .gpkg, .nc) are never touched.

Usage:
    python tools/list_stale_figures.py              # list, per basin
    python tools/list_stale_figures.py --delete     # delete what is listed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "workflow"))
from src.io import site_setting  # noqa: E402

IMAGE_SUFFIXES = (".png", ".mp4", ".gif")
# Per basin, relative to <results_dir>/{basin_id}/.
STALE_DIRS = ("preprocessing_inputs/visuals", "spin_up", "runs", "adaptation_lines")
# analysis/ scripts' former output folder, relative to the repository.
STALE_REPO_DIR = "figs"
# plot_dag.py still writes here.
KEEP_REPO_DIRS = ("figs/dag",)


def stale_figures(results_dir: Path, repo_root: Path) -> dict[str, list[Path]]:
    """Return image files outside the figures/ folders, grouped by basin id or 'repo'."""
    found: dict[str, list[Path]] = {}
    for basin_dir in sorted(
        p for p in results_dir.iterdir() if p.is_dir() and p.name.isdigit()
    ):
        files = sorted(
            f
            for d in STALE_DIRS
            if (basin_dir / d).is_dir()
            for f in (basin_dir / d).rglob("*")
            if f.suffix.lower() in IMAGE_SUFFIXES and f.is_file()
        )
        if files:
            found[basin_dir.name] = files
    repo_dir = repo_root / STALE_REPO_DIR
    if repo_dir.is_dir():
        keep = [repo_root / k for k in KEEP_REPO_DIRS]
        files = sorted(
            f
            for f in repo_dir.rglob("*")
            if f.is_file()
            and f.suffix.lower() in IMAGE_SUFFIXES
            and not any(f.is_relative_to(k) for k in keep)
        )
        if files:
            found["repo"] = files
    return found


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--delete", action="store_true", help="delete the listed files")
    args = parser.parse_args()

    found = stale_figures(Path(site_setting("results_dir")), REPO_ROOT)
    total = 0
    for group, files in found.items():
        print(f"{group}: {len(files)} file(s)")
        for f in files:
            print(f"  {f}")
            if args.delete:
                f.unlink()
        total += len(files)
    print(f"{'Deleted' if args.delete else 'Found'} {total} stale figure file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
