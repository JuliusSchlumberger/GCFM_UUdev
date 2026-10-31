"""Phase 1.9: flood volume and extent of the default scenario with the burned river depth scaled.

`run`: for each scale, copies the basin's preprocessing_inputs (rules 01-10
and the old rule-11 outputs) to a scratch results root
(<results_dir>/../scratch_phase1/depth_<scale>/), then runs Snakemake there
(GCFM_RESULTS_DIR) with river_depth_modelling.depth_scale = scale (a
--configfile overlay) and --forcerun modelled_depth_estimation, so rule 11
(burn and calibration) and the build (skeleton, spin-up, event run, metrics)
run on the scaled bed. Production results are not touched.
`compare`: flooded area and volume of each scale relative to the baseline
run in <results_dir>.

Writes phase1_depth_sensitivity_summary.csv to <results_dir>/figures/analysis/.

Usage: python analysis/phase1_depth_sensitivity.py run|compare [basin_id] [--scales 0.7 1.3] [--cores 14]
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from _common import REPO_ROOT, analysis_path, script_parser, site_path
from _phase1 import write_summary


def scratch_root(scale: float) -> Path:
    return site_path("results_dir").parent / "scratch_phase1" / f"depth_{scale:g}"


def run(basin_id: str, scale: float, cores: int) -> int:
    root = scratch_root(scale)
    src = site_path("results_dir") / basin_id / "preprocessing_inputs"
    dst = root / basin_id / "preprocessing_inputs"
    if dst.exists():
        shutil.rmtree(root / basin_id)
    shutil.copytree(src, dst)
    overlay = root / "depth_scale.yml"
    overlay.write_text(
        f"river_processing:\n  river_depth_modelling:\n    depth_scale: {scale}\n"
    )
    env = {**os.environ, "GCFM_RESULTS_DIR": root.as_posix()}
    cmd = [
        "snakemake",
        "build",
        "--cores",
        str(cores),
        "--configfile",
        "config/config.yml",
        str(overlay),
        "--forcerun",
        "modelled_depth_estimation",
        "--config",
        f"target_basins=[{basin_id}]",
        "target_scenarios=['default']",
    ]
    with open(root / f"snakemake_{basin_id}.log", "w") as log:
        return subprocess.run(
            cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
        ).returncode


def metrics(results_root: Path, basin_id: str) -> dict:
    p = results_root / basin_id / "runs" / "default" / "metrics" / "flood_metrics.csv"
    return pd.read_csv(p).iloc[0].to_dict() if p.exists() else {}


def compare(basin_id: str, scales: list[float]) -> list[dict]:
    base = metrics(site_path("results_dir"), basin_id)
    rows = []
    for scale in [1.0, *scales]:
        m = base if scale == 1.0 else metrics(scratch_root(scale), basin_id)
        rows.append(
            {
                "basin_id": basin_id,
                "depth_scale": scale,
                "flooded_area_km2": m.get("flooded_area_km2"),
                "volume_m3": m.get("volume_m3"),
                "area_rel_baseline": (
                    m.get("flooded_area_km2") / base["flooded_area_km2"]
                )
                if m and base
                else None,
                "volume_rel_baseline": (m.get("volume_m3") / base["volume_m3"])
                if m and base
                else None,
            }
        )
    return rows


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("action", choices=["run", "compare"])
    parser.add_argument("basin_id", nargs="?", default="2433835")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.7, 1.3])
    parser.add_argument("--cores", type=int, default=14)
    args = parser.parse_args()
    if args.action == "run":
        return max(run(args.basin_id, s, args.cores) for s in args.scales)
    rows = compare(args.basin_id, args.scales)
    print(
        write_summary(rows, analysis_path(None, __file__, "summary.csv")).to_string(
            index=False
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
