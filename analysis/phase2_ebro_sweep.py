"""Phase 2a: Ebro round-1 settings and roughness sweep against the 1D-2D gap.

`run`: per variant, copies the basin's preprocessing_inputs to
<scratch>/p2a/<variant>/ (GCFM_PLAN_SCRATCH, default
<results_dir>/../scratch_plan), applies the variant and reruns rule 11 there
(snakemake preprocess --forcerun modelled_depth_estimation, a --configfile
overlay). Variants: nr_levels 50; advection 0; nuvisc 0; huthresh 0.05
(calibration_sfincs_settings); calibration discharge x 0.5
(calibration_q_scale; the 1D solve uses the same Q); Manning n 0.03 on the
whole roughness raster; n 0.03 on the channel (pixels at channel_manning_n)
plus one grid cell either side.
`compare`: per variant, the 1.7 gap per reach (median, p5, p95 over flowing
reaches, phase1_gvf_gap.cell_gaps), the median slope excess
(phase2_gap_slope.reach_table) and the steady check. For the baseline also the
share of centreline cells whose native roughness pixels mix channel and land
values, and the share of those whose end level stands above the 75th percentile of the cell's
subgrid elevation (banks wet on the cell's faces).

Writes phase2_ebro_sweep_summary.csv and _gap.png to <results_dir>/figures/analysis/.

Usage: python analysis/phase2_ebro_sweep.py run|compare [basin_id] [--variants ...] [--cores 14]
"""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml
from _common import REPO_ROOT, analysis_path, script_parser, site_path
from _phase1 import basin_paths, git_hash, load_config, pct, write_summary
from phase1_gvf_gap import cell_gaps
from phase2_gap_slope import reach_table
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

RDM = ("river_processing", "river_depth_modelling")
VARIANTS = {
    "baseline": {},
    "nr_levels_50": {"overlay": {"sfincs": {"subgrid": {"nr_levels": 50}}}},
    "advection_0": {"sfincs": {"advection": 0}},
    "nuvisc_0": {"sfincs": {"nuvisc": 0.0}},
    "huthresh_0.05": {"sfincs": {"huthresh": 0.05}},
    "q_half": {"q_scale": 0.5},
    "manning_uniform": {"roughness": "uniform"},
    "manning_corridor": {"roughness": "corridor"},
}
N_UNIFORM = 0.03


def scratch_root() -> Path:
    return (
        Path(
            os.environ.get("GCFM_PLAN_SCRATCH")
            or site_path("results_dir").parent / "scratch_plan"
        )
        / "p2a"
    )


def _overlay(spec: dict) -> dict:
    rdm: dict = {}
    if "sfincs" in spec:
        rdm["calibration_sfincs_settings"] = spec["sfincs"]
    if "q_scale" in spec:
        rdm["calibration_q_scale"] = spec["q_scale"]
    out = dict(spec.get("overlay", {}))
    if rdm:
        out[RDM[0]] = {RDM[1]: rdm}
    return out


def _set_roughness(path: Path, mode: str, channel_n: float, grid_dx_m: float) -> None:
    with rasterio.open(path) as src:
        n = src.read(1)
        profile = src.profile
        px = abs(src.transform.a)
        nodata = src.nodata
    valid = n != nodata if nodata is not None else np.isfinite(n)
    if mode == "uniform":
        target = valid
    else:
        channel = valid & (np.abs(n - channel_n) < 1e-4)
        target = valid & binary_dilation(
            channel, iterations=max(1, round(grid_dx_m / px))
        )
    n = np.where(target, N_UNIFORM, n).astype(profile["dtype"])
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(n, 1)


def run(basin_id: str, variants: list[str], cores: int) -> None:
    from src.grid import load_grid_def

    src_prep = basin_paths(basin_id)["prep"]
    cfg = load_config()
    channel_n = float(cfg["river_processing"]["channel_manning_n"])
    grid = load_grid_def(basin_paths(basin_id)["sfincs_grid"])
    dx = float(np.hypot(grid["transform"].a, grid["transform"].d))
    for name in variants:
        spec = VARIANTS[name]
        root = scratch_root() / name
        prep = root / basin_id / "preprocessing_inputs"
        if prep.exists():
            shutil.rmtree(prep)
        shutil.copytree(src_prep, prep)
        if "roughness" in spec:
            _set_roughness(
                prep / "domain" / f"{basin_id}_roughness.tif",
                spec["roughness"],
                channel_n,
                dx,
            )
        overlay = root / "overlay.yml"
        overlay.write_text(yaml.safe_dump(_overlay(spec)), encoding="utf-8")
        cmd = [
            "snakemake",
            "preprocess",
            "--cores",
            str(cores),
            "--configfile",
            "config/config.yml",
            str(overlay),
            "--forcerun",
            "modelled_depth_estimation",
            "--config",
            f"target_basins=[{basin_id}]",
            "target_scenarios=['coast_100_03SLR']",
        ]
        env = dict(os.environ, GCFM_RESULTS_DIR=str(root))
        with open(root / f"snakemake_{basin_id}.log", "w") as log:
            code = subprocess.run(
                cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=log
            ).returncode
        print(f"{name}: exit {code}", flush=True)
        if code:
            raise SystemExit(
                f"{name}: snakemake failed, see {root / f'snakemake_{basin_id}.log'}"
            )


def mixed_faces(basin_id: str, cells: pd.DataFrame, channel_n: float) -> dict:
    p = basin_paths(basin_id)
    with rasterio.open(p["prep"] / "domain" / f"{basin_id}_roughness.tif") as src:
        n = src.read(1)
        tr = src.transform
    with rasterio.open(p["round1"] / "subgrid" / "dep_subgrid.tif") as src:
        z = src.read(1)
        tz = src.transform
    half = 50.0
    mixed, banks_wet = 0, 0
    for x, y, zs in zip(cells["x"], cells["y"], cells["zs_end_m"]):
        r0, c0 = rasterio.transform.rowcol(tr, x - half, y + half)
        r1, c1 = rasterio.transform.rowcol(tr, x + half, y - half)
        block = n[max(r0, 0) : r1, max(c0, 0) : c1]
        is_ch = np.abs(block - channel_n) < 1e-4
        if not (is_ch.any() and (~is_ch).any()):
            continue
        mixed += 1
        zr0, zc0 = rasterio.transform.rowcol(tz, x - half, y + half)
        zr1, zc1 = rasterio.transform.rowcol(tz, x + half, y - half)
        zb = z[max(zr0, 0) : zr1, max(zc0, 0) : zc1]
        zb = zb[np.isfinite(zb)]
        if zb.size and zs > np.percentile(zb, 75):
            banks_wet += 1
    return {
        "share_cells_mixed_roughness": mixed / len(cells),
        "share_mixed_with_banks_wet": banks_wet / mixed if mixed else np.nan,
    }


def compare(basin_id: str, variants: list[str]) -> pd.DataFrame:
    cfg = load_config()
    channel_n = float(cfg["river_processing"]["channel_manning_n"])
    tag = git_hash()
    results_dir = os.environ.get("GCFM_RESULTS_DIR")
    base = scratch_root()
    rows = []
    for name in variants:
        os.environ["GCFM_RESULTS_DIR"] = str(base / name)
        cells, grid, _bed, rivers, q_calib, _sea = cell_gaps(
            basin_id, cfg, VARIANTS[name].get("q_scale", 1.0)
        )
        nodes = pd.read_csv(basin_paths(basin_id)["node_table"])
        _, k = cKDTree(nodes[["x", "y"]].to_numpy()).query(cells[["x", "y"]].to_numpy())
        cells["dist_to_sea_km"] = nodes["dist_to_sea_m"].to_numpy()[k] / 1000.0
        t = reach_table(cells, rivers, grid, q_calib)
        flowing = t[t["q_calib_m3s"] > 0]
        check = pd.read_csv(basin_paths(basin_id)["round1"] / "steady_check.csv").iloc[
            0
        ]
        row = {
            "variant": name,
            "gap_median_m": pct(flowing["gap_median_m"], 50),
            "gap_p5_m": pct(flowing["gap_median_m"], 5),
            "gap_p95_m": pct(flowing["gap_median_m"], 95),
            "slope_excess_median_m_per_km": pct(flowing["slope_excess_m_per_km"], 50),
            "cells_rising": int(check["cells_rising"]),
            "run_hr": float(check["run_hr"]),
        }
        if name == "baseline":
            row.update(mixed_faces(basin_id, cells, channel_n))
        rows.append(row)
        print(f"{name}: gap median {row['gap_median_m']:+.3f} m", flush=True)
    if results_dir is None:
        os.environ.pop("GCFM_RESULTS_DIR")
    else:
        os.environ["GCFM_RESULTS_DIR"] = results_dir
    df = write_summary(rows, analysis_path(None, __file__, "summary.csv"))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(df))
    ax.errorbar(
        df["gap_median_m"],
        y,
        xerr=[
            df["gap_median_m"] - df["gap_p5_m"],
            df["gap_p95_m"] - df["gap_median_m"],
        ],
        fmt="o",
        capsize=3,
    )
    ax.axvline(0, color="k", lw=0.8)
    ax.axvspan(
        -0.2, 0.2, color="tab:green", alpha=0.12, label="gate: |median| <= 0.2 m"
    )
    ax.set_yticks(y, df["variant"])
    ax.invert_yaxis()
    ax.set_xlabel("1D-2D gap per reach: median, p5-p95 (m)")
    ax.set_title(f"phase2 2a Ebro sweep, round 1 | {basin_id} | {tag}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(analysis_path(None, __file__, "gap.png"), dpi=120)
    plt.close(fig)
    return df


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("action", choices=["run", "compare"])
    parser.add_argument("basin_id", nargs="?", default="2433835")
    parser.add_argument(
        "--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS)
    )
    parser.add_argument("--cores", type=int, default=14)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    sys.path.insert(0, str(REPO_ROOT / "workflow"))
    if args.action == "run":
        run(args.basin_id, args.variants, args.cores)
    else:
        df = compare(args.basin_id, args.variants)
        with pd.option_context("display.width", 250, "display.max_columns", None):
            print(df.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
