"""Phase 6.2: crest level under a held sea against the design storm-tide hydrograph.

`run`: copies the basin's round-1 calibration model (rivers ramped to the
protection discharge and held; sea ramped to each station's storm tide at the
coastal protection RP and held) to a scratch folder (GCFM_PLAN_SCRATCH, default
<results_dir>/../scratch_phase1; p62/<basin>/hydrograph/) and replaces its sea:
calm level while the rivers settle (round 1's run length), a 6 h half-cosine to
the hydrograph's first value, then the design storm tide over HALF_WINDOW_HR
either side of its peak: average tide plus the surge peaking on high water
(src.surge.build_design_surge_matrix, as rule 14 forces events) at the next
COAST-RP return period up, its surge scaled per station so the peak equals
round 1's held level at the protection RP. Rivers stay held. The
production round-1 folder is not touched.
`compare`: the water-side maximum level along the round-1 production weir
(the larger zsmax of the cells within one cell of each weir vertex) in the
hydrograph run minus round 1, on weir vertices within LOWER_KM of a
water-level boundary point.

Writes per basin 90_phase6_crest_hydrograph_{points.csv,map.png} to
<results_dir>/<basin>/figures/analysis/ and phase6_crest_hydrograph_summary.csv
to <results_dir>/figures/analysis/.

Usage: python analysis/phase6_crest_hydrograph.py run|compare [basin_id ...]
"""

import json
import logging
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from _common import analysis_path, script_parser, site_path
from _phase1 import PHASE1_BASINS, basin_paths, pct, write_summary
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)
TABULATED_RP = (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000)  # COAST-RP
HALF_WINDOW_HR = 60
BLEND_HR = 6
LOWER_KM = 50.0
TIME_FMT = "%Y%m%d %H%M%S"


def scratch_dir(basin_id: str) -> Path:
    root = os.environ.get("GCFM_PLAN_SCRATCH") or str(
        site_path("results_dir").parent / "scratch_phase1"
    )
    return Path(root) / "p62" / basin_id / "hydrograph"


def _read_inp(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _write_inp(path: Path, inp: dict[str, str]) -> None:
    path.write_text("".join(f"{k:<20} = {v}\n" for k, v in inp.items()))


def sea_series(basin_id: str, bnd_xy: np.ndarray, crs, run_hr: float):
    """Return (hours, levels[time, bnd]) of the hydrograph sea, and the peak hour."""
    from pyproj import Transformer
    from src.surge import (
        build_design_surge_matrix,
        read_calm_sea_level_m,
        storm_tide_at_rp_interpolated,
    )

    p = basin_paths(basin_id)
    rp = float(json.loads(p["protection_levels"].read_text())["surge_protection_rp_yr"])
    with xr.open_dataset(p["surge_forcing"], decode_times=False) as ds:
        calm = read_calm_sea_level_m(ds)
        # COAST-RP tabulates fixed RPs: take the surge shape of the next one up
        # and scale it, per station, so the peak is round 1's interpolated level.
        rp_tab = min(r for r in TABULATED_RP if r >= rp)
        tide = build_design_surge_matrix(ds, "tide")  # (station, time), hourly
        surge = build_design_surge_matrix(ds, rp_tab) - tide
        target = np.asarray(storm_tide_at_rp_interpolated(ds, rp), dtype=float)
        k = np.argmax(surge + tide, axis=1)
        idx = np.arange(surge.shape[0])
        scale = (target - tide[idx, k]) / surge[idx, k]
        hg = tide + scale[:, None] * surge
        sx, sy = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
            ds["longitude"].values, ds["latitude"].values
        )
    dist, st = cKDTree(np.column_stack([sx, sy])).query(bnd_xy)
    if dist.max() > 1.0:
        raise ValueError(
            f"{basin_id}: a boundary point is {dist.max():.0f} m from every surge station"
        )
    hg = hg[st]  # (bnd, time)
    peak = int(np.argmax(hg.max(axis=0)))
    lo, hi = max(peak - HALF_WINDOW_HR, 0), min(peak + HALF_WINDOW_HR, hg.shape[1] - 1)
    window = hg[:, lo : hi + 1]
    hold = np.arange(0.0, run_hr + 1e-9, 1.0)
    blend_t = np.arange(1.0, BLEND_HR + 1e-9, 1.0)
    w = 0.5 * (1.0 - np.cos(np.pi * blend_t / BLEND_HR))
    blend = calm + w[None, :] * (window[:, :1] - calm)
    hours = np.r_[
        hold, run_hr + blend_t, run_hr + BLEND_HR + np.arange(1, window.shape[1] + 1)
    ]
    levels = np.c_[np.full((len(bnd_xy), hold.size), calm), blend, window]
    return hours, levels.T, run_hr + BLEND_HR + 1 + (peak - lo)


def run(basin_id: str, threads: int) -> None:
    from src.grid import load_grid_def
    from src.io import site_setting
    from src.sfincs_run import run_sfincs_subprocess

    src, dst = basin_paths(basin_id)["round1"], scratch_dir(basin_id)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("sfincs_map.nc", "sfincs_his.nc", "*.rst"),
    )
    inp = _read_inp(dst / "sfincs.inp")
    tref = datetime.strptime(inp["tref"], TIME_FMT)
    run_hr = (datetime.strptime(inp["tstop"], TIME_FMT) - tref).total_seconds() / 3600.0
    bnd = np.loadtxt(dst / "sfincs.bnd", ndmin=2)
    crs = load_grid_def(basin_paths(basin_id)["sfincs_grid"])["crs"]
    hours, levels, peak_hr = sea_series(basin_id, bnd[:, :2], crs, run_hr)
    np.savetxt(
        dst / "sfincs.bzs",
        np.c_[hours * 3600.0, levels],
        fmt=["%.1f"] + ["%.3f"] * levels.shape[1],
    )
    dis = np.loadtxt(dst / "sfincs.dis", ndmin=2)
    total_s = float(hours[-1] * 3600.0)
    np.savetxt(
        dst / "sfincs.dis", np.r_[dis, np.r_[total_s, dis[-1, 1:]][None, :]], fmt="%.2f"
    )
    inp["tstop"] = (tref + timedelta(seconds=total_s)).strftime(TIME_FMT)
    inp["dtmaxout"] = f"{total_s:.1f}"
    inp.pop("trstout", None)
    _write_inp(dst / "sfincs.inp", inp)
    log.info(
        f"{basin_id}: sea calm for {run_hr:.0f} h, hydrograph to {hours[-1]:.0f} h (peak at {peak_hr:.0f} h); "
        f"running SFINCS in {dst}"
    )
    run_sfincs_subprocess(
        site_setting("sfincs_exe"),
        dst,
        12 * 3600,
        log,
        label=f"6.2 hydrograph ({basin_id})",
        n_threads=threads,
    )


def _zsmax(run_dir: Path):
    with xr.open_dataset(run_dir / "sfincs_map.nc") as ds:
        return ds["x"].values, ds["y"].values, np.nanmax(ds["zsmax"].values, axis=0)


def _weir_points(path: Path, crs, step_m: float) -> np.ndarray:
    weir = gpd.read_file(path).to_crs(crs)
    pts = []
    for g in weir.geometry:
        for line in getattr(g, "geoms", [g]):
            d = np.arange(0.0, line.length + 1e-9, step_m)
            pts += [(q.x, q.y) for q in (line.interpolate(v) for v in d)]
    return np.array(pts)


def compare(basin_id: str) -> dict:
    from src.grid import load_grid_def

    p = basin_paths(basin_id)
    crs = load_grid_def(p["sfincs_grid"])["crs"]
    x, y, held = _zsmax(p["round1"])
    _, _, hydro = _zsmax(scratch_dir(basin_id))
    dx = float(np.hypot(x[0, 1] - x[0, 0], y[0, 1] - y[0, 0]))
    pts = _weir_points(
        p["prep"] / "domain" / f"{basin_id}_coastal_protection_weir.gpkg", crs, dx
    )
    tree = cKDTree(np.column_stack([x.ravel(), y.ravel()]))
    near = tree.query_ball_point(pts, 1.5 * dx)

    def side_max(z):
        flat = z.ravel()
        return np.array([np.nanmax(flat[k]) if k else np.nan for k in near])

    bnd = np.loadtxt(p["round1"] / "sfincs.bnd", ndmin=2)[:, :2]
    d_bnd, _ = cKDTree(bnd).query(pts)
    t = pd.DataFrame(
        {
            "x": pts[:, 0],
            "y": pts[:, 1],
            "dist_to_boundary_km": d_bnd / 1e3,
            "zsmax_held_m": side_max(held),
            "zsmax_hydrograph_m": side_max(hydro),
        }
    )
    t["diff_m"] = t["zsmax_hydrograph_m"] - t["zsmax_held_m"]
    t.to_csv(
        analysis_path(basin_id, __file__, "points.csv"),
        index=False,
        float_format="%.3f",
    )
    low = t[(t["dist_to_boundary_km"] <= LOWER_KM) & np.isfinite(t["diff_m"])]
    _figure(basin_id, t, low)
    return {
        "basin_id": basin_id,
        "n_points_lower": int(len(low)),
        "diff_p5_m": pct(low["diff_m"].to_numpy(), 5),
        "diff_median_m": pct(low["diff_m"].to_numpy(), 50),
        "diff_p95_m": pct(low["diff_m"].to_numpy(), 95),
        "diff_max_m": float(low["diff_m"].max()) if len(low) else np.nan,
        "share_hydrograph_higher_0.1m": float((low["diff_m"] > 0.1).mean())
        if len(low)
        else np.nan,
    }


def _figure(basin_id: str, t: pd.DataFrame, low: pd.DataFrame) -> None:
    fig, ax = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    sc = ax[0].scatter(
        t["x"], t["y"], c=t["diff_m"], s=3, cmap="RdBu_r", vmin=-0.5, vmax=0.5
    )
    fig.colorbar(
        sc, ax=ax[0], label="zsmax hydrograph - held (m), water side of the weir"
    )
    ax[0].set_aspect("equal")
    ax[0].tick_params(labelsize=7)
    ax[0].set_title(
        f"{basin_id}: crest basis, hydrograph sea minus held sea", fontsize=9
    )
    ax[1].hist(low["diff_m"].dropna(), bins=60, color="tab:blue")
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set_xlabel(f"difference (m), weir within {LOWER_KM:g} km of the sea boundary")
    ax[1].set_ylabel("weir vertices")
    fig.suptitle(f"phase6 6.2 | {basin_id} | held vs hydrograph sea", fontsize=10)
    fig.tight_layout()
    fig.savefig(analysis_path(basin_id, __file__, "map.png"), dpi=120)
    plt.close(fig)


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("action", choices=["run", "compare"])
    parser.add_argument("basin_ids", nargs="*", default=PHASE1_BASINS)
    parser.add_argument("--threads", type=int, default=14)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    if args.action == "run":
        for b in args.basin_ids:
            run(str(b), args.threads)
        return 0
    rows = [compare(str(b)) for b in args.basin_ids]
    print(
        write_summary(rows, analysis_path(None, __file__, "summary.csv")).to_string(
            index=False
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
