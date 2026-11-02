"""Phase 2a: SFINCS steady level in a synthetic sub-cell trench, advection on and off, against normal depth.

A rectangular trench (width TRENCH_WIDTH_M, Manning n, bed slope BED_SLOPE)
between walls WALL_HEIGHT_M above the bed carries a constant discharge to a
water-level boundary held at normal depth, so the exact steady solution is
uniform flow at normal depth, h = (Q n / (W sqrt(S)))^(3/5) (wide channel,
as the subgrid conveyance h^(5/3)/n per pixel). Velocity is uniform, so the
advection term contributes nothing physically. The trench is laid out three
ways on the model grid:

  aligned: straight along a grid row, the trench exactly one cell wide;
  diagonal: straight at 45 degrees, a staircase of cells;
  meander: a sine centreline (amplitude MEANDER_AMPLITUDE_M, wavelength
  MEANDER_WAVELENGTH_M), the cell path turning both ways.

Each layout runs at each discharge with advection 0 and 1, at each SFINCS
time-step factor alpha in ALPHAS: at alpha 0.5 a deep trench one cell wide
goes unstable in subgrid mode and stands metres too high with or without
advection, so the advection effect is read at the stable alpha. Grid size, subgrid
pixels and levels come from config.yml (sfincs.grid, sfincs.subgrid). The
level error per centreline cell is SFINCS end level minus bed + normal
depth, away from both ends (EDGE_SKIP_M).

Writes advection_channel_summary.csv and the same-name _profiles.png and
_map.png to <results_dir>/figures/analysis/; model runs to --run-dir.

Usage: python analysis/phase2_advection_channel.py [--run-dir DIR] [--skip-run]
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import REPO_ROOT, analysis_path, script_parser, site_path
from scipy.spatial import cKDTree

log = logging.getLogger("advection_channel")

TRENCH_WIDTH_M = 100.0
MANNING_N = 0.03
BED_SLOPE = 1.0e-4
WALL_HEIGHT_M = 25.0
CHANNEL_LENGTH_M = 20_000.0
MEANDER_AMPLITUDE_M = 1500.0
MEANDER_WAVELENGTH_M = 6000.0
CORRIDOR_HALF_WIDTH_M = 200.0
EDGE_SKIP_M = 2000.0
DISCHARGES_M3S = (3000.0, 500.0)
LAYOUTS = ("aligned", "diagonal", "meander")
# SFINCS time-step factor; 0.5 is the SFINCS default and the pipeline's value
ALPHAS = (0.5, 0.25)
RAMP_HR = 2.0
RUN_HR = 12.0
CENTRELINE_STEP_M = 2.0
EPSG = 32631
ORIGIN_XY = (500_000.0, 5_000_000.0)


def normal_depth_m(q_m3s: float, width_m: float, n: float, slope: float) -> float:
    """Return wide-channel Manning normal depth (m): Q = W h^(5/3) sqrt(S) / n."""
    return float((q_m3s * n / (width_m * np.sqrt(slope))) ** 0.6)


def centreline(layout: str, length_m: float, step_m: float = CENTRELINE_STEP_M):
    """Return (xy, s): points along the centreline from upstream, and their path distance (m)."""
    if layout == "aligned":
        t = np.arange(0.0, length_m + step_m, step_m)
        xy = np.c_[t, np.zeros_like(t)]
    elif layout == "diagonal":
        t = np.arange(0.0, length_m + step_m, step_m) / np.sqrt(2.0)
        xy = np.c_[t, t]
    elif layout == "meander":
        t = np.arange(0.0, length_m, step_m / 4.0)
        y = MEANDER_AMPLITUDE_M * np.sin(2 * np.pi * t / MEANDER_WAVELENGTH_M)
        s = np.r_[0.0, np.cumsum(np.hypot(np.diff(t), np.diff(y)))]
        keep = s <= length_m
        s_new = np.arange(0.0, s[keep][-1], step_m)
        xy = np.c_[np.interp(s_new, s, t), np.interp(s_new, s, y)]
    else:
        raise ValueError(f"unknown layout {layout!r}")
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    return xy, s


def channel_rasters(
    layout: str, dx_m: float, nr_subgrid_pixels: int, length_m: float = CHANNEL_LENGTH_M
) -> dict:
    """Return the subgrid-resolution bed raster of one layout, and the grid around it.

    Bed: BED_SLOPE * (L - s) in the trench (L = length_m, s the nearest
    centreline point's distance from the inflow), WALL_HEIGHT_M higher outside.
    The trench runs on past both ends so the outlet's boundary cells lie in
    it. The grid origin puts an aligned trench exactly on one cell row.
    """
    ext = CORRIDOR_HALF_WIDTH_M + dx_m
    xy, s = centreline(layout, length_m + 2 * ext)
    s = s - ext
    length = length_m
    pad = CORRIDOR_HALF_WIDTH_M + 2 * dx_m
    x0 = np.floor((xy[:, 0].min() - pad) / dx_m) * dx_m
    y0 = np.floor((xy[:, 1].min() - pad) / dx_m) * dx_m - dx_m / 2.0
    mmax = int(np.ceil((xy[:, 0].max() + pad - x0) / dx_m))
    nmax = int(np.ceil((xy[:, 1].max() + pad - y0) / dx_m))
    px = dx_m / nr_subgrid_pixels
    xs = x0 + (np.arange(mmax * nr_subgrid_pixels) + 0.5) * px
    ys = y0 + (np.arange(nmax * nr_subgrid_pixels) + 0.5) * px
    gx, gy = np.meshgrid(xs, ys)
    dist, k = cKDTree(xy).query(np.c_[gx.ravel(), gy.ravel()])
    dist, s_px = dist.reshape(gx.shape), s[k].reshape(gx.shape)
    bed = BED_SLOPE * (length - s_px)
    z = np.where(dist <= TRENCH_WIDTH_M / 2.0, bed, bed + WALL_HEIGHT_M)
    return {
        "xy": xy,
        "s": s,
        "length_m": length,
        "dx_m": dx_m,
        "x0": float(x0),
        "y0": float(y0),
        "mmax": mmax,
        "nmax": nmax,
        "px_m": px,
        "z": z.astype(np.float32),
        "trench": dist <= TRENCH_WIDTH_M / 2.0,
    }


def _write_raster(path: Path, data: np.ndarray, r: dict) -> None:
    import rasterio
    from rasterio.transform import from_origin

    top = r["y0"] + data.shape[0] * r["px_m"]
    transform = from_origin(
        r["x0"] + ORIGIN_XY[0], top + ORIGIN_XY[1], r["px_m"], r["px_m"]
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=f"EPSG:{EPSG}",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(np.flipud(data).astype(np.float32), 1)


def build_case(
    root: Path,
    layout: str,
    q_m3s: float,
    advection: int,
    alpha: float,
    cfg: dict,
    length_m: float = CHANNEL_LENGTH_M,
) -> dict:
    """Write one SFINCS model of the trench to ``root``; return its rasters and normal depth."""
    import geopandas as gpd
    import yaml
    from hydromt_sfincs import SfincsModel
    from shapely.geometry import LineString, Point
    from src.hydromt_sfincs_patch import (
        patch_nearest_extrapolation,
        patch_subgrid_write,
        raster_catalog_entry,
    )

    patch_subgrid_write()
    patch_nearest_extrapolation()
    dx = float(cfg["sfincs"]["grid"]["resolution_m"])
    sub = cfg["sfincs"]["subgrid"]
    r = channel_rasters(layout, dx, int(sub["nr_subgrid_pixels"]), length_m)
    h_n = normal_depth_m(q_m3s, TRENCH_WIDTH_M, MANNING_N, BED_SLOPE)
    root.mkdir(parents=True, exist_ok=True)
    _write_raster(root / "bed.tif", r["z"], r)
    _write_raster(root / "manning.tif", np.full_like(r["z"], MANNING_N), r)
    catalog = {
        "meta": {"root": str(root)},
        "bed": raster_catalog_entry(root / "bed.tif"),
        "manning": raster_catalog_entry(root / "manning.tif"),
    }
    (root / "catalog.yml").write_text(yaml.safe_dump(catalog), encoding="utf-8")

    sf = SfincsModel(
        data_libs=[str(root / "catalog.yml")],
        root=str(root),
        mode="w+",
        write_gis=False,
    )
    sf.grid.create(
        x0=r["x0"] + ORIGIN_XY[0],
        y0=r["y0"] + ORIGIN_XY[1],
        dx=dx,
        dy=dx,
        nmax=r["nmax"],
        mmax=r["mmax"],
        rotation=0.0,
        epsg=EPSG,
    )
    tref = datetime(2000, 1, 1)
    sf.config.update(
        {
            "tref": tref,
            "tstart": tref,
            "tstop": tref + timedelta(hours=RUN_HR),
            "dtmapout": 3600,
            "dthisout": 3600,
            "advection": advection,
            "alpha": alpha,
            "zsini": float(BED_SLOPE * r["length_m"] + h_n),
        }
    )
    sf.elevation.create(elevation_list=[{"elevation": "bed"}])
    line = LineString(r["xy"] + np.array(ORIGIN_XY))
    corridor = gpd.GeoDataFrame(
        geometry=[line.buffer(CORRIDOR_HALF_WIDTH_M, cap_style="flat")], crs=EPSG
    )
    sf.mask.create_active(include_polygon=corridor, reset_mask=True)
    # The outlet boundary is the one trench cell at the line's end; hydromt's
    # create_boundary would also take the wall cells beside it.
    mask = sf.grid.data["mask"]
    end_x, end_y = r["xy"][-1] + np.array(ORIGIN_XY)
    iy = int(np.abs(mask["y"].values - end_y).argmin())
    ix = int(np.abs(mask["x"].values - end_x).argmin())
    if mask.values[iy, ix] != 1:
        raise RuntimeError(f"outlet cell ({iy}, {ix}) is not active")
    mask.values[iy, ix] = 2
    sf.roughness.create(roughness_list=[{"manning": "manning"}])
    sf.subgrid.create(
        elevation_list=[{"elevation": "bed"}],
        roughness_list=[{"manning": "manning"}],
        river_list=[],
        nr_subgrid_pixels=int(sub["nr_subgrid_pixels"]),
        nr_levels=int(sub["nr_levels"]),
        nrmax=int(sub["nrmax"]),
    )
    sf.water_level.create_boundary_points_from_mask(bnd_dist=5000.0)
    # Normal-depth level at the outlet, beyond s = L.
    sf.water_level.create_timeseries(
        shape="constant", offset=h_n + BED_SLOPE * (r["length_m"] - r["s"][-1])
    )
    times = pd.date_range(tref, tref + timedelta(hours=RUN_HR), freq="10min")
    hr = (times - times[0]).total_seconds().to_numpy() / 3600.0
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.clip(hr / RAMP_HR, 0.0, 1.0))
    start = Point(*(r["xy"][np.searchsorted(r["s"], 0.0)] + np.array(ORIGIN_XY)))
    src = gpd.GeoDataFrame({"index": [1]}, geometry=[start], crs=EPSG)
    q = pd.DataFrame({1: q_m3s * (0.1 + 0.9 * ramp)}, index=times)
    sf.discharge_points.create(timeseries=q, locations=src)
    sf.write()
    return {**r, "h_n": h_n}


def level_errors(root: Path, r: dict) -> pd.DataFrame:
    """Return per centreline cell: path distance, SFINCS end level, normal-depth level and error (m)."""
    import xarray as xr

    with xr.open_dataset(root / "sfincs_map.nc") as ds:
        zs = ds["zs"].values
    col = np.floor((r["xy"][:, 0] - r["x0"]) / r["dx_m"]).astype(int)
    row = np.floor((r["xy"][:, 1] - r["y0"]) / r["dx_m"]).astype(int)
    cells = (
        pd.DataFrame({"row": row, "col": col, "s_m": r["s"]})
        .groupby(["row", "col"], sort=False)["s_m"]
        .median()
    )
    out = cells.reset_index()
    out["zs_end_m"] = zs[-1][out["row"], out["col"]]
    out["zs_prev_m"] = zs[-2][out["row"], out["col"]]
    out["level_normal_m"] = BED_SLOPE * (r["length_m"] - out["s_m"]) + r["h_n"]
    out["error_m"] = out["zs_end_m"] - out["level_normal_m"]
    inner = (out["s_m"] > EDGE_SKIP_M) & (out["s_m"] < r["length_m"] - EDGE_SKIP_M)
    return out[inner].sort_values("s_m").reset_index(drop=True)


def summarise(
    err: pd.DataFrame, layout: str, q_m3s: float, advection: int, alpha: float, r: dict
) -> dict:
    steps = np.abs(
        np.diff(err.sort_values("s_m")[["row", "col"]].to_numpy(), axis=0)
    ).sum()
    span = err["s_m"].max() - err["s_m"].min()
    return {
        "layout": layout,
        "q_m3s": q_m3s,
        "advection": advection,
        "alpha": alpha,
        "normal_depth_m": r["h_n"],
        "n_cells": len(err),
        "path_ratio": float(steps * r["dx_m"] / span) if span > 0 else np.nan,
        "error_median_m": float(err["error_m"].median()),
        "error_p95_m": float(err["error_m"].quantile(0.95)),
        # positive when the error grows upstream, as the slope excess of phase2_gap_slope.py
        "error_slope_m_per_km": float(
            np.polyfit((r["length_m"] - err["s_m"]) / 1000.0, err["error_m"], 1)[0]
        ),
        "last_hour_change_max_m": float(
            (err["zs_end_m"] - err["zs_prev_m"]).abs().max()
        ),
    }


def plot(results: dict, tag: str) -> None:
    combos = [(q, a) for q in DISCHARGES_M3S for a in ALPHAS]
    fig, axes = plt.subplots(
        len(combos),
        len(LAYOUTS),
        figsize=(15, 3 * len(combos)),
        sharey="row",
        squeeze=False,
    )
    for i, (q, alpha) in enumerate(combos):
        for j, layout in enumerate(LAYOUTS):
            ax = axes[i, j]
            for adv, color in ((0, "tab:blue"), (1, "tab:red")):
                e = results.get((layout, q, adv, alpha))
                if e is not None:
                    ax.plot(
                        e["s_m"] / 1000.0,
                        e["error_m"],
                        color=color,
                        lw=1,
                        label=f"advection {adv}",
                    )
            ax.axhline(0, color="k", lw=0.8)
            ax.set_title(f"{layout}, Q {q:.0f} m3/s, alpha {alpha:g}", fontsize=9)
            ax.grid(alpha=0.3)
            if i == len(combos) - 1:
                ax.set_xlabel("distance from the inflow (km)")
            if j == 0:
                ax.set_ylabel("SFINCS - normal-depth level (m)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"phase2 2a synthetic trench W {TRENCH_WIDTH_M:.0f} m, n {MANNING_N}, S {BED_SLOPE:g}: "
        f"steady level error | {tag}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(analysis_path(None, __file__, "profiles.png"), dpi=120)
    plt.close(fig)


def plot_map(models: dict, results: dict, tag: str) -> None:
    q, alpha = DISCHARGES_M3S[0], min(ALPHAS)
    fig, axes = plt.subplots(2, len(LAYOUTS), figsize=(15, 9), squeeze=False)
    for j, layout in enumerate(LAYOUTS):
        for i, adv in enumerate((0, 1)):
            key = (layout, q, adv, alpha)
            if key not in results:
                continue
            ax, r, e = axes[i, j], models[key], results[key]
            dx = r["dx_m"]
            ax.imshow(
                r["trench"],
                origin="lower",
                extent=(
                    r["x0"],
                    r["x0"] + r["mmax"] * dx,
                    r["y0"],
                    r["y0"] + r["nmax"] * dx,
                ),
                cmap="Greys",
                alpha=0.4,
            )
            sc = ax.scatter(
                r["x0"] + (e["col"] + 0.5) * dx,
                r["y0"] + (e["row"] + 0.5) * dx,
                c=e["error_m"],
                cmap="RdBu_r",
                norm=matplotlib.colors.TwoSlopeNorm(vcenter=0.0, vmin=-2.0, vmax=12.0),
                s=6,
            )
            ax.set_aspect("equal")
            ax.set_title(f"{layout}, advection {adv}: centreline cells", fontsize=9)
            fig.colorbar(sc, ax=ax, shrink=0.7, label="level error (m)")
    fig.suptitle(
        f"phase2 2a synthetic trench, Q {q:.0f} m3/s, alpha {alpha:g} | {tag}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(analysis_path(None, __file__, "map.png"), dpi=120)
    plt.close(fig)


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    sys.path.insert(0, str(REPO_ROOT / "workflow"))
    from _phase1 import git_hash, load_config
    from src.sfincs_run import run_sfincs_subprocess

    cfg, tag = load_config(), git_hash()
    dx = float(cfg["sfincs"]["grid"]["resolution_m"])
    nr_subgrid_pixels = int(cfg["sfincs"]["subgrid"]["nr_subgrid_pixels"])
    run_dir = (
        args.run_dir
        or site_path("results_dir").parent / "scratch" / "advection_channel"
    )
    models, results, rows = {}, {}, []
    for layout in LAYOUTS:
        for q in DISCHARGES_M3S:
            for alpha in ALPHAS:
                for adv in (0, 1):
                    key = (layout, q, adv, alpha)
                    root = run_dir / f"{layout}_q{q:.0f}_adv{adv}_alpha{alpha:g}"
                    if args.skip_run:
                        r = channel_rasters(layout, dx, nr_subgrid_pixels)
                        r["h_n"] = normal_depth_m(
                            q, TRENCH_WIDTH_M, MANNING_N, BED_SLOPE
                        )
                    else:
                        r = build_case(root, layout, q, adv, alpha, cfg)
                        run_sfincs_subprocess(
                            site_path("sfincs_exe"),
                            root,
                            3600,
                            log,
                            label=root.name,
                            n_threads=args.threads,
                        )
                    err = level_errors(root, r)
                    models[key], results[key] = r, err
                    rows.append(summarise(err, layout, q, adv, alpha, r))
                    print(
                        pd.DataFrame(rows[-1:])
                        .round(3)
                        .to_string(index=False, header=len(rows) == 1),
                        flush=True,
                    )
    summary = pd.DataFrame(rows)
    summary.to_csv(
        analysis_path(None, __file__, "summary.csv"), index=False, float_format="%.4g"
    )
    plot(results, tag)
    plot_map(models, results, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
