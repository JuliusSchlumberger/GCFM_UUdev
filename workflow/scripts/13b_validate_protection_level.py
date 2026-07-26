"""
13b_validate_protection_level.py -- validate that the BUILT production
model (real burned channel + real calibrated/coastal weir + real floodplain,
not a re-confined corridor) actually holds at its own protection standard.

Runs two short, steady discharge scenarios against the production model's
OWN static files (dep/msk/ind/manning/sbg/weir/obs) -- referenced via
relative "../" paths, not rebuilt, mirroring 14_run_spinup.py's proven
approach for reusing an already-built model with different timing/forcing:

  1. protection-level design discharge (the same target rule
     modelled_depth_estimation's own calibration used --
     protection_levels.json's riverine_rp_yr, bankfull as the fallback).
  2. that same discharge x sfincs.protection_validation.higher_rp_factor
     (a modest beyond-design-standard event).

Per-point steady discharge is derived by matching each of the production
model's own sfincs.src points (by nearest coordinate) to its river_forcing.nc
crossing, then looking up that crossing's own discharge_rp_table at the
target return period -- NOT by re-deriving which crossings ended up in
sfincs.src (that filtering/snapping logic lives in 13_build_sfincs.py and
re-deriving it here would risk a point-order mismatch); this way the
mapping is anchored directly to the real, already-placed source points.

The coastal boundary is forced flat and dry (same river_only_flat_level_m
convention as rule modelled_depth_estimation's own calibration), isolating the river's own
response regardless of the production model's sfincs.boundary_setup.mode.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyproj
import xarray as xr
from shapely.geometry import Polygon
from typing import cast
import geopandas as gpd

from src.domain import load_domain
from src.log import setup_logging
from src.plots import plot_max_inundation_map, plot_water_level_timeseries
from src.postprocessing import compute_max_inundation
from src.river_depth_calibration import check_convergence
from src.river_forcing import interpolate_discharge_at_rp
from src.sfincs_run import run_sfincs_subprocess

log = setup_logging(snakemake.log[0])

sfincs_root = Path(snakemake.params.sfincs_root)
sfincs_exe  = Path(snakemake.params.sfincs_exe)
timeout_s   = int(snakemake.params.timeout_s)
include_subgrid = bool(snakemake.params.include_subgrid)
higher_rp_factor = float(snakemake.params.higher_rp_factor)
simulation_days  = float(snakemake.params.simulation_days)
window_hours = float(snakemake.params.convergence_window_hours)
abs_tol_m    = float(snakemake.params.convergence_abs_tolerance_m)
rel_tol      = float(snakemake.params.convergence_rel_tolerance)
river_only_flat_level_m = float(snakemake.params.river_only_flat_level_m)

land_polygons_path = Path(snakemake.input.land_polygons)
landuse_path       = Path(snakemake.input.landuse)
river_network_path = Path(snakemake.input.clean_river_network)

# ── domain polygon (WGS84, for plot overlays) ────────────────────────────────
_domain_gdf = gpd.read_file(snakemake.input.domain_gpkg)
if _domain_gdf.crs is not None and _domain_gdf.crs.to_epsg() != 4326:
    _domain_gdf = _domain_gdf.to_crs("EPSG:4326")
_union = _domain_gdf.geometry.union_all()
domain_poly = cast(Polygon, _union if isinstance(_union, Polygon) else _union.convex_hull)

# ── parse production sfincs.inp (same technique as 14_run_spinup.py) ────────
main_inp = sfincs_root / "sfincs.inp"
cfg: dict[str, str] = {}
with open(main_inp) as fh:
    for line in fh:
        line = line.strip()
        if "=" in line and not line.startswith("!"):
            key, _, val = line.partition("=")
            cfg[key.strip().lower()] = val.strip()
log.info(f"Parsed production sfincs.inp: {len(cfg)} parameters")

epsg = int(cfg["epsg"])
utm_crs = f"EPSG:{epsg}"
tref = datetime.strptime(cfg.get("tref", "20000101 000000").replace("  ", " "), "%Y%m%d %H%M%S")

# ── production discharge source points (sfincs.src) ─────────────────────────
src_path = sfincs_root / "sfincs.src"
src_xy = np.loadtxt(src_path, usecols=(0, 1)) if src_path.exists() and src_path.stat().st_size > 0 else np.empty((0, 2))
if src_xy.ndim == 1 and src_xy.size:
    src_xy = src_xy[np.newaxis, :]
n_src = len(src_xy)
log.info(f"Production discharge source points: {n_src}")

# ── production boundary points (sfincs.bnd) -- count only, values rewritten ─
bnd_path = sfincs_root / "sfincs.bnd"
bnd_xy = np.loadtxt(bnd_path, usecols=(0, 1)) if bnd_path.exists() and bnd_path.stat().st_size > 0 else np.empty((0, 2))
if bnd_xy.ndim == 1 and bnd_xy.size:
    bnd_xy = bnd_xy[np.newaxis, :]
n_bnd = len(bnd_xy)
log.info(f"Production boundary points: {n_bnd}")

# ── match each src point to its river_forcing.nc crossing (nearest coord) ───
transformer = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
src_lon, src_lat = transformer.transform(src_xy[:, 0], src_xy[:, 1]) if n_src else ([], [])

with xr.open_dataset(snakemake.input.river_forcing, decode_times=False) as river_ds:
    active = river_ds["has_glofas"].values.astype(bool)
    cross_lon = river_ds["longitude"].values[active]
    cross_lat = river_ds["latitude"].values[active]
    bankfull_q = river_ds["bankfull_discharge"].values[active]
    rp_table = river_ds["discharge_rp_table"].values[active]
    table_rps = river_ds["return_period"].values

with open(snakemake.input.protection_levels) as f:
    protection = json.load(f)
protection_rp_yr = None
_rp = protection.get("riverine_rp_yr")
if _rp is not None and np.isfinite(float(_rp)):
    protection_rp_yr = float(_rp)
log.info(f"Protection-level RP: {protection_rp_yr}")

protection_q_per_src = np.zeros(n_src)
for i in range(n_src):
    d2 = (cross_lon - src_lon[i]) ** 2 + (cross_lat - src_lat[i]) ** 2
    j = int(np.argmin(d2))
    if protection_rp_yr is not None:
        protection_q_per_src[i] = float(
            interpolate_discharge_at_rp(rp_table[j][np.newaxis, :], table_rps, protection_rp_yr)[0]
        )
    else:
        protection_q_per_src[i] = float(bankfull_q[j])
log.info(
    f"Protection-level discharge per source point: "
    f"{np.array2string(protection_q_per_src, precision=1)} m3/s"
)

scenarios = {
    "protection_level": protection_q_per_src,
    "higher_rp": protection_q_per_src * higher_rp_factor,
}


def _write_scenario(name: str, discharge_per_src: np.ndarray) -> Path:
    scenario_dir = sfincs_root / f"validation_{name}"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    tstop = tref + timedelta(days=simulation_days)
    n_steps = max(int(simulation_days * 24), 2)
    times_s = np.linspace(0.0, simulation_days * 86400.0, n_steps)

    # ── steady discharge (sfincs.dis) ────────────────────────────────────────
    if n_src:
        dis_arr = np.column_stack([times_s] + [np.full(n_steps, q) for q in discharge_per_src])
        np.savetxt(scenario_dir / "sfincs.dis", dis_arr, fmt="%10.1f")

    # ── flat, dry boundary (sfincs.bzs) -- river_only style, isolates the
    # river's own response regardless of the production model's own
    # sfincs.boundary_setup.mode ─────────────────────────────────────────────
    if n_bnd:
        bzs_arr = np.column_stack(
            [times_s] + [np.full(n_steps, river_only_flat_level_m) for _ in range(n_bnd)]
        )
        np.savetxt(scenario_dir / "sfincs.bzs", bzs_arr, fmt="%10.3f")

    # ── scenario sfincs.inp: grid/physics copied verbatim, new timing, every
    # other "*file" entry referenced unchanged via ../ (same technique as
    # 14_run_spinup.py) except disfile/bzsfile, rewritten above ──────────────
    def fmt_dt(dt: datetime) -> str:
        return dt.strftime("%Y%m%d %H%M%S")

    trstout_sec = int(simulation_days * 86400)
    lines: list[str] = []
    for key in ("mmax", "nmax", "dx", "dy", "x0", "y0", "rotation", "epsg", "crsgeo"):
        if key in cfg:
            lines.append(f"{key:<20} = {cfg[key]}")
    lines += [
        f"{'tref':<20} = {fmt_dt(tref)}",
        f"{'tstart':<20} = {fmt_dt(tref)}",
        f"{'tstop':<20} = {fmt_dt(tstop)}",
        f"{'dthisout':<20} = 3600",
        f"{'dtmapout':<20} = 3600",
        f"{'dtmaxout':<20} = {trstout_sec}",
        f"{'zsini':<20} = -9999.0",
    ]
    for key in ("alpha", "huthresh", "advection", "viscosity", "nuvisc", "coriolis",
                "baro", "rhoa", "rhow", "pavbnd", "btfilter", "latitude"):
        if key in cfg:
            lines.append(f"{key:<20} = {cfg[key]}")
    lines += [
        f"{'storevel':<20} = 0",
        f"{'storevelmax':<20} = 0",
        f"{'storecumprcp':<20} = 0",
        f"{'storemeteo':<20} = 0",
        f"{'storetwet':<20} = 0",
        f"{'inputformat':<20} = {cfg.get('inputformat', 'bin')}",
        f"{'outputformat':<20} = net",
    ]
    for key, value in cfg.items():
        if not key.endswith("file") or key in ("rstfile", "disfile", "bzsfile"):
            continue
        fpath = sfincs_root / value
        if fpath.exists() and fpath.stat().st_size > 0:
            lines.append(f"{key:<20} = ../{value}")
    if n_src:
        lines.append(f"{'disfile':<20} = sfincs.dis")
    if n_bnd:
        lines.append(f"{'bzsfile':<20} = sfincs.bzs")

    with open(scenario_dir / "sfincs.inp", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return scenario_dir


def _run_scenario(name: str, discharge_per_src: np.ndarray, plot_inundation_path: Path, plot_water_level_path: Path):
    scenario_dir = _write_scenario(name, discharge_per_src)
    log.info(f"[{name}] discharge={np.array2string(discharge_per_src, precision=1)} m3/s, running {simulation_days:.0f} days")
    run_sfincs_subprocess(sfincs_exe, scenario_dir, timeout_s, log, label=f"SFINCS validation ({name})")

    his_path = scenario_dir / "sfincs_his.nc"
    if his_path.exists():
        ds = xr.open_dataset(his_path, decode_times=False)
        zs_var = next((v for v in ("point_zs", "zs") if v in ds), None)
        if zs_var is not None:
            zs = ds[zs_var].values
            times_his = np.asarray(ds["time"].values, dtype=float)
            if zs.shape[0] != len(times_his):
                zs = zs.T
            result = check_convergence(zs, times_his, window_hours, abs_tol_m, rel_tol)
            n_pending = int((~result["converged"]).sum())
            if n_pending:
                log.warning(
                    f"[{name}] {n_pending}/{len(result['converged'])} obs point(s) "
                    f"not converged after {simulation_days:.0f} days (diagnostic-only, not a hard failure)"
                )
            else:
                log.info(f"[{name}] all obs points converged")
            plot_water_level_timeseries(
                times_his / 86400.0, zs, plot_water_level_path,
                day_markers=[(simulation_days, f"Day {simulation_days:.0f} (end)")],
                basin_id=f"Basin {sfincs_root.parent.name}",
                run_label=f"Protection validation -- {name}",
            )
            log.info(f"[{name}] water-level plot written: {plot_water_level_path}")
        else:
            plot_water_level_path.touch()
    else:
        log.warning(f"[{name}] no sfincs_his.nc produced -- skipping water-level plot")
        plot_water_level_path.touch()

    da_hmax, _da_dep = compute_max_inundation(
        scenario_dir, sfincs_root, landuse_path, hmin=0.0, include_subgrid=include_subgrid,
    )
    if da_hmax is None:
        log.warning(f"[{name}] no max inundation data available -- creating empty plot sentinel")
        plot_inundation_path.touch()
    else:
        plot_max_inundation_map(
            da_hmax, domain_poly, str(land_polygons_path), str(river_network_path),
            str(plot_inundation_path), basin_id=sfincs_root.parent.name, run_label=f"protection validation ({name})",
        )
        log.info(f"[{name}] max inundation plot written: {plot_inundation_path}")


_run_scenario(
    "protection_level", scenarios["protection_level"],
    Path(snakemake.output.plot_max_inundation_protection),
    Path(snakemake.output.plot_water_level_protection),
)
_run_scenario(
    "higher_rp", scenarios["higher_rp"],
    Path(snakemake.output.plot_max_inundation_higher),
    Path(snakemake.output.plot_water_level_higher),
)

log.info("Done")
