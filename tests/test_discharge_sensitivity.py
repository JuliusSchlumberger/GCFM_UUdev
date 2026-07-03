"""
test_discharge_sensitivity.py — Hypothesis test: how does spin-up flood
extent respond to river discharge magnitude alone, with the coastal boundary
held dry so it cannot contribute or interfere?

For basin 2444235, sweeps a multiplicative scale factor over the production
river-discharge timeseries (river_forcing.nc, the same per-crossing
hydrograph used in production, just scaled up/down) while the coastal
water-level boundary is flat and fixed at COASTAL_BOUNDARY_LEVEL_M = -10 m —
well below local bed elevation almost everywhere, so the sea boundary acts as
a free drain that never itself floods anything (mirrors the flat-boundary
pattern in tests/test_river_smoothing_sfincs.py and
tests/test_zsini_sensitivity.py, but fixed low here specifically to isolate
the river-discharge effect rather than the initial-condition effect).
zsini matches that same convention: sea cells = COASTAL_BOUNDARY_LEVEL_M (no
mismatch transient against the boundary), land cells stay NODATA (dry at
t=0, SFINCS falls back to local bed level).

All other inputs (elevation, roughness, mask, river network/depth, grid
resolution) are held identical across scale factors — copied verbatim from
the basin's existing build at results/2444235/{inputs,sfincs}/.

For each scale factor, runs the spin-up only (workflow/scripts/
14_run_spinup.py logic, replicated here) and counts wet cells via
src.postprocessing.compute_max_inundation — the same function rule 15 uses —
at the project's configured min_inundation_depth_m threshold. Produces:
  - a line plot of discharge scale vs. wet-cell count (CSV + PNG), and
  - a 4-panel max-inundation comparison figure across the sweep range
    (mirrors tests/test_zsini_sensitivity.py's panel figure).

Per-scale SFINCS model directories (full builds, large binary subgrid
tables) are written outside both results/ and the git-tracked tests/ folder,
to D:/GCFM_UU/experiments/discharge_sensitivity/<basin_id>/. The CSV summary
and plots are written to figs/discharge_sensitivity/.

Usage:
    conda run -n hmt_sfincs_dev python tests/test_discharge_sensitivity.py            # default sweep: 0, 0.5, 1, 2, 5, 10x
    conda run -n hmt_sfincs_dev python tests/test_discharge_sensitivity.py 0 1 5 20    # only these scale factors
"""

import logging
import subprocess
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))
from src.postprocessing import compute_max_inundation
from hydromt_sfincs import SfincsModel

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
for _name in ("hydromt", "hydromt_sfincs"):
    logging.getLogger(_name).setLevel(logging.WARNING)

plt.ioff()

REPO_ROOT = Path(__file__).resolve().parents[1]
BASIN_ID = "2444235"
DISCHARGE_SCALES = [
    0.2,
    0.5,
    0.7,
    1.0,
    1.2,
    1.5,
    2.0,
]  # default sweep: x production discharge
COASTAL_BOUNDARY_LEVEL_M = (
    -10.0
)  # fixed, dry coastal boundary -- isolates the river-discharge effect
BND_DIST_M = 5000.0  # spacing for boundary points generated from the mask
N_PANELS = 4

with open(REPO_ROOT / "config" / "config.yml") as fh:
    config = yaml.safe_load(fh)

RESULTS_DIR = Path(config["results_dir"])
EXPERIMENTS_DIR = Path("D:/GCFM_UU/experiments/discharge_sensitivity") / BASIN_ID
FIGS_DIR = REPO_ROOT / "figs" / "discharge_sensitivity"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

basin_inputs_dir = RESULTS_DIR / BASIN_ID / "inputs"

domain_path = basin_inputs_dir / "domain" / f"{BASIN_ID}_domain.gpkg"
elevation_merged_path = basin_inputs_dir / "domain" / f"{BASIN_ID}_elevation_merged.tif"
roughness_path = basin_inputs_dir / "domain" / f"{BASIN_ID}_roughness.tif"
land_polygons_path = basin_inputs_dir / "domain" / f"{BASIN_ID}_land_polygons.gpkg"
landuse_path = basin_inputs_dir / "domain" / f"{BASIN_ID}_landuse.tif"
river_network_path = (
    basin_inputs_dir / "domain" / f"{BASIN_ID}_river_network_processed.gpkg"
)
surge_forcing_path = basin_inputs_dir / "forcing" / "surge_forcing.nc"
river_forcing_path = basin_inputs_dir / "forcing" / "river_forcing.nc"
zsini_raw_path = basin_inputs_dir / "domain" / f"{BASIN_ID}_zsini.tif"

sfincs_cfg = config["sfincs"]
resolution = sfincs_cfg["grid"]["resolution"]
include_subgrid = sfincs_cfg["subgrid"]["enabled"]
nr_subgrid_pixels = sfincs_cfg["subgrid"]["nr_subgrid_pixels"]
nr_levels = sfincs_cfg["subgrid"]["nr_levels"]
nrmax = sfincs_cfg["subgrid"]["nrmax"]
tref_str = sfincs_cfg["simulation"]["tref"]
dtmapout = sfincs_cfg["simulation"]["dtmapout"]
dtmaxout = sfincs_cfg["simulation"]["dtmaxout"]
dthisout = sfincs_cfg["simulation"]["dthisout"]
storevelmax = sfincs_cfg["simulation"]["storevelmax"]
storetwet = sfincs_cfg["simulation"]["storetwet"]
sfincs_exe = Path(sfincs_cfg["simulation"]["sfincs_exe"]).resolve()
spinup_days = sfincs_cfg["spinup"]["spinup_days"]
spinup_dtmapout = sfincs_cfg["spinup"]["dtmapout_s"]
spinup_dthisout = sfincs_cfg["spinup"]["dthisout_s"]
spinup_timeout_s = sfincs_cfg["spinup"]["timeout_s"]
velocity_animation_enabled = sfincs_cfg["sanity_checks"]["velocity_animation"][
    "enabled"
]
min_inundation_depth_m = sfincs_cfg["sanity_checks"]["min_inundation_depth_m"]

# ── scenario-independent inputs, loaded once ───────────────────────────────────
delta_domain = gpd.read_file(domain_path)
land_polygons_empty = gpd.read_file(land_polygons_path).empty
boundary_kwargs = (
    {} if land_polygons_empty else {"exclude_polygon": "local_land_polygons"}
)

with xr.open_dataset(surge_forcing_path, decode_times=False) as _ds:
    surge_end_hr = float(_ds.time.max())
with xr.open_dataset(river_forcing_path, decode_times=False) as _ds:
    river_end_hr = float(_ds.time.max())

tref = datetime.strptime(tref_str, "%Y-%m-%d %H:%M:%S")
tstop = tref + timedelta(hours=max(surge_end_hr, river_end_hr))

river_ds = xr.open_dataset(river_forcing_path, decode_times=False)
active = river_ds.has_glofas.values.astype(bool)
n_active = int(active.sum())
river_times = pd.DatetimeIndex(
    [tref + timedelta(hours=float(h)) for h in river_ds.time.values]
)
crossings_gdf = gpd.GeoDataFrame(
    {"index": range(n_active)},
    geometry=gpd.points_from_xy(
        river_ds.longitude.values[active], river_ds.latitude.values[active]
    ),
    crs="EPSG:4326",
)
dis_df_production = pd.DataFrame(
    data=river_ds.discharge.values[active, :].T,
    index=river_times,
    columns=range(n_active),
)
log.info(
    f"Loaded production discharge: {n_active} crossing(s), "
    f"max Q = {dis_df_production.to_numpy().max():.1f} m3/s"
)

rivers = gpd.read_file(river_network_path)
rivers["rivwth"] = rivers["max_width"].fillna(1.0).astype(float)
log.info(f"Loaded {len(rivers)} reaches from {river_network_path}")

# ── fixed, dry coastal zsini (sea = COASTAL_BOUNDARY_LEVEL_M, land = NODATA) ───
# Built once -- it never varies across the discharge sweep, only matches the
# flat boundary level so there's no initial mismatch/transient at the coast.
with rasterio.open(zsini_raw_path) as _src:
    zsini_raw = _src.read(1).astype(np.float32)
    zsini_meta = _src.meta.copy()
    zsini_nodata = np.float32(_src.nodata if _src.nodata is not None else -9999.0)
sea_mask = zsini_raw != zsini_nodata

zsini_path = EXPERIMENTS_DIR / "zsini_coastal_dry.tif"
_arr = np.full(
    (zsini_meta["height"], zsini_meta["width"]), zsini_nodata, dtype=np.float32
)
_arr[sea_mask] = np.float32(COASTAL_BOUNDARY_LEVEL_M)
with rasterio.open(zsini_path, "w", **zsini_meta) as _dst:
    _dst.write(_arr, 1)
log.info(
    f"Fixed coastal-dry zsini written: {zsini_path} (sea cells = {COASTAL_BOUNDARY_LEVEL_M:+.2f} m)"
)

cli_scales = [float(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else None
DISCHARGE_SCALES = cli_scales or DISCHARGE_SCALES
log.info(f"Discharge scale factors to test: {DISCHARGE_SCALES}")

# Panels for the 4-comparison max-inundation figure: evenly spaced across the
# full sweep range, snapped to the nearest scale actually being tested (so no
# extra SFINCS runs are needed beyond the main sweep).
_panel_targets = np.linspace(min(DISCHARGE_SCALES), max(DISCHARGE_SCALES), N_PANELS)
panel_scales = sorted(
    {min(DISCHARGE_SCALES, key=lambda s: abs(s - t)) for t in _panel_targets}
)
log.info(f"Discharge scales selected for the panel figure: {panel_scales}")


def build_model(scale: float, sfincs_root: Path) -> SfincsModel:
    """Full SFINCS build for one discharge scale factor (mirrors 13_build_sfincs.py)."""
    sfincs_root.mkdir(parents=True, exist_ok=True)

    local_catalog_path = sfincs_root / "data_catalog_local.yml"
    local_catalog = {
        "meta": {"root": str(basin_inputs_dir)},
        "local_elevation_merged": {
            "data_type": "RasterDataset",
            "uri": str(elevation_merged_path),
            "driver": "rasterio",
        },
        "local_roughness": {
            "data_type": "RasterDataset",
            "uri": str(roughness_path),
            "driver": "rasterio",
        },
        "local_land_polygons": {
            "data_type": "GeoDataFrame",
            "uri": str(land_polygons_path),
            "driver": "pyogrio",
        },
        "local_zsini": {
            "data_type": "RasterDataset",
            "uri": str(zsini_path),
            "driver": "rasterio",
        },
    }
    with open(local_catalog_path, "w") as fh:
        yaml.dump(local_catalog, fh, sort_keys=False)

    sf = SfincsModel(
        data_libs=[str(local_catalog_path)],
        root=str(sfincs_root),
        mode="w+",
        write_gis=False,
    )

    sf.grid.create_from_region(
        region={"geom": delta_domain}, res=resolution, crs="utm", rotated=False
    )
    sf.elevation.create(elevation_list=[{"elevation": "local_elevation_merged"}])
    sf.mask.create_active(include_polygon=delta_domain)
    sf.mask.create_boundary(btype="waterlevel", reset_bounds=True, **boundary_kwargs)
    sf.initial_conditions.create(ini="local_zsini", reproj_method="nearest")
    sf.roughness.create(roughness_list=[{"manning": "local_roughness"}])

    sf.config.update(
        {
            "tref": tref,
            "tstart": tref,
            "tstop": tstop,
            "dtmapout": dtmapout,
            "dtmaxout": dtmaxout,
            "dthisout": dthisout,
            "storevelmax": storevelmax,
            "storetwet": storetwet,
            "baro": 0,
        }
    )

    sf.subgrid.create(
        elevation_list=[{"elevation": "local_elevation_merged"}],
        roughness_list=[{"manning": "local_roughness"}],
        river_list=[{"centerlines": rivers}],
        nr_subgrid_pixels=nr_subgrid_pixels,
        nr_levels=nr_levels,
        write_dep_tif=True,
        write_man_tif=True,
        nrmax=nrmax,
    )

    # Flat, fixed-dry boundary -- no surge/tide variability and no flooding
    # contribution of its own, so any wet cells come from river discharge alone.
    sf.water_level.create_boundary_points_from_mask(bnd_dist=BND_DIST_M)
    sf.water_level.create_timeseries(shape="constant", offset=COASTAL_BOUNDARY_LEVEL_M)

    if n_active > 0:
        buf_deg = float(resolution) / 111_000.0
        region_geom = sf.region.to_crs("EPSG:4326").geometry.union_all().buffer(buf_deg)
        crossings_filt = crossings_gdf[
            crossings_gdf.geometry.within(region_geom)
        ].copy()
        if not crossings_filt.empty:
            sf.discharge_points.create(
                timeseries=dis_df_production * scale, locations=crossings_filt
            )

    sf.write()
    return sf


def run_spinup(sfincs_root: Path) -> Path:
    """Run the spin-up only (mirrors 14_run_spinup.py)."""
    spinup_dir = sfincs_root / "spinup"
    spinup_dir.mkdir(parents=True, exist_ok=True)
    spinup_inp = spinup_dir / "sfincs.inp"

    cfg: dict[str, str] = {}
    with open(sfincs_root / "sfincs.inp") as fh:
        for line in fh:
            line = line.strip()
            if "=" in line and not line.startswith("!"):
                key, _, val = line.partition("=")
                cfg[key.strip().lower()] = val.strip()

    tref_main = datetime.strptime(
        cfg.get("tref", "20000101 000000").replace("  ", " "), "%Y%m%d %H%M%S"
    )
    tstop_spinup = tref_main + timedelta(days=spinup_days)
    trstout_sec = int(spinup_days * 86400)

    def fmt_dt(dt: datetime) -> str:
        return dt.strftime("%Y%m%d %H%M%S")

    lines: list[str] = []
    for key in ("mmax", "nmax", "dx", "dy", "x0", "y0", "rotation", "epsg", "crsgeo"):
        if key in cfg:
            lines.append(f"{key:<20} = {cfg[key]}")

    lines += [
        f"{'tref':<20} = {fmt_dt(tref_main)}",
        f"{'tstart':<20} = {fmt_dt(tref_main)}",
        f"{'tstop':<20} = {fmt_dt(tstop_spinup)}",
        f"{'trstout':<20} = {trstout_sec}",
        f"{'dthisout':<20} = {spinup_dthisout}",
        f"{'dtmapout':<20} = {spinup_dtmapout}",
        f"{'dtmaxout':<20} = {trstout_sec}",
        f"{'dtrstout':<20} = 0",
    ]

    for key in (
        "alpha",
        "huthresh",
        "advection",
        "viscosity",
        "nuvisc",
        "coriolis",
        "baro",
        "rhoa",
        "rhow",
        "pavbnd",
        "btfilter",
        "latitude",
    ):
        if key in cfg:
            lines.append(f"{key:<20} = {cfg[key]}")

    lines += [
        f"{'storevel':<20} = {'1' if velocity_animation_enabled else '0'}",
        f"{'storevelmax':<20} = 0",
        f"{'storecumprcp':<20} = 0",
        f"{'storemeteo':<20} = 0",
        f"{'storetwet':<20} = 0",
        f"{'inputformat':<20} = {cfg.get('inputformat', 'bin')}",
        f"{'outputformat':<20} = net",
    ]

    for key, value in cfg.items():
        if not key.endswith("file") or key == "rstfile":
            continue
        fpath = sfincs_root / value
        if fpath.exists() and fpath.stat().st_size > 0:
            lines.append(f"{key:<20} = ../{value}")

    with open(spinup_inp, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    proc = subprocess.Popen(
        [str(sfincs_exe)],
        cwd=str(spinup_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def _forward(pipe, prefix):
        for line in pipe:
            line = line.rstrip()
            if line:
                print(f"{prefix} {line}", file=sys.stderr, flush=True)

    t_out = threading.Thread(target=_forward, args=(proc.stdout, "[sfincs out]"))
    t_err = threading.Thread(target=_forward, args=(proc.stderr, "[sfincs err]"))
    t_out.start()
    t_err.start()
    t_out.join()
    t_err.join()

    try:
        proc.wait(timeout=spinup_timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError(f"SFINCS spin-up exceeded {spinup_timeout_s}s timeout")

    if proc.returncode != 0:
        raise RuntimeError(f"SFINCS spin-up failed with exit code {proc.returncode}")

    return spinup_dir


# ── main sweep ─────────────────────────────────────────────────────────────────
csv_path = FIGS_DIR / f"discharge_sensitivity_{BASIN_ID}.csv"
results: list[dict] = []
hmax_by_scale: dict[float, xr.DataArray] = {}

for scale in DISCHARGE_SCALES:
    tag = f"discharge_{scale:.2f}x"
    sfincs_root = EXPERIMENTS_DIR / tag
    log.info(f"=== discharge scale={scale:.2f}x -> {sfincs_root} ===")

    try:
        build_model(scale, sfincs_root)
        spinup_dir = run_spinup(sfincs_root)
        da_hmax, da_dep = compute_max_inundation(
            spinup_dir,
            sfincs_root,
            landuse_path,
            hmin=min_inundation_depth_m,
            include_subgrid=include_subgrid,
        )
        if da_hmax is None or da_dep is None:
            log.warning(
                f"discharge scale={scale} x: no zsmax/bed level available -- skipping"
            )
            continue

        if scale in panel_scales:
            hmax_by_scale[scale] = da_hmax

        n_land = int(da_dep.notnull().sum().item())
        n_flooded = int(da_hmax.notnull().sum().item())
        frac = n_flooded / n_land if n_land > 0 else 0.0
        log.info(
            f"discharge scale={scale:.2f}x: {n_flooded:,}/{n_land:,} wet cells ({frac:.2%})"
        )
        results.append(
            {
                "discharge_scale": scale,
                "n_flooded": n_flooded,
                "n_land": n_land,
                "frac_flooded": frac,
            }
        )
    except Exception:
        log.exception(f"discharge scale={scale}x failed -- skipping")
        continue

    pd.DataFrame(results).to_csv(csv_path, index=False)

if results:
    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["discharge_scale"], df["n_flooded"], marker="o")
    ax.set_xlabel("River discharge scale factor (× production hydrograph)")
    ax.set_ylabel(f"Wet cells (h > {min_inundation_depth_m} m)")
    ax.set_title(
        f"Basin {BASIN_ID} -- spin-up wet-cell count vs. discharge "
        f"(coastal boundary fixed at {COASTAL_BOUNDARY_LEVEL_M:+.0f} m)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = FIGS_DIR / f"discharge_sensitivity_{BASIN_ID}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    log.info(f"Plot written: {plot_path}")
    log.info(f"CSV written: {csv_path}")
else:
    log.warning("No successful runs -- nothing to plot")

if hmax_by_scale:
    valid_vals = np.concatenate(
        [da.values[~np.isnan(da.values)].ravel() for da in hmax_by_scale.values()]
    )
    vmax_panel = float(np.percentile(valid_vals, 99)) if valid_vals.size else 1.0
    vmax_panel = max(vmax_panel, 0.01)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, scale in zip(axes.flat, panel_scales):
        da_hmax_scale = hmax_by_scale.get(scale)
        if da_hmax_scale is None:
            ax.set_title(f"discharge = {scale:.2f}x (run failed)")
            ax.axis("off")
            continue

        da_wgs = da_hmax_scale.squeeze().rio.reproject("EPSG:4326")
        arr = da_wgs.values.astype(np.float32)
        left, bottom, right, top = da_wgs.rio.bounds()
        im = ax.imshow(
            arr,
            cmap="Blues",
            vmin=0,
            vmax=vmax_panel,
            extent=(left, right, bottom, top),
            origin="upper",
            aspect="auto",
        )
        n_flooded_panel = int(np.isfinite(arr).sum())
        ax.set_title(f"discharge = {scale:.2f}x  ({n_flooded_panel:,} flooded cells)")
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")

    for ax in axes.flat[len(panel_scales) :]:
        ax.axis("off")

    fig.colorbar(
        im, ax=axes, shrink=0.8, extend="max", label="Max inundation depth (m)"
    )
    fig.suptitle(
        f"Basin {BASIN_ID} -- max inundation depth on land across the discharge sweep "
        f"(coastal boundary fixed at {COASTAL_BOUNDARY_LEVEL_M:+.0f} m)"
    )
    panels_path = FIGS_DIR / f"discharge_sensitivity_{BASIN_ID}_panels.png"
    fig.savefig(panels_path, dpi=150)
    plt.close(fig)
    log.info(f"Panel figure written: {panels_path}")
else:
    log.warning("No successful runs among the panel scales -- skipping panel figure")
