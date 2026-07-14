"""
test_zsini_sensitivity.py — Hypothesis test: how sensitive is the spin-up
flood result to the assumed initial water level (zsini), independent of any
dynamic forcing?

zsini.tif (rule 03a) normally encodes 0.0 (mean sea level) for sea cells and
NODATA for land cells (SFINCS then falls back to local bed level on land, i.e.
dry at t=0). This script keeps that convention -- land cells stay NODATA --
and only sweeps the *sea*-cell value, from the minimum land elevation (a
"drier than the lowest point on land" extreme) up to mean sea level (the
value the original zsini.tif already uses for sea cells), for basin 2444235.

All other inputs (elevation, roughness, mask, river network/depth, grid
resolution) are held identical across levels — copied verbatim from the
basin's existing build at results/2444235/{inputs,sfincs}/. To isolate the
zsini effect from dynamic forcing:
  - River discharge is zeroed at all crossings (no river forcing).
  - The coastal water-level boundary is flat and held at the *same* level as
    that scenario's zsini (not the real COAST-RP surge/tide forcing) — this
    avoids an initial mismatch between the sea-cell zsini and the boundary
    that would otherwise create its own draining/filling transient unrelated
    to the question being tested (mirrors tests/test_river_smoothing_sfincs.py's
    RIVER_ONLY pattern, but the flat level tracks the swept zsini value).

For each level, runs the spin-up only (workflow/scripts/14_run_spinup.py
logic, replicated here) and counts wet cells via
src.postprocessing.compute_max_inundation — the same function rule 10 uses —
at the project's configured min_inundation_depth_m threshold. Land starts dry
(bed level) in every scenario, so any land inundation that shows up here, with
no discharge and no surge variability, must come from the assumed sea-cell
zsini/boundary level propagating inland during spin-up rather than real flood
signal. Produces a line plot of zsini level vs. wet-cell count.

Per-level SFINCS model directories (full builds, large binary subgrid
tables) are written outside both results/ and the git-tracked tests/ folder,
to D:/GCFM_UU/experiments/zsini_sensitivity/<basin_id>/. The CSV summary and
line plot are written to figs/zsini_sensitivity/.

Usage:
    conda run -n hmt_sfincs_dev python tests/test_zsini_sensitivity.py            # auto-derived sweep, ~0.5 m steps
    conda run -n hmt_sfincs_dev python tests/test_zsini_sensitivity.py -6 -3 0    # only these levels
"""

import logging
import subprocess
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import math
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))
from src.plots import reproject_max_for_plot
from src.postprocessing import compute_max_inundation
from hydromt_sfincs import SfincsModel

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
for _name in ("hydromt", "hydromt_sfincs"):
    logging.getLogger(_name).setLevel(logging.WARNING)

plt.ioff()

REPO_ROOT = Path(__file__).resolve().parents[1]
BASIN_ID = "2444235"
LEVEL_STEP_M = 0.2  # default spacing between swept zsini levels
BND_DIST_M = 5000.0  # spacing for boundary points generated from the mask

with open(REPO_ROOT / "config" / "config.yml") as fh:
    config = yaml.safe_load(fh)

RESULTS_DIR = Path(config["results_dir"])
EXPERIMENTS_DIR = Path("D:/GCFM_UU/experiments/zsini_sensitivity") / BASIN_ID
FIGS_DIR = REPO_ROOT / "figs" / "zsini_sensitivity"
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
dis_df = pd.DataFrame(data=0.0, index=river_times, columns=range(n_active))
log.info("River discharge forced to 0.0 m3/s at all crossings (no river forcing)")

rivers = gpd.read_file(river_network_path)
rivers["rivwth"] = rivers["max_width"].fillna(1.0).astype(float)
log.info(f"Loaded {len(rivers)} reaches from {river_network_path}")

# ── derive the zsini sweep range from the basin's own elevation/zsini files ────
with rasterio.open(zsini_raw_path) as _src:
    zsini_raw = _src.read(1).astype(np.float32)
    zsini_meta = _src.meta.copy()
    zsini_nodata = np.float32(_src.nodata if _src.nodata is not None else -9999.0)
with rasterio.open(elevation_merged_path) as _src:
    elev_raw = _src.read(1).astype(np.float32)
    elev_nodata = np.float32(_src.nodata if _src.nodata is not None else -9999.0)

sea_mask = zsini_raw != zsini_nodata
land_mask = (zsini_raw == zsini_nodata) & (elev_raw != elev_nodata)
mean_sea_level_m = (
    float(math.ceil(zsini_raw[sea_mask].max()) + 0.05) if sea_mask.any() else 0.0
)
min_land_elev_m = float(math.floor(elev_raw[land_mask].min()))
zsini_low_m, zsini_high_m = min_land_elev_m, mean_sea_level_m
log.info(
    f"Basin {BASIN_ID}: min land elevation = {min_land_elev_m:.2f} m, "
    f"mean sea level (zsini.tif) = {mean_sea_level_m:.2f} m -- "
    f"sweep range = [{zsini_low_m:.2f}, {zsini_high_m:.2f}] m"
)

cli_levels = [float(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else None
if cli_levels:
    ZSINI_LEVELS_M = cli_levels
else:
    n_steps = max(int(round((zsini_high_m - zsini_low_m) / LEVEL_STEP_M)), 4)
    ZSINI_LEVELS_M = [
        round(v, 2) for v in np.linspace(zsini_low_m, zsini_high_m, n_steps + 1)
    ]
log.info(f"Zsini levels to test (m): {ZSINI_LEVELS_M}")

# Levels for the 4-panel max-inundation comparison figure: evenly spaced across
# the full [zsini_low_m, zsini_high_m] range, snapped to the nearest level
# actually being swept (so no extra SFINCS runs are needed beyond the main sweep).
N_PANELS = 4
_panel_targets = np.linspace(zsini_low_m, zsini_high_m, N_PANELS)
panel_levels_m = sorted(
    {min(ZSINI_LEVELS_M, key=lambda lv: abs(lv - t)) for t in _panel_targets}
)
log.info(f"Zsini levels selected for the panel figure (m): {panel_levels_m}")


def write_zsini_sea_level(level_m: float, out_path: Path) -> Path:
    """Sea cells set to the swept initial water level; land cells left at NODATA
    (matches zsini.tif's original convention -- SFINCS falls back to local bed
    level on land, i.e. dry at t=0)."""
    arr = np.full(
        (zsini_meta["height"], zsini_meta["width"]), zsini_nodata, dtype=np.float32
    )
    arr[sea_mask] = np.float32(level_m)
    with rasterio.open(out_path, "w", **zsini_meta) as dst:
        dst.write(arr, 1)
    return out_path


def build_model(level_m: float, zsini_path: Path, sfincs_root: Path) -> SfincsModel:
    """Full SFINCS build for one zsini level (mirrors 13_build_sfincs.py)."""
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
    # Default reproj_method="average" lets GDAL average only the *valid* (sea)
    # sub-pixels in a destination cell, ignoring nodata (land) ones -- a coastal
    # grid cell that's mostly land by area still gets the full sea-cell zsini
    # value, "starting wet" even where its bed elevation is well above that
    # level (same artifact documented in test_river_smoothing_sfincs.py and
    # tests/check_zsini_resampling.py). "nearest" avoids this blending.
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

    # Flat boundary held at the same level as this scenario's sea-cell zsini -- no
    # initial mismatch between the two, so any wet land cells that appear are due
    # to that level propagating inland during spin-up, not a boundary transient.
    sf.water_level.create_boundary_points_from_mask(bnd_dist=BND_DIST_M)
    sf.water_level.create_timeseries(shape="constant", offset=level_m)

    if n_active > 0:
        buf_deg = float(resolution) / 111_000.0
        region_geom = sf.region.to_crs("EPSG:4326").geometry.union_all().buffer(buf_deg)
        crossings_filt = crossings_gdf[
            crossings_gdf.geometry.within(region_geom)
        ].copy()
        if not crossings_filt.empty:
            sf.discharge_points.create(timeseries=dis_df, locations=crossings_filt)

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
        f"{'storevel':<20} = 0",
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
csv_path = FIGS_DIR / f"zsini_sensitivity_{BASIN_ID}.csv"
results: list[dict] = []
hmax_by_level: dict[float, xr.DataArray] = {}

for level_m in ZSINI_LEVELS_M:
    tag = f"zsini_{level_m:.2f}m"
    sfincs_root = EXPERIMENTS_DIR / tag
    log.info(f"=== zsini level={level_m:.2f} m -> {sfincs_root} ===")

    try:
        zsini_path = write_zsini_sea_level(
            level_m, EXPERIMENTS_DIR / f"{tag}_zsini.tif"
        )
        build_model(level_m, zsini_path, sfincs_root)
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
                f"zsini level={level_m} m: no zsmax/bed level available -- skipping"
            )
            continue

        if level_m in panel_levels_m:
            hmax_by_level[level_m] = da_hmax

        n_land = int(da_dep.notnull().sum().item())
        n_flooded = int(da_hmax.notnull().sum().item())
        frac = n_flooded / n_land if n_land > 0 else 0.0
        log.info(
            f"zsini level={level_m:.2f} m: {n_flooded:,}/{n_land:,} wet cells ({frac:.2%})"
        )
        results.append(
            {
                "zsini_level_m": level_m,
                "n_flooded": n_flooded,
                "n_land": n_land,
                "frac_flooded": frac,
            }
        )
    except Exception:
        log.exception(f"zsini level={level_m} m failed -- skipping")
        continue

    pd.DataFrame(results).to_csv(csv_path, index=False)

if results:
    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["zsini_level_m"], df["n_flooded"], marker="o")
    ax.axvline(
        mean_sea_level_m,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="mean sea level (original zsini)",
    )
    ax.set_xlabel("Sea-cell initial water level, zsini (m)")
    ax.set_ylabel(f"Wet cells (h > {min_inundation_depth_m} m)")
    ax.set_title(
        f"Basin {BASIN_ID} -- spin-up wet-cell count vs. zsini level (no discharge, flat boundary)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = FIGS_DIR / f"zsini_sensitivity_{BASIN_ID}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    log.info(f"Plot written: {plot_path}")
    log.info(f"CSV written: {csv_path}")
else:
    log.warning("No successful runs -- nothing to plot")

if hmax_by_level:
    valid_vals = np.concatenate(
        [da.values[~np.isnan(da.values)].ravel() for da in hmax_by_level.values()]
    )
    vmax_panel = float(np.percentile(valid_vals, 99)) if valid_vals.size else 1.0
    vmax_panel = max(vmax_panel, 0.01)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, level_m in zip(axes.flat, panel_levels_m):
        da_hmax_level = hmax_by_level.get(level_m)
        if da_hmax_level is None:
            ax.set_title(f"zsini = {level_m:+.2f} m (run failed)")
            ax.axis("off")
            continue

        # reproject_max_for_plot reprojects directly at a coarse resolution
        # with max-value resampling -- reprojecting at native (subgrid)
        # resolution first can produce tens of millions of pixels, which
        # blows up matplotlib's imshow/RGBA rendering.
        da_wgs = reproject_max_for_plot(da_hmax_level.squeeze())
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
        ax.set_title(f"zsini = {level_m:+.2f} m  ({n_flooded_panel:,} flooded cells)")
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")

    for ax in axes.flat[len(panel_levels_m) :]:
        ax.axis("off")

    fig.colorbar(
        im, ax=axes, shrink=0.8, extend="max", label="Max inundation depth (m)"
    )
    fig.suptitle(
        f"Basin {BASIN_ID} -- max inundation depth on land across the zsini range (no discharge, flat boundary)"
    )
    panels_path = FIGS_DIR / f"zsini_sensitivity_{BASIN_ID}_panels.png"
    fig.savefig(panels_path, dpi=150)
    plt.close(fig)
    log.info(f"Panel figure written: {panels_path}")
else:
    log.warning("No successful runs among the panel levels -- skipping panel figure")
