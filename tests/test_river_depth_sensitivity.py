"""
test_river_depth_sensitivity.py — Hypothesis test: does adding extra depth to
the river network's burned-in channel bathymetry change spin-up flood extent,
and does that effect depend on regular vs. quadtree grid refinement?

For basin 2444235, builds a fresh SFINCS model (mode="w+", required for a full
rebuild) for every combination of {regular, quadtree} grid type and several
"river-depth buffer" values, adding the buffer to the river network's
`rivdph` column before it is burned into the subgrid bathymetry via
`river_list=` in `subgrid.create()` (workflow/scripts/13_build_sfincs.py
section 7). All other inputs (elevation, roughness, mask, forcing, zsini,
grid resolution, quadtree refinement zones) are held identical across buffer
values within each grid type — they are copied verbatim from the basin's
existing build at results/2444235/{inputs,sfincs}/.

For each (grid_type, buffer) combination, runs the spin-up only (workflow/
scripts/14_run_spinup.py logic, replicated here) and counts wet cells via
`src.postprocessing.compute_max_inundation` — the same function rule 15 uses,
already grid-type agnostic via hydromt_sfincs's native ugrid handling — at the
project's configured `min_inundation_depth_m` threshold. Produces a line plot
of buffer depth vs. wet-cell count with one line per grid type.

Per-combination SFINCS model directories (full builds, large binary subgrid
tables) are written outside both results/ and the git-tracked tests/ folder,
to D:/GCFM_UU/experiments/river_depth_sensitivity/<basin_id>/. The CSV
summary and line plot are written to figs/river_depth_sensitivity/.

Quadtree builds are far slower than regular (~20-25 min vs. ~30s per buffer:
the subgrid table is built separately per refinement level, with the finest
level resolving to a 1.5 m subgrid pixel size). To keep total runtime
practical, the regular grid type runs the full buffer sweep by default while
quadtree only runs QUADTREE_BUFFERS_M (a handful of representative values).
Passing explicit buffers on the command line overrides both.

Usage:
    conda run -n hmt_sfincs_dev python tests/test_river_depth_sensitivity.py            # full 0.5-10m sweep (regular); reduced set (quadtree)
    conda run -n hmt_sfincs_dev python tests/test_river_depth_sensitivity.py 0.5 1.0     # only these buffers, both grid types
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
import pandas as pd
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))
from src.postprocessing import compute_max_inundation
from src.geometry import pick_utm_crs
from hydromt_sfincs import SfincsModel
from hydromt.gis import parse_crs

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
for _name in ("hydromt", "hydromt_sfincs"):
    logging.getLogger(_name).setLevel(logging.WARNING)

plt.ioff()

REPO_ROOT = Path(__file__).resolve().parents[1]
BASIN_ID = "2444235"
BUFFERS_M = [round(0.5 * i, 1) for i in range(1, 21)]  # 0.5, 1.0, ..., 10.0
QUADTREE_BUFFERS_M = [
    0.5,
    2.0,
    5.0,
    10.0,
]  # reduced set -- quadtree builds are ~20-25 min each

with open(REPO_ROOT / "config" / "config.yml") as fh:
    config = yaml.safe_load(fh)

RESULTS_DIR = Path(config["results_dir"])
EXPERIMENTS_DIR = Path("D:/GCFM_UU/experiments/river_depth_sensitivity") / BASIN_ID
FIGS_DIR = REPO_ROOT / "figs" / "river_depth_sensitivity"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

basin_inputs_dir = RESULTS_DIR / BASIN_ID / "inputs"
basin_sfincs_dir = RESULTS_DIR / BASIN_ID / "sfincs"

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
# Buffer-independent (depends only on surge baseline_m + zsini.tif): reuse the
# basin's existing build output rather than recomputing it per buffer.
zsini_baseline_path = basin_sfincs_dir / "zsini_baseline.tif"
if not zsini_baseline_path.exists():
    zsini_baseline_path = basin_inputs_dir / "domain" / f"{BASIN_ID}_zsini.tif"

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

quadtree_cfg = sfincs_cfg["grid"]["quadtree"]
river_refinement_level = quadtree_cfg["river_refinement_level"]
river_buffer_factor = quadtree_cfg["river_buffer_factor"]

GRID_TYPES = ["regular", "quadtree"]

# ── buffer-independent inputs, loaded once ────────────────────────────────────
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

surge_ds = xr.open_dataset(surge_forcing_path, decode_times=False)
surge_times = pd.DatetimeIndex(
    [tref + timedelta(hours=float(h)) for h in surge_ds.time.values]
)
n_stations = surge_ds.sizes["station"]
stations_gdf = gpd.GeoDataFrame(
    {"index": range(n_stations)},
    geometry=gpd.points_from_xy(surge_ds.longitude.values, surge_ds.latitude.values),
    crs="EPSG:4326",
)
wl_df = pd.DataFrame(
    data=surge_ds.water_level.values.T, index=surge_times, columns=range(n_stations)
)

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
dis_df = pd.DataFrame(
    data=river_ds.discharge.values[active, :].T,
    index=river_times,
    columns=range(n_active),
)

# Quadtree refinement zone: river network only (no coastal refinement — that
# zone covers the entire coastline perimeter and was making builds far too
# slow). Depends only on the (buffer-independent) river network widths, not
# on the rivdph buffer — computed once and reused across all buffer values.
bounds_4326 = delta_domain.to_crs("EPSG:4326").total_bounds
target_crs_quadtree = parse_crs("utm", bbox=list(bounds_4326))

_rivers_for_refinement = gpd.read_file(river_network_path)
_metric_crs = (
    pick_utm_crs(_rivers_for_refinement)
    if _rivers_for_refinement.crs.is_geographic
    else _rivers_for_refinement.crs
)
_rivers_m = _rivers_for_refinement.to_crs(_metric_crs)
_buffer_dist = _rivers_m["max_width"].fillna(0.0).clip(lower=0.0) * river_buffer_factor
refinement_gdf = gpd.GeoDataFrame(
    geometry=[_rivers_m.geometry.buffer(_buffer_dist).union_all()],
    crs=_metric_crs,
)
refinement_gdf["refinement_level"] = river_refinement_level
refinement_gdf = refinement_gdf.to_crs(target_crs_quadtree)
# quadtree_builder.refine_in_polygon() expects a simple Polygon per row (reads
# geometry.exterior directly) — explode any MultiPolygon from union_all().
refinement_gdf = refinement_gdf.explode(index_parts=False).reset_index(drop=True)
log.info(
    f"Quadtree river refinement zone: {len(refinement_gdf)} part(s), target CRS={target_crs_quadtree}"
)


def build_model(
    buffer_m: float, sfincs_root: Path, quadtree_enabled: bool
) -> SfincsModel:
    """Full SFINCS build for one (grid_type, river-depth buffer) combination (mirrors 13_build_sfincs.py)."""
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
            "uri": str(zsini_baseline_path),
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

    if quadtree_enabled:
        sf.quadtree_grid.create_from_region(
            region={"geom": delta_domain},
            res=resolution,
            crs=target_crs_quadtree,
            rotated=False,
            refinement_polygons=refinement_gdf,
        )
    else:
        sf.grid.create_from_region(
            region={"geom": delta_domain}, res=resolution, crs="utm", rotated=False
        )

    elevation_component = sf.quadtree_elevation if quadtree_enabled else sf.elevation
    elevation_component.create(elevation_list=[{"elevation": "local_elevation_merged"}])

    mask_component = sf.quadtree_mask if quadtree_enabled else sf.mask
    mask_component.create_active(include_polygon=delta_domain)
    mask_component.create_boundary(
        btype="waterlevel", reset_bounds=True, **boundary_kwargs
    )

    initial_conditions_component = (
        sf.quadtree_initial_conditions if quadtree_enabled else sf.initial_conditions
    )
    if quadtree_enabled:
        # quadtree_initial_conditions.create()'s default reproj_method="average" is not a
        # valid xugrid.OverlapRegridder method in the installed xugrid version (valid: "mean",
        # "harmonic_mean", ...) — override explicitly to work around this hydromt_sfincs bug.
        initial_conditions_component.create(ini="local_zsini", reproj_method="mean")
    else:
        initial_conditions_component.create(ini="local_zsini")

    roughness_component = sf.quadtree_roughness if quadtree_enabled else sf.roughness
    roughness_component.create(roughness_list=[{"manning": "local_roughness"}])

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

    # ── the experiment's only varying input ──────────────────────────────────
    rivers = gpd.read_file(river_network_path)
    rivers["rivwth"] = rivers["max_width"].fillna(1.0).astype(float)
    rivers["rivdph"] = rivers["rivdph"].clip(lower=0.0).astype(float) + buffer_m

    subgrid_component = sf.quadtree_subgrid if quadtree_enabled else sf.subgrid
    subgrid_component.create(
        elevation_list=[{"elevation": "local_elevation_merged"}],
        roughness_list=[{"manning": "local_roughness"}],
        river_list=[{"centerlines": rivers}],
        nr_subgrid_pixels=nr_subgrid_pixels,
        nr_levels=nr_levels,
        write_dep_tif=True,
        write_man_tif=True,
        nrmax=nrmax,
    )

    sf.water_level.create(timeseries=wl_df, locations=stations_gdf, buffer=25e3)

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
cli_buffers = [float(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else None
buffers_by_grid_type = {
    "regular": cli_buffers or BUFFERS_M,
    "quadtree": cli_buffers or QUADTREE_BUFFERS_M,
}
csv_path = FIGS_DIR / f"river_depth_sensitivity_{BASIN_ID}.csv"
results: list[dict] = []

for grid_type in GRID_TYPES:
    quadtree_enabled = grid_type == "quadtree"
    for buffer_m in buffers_by_grid_type[grid_type]:
        tag = f"{grid_type}_buf_{buffer_m:.1f}m"
        sfincs_root = EXPERIMENTS_DIR / tag
        log.info(
            f"=== grid_type={grid_type} buffer={buffer_m:.1f} m -> {sfincs_root} ==="
        )

        try:
            build_model(buffer_m, sfincs_root, quadtree_enabled)
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
                    f"grid_type={grid_type} buffer={buffer_m}: no zsmax/bed level available -- skipping"
                )
                continue

            n_land = int(da_dep.notnull().sum().item())
            n_flooded = int(da_hmax.notnull().sum().item())
            frac = n_flooded / n_land if n_land > 0 else 0.0
            log.info(
                f"grid_type={grid_type} buffer={buffer_m:.1f} m: {n_flooded:,}/{n_land:,} wet cells ({frac:.2%})"
            )
            results.append(
                {
                    "grid_type": grid_type,
                    "buffer_m": buffer_m,
                    "n_flooded": n_flooded,
                    "n_land": n_land,
                    "frac_flooded": frac,
                }
            )
        except Exception:
            log.exception(
                f"grid_type={grid_type} buffer={buffer_m} m failed -- skipping"
            )
            continue

        pd.DataFrame(results).to_csv(csv_path, index=False)

if results:
    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(8, 5))
    for grid_type, df_g in df.groupby("grid_type"):
        ax.plot(df_g["buffer_m"], df_g["n_flooded"], marker="o", label=grid_type)
    ax.set_xlabel("Additional river-depth buffer (m)")
    ax.set_ylabel(f"Wet cells (h > {min_inundation_depth_m} m)")
    ax.set_title(f"Basin {BASIN_ID} -- spin-up wet-cell count vs. river-depth buffer")
    ax.legend(title="Grid type")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = FIGS_DIR / f"river_depth_sensitivity_{BASIN_ID}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    log.info(f"Plot written: {plot_path}")
    log.info(f"CSV written: {csv_path}")
else:
    log.warning("No successful runs -- nothing to plot")
