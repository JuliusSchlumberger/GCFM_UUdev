"""
08_build_sfincs.py — Build a SFINCS model for a single delta basin.

The model is constructed in one continuous in-memory session (mode="w+")
so that no intermediate binary files need to be written and re-read between
steps.  Each major setup step is in its own clearly labelled section; add
new steps (mask boundary, roughness, subgrid, forcing …) below as they are
developed.

Inputs (from Snakemake)
-----------------------
delta_polygon     GeoPackage of the basin delta polygon
elevation_merged  Blended DiluviumDEM+GEBCO GeoTIFF (UTM, from rule 03f)
roughness         Manning's n GeoTIFF (from rule 03d)

Outputs
-------
sfincs.inp        SFINCS config / sentinel that the model was written
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import yaml
from hydromt_sfincs import SfincsModel

plt.ioff()

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=snakemake.log[0],
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)
for _name in ("hydromt", "hydromt_sfincs"):
    _l = logging.getLogger(_name)
    _l.setLevel(logging.INFO)
    _l.handlers = []
    _l.addHandler(logging.FileHandler(snakemake.log[0]))

# ── paths & params ────────────────────────────────────────────────────────────
domain_path           = Path(snakemake.input.domain_gpkg)
elevation_merged_path = Path(snakemake.input.elevation_merged)
roughness_path        = Path(snakemake.input.roughness)
land_polygons_path    = Path(snakemake.input.land_polygons)
river_network_path    = Path(snakemake.input.river_network)
surge_forcing_path    = Path(snakemake.input.surge_forcing)
river_forcing_path    = Path(snakemake.input.river_forcing)

inputs_dir        = Path(snakemake.params.inputs_dir)
sfincs_root       = Path(snakemake.params.sfincs_root)
resolution        = snakemake.params.resolution
include_subgrid   = snakemake.params.include_subgrid
clip_elevation_m  = snakemake.params.clip_elevation_m
nr_subgrid_pixels = snakemake.params.nr_subgrid_pixels
nr_levels         = snakemake.params.nr_levels
nrmax             = snakemake.params.nrmax
tref_str          = snakemake.params.tref
dtmapout          = snakemake.params.dtmapout
dtmaxout          = snakemake.params.dtmaxout
dthisout          = snakemake.params.dthisout
storevelmax       = snakemake.params.storevelmax
storetwet         = snakemake.params.storetwet
include_rstart    = snakemake.params.include_rstart
spinup_days       = snakemake.params.spinup_days

sfincs_root.mkdir(parents=True, exist_ok=True)

# ── data catalog ──────────────────────────────────────────────────────────────
# HydroMT v1.3.1 catalog schema: 'uri' (not 'path'), driver string name,
# no 'filesystem' or 'crs' top-level fields (Pydantic forbids extras).
local_catalog_path = sfincs_root / "data_catalog_local.yml"
local_catalog = {
    "meta": {"root": str(inputs_dir)},
    "local_elevation_merged": {
        "data_type": "RasterDataset",
        "uri": str(elevation_merged_path),   # absolute — lives under sfincs_root
        "driver": "rasterio",
    },
    "local_roughness": {
        "data_type": "RasterDataset",
        "uri": str(roughness_path),       # relative to inputs_dir
        "driver": "rasterio",
    },
    "local_land_polygons": {
        "data_type": "GeoDataFrame",
        "uri": str(land_polygons_path),  # relative to inputs_dir
        "driver": "pyogrio",
    },
    "local_zsini": {
        "data_type": "RasterDataset",
        "uri": "domain/zsini.tif",           # relative to inputs_dir
        "driver": "rasterio",
    },
}
with open(local_catalog_path, "w") as fh:
    yaml.dump(local_catalog, fh, sort_keys=False)
log.info(f"Data catalog written: {local_catalog_path}")

# ── load domain boundary ────────────────────────────────────────────────────────
delta_domain = gpd.read_file(domain_path)
log.info(f"Domain boundary: {len(delta_domain)} feature(s), CRS={delta_domain.crs}")

# ── initialise model ──────────────────────────────────────────────────────────
sf = SfincsModel(
    data_libs=[str(local_catalog_path)],
    root=str(sfincs_root),
    mode="w+",
    write_gis=True,
)
log.info("SfincsModel initialised")

# ── 1. Grid ───────────────────────────────────────────────────────────────────
sf.grid.create_from_region(
    region={"geom": delta_domain},
    res=resolution,
    crs="utm",
    rotated=False,
)
log.info(f"Grid created: {resolution} m, auto-UTM")

# ── 2. Elevation ──────────────────────────────────────────────────────────────
sf.elevation.create(
    elevation_list=[{"elevation": "local_elevation_merged"}],
)
log.info("Elevation set from 'local_elevation_merged'")

# ── 3. Mask: active cells ─────────────────────────────────────────────────────
# Use include_polygon + include_zmax (not global zmax) so the elevation filter
# is scoped to cells within the polygon.  Global zmax activates ALL grid cells
# below the threshold first, then include_polygon *adds* (overrides), which
# leaves ocean cells in the rectangular bounding box corners active.
# sf.mask.create_active(include_polygon=delta_domain, include_zmax=clip_elevation_m)
# log.info(
#     f"Active mask created: cells within delta polygon AND below {clip_elevation_m} m"
# )
sf.mask.create_active(include_polygon=delta_domain)
log.info(
    f"Active mask created: cells within delta polygon."
)

# ── 4. Mask: waterlevel boundary ──────────────────────────────────────────────
# Active-domain edge cells NOT covered by land polygons become waterlevel
# boundary (mask=2).  This marks the coastal / open-water perimeter as the
# tidal forcing boundary.  'local_land_polygons' is a catalog key so
# create_boundary() resolves it via get_geodataframe() — but for basins where
# OSM land is entirely outside the domain bbox, that file is empty and
# hydromt's pyogrio driver raises NoDataException on read.  In that case there
# is no land to exclude, so every active-domain edge cell should become a
# waterlevel boundary cell — simply omit exclude_polygon.
land_polygons_empty = gpd.read_file(land_polygons_path).empty
boundary_kwargs = {} if land_polygons_empty else {"exclude_polygon": "local_land_polygons"}
if land_polygons_empty:
    log.info("No land polygons in domain — boundary covers the full active-domain edge")
sf.mask.create_boundary(
    btype="waterlevel",
    reset_bounds=True,
    **boundary_kwargs,
)
log.info("Waterlevel boundary set: edge cells not on land → mask=2")

# ── 5. Initial conditions (spatially varying zsini) ───────────────────────────
# zsini = elevation_merged - 0.01 m, computed in rule 03f alongside the DEM.
# create() calls get_rasterdataset() then reproject_like(mask), so the mask
# must already exist (steps 3 & 4 above).  Where zsini is nodata (-9999),
# SFINCS falls back to using the local bed level as the initial water level.
sf.initial_conditions.create(ini="local_zsini")
log.info("Initial conditions set from 'local_zsini' (elevation − 1 cm)")

# ── 6. Roughness ─────────────────────────────────────────────────────────────
# roughness.create() option (1): {'manning': catalog_key} loads a pre-computed
# Manning's n raster directly — no landuse reclassification needed.
# 'local_roughness' → domain/roughness.tif (registered in data_catalog_local.yml).
sf.roughness.create(
    roughness_list=[{"manning": "local_roughness"}],
)
log.info("Manning roughness set from 'local_roughness'")

# ── 6. Simulation config ──────────────────────────────────────────────────────
# tref = tstart: forcing timeseries are in "hours since simulation start" so
# any fixed reference date works — the origin t=0 maps to tref exactly.
# tstop is derived from the actual end of the forcing files so it stays
# consistent even if config parameters (lead_days, period_hr) change.

tref = datetime.strptime(tref_str, "%Y-%m-%d %H:%M:%S")

# Read simulation duration from forcing files (time coord is hours since start,
# stored as float; decode_times=False is required for this non-CF time unit).
with xr.open_dataset(surge_forcing_path, decode_times=False) as surge_ds:
    surge_end_hr = float(surge_ds.time.max())
with xr.open_dataset(river_forcing_path, decode_times=False) as river_ds:
    river_end_hr = float(river_ds.time.max())

sim_hours = max(surge_end_hr, river_end_hr)
tstop = tref + timedelta(hours=sim_hours)

log.info(
    f"Simulation period: {tref} → {tstop} "
    f"({sim_hours:.0f} h from forcing files)"
)

# Phase 1: set tref/tstop/output params NOW so the rest of the build uses them.
# tstart is intentionally left as tref here so that HydroMT writes the forcing
# files (sections 8–9) starting from day 0.  If we set tstart=day10 at this
# point, HydroMT would clip the forcing to [day10,tstop] and the spinup (which
# runs from day 0) would find forcing that doesn't cover its simulation period.
sf.config.update({
    "tref":        tref,
    "tstart":      tref,        # placeholder — overridden before sf.write()
    "tstop":       tstop,
    "dtmapout":    dtmapout,
    "dtmaxout":    dtmaxout,
    "dthisout":    dthisout,
    "storevelmax": storevelmax,
    "storetwet":   storetwet,
    "baro":        0,           # no wind/atmosphere data
})
log.info("Simulation config set (tstart=tref placeholder, updated before write)")

# ── 7. Subgrid table ──────────────────────────────────────────────────────────
# Controlled by config["sfincs"]["include_subgrid"].
# When disabled the rule still writes sfincs.sbg as a placeholder (empty file)
# so Snakemake's output check passes; sfincs.inp will NOT contain a sbgfile
# directive, so SFINCS can run without subgrid (coarser, but valid).
rivers = gpd.read_file(river_network_path)
rivers["rivwth"] = rivers["width"].fillna(1.0).astype(float)
rivers["rivdph"] = rivers["rivdph"].clip(lower=0.0).astype(float)

if include_subgrid:
    log.info(
        f"River network for subgrid: {len(rivers)} reaches, "
        f"rivwth [{rivers['rivwth'].min():.1f}–{rivers['rivwth'].max():.1f} m], "
        f"rivdph [{rivers['rivdph'].min():.2f}–{rivers['rivdph'].max():.2f} m]"
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
    log.info(
        f"Subgrid table created: {nr_subgrid_pixels} px/cell → "
        f"{resolution / nr_subgrid_pixels:.0f} m effective resolution"
    )
else:
    log.info("Subgrid skipped (include_subgrid=false in config)")

# ── 8. Water-level boundary forcing (surge) ──────────────────────────────────
# surge_forcing.nc stores time as "hours since simulation start" (non-CF).
# We convert to absolute datetimes using tref (= tstart) so HydroMT can match
# the timeseries to the model's time window.
#
# sf.water_level.create() with timeseries + locations writes ASCII .bnd/.bzs
# files and spatially matches each station to the nearest boundary cell (mask=2)
# within the given buffer.

surge_ds = xr.open_dataset(surge_forcing_path, decode_times=False)
surge_times = pd.DatetimeIndex(
    [tref + timedelta(hours=float(h)) for h in surge_ds.time.values]
)

n_stations = surge_ds.sizes["station"]
stations_gdf = gpd.GeoDataFrame(
    {"index": range(n_stations)},
    geometry=gpd.points_from_xy(
        surge_ds.longitude.values,
        surge_ds.latitude.values,
    ),
    crs="EPSG:4326",
)

# water_level dims: (station, time) → transpose to (time, station) for DataFrame
wl_df = pd.DataFrame(
    data=surge_ds.water_level.values.T,
    index=surge_times,
    columns=range(n_stations),
)

sf.water_level.create(
    timeseries=wl_df,
    locations=stations_gdf,
    buffer=25e3,
)
log.info(f"Water-level forcing: {n_stations} stations, {len(surge_times)} time steps")

# ── 9. River discharge forcing ────────────────────────────────────────────────
# river_forcing.nc holds one timeseries per boundary crossing.  Only crossings
# with has_glofas=1 have a calibrated EVA fit and a meaningful discharge signal;
# crossings without GloFAS data carry zero discharge and are excluded.
#
# sf.discharge_points.create() places source cells at the crossing coordinates
# (snapped to the nearest active grid cell) and assigns the timeseries.
# The colleague's combined_dataset_deltas is NOT needed: our crossing coordinates
# and timeseries come directly from the river forcing NetCDF.

river_ds = xr.open_dataset(river_forcing_path, decode_times=False)
active = river_ds.has_glofas.values.astype(bool)
n_active = int(active.sum())
n_total  = len(active)
log.info(f"River forcing: {n_active}/{n_total} crossings with valid GloFAS data")

if n_active == 0:
    log.warning("No active river crossings — discharge forcing skipped")
else:
    river_times = pd.DatetimeIndex(
        [tref + timedelta(hours=float(h)) for h in river_ds.time.values]
    )

    crossings_gdf = gpd.GeoDataFrame(
        {"index": range(n_active)},
        geometry=gpd.points_from_xy(
            river_ds.longitude.values[active],
            river_ds.latitude.values[active],
        ),
        crs="EPSG:4326",
    )

    # discharge dims: (crossing, time) → transpose to (time, crossing) for DataFrame
    dis_df = pd.DataFrame(
        data=river_ds.discharge.values[active, :].T,
        index=river_times,
        columns=range(n_active),
    )

    # Filter crossing points to those within the active SFINCS region.
    # Centroids of inside-domain reaches can land in a cell that is above
    # clip_elevation_m (therefore inactive); discharge_points.create() clips
    # locations to the active-cells polygon and raises NoDataException when
    # none survive.  A 1-cell buffer prevents false negatives at cell edges.
    buf_deg = float(resolution) / 111_000.0
    region_geom = sf.region.to_crs("EPSG:4326").geometry.union_all().buffer(buf_deg)
    in_region = crossings_gdf.geometry.within(region_geom)
    crossings_filt = crossings_gdf[in_region].copy()
    n_outside = int((~in_region).sum())
    if n_outside:
        log.warning(
            f"  {n_outside}/{n_active} discharge crossing(s) outside active SFINCS "
            f"region (cell elevation ≥ {clip_elevation_m} m) — skipped"
        )

    if crossings_filt.empty:
        log.warning("All discharge crossings outside active region — discharge forcing skipped")
    else:
        # Pass full dis_df — discharge_points.create() reindexes it to only the
        # columns matching crossings_filt["index"] (the original crossing indices).
        sf.discharge_points.create(
            timeseries=dis_df,
            locations=crossings_filt,
        )
        log.info(
            f"Discharge forcing: {len(crossings_filt)}/{n_active} source point(s), "
            f"{len(river_times)} time steps"
        )

# ── 10. Observation points ────────────────────────────────────────────────────
# For each of the 2 boundary crossings with the highest bankfull discharge,
# place 5 evenly-spaced observation points along the downstream river path.
# river_ds and active/n_active are still in scope from section 9.

N_OBS_PER_CROSSING = 5
N_TOP_CROSSINGS    = 2

def _norm_rid(x):
    """Normalise a reach ID to a plain int string, or None."""
    s = str(x).strip()
    return None if s.lower() in ("nan", "none", "<na>", "") else (
        str(int(float(s))) if s.replace(".", "").lstrip("-").isdigit() else s or None
    )

obs_points_list = []

if n_active >= 1 and not rivers.empty:
    bankfull_vals = river_ds.bankfull_discharge.values.copy()
    bankfull_vals[~active] = -np.inf   # exclude inactive crossings
    src_lons_all  = river_ds.longitude.values
    src_lats_all  = river_ds.latitude.values

    top_indices = np.argsort(bankfull_vals)[::-1][:N_TOP_CROSSINGS]
    top_indices = [i for i in top_indices if bankfull_vals[i] > 0]

    rivers_utm     = rivers.to_crs(sf.crs)
    centroids_utm  = rivers_utm.geometry.centroid
    reach_lookup   = {_norm_rid(r["reach_id"]): r
                      for _, r in rivers_utm.iterrows()
                      if _norm_rid(r.get("reach_id"))}

    obs_id = 1
    for rank, ci in enumerate(top_indices):
        src_pt = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([src_lons_all[ci]], [src_lats_all[ci]]),
            crs="EPSG:4326",
        ).to_crs(sf.crs).geometry.iloc[0]

        nearest_rid = _norm_rid(
            rivers_utm.iloc[centroids_utm.distance(src_pt).idxmin()]["reach_id"]
        )

        # Walk downstream, collect reach geometries
        dn_rows, current, visited = [], nearest_rid, set()
        for _ in range(200):
            if not current or current in visited or current not in reach_lookup:
                break
            visited.add(current)
            row = reach_lookup[current]
            dn_rows.append(row)
            dn_raw = str(row.get("rch_id_dn", "") or "").strip().strip("[]")
            if not dn_raw or dn_raw.lower() in ("nan", "none", "<na>"):
                break
            nxt = next((_norm_rid(t.strip()) for t in dn_raw.split(",")
                        if _norm_rid(t.strip())), None)
            if not nxt:
                break
            current = nxt

        if not dn_rows:
            log.warning(f"Crossing rank {rank+1}: no downstream reaches — skipping")
            continue

        dn_line   = gpd.GeoDataFrame(dn_rows, crs=sf.crs).geometry.union_all()
        total_len = dn_line.length
        for i in range(N_OBS_PER_CROSSING):
            dist = (i + 0.5) * total_len / N_OBS_PER_CROSSING
            obs_points_list.append({"obs_id": obs_id, "geometry": dn_line.interpolate(dist)})
            obs_id += 1

        log.info(
            f"Crossing rank {rank+1} (Q={bankfull_vals[ci]:.0f} m³/s): "
            f"{N_OBS_PER_CROSSING} obs points along {len(dn_rows)} downstream reaches"
        )

if obs_points_list:
    obs_gdf = gpd.GeoDataFrame(obs_points_list, crs=sf.crs)
    sf.observation_points.create(locations=obs_gdf, merge=False)
    log.info(f"Observation points: {len(obs_points_list)} total")
else:
    log.warning("No observation points placed")

# ── Phase 2 config: set final tstart and rstfile ─────────────────────────────
# Now that all forcing files are built (starting from tref/day0), update tstart
# to the spinup end so the event model picks up exactly where the spinup left off.
if include_rstart:
    tstart_event = tref + timedelta(days=spinup_days)
    # SFINCS v2.3 names rst files sfincs.YYYYMMDD.HHMMSS.rst (timestamp at trstout)
    rst_fname = f"sfincs.{tstart_event.strftime('%Y%m%d.%H%M%S')}.rst"
    sf.config.update({
        "tstart": tstart_event,
        "rstfile": f"spinup/{rst_fname}",
    })
    log.info(
        f"tstart updated to {tstart_event} ({spinup_days} days after tref); "
        f"rstfile = spinup/sfincs.rst"
    )

# ── write ─────────────────────────────────────────────────────────────────────
sf.write()
log.info(f"Model written to {sfincs_root}")

# When subgrid is disabled, sf.write() does not create sfincs.sbg.
# Touch a placeholder so Snakemake's output check passes.
sbg_path = Path(snakemake.output.sfincs_sbg)
if not sbg_path.exists():
    sbg_path.touch()
    log.info("sfincs.sbg placeholder written (subgrid not built)")

# ── diagnostic plots ──────────────────────────────────────────────────────────
# Plots are built manually (not via fn_out) so we can:
#   • add a 15% buffer around the domain extent before saving
#   • show observation points only on the elevation (dep) plot

figs_dir = sfincs_root / "figs"
figs_dir.mkdir(parents=True, exist_ok=True)

def _save_with_buffer(fig, ax, fname, buffer_frac=0.15):
    """Expand axes extent by buffer_frac, then save."""
    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()
    dx, dy = (xr - xl) * buffer_frac, (yt - yb) * buffer_frac
    ax.set_xlim(xl - dx, xr + dx)
    ax.set_ylim(yb - dy, yt + dy)
    fig.savefig(figs_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Plot written: {fname}")

# 1. Grid extent
fig, ax = sf.plot_basemap(variable="grid", plot_region=True, bmap="sat",
                          plot_geoms=False)
_save_with_buffer(fig, ax, "01_grid.png")

# 2. Elevation — obs points shown here only (plot_geoms=True is default)
fig, ax = sf.plot_basemap(variable="dep", bmap="sat", vmin=-80, vmax=80)
_save_with_buffer(fig, ax, "02_elevation.png")

# 3. Mask
fig, ax = sf.plot_basemap(variable="mask", plot_bounds=False, bmap="sat",
                          plot_geoms=False)
_save_with_buffer(fig, ax, "03_mask.png")

# 4. Roughness
fig, ax = sf.plot_basemap(variable="manning", plot_bounds=False, bmap="sat",
                          plot_geoms=False)
_save_with_buffer(fig, ax, "04_roughness.png")

# 5. Forcing
fig, ax = sf.plot_forcing()
if fig is not None:
    fig.savefig(figs_dir / "05_forcing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot written: 05_forcing.png")
