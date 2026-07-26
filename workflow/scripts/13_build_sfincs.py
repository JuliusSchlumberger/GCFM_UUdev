"""
13_build_sfincs.py — Build a SFINCS model for a single delta basin.

The model is constructed in one continuous in-memory session (mode="w+")
so that no intermediate binary files need to be written and re-read between
steps.  Each major setup step is in its own clearly labelled section; add
new steps (mask boundary, roughness, subgrid, forcing …) below as they are
developed.

Inputs (from Snakemake, see 13_build_sfincs.smk for the full list)
--------------------------------------------------------------------
elevation_merged  Blended FathomDEM+GEBCO GeoTIFF, or the conditioned DEM
                  when river conditioning is enabled (UTM, from rule 05a/10)
roughness         Manning's n GeoTIFF (from rule 05c)

Outputs
-------
sfincs.inp        SFINCS config / sentinel that the model was written

Forcing mode (boundary_setup.mode in config.yml)
---------------------------------------------------
Controls which forcing(s) actually drive the model, independent of the
input files themselves (always loaded so duration/observation points stay
consistent across modes):
  "compound"     — real surge/tide boundary + real river discharge (default).
                   boundary_setup.compound.lag_hr optionally shifts the
                   river discharge timeseries relative to the surge
                   timeseries (positive = river peak arrives later; negative
                   = earlier) — see the "compound lag" section below.
  "coastal_only" — real surge/tide boundary; river discharge forced to 0 at
                   every crossing (discharge points are still created, just
                   with a zeroed timeseries, so the model structure matches
                   the other modes).
  "river_only"   — real river discharge; the coastal water-level boundary is
                   replaced by a flat constant at zsini's sea-level baseline
                   (baseline_m) — no surge/tide variability, and no initial
                   mismatch against zsini since both share baseline_m.
"""

import logging
import math
import re
from datetime import datetime, timedelta
from pathlib import Path

import rasterio
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import yaml
from scipy.ndimage import label as _ndimage_label
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from hydromt_sfincs import SfincsModel

from src.river_forcing import build_design_discharge_matrix
from src.surge import build_design_surge_matrix

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
delta_outflow_points_path = Path(snakemake.input.delta_outflow_points)
zsini_path            = Path(snakemake.input.zsini)
surge_forcing_path    = Path(snakemake.input.surge_forcing)
river_forcing_path    = Path(snakemake.input.river_forcing)
burn_rivers_enabled   = snakemake.params.burn_rivers_enabled
river_burned_dem_path = (
    Path(snakemake.input.river_burned_dem) if burn_rivers_enabled else None
)

inputs_dir        = Path(snakemake.params.inputs_dir)
sfincs_root       = Path(snakemake.params.sfincs_root)
resolution        = snakemake.params.resolution
include_subgrid   = snakemake.params.include_subgrid
nr_subgrid_pixels = snakemake.params.nr_subgrid_pixels
nr_levels         = snakemake.params.nr_levels
nrmax             = snakemake.params.nrmax
tref_str          = snakemake.params.tref
dtmapout          = snakemake.params.dtmapout
dtmaxout          = snakemake.params.dtmaxout
dthisout          = snakemake.params.dthisout
storevelmax       = snakemake.params.storevelmax
storetwet         = snakemake.params.storetwet
forcing_mode      = snakemake.params.forcing_mode
design_rp_river_yr = snakemake.params.design_rp_river_yr # Removed Float 
design_rp_surge_yr = snakemake.params.design_rp_surge_yr # Added 
compound_lag_hr   = float(snakemake.params.compound_lag_hr)
flat_boundary_point_spacing_m = snakemake.params.flat_boundary_point_spacing_m
waterlevel_buffer_m = snakemake.params.waterlevel_buffer_m
outflow_buffer_m   = snakemake.params.outflow_buffer_m
n_top_crossings    = snakemake.params.n_top_crossings
n_per_crossing     = snakemake.params.n_per_crossing
max_downstream_hops = snakemake.params.max_downstream_hops
include_rstart    = snakemake.params.include_rstart
spinup_days       = snakemake.params.spinup_days
preburn_enabled   = snakemake.params.preburn_enabled
quadtree_enabled  = snakemake.params.quadtree_enabled
river_refinement_level     = snakemake.params.river_refinement_level
river_buffer_factor        = snakemake.params.river_buffer_factor
coastal_refinement_enabled = snakemake.params.coastal_refinement_enabled
coastal_refinement_level   = snakemake.params.coastal_refinement_level
coastal_buffer_m           = snakemake.params.coastal_buffer_m

sfincs_root.mkdir(parents=True, exist_ok=True)
log.info(f"Forcing mode: {forcing_mode!r}")

# ── baseline water level from surge forcing ───────────────────────────────────
# Read baseline_m from surge_forcing.nc: the mean vertical correction (MDT +
# SLR) applied to rp_level across all selected boundary stations.  This value
# must be used as the initial ocean water level so the model's spin-up starts
# at the same vertical reference as the lead period of the boundary forcing.
# It is 0.0 when both corrections are disabled (default).
with xr.open_dataset(surge_forcing_path, decode_times=False) as _ds:
    baseline_m = float(_ds["baseline_m"].values) if "baseline_m" in _ds else 0.0
log.info(f"Surge boundary baseline read from surge_forcing.nc: {baseline_m:+.4f} m")

# ── corrected zsini raster ────────────────────────────────────────────────────
# zsini.tif (rule 05b) has two values: 0.0 for sea cells (not on OSM land
# polygons), -9999 nodata for land cells.  When baseline_m != 0 we write a
# corrected copy shifting sea cells from 0.0 to baseline_m so the initial
# ocean state matches the boundary forcing vertical reference.
_zsini_src = zsini_path
if baseline_m != 0.0:
    _zsini_corr = sfincs_root / "zsini_baseline.tif"
    with rasterio.open(_zsini_src) as _src:
        _arr = _src.read(1).astype(np.float32)
        _meta = _src.meta.copy()
        _nodata_val = np.float32(_src.nodata if _src.nodata is not None else -9999.0)
    _arr = np.where(_arr == np.float32(0.0), np.float32(baseline_m), _arr)

    # ── connected-component dry-out of isolated sea cells ─────────────────────
    # When baseline_m is significantly negative (e.g. −0.50 m after protection-
    # level correction) shallow connections between isolated depressions and the
    # main ocean become dry, trapping pockets of water that were previously able
    # to drain.  Fix: label connected components of sea cells (value=baseline_m),
    # keep only those reachable from the domain boundary, and set isolated wet
    # cells (dep < baseline_m, so water depth > 0) to dep = dry start.
    # Both zsini.tif and elevation_merged.tif share the same grid (zsini.tif
    # is reprojected onto elevation_merged.tif's grid in rule 05b), so
    # pixel-aligned comparison is exact.
    with rasterio.open(elevation_merged_path) as _dep_src:
        _dep = _dep_src.read(1).astype(np.float32)
        _dep_nd = np.float32(_dep_src.nodata if _dep_src.nodata is not None else -9999.0)
    _dep_valid = np.where(np.isclose(_dep, _dep_nd), np.float32(1e6), _dep)

    _sea_mask = np.isclose(_arr, np.float32(baseline_m))           # cells carrying baseline_m
    _labeled, _n = _ndimage_label(_sea_mask)                        # 4-connected components
    # Edge labels = components touching the raster boundary (= open-ocean side)
    _edge_lbl: set[int] = set()
    for _edge in (_labeled[0, :], _labeled[-1, :], _labeled[:, 0], _labeled[:, -1]):
        _edge_lbl.update(_edge.tolist())
    _edge_lbl.discard(0)
    _connected = np.isin(_labeled, list(_edge_lbl)) & _sea_mask
    _isolated_wet = _sea_mask & ~_connected & (_dep_valid < np.float32(baseline_m))
    _n_fix = int(_isolated_wet.sum())
    if _n_fix > 0:
        # Set isolated wet sea cells to their bed elevation → water depth = 0 (dry).
        _arr = np.where(_isolated_wet, _dep_valid, _arr)
        log.info(
            f"zsini connectivity fix: {_n_fix} isolated wet sea-cell(s) set to dep "
            f"(not reachable from domain boundary at baseline_m={baseline_m:+.4f} m)"
        )
    else:
        log.info("zsini connectivity fix: no isolated wet sea cells found")

    with rasterio.open(_zsini_corr, "w", **_meta) as _dst:
        _dst.write(_arr, 1)
    _zsini_uri = str(_zsini_corr)
    log.info(f"Corrected zsini written: {_zsini_corr} (sea cells = {baseline_m:+.4f} m)")
else:
    _zsini_uri = str(zsini_path.relative_to(inputs_dir))

# ── delta-outline outflow points ──────────────────────────────────────────────
# Rule clean_river_network's identify_delta_outflow_points always produces this
# file, but it is typically empty (0 features) -- only register/consume it
# when it actually has points, so an empty file never reaches hydromt_sfincs.
delta_outflow_gdf = gpd.read_file(delta_outflow_points_path)
delta_outflow_enabled = not delta_outflow_gdf.empty
log.info(f"Delta-outline outflow points: {len(delta_outflow_gdf)}")

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
    **({
        "local_river_burned": {
            "data_type": "RasterDataset",
            "uri": str(river_burned_dem_path),
            "driver": "rasterio",
        },
    } if burn_rivers_enabled else {}),
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
    **({
        "local_delta_outflow_points": {
            "data_type": "GeoDataFrame",
            "uri": str(delta_outflow_points_path),
            "driver": "pyogrio",
        },
    } if delta_outflow_enabled else {}),
    "local_zsini": {
        "data_type": "RasterDataset",
        "uri": _zsini_uri,
        "driver": "rasterio",
    },
}
with open(local_catalog_path, "w") as fh:
    yaml.dump(local_catalog, fh, sort_keys=False)
log.info(f"Data catalog written: {local_catalog_path}")

# When burn_rivers is enabled, 'local_river_burned' (channel-only, native
# resolution) takes priority over 'local_elevation_merged' wherever it has
# valid data (hydromt_sfincs's elevation_list/merge_multi_dataarrays merges
# by priority: first source wins, later sources fill gaps) — so ocean/
# floodplain/gap cells transparently fall back to the existing DEM.
elevation_list = (
    [{"elevation": "local_river_burned"}, {"elevation": "local_elevation_merged"}]
    if burn_rivers_enabled
    else [{"elevation": "local_elevation_merged"}]
)

# ── load domain boundary ────────────────────────────────────────────────────────
delta_domain = gpd.read_file(domain_path)
log.info(f"Domain boundary: {len(delta_domain)} feature(s), CRS={delta_domain.crs}")

# ── quadtree refinement zones ─────────────────────────────────────────────────
# Resolve the UTM CRS upfront (instead of the magic string "utm") so refinement
# polygons can be reprojected to the model's exact target CRS before
# create_from_region runs — quadtree_grid.create_from_region does not reproject
# refinement polygons internally.
if quadtree_enabled:
    from hydromt.gis import parse_crs
    from src.quadtree_refinement import build_refinement_polygons

    bounds_4326 = delta_domain.to_crs("EPSG:4326").total_bounds
    target_crs = parse_crs("utm", bbox=list(bounds_4326))
    log.info(f"Quadtree enabled — resolved target CRS: {target_crs}")

    refinement_gdf = build_refinement_polygons(
        river_network_path, land_polygons_path,
        river_refinement_level, river_buffer_factor,
        coastal_refinement_enabled, coastal_refinement_level, coastal_buffer_m,
        target_crs=target_crs,
    )
    refinement_gdf.to_file(snakemake.output.refinement_polygons)
    log.info(f"Refinement polygons written: {len(refinement_gdf)} zone(s)")
else:
    target_crs = "utm"

# ── initialise model ──────────────────────────────────────────────────────────
sf = SfincsModel(
    data_libs=[str(local_catalog_path)],
    root=str(sfincs_root),
    mode="w+",
    write_gis=True,
)
log.info("SfincsModel initialised")

# ── 1. Grid ───────────────────────────────────────────────────────────────────
if quadtree_enabled:
    sf.quadtree_grid.create_from_region(
        region={"geom": delta_domain},
        res=resolution,
        crs=target_crs,
        rotated=False,
        refinement_polygons=refinement_gdf,
    )
    log.info(f"Quadtree grid created: {resolution} m base resolution, CRS={target_crs}")
else:
    sf.grid.create_from_region(
        region={"geom": delta_domain},
        res=resolution,
        crs="utm",
        rotated=False,
    )
    log.info(f"Grid created: {resolution} m, auto-UTM")

# ── 2. Elevation ──────────────────────────────────────────────────────────────
elevation_component = sf.quadtree_elevation if quadtree_enabled else sf.elevation
elevation_component.create(
    elevation_list=elevation_list,
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
mask_component = sf.quadtree_mask if quadtree_enabled else sf.mask
mask_component.create_active(include_polygon=delta_domain)
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
mask_component.create_boundary(
    btype="waterlevel",
    reset_bounds=True,
    **boundary_kwargs,
)
log.info("Waterlevel boundary set: edge cells not on land → mask=2")

# ── 4b. Mask: delta-outline outflow boundary ─────────────────────────────────
# Reaches that cross the delta polygon's outline but are neither seed nor
# mouth (identify_delta_outflow_points, rule clean_river_network) are genuine
# places where flow exits the modelled network -- registered as a free
# outflow boundary (mask=3, no prescribed water level) at that location,
# rather than being dropped from the network. include_polygon_buffer turns
# each point into a small polygon so at least one grid cell is captured
# regardless of exactly where within a cell the point falls.
if delta_outflow_enabled:
    mask_component.create_boundary(
        btype="outflow",
        include_polygon="local_delta_outflow_points",
        include_polygon_buffer=outflow_buffer_m,
        reset_bounds=False,
    )
    log.info(
        f"Outflow boundary set: {len(delta_outflow_gdf)} delta-outline "
        f"crossing(s), {outflow_buffer_m:.0f} m buffer → mask=3"
    )

# ── 5. Initial conditions ──────────────────────────────────────────────────────
# zsini.tif (rule 05b) has two values: 0.0 for sea cells (not on OSM land
# polygons) and -9999 nodata for land cells.  When baseline_m != 0 the
# catalog points to a corrected copy with sea cells shifted to baseline_m
# so the initial ocean state matches the boundary forcing lead period.
# create() reprojects the raster onto the model grid (mask must exist first).
# Where nodata: SFINCS falls back to local bed level as initial water level.
initial_conditions_component = sf.quadtree_initial_conditions if quadtree_enabled else sf.initial_conditions
if quadtree_enabled:
    # quadtree_initial_conditions.create()'s default reproj_method="average" is not a
    # valid xugrid.OverlapRegridder method in the installed xugrid version (valid: "mean",
    # "harmonic_mean", ...) — override explicitly to work around this hydromt_sfincs bug.
    initial_conditions_component.create(ini="local_zsini", reproj_method="mean")
    # Second, separate hydromt_sfincs bug: create() sets config key "ncinifile"
    # (SfincsQuadtreeInitialConditions.create()), but SfincsQuadtreeGrid.write()
    # (the code that actually writes the netCDF) looks up "inifile" (no "nc"
    # prefix) to decide whether/where to write the "ini" variable — a key-name
    # mismatch means the carefully-computed spatially-varying initial water
    # level is silently NEVER written for a quadtree build; SFINCS falls back
    # to its own uniform default with no warning. Set the key write() actually
    # reads, matching the same filename create() already put in "ncinifile".
    # (This is removed again from the final sfincs.inp after sf.write(), once
    # it has served its purpose -- see the write() section below.)
    sf.config.update({"inifile": sf.config.get("ncinifile")})
else:
    initial_conditions_component.create(ini="local_zsini")
# HydroMT hardcodes zsini=0.0 in sfincs.inp after create().  Override to
# -9999 so any cell not covered by sfincs.ini falls back to bed level (dry),
# consistent with how sfincs.ini itself handles land cells.
sf.config.update({"zsini": -9999.0})
log.info(f"Initial conditions set: sea = {baseline_m:+.4f} m, land = -9999 (bed level / dry)")

# ── 6. Roughness ─────────────────────────────────────────────────────────────
# roughness.create() option (1): {'manning': catalog_key} loads a pre-computed
# Manning's n raster directly — no landuse reclassification needed.
# 'local_roughness' → domain/roughness.tif (registered in data_catalog_local.yml).
roughness_component = sf.quadtree_roughness if quadtree_enabled else sf.roughness
roughness_component.create(
    roughness_list=[{"manning": "local_roughness"}],
)
log.info("Manning roughness set from 'local_roughness'")

# ── 7. Simulation config ──────────────────────────────────────────────────────
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

# ── 8. Subgrid table ──────────────────────────────────────────────────────────
# Controlled by config["sfincs"]["subgrid"]["enabled"]. quadtree mode requires
# subgrid to be enabled (validated in 00_common.smk), since the subgrid's
# dep_subgrid.tif is the only regular reference raster available for
# postprocessing a quadtree run.
# When disabled the rule still writes sfincs_subgrid.nc as a placeholder
# (empty file) so Snakemake's output check passes; sfincs.inp will NOT
# contain a qtrfile/sbgfile directive, so SFINCS can run without subgrid
# (coarser, but valid).
rivers = gpd.read_file(river_network_path)
rivers["rivwth"] = rivers["width"].fillna(1.0).astype(float)
rivers["rivdph"] = rivers["rivdph"].clip(lower=0.0).astype(float)

# When river conditioning is enabled (and burn_rivers is NOT), rule 11
# (burn_river_bed) has pre-computed smooth bed anchor points
# (zbed_anchors.gpkg) with absolute rivbed [m+REF] values. These are passed
# directly to burn_river_rect as gdf_zb, which interpolates along merged
# river lines and only lowers the subgrid DEM where rivbed < DEM.
# rivdph is completely bypassed when gdf_zb is provided — no need to zero it.
#
# When burn_rivers IS enabled, the channel is already correctly burned into
# 'local_river_burned' (see elevation_list above) — burn_river_rect is
# skipped entirely (river_list=[] below) rather than running it a second
# time on an already-burned raster, since it's the actual source of the
# per-tile contamination artifact this feature exists to avoid.
zbed_gdf = None
if preburn_enabled and not burn_rivers_enabled:
    zbed_path = snakemake.input.zbed_anchors
    if zbed_path:   # empty list [] when disabled
        zbed_gdf = gpd.read_file(zbed_path)
        log.info(
            f"Loaded {len(zbed_gdf)} river bed anchor points from zbed_anchors.gpkg — "
            "passed as gdf_zb (rivbed) to subgrid; rivdph bypassed"
        )

# Clear any subgrid files from a previous build before writing new ones.
# A regular build writes a single dep_subgrid.tif/manning_subgrid.tif; a
# quadtree build writes one dep_subgrid_levN.tif/manning_subgrid_levN.tif
# pair per refinement level instead. Switching grid types for the same
# sfincs_root would otherwise leave stale files from the previous grid type
# behind — src.postprocessing.get_bed_level would then risk reading the
# wrong (stale) bed level for postprocessing.
subgrid_dir = sfincs_root / "subgrid"
if subgrid_dir.exists():
    for _stale in subgrid_dir.glob("*subgrid*.tif"):
        _stale.unlink()

if include_subgrid:
    log.info(
        f"River network for subgrid: {len(rivers)} reaches, "
        f"rivwth [{rivers['rivwth'].min():.1f}–{rivers['rivwth'].max():.1f} m], "
        f"rivdph [{rivers['rivdph'].min():.2f}–{rivers['rivdph'].max():.2f} m]"
    )
    subgrid_component = sf.quadtree_subgrid if quadtree_enabled else sf.subgrid
    if burn_rivers_enabled:
        river_list = []
        log.info("burn_rivers enabled — skipping burn_river_rect (channel already burned)")
    else:
        river_entry = {"centerlines": rivers}
        if zbed_gdf is not None:
            river_entry["gdf_zb"] = zbed_gdf
        river_list = [river_entry]
    subgrid_component.create(
        elevation_list=elevation_list,
        roughness_list=[{"manning": "local_roughness"}],
        river_list=river_list,
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

# ── 9. Water-level boundary forcing (surge) ──────────────────────────────────
# surge_forcing.nc stores time as "hours since simulation start" (non-CF).
# We convert to absolute datetimes using tref (= tstart) so HydroMT can match
# the timeseries to the model's time window.
#
# sf.water_level.create() with timeseries + locations writes ASCII .bnd/.bzs
# files and spatially matches each station to the nearest boundary cell (mask=2)
# within the given buffer.
#
# forcing_mode="river_only": replace the real surge/tide boundary with a flat
# constant at baseline_m -- the same value zsini's sea cells already carry
# (section 5), so there's no initial mismatch/transient at the coast, and no
# surge/tide variability to confound the river-discharge contribution.
if forcing_mode == "river_only":
    sf.water_level.create_boundary_points_from_mask(bnd_dist=flat_boundary_point_spacing_m)
    sf.water_level.create_timeseries(shape="constant", offset=baseline_m)
    log.info(
        f"Water-level forcing: flat constant boundary at {baseline_m:+.4f} m "
        f"(forcing_mode='river_only')"
    )
else:
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
        data=build_design_surge_matrix(surge_ds, design_rp_surge_yr).T,
        index=surge_times,
        columns=range(n_stations),
    )

    sf.water_level.create(
        timeseries=wl_df,
        locations=stations_gdf,
        buffer=waterlevel_buffer_m,
    )
    log.info(f"Water-level forcing: {n_stations} stations, {len(surge_times)} time steps")

# ── 10. River discharge forcing ───────────────────────────────────────────────
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

    # Built here (SFINCS-build time), not precomputed in river_forcing.nc --
    # looks up the design discharge at design_rp_river_yr from the stored
    # per-crossing GPD return-value table (log-RP interpolated), applies the
    # protection-discharge floor, then reconstructs the sinusoidal-wave
    # hydrograph -- see src.river_forcing.build_design_discharge_matrix.
    # discharge dims: (crossing, time) → transpose to (time, crossing) for DataFrame
    dis_df = pd.DataFrame(
        data=build_design_discharge_matrix(river_ds, active, design_rp_river_yr).T,
        index=river_times,
        columns=range(n_active),
    )

    # ── compound lag: shift river discharge relative to surge ────────────────
    # Rolls the whole discharge array by compound_lag_hr (converted to whole
    # time steps at the river forcing's own dt) and pads the side vacated by
    # the shift with each crossing's own bankfull_discharge (the same
    # baseline value the unshifted timeseries already rests at outside its
    # event window) so the array stays the same length as the shared time
    # axis. Positive lag delays the river peak relative to the surge peak;
    # negative lag brings it forward.
    if forcing_mode == "compound" and compound_lag_hr != 0 and len(river_times) > 1:
        dt_hr = float((river_times[1] - river_times[0]).total_seconds() / 3600.0)
        shift_steps = int(round(compound_lag_hr / dt_hr))
        if shift_steps != 0:
            bankfull_active = river_ds.bankfull_discharge.values[active]
            arr = dis_df.to_numpy()
            shifted = np.empty_like(arr)
            n = min(abs(shift_steps), arr.shape[0])
            if shift_steps > 0:
                shifted[:n, :] = bankfull_active[np.newaxis, :]
                shifted[n:, :] = arr[: arr.shape[0] - n, :]
            else:
                shifted[: arr.shape[0] - n, :] = arr[n:, :]
                shifted[arr.shape[0] - n :, :] = bankfull_active[np.newaxis, :]
            dis_df = pd.DataFrame(shifted, index=dis_df.index, columns=dis_df.columns)
            log.info(
                f"Compound lag applied: river discharge shifted {compound_lag_hr:+.1f} h "
                f"relative to surge ({shift_steps:+d} step(s) at dt={dt_hr:.2f} h); "
                f"{'start' if shift_steps > 0 else 'end'} padded with each "
                f"crossing's bankfull discharge"
            )

    if forcing_mode == "coastal_only":
        # Discharge points are still created below (zeroed) rather than skipped
        # entirely, so the model structure matches the other two modes.
        dis_df.loc[:, :] = 0.0
        log.info("River discharge forced to 0.0 m3/s at all crossings (forcing_mode='coastal_only')")

    # Filter crossing points to those within the active SFINCS region.
    # Centroids of inside-domain reaches can land in a cell outside the
    # active-cells polygon (delta_domain); discharge_points.create() clips
    # locations against self.model.region using an UNBUFFERED 'intersects'
    # check and raises NoDataException when none survive. Its own `buffer=`
    # parameter does NOT expand that check outward -- it instead restricts
    # acceptance to a ring near the boundary that is itself clipped back to
    # the same unbuffered region, so it can never rescue a point that falls
    # just outside (confirmed by reading hydromt_sfincs' discharge_points.
    # create(): `region.boundary.buffer(buffer).clip(self.model.region)`).
    # A 1-cell buffer is used here only to decide which crossings to keep;
    # any kept point that falls just outside the exact (unbuffered) region
    # is then snapped onto its boundary (nearest_points; a no-op for points
    # already inside) so every kept point is guaranteed to satisfy hydromt's
    # own 'intersects' check regardless of its buffer setting.
    buf_deg = float(resolution) / 111_000.0
    region_wgs84 = sf.region.to_crs("EPSG:4326").geometry.union_all()
    in_region = crossings_gdf.geometry.within(region_wgs84.buffer(buf_deg))
    crossings_filt = crossings_gdf[in_region].copy()
    n_outside = int((~in_region).sum())
    if n_outside:
        log.warning(
            f"  {n_outside}/{n_active} discharge crossing(s) outside active SFINCS "
            f"region — skipped"
        )

    if crossings_filt.empty:
        log.warning("All discharge crossings outside active region — discharge forcing skipped")
    else:
        # nearest_points() alone is not enough: it returns a point that is
        # mathematically ON the region boundary, but GEOS's floating-point
        # arithmetic can place it a hair outside, which then still fails
        # hydromt's own 'intersects' check inside create() (confirmed live:
        # a point snapped this way at R=60/n_cells=2 still reported
        # region.intersects(point) == False). Nudge REGION_SNAP_NUDGE_M
        # further from the boundary point, towards a point guaranteed to be
        # inside the region (representative_point()), so the result is
        # safely interior rather than balanced on a numerically fragile edge.
        REGION_SNAP_NUDGE_M = 1.0
        nudge_deg = REGION_SNAP_NUDGE_M / 111_000.0
        interior_anchor = region_wgs84.representative_point()

        def _snap_into_region(g):
            if region_wgs84.contains(g):
                return g
            boundary_pt = nearest_points(region_wgs84, g)[0]
            dx = interior_anchor.x - boundary_pt.x
            dy = interior_anchor.y - boundary_pt.y
            dist = math.hypot(dx, dy)
            if dist == 0:
                return boundary_pt
            frac = min(nudge_deg / dist, 1.0)
            return Point(boundary_pt.x + dx * frac, boundary_pt.y + dy * frac)

        crossings_filt["geometry"] = crossings_filt.geometry.apply(_snap_into_region)
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

# ── 11. Observation points ────────────────────────────────────────────────────
# For each of the N_TOP_CROSSINGS boundary crossings with the highest bankfull
# discharge, place N_OBS_PER_CROSSING evenly-spaced observation points along
# the downstream river path (up to max_downstream_hops reaches).
# river_ds and active/n_active are still in scope from section 10.

N_OBS_PER_CROSSING = n_per_crossing
N_TOP_CROSSINGS    = n_top_crossings

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
        for _ in range(max_downstream_hops):
            if not current or current in visited or current not in reach_lookup:
                break
            visited.add(current)
            row = reach_lookup[current]
            dn_rows.append(row)
            dn_raw = str(row.get("rch_id_dn", "") or "").strip().strip("[]")
            if not dn_raw or dn_raw.lower() in ("nan", "none", "<na>"):
                break
            nxt = next((_norm_rid(t.strip()) for t in re.split(r"[,\s]+", dn_raw)
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
        f"rstfile = spinup/{rst_fname}"
    )

# ── write ─────────────────────────────────────────────────────────────────────
sf.write()
log.info(f"Model written to {sfincs_root}")

if quadtree_enabled:
    # SfincsQuadtreeGrid.write() only splits the "ini" variable into its own
    # sfincs_ini.nc when config["inifile"] is set (write()'s generic per-variable
    # split loop keys off "{var}file", i.e. "inifile" for var="ini" -- it does
    # NOT check "ncinifile", which is why we set "inifile" earlier). That data
    # is now correctly written to disk. But the SFINCS kernel itself appears to
    # treat "inifile" (legacy regular-grid ASCII/binary ini format) and
    # "ncinifile" (netCDF, for quadtree) as mutually exclusive: leaving BOTH
    # keys in sfincs.inp pointing at the same netCDF file caused an immediate
    # access-violation crash at simulation start (observed for basin 4267691).
    # Drop the now-redundant "inifile" key and re-serialize sfincs.inp so the
    # kernel only sees "ncinifile", matching the actual file format on disk.
    sf.config.set("inifile", None)
    sf.config.write()
    log.info("Removed redundant 'inifile' key from sfincs.inp (kept 'ncinifile')")

# When subgrid is disabled, sf.write() does not create sfincs_subgrid.nc.
# Touch a placeholder so Snakemake's output check passes.
subgrid_path = Path(snakemake.output.sfincs_subgrid)
if not subgrid_path.exists():
    subgrid_path.touch()
    log.info("sfincs_subgrid.nc placeholder written (subgrid not built)")

# ── diagnostic plots ──────────────────────────────────────────────────────────
# Plots are built manually (not via fn_out) so we can:
#   • add a 15% buffer around the domain extent before saving
#   • show observation points only on the elevation (dep) plot

# Derived from the tracked plot_grid output rather than hardcoded relative to
# sfincs_root, so this directory always matches wherever the rule's output:
# block says these plots should go.
figs_dir = Path(snakemake.output.plot_grid).parent
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

# 1b. Quadtree refinement zones (quadtree mode only)
if quadtree_enabled:
    from src.plots import plot_refinement_zones

    _domain_wgs = delta_domain if delta_domain.crs.to_epsg() == 4326 else delta_domain.to_crs("EPSG:4326")
    _domain_union = _domain_wgs.geometry.union_all()
    _domain_poly = _domain_union if isinstance(_domain_union, Polygon) else _domain_union.convex_hull
    plot_refinement_zones(
        refinement_gdf, _domain_poly, str(land_polygons_path), str(river_network_path),
        str(figs_dir / "01b_refinement_zones.png"), basin_id=sfincs_root.parent.name,
    )

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

# 5. Forcing — custom plot reading directly from the forcing NC files so the
# full timeseries (lead period + event) is always shown, independent of
# tstart/tstop.  A vertical dashed line marks tstart_event (end of spinup).
import matplotlib.dates as _mdates

def _hours_to_dt(hours_arr):
    return [tref + timedelta(hours=float(h)) for h in hours_arr]

with xr.open_dataset(surge_forcing_path, decode_times=False) as _sds:
    _surge_times = _hours_to_dt(_sds.time.values)
    _wl = _sds["water_level"].values        # (station, time)
    _n_stn = _wl.shape[0]

with xr.open_dataset(river_forcing_path, decode_times=False) as _rds:
    _river_times  = _hours_to_dt(_rds.time.values)
    _active_mask  = _rds.has_glofas.values.astype(bool)
    _n_cross      = int(_active_mask.sum())
    _dis_active   = (
        build_design_discharge_matrix(_rds, _active_mask, design_rp_river_yr)
        if _n_cross > 0 else np.zeros((0, len(_river_times)))
    )

_tstart_event = tref + timedelta(days=spinup_days) if include_rstart else None

fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# — surge water level
for _i in range(_n_stn):
    ax5a.plot(_surge_times, _wl[_i, :], linewidth=0.8, alpha=0.7)
ax5a.set_ylabel("Water level (m+ref)")
ax5a.set_title("Surge boundary — water level (all stations)")
ax5a.grid(True, alpha=0.3)

# — river discharge
for _i in range(_n_cross):
    ax5b.plot(_river_times, _dis_active[_i, :], linewidth=0.8, alpha=0.7)
ax5b.set_ylabel("Discharge (m³/s)")
ax5b.set_title(f"River discharge ({_n_cross} active GloFAS crossing(s))")
ax5b.grid(True, alpha=0.3)

# Mark tstart_event on both panels
if _tstart_event is not None:
    for _ax in (ax5a, ax5b):
        _ax.axvline(_tstart_event, color="black", linewidth=1.2,
                    linestyle="--", label=f"event start (after {spinup_days}d spinup)")
        _ax.legend(fontsize=8, loc="upper right")

_locator   = _mdates.AutoDateLocator()
_formatter = _mdates.ConciseDateFormatter(_locator)
ax5b.xaxis.set_major_locator(_locator)
ax5b.xaxis.set_major_formatter(_formatter)

fig5.suptitle(
    f"SFINCS forcing — basin {sfincs_root.parent.name} "
    f"(full timeseries; dashed = event model tstart)",
    fontsize=10,
)
fig5.tight_layout()
fig5.savefig(figs_dir / "05_forcing.png", dpi=150, bbox_inches="tight")
plt.close(fig5)
log.info("Plot written: 05_forcing.png")

# 6. Initial conditions (zsini)
_zsini_plot_path = _zsini_corr if baseline_m != 0.0 else _zsini_src
with rasterio.open(_zsini_plot_path) as _src:
    _zsini_arr = _src.read(1).astype(np.float32)
    _nodata_val = _src.nodata
    _bounds = _src.bounds

if _nodata_val is not None:
    _zsini_arr = np.where(
        np.isclose(_zsini_arr, np.float32(_nodata_val)), np.nan, _zsini_arr
    )

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(
    _zsini_arr,
    extent=[_bounds.left, _bounds.right, _bounds.bottom, _bounds.top],
    origin="upper",
    cmap="Blues",
    aspect="auto",
)
plt.colorbar(im, ax=ax, label="Initial water level (m)")
ax.set_title(
    f"Initial conditions (zsini)\n"
    f"sea = {baseline_m:+.4f} m  |  land = nodata"
)
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
_save_with_buffer(fig, ax, "06_zsini.png")
