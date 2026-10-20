"""Calibrate river depth and riverbank weir crests with SFINCS; run by rule modelled_depth_estimation.

Three fixed-duration runs of one minimal, disposable SFINCS model (regular
grid, production subgrid setup), all with the production weir geometry
(src.protection_weir.build_coastal_protection_weir: coast and river banks in
one water_like boundary). River forcing: a steady discharge at the protection
RP (bankfull as fallback), ramped linearly from each crossing's bankfull over
discharge_ramp_hr: a step to several times bankfull into a near-dry channel
gives a startup shock that can oscillate for the whole run.

Round 0 -- confined, un-excavated, calm sea at calm_sea_level_m. Every edge, coast
too, at weir_crest_m (e.g. 1000 m): the confined level can exceed a low real
coastal crest near the mouth and leak. Method modelled, per centerline cell
(src.river_depth_calibration.compute_excavation_depth):

    rise   = period_max_water_level - DEM_at_cell
    river_depth_m = excavation_fraction * max(rise, 0)

then the mouth and no-dig rules. Method gvf: steady gradually-varied flow
fitted to SWORD water levels at the mean discharge (src.gvf), no SFINCS run.

Round 1 -- confined, excavated, sea boundary a storm tide at the FLOPROS
coastal RP (average tide plus surge shape, peaking on tidal high water once the
river is at full discharge; a half-cosine without hydrograph components). With
no overbank relief this is the highest level reached, including coastal
amplification (shoaling, reflection off the dike), so per edge, no iteration:

    crest(edge) = ceil_cm( max(zsmax(water-side cell),
                                coastal_protection_crest_m) + freeboard_m )

zsmax: SFINCS's maximum over every computational timestep. ceil_cm: rounded
up to the next centimetre (src.surge.ceil_crest), the sfincs.weir precision,
so a crest is never below its zsmax. A never-wetted water side falls back to
coastal_protection_crest_m, so no edge is dropped.

Round 2 -- verification with the production weir. Changes nothing; reports
every edge whose water side exceeds its crest.

period_max_water_level (excavation depth, per-cell diagnostics) is each
cell's own MAXIMUM simulated water level over the entire run
(src.river_depth_calibration.compute_period_max_zs, from the hourly zs
field each round writes to its own sfincs_map.nc), not a
windowed-flatness "converged"/stabilized value -- a persistent, non-damping
oscillation at a discharge-injection cell never satisfies a flatness/trend
tolerance no matter how long the run is extended. Sizing channel depth/
crest only needs the worst-case (peak) forced water level, which the period
maximum gives directly, oscillation or not, with no convergence check,
retry, or restart-extension logic at all.

A centerline cell's water level is the MAX across its channel cross-section:
the shorter of its row and column channel_mask runs, extended one cell each
side to the land cell the weir sits on. Axis-aligned, not reach-normal.

Mouth reaches (n_rch_dn == 0): the last cell gets river_depth_m=0 (bed = natural
bathymetry). Upstream cells shallower than that bed are floored to it, walking
upstream until a cell is already deeper or a reach has zero or several
upstream neighbours. Delta-outline outflow points get a free outflow boundary
(mask=3), not mouth treatment.

Elevation: the coarse dep is elevation_conditioned_sfincs_grid.tif, the same
raster rule 13 uses; the subgrid uses native elevation_conditioned.tif. Rounds
1-2 burn the channel once on the subgrid pixel grid; channel cells' main-grid
dep is its per-cell mean. Rule 13 imports both rasters and the weir as-is.

Outputs: river_network_depth_estimated holds per-reach MEDIAN river_depth_m and
weir_crest_calibrated_m (summary only; production uses the per-edge weir file).
calib_root/round{i}/calibration_state.csv (not a declared output) holds the
per-cell state of each round; weir_crest is NaN in round 0.
"""

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import yaml
from affine import Affine
from hydromt_sfincs import SfincsModel
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

from src.protection_weir import GridArrays, build_coastal_protection_weir, ocean_linked_wetland_mask
from src.domain import load_domain
from src.grid import create_model_grid, load_grid_def
from src.hydromt_sfincs_patch import use_float32_subgrid_write
from src.io import require_keys
from src.landuse import LANDUSE_SEA_CODE
from src.log import setup_logging
from src.plots import (
    animate_flood_progression,
    plot_calibration_round_profiles,
    plot_crest_gap_map,
    plot_max_inundation_map,
    plot_river_depth,
    plot_river_depth_width,
    plot_water_level_timeseries,
    plot_wetland_dike_positions,
)
from src.postprocessing import compute_flood_progression, compute_max_inundation
from src.raster import FLOAT_NODATA, dry_isolated_sea_cells, read_on_grid
from src.river_burn import (
    build_centerline_cells_regular,
    build_channel_mask_regular,
    burn_river_channel,
    constrain_to_coarse_channel_mask,
    mean_elevation_on_coarse_grid,
)
from src.river_depth_calibration import (
    build_calibration_seed_discharge,
    cell_depth_segments,
    compute_excavation_depth,
    compute_end_of_run_zs,
    compute_period_max_zs,
    is_steady,
    steady_state_metrics,
    gather_calibration_round_profile,
)
from src.river_network import (
    accumulate_discharge,
    build_downstream_adjacency,
    normalize_reach_id,
    outflow_crossing_mask,
)
from src.sfincs_run import (
    create_active_mask,
    place_discharge_points,
    run_sfincs_subprocess,
    set_sea_and_outflow_boundaries,
    set_sfincs_inp_keys,
    write_weir_file,
)
from src.surge import ceil_crest, read_calm_sea_level_m, sinusoidal_wave, storm_tide_at_rp_interpolated, storm_tide_event

log = setup_logging(snakemake.log[0])
use_float32_subgrid_write()

# paths & params
elevation_path       = Path(snakemake.input.elevation_conditioned)
elevation_sfincs_grid_path = Path(snakemake.input.elevation_conditioned_sfincs_grid)
river_elevation_max_path = Path(snakemake.input.river_elevation_max)
river_network_path   = Path(snakemake.input.clean_river_network)
river_forcing_path   = Path(snakemake.input.river_forcing)
protection_levels_path = Path(snakemake.input.protection_levels)
# Grid-aligned land mask (rule grid_align_landuse) -- plot background only.
land_mask_path       = Path(snakemake.input.land_mask_on_grid)
domain_gpkg_path     = Path(snakemake.input.domain_gpkg)
# Already on the SFINCS grid (rule 09b); read without reprojection.
landuse_on_grid_path = Path(snakemake.input.landuse_on_grid)
roughness_on_grid_path = Path(snakemake.input.roughness_on_grid)
# Native-resolution roughness (rule 05c): subgrid table only, which needs
# sub-cell detail; the main-grid "manning" uses roughness_on_grid_path.
roughness_path        = Path(snakemake.input.roughness)
surge_forcing_path   = Path(snakemake.input.surge_forcing)
delta_outflow_points_path = Path(snakemake.input.delta_outflow_points)

basin_id    = str(snakemake.wildcards.basin_id)  # plot titles
calib_root  = Path(snakemake.params.calib_root)
sfincs_exe  = Path(snakemake.params.sfincs_exe)
timeout_s   = int(snakemake.params.timeout_s)

depth_method       = str(snakemake.params.depth_method)
gvf_cfg            = dict(snakemake.params.gvf)
channel_manning_n  = float(snakemake.params.channel_manning_n)
if depth_method not in ("modelled", "gvf"):
    raise ValueError(f"river_depth_modelling.method must be modelled or gvf, got {depth_method!r}")
elevation_merged_path = Path(snakemake.input.elevation_merged)
calibration_days   = float(snakemake.params.calibration_days)
gebco_max_depth_m  = float(snakemake.params.gebco_max_depth_m)
round0_level       = str(snakemake.params.round0_level)
steady_cfg         = dict(snakemake.params.steady_state)
require_keys(
    steady_cfg,
    ("median_dzs_3h_m", "p95_dzs_3h_m", "chunk_days", "max_days", "flow_balance_tol", "fail"),
    "river_processing.river_depth_modelling.steady_state",
)
if round0_level not in ("end_of_run", "period_max"):
    raise ValueError(f"river_depth_modelling.round0_level must be end_of_run or period_max, got {round0_level!r}")
# Forcing covers the longest round-0 run (continued until steady), so a
# continuation never runs past the end of its own discharge/water-level series.
forcing_days       = max(calibration_days, float(steady_cfg["max_days"]))
discharge_ramp_hr = float(snakemake.params.discharge_ramp_hr)
weir_crest_m       = float(snakemake.params.weir_crest_m)
weir_par1          = float(snakemake.params.weir_par1)
excavation_fraction = float(snakemake.params.excavation_fraction)
# The only margin on the production weir, on top of round 1's water level.
weir_freeboard_m = float(snakemake.params.weir_freeboard_m)
# True: wetland/lagoon patches linked to the open sea stay seaward of the
# coastal dike in all three rounds.
unprotected_ocean_wetlands = bool(snakemake.params.unprotected_ocean_wetlands)
min_component_cells = int(snakemake.params.min_component_cells)
nr_subgrid_pixels  = int(snakemake.params.nr_subgrid_pixels)
nr_levels          = int(snakemake.params.nr_levels)
nrmax              = int(snakemake.params.nrmax)
animation_fps      = int(snakemake.params.animation_fps)
active_mask_enabled = snakemake.params.active_mask_enabled
active_mask_elevation_buffer_m = float(snakemake.params.active_mask_elevation_buffer_m)
outflow_buffer_m = float(snakemake.params.outflow_buffer_m)
water_level_buffer_m = float(snakemake.params.water_level_buffer_m)

calib_root.mkdir(parents=True, exist_ok=True)

# river network + calibration discharge
rivers = gpd.read_file(river_network_path)

protection_rp_yr = None
with open(protection_levels_path) as f:
    protection = json.load(f)
rp_yr = protection.get("river_protection_rp_yr")
if rp_yr is not None and np.isfinite(float(rp_yr)):
    protection_rp_yr = float(rp_yr)
# FLOPROS coastal RP for rounds 1-2's storm tide; None keeps the calm sea.
surge_protection_rp_yr = protection.get("surge_protection_rp_yr")
surge_protection_rp_yr = float(surge_protection_rp_yr) if surge_protection_rp_yr is not None and np.isfinite(float(surge_protection_rp_yr)) else None

with xr.open_dataset(river_forcing_path, decode_times=False) as river_ds:
    seed_q = build_calibration_seed_discharge(river_ds, protection_rp_yr)

# Rule 08's decisions, reused: seeds that flow OUT of the network inject
# nothing, and canal arms get no share (src.river_network).
if "seed_flows_out" in rivers.columns:
    _outward = {normalize_reach_id(r) for r, f in zip(rivers["reach_id"], rivers["seed_flows_out"]) if bool(f)}
    seed_q = {k: v for k, v in seed_q.items() if normalize_reach_id(k) not in _outward}
_excluded_arms = (
    {normalize_reach_id(r) for r, f in zip(rivers["reach_id"], rivers["zero_discharge_share"]) if bool(f)}
    if "zero_discharge_share" in rivers.columns else set()
)
adjacency = build_downstream_adjacency(rivers)
q_calib = accumulate_discharge(rivers, seed_q, adjacency, excluded_arms=_excluded_arms)
rivers["q_calib"] = q_calib
log.info(
    f"Calibration discharge: {int((q_calib > 0).sum())}/{len(rivers)} reach(es) "
    f"with Q > 0, max Q = {q_calib.max():.1f} m3/s"
)

# Real, steady coastal baseline (mean sea level + SLR/MDT correction) -- same
# value 13_build_sfincs.py/14_run_spinup.py use for their own initial/steady
# conditions -- applied uniformly to every round.
# coastal_protection_crest_m: the production coastal crest, the floor under
# every production weir edge's crest. Rounds 0 and 1 use weir_crest_m instead.
with xr.open_dataset(surge_forcing_path, decode_times=False) as _surge_ds:
    # Rounded to 1 cm, as production's zsini.
    calm_sea_level_m = read_calm_sea_level_m(_surge_ds)
    coastal_protection_crest_m = (
        float(_surge_ds["coastal_protection_crest_m"].values)
        if "coastal_protection_crest_m" in _surge_ds else 0.0
    )
log.info(
    f"Coastal boundary baseline read from surge_forcing.nc: {calm_sea_level_m:+.4f} m, "
    f"real coastal protection crest: {coastal_protection_crest_m:+.4f} m"
)

# minimal SFINCS model (regular grid)
resolution_m = float(snakemake.params.resolution_m)

delta_domain = gpd.read_file(domain_gpkg_path)

# Delta-outline outflow points (rule 08): where flow leaves the network without
# a coastal mouth, e.g. a distributary clipped by the domain. Without the free
# outflow boundary (mask=3, as in rule 13) the reach would be walled off and its
# confined level, and so its crest, would climb.
delta_outflow_gdf = gpd.read_file(delta_outflow_points_path)
delta_outflow_enabled = not delta_outflow_gdf.empty
log.info(f"Delta-outline outflow points: {len(delta_outflow_gdf)}")

# river_burned_subgrid is registered now although it is written only after
# round 0: the catalog needs the URI at model construction.
river_burned_subgrid_path = Path(snakemake.output.river_burned_subgrid)
local_catalog_path = calib_root / "data_catalog_local.yml"
local_catalog = {
    "meta": {"root": str(elevation_path.parent)},
    "local_elevation_conditioned": {
        "data_type": "RasterDataset",
        "uri": str(elevation_path),
        "driver": "rasterio",
    },
    "local_elevation_conditioned_sfincs_grid": {
        "data_type": "RasterDataset",
        "uri": str(elevation_sfincs_grid_path),
        "driver": "rasterio",
    },
    "local_roughness": {
        "data_type": "RasterDataset",
        "uri": str(roughness_on_grid_path),
        "driver": "rasterio",
    },
    # Native resolution, subgrid table only.
    "local_roughness_native": {
        "data_type": "RasterDataset",
        "uri": str(roughness_path),
        "driver": "rasterio",
    },
    **({
        "local_delta_outflow_points": {
            "data_type": "GeoDataFrame",
            "uri": str(delta_outflow_points_path),
            "driver": "pyogrio",
        },
    } if delta_outflow_enabled else {}),
    "local_river_burned_subgrid": {
        "data_type": "RasterDataset",
        "uri": str(river_burned_subgrid_path),
        "driver": "rasterio",
    },
    # Channel Manning n under the burned channel (method gvf), written with
    # the burn below; listed before local_roughness_native in rounds 1-2.
    "local_river_roughness_subgrid": {
        "data_type": "RasterDataset",
        "uri": str(snakemake.output.river_roughness_subgrid),
        "driver": "rasterio",
    },
}
with open(local_catalog_path, "w") as fh:
    yaml.dump(local_catalog, fh, sort_keys=False)

sf = SfincsModel(
    data_libs=[str(local_catalog_path)],
    root=str(calib_root),
    mode="w+",
    write_gis=False,
)
create_model_grid(sf, load_grid_def(snakemake.input.sfincs_grid))
log.info(
    f"Calibration grid created from the shared grid definition: {resolution_m} m, "
    f"rotation {sf.config.get('rotation')} deg"
)

# Time window set before forcing: discharge_points.create() slices to it and
# raises NoDataException while it is unset.
tref = datetime(2000, 1, 1)
tstop = tref + timedelta(days=calibration_days)
trstout_s = int(calibration_days * 86400)
# Forcing is built over forcing_days (round 0 may be continued up to
# steady_state.max_days); each round's own run window is set in
# _run_calibration_round.
forcing_tstop = tref + timedelta(days=forcing_days)
sf.config.update({
    "tref": tref, "tstart": tref, "tstop": forcing_tstop,
    "dthisout": 3600, "dtmapout": 3600,
    "dtmaxout": trstout_s,
    "trstout": trstout_s, "dtrstout": 0,
    "zsini": -9999.0,
})

# round-0 elevation (un-excavated), mask, roughness
# Same coarse background raster as rule 13.
sf.elevation.create(elevation_list=[{"elevation": "local_elevation_conditioned_sfincs_grid"}])
# Elevation ceiling on the active mask, as in rule 13.
river_elevation_max_m = None
if active_mask_enabled:
    with open(river_elevation_max_path) as f:
        river_elevation_max_m = float(json.load(f)["river_elevation_max_m"])
create_active_mask(sf, delta_domain, river_elevation_max_m, active_mask_elevation_buffer_m)

# Water-level boundary on open sea only (landuse_on_grid == 200), as rule 13.
with rasterio.open(landuse_on_grid_path) as _lu_src_bnd:
    _landuse_bnd = _lu_src_bnd.read(1)
set_sea_and_outflow_boundaries(
    sf, _landuse_bnd, len(delta_outflow_gdf) if delta_outflow_enabled else 0, outflow_buffer_m
)

# spatially-varying zsini: sea cells start AT calm_sea_level_m, not bone-dry
# Compared against sf.grid.data["dep"], not elevation_conditioned_sfincs_grid.tif,
# which extends beyond the grid bounds. A coastal cell starting dry fills with
# a transient spike above its settled level, which the run maximum would add
# to its crest.
_lu_on_grid_zsini = read_on_grid(landuse_on_grid_path, sf.grid.data["dep"].shape)
_zsini_arr = np.where(_lu_on_grid_zsini == LANDUSE_SEA_CODE, np.float32(calm_sea_level_m), np.nan)

# Isolated sea cells (not the main ocean component) below calm_sea_level_m start
# dry at dep, as in rule 13.
_zsini_arr, _n_fix_zsini = dry_isolated_sea_cells(
    _zsini_arr,
    sf.grid.data["dep"].values,
    sf.grid.data["dep"].rio.nodata or FLOAT_NODATA,
    calm_sea_level_m,
)
if _n_fix_zsini > 0:
    log.info(f"zsini connectivity fix: {_n_fix_zsini} isolated wet sea-cell(s) set to dep")

# Nearest: zsini is near-binary and the grids are identical. The mask must
# exist first.
_da_zsini = xr.DataArray(
    _zsini_arr, dims=sf.grid.data["dep"].dims, coords=sf.grid.data["dep"].coords,
)
_da_zsini = _da_zsini.rio.write_crs(sf.grid.data["dep"].rio.crs)
_da_zsini = _da_zsini.rio.write_transform(sf.grid.data["dep"].raster.transform)
_da_zsini.raster.set_nodata(np.nan)
sf.initial_conditions.create(ini=_da_zsini, reproj_method="nearest")
log.info("Initial conditions: spatially-varying zsini (sea cells start at calm_sea_level_m, land dry)")

sf.roughness.create(roughness_list=[{"manning": "local_roughness"}])

# shared, round-invariant pieces: grid arrays, channel_mask, landuse
# Coast and channel in one water_like boundary so the mouth is never sealed.
# GridArrays copies dep/mask at construction, so it stays valid after the
# excavation. channel_mask (weir corridor and excavation limit) matches what
# rules 11b/13 compute.
rivers_utm = rivers.to_crs(sf.crs)
grid = GridArrays.from_regular(sf.grid.data["dep"], sf.grid.data["mask"], sf.crs)
channel_mask = build_channel_mask_regular(rivers_utm, "width", grid.shape, grid.transform)
landuse_on_grid = read_on_grid(landuse_on_grid_path, grid.shape)
ocean_mask_grid = landuse_on_grid == LANDUSE_SEA_CODE

# Written to file: compute_max_inundation/compute_flood_progression take a path.
# Not weir-aware (no protected_pocket_mask before round 1).
_round_ocean_mask_path = calib_root / "ocean_mask_on_grid.tif"
with rasterio.open(
    _round_ocean_mask_path, "w", driver="GTiff", dtype="float32", count=1,
    height=ocean_mask_grid.shape[0], width=ocean_mask_grid.shape[1],
    crs=grid.crs, transform=grid.transform, nodata=FLOAT_NODATA,
) as _dst:
    _dst.write(np.where(ocean_mask_grid, np.float32(1.0), np.float32(FLOAT_NODATA)), 1)

# Centerline calibration cells: every grid cell a reach's centerline crosses
# Per cell from sfincs_map.nc, not observation points: a sparse point can miss
# a sharp spike at the discharge-injection cell.
cell_gdf = build_centerline_cells_regular(rivers_utm, grid.shape, grid.transform)
if cell_gdf.empty:
    raise RuntimeError("No centerline calibration cells could be identified -- empty river network?")
dem_at_cell = grid.bed_elevation_m[cell_gdf["row"].to_numpy(), cell_gdf["col"].to_numpy()]
log.info(
    f"Centerline calibration cells: {len(cell_gdf)} across "
    f"{cell_gdf['reach_id'].nunique()} reach(es)"
)

# Per-cell cross-section zs sampling
# A cell's zs is the MAX over its channel cross-section: the shorter of the
# channel_mask runs along its row and column (the shorter is across the
# channel). Not their union: for a reach along one grid axis the other run is
# the whole reach, giving every cell the same global maximum. Each run extends
# one cell past channel_mask to the land cell the weir sits on. Axis-aligned,
# so inexact where the channel runs near-diagonally.
def _channel_run_through(mask_1d: np.ndarray, idx: int) -> np.ndarray:
    # idx is kept even if mask_1d[idx] is False.
    if not mask_1d[idx]:
        left = right = idx
    else:
        left = idx
        while left > 0 and mask_1d[left - 1]:
            left -= 1
        right = idx
        while right < len(mask_1d) - 1 and mask_1d[right + 1]:
            right += 1
    left = max(0, left - 1)
    right = min(len(mask_1d) - 1, right + 1)
    return np.arange(left, right + 1)


_cell_rows = cell_gdf["row"].to_numpy()
_cell_cols = cell_gdf["col"].to_numpy()
cell_cross_section_rowcol: list[np.ndarray] = []  # one (n_i, 2) row/col array per cell_gdf row
for _r, _c in zip(_cell_rows, _cell_cols):
    _row_run_cols = _channel_run_through(channel_mask[_r, :], _c)
    _col_run_rows = _channel_run_through(channel_mask[:, _c], _r)
    if len(_row_run_cols) <= len(_col_run_rows):
        _rc = np.column_stack([np.full(len(_row_run_cols), _r), _row_run_cols])
    else:
        _rc = np.column_stack([_col_run_rows, np.full(len(_col_run_rows), _c)])
    cell_cross_section_rowcol.append(_rc)

# Deduplicated: adjacent centerline cells share cross-section cells.
_all_cross_section_rc = np.concatenate(cell_cross_section_rowcol, axis=0)
_unique_rc, _cross_section_inverse = np.unique(_all_cross_section_rc, axis=0, return_inverse=True)
cross_section_unique_rows = _unique_rc[:, 0]
cross_section_unique_cols = _unique_rc[:, 1]
cross_section_unique_x, cross_section_unique_y = rasterio.transform.xy(
    grid.transform, cross_section_unique_rows, cross_section_unique_cols
)
cross_section_unique_x = np.asarray(cross_section_unique_x)
cross_section_unique_y = np.asarray(cross_section_unique_y)
# cross_section_group_idx[i]: indices into the unique cells for cell i.
_group_sizes = [len(a) for a in cell_cross_section_rowcol]
_group_bounds = np.cumsum([0] + _group_sizes)
cross_section_group_idx = [
    _cross_section_inverse[_group_bounds[i]:_group_bounds[i + 1]] for i in range(len(cell_gdf))
]
log.info(
    f"Cross-section zs sampling: {len(cell_gdf)} centerline cell(s) map onto "
    f"{len(cross_section_unique_rows)} unique grid cell(s) total "
    f"(median cross-section width: {int(np.median(_group_sizes))} cell(s))"
)

# sfincs_map.nc (n, m) indices, resolved lazily once; the grid is round-invariant.
map_cell_idx = None
cross_section_map_cell_idx = None
field_map_cell_idx = None

# One cell per reach (nearest its along_m midpoint) for the timeseries plot.
# Positional (cell_gdf has a RangeIndex), so also column indices into final_zs.
_reach_along_mid = cell_gdf.groupby("reach_id")["along_m"].transform(lambda s: s.max() / 2.0)
_dist_to_mid = (cell_gdf["along_m"] - _reach_along_mid).abs()
representative_cell_pos = _dist_to_mid.groupby(cell_gdf["reach_id"]).idxmin().to_numpy()
representative_reach_ids = cell_gdf.loc[representative_cell_pos, "reach_id"].to_numpy()

# discharge points (constant calibration discharge per crossing)
n_steps = max(int(forcing_days * 24), 2)
calib_times = pd.date_range(tref, forcing_tstop, periods=n_steps)

reach_q = dict(zip(rivers["reach_id"].astype(str), q_calib))
with xr.open_dataset(river_forcing_path, decode_times=False) as river_ds:
    # Crossings on a seed that flows out of the network inject nothing.
    active = river_ds["has_glofas"].values.astype(bool) & ~outflow_crossing_mask(river_ds, rivers)
    inside_ids = river_ds["inside_reach_id"].values[active] if "inside_reach_id" in river_ds else []
    cross_lons = river_ds["longitude"].values[active]
    cross_lats = river_ds["latitude"].values[active]
    # Ramp start value.
    cross_bankfull_q = river_ds["bankfull_discharge_m3s"].values[active]

crossing_q = [reach_q.get(str(reach_id), np.nan) for reach_id in inside_ids]
crossings_gdf = gpd.GeoDataFrame(
    {"index": range(len(crossing_q))},
    geometry=gpd.points_from_xy(cross_lons, cross_lats),
    crs="EPSG:4326",
)
valid = np.isfinite(crossing_q)
crossings_gdf = crossings_gdf[valid].reset_index(drop=True)
crossing_q = np.asarray(crossing_q)[valid]
crossing_bankfull_q = np.asarray(cross_bankfull_q)[valid]
inside_ids_valid = np.asarray(inside_ids)[valid] if len(inside_ids) else [None] * len(crossing_q)

if crossings_gdf.empty:
    raise RuntimeError("No discharge crossings resolved to a calibration discharge -- cannot calibrate")

# Linear ramp from bankfull to the calibration discharge, then flat: a step
# into a near-dry channel oscillates for the whole run (basin 2433835:
# multi-metre hourly swings at the seed cell). NaN bankfull ramps from 0.
_ramp_time_hr = np.arange(n_steps) * (forcing_days * 24.0) / (n_steps - 1)
_ramp_frac = np.clip(_ramp_time_hr / max(discharge_ramp_hr, 1e-6), 0.0, 1.0)[:, None]
_bankfull_start = np.where(np.isfinite(crossing_bankfull_q), crossing_bankfull_q, 0.0)
dis_values = _bankfull_start[None, :] + (crossing_q[None, :] - _bankfull_start[None, :]) * _ramp_frac
dis_df = pd.DataFrame(dis_values, index=calib_times, columns=crossings_gdf["index"].to_numpy())

# Snap to the crossing reach's centerline cell: an unsnapped point can land in
# a floodplain cell without conveyance (basin 4267691: 369 m water level in
# round 0). Own reach first, else nearest cell of any reach.
crossings_filt = place_discharge_points(sf, crossings_gdf, dis_df, cell_gdf, inside_ids_valid, resolution_m)
dis_df = dis_df[crossings_filt["index"].to_numpy()]
log.info(
    f"Discharge forcing: {len(crossings_filt)} source point(s), ramped from bankfull to the full "
    f"calibration discharge over {discharge_ramp_hr:.1f} h, held constant for the remaining "
    f"{max(calibration_days * 24.0 - discharge_ramp_hr, 0.0):.1f} h"
)

# For round 2's crest gap map.
crossings_utm = crossings_filt.to_crs(grid.crs)

# coastal water-level boundary (steady calm_sea_level_m)
sf.water_level.create_boundary_points_from_mask(bnd_dist=5000.0)
sf.water_level.create_timeseries(shape="constant", offset=calm_sea_level_m)
log.info(f"Water-level forcing: flat constant boundary at {calm_sea_level_m:+.4f} m (calm_sea_level_m)")


def _resolve_map_cell_idx(map_nc_path: Path, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Matched on x/y coordinates: SFINCS's (n, m) order relative to
    # grid.transform is not guaranteed.
    with xr.open_dataset(map_nc_path) as ds:
        grid_x = ds["x"].values.ravel()
        grid_y = ds["y"].values.ravel()
        n_dim, m_dim = ds["x"].shape
    valid = np.isfinite(grid_x) & np.isfinite(grid_y)
    tree = cKDTree(np.column_stack([grid_x[valid], grid_y[valid]]))
    _dist, flat_idx = tree.query(np.column_stack([x, y]))
    orig_flat_idx = np.where(valid)[0][flat_idx]
    n_idx, m_idx = np.unravel_index(orig_flat_idx, (n_dim, m_dim))
    return n_idx, m_idx


def _read_map_zs_at_cells(run_dir: Path, n_idx: np.ndarray, m_idx: np.ndarray):
    # `with` releases the handle; otherwise the next round's write fails on
    # Windows ("Permission denied"). dims="cell" pairs indices instead of the
    # outer product, so the full field is never loaded.
    with xr.open_dataset(run_dir / "sfincs_map.nc") as ds:
        zs = ds["zs"].isel(
            n=xr.DataArray(n_idx, dims="cell"), m=xr.DataArray(m_idx, dims="cell"),
        ).values
        times_s = (ds["time"].values - ds["time"].values[0]) / np.timedelta64(1, "s")
    return zs, times_s


def _read_zs_and_resolve(round_root: Path):
    global map_cell_idx
    if map_cell_idx is None:
        map_cell_idx = _resolve_map_cell_idx(
            round_root / "sfincs_map.nc", cell_gdf["x"].to_numpy(), cell_gdf["y"].to_numpy(),
        )
        log.info(f"Resolved {len(map_cell_idx[0])} centerline cell(s) against sfincs_map.nc's own grid indices")
    return _read_map_zs_at_cells(round_root, *map_cell_idx)


def _read_cross_section_period_max_zs(round_root: Path) -> np.ndarray:
    global cross_section_map_cell_idx
    if cross_section_map_cell_idx is None:
        cross_section_map_cell_idx = _resolve_map_cell_idx(
            round_root / "sfincs_map.nc", cross_section_unique_x, cross_section_unique_y,
        )
        log.info(
            f"Resolved {len(cross_section_map_cell_idx[0])} unique cross-section grid cell(s) "
            f"against sfincs_map.nc's own grid indices"
        )
    zs, _times_s = _read_map_zs_at_cells(round_root, *cross_section_map_cell_idx)
    per_unique_cell_period_max = compute_period_max_zs(zs)
    return np.array(
        [np.nanmax(per_unique_cell_period_max[idx]) for idx in cross_section_group_idx],
        dtype=np.float32,
    )


def _read_cross_section_end_zs(round_root: Path) -> np.ndarray:
    # Mean of the last two outputs per cell, then max across the cross-section.
    global cross_section_map_cell_idx
    if cross_section_map_cell_idx is None:
        cross_section_map_cell_idx = _resolve_map_cell_idx(
            round_root / "sfincs_map.nc", cross_section_unique_x, cross_section_unique_y,
        )
    zs, _times_s = _read_map_zs_at_cells(round_root, *cross_section_map_cell_idx)
    per_unique_cell_end = compute_end_of_run_zs(zs)
    return np.array(
        [np.nanmax(per_unique_cell_end[idx]) for idx in cross_section_group_idx],
        dtype=np.float32,
    )


def _read_exit_discharge(round_root: Path) -> np.ndarray | None:
    # m3/s, absolute, mean of the last two outputs; None without cross-section output.
    his = round_root / "sfincs_his.nc"
    if not his.exists():
        return None
    with xr.open_dataset(his, decode_times=False) as ds:
        names = [v for v in ds.data_vars if "crosssection" in v.lower() and "discharge" in v.lower()]
        if not names:
            return None
        q = np.asarray(ds[names[0]].values, dtype=float)
    if q.ndim == 1:
        q = q[:, None]
    return np.abs(np.nanmean(q[-2:], axis=0))


def _continue_until_steady(round_label: str, round_root: Path) -> dict:
    # Restart in steady_state.chunk_days chunks until steady (median/p95 |dzs|
    # over the last 3 h) or steady_state.max_days. sfincs_map.nc/sfincs_his.nc
    # then hold only the LAST chunk.
    run_end = tstop
    chunk = timedelta(days=float(steady_cfg["chunk_days"]))
    max_end = tref + timedelta(days=float(steady_cfg["max_days"]))
    while True:
        zs, _ = _read_zs_and_resolve(round_root)
        metrics = steady_state_metrics(zs, window_steps=3)
        steady = is_steady(metrics, float(steady_cfg["median_dzs_3h_m"]), float(steady_cfg["p95_dzs_3h_m"]))
        days = (run_end - tref).total_seconds() / 86400.0
        log.info(
            f"[{round_label}] after {days:.1f} day(s): centreline |dzs| over the last 3 h median "
            f"{metrics['median']:.4f} m, p95 {metrics['p95']:.4f} m, {100 * metrics['frac_rising_1cm']:.0f}% "
            f"of cells still rising > 1 cm -- {'steady' if steady else 'not steady'}"
        )
        if steady or run_end + chunk > max_end + timedelta(seconds=1):
            break
        rst = sorted(round_root.glob("sfincs.*.rst"), key=lambda p: p.stat().st_mtime)
        if not rst:
            raise RuntimeError(
                f"[{round_label}] not steady after {days:.1f} day(s) and no sfincs.*.rst restart "
                f"file in {round_root} -- cannot continue the run to steady state"
            )
        new_end = run_end + chunk
        set_sfincs_inp_keys(round_root / "sfincs.inp", {
            "rstfile": rst[-1].name, "inifile": None, "zsini": None,
            "tstart": run_end.strftime("%Y%m%d %H%M%S"), "tstop": new_end.strftime("%Y%m%d %H%M%S"),
            "trstout": int((new_end - tref).total_seconds()),
            "dtmaxout": int(chunk.total_seconds()),
        })
        log.info(f"[{round_label}] continuing from {rst[-1].name} to day {(new_end - tref).total_seconds() / 86400.0:.1f}")
        run_sfincs_subprocess(
            sfincs_exe, round_root, timeout_s, log,
            label=f"SFINCS calibration ({round_label}, continuation)", n_threads=snakemake.threads,
        )
        run_end = new_end
    metrics["days"] = (run_end - tref).total_seconds() / 86400.0
    metrics["steady"] = steady
    return metrics


def _read_period_max_zs_field(round_root: Path) -> np.ndarray:
    # zsmax on grid.shape (NaN never wetted/inactive): the maximum over every
    # computational timestep (dtmaxout = whole run), not the hourly output, since
    # a weir overtops between outputs (basin 2433835: hourly max up to ~0.1 m
    # below zsmax at 1% of wet cells).
    global field_map_cell_idx
    map_nc_path = round_root / "sfincs_map.nc"
    if field_map_cell_idx is None:
        _rows, _cols = np.indices(grid.shape)
        _xs, _ys = grid.transform * (_cols.ravel() + 0.5, _rows.ravel() + 0.5)
        field_map_cell_idx = _resolve_map_cell_idx(map_nc_path, np.asarray(_xs), np.asarray(_ys))
        log.info(f"Resolved {grid.shape[0] * grid.shape[1]:,} grid cell(s) against sfincs_map.nc's own grid indices")
    with xr.open_dataset(map_nc_path) as ds:
        zsmax = ds["zsmax"]
        zsmax = zsmax.max(dim=[d for d in zsmax.dims if d not in ("n", "m")]).values
    n_idx, m_idx = field_map_cell_idx
    return zsmax[n_idx, m_idx].reshape(grid.shape).astype(np.float32)


def _run_calibration_round(round_label: str, round_root: Path, until_steady: bool = False):
    """Write and run whatever elevation/weir the caller just set up on the
    (shared) `sf` object, into round_root (the caller has already pointed
    sf.root at this via sf.root.set()) -- a single fixed-duration run, no
    restart/extension. Each round starts cold (no restart chained across
    rounds -- the geometry changed between rounds, so resuming a previous
    round's dynamic state would be physically invalid).

    No convergence/stabilization check: the per-cell water level used for
    calibration is simply each cell's own maximum over the full run
    (compute_period_max_zs), not a windowed-flatness "stabilized" value --
    see that function's own docstring for why (a persistent, non-damping
    oscillation at a discharge-injection cell never satisfies a flatness
    tolerance no matter how long the run is extended: the peak is
    exactly what calibration needs, converged or not).

    until_steady (round 0): continue the run from its restart file until the
    water level is steady -- see _continue_until_steady. The returned
    period_max_zs is then the maximum over the LAST chunk."""
    sf.config.set("rstfile", None)
    sf.config.update({
        "tstart": tref, "tstop": tstop, "trstout": trstout_s, "dtrstout": 0,
        "dtmaxout": trstout_s, "zsini": -9999.0,
    })
    sf.write()
    write_weir_file(sf)  # crests at 1 cm, not hydromt_sfincs' 0.1 m
    log.info(f"Running SFINCS calibration ({round_label}, {calibration_days:.0f} days)")
    run_sfincs_subprocess(
        sfincs_exe, round_root, timeout_s, log,
        label=f"SFINCS calibration ({round_label})", n_threads=snakemake.threads,
    )
    if until_steady:
        global round0_steady
        round0_steady = _continue_until_steady(round_label, round_root)

    # zs (single intersected cell) is for the timeseries plot only.
    zs, times_s = _read_zs_and_resolve(round_root)
    period_max_zs = _read_cross_section_period_max_zs(round_root)
    log.info(
        f"[{round_label}] Period max water level (cross-section): min={period_max_zs.min():.3f} m, "
        f"max={period_max_zs.max():.3f} m, median={np.median(period_max_zs):.3f} m "
        f"(over the run's last {zs.shape[0]} output step(s))"
    )
    return period_max_zs, zs, times_s


wgs84_bounds, domain_crs, domain_poly = load_domain(snakemake.input.spec_basins_meta, domain_gpkg_path)

# Not declared outputs. Round number as suffix so listings group by plot kind.
round_visuals_dir = Path(snakemake.output.plot_calibration).parent / "rounds"
round_visuals_dir.mkdir(parents=True, exist_ok=True)

N_ROUNDS_AFTER_0 = 2
ROUND_TITLES = [
    "round 0 (confined, un-excavated)",
    "round 1 (confined, excavated)",
    "round 2 (verification, real crests)",
]


def _round_visual_paths(round_idx: int) -> dict[str, Path]:
    return {
        "plot_calibration": round_visuals_dir / f"river_depth_round{round_idx}.png",
        "plot_water_level": round_visuals_dir / f"water_level_timeseries_round{round_idx}.png",
        "plot_max_inundation": round_visuals_dir / f"max_inundation_round{round_idx}.png",
        "animation": round_visuals_dir / f"flood_animation_round{round_idx}.mp4",
        "crest_gap_map": round_visuals_dir / f"crest_gap_map_round{round_idx}.png",
    }


def _start_round(round_idx: int) -> Path:
    # One directory per round: reusing filenames fails SFINCS's netCDF creation
    # on Windows ("Permission denied"/"NetCDF: Not a valid ID") once a previous
    # round's diagnostics read them.
    round_root = calib_root / f"round{round_idx}"
    round_root.mkdir(parents=True, exist_ok=True)
    sf.root.set(round_root)
    return round_root


def _create_subgrid(elevation_list: list[dict], round_label: str, channel_roughness: bool = False) -> None:
    # Rebuilt every round: each round's diagnostics read its own dep_subgrid.tif.
    # channel_roughness: channel Manning n first, so it wins under the channel.
    roughness_list = [{"manning": "local_roughness_native"}]
    if channel_roughness:
        roughness_list = [{"manning": "local_river_roughness_subgrid"}] + roughness_list
    sf.subgrid.create(
        elevation_list=elevation_list,
        roughness_list=roughness_list,
        river_list=[],
        nr_subgrid_pixels=nr_subgrid_pixels,
        nr_levels=nr_levels,
        write_dep_tif=True,
        write_man_tif=True,
        nrmax=nrmax,
    )
    log.info(f"[{round_label}] Subgrid table created: {nr_subgrid_pixels} px/cell")


def _set_weir(weir_gdf: gpd.GeoDataFrame, weir_diagnostics: dict, round_label: str, description: str) -> None:
    if weir_diagnostics.get("applicable", True) and not weir_gdf.empty:
        sf.weirs.set(weir_gdf, merge=False)
        log.info(f"[{round_label}] Weir set: {len(weir_gdf)} segment(s), {description}")
    else:
        log.warning(f"[{round_label}] Coastal protection weir: not applicable -- this round is not confined")


def _update_reach_columns(river_depth_m: np.ndarray, weir_crest: np.ndarray | None = None) -> None:
    # Median, so a mouth reach's forced river_depth_m=0 last cell does not skew it.
    cell_state = cell_gdf[["reach_id"]].assign(river_depth_m=river_depth_m)
    if weir_crest is not None:
        cell_state = cell_state.assign(crest=weir_crest)
    reach_medians = cell_state.groupby("reach_id").median()
    rivers["river_depth_m"] = rivers["reach_id"].astype(str).map(reach_medians["river_depth_m"])
    if weir_crest is not None:
        rivers["weir_crest_calibrated_m"] = rivers["reach_id"].astype(str).map(reach_medians["crest"])
        rivers_utm["weir_crest_calibrated_m"] = rivers_utm["reach_id"].astype(str).map(reach_medians["crest"])


def _write_calibration_state(round_root: Path, river_depth_m: np.ndarray, weir_crest: np.ndarray, zs: np.ndarray) -> None:
    cell_gdf.drop(columns="geometry").assign(
        dem=dem_at_cell, river_depth_m=river_depth_m, weir_crest=weir_crest, zs=zs,
    ).to_csv(round_root / "calibration_state.csv", index=False)


def _round_diagnostics(
    round_idx: int, round_root: Path, final_zs: np.ndarray, final_times_s: np.ndarray,
    weir_gdf: gpd.GeoDataFrame, crest_gap_crest_on_grid: np.ndarray | None = None,
) -> None:
    # Run before the next round's sf.write(). crest_gap_crest_on_grid: round 2
    # only; meaningless for the 1000 m confinement rounds.
    round_label = ROUND_TITLES[round_idx]
    paths = _round_visual_paths(round_idx)

    rivers_wgs = rivers.to_crs("EPSG:4326") if rivers.crs is not None and rivers.crs.to_epsg() != 4326 else rivers
    plot_river_depth(
        rivers_wgs=rivers_wgs,
        bbox_poly=domain_poly,
        land_polygons_path=str(land_mask_path),
        output_path=str(paths["plot_calibration"]),
    )

    plot_water_level_timeseries(
        final_times_s / 86400.0, final_zs[:, representative_cell_pos], str(paths["plot_water_level"]),
        day_markers=[(calibration_days, f"Day {calibration_days:.0f} (run end)")],
        station_labels=[f"reach {reach_id}" for reach_id in representative_reach_ids],
        basin_id=f"Basin {basin_id}", run_label=f"Depth calibration -- {round_label}",
    )

    da_hmax, _da_dep = compute_max_inundation(
        round_root, round_root, _round_ocean_mask_path, hmin=0.0,
    )
    if da_hmax is None:
        raise RuntimeError(f"[{round_label}] no zsmax in the SFINCS output of {round_root}")
    plot_max_inundation_map(
        da_hmax, domain_poly, str(land_mask_path), str(river_network_path),
        str(paths["plot_max_inundation"]),
        basin_id=basin_id, run_label=f"calibration -- {round_label}",
    )

    da_h = compute_flood_progression(round_root, _round_ocean_mask_path)
    if da_h is None:
        raise RuntimeError(f"[{round_label}] no zs or zb in the SFINCS output of {round_root}")
    animate_flood_progression(
        da_h, domain_poly, str(land_mask_path), str(river_network_path),
        str(paths["animation"]),
        basin_id=basin_id, run_label=f"calibration -- {round_label}", fps=animation_fps,
    )

    if round_idx == N_ROUNDS_AFTER_0:
        # None when the production weir is not applicable (no ocean cells on
        # the grid); all-NaN when it has no seaward edge. Both leave no crest to map.
        if crest_gap_crest_on_grid is None or not np.isfinite(crest_gap_crest_on_grid).any():
            log.warning(f"[{round_label}] no weir crest cells -- writing empty crest gap map sentinel")
            paths["crest_gap_map"].touch()
        else:
            plot_crest_gap_map(
                crest_gap_crest_on_grid, grid.transform, str(round_root / "sfincs_map.nc"),
                weir_gdf, rivers_utm, crossings_utm.geometry.x.to_numpy(), crossings_utm.geometry.y.to_numpy(),
                str(paths["crest_gap_map"]),
                basin_id=basin_id, run_label=f"calibration -- {round_label}",
            )
    log.info(f"[{round_label}] Diagnostic plots written under {round_visuals_dir}")


# Confinement weir (rounds 0 and 1): all-NaN water_side_crest_on_grid puts
# every edge at weir_crest_m. channel_mask_gap_free=True: the mask is
# all_touched, so close_gaps() is skipped and the weir hugs the corridor; the
# production weir uses the same flag, so all rounds share one geometry.
confinement_weir_gdf, confinement_weir_diagnostics = build_coastal_protection_weir(
    grid, landuse_on_grid, channel_mask, weir_crest_m, min_component_cells, weir_par1,
    water_side_crest_on_grid=np.full(grid.shape, np.nan, dtype=np.float32),
    channel_mask_gap_free=True, unprotected_ocean_wetlands=unprotected_ocean_wetlands,
)

# dike position with unprotected_ocean_wetlands off vs on (figure only)
# The other setting is traced for the figure only, never simulated or exported.
log.info(
    f"Tracing the weir with unprotected_ocean_wetlands={not unprotected_ocean_wetlands} "
    f"(comparison figure only, not simulated)"
)
_alt_weir_gdf, _ = build_coastal_protection_weir(
    grid, landuse_on_grid, channel_mask, weir_crest_m, min_component_cells, weir_par1,
    water_side_crest_on_grid=np.full(grid.shape, np.nan, dtype=np.float32),
    channel_mask_gap_free=True, unprotected_ocean_wetlands=not unprotected_ocean_wetlands,
)
_weir_off, _weir_on = (
    (_alt_weir_gdf, confinement_weir_gdf) if unprotected_ocean_wetlands else (confinement_weir_gdf, _alt_weir_gdf)
)
plot_wetland_dike_positions(
    landuse_on_grid, grid.transform, _weir_off, _weir_on,
    ocean_linked_wetland_mask(landuse_on_grid, ocean_mask_grid, grid),
    str(snakemake.output.plot_dike_positions), active_on=unprotected_ocean_wetlands, basin_id=basin_id,
)

# round 0: confined, un-excavated
round_label = ROUND_TITLES[0]
round_root = _start_round(0)
_set_weir(
    confinement_weir_gdf, confinement_weir_diagnostics, round_label,
    f"uniform confinement crest={weir_crest_m:.0f} m",
)
# Either method leaves the confinement weir on sf for round 1.
if depth_method == "gvf":
    from pyproj import Transformer

    from src.gvf import geoid_offset_at, mean_seed_discharge, run_gvf, slot_levels, values_at_points
    from src.gvf_solver import slot_diagnostics

    # Mean discharge, routed through the network exactly as q_calib is.
    with xr.open_dataset(river_forcing_path, decode_times=False) as _rds:
        _active = _rds["has_glofas"].values.astype(bool) & ~outflow_crossing_mask(_rds, rivers)
        _seed_mean = mean_seed_discharge(_rds, _active)
    q_gvf = accumulate_discharge(rivers, _seed_mean, adjacency, excluded_arms=_excluded_arms)
    rivers["q_gvf"] = q_gvf
    log.info(
        f"[{round_label}] GVF discharge (GloFAS mean): seeds "
        f"{ {k: round(v) for k, v in _seed_mean.items()} }, {int((q_gvf > 0).sum())}/{len(rivers)} reach(es) with Q > 0"
    )

    # SWORD wse (EGM2008) -> model datum (GOCO06s) at each reach midpoint.
    _mid = rivers_utm.geometry.interpolate(0.5, normalized=True).to_crs("EPSG:4326")
    _offset = geoid_offset_at(snakemake.input.goco06s_gfc, snakemake.input.egm2008_gfc, _mid.x.values, _mid.y.values)
    log.info(f"[{round_label}] geoid offset N_EGM2008 - N_GOCO06s at the reaches: {_offset.min():+.3f}..{_offset.max():+.3f} m")

    def _sampler(path):
        src = rasterio.open(path)
        to_src = Transformer.from_crs(sf.crs, src.crs, always_xy=True)
        return src, to_src

    def _sample_dem(x, y):
        src, to_src = _sampler(elevation_merged_path)
        with src:
            xs, ys = to_src.transform(np.asarray(x), np.asarray(y))
            vals = np.array([v[0] for v in src.sample(zip(xs, ys))], dtype=float)
            if src.nodata is not None:
                vals[vals == src.nodata] = np.nan
        return vals

    def _sample_sea_fraction(x, y, width_m):
        # Share of sea class (200) over a disc of the channel width (21 points).
        offs = np.linspace(-0.5, 0.5, 5)
        disc = [(a, b) for a in offs for b in offs if a * a + b * b <= 0.25]
        src, to_src = _sampler(snakemake.input.landuse)
        out = np.zeros(len(x))
        with src:
            for k, (xk, yk, wk) in enumerate(zip(x, y, width_m)):
                px, py = to_src.transform([xk + a * wk for a, _ in disc], [yk + b * wk for _, b in disc])
                out[k] = np.mean(np.array([v[0] for v in src.sample(zip(px, py))]) == LANDUSE_SEA_CODE)
        return out

    _outflow_ids = (
        {normalize_reach_id(r) for r, f in zip(rivers["reach_id"], rivers["is_delta_outflow"]) if bool(f)}
        if "is_delta_outflow" in rivers.columns else set()
    )
    gvf_run = run_gvf(
        rivers_utm,
        dict(zip(rivers["reach_id"].map(normalize_reach_id), np.asarray(q_gvf, dtype=float))),
        _offset,
        _sample_sea_fraction,
        _sample_dem,
        sea_level_m=calm_sea_level_m,
        gebco_floor_m=calm_sea_level_m - gebco_max_depth_m,
        outflow_reaches=_outflow_ids,
        manning_n=channel_manning_n,
        cfg=gvf_cfg,
    )

    # Bed and level at the centreline cells, by along-reach distance.
    _cx, _cy = cell_gdf["x"].to_numpy(), cell_gdf["y"].to_numpy()
    rivbed_cell = values_at_points(gvf_run, gvf_run.result.z, cell_gdf["reach_id"], _cx, _cy)
    round0_zs = values_at_points(gvf_run, slot_levels(gvf_run), cell_gdf["reach_id"], _cx, _cy)
    _missing = ~np.isfinite(rivbed_cell)
    if _missing.any():
        log.warning(f"[{round_label}] {int(_missing.sum())} centreline cell(s) without a GVF bed -- left at the DEM")
        rivbed_cell = np.where(_missing, dem_at_cell, rivbed_cell)
    rivdph_current = np.maximum(dem_at_cell - rivbed_cell, 0.0)
    log.info(
        f"[{round_label}] GVF bed at {len(cell_gdf):,} centreline cell(s): depth below the conditioned DEM "
        f"median {np.nanmedian(rivdph_current):.2f} m, max {np.nanmax(rivdph_current):.2f} m"
    )

    # Diagnostics next to the canonical 10_calibration visuals.
    _gvf_dir = Path(snakemake.output.plot_calibration).parent / "gvf"
    _gvf_dir.mkdir(parents=True, exist_ok=True)
    _gvf_table = slot_diagnostics(gvf_run.problem, gvf_run.result)
    _gvf_table.to_csv(_gvf_dir / f"{basin_id}_river_gvf_nodes.csv", index=False)
    gpd.GeoDataFrame(
        _gvf_table, geometry=gpd.points_from_xy(_gvf_table["x"], _gvf_table["y"]), crs=sf.crs
    ).to_file(_gvf_dir / f"{basin_id}_river_gvf_nodes.gpkg", driver="GPKG")
    _per_reach = _gvf_table.groupby("reach_id").agg(
        gvf_eta_target=("eta_target", "median"), gvf_depth=("h", "median"), gvf_class=("node_class", "first"),
    )
    _per_reach["gvf_rmse"] = pd.Series(gvf_run.result.reach_rmse)
    for _col in _per_reach.columns:
        rivers[_col] = rivers["reach_id"].map(normalize_reach_id).map(_per_reach[_col])
    log.info(f"[{round_label}] GVF diagnostics written to {_gvf_dir}")

    _update_reach_columns(rivdph_current)
    _write_calibration_state(
        round_root, river_depth_m=np.zeros(len(cell_gdf)), weir_crest=np.full(len(cell_gdf), np.nan), zs=round0_zs,
    )
else:
    _create_subgrid([{"elevation": "local_elevation_conditioned"}], round_label)

    # Exit cross-sections for round 0's flow balance: across the channel at the
    # last centreline cell of every network end inside the domain (mouths), and
    # at the centreline cell nearest every delta-outline outflow point.
    def _exit_cross_sections() -> gpd.GeoDataFrame:
        lines, names = [], []
        _ends = [r for r in rivers["reach_id"].map(normalize_reach_id) if not adjacency.get(r)]
        _anchor_cells = []
        for reach_id in _ends:
            c = cell_gdf[cell_gdf["reach_id"].map(normalize_reach_id) == reach_id]
            if not c.empty:
                _anchor_cells.append((f"mouth {reach_id}", c.loc[c["along_m"].idxmax()]))
        if delta_outflow_enabled:
            _cells_xy = np.column_stack([cell_gdf["x"], cell_gdf["y"]])
            for k, pt in enumerate(delta_outflow_gdf.to_crs(sf.crs).geometry):
                j = int(np.argmin(np.hypot(_cells_xy[:, 0] - pt.x, _cells_xy[:, 1] - pt.y)))
                _anchor_cells.append((f"outflow {k}", cell_gdf.iloc[j]))
        _lines_by_reach_id = {normalize_reach_id(r): g for r, g in zip(rivers_utm["reach_id"], rivers_utm.geometry)}
        _width_by_reach_id = {normalize_reach_id(r): float(w) for r, w in zip(rivers_utm["reach_id"], rivers_utm["width"])}
        for name, cell in _anchor_cells:
            reach_id = normalize_reach_id(cell["reach_id"])
            line = _lines_by_reach_id.get(reach_id)
            if line is None:
                continue
            s = line.project(Point(cell["x"], cell["y"]))
            a, b = line.interpolate(max(s - 50.0, 0.0)), line.interpolate(min(s + 50.0, line.length))
            tx, ty = b.x - a.x, b.y - a.y
            norm = float(np.hypot(tx, ty)) or 1.0
            nx_, ny_ = -ty / norm, tx / norm
            half = max(0.75 * _width_by_reach_id.get(reach_id, resolution_m), 1.5 * resolution_m)
            lines.append(LineString([(cell["x"] - nx_ * half, cell["y"] - ny_ * half), (cell["x"] + nx_ * half, cell["y"] + ny_ * half)]))
            names.append(name)
        return gpd.GeoDataFrame({"name": names}, geometry=lines, crs=sf.crs)

    exit_crs = _exit_cross_sections()
    if not exit_crs.empty:
        sf.cross_sections.create(locations=exit_crs, merge=False)
        log.info(f"[{round_label}] {len(exit_crs)} exit cross-section(s) for the flow balance: {list(exit_crs['name'])}")

    round0_steady = None
    period_max_zs_0, final_zs, final_times_s = _run_calibration_round(round_label, round_root, until_steady=True)

    # Flow balance at the end of round 0: total discharge out through the exits
    # against the total injected at the crossings.
    _q_in = float(dis_df.iloc[-1].sum())
    _q_exit = _read_exit_discharge(round_root)
    _imbalance = abs(_q_in - float(np.nansum(_q_exit))) / _q_in if (_q_exit is not None and _q_in > 0) else np.nan
    _problems = []
    if round0_steady is not None and not round0_steady["steady"]:
        _problems.append(f"not steady after {round0_steady['days']:.1f} day(s)")
    if _q_exit is None:
        log.warning(f"[{round_label}] no cross-section discharge in sfincs_his.nc -- flow balance not checked")
    else:
        log.info(
            f"[{round_label}] flow balance: inflow {_q_in:.0f} m3/s, outflow through {len(_q_exit)} exit(s) "
            f"{float(np.nansum(_q_exit)):.0f} m3/s ({dict(zip(exit_crs['name'], np.round(_q_exit, 0)))}), "
            f"imbalance {100 * _imbalance:.1f}%"
        )
        if _imbalance > float(steady_cfg["flow_balance_tol"]):
            _problems.append(f"flow imbalance {100 * _imbalance:.1f}% > {100 * float(steady_cfg['flow_balance_tol']):.0f}%")
    if _problems:
        _msg = f"[{round_label}] " + "; ".join(_problems)
        if bool(steady_cfg["fail"]):
            raise RuntimeError(_msg + " (river_depth_modelling.steady_state.fail is true)")
        log.warning(_msg + " -- continuing (river_depth_modelling.steady_state.fail is false)")

    # End of the (steady) run, or the maximum over the run (round0_level: period_max).
    round0_zs = (
        _read_cross_section_end_zs(round_root) if round0_level == "end_of_run" else period_max_zs_0
    )
    log.info(
        f"[{round_label}] excavation water level: {round0_level} (median {np.nanmedian(round0_zs):.3f} m)"
    )

    rivdph_current = compute_excavation_depth(round0_zs, dem_at_cell, excavation_fraction)
    log.info(
        f"[{round_label}] Excavation depth: min={np.nanmin(rivdph_current):.3f} m, "
        f"max={np.nanmax(rivdph_current):.3f} m, median={np.nanmedian(rivdph_current):.3f} m "
        f"(excavation_fraction={excavation_fraction:.3f})"
    )

    # Mouth (n_rch_dn == 0; is_delta_outflow is unreliable): the last cell is
    # natural bathymetry (river_depth_m=0), a physical constraint. A channel does not
    # get shallower before the sea, so beds shallower than that are floored to
    # it, walking upstream and stopping for good at the first cell already
    # deeper; a blanket min() would deepen unrelated upstream reaches.
    reach_is_mouth = dict(zip(rivers["reach_id"].astype(str), rivers["n_rch_dn"] == 0))
    cell_is_mouth_reach = cell_gdf["reach_id"].map(reach_is_mouth).fillna(False).to_numpy(dtype=bool)

    if cell_is_mouth_reach.any():
        along_m_arr = cell_gdf["along_m"].to_numpy()
        reach_id_arr = cell_gdf["reach_id"].to_numpy()
        reach_cell_pos: dict[str, np.ndarray] = {
            reach_id: (pos := np.flatnonzero(reach_id_arr == reach_id))[np.argsort(along_m_arr[pos])]  # upstream -> downstream
            for reach_id in cell_gdf["reach_id"].unique()
        }
        downstream_adjacency = build_downstream_adjacency(rivers)
        upstream_of: dict[str, list[str]] = {}
        for reach_id, dn_list in downstream_adjacency.items():
            for dn in dn_list:
                upstream_of.setdefault(dn, []).append(reach_id)

        for mouth_reach_id in cell_gdf.loc[cell_is_mouth_reach, "reach_id"].unique():
            reach_pos = reach_cell_pos[mouth_reach_id]
            last_pos = reach_pos[-1]
            coastal_endpoint_bed = dem_at_cell[last_pos]
            rivdph_current[last_pos] = 0.0

            current_reach_id = mouth_reach_id
            cell_positions = reach_pos[-2::-1]  # this reach's own cells, excl. coastal endpoint, downstream -> upstream
            found_intersection = False
            while not found_intersection:
                for pos in cell_positions:
                    bed_normal = dem_at_cell[pos] - rivdph_current[pos]
                    if bed_normal > coastal_endpoint_bed:
                        rivdph_current[pos] = dem_at_cell[pos] - coastal_endpoint_bed
                    else:
                        found_intersection = True
                        break
                if found_intersection:
                    break
                upstream_reach_ids = upstream_of.get(current_reach_id, [])
                if len(upstream_reach_ids) != 1:
                    break  # no further single upstream reach (headwater or confluence) -- stop clamping
                current_reach_id = upstream_reach_ids[0]
                cell_positions = reach_cell_pos[current_reach_id][::-1]  # next reach, downstream -> upstream
    else:
        log.warning(f"[{round_label}] No mouth (outlet) reach found")

    # No excavation on open-water cells: land-use sea, or at the GEBCO depth
    # limit (rule 05a clamp, spread downstream by rule 09's running minimum).
    # The rise there is over a sea bed; excavating below it double-counts the
    # depth (basin 1248635: beds around -18 m on 36 reaches).
    _cell_rows = cell_gdf["row"].to_numpy()
    _cell_cols = cell_gdf["col"].to_numpy()
    _sea_cell = landuse_on_grid[_cell_rows, _cell_cols] == LANDUSE_SEA_CODE
    _gebco_floor_cell = dem_at_cell <= (calm_sea_level_m - gebco_max_depth_m + 0.1)
    _no_dig = (_sea_cell | _gebco_floor_cell) & (rivdph_current > 0)
    if _no_dig.any():
        log.info(
            f"[{round_label}] No excavation on {int(_no_dig.sum()):,} of {len(cell_gdf):,} centreline cell(s): "
            f"{int((_sea_cell & (rivdph_current > 0)).sum()):,} land-use sea, "
            f"{int((_gebco_floor_cell & (rivdph_current > 0)).sum()):,} at the GEBCO depth limit "
            f"(<= {calm_sea_level_m - gebco_max_depth_m + 0.1:+.2f} m)"
        )
        rivdph_current = np.where(_no_dig, 0.0, rivdph_current)

    _update_reach_columns(rivdph_current)
    _write_calibration_state(
        round_root, river_depth_m=np.zeros(len(cell_gdf)), weir_crest=np.full(len(cell_gdf), np.nan), zs=round0_zs,
    )
    _round_diagnostics(0, round_root, final_zs, final_times_s, confinement_weir_gdf)
    rivbed_cell = dem_at_cell - rivdph_current

# river network map: depth (colour) and the width used (line width)
# method gvf: water depth at the mean discharge per GVF node pair, dashed where
# it comes from the width-depth fit; method modelled: the excavated depth per
# centreline cell pair.
if depth_method == "gvf":
    from src.gvf import depth_segments

    _depth_segments = depth_segments(gvf_run, crs=sf.crs)
    _depth_label = "Water depth at mean discharge (m)"
    _depth_title = f"Basin {basin_id}: river depth from GVF (SWORD wse, GloFAS mean Q, n = {channel_manning_n})"
else:
    _depth_segments = cell_depth_segments(
        cell_gdf, rivdph_current,
        {normalize_reach_id(r): float(w) for r, w in zip(rivers_utm["reach_id"], rivers_utm["width"])},
    )
    _depth_label = "Excavated depth below the conditioned DEM, river_depth_m (m)"
    _depth_title = f"Basin {basin_id}: river depth from round 0 (excavation_fraction {excavation_fraction:g})"
plot_river_depth_width(
    _depth_segments,
    land_mask_path=str(land_mask_path),
    domain_outline=sf.region.to_crs(sf.crs).geometry.union_all(),
    output_path=str(snakemake.output.plot_river_network_depth),
    depth_label=_depth_label,
    title=_depth_title,
)

# excavation (shared by rounds 1 and 2)
# One anchor per centerline cell: rivbed = DEM_at_cell - river_depth_m.
zbed_anchors = cell_gdf.assign(rivbed=rivbed_cell)

# Burned on the subgrid's own pixel grid: hydromt resampling another grid leaks
# excavated sub-pixels past channel_mask, and SFINCS's subgrid zb is about the
# MINIMUM sub-pixel elevation, so a few leaked pixels deepen a whole cell.
# natural_dem_path floors the burn: anchor interpolation can overshoot.
# Rule 13 uses this raster unresampled.
subgrid_transform = grid.transform * Affine.scale(1.0 / nr_subgrid_pixels)
subgrid_shape = (grid.shape[0] * nr_subgrid_pixels, grid.shape[1] * nr_subgrid_pixels)
burned_arr_subgrid, transform_subgrid, terrain_subgrid, stats_subgrid = burn_river_channel(
    rivers=rivers_utm, zbed_anchors=zbed_anchors,
    # GVF: floor with the unconditioned merged DEM -- rule 09's running
    # minimum would otherwise override GVF beds above its sticky floors.
    natural_dem_path=elevation_merged_path if depth_method == "gvf" else elevation_path,
    utm_crs=sf.crs, resolution_m=grid.cell_size_m / nr_subgrid_pixels,
    out_transform=subgrid_transform, out_shape=subgrid_shape,
)
# A reach's buf_poly can rasterize edge pixels outside channel_mask.
burned_arr_subgrid = constrain_to_coarse_channel_mask(
    burned_arr_subgrid, transform_subgrid, sf.crs, channel_mask, grid.transform,
)
river_burned_subgrid_path.parent.mkdir(parents=True, exist_ok=True)
with rasterio.open(
    river_burned_subgrid_path, "w", driver="GTiff", dtype="float32",
    width=burned_arr_subgrid.shape[1], height=burned_arr_subgrid.shape[0],
    count=1, crs=sf.crs, transform=transform_subgrid,
    nodata=FLOAT_NODATA, compress="deflate", tiled=True,
) as dst:
    dst.write(np.where(np.isnan(burned_arr_subgrid), FLOAT_NODATA, burned_arr_subgrid).astype(np.float32), 1)
# Channel Manning n on the same pixels (method gvf: the solver's n, so SFINCS
# sees the channel the bed was fitted for); all nodata for method modelled.
_channel_n = (
    np.where(np.isfinite(burned_arr_subgrid), np.float32(channel_manning_n), np.float32(FLOAT_NODATA))
    if depth_method == "gvf" else np.full(burned_arr_subgrid.shape, FLOAT_NODATA, dtype=np.float32)
)
with rasterio.open(
    snakemake.output.river_roughness_subgrid, "w", driver="GTiff", dtype="float32",
    width=_channel_n.shape[1], height=_channel_n.shape[0],
    count=1, crs=sf.crs, transform=transform_subgrid,
    nodata=FLOAT_NODATA, compress="deflate", tiled=True,
) as dst:
    dst.write(_channel_n.astype(np.float32), 1)
log.info(
    f"Written: {snakemake.output.river_roughness_subgrid} "
    f"({int((_channel_n != FLOAT_NODATA).sum()):,} channel pixel(s) at n = "
    f"{channel_manning_n if depth_method == 'gvf' else '-'})"
)
del _channel_n
log.info(
    f"Written: {river_burned_subgrid_path} ({stats_subgrid['n_reaches_burned']} reach(es) burned, "
    f"{int(np.isfinite(burned_arr_subgrid).sum()):,} subgrid pixel(s) @ "
    f"{grid.cell_size_m / nr_subgrid_pixels:g} m)"
)

# Main-grid dep of channel cells: per-cell MEAN of the burned subgrid, so a
# channel narrower than a cell lowers it in proportion to its area.
burned_arr_sfincs = mean_elevation_on_coarse_grid(
    burned_arr_subgrid, terrain_subgrid, nr_subgrid_pixels,
)
del terrain_subgrid
valid_burn = np.isfinite(burned_arr_sfincs)
sf.grid.data["dep"].values[valid_burn] = burned_arr_sfincs[valid_burn]
log.info(
    f"Excavated {int(valid_burn.sum()):,} cell(s) on the main grid "
    f"(per-cell mean of the subgrid burn)"
)
del burned_arr_subgrid

excavated_subgrid_elevation_list = [
    {"elevation": "local_river_burned_subgrid"},
    {"elevation": "local_elevation_conditioned"},
]

# coastal forcing for rounds 1 and 2: the protection-level storm tide
# The coast amplifies the boundary level (shoaling, reflection off the dike,
# bays): basin 2433835 reaches ~0.8 m at the dike under a 0.4 m boundary, so
# coastal crests come from the simulated level at the dike. Peaks per station,
# linear in RP between COAST-RP's tabulated RPs, MDT-corrected, rounded to 1 cm.
# River and surge both at protection RP is conservative near the mouth.
# Shape (src.surge.storm_tide_event, as rule 13): average tide plus the RP's surge
# shape, peak midway between ramp end and run end, blended in linearly from
# calm_sea_level_m over the ramp. Without hydrograph components: a half-cosine over
# surge period_hr starting at ramp end.
if surge_protection_rp_yr is not None:
    _surge_time_hr = np.arange(0.0, calibration_days * 24.0 + 1e-9, 0.25)
    with xr.open_dataset(surge_forcing_path, decode_times=False) as _sds:
        coastal_design_peaks = storm_tide_at_rp_interpolated(_sds, surge_protection_rp_yr)
        surge_period_hr = float(_sds.attrs["period_hr"])
        _surge_stations = gpd.GeoDataFrame(
            {"index": range(_sds.sizes["station"])},
            geometry=gpd.points_from_xy(_sds["longitude"].values, _sds["latitude"].values),
            crs="EPSG:4326",
        )
        _tidal = "hg_tide_m" in _sds
        if _tidal:
            _surge_peak_hr = discharge_ramp_hr + (calibration_days * 24.0 - discharge_ramp_hr) / 2.0
            _event = storm_tide_event(
                _sds, _surge_time_hr - _surge_peak_hr, coastal_design_peaks, rp_yr=surge_protection_rp_yr
            )
    if _tidal:
        _w = (
            np.clip(_surge_time_hr / discharge_ramp_hr, 0.0, 1.0)
            if discharge_ramp_hr > 0 else np.ones_like(_surge_time_hr)
        )
        _surge_wl = ((1.0 - _w) * calm_sea_level_m + _w * _event).T
        _wave_desc = (
            f"average tide + surge, peak at {_surge_peak_hr:.1f} h "
            f"(blended in from {calm_sea_level_m:+.2f} m over the first {discharge_ramp_hr:.1f} h)"
        )
    else:
        if discharge_ramp_hr + surge_period_hr > calibration_days * 24.0:
            raise ValueError(
                f"calibration run ({calibration_days * 24.0:.0f} h) too short for the storm tide: "
                f"discharge ramp {discharge_ramp_hr:.0f} h + surge period {surge_period_hr:.0f} h -- "
                f"increase river_depth_modelling.calibration_days"
            )
        _surge_wl = np.stack([
            sinusoidal_wave(calm_sea_level_m, float(peak), _surge_time_hr, discharge_ramp_hr / 24.0, surge_period_hr)
            for peak in coastal_design_peaks
        ], axis=1)
        _wave_desc = (
            f"{surge_period_hr:.0f} h half-cosine from {calm_sea_level_m:+.2f} m starting at {discharge_ramp_hr:.1f} h "
            f"(no storm-tide hydrograph components in surge_forcing.nc)"
        )
    sf.water_level.create(
        timeseries=pd.DataFrame(
            _surge_wl, index=pd.DatetimeIndex([tref + timedelta(hours=float(h)) for h in _surge_time_hr]),
            columns=range(len(_surge_stations)),
        ),
        locations=_surge_stations, buffer=water_level_buffer_m, merge=False,
    )
    log.info(
        f"Rounds 1-2 coastal forcing: RP {surge_protection_rp_yr:.1f} storm tide at {len(_surge_stations)} station(s), "
        f"peak {coastal_design_peaks.min():+.2f}..{coastal_design_peaks.max():+.2f} m "
        f"(boundary series {_surge_wl.min():+.2f}..{_surge_wl.max():+.2f} m), {_wave_desc}"
    )
else:
    log.warning("No coastal protection RP -- rounds 1-2 keep the steady calm-sea boundary")

# round 1: confined, excavated
# sf.weirs still holds round 0's confinement weir -- unchanged here.
round_label = ROUND_TITLES[1]
round_root = _start_round(1)
_create_subgrid(excavated_subgrid_elevation_list, round_label, channel_roughness=True)
period_max_zs_1, final_zs, final_times_s = _run_calibration_round(round_label, round_root)
zs_field_1 = _read_period_max_zs_field(round_root)

# Production weir, exported as-is; round 2 verifies exactly this weir.
weir_gdf, final_weir_diagnostics = build_coastal_protection_weir(
    grid, landuse_on_grid, channel_mask, coastal_protection_crest_m, min_component_cells, weir_par1,
    water_side_crest_on_grid=zs_field_1, freeboard_m=weir_freeboard_m, channel_mask_gap_free=True,
    unprotected_ocean_wetlands=unprotected_ocean_wetlands,
)

# Per-centerline-cell summary of the same rule; reporting only.
weir_crest_cell = ceil_crest(np.fmax(period_max_zs_1, coastal_protection_crest_m) + weir_freeboard_m)
log.info(
    f"[{round_label}] Crest (centerline summary): min={np.nanmin(weir_crest_cell):.3f} m, "
    f"max={np.nanmax(weir_crest_cell):.3f} m, median={np.nanmedian(weir_crest_cell):.3f} m "
    f"(+{weir_freeboard_m:.2f} m freeboard)"
)
_update_reach_columns(rivdph_current, weir_crest_cell)
_write_calibration_state(round_root, river_depth_m=rivdph_current, weir_crest=weir_crest_cell, zs=period_max_zs_1)
_round_diagnostics(1, round_root, final_zs, final_times_s, confinement_weir_gdf)

# round 2: verification with the real production weir
round_label = ROUND_TITLES[2]
round_root = _start_round(2)
_set_weir(weir_gdf, final_weir_diagnostics, round_label, "production crests (per edge, from round 1)")
_create_subgrid(excavated_subgrid_elevation_list, round_label, channel_roughness=True)
period_max_zs_2, final_zs, final_times_s = _run_calibration_round(round_label, round_root)

edge_crest_on_grid = None
if final_weir_diagnostics.get("edge_water_side_mask") is not None:
    zs_field_2 = _read_period_max_zs_field(round_root)
    _edge_mask = final_weir_diagnostics["edge_water_side_mask"]
    edge_crest_on_grid = np.where(_edge_mask, final_weir_diagnostics["crest_surface"], np.nan).astype(np.float32)
    # NaN (never-wetted) water side compares False -- nothing to overtop there.
    _exceedance = zs_field_2[_edge_mask] - edge_crest_on_grid[_edge_mask]
    _n_overtopped = int((_exceedance > 0).sum())
    if _n_overtopped:
        log.warning(
            f"[{round_label}] {_n_overtopped} of {int(_edge_mask.sum()):,} edge water-side cell(s) exceed "
            f"their own crest (max exceedance {np.nanmax(_exceedance):.3f} m) -- see the crest gap map; "
            f"crests are NOT adjusted (verification only)"
        )
    else:
        log.info(
            f"[{round_label}] Verified: no edge water-side cell exceeds its own crest "
            f"({int(_edge_mask.sum()):,} cell(s) checked)"
        )
_write_calibration_state(round_root, river_depth_m=rivdph_current, weir_crest=weir_crest_cell, zs=period_max_zs_2)
_round_diagnostics(2, round_root, final_zs, final_times_s, weir_gdf, crest_gap_crest_on_grid=edge_crest_on_grid)

# canonical production outputs: burned river channel + weir
with rasterio.open(
    snakemake.output.river_burned_dem_sfincs_grid, "w", driver="GTiff", dtype="float32",
    width=burned_arr_sfincs.shape[1], height=burned_arr_sfincs.shape[0],
    count=1, crs=sf.crs, transform=grid.transform,
    nodata=FLOAT_NODATA, compress="deflate",
) as dst:
    dst.write(np.where(np.isnan(burned_arr_sfincs), FLOAT_NODATA, burned_arr_sfincs).astype(np.float32), 1)
log.info(
    f"Written: {snakemake.output.river_burned_dem_sfincs_grid} "
    f"({int(np.isfinite(burned_arr_sfincs).sum()):,} channel cell(s))"
)

Path(snakemake.output.coastal_protection_weir).parent.mkdir(parents=True, exist_ok=True)
weir_gdf.to_file(snakemake.output.coastal_protection_weir, driver="GPKG")
log.info(f"Written: {snakemake.output.coastal_protection_weir} ({len(weir_gdf)} segment(s))")

# Real open sea on this rule's grid, no reprojection
# Mostly a GRID-RESOLUTION MISALIGNMENT fix, not an isolated-pocket one --
# the weir is traced on landuse_on_grid (rule grid_align_landuse, 09b:
# native landuse.tif resampled via nearest-neighbor onto the much coarser
# SFINCS regular grid, ONCE, upstream), so the protected side disagrees with
# native landuse.tif along the whole coast where coarse cell edges cut the
# coastline (basin 2433835 has one connected sea component at native
# resolution: a coastline-wide fringe, not discrete pockets).
# The single sea/land classification for rule 13's zsini (uncorrected, fringe
# cells would start wet behind the weir) and for flood diagnostics
# (src.postprocessing sea_mask_path), which reproject it onto subgrid- or
# cell-resolution grids; that is alignment-safe only from a grid-aligned source.
_protected_pocket_mask = final_weir_diagnostics.get(
    "protected_pocket_mask", np.zeros(landuse_on_grid.shape, dtype=bool)
)
_sea_cells_on_grid = (landuse_on_grid == LANDUSE_SEA_CODE) & ~_protected_pocket_mask
_sea_cells_arr = np.where(_sea_cells_on_grid, np.float32(1.0), np.float32(FLOAT_NODATA))
_sea_cells_profile = {
    "driver": "GTiff", "height": _sea_cells_arr.shape[0], "width": _sea_cells_arr.shape[1],
    "count": 1, "dtype": "float32", "crs": grid.crs, "transform": grid.transform,
    "nodata": FLOAT_NODATA,
}
Path(snakemake.output.zsini_sea_cells).parent.mkdir(parents=True, exist_ok=True)
with rasterio.open(snakemake.output.zsini_sea_cells, "w", **_sea_cells_profile) as _dst:
    _dst.write(_sea_cells_arr, 1)
log.info(
    f"zsini sea cells (this rule's own grid, no reprojection): "
    f"{int(_sea_cells_on_grid.sum()):,} real open-sea cell(s). "
    f"Written: {snakemake.output.zsini_sea_cells}"
)

# seed-to-mouth round-profile diagnostics (bed/crest/water-level per round)
# Not declared outputs. Read from each round's calibration_state.csv; written
# to the visuals directory, not the disposable calib_root.
visuals_dir = Path(snakemake.output.plot_calibration).parent
visuals_dir.mkdir(parents=True, exist_ok=True)
seed_ids = sorted(
    {normalize_reach_id(reach_id) for reach_id, s in zip(rivers["reach_id"], rivers["is_seed"]) if bool(s) is True} - {None}
)
if not seed_ids:
    log.warning("No is_seed reach found -- skipping seed-to-mouth round-profile diagnostics")
if depth_method == "gvf":
    _round0_level_label = "water level, GVF steady"
elif round0_level == "end_of_run":
    _round0_level_label = "water level, end of run"
else:
    _round0_level_label = "water level, period max"
_water_level_labels = [_round0_level_label] + ["water level, period max"] * N_ROUNDS_AFTER_0
for _seed in seed_ids:
    _path_reach_ids, _profiles_by_round = gather_calibration_round_profile(
        calib_root, rivers_utm, _seed, N_ROUNDS_AFTER_0
    )
    plot_calibration_round_profiles(
        basin_id=basin_id, seed=_seed, profiles_by_round=_profiles_by_round,
        n_rounds=N_ROUNDS_AFTER_0,
        output_subplots_path=str(visuals_dir / f"round_profiles_seed{_seed}_subplots.png"),
        output_combined_path=str(visuals_dir / f"round_profiles_seed{_seed}_combined.png"),
        round_titles=ROUND_TITLES, water_level_labels=_water_level_labels,
    )
    pd.concat(
        [df.assign(round=i) for i, df in _profiles_by_round.items()], ignore_index=True
    ).to_csv(visuals_dir / f"round_profiles_seed{_seed}.csv", index=False)
    log.info(f"Seed {_seed}: round-profile diagnostics written (path of {len(_path_reach_ids)} reach(es))")

n_missing = rivers["river_depth_m"].isna().sum()
if n_missing:
    log.warning(f"{n_missing} reach(es) missing a calibrated depth (no centerline cell?) -- left as NaN")

Path(snakemake.output.depth_estimated_river_network).parent.mkdir(parents=True, exist_ok=True)
rivers.to_file(snakemake.output.depth_estimated_river_network, driver="GPKG")
log.info(f"Written: {snakemake.output.depth_estimated_river_network}")

# canonical outputs: copies of round 2's own per-round files
Path(snakemake.output.plot_calibration).parent.mkdir(parents=True, exist_ok=True)
_last_round_paths = _round_visual_paths(N_ROUNDS_AFTER_0)
for _canonical, _kind in (
    (snakemake.output.plot_calibration, "plot_calibration"),
    (snakemake.output.plot_water_level_timeseries, "plot_water_level"),
    (snakemake.output.plot_max_inundation, "plot_max_inundation"),
    (snakemake.output.animation_flood_progress, "animation"),
    (snakemake.output.plot_crest_gap_map, "crest_gap_map"),
):
    shutil.copy(_last_round_paths[_kind], _canonical)
log.info(f"Canonical diagnostic outputs copied from round {N_ROUNDS_AFTER_0}'s own files")
log.info("Done")

