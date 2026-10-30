"""Estimate the river bed (water-surface slope) and calibrate riverbank weir crests with SFINCS; run by rule modelled_depth_estimation.

Round 0 -- river bed = rule 10's water surface - Manning depth at the GloFAS design
discharge and the surface slope (src.river_bed); no SFINCS run.

Rounds 1-2 -- runs (round 2 only with verify_crests) of one minimal, disposable SFINCS
model (regular grid, production subgrid setup) with the production weir
geometry (src.protection_weir.build_coastal_protection_weir: coast and river
banks in one water_like boundary). Initial level: river channels at the steady
level of the mean discharge on round 0's bed (as rule 12a's zsini), sea cells
at calm_sea_level_m. Forcing: every sea station gets the storm-tide hydrograph
of the event runs (src.surge.storm_tide_event: average tide + surge shape,
peaking at its storm tide at the FLOPROS coastal RP) with the peak at
ramp_hr + T_walled, when the calibration discharge reaches the coast, blended
in linearly from the calm level over ramp_hr; every crossing ramped over
ramp_hr (half-cosine, src.river_forcing.cosine_ramp)
from its mean discharge to the discharge at the protection RP (bankfull as
fallback), then held to the run end. Run length: ramp_hr + T_walled +
calibration_margin_hr, with T_walled the slowest walled travel time at the
calibration discharge (src.river_profile.walled_travel_time): per node the
gravity speed where friction acts slower than the ramp (r ramp_hr below
wave_regime_rT[0], r = g n^2 u / h^(4/3)), else the kinematic 5/3 u,
from a crossing to its network end on round 0's bed, so the full discharge
reaches the coast and the levels settle before the run ends (checked per
round: steady_check).

Round 1 -- confined, excavated. With no overbank relief the held discharge and
storm tide give the highest level reached, including coastal amplification
(shoaling, reflection off the dike), so per edge, no iteration:

    crest(edge) = ceil_cm( max(zsmax(water-side cell),
                                coastal_protection_crest_m) + freeboard_m )

zsmax: SFINCS's maximum over every computational timestep. ceil_cm: rounded
up to the next centimetre (src.surge.ceil_crest), the sfincs.weir precision,
so a crest is never below its zsmax. A never-wetted water side falls back to
coastal_protection_crest_m, so no edge is dropped.

Round 2 -- verification with the production weir. Changes nothing; reports
every edge whose water side exceeds its crest.

period_max_water_level (rounds 1-2, per-cell diagnostics) is each cell's own
MAXIMUM simulated water level over the run
(src.river_depth_calibration.compute_period_max_zs, from the hourly zs
field each round writes to its own sfincs_map.nc), not a
windowed-flatness "converged"/stabilized value -- a persistent, non-damping
oscillation at a discharge-injection cell never satisfies a flatness/trend
tolerance no matter how long the run is extended. Sizing crests only needs
the worst-case (peak) forced water level, which the period maximum gives
directly, oscillation or not.

A centerline cell's water level is the MAX across its channel cross-section:
the shorter of its row and column channel_mask runs, extended one cell each
side to the land cell the weir sits on. Axis-aligned, not reach-normal.

Delta-outline outflow points get a free outflow boundary (mask=3).

Elevation: the coarse dep is elevation_conditioned_sfincs_grid.tif, the same
raster rule 12a uses; the subgrid uses native elevation_conditioned.tif. Rounds
1-2 burn the channel once on the subgrid pixel grid; channel cells' main-grid
dep is its per-cell mean. Rule 13 imports both rasters and the weir as-is.

Outputs: river_network_depth_estimated holds per-reach MEDIAN river_depth_m and
weir_crest_calibrated_m (summary only; production uses the per-edge weir file).
calib_root/round{i}/calibration_state.csv (not a declared output) holds the
per-cell state of each round; weir_crest is NaN in round 0 (estimated bed).
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

from src.protection_weir import (
    GridArrays,
    build_coastal_protection_weir,
    floor_crests_above_face_zmin,
    ocean_linked_wetland_mask,
)
from src.domain import load_domain
from src.grid import create_model_grid, load_grid_def
from src.hydromt_sfincs_patch import (
    ReusableSubgrid,
    patch_nearest_extrapolation,
    patch_subgrid_write,
    raster_catalog_entry,
)
from src.landuse import LANDUSE_SEA_CODE
from src.log import setup_logging
from src.plots import (
    plot_crest_profiles,
    plot_max_inundation_map,
    plot_run_forcing,
)
from src.postprocessing import (
    compute_max_inundation,
    land_bed_level,
)
from src.raster import (
    FLOAT_NODATA,
    dry_isolated_sea_cells,
    read_on_grid,
    write_float_raster,
)
from src.river_burn import (
    build_centerline_cells_regular,
    build_channel_mask_regular,
    burn_river_channel,
    constrain_to_coarse_channel_mask,
    mean_elevation_on_coarse_grid,
)
from src.river_depth_calibration import (
    bifurcation_cross_sections,
    build_calibration_seed_discharge,
    compute_period_max_zs,
    gather_calibration_round_profile,
)
from src.river_bed import (
    depth_segments,
    estimate_river_bed_from_files,
    values_at_points,
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
    read_face_zmin,
    run_sfincs_subprocess,
    set_sea_and_outflow_boundaries,
    write_weir_file,
)
from src.river_forcing import cosine_ramp
from src.river_profile import mainstem_profile, walled_travel_time
from src.surge import (
    CREST_PRECISION_M,
    ceil_crest,
    read_calm_sea_level_m,
    storm_tide_at_rp_interpolated,
    storm_tide_event,
)
from src.water_surface import river_cells_at_surface

log = setup_logging(snakemake.log[0])
patch_subgrid_write()
patch_nearest_extrapolation()

# paths & params
elevation_path = Path(snakemake.input.elevation_conditioned)
elevation_sfincs_grid_path = Path(snakemake.input.elevation_conditioned_sfincs_grid)
river_elevation_max_path = Path(snakemake.input.river_elevation_max)
river_network_path = Path(snakemake.input.clean_river_network)
river_forcing_path = Path(snakemake.input.river_forcing)
protection_levels_path = Path(snakemake.input.protection_levels)
# Grid-aligned land mask (rule grid_align_landuse) -- plot background only.
land_mask_path = Path(snakemake.input.land_mask_on_grid)
domain_gpkg_path = Path(snakemake.input.domain_gpkg)
# Already on the SFINCS grid (rule 09); read without reprojection.
landuse_on_grid_path = Path(snakemake.input.landuse_on_grid)
roughness_on_grid_path = Path(snakemake.input.roughness_on_grid)
# Native-resolution roughness (rule 09): subgrid table only, which needs
# sub-cell detail; the main-grid "manning" uses roughness_on_grid_path.
roughness_path = Path(snakemake.input.roughness)
surge_forcing_path = Path(snakemake.input.surge_forcing)
delta_outflow_points_path = Path(snakemake.input.delta_outflow_points)

basin_id = str(snakemake.wildcards.basin_id)  # plot titles
calib_root = Path(snakemake.params.calib_root)
sfincs_exe = Path(snakemake.params.sfincs_exe)
timeout_s = int(snakemake.params.timeout_s)

bed_cfg = dict(snakemake.params.bed)
gebco_min_depth_m = float(snakemake.params.gebco_min_depth_m)
channel_manning_n = float(snakemake.params.channel_manning_n)
water_surface_cfg = dict(snakemake.params.water_surface)
calibration_margin_hr = float(snakemake.params.calibration_margin_hr)
calibration_max_days = float(snakemake.params.calibration_max_days)
wave_regime_rT = tuple(float(v) for v in snakemake.params.wave_regime_rT)
steady_check_cfg = dict(snakemake.params.steady_check)
verify_crests = bool(snakemake.params.verify_crests)
crosssections = bool(snakemake.params.crosssections)
depth_scale = float(snakemake.params.depth_scale)
ramp_hr = float(snakemake.params.ramp_hr)
weir_crest_m = float(snakemake.params.weir_crest_m)
weir_par1 = float(snakemake.params.weir_par1)
# The only margin on the production weir, on top of round 1's water level.
weir_freeboard_m = float(snakemake.params.weir_freeboard_m)
# True: wetland/lagoon patches linked to the open sea stay seaward of the
# coastal dike in all three rounds.
unprotected_ocean_wetlands = bool(snakemake.params.unprotected_ocean_wetlands)
min_component_cells = int(snakemake.params.min_component_cells)
nr_subgrid_pixels = int(snakemake.params.nr_subgrid_pixels)
nr_levels = int(snakemake.params.nr_levels)
nrmax = int(snakemake.params.nrmax)
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
surge_protection_rp_yr = (
    float(surge_protection_rp_yr)
    if surge_protection_rp_yr is not None and np.isfinite(float(surge_protection_rp_yr))
    else None
)

with xr.open_dataset(river_forcing_path, decode_times=False) as river_ds:
    seed_q = build_calibration_seed_discharge(river_ds, protection_rp_yr)

# Rule 08's decisions, reused: seeds that flow OUT of the network inject
# nothing, and canal arms get no share (src.river_network).
if "seed_flows_out" in rivers.columns:
    _outward = {
        normalize_reach_id(r)
        for r, f in zip(rivers["reach_id"], rivers["seed_flows_out"])
        if bool(f)
    }
    seed_q = {k: v for k, v in seed_q.items() if normalize_reach_id(k) not in _outward}
_excluded_arms = (
    {
        normalize_reach_id(r)
        for r, f in zip(rivers["reach_id"], rivers["zero_discharge_share"])
        if bool(f)
    }
    if "zero_discharge_share" in rivers.columns
    else set()
)
adjacency = build_downstream_adjacency(rivers)
q_calib = accumulate_discharge(rivers, seed_q, adjacency, excluded_arms=_excluded_arms)
rivers["q_calib"] = q_calib
log.info(
    f"Calibration discharge: {int((q_calib > 0).sum())}/{len(rivers)} reach(es) "
    f"with Q > 0, max Q = {q_calib.max():.1f} m3/s"
)

# Real, steady coastal baseline (calm sea level, MDT-corrected, no SLR) -- same
# value 14_build_sfincs.py/14_run_spinup.py use for their own initial/steady
# conditions -- applied uniformly to every round.
# coastal_protection_crest_m: the production coastal crest, the floor under
# every production weir edge's crest. Rounds 0 and 1 use weir_crest_m instead.
with xr.open_dataset(surge_forcing_path, decode_times=False) as _surge_ds:
    # Rounded to 1 cm, as production's zsini.
    calm_sea_level_m = read_calm_sea_level_m(_surge_ds)
    coastal_protection_crest_m = (
        float(_surge_ds["coastal_protection_crest_m"].values)
        if "coastal_protection_crest_m" in _surge_ds
        else 0.0
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
# outflow boundary (mask=3, as in rule 12a) the reach would be walled off and its
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
    "local_elevation_conditioned": raster_catalog_entry(elevation_path),
    "local_elevation_conditioned_sfincs_grid": raster_catalog_entry(
        elevation_sfincs_grid_path
    ),
    "local_roughness": raster_catalog_entry(roughness_on_grid_path),
    # Native resolution, subgrid table only.
    "local_roughness_native": raster_catalog_entry(roughness_path),
    **(
        {
            "local_delta_outflow_points": {
                "data_type": "GeoDataFrame",
                "uri": str(delta_outflow_points_path),
                "driver": "pyogrio",
            },
        }
        if delta_outflow_enabled
        else {}
    ),
    "local_river_burned_subgrid": raster_catalog_entry(river_burned_subgrid_path),
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
subgrid_builder = ReusableSubgrid(sf)
log.info(f"Calibration grid created from the shared grid definition: {resolution_m} m")

# tstop, dtmaxout and trstout follow from round 0's bed (calibration forcing,
# before round 1).
tref = datetime(2000, 1, 1)
sf.config.update(
    {
        "tref": tref,
        "tstart": tref,
        "dthisout": 3600,
        "dtmapout": 3600,
        "dtrstout": 0,
        "zsini": -9999.0,
    }
)

# round-0 elevation (un-excavated), mask, roughness
# Same coarse background raster as rule 12a.
sf.elevation.create(
    elevation_list=[{"elevation": "local_elevation_conditioned_sfincs_grid"}]
)
# Elevation ceiling on the active mask, as in rule 12a.
river_elevation_max_m = None
if active_mask_enabled:
    with open(river_elevation_max_path) as f:
        river_elevation_max_m = float(json.load(f)["river_elevation_max_m"])
create_active_mask(
    sf, delta_domain, river_elevation_max_m, active_mask_elevation_buffer_m
)

# Water-level boundary on open sea only (landuse_on_grid == 200), as rule 12a.
with rasterio.open(landuse_on_grid_path) as _lu_src_bnd:
    _landuse_bnd = _lu_src_bnd.read(1)
set_sea_and_outflow_boundaries(
    sf,
    _landuse_bnd,
    len(delta_outflow_gdf) if delta_outflow_enabled else 0,
    outflow_buffer_m,
)

# spatially-varying zsini: sea cells start AT calm_sea_level_m, not bone-dry
# Compared against sf.grid.data["dep"], not elevation_conditioned_sfincs_grid.tif,
# which extends beyond the grid bounds. A coastal cell starting dry fills with
# a transient spike above its settled level, which the run maximum would add
# to its crest.
_lu_on_grid_zsini = read_on_grid(landuse_on_grid_path, sf.grid.data["dep"].shape)
_zsini_arr = np.where(
    _lu_on_grid_zsini == LANDUSE_SEA_CODE, np.float32(calm_sea_level_m), np.nan
)

# Isolated sea cells (not the main ocean component) below calm_sea_level_m start
# dry at dep, as in rule 12a.
_zsini_arr, _n_fix_zsini = dry_isolated_sea_cells(
    _zsini_arr,
    sf.grid.data["dep"].values,
    sf.grid.data["dep"].rio.nodata or FLOAT_NODATA,
    calm_sea_level_m,
)
if _n_fix_zsini > 0:
    log.info(
        f"zsini connectivity fix: {_n_fix_zsini} isolated wet sea-cell(s) set to dep"
    )
# River channel cells follow round 0's bed (calibration forcing, before round 1).

sf.roughness.create(roughness_list=[{"manning": "local_roughness"}])

# shared, round-invariant pieces: grid arrays, channel_mask, landuse
# Coast and channel in one water_like boundary so the mouth is never sealed.
# GridArrays copies dep/mask at construction, so it stays valid after the
# excavation. channel_mask (weir corridor and excavation limit) matches what
# rule 12a computes.
rivers_utm = rivers.to_crs(sf.crs)
grid = GridArrays.from_regular(sf.grid.data["dep"], sf.grid.data["mask"], sf.crs)
channel_mask = build_channel_mask_regular(
    rivers_utm, "width", grid.shape, grid.transform
)
landuse_on_grid = read_on_grid(landuse_on_grid_path, grid.shape)
ocean_mask_grid = landuse_on_grid == LANDUSE_SEA_CODE

# Written to file: compute_max_inundation/compute_flood_progression take a path.
# Not weir-aware (no protected_pocket_mask before round 1).
_round_ocean_mask_path = calib_root / "ocean_mask_on_grid.tif"
with rasterio.open(
    _round_ocean_mask_path,
    "w",
    driver="GTiff",
    dtype="float32",
    count=1,
    height=ocean_mask_grid.shape[0],
    width=ocean_mask_grid.shape[1],
    crs=grid.crs,
    transform=grid.transform,
    nodata=FLOAT_NODATA,
) as _dst:
    _dst.write(np.where(ocean_mask_grid, np.float32(1.0), np.float32(FLOAT_NODATA)), 1)

# Centerline calibration cells: every grid cell a reach's centerline crosses
# Per cell from sfincs_map.nc, not observation points: a sparse point can miss
# a sharp spike at the discharge-injection cell.
cell_gdf = build_centerline_cells_regular(rivers_utm, grid.shape, grid.transform)
if cell_gdf.empty:
    raise RuntimeError(
        "No centerline calibration cells could be identified -- empty river network?"
    )
dem_at_cell = grid.bed_elevation_m[
    cell_gdf["row"].to_numpy(), cell_gdf["col"].to_numpy()
]
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
cell_cross_section_rowcol: list[
    np.ndarray
] = []  # one (n_i, 2) row/col array per cell_gdf row
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
_unique_rc, _cross_section_inverse = np.unique(
    _all_cross_section_rc, axis=0, return_inverse=True
)
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
    _cross_section_inverse[_group_bounds[i] : _group_bounds[i + 1]]
    for i in range(len(cell_gdf))
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
_reach_along_mid = cell_gdf.groupby("reach_id")["along_m"].transform(
    lambda s: s.max() / 2.0
)
_dist_to_mid = (cell_gdf["along_m"] - _reach_along_mid).abs()
representative_cell_pos = _dist_to_mid.groupby(cell_gdf["reach_id"]).idxmin().to_numpy()
representative_reach_ids = cell_gdf.loc[representative_cell_pos, "reach_id"].to_numpy()


def _resolve_map_cell_idx(
    map_nc_path: Path, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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
        zs = (
            ds["zs"]
            .isel(
                n=xr.DataArray(n_idx, dims="cell"),
                m=xr.DataArray(m_idx, dims="cell"),
            )
            .values
        )
        times_s = (ds["time"].values - ds["time"].values[0]) / np.timedelta64(1, "s")
    return zs, times_s


def _read_zs_and_resolve(round_root: Path):
    global map_cell_idx
    if map_cell_idx is None:
        map_cell_idx = _resolve_map_cell_idx(
            round_root / "sfincs_map.nc",
            cell_gdf["x"].to_numpy(),
            cell_gdf["y"].to_numpy(),
        )
        log.info(
            f"Resolved {len(map_cell_idx[0])} centerline cell(s) against sfincs_map.nc's own grid indices"
        )
    return _read_map_zs_at_cells(round_root, *map_cell_idx)


def _read_cross_section_period_max_zs(round_root: Path) -> np.ndarray:
    global cross_section_map_cell_idx
    if cross_section_map_cell_idx is None:
        cross_section_map_cell_idx = _resolve_map_cell_idx(
            round_root / "sfincs_map.nc",
            cross_section_unique_x,
            cross_section_unique_y,
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
        field_map_cell_idx = _resolve_map_cell_idx(
            map_nc_path, np.asarray(_xs), np.asarray(_ys)
        )
        log.info(
            f"Resolved {grid.shape[0] * grid.shape[1]:,} grid cell(s) against sfincs_map.nc's own grid indices"
        )
    with xr.open_dataset(map_nc_path) as ds:
        zsmax = ds["zsmax"]
        zsmax = zsmax.max(dim=[d for d in zsmax.dims if d not in ("n", "m")]).values
    n_idx, m_idx = field_map_cell_idx
    return zsmax[n_idx, m_idx].reshape(grid.shape).astype(np.float32)


def _run_calibration_round(round_label: str, round_root: Path):
    """Write and run whatever elevation/weir the caller just set up on the
    (shared) `sf` object, into round_root (the caller has already pointed
    sf.root at this via sf.root.set()) -- one run of run_hr. Each
    round starts cold (no restart chained across rounds -- the geometry
    changed between rounds, so resuming a previous round's dynamic state
    would be physically invalid).

    No convergence check: the per-cell water level used for calibration is
    each cell's own maximum over the full run (compute_period_max_zs), not a
    windowed-flatness "stabilized" value -- a persistent, non-damping
    oscillation at a discharge-injection cell never satisfies a flatness
    tolerance no matter how long the run is extended; the peak is exactly
    what calibration needs, converged or not."""
    sf.config.set("rstfile", None)
    sf.config.update(
        {
            "tstart": tref,
            "tstop": tstop,
            "trstout": trstout_s,
            "dtrstout": 0,
            "dtmaxout": trstout_s,
            "zsini": -9999.0,
        }
    )
    sf.write()
    write_weir_file(sf)  # crests at 1 cm, not hydromt_sfincs' 0.1 m
    log.info(f"Running SFINCS calibration ({round_label}, {run_hr:.0f} h)")
    run_sfincs_subprocess(
        sfincs_exe,
        round_root,
        timeout_s,
        log,
        label=f"SFINCS calibration ({round_label})",
        n_threads=snakemake.threads,
    )

    # zs (single intersected cell) is for the timeseries plot only.
    zs, times_s = _read_zs_and_resolve(round_root)
    period_max_zs = _read_cross_section_period_max_zs(round_root)
    log.info(
        f"[{round_label}] Period max water level (cross-section): min={period_max_zs.min():.3f} m, "
        f"max={period_max_zs.max():.3f} m, median={np.median(period_max_zs):.3f} m "
        f"(over the run's last {zs.shape[0]} output step(s))"
    )
    return period_max_zs, zs, times_s


wgs84_bounds, domain_crs, domain_poly = load_domain(
    snakemake.input.spec_basins_meta, domain_gpkg_path
)

N_ROUNDS_AFTER_0 = 2 if verify_crests else 1
ROUND_TITLES = [
    "round 0 (estimated river bed, no SFINCS run)",
    "round 1 (confined, excavated)",
    "round 2 (verification, real crests)",
]


def _start_round(round_idx: int) -> Path:
    # One directory per round: reusing filenames fails SFINCS's netCDF creation
    # on Windows ("Permission denied"/"NetCDF: Not a valid ID") once a previous
    # round's diagnostics read them.
    round_root = calib_root / f"round{round_idx}"
    round_root.mkdir(parents=True, exist_ok=True)
    sf.root.set(round_root)
    return round_root


def _create_subgrid(elevation_list: list[dict], round_label: str) -> None:
    # Each round's diagnostics read its own dep_subgrid.tif; a round whose
    # inputs equal the previous round's (round 2) reuses that round's table.
    # The channel's channel_manning_n is already in roughness.tif (rule 09).
    roughness_list = [{"manning": "local_roughness_native"}]
    sources = [d["elevation"] for d in elevation_list] + [
        d["manning"] for d in roughness_list
    ]
    built = subgrid_builder.create(
        [Path(local_catalog[name]["uri"]) for name in sources],
        elevation_list=elevation_list,
        roughness_list=roughness_list,
        river_list=[],
        nr_subgrid_pixels=nr_subgrid_pixels,
        nr_levels=nr_levels,
        write_dep_tif=True,
        write_man_tif=True,
        nrmax=nrmax,
    )
    log.info(
        f"[{round_label}] Subgrid table {'created' if built else 'reused'}: {nr_subgrid_pixels} px/cell"
    )


def _set_weir(
    weir_gdf: gpd.GeoDataFrame,
    weir_diagnostics: dict,
    round_label: str,
    description: str,
) -> None:
    if weir_diagnostics.get("applicable", True) and not weir_gdf.empty:
        sf.weirs.set(weir_gdf, merge=False)
        log.info(f"[{round_label}] Weir set: {len(weir_gdf)} segment(s), {description}")
    else:
        log.warning(
            f"[{round_label}] Coastal protection weir: not applicable -- this round is not confined"
        )


def _update_reach_columns(
    river_depth_m: np.ndarray, weir_crest: np.ndarray | None = None
) -> None:
    # Median, so a mouth reach's forced river_depth_m=0 last cell does not skew it.
    cell_state = cell_gdf[["reach_id"]].assign(river_depth_m=river_depth_m)
    if weir_crest is not None:
        cell_state = cell_state.assign(crest=weir_crest)
    reach_medians = cell_state.groupby("reach_id").median()
    rivers["river_depth_m"] = (
        rivers["reach_id"].astype(str).map(reach_medians["river_depth_m"])
    )
    if weir_crest is not None:
        rivers["weir_crest_calibrated_m"] = (
            rivers["reach_id"].astype(str).map(reach_medians["crest"])
        )
        rivers_utm["weir_crest_calibrated_m"] = (
            rivers_utm["reach_id"].astype(str).map(reach_medians["crest"])
        )


def _write_calibration_state(
    round_root: Path, river_depth_m: np.ndarray, weir_crest: np.ndarray, zs: np.ndarray
) -> None:
    cell_gdf.drop(columns="geometry").assign(
        dem_m=dem_at_cell,
        river_depth_m=river_depth_m,
        weir_crest_m=weir_crest,
        zs_m=zs,
    ).to_csv(round_root / "calibration_state.csv", index=False)


def _round_diagnostics(
    round_idx: int,
    round_root: Path,
    final_zs: np.ndarray,
    final_times_s: np.ndarray,
    weir_gdf: gpd.GeoDataFrame,
    crest_gap_crest_on_grid: np.ndarray | None = None,
) -> None:
    # Run before the next round's sf.write(). crest_gap_crest_on_grid: last
    # round only, with the production crests.
    round_label = ROUND_TITLES[round_idx]
    run = f"calib_r{round_idx}"
    out = snakemake.output

    plot_run_forcing(
        round_root, getattr(out, f"plot_forcing_r{round_idx}"), basin_id, run
    )
    da_hmax, _da_dep = compute_max_inundation(
        round_root,
        land_bed_level(round_root, _round_ocean_mask_path),
        hmin=0.0,
    )
    if da_hmax is None:
        raise RuntimeError(
            f"[{round_label}] no zsmax in the SFINCS output of {round_root}"
        )
    plot_max_inundation_map(
        da_hmax,
        domain_poly,
        str(land_mask_path),
        str(river_network_path),
        str(getattr(out, f"plot_max_depth_r{round_idx}")),
        basin_id=basin_id,
        run_label=run,
    )
    # Water level at one centreline cell per reach, for the analysis figures.
    pd.DataFrame(
        final_zs[:, representative_cell_pos],
        columns=[f"reach {reach_id}" for reach_id in representative_reach_ids],
    ).assign(time_s=final_times_s).to_csv(
        getattr(out, f"water_level_r{round_idx}"), index=False
    )
    if round_idx == N_ROUNDS_AFTER_0:
        # Production crest on each edge's water-side cell; all-NaN when the weir
        # is not applicable (no ocean cells) or has no seaward edge.
        crest = (
            np.full(grid.shape, np.nan, dtype=np.float32)
            if crest_gap_crest_on_grid is None
            else crest_gap_crest_on_grid
        )
        write_float_raster(out.crest_on_grid, crest, grid.transform, grid.crs)
    log.info(f"[{round_label}] Diagnostics written")


def _edge_crest_on_grid(weir_diagnostics: dict) -> np.ndarray | None:
    # Production crest on each edge's water-side cell, NaN elsewhere; None
    # when the weir is not applicable.
    edge_mask = weir_diagnostics.get("edge_water_side_mask")
    if edge_mask is None:
        return None
    return np.where(edge_mask, weir_diagnostics["crest_surface"], np.nan).astype(
        np.float32
    )


# Confinement weir (rounds 0 and 1): all-NaN water_side_crest_on_grid puts
# every edge at weir_crest_m. channel_mask_gap_free=True: the mask is
# all_touched, so close_gaps() is skipped and the weir hugs the corridor; the
# production weir uses the same flag, so all rounds share one geometry.
confinement_weir_gdf, confinement_weir_diagnostics = build_coastal_protection_weir(
    grid,
    landuse_on_grid,
    channel_mask,
    weir_crest_m,
    min_component_cells,
    weir_par1,
    water_side_crest_on_grid=np.full(grid.shape, np.nan, dtype=np.float32),
    channel_mask_gap_free=True,
    unprotected_ocean_wetlands=unprotected_ocean_wetlands,
)

# Dike position with unprotected_ocean_wetlands off vs on, for the analysis
# figure. The other setting is traced only for that, never simulated or exported.
log.info(
    f"Tracing the weir with unprotected_ocean_wetlands={not unprotected_ocean_wetlands} "
    f"(diagnostic only, not simulated)"
)
_alt_weir_gdf, _ = build_coastal_protection_weir(
    grid,
    landuse_on_grid,
    channel_mask,
    weir_crest_m,
    min_component_cells,
    weir_par1,
    water_side_crest_on_grid=np.full(grid.shape, np.nan, dtype=np.float32),
    channel_mask_gap_free=True,
    unprotected_ocean_wetlands=not unprotected_ocean_wetlands,
)
_weir_off, _weir_on = (
    (_alt_weir_gdf, confinement_weir_gdf)
    if unprotected_ocean_wetlands
    else (confinement_weir_gdf, _alt_weir_gdf)
)
_weir_off.to_file(snakemake.output.dike_wetlands_off, driver="GPKG")
_weir_on.to_file(snakemake.output.dike_wetlands_on, driver="GPKG")
write_float_raster(
    snakemake.output.ocean_wetland_mask,
    ocean_linked_wetland_mask(landuse_on_grid, ocean_mask_grid, grid).astype(
        np.float32
    ),
    grid.transform,
    grid.crs,
)

# round 0: river bed from the water-surface slope (no SFINCS run)
round_label = ROUND_TITLES[0]
round_root = _start_round(0)
_set_weir(
    confinement_weir_gdf,
    confinement_weir_diagnostics,
    round_label,
    f"uniform confinement crest={weir_crest_m:.0f} m",
)
# The confinement weir stays on sf for round 1.

# Design discharge for the depth (GloFAS mean or bankfull at the seed
# crossings), routed through the network exactly as q_calib is; rule 10's
# surface and slope set the level and the depth.
river_bed, q_design, _seed_q, q_mean = estimate_river_bed_from_files(
    rivers,
    river_forcing_path,
    snakemake.input.water_surface,
    snakemake.input.sword_regions,
    wgs84_bounds,
    sf.crs,
    sea_level_m=calm_sea_level_m,
    gebco_min_depth_m=gebco_min_depth_m,
    manning_n=channel_manning_n,
    surface_cfg=water_surface_cfg,
    cfg=bed_cfg,
)
rivers["q_design_m3s"] = q_design
rivers["q_mean_m3s"] = q_mean
log.info(
    f"[{round_label}] Design discharge for the depth ({bed_cfg['design_discharge']}, GloFAS): seeds "
    f"{ {k: round(v) for k, v in _seed_q.items()} }, {int((q_design > 0).sum())}/{len(rivers)} reach(es) with Q > 0"
)

# Bed and level at the centreline cells, by along-reach distance.
_cx, _cy = cell_gdf["x"].to_numpy(), cell_gdf["y"].to_numpy()
rivbed_cell = values_at_points(
    river_bed.net, river_bed.bed_m, cell_gdf["reach_id"], _cx, _cy
)
round0_zs = values_at_points(
    river_bed.net, river_bed.level_m, cell_gdf["reach_id"], _cx, _cy
)
if depth_scale != 1.0:
    # Sensitivity runs only: scale the depth below the surface, not the surface.
    rivbed_cell = round0_zs - depth_scale * (round0_zs - rivbed_cell)
    log.warning(f"[{round_label}] Bed depth scaled by depth_scale = {depth_scale:g}")
_missing = ~np.isfinite(rivbed_cell)
if _missing.any():
    log.warning(
        f"[{round_label}] {int(_missing.sum())} centreline cell(s) without an estimated bed -- left at the DEM"
    )
    rivbed_cell = np.where(_missing, dem_at_cell, rivbed_cell)
rivdph_current = np.maximum(dem_at_cell - rivbed_cell, 0.0)
log.info(
    f"[{round_label}] Bed at {len(cell_gdf):,} centreline cell(s): depth below the conditioned DEM "
    f"median {np.nanmedian(rivdph_current):.2f} m, max {np.nanmax(rivdph_current):.2f} m"
)

# Node table (declared output) and its GeoPackage twin.
_bed_dir = Path(snakemake.output.river_bed_nodes).parent
_bed_dir.mkdir(parents=True, exist_ok=True)
_bed_table = river_bed.node_table()
_bed_table.to_csv(_bed_dir / f"{basin_id}_river_bed_nodes.csv", index=False)
gpd.GeoDataFrame(
    _bed_table,
    geometry=gpd.points_from_xy(_bed_table["x"], _bed_table["y"]),
    crs=sf.crs,
).to_file(_bed_dir / f"{basin_id}_river_bed_nodes.gpkg", driver="GPKG")
_per_reach = _bed_table.groupby("reach_id").agg(
    water_slope=("slope", "median"),
    water_level_m=("level_m", "median"),
    mean_flow_depth_m=("depth_m", "median"),
    depth_on_slope_floor=("on_floor", "mean"),
)
for _col in _per_reach.columns:
    rivers[_col] = rivers["reach_id"].map(normalize_reach_id).map(_per_reach[_col])
log.info(f"[{round_label}] River-bed diagnostics written to {_bed_dir}")

_update_reach_columns(rivdph_current)
# Round 0 is the estimated bed itself; its state carries that depth for the profiles.
_write_calibration_state(
    round_root,
    river_depth_m=rivdph_current,
    weir_crest=np.full(len(cell_gdf), np.nan),
    zs=round0_zs,
)

# River depth and forward check per node pair, and the model region, for the
# analysis figures of the network.
depth_segments(river_bed, crs=sf.crs).to_file(
    snakemake.output.depth_segments, driver="GPKG"
)
sf.region.to_crs(sf.crs).to_file(snakemake.output.model_region, driver="GPKG")

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
burned_arr_subgrid, transform_subgrid, terrain_subgrid, stats_subgrid = (
    burn_river_channel(
        rivers=rivers_utm,
        zbed_anchors=zbed_anchors,
        natural_dem_path=elevation_path,
        utm_crs=sf.crs,
        resolution_m=grid.cell_size_m / nr_subgrid_pixels,
        out_transform=subgrid_transform,
        out_shape=subgrid_shape,
    )
)
# A reach's buf_poly can rasterize edge pixels outside channel_mask.
burned_arr_subgrid = constrain_to_coarse_channel_mask(
    burned_arr_subgrid,
    transform_subgrid,
    sf.crs,
    channel_mask,
    grid.transform,
)
river_burned_subgrid_path.parent.mkdir(parents=True, exist_ok=True)
with rasterio.open(
    river_burned_subgrid_path,
    "w",
    driver="GTiff",
    dtype="float32",
    width=burned_arr_subgrid.shape[1],
    height=burned_arr_subgrid.shape[0],
    count=1,
    crs=sf.crs,
    transform=transform_subgrid,
    nodata=FLOAT_NODATA,
    compress="deflate",
    tiled=True,
) as dst:
    dst.write(
        np.where(np.isnan(burned_arr_subgrid), FLOAT_NODATA, burned_arr_subgrid).astype(
            np.float32
        ),
        1,
    )
log.info(
    f"Written: {river_burned_subgrid_path} ({stats_subgrid['n_reaches_burned']} reach(es) burned, "
    f"{int(np.isfinite(burned_arr_subgrid).sum()):,} subgrid pixel(s) @ "
    f"{grid.cell_size_m / nr_subgrid_pixels:g} m)"
)

# Main-grid dep of channel cells: per-cell MEAN of the burned subgrid, so a
# channel narrower than a cell lowers it in proportion to its area.
burned_arr_sfincs = mean_elevation_on_coarse_grid(
    burned_arr_subgrid,
    terrain_subgrid,
    nr_subgrid_pixels,
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

# calibration forcing for rounds 1 and 2: rivers ramped over ramp_hr, then held
# Sea: each station's storm-tide hydrograph peaking at its storm tide at the
# FLOPROS coastal RP (linear in RP between COAST-RP's tabulated RPs,
# MDT-corrected, rounded to 1 cm), set up below once the run length is known. The coast amplifies the boundary level (shoaling, reflection off the
# dike, bays): basin 2433835 reaches ~0.8 m at the dike under a 0.4 m boundary,
# so coastal crests come from the simulated level at the dike. River and surge
# both at protection RP is conservative near the mouth.
with xr.open_dataset(surge_forcing_path, decode_times=False) as _sds:
    _surge_stations = gpd.GeoDataFrame(
        {"index": range(_sds.sizes["station"])},
        geometry=gpd.points_from_xy(_sds["longitude"].values, _sds["latitude"].values),
        crs="EPSG:4326",
    )
    coastal_design_peaks = (
        np.asarray(
            storm_tide_at_rp_interpolated(_sds, surge_protection_rp_yr), dtype=float
        )
        if surge_protection_rp_yr is not None
        else np.full(_sds.sizes["station"], calm_sea_level_m)
    )

# Initial condition: river channel cells at the steady level of the mean
# discharge on round 0's bed, not dry: a dry channel fills with a transient
# wave that the run maximum would add to the riverbank crests, as it would for
# dry coastal cells. The sea starts calm, as its boundary does.
calib_sea_level_m = float(coastal_design_peaks.max())
_mean_level_nodes = gpd.GeoDataFrame(
    {"level_m": river_bed.mean_level_m},
    geometry=gpd.points_from_xy(river_bed.net.node_x, river_bed.net.node_y),
    crs=sf.crs,
)
_channel_mask = build_channel_mask_regular(
    rivers.to_crs(sf.crs),
    "width",
    _zsini_arr.shape,
    sf.grid.data["dep"].raster.transform,
)
_zsini_arr, _n_river_zsini = river_cells_at_surface(
    _zsini_arr,
    _mean_level_nodes[np.isfinite(_mean_level_nodes["level_m"])],
    _channel_mask,
    sf.grid.data["dep"].raster.transform,
)
# Nearest: zsini is near-binary and the grids are identical.
_da_zsini = xr.DataArray(
    _zsini_arr,
    dims=sf.grid.data["dep"].dims,
    coords=sf.grid.data["dep"].coords,
)
_da_zsini = _da_zsini.rio.write_crs(sf.grid.data["dep"].rio.crs)
_da_zsini = _da_zsini.rio.write_transform(sf.grid.data["dep"].raster.transform)
_da_zsini.raster.set_nodata(np.nan)
sf.initial_conditions.create(ini=_da_zsini, reproj_method="nearest")
log.info(
    f"Initial conditions: {_n_river_zsini:,} river channel cell(s) at the mean-discharge level; "
    f"sea at {calm_sea_level_m:+.2f} m; land dry"
)

# Rivers: each crossing from its reach's mean discharge (the flow that holds
# zsini's river level) to its calibration discharge.
reach_q = dict(zip(rivers["reach_id"].astype(str), q_calib))
reach_q_start = dict(zip(rivers["reach_id"].astype(str), rivers["q_mean_m3s"]))
with xr.open_dataset(river_forcing_path, decode_times=False) as river_ds:
    # Crossings on a seed that flows out of the network inject nothing.
    active = river_ds["has_glofas"].values.astype(bool) & ~outflow_crossing_mask(
        river_ds, rivers
    )
    inside_ids = (
        river_ds["inside_reach_id"].values[active]
        if "inside_reach_id" in river_ds
        else []
    )
    cross_lons = river_ds["longitude"].values[active]
    cross_lats = river_ds["latitude"].values[active]

crossing_q = [reach_q.get(str(reach_id), np.nan) for reach_id in inside_ids]
crossing_q_start = [reach_q_start.get(str(reach_id), np.nan) for reach_id in inside_ids]
crossings_gdf = gpd.GeoDataFrame(
    {"index": range(len(crossing_q))},
    geometry=gpd.points_from_xy(cross_lons, cross_lats),
    crs="EPSG:4326",
)
valid = np.isfinite(crossing_q)
crossings_gdf = crossings_gdf[valid].reset_index(drop=True)
crossing_q = np.asarray(crossing_q)[valid]
crossing_q_start = np.asarray(crossing_q_start, dtype=float)[valid]
inside_ids_valid = (
    np.asarray(inside_ids)[valid] if len(inside_ids) else [None] * len(crossing_q)
)

if crossings_gdf.empty:
    raise RuntimeError(
        "No discharge crossings resolved to a calibration discharge -- cannot calibrate"
    )

# Run length: the slowest walled travel time from a crossing to its network
# end at the calibration discharge on round 0's bed, sea at the highest
# station level (deeper, so slower, near the mouth).
_nodes = _bed_table.assign(reach_id=_bed_table["reach_id"].map(normalize_reach_id))
_node_reaches = set(_nodes["reach_id"])
_q_calib_by_reach = dict(
    zip(rivers["reach_id"].map(normalize_reach_id), np.asarray(q_calib, dtype=float))
)
_walled = []
for _reach in sorted({normalize_reach_id(r) for r in inside_ids_valid} - {None}):
    if _reach not in _node_reaches:
        log.warning(
            f"Crossing reach {_reach} has no river-bed nodes: no walled travel time"
        )
        continue
    _prof = mainstem_profile(_nodes, adjacency, _reach, reach_q=_q_calib_by_reach)
    _walled.append(
        walled_travel_time(
            _prof,
            _prof["reach_id"].map(_q_calib_by_reach).to_numpy(float),
            float(coastal_design_peaks.max()),
            channel_manning_n,
            forcing_hr=ramp_hr,
            regime_rT=wave_regime_rT,
        ).assign(crossing_reach_id=_reach, reach_id=_prof["reach_id"].to_numpy())
    )
if not _walled:
    raise RuntimeError(
        "No crossing reach has river-bed nodes -- cannot size the calibration run"
    )
walled = pd.concat(_walled, ignore_index=True)
walled.to_csv(_bed_dir / f"{basin_id}_walled_travel.csv", index=False)
_ends = walled.groupby("crossing_reach_id")[
    ["t_wave_hr", "t_kinematic_hr", "t_gravity_hr"]
].last()
t_walled_hr = float(_ends["t_wave_hr"].max())
run_hr = float(np.ceil(ramp_hr + t_walled_hr + calibration_margin_hr))
log.info(
    f"Walled travel time to the network end per crossing reach (regime / all-kinematic / all-gravity, h): "
    f"{ {r: (round(t, 1), round(k, 1), round(g, 1)) for r, (t, k, g) in _ends.iterrows()} }; "
    f"inertial nodes {int((walled['regime'] == 'inertial').sum())} of {len(walled)}; run = {ramp_hr:g} h ramp + "
    f"{t_walled_hr:.1f} h + {calibration_margin_hr:g} h margin = {run_hr:.0f} h"
)
if not np.isfinite(run_hr) or run_hr > calibration_max_days * 24.0:
    raise ValueError(
        f"calibration run of {run_hr} h exceeds calibration_max_days ({calibration_max_days:g} d): walled "
        f"travel time {t_walled_hr:.1f} h (see {_bed_dir / f'{basin_id}_walled_travel.csv'}) -- "
        f"raise river_depth_modelling.calibration_max_days or check the bed"
    )

tstop = tref + timedelta(hours=run_hr)
trstout_s = int(run_hr * 3600)
sf.config.update({"tstop": tstop, "dtmaxout": trstout_s, "trstout": trstout_s})

calib_times = pd.date_range(tref, tstop, periods=int(run_hr) + 1)
_ramp_frac = cosine_ramp(np.linspace(0.0, run_hr, len(calib_times)), 0.0, ramp_hr)[
    :, None
]
_start_q = np.where(np.isfinite(crossing_q_start), crossing_q_start, 0.0)
dis_values = _start_q[None, :] + (crossing_q[None, :] - _start_q[None, :]) * _ramp_frac
dis_df = pd.DataFrame(
    dis_values, index=calib_times, columns=crossings_gdf["index"].to_numpy()
)

# Snap to the crossing reach's centerline cell: an unsnapped point can land in
# a floodplain cell without conveyance (basin 4267691: 369 m water level in
# round 0). Own reach first, else nearest cell of any reach.
crossings_filt = place_discharge_points(
    sf, crossings_gdf, dis_df, cell_gdf, inside_ids_valid, resolution_m
)
dis_df = dis_df[crossings_filt["index"].to_numpy()]
log.info(
    f"Discharge forcing: {len(crossings_filt)} source point(s), cosine-ramped from the mean discharge to the "
    f"calibration discharge over {ramp_hr:g} h, held for the remaining {run_hr - ramp_hr:.0f} h"
)

# For round 2's crest gap map.
crossings_utm = crossings_filt.to_crs(grid.crs)
crossings_utm.to_file(snakemake.output.discharge_points, driver="GPKG")

sf.water_level.create_boundary_points_from_mask(bnd_dist=5000.0)
if surge_protection_rp_yr is not None:
    # The event runs' storm-tide wave, peaking when the calibration discharge
    # reaches the coast, so river and surge maxima coincide (conservative at
    # the mouth); blended in from the calm level, where the sea starts.
    # On a boundary sample (hourly), so the forcing reaches the peak level exactly.
    surge_peak_hr = float(np.round(ramp_hr + t_walled_hr))
    _surge_time_hr = np.arange(0.0, run_hr + 1e-9, 1.0)
    with xr.open_dataset(surge_forcing_path, decode_times=False) as _sds:
        _event = storm_tide_event(
            _sds,
            _surge_time_hr - surge_peak_hr,
            coastal_design_peaks,
            rp_yr=surge_protection_rp_yr,
        )
    _blend = (
        np.clip(_surge_time_hr / ramp_hr, 0.0, 1.0)
        if ramp_hr > 0
        else np.ones_like(_surge_time_hr)
    )
    _sea = np.round(calm_sea_level_m + _blend[None, :] * (_event - calm_sea_level_m), 2)
    sf.water_level.create(
        timeseries=pd.DataFrame(
            _sea.T,
            index=pd.DatetimeIndex(
                [tref + timedelta(hours=float(h)) for h in _surge_time_hr]
            ),
            columns=range(len(_surge_stations)),
        ),
        locations=_surge_stations,
        buffer=water_level_buffer_m,
        merge=False,
    )
    log.info(
        f"Rounds 1-2 coastal forcing: RP {surge_protection_rp_yr:.1f} storm-tide hydrograph at "
        f"{len(_surge_stations)} station(s), peaks {coastal_design_peaks.min():+.2f}..{coastal_design_peaks.max():+.2f} m "
        f"at {surge_peak_hr:.1f} h (river arrival), blended in from {calm_sea_level_m:+.2f} m over {ramp_hr:g} h"
    )
else:
    sf.water_level.create_timeseries(shape="constant", offset=calm_sea_level_m)
    log.warning(
        f"No coastal protection RP -- rounds 1-2 keep the steady calm-sea boundary at {calm_sea_level_m:+.4f} m"
    )


def _steady_check(
    round_label: str, final_zs: np.ndarray, final_times_s: np.ndarray
) -> None:
    window_hr = float(steady_check_cfg["window_hr"])
    tol_m = float(steady_check_cfg["tol_m"])
    i0 = int(np.searchsorted(final_times_s, final_times_s[-1] - window_hr * 3600.0))
    rise = final_zs[-1] - final_zs[i0]
    rising = np.isfinite(rise) & (rise > tol_m)
    msg = (
        f"[{round_label}] {int(rising.sum())}/{rise.size} centreline cell(s) rose more than {tol_m:g} m "
        f"over the run's last {window_hr:g} h (max {np.nanmax(rise):+.2f} m)"
    )
    if rising.any():
        log.warning(msg + " -- not steady at the run end: raise calibration_margin_hr")
    else:
        log.info(msg)


# round 1: confined, excavated
# sf.weirs still holds round 0's confinement weir -- unchanged here.
round_label = ROUND_TITLES[1]
round_root = _start_round(1)
_create_subgrid(excavated_subgrid_elevation_list, round_label)
if crosssections:
    # Discharge through each arm of the seeds' bifurcations (sfincs_his.nc),
    # against the width-proportional split of accumulate_discharge.
    _crs = bifurcation_cross_sections(
        rivers_utm,
        sorted(
            {
                normalize_reach_id(r)
                for r, s in zip(rivers["reach_id"], rivers["is_seed"])
                if bool(s)
            }
            - {None}
        ),
    )
    sf.cross_sections.set(_crs[["name", "geometry"]], merge=False)
    _crs.to_file(calib_root / "cross_sections.gpkg", driver="GPKG")
    log.info(f"[{round_label}] {len(_crs)} bifurcation cross-section(s) set")
period_max_zs_1, final_zs, final_times_s = _run_calibration_round(
    round_label, round_root
)
_steady_check(round_label, final_zs, final_times_s)
zs_field_1 = _read_period_max_zs_field(round_root)

# Production weir, exported as-is; round 2 verifies exactly this weir.
weir_gdf, final_weir_diagnostics = build_coastal_protection_weir(
    grid,
    landuse_on_grid,
    channel_mask,
    coastal_protection_crest_m,
    min_component_cells,
    weir_par1,
    water_side_crest_on_grid=zs_field_1,
    freeboard_m=weir_freeboard_m,
    channel_mask_gap_free=True,
    unprotected_ocean_wetlands=unprotected_ocean_wetlands,
)
# SFINCS drops a weir face whose crest is not above the face's subgrid
# minimum (uv_zmin); floor every crest there so all runs keep round 1's dike.
# Round 2 and rule 12a use the same subgrid table, so its faces apply there.
if final_weir_diagnostics.get("applicable", True) and not weir_gdf.empty:
    _u_zmin, _v_zmin = read_face_zmin(
        round_root / "sfincs_subgrid.nc", sf.grid.data["mask"]
    )
    weir_gdf, _floored_surface, _n_raised = floor_crests_above_face_zmin(
        weir_gdf,
        _u_zmin,
        _v_zmin,
        final_weir_diagnostics["edge_water_side_mask"],
        grid.transform,
    )
    final_weir_diagnostics["crest_surface"] = np.where(
        np.isfinite(_floored_surface),
        _floored_surface,
        final_weir_diagnostics["crest_surface"],
    )
    log.info(
        f"[{round_label}] {_n_raised} of {len(weir_gdf)} weir edge crest(s) raised to the face's subgrid "
        f"minimum (uv_zmin) + {CREST_PRECISION_M * 100:g} cm, so SFINCS keeps every weir face"
    )

# Per-centerline-cell summary of the same rule; reporting only.
weir_crest_cell = ceil_crest(
    np.fmax(period_max_zs_1, coastal_protection_crest_m) + weir_freeboard_m
)
log.info(
    f"[{round_label}] Crest (centerline summary): min={np.nanmin(weir_crest_cell):.3f} m, "
    f"max={np.nanmax(weir_crest_cell):.3f} m, median={np.nanmedian(weir_crest_cell):.3f} m "
    f"(+{weir_freeboard_m:.2f} m freeboard)"
)
_update_reach_columns(rivdph_current, weir_crest_cell)
_write_calibration_state(
    round_root,
    river_depth_m=rivdph_current,
    weir_crest=weir_crest_cell,
    zs=period_max_zs_1,
)
if verify_crests:
    _round_diagnostics(1, round_root, final_zs, final_times_s, confinement_weir_gdf)
else:
    # Round 1 is the last round: its crest gap map shows the production crests
    # against round 1's own water level.
    _round_diagnostics(
        1,
        round_root,
        final_zs,
        final_times_s,
        weir_gdf,
        crest_gap_crest_on_grid=_edge_crest_on_grid(final_weir_diagnostics),
    )
    # A previous run's round 2 would sit next to this round 1 as if current.
    shutil.rmtree(calib_root / "round2", ignore_errors=True)
    log.info(f"[{ROUND_TITLES[2]}] skipped (verify_crests: false)")

# round 2: verification with the real production weir
if verify_crests:
    round_label = ROUND_TITLES[2]
    round_root = _start_round(2)
    _set_weir(
        weir_gdf,
        final_weir_diagnostics,
        round_label,
        "production crests (per edge, from round 1)",
    )
    _create_subgrid(excavated_subgrid_elevation_list, round_label)
    period_max_zs_2, final_zs, final_times_s = _run_calibration_round(
        round_label, round_root
    )
    _steady_check(round_label, final_zs, final_times_s)

    edge_crest_on_grid = _edge_crest_on_grid(final_weir_diagnostics)
    if edge_crest_on_grid is not None:
        zs_field_2 = _read_period_max_zs_field(round_root)
        _edge_mask = final_weir_diagnostics["edge_water_side_mask"]
        # NaN (never-wetted) water side compares False -- nothing to overtop there.
        _exceedance = zs_field_2[_edge_mask] - edge_crest_on_grid[_edge_mask]
        _n_overtopped = int((_exceedance > CREST_PRECISION_M).sum())
        if _n_overtopped:
            log.warning(
                f"[{round_label}] {_n_overtopped} of {int(_edge_mask.sum()):,} edge water-side cell(s) exceed "
                f"their own crest by > {CREST_PRECISION_M * 100:g} cm "
                f"(max exceedance {np.nanmax(_exceedance):.3f} m) -- see the crest gap map; "
                f"crests are NOT adjusted (verification only)"
            )
        else:
            log.info(
                f"[{round_label}] Verified: no edge water-side cell exceeds its own crest by > {CREST_PRECISION_M * 100:g} cm "
                f"({int(_edge_mask.sum()):,} cell(s) checked)"
            )
    _write_calibration_state(
        round_root,
        river_depth_m=rivdph_current,
        weir_crest=weir_crest_cell,
        zs=period_max_zs_2,
    )
    _round_diagnostics(
        2,
        round_root,
        final_zs,
        final_times_s,
        weir_gdf,
        crest_gap_crest_on_grid=edge_crest_on_grid,
    )

# canonical production outputs: burned river channel + weir
with rasterio.open(
    snakemake.output.river_burned_dem_sfincs_grid,
    "w",
    driver="GTiff",
    dtype="float32",
    width=burned_arr_sfincs.shape[1],
    height=burned_arr_sfincs.shape[0],
    count=1,
    crs=sf.crs,
    transform=grid.transform,
    nodata=FLOAT_NODATA,
    compress="deflate",
) as dst:
    dst.write(
        np.where(np.isnan(burned_arr_sfincs), FLOAT_NODATA, burned_arr_sfincs).astype(
            np.float32
        ),
        1,
    )
log.info(
    f"Written: {snakemake.output.river_burned_dem_sfincs_grid} "
    f"({int(np.isfinite(burned_arr_sfincs).sum()):,} channel cell(s))"
)

Path(snakemake.output.coastal_protection_weir).parent.mkdir(parents=True, exist_ok=True)
weir_gdf.to_file(snakemake.output.coastal_protection_weir, driver="GPKG")
log.info(
    f"Written: {snakemake.output.coastal_protection_weir} ({len(weir_gdf)} segment(s))"
)

# Real open sea on this rule's grid, no reprojection
# Mostly a GRID-RESOLUTION MISALIGNMENT fix, not an isolated-pocket one --
# the weir is traced on landuse_on_grid (rule grid_align_landuse, 09b:
# native landuse.tif resampled by mode onto the much coarser
# SFINCS regular grid, ONCE, upstream), so the protected side disagrees with
# native landuse.tif along the whole coast where coarse cell edges cut the
# coastline (basin 2433835 has one connected sea component at native
# resolution: a coastline-wide fringe, not discrete pockets).
# The single sea/land classification for rule 12a's zsini (uncorrected, fringe
# cells would start wet behind the weir) and for flood diagnostics
# (src.postprocessing sea_mask_path), which reproject it onto subgrid- or
# cell-resolution grids; that is alignment-safe only from a grid-aligned source.
_protected_pocket_mask = final_weir_diagnostics.get(
    "protected_pocket_mask", np.zeros(landuse_on_grid.shape, dtype=bool)
)
_sea_cells_on_grid = (landuse_on_grid == LANDUSE_SEA_CODE) & ~_protected_pocket_mask
_sea_cells_arr = np.where(_sea_cells_on_grid, np.float32(1.0), np.float32(FLOAT_NODATA))
_sea_cells_profile = {
    "driver": "GTiff",
    "height": _sea_cells_arr.shape[0],
    "width": _sea_cells_arr.shape[1],
    "count": 1,
    "dtype": "float32",
    "crs": grid.crs,
    "transform": grid.transform,
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

# seed-to-mouth profiles: DEM, bed and crest of the last round, from each
# round's calibration_state.csv. The per-round table goes next to the bed nodes.
_profile_dir = Path(snakemake.output.river_bed_nodes).parent.parent
seed_ids = sorted(
    {
        normalize_reach_id(reach_id)
        for reach_id, s in zip(rivers["reach_id"], rivers["is_seed"])
        if bool(s) is True
    }
    - {None}
)
if not seed_ids:
    log.warning("No is_seed reach found -- the crest profile figure has no panel")
_crest_profiles = {}
for _seed in seed_ids:
    _path_reach_ids, _profiles_by_round = gather_calibration_round_profile(
        calib_root, rivers_utm, _seed, N_ROUNDS_AFTER_0
    )
    _crest_profiles[_seed] = _profiles_by_round[N_ROUNDS_AFTER_0]
    pd.concat(
        [df.assign(round=i) for i, df in _profiles_by_round.items()], ignore_index=True
    ).to_csv(_profile_dir / f"round_profiles_seed{_seed}.csv", index=False)
    log.info(
        f"Seed {_seed}: profile written (path of {len(_path_reach_ids)} reach(es))"
    )
plot_crest_profiles(basin_id, _crest_profiles, snakemake.output.plot_crest_profile)

n_missing = rivers["river_depth_m"].isna().sum()
if n_missing:
    log.warning(
        f"{n_missing} reach(es) missing a calibrated depth (no centerline cell?) -- left as NaN"
    )

Path(snakemake.output.depth_estimated_river_network).parent.mkdir(
    parents=True, exist_ok=True
)
rivers.to_file(snakemake.output.depth_estimated_river_network, driver="GPKG")
log.info(f"Written: {snakemake.output.depth_estimated_river_network}")

# Run length and round titles, for the analysis figures.
with open(snakemake.output.calibration_rounds, "w") as _f:
    json.dump(
        {"run_hr": run_hr, "n_rounds": N_ROUNDS_AFTER_0, "titles": ROUND_TITLES},
        _f,
        indent=2,
    )
log.info("Done")
