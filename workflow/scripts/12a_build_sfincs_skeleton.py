"""Build the scenario-independent SFINCS skeleton of a basin.

Run by rule build_sfincs_skeleton, once per basin: grid, elevation, masks
(active, water-level and outflow boundary), coastal protection weir,
roughness, subgrid table, observation points and the default zsini.tif.

The skeleton sfincs.inp has no forcing and no tref/tstart/tstop, so it is not
runnable; build_sfincs and run_spinup reference its geometry files by relative
path. Every grid input is already on this model's grid (resampled once
upstream), so nothing is reprojected here; zsini is built from
zsini_sea_cells_on_grid.tif (sea = calm_sea_level_m) and rule 11's steady
mean-discharge level on the bed (river channel cells); land = nodata.
"""

import json
import logging
from pathlib import Path

import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from rasterio.transform import Affine
from shapely.geometry import Polygon
from hydromt_sfincs import SfincsModel

from src.grid import create_model_grid, load_grid_def
from src.hydromt_sfincs_patch import patch_nearest_extrapolation, patch_subgrid_write
from src.landuse import LANDUSE_SEA_CODE
from src.log import setup_logging
from src.sfincs_run import create_active_mask, set_sea_and_outflow_boundaries, write_weir_file
from src.plots import add_land_background_to_geoaxes, imshow_on_grid, save_figure
from src.protection_weir import GridArrays
from src.raster import FLOAT_NODATA, dry_isolated_sea_cells, read_on_grid
from src.river_burn import build_channel_mask_regular
from src.river_network import normalize_reach_id
from src.surge import read_calm_sea_level_m
from src.water_surface import river_cells_at_surface

plt.ioff()

# ── logging ──
setup_logging(snakemake.log[0], verbose=True)
log = logging.getLogger(__name__)
patch_subgrid_write()
patch_nearest_extrapolation()
for _name in ("hydromt", "hydromt_sfincs"):
    _l = logging.getLogger(_name)
    _l.setLevel(logging.INFO)
    _l.handlers = []
    _l.addHandler(logging.FileHandler(snakemake.log[0]))

# ── paths & params ──
domain_path           = Path(snakemake.input.domain_gpkg)
# Native-resolution conditioned (post-monotonicity) elevation.
elevation_conditioned_path = Path(snakemake.input.elevation_conditioned)
# Native resolution, subgrid table only; the regular-grid manning field and
# weir diagnostics use the grid-aligned rasters (rule grid_align_landuse).
roughness_path        = Path(snakemake.input.roughness)
roughness_on_grid_path = Path(snakemake.input.roughness_on_grid)
landuse_on_grid_path  = Path(snakemake.input.landuse_on_grid)
# Plot background only.
land_mask_path        = Path(snakemake.input.land_mask_on_grid)
river_network_path    = Path(snakemake.input.river_network)
delta_outflow_points_path = Path(snakemake.input.delta_outflow_points)
# The only sea/land classification zsini is built from (rule modelled_depth_estimation).
zsini_sea_cells_path = Path(snakemake.input.zsini_sea_cells)
surge_forcing_path    = Path(snakemake.input.surge_forcing)
river_forcing_path    = Path(snakemake.input.river_forcing)
river_burned_subgrid_path = Path(snakemake.input.river_burned_subgrid)
river_burned_dem_sfincs_grid_path = Path(snakemake.input.river_burned_dem_sfincs_grid)
elevation_conditioned_sfincs_grid_path = Path(snakemake.input.elevation_conditioned_sfincs_grid)
river_elevation_max_path = Path(snakemake.input.river_elevation_max)
coastal_protection_weir_path = Path(snakemake.input.coastal_protection_weir)

inputs_dir        = Path(snakemake.params.inputs_dir)
skeleton_root     = Path(snakemake.params.skeleton_root)
resolution_m        = snakemake.params.resolution_m
nr_subgrid_pixels = snakemake.params.nr_subgrid_pixels
nr_levels         = snakemake.params.nr_levels
nrmax             = snakemake.params.nrmax
active_mask_enabled = snakemake.params.active_mask_enabled
active_mask_elevation_buffer_m = float(snakemake.params.active_mask_elevation_buffer_m)
outflow_buffer_m   = snakemake.params.outflow_buffer_m
n_top_crossings    = snakemake.params.n_top_crossings
n_per_crossing     = snakemake.params.n_per_crossing
max_downstream_hops = snakemake.params.max_downstream_hops

skeleton_root.mkdir(parents=True, exist_ok=True)

# ── baseline water level from surge forcing ──
# calm_sea_level_m (mean vertical correction, rounded to 1 cm) and
# coastal_protection_crest_m (FLOPROS coastal standard) are basin-level, not
# derived from any design RP.
with xr.open_dataset(surge_forcing_path, decode_times=False) as _ds:
    calm_sea_level_m = read_calm_sea_level_m(_ds)
    coastal_protection_crest_m = (
        float(_ds["coastal_protection_crest_m"].values)
        if "coastal_protection_crest_m" in _ds else 0.0
    )
log.info(f"Surge boundary baseline read from surge_forcing.nc: {calm_sea_level_m:+.2f} m (rounded to 1 cm)")

_zsini_out = Path(snakemake.output.zsini)

# ── delta-outline outflow points ──
delta_outflow_gdf = gpd.read_file(delta_outflow_points_path)
delta_outflow_enabled = not delta_outflow_gdf.empty
log.info(f"Delta-outline outflow points: {len(delta_outflow_gdf)}")

# ── data catalog ──
local_catalog_path = skeleton_root / "data_catalog_local.yml"
local_catalog = {
    "meta": {"root": str(inputs_dir)},
    "local_elevation_conditioned": {
        "data_type": "RasterDataset",
        "uri": str(elevation_conditioned_path),
        "driver": "rasterio",
    },
    "local_river_burned_subgrid": {
        "data_type": "RasterDataset",
        "uri": str(river_burned_subgrid_path),
        "driver": "rasterio",
    },
    "local_river_burned_sfincs_grid": {
        "data_type": "RasterDataset",
        "uri": str(river_burned_dem_sfincs_grid_path),
        "driver": "rasterio",
    },
    "local_elevation_conditioned_sfincs_grid": {
        "data_type": "RasterDataset",
        "uri": str(elevation_conditioned_sfincs_grid_path),
        "driver": "rasterio",
    },
    # On the model grid; regular-grid "manning" field only, never subgrid.
    "local_roughness": {
        "data_type": "RasterDataset",
        "uri": str(roughness_on_grid_path),
        "driver": "rasterio",
    },
    # Native resolution: the subgrid table needs sub-cell roughness detail.
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
}
with open(local_catalog_path, "w") as fh:
    yaml.dump(local_catalog, fh, sort_keys=False)
log.info(f"Data catalog written: {local_catalog_path}")

# First source wins, later sources fill gaps: the burned channel (rule
# modelled_depth_estimation) over the conditioned DEM, at each consumer's
# resolution so nothing is reprojected. Main "dep": per-cell mean of the burned
# subgrid pixels over rule 10's grid-resolution DEM, the same dep the
# calibration rounds 1-2 ran on. Subgrid: the exact burned raster calibration
# used, on this model's subgrid pixel grid (checked below), over the native DEM.
elevation_list_main = [
    {"elevation": "local_river_burned_sfincs_grid"}, {"elevation": "local_elevation_conditioned_sfincs_grid"},
]
elevation_list_subgrid = [{"elevation": "local_river_burned_subgrid"}, {"elevation": "local_elevation_conditioned"}]

# ── domain boundary ──
delta_domain = gpd.read_file(domain_path)
log.info(f"Domain boundary: {len(delta_domain)} feature(s), CRS={delta_domain.crs}")

# ── initialise model ──
sf = SfincsModel(
    data_libs=[str(local_catalog_path)],
    root=str(skeleton_root),
    mode="w+",
    write_gis=True,
)
log.info("SfincsModel initialised")

# ── 1. Grid ──
create_model_grid(sf, load_grid_def(snakemake.input.sfincs_grid))
log.info(
    f"Grid created from the shared grid definition: {resolution_m} m"
)

# ── 2. Elevation ──
elevation_component = sf.elevation
elevation_component.create(
    elevation_list=elevation_list_main,
)
log.info(f"Elevation set from {[e['elevation'] for e in elevation_list_main]}")

# river_burned_subgrid.tif must be exactly this model's subgrid pixel grid so
# sf.subgrid.create() reads it without resampling and the channel matches the
# calibrated one.
_expected_transform = sf.grid.data["dep"].raster.transform * Affine.scale(1.0 / nr_subgrid_pixels)
_expected_shape = tuple(int(n) * int(nr_subgrid_pixels) for n in sf.grid.data["dep"].shape)
with rasterio.open(river_burned_subgrid_path) as _src:
    _burn_transform, _burn_shape = _src.transform, (_src.height, _src.width)
if _burn_shape != _expected_shape or not _burn_transform.almost_equals(_expected_transform, precision=1e-6):
    raise ValueError(
        f"{river_burned_subgrid_path.name} is not on this model's own subgrid pixel grid: "
        f"shape {_burn_shape} vs expected {_expected_shape}, transform {tuple(_burn_transform)[:6]} "
        f"vs expected {tuple(_expected_transform)[:6]} -- rerun rule modelled_depth_estimation "
        f"(same grid resolution and nr_subgrid_pixels as this rule)"
    )

# ── 3. Mask: active cells ──
river_elevation_max_m = None
if active_mask_enabled:
    with open(river_elevation_max_path) as f:
        river_elevation_max_m = float(json.load(f)["river_elevation_max_m"])
create_active_mask(sf, delta_domain, river_elevation_max_m, active_mask_elevation_buffer_m)

# ── 4. Mask: water-level boundary (open sea only, as in rule 11) and outflow ──
with rasterio.open(landuse_on_grid_path) as _lu_src_bnd:
    _landuse_bnd = _lu_src_bnd.read(1)
set_sea_and_outflow_boundaries(
    sf, _landuse_bnd, len(delta_outflow_gdf) if delta_outflow_enabled else 0, outflow_buffer_m
)

# ── 4c. Coastal protection weir ──
rivers = gpd.read_file(river_network_path)

_domain_wgs84 = delta_domain.to_crs("EPSG:4326")
_domain_union = _domain_wgs84.geometry.union_all()
_domain_poly_4326 = _domain_union if isinstance(_domain_union, Polygon) else _domain_union.convex_hull

rivers_utm = rivers.to_crs(sf.crs)
weir_grid = GridArrays.from_regular(sf.grid.data["dep"], sf.grid.data["mask"], sf.crs)
weir_gdf = gpd.read_file(coastal_protection_weir_path)
if not weir_gdf.empty:
    sf.weirs.set(weir_gdf, merge=False)
    log.info(f"Coastal protection weir imported directly from rule 11: {len(weir_gdf)} segment(s)")
else:
    log.info("Coastal protection weir: rule 11's own file has no segments -- sf.weirs left empty")
with rasterio.open(landuse_on_grid_path) as _lu_src:
    landuse_on_grid = _lu_src.read(1)
river_channel_mask = build_channel_mask_regular(rivers_utm, "width", weir_grid.shape, weir_grid.transform)
weir_diagnostics = {
    "ocean_mask": landuse_on_grid == LANDUSE_SEA_CODE,
    "river_channel_mask": river_channel_mask,
    "weir_lines": list(weir_gdf.geometry),
    "crest_elevation_m": coastal_protection_crest_m,
}

Path(snakemake.output.weir_gpkg).parent.mkdir(parents=True, exist_ok=True)
# An empty weir keeps rule 11's schema and CRS, so the GeoPackage stays valid.
weir_gdf.to_file(snakemake.output.weir_gpkg, driver="GPKG")
log.info(f"Weir GeoPackage written: {snakemake.output.weir_gpkg} ({len(weir_gdf)} segment(s))")

from src.plots import plot_coastal_protection_weir
plot_coastal_protection_weir(
    weir_grid, weir_diagnostics, _domain_poly_4326,
    str(land_mask_path), str(river_network_path),
    str(snakemake.output.plot_coastal_protection_weir),
    basin_id=skeleton_root.parent.name, weir_gdf=weir_gdf,
)

# ── 5. Initial conditions (default; river_only skips the inifile) ──
# "ini" is part of the grid Dataset and is written by sf.write(), so
# build_sfincs forwards it as "inifile" like any geometry file. A DataArray on
# sf.grid.data["dep"]'s own coords makes .create()'s reproject_like a no-op;
# resampling the native-resolution sea mask instead leaves coastal-fringe
# cells wet (two nearest-neighbour passes do not invert each other).
initial_conditions_component = sf.initial_conditions
# Same domain_gpkg + grid resolution feed both rules, so the grids match.
_sea_cells_on_grid = read_on_grid(zsini_sea_cells_path, sf.grid.data["dep"].shape)
# Dry cells are np.nan, not -9999: xarray ops such as .where() do not
# reliably preserve rio/nodata attrs across versions.
_ini_coarse = np.where(_sea_cells_on_grid == np.float32(1.0), np.float32(calm_sea_level_m), np.nan)

# ── dry out isolated sea cells ──
# Sea pockets not connected to the open ocean (shallow links dried by a
# negative calm_sea_level_m, or inland lagoons at any calm_sea_level_m) would hold trapped
# water. Weir-protected pockets are already excluded upstream. Isolated wet
# cells (dep < calm_sea_level_m) start dry at bed level.
_ini_coarse, _n_fix_coarse = dry_isolated_sea_cells(
    _ini_coarse,
    sf.grid.data["dep"].values,
    sf.grid.data["dep"].rio.nodata or FLOAT_NODATA,
    calm_sea_level_m,
)
if _n_fix_coarse > 0:
    log.info(
        f"zsini connectivity fix: {_n_fix_coarse} isolated wet sea-cell(s) set to dep "
        f"(not reachable from domain boundary at calm_sea_level_m={calm_sea_level_m:+.4f} m)"
    )
else:
    log.info("zsini connectivity fix: no isolated wet sea cells found")

# River channel cells start at rule 11's steady mean-discharge level on the
# bed, the level the spin-up's mean river flow holds, instead of dry. The node
# table is in the model CRS (rule 11 shares this grid definition).
_bed_nodes = pd.read_csv(snakemake.input.river_bed_nodes, usecols=["x", "y", "mean_level_m"])
_bed_nodes = _bed_nodes[np.isfinite(_bed_nodes["mean_level_m"])]
_ini_coarse, _n_river_coarse = river_cells_at_surface(
    _ini_coarse,
    gpd.GeoDataFrame(
        {"level_m": _bed_nodes["mean_level_m"].to_numpy()},
        geometry=gpd.points_from_xy(_bed_nodes["x"], _bed_nodes["y"]),
        crs=sf.crs,
    ),
    river_channel_mask,
    sf.grid.data["dep"].raster.transform,
)

# "nearest", not the default "average": zsini is near-binary (calm_sea_level_m at
# sea, NaN on land); grids are identical, so the resample is a no-op anyway.
_da_ini_coarse = xr.DataArray(
    _ini_coarse, dims=sf.grid.data["dep"].dims, coords=sf.grid.data["dep"].coords,
)
_da_ini_coarse = _da_ini_coarse.rio.write_crs(sf.grid.data["dep"].rio.crs)
_da_ini_coarse = _da_ini_coarse.rio.write_transform(sf.grid.data["dep"].raster.transform)
_da_ini_coarse.raster.set_nodata(np.nan)

initial_conditions_component.create(ini=_da_ini_coarse, reproj_method="nearest")
_n_wet_coarse = int((sf.grid.data["ini"].values > -9998.0).sum())
log.info(
    f"Default initial conditions set: sea = {calm_sea_level_m:+.4f} m, {_n_river_coarse:,} river channel "
    f"cell(s) at the mean-discharge level, land = -9999 (bed level / dry) "
    f"-- {_n_wet_coarse:,} wet cell(s) at this model's own grid resolution"
)

# zsini.tif mirrors sf.grid.data["ini"] exactly.
_zsini_meta_coarse = {
    "driver": "GTiff", "dtype": "float32", "count": 1,
    "height": _ini_coarse.shape[0], "width": _ini_coarse.shape[1],
    "transform": sf.grid.data["dep"].raster.transform, "crs": sf.grid.data["dep"].rio.crs,
    "nodata": np.float32(FLOAT_NODATA), "compress": "deflate",
}
with rasterio.open(_zsini_out, "w", **_zsini_meta_coarse) as _dst:
    _dst.write(np.where(np.isnan(_ini_coarse), np.float32(FLOAT_NODATA), _ini_coarse).astype(np.float32), 1)
log.info(f"zsini written: {_zsini_out} (sea cells = {calm_sea_level_m:+.4f} m, river channel cells = water surface)")

# ── 6. Roughness ──
roughness_component = sf.roughness
roughness_component.create(
    roughness_list=[{"manning": "local_roughness"}],
)
log.info("Manning roughness set from 'local_roughness'")

# ── 7. Subgrid table ──
rivers["rivwth"] = rivers["width"].fillna(1.0).astype(float)
rivers["river_depth_m"] = rivers["river_depth_m"].clip(lower=0.0).astype(float)
river_list = []

subgrid_dir = skeleton_root / "subgrid"
if subgrid_dir.exists():
    for _stale in subgrid_dir.glob("*subgrid*.tif"):
        _stale.unlink()

log.info(
    f"River network for subgrid: {len(rivers)} reaches, "
    f"rivwth [{rivers['rivwth'].min():.1f}–{rivers['rivwth'].max():.1f} m], "
    f"river_depth_m [{rivers['river_depth_m'].min():.2f}–{rivers['river_depth_m'].max():.2f} m]"
)
subgrid_component = sf.subgrid
subgrid_component.create(
    elevation_list=elevation_list_subgrid,
    # The channel's channel_manning_n is already in roughness.tif (rule 09).
    roughness_list=[{"manning": "local_roughness_native"}],
    river_list=river_list,
    nr_subgrid_pixels=nr_subgrid_pixels,
    nr_levels=nr_levels,
    write_dep_tif=True,
    write_man_tif=True,
    nrmax=nrmax,
)
log.info(
    f"Subgrid table created: {nr_subgrid_pixels} px/cell → "
    f"{resolution_m / nr_subgrid_pixels:.0f} m effective resolution"
)

# ── 8. Observation points ──
# N_OBS_PER_CROSSING evenly spaced points along the downstream path (up to
# max_downstream_hops reaches) of the N_TOP_CROSSINGS crossings with the
# highest bankfull discharge; independent of any design RP.
import re as _re

river_ds = xr.open_dataset(river_forcing_path, decode_times=False)
active = river_ds.has_glofas.values.astype(bool)
n_active = int(active.sum())
log.info(f"River forcing: {n_active}/{len(active)} crossings with valid GloFAS data (for observation points)")

N_OBS_PER_CROSSING = n_per_crossing
N_TOP_CROSSINGS    = n_top_crossings


obs_points_list = []

if n_active >= 1 and not rivers.empty:
    bankfull_vals = river_ds.bankfull_discharge_m3s.values.copy()
    bankfull_vals[~active] = -np.inf
    src_lons_all  = river_ds.longitude.values
    src_lats_all  = river_ds.latitude.values

    top_indices = np.argsort(bankfull_vals)[::-1][:N_TOP_CROSSINGS]
    top_indices = [i for i in top_indices if bankfull_vals[i] > 0]

    centroids_utm  = rivers_utm.geometry.centroid
    reach_lookup   = {normalize_reach_id(r["reach_id"]): r
                      for _, r in rivers_utm.iterrows()
                      if normalize_reach_id(r.get("reach_id"))}

    obs_id = 1
    for rank, ci in enumerate(top_indices):
        src_pt = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([src_lons_all[ci]], [src_lats_all[ci]]),
            crs="EPSG:4326",
        ).to_crs(sf.crs).geometry.iloc[0]

        nearest_reach_id = normalize_reach_id(
            rivers_utm.iloc[centroids_utm.distance(src_pt).idxmin()]["reach_id"]
        )

        dn_rows, current, visited = [], nearest_reach_id, set()
        for _ in range(max_downstream_hops):
            if not current or current in visited or current not in reach_lookup:
                break
            visited.add(current)
            row = reach_lookup[current]
            dn_rows.append(row)
            dn_raw = str(row.get("rch_id_dn", "") or "").strip().strip("[]")
            if not dn_raw or dn_raw.lower() in ("nan", "none", "<na>"):
                break
            nxt = next((normalize_reach_id(t.strip()) for t in _re.split(r"[,\s]+", dn_raw)
                        if normalize_reach_id(t.strip())), None)
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

# ── write ──
# No forcing component was created, so sf.write() writes none.
sf.write()
write_weir_file(sf)  # crests at 1 cm, not hydromt_sfincs' 0.1 m
log.info(f"Skeleton written to {skeleton_root}")

subgrid_path = Path(snakemake.output.sfincs_subgrid)
if not subgrid_path.exists():
    subgrid_path.touch()
    log.info("sfincs_subgrid.nc placeholder written (subgrid not built)")

weir_path = Path(snakemake.output.sfincs_weir)
if not weir_path.exists():
    weir_path.touch()
    log.info("sfincs.weir placeholder written (no weir set)")

# ── diagnostic plots ──
figs_dir = Path(snakemake.output.plot_grid).parent
figs_dir.mkdir(parents=True, exist_ok=True)


def _save_with_buffer(fig, ax, fname, buffer_frac=0.15):
    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()
    dx, dy = (xr - xl) * buffer_frac, (yt - yb) * buffer_frac
    ax.set_xlim(xl - dx, xr + dx)
    ax.set_ylim(yb - dy, yt + dy)
    save_figure(fig, figs_dir / fname, dpi=150, bbox_inches="tight")
    log.info(f"Plot written: {fname}")


# 1. Grid extent
fig, ax = sf.plot_basemap(variable="grid", plot_region=True, plot_geoms=False)
add_land_background_to_geoaxes(ax, str(land_mask_path), sf.crs)
_save_with_buffer(fig, ax, "01_grid.png")

# 2. Elevation; obs points shown here only (plot_geoms defaults to True)
fig, ax = sf.plot_basemap(variable="dep", vmin=-80, vmax=80)
add_land_background_to_geoaxes(ax, str(land_mask_path), sf.crs)
_save_with_buffer(fig, ax, "02_elevation.png")

# 3. Mask
fig, ax = sf.plot_basemap(variable="mask", plot_bounds=False, plot_geoms=False)
add_land_background_to_geoaxes(ax, str(land_mask_path), sf.crs)
_save_with_buffer(fig, ax, "03_mask.png")

# 4. Roughness
fig, ax = sf.plot_basemap(variable="manning", plot_bounds=False, plot_geoms=False)
add_land_background_to_geoaxes(ax, str(land_mask_path), sf.crs)
_save_with_buffer(fig, ax, "04_roughness.png")

# 5. Initial conditions (zsini); untracked side-effect plot, not a rule output.
with rasterio.open(_zsini_out) as _src:
    _zsini_arr = _src.read(1).astype(np.float32)
    _nodata_val = _src.nodata
    _zsini_transform = _src.transform

if _nodata_val is not None:
    _zsini_arr = np.where(
        np.isclose(_zsini_arr, np.float32(_nodata_val)), np.nan, _zsini_arr
    )

fig, ax = plt.subplots(figsize=(8, 6))
im = imshow_on_grid(ax, _zsini_arr, _zsini_transform, cmap="Blues")
plt.colorbar(im, ax=ax, label="Initial water level (m)")
ax.set_title(
    f"Initial conditions (zsini)\n"
    f"sea = {calm_sea_level_m:+.4f} m  |  land = nodata"
)
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
_save_with_buffer(fig, ax, "06_zsini.png")
