"""
10_depth_estimation_empirical.py -- empirical (formula-driven, no
simulation) river channel depth estimate: power-law (Leopold-Maddock)
hydraulic depth, optionally refined with the Nienhuis/O'Brien estuarine
depth model near the coast. river_processing.depth_method == "empirical"
only -- see rule modelled_depth_estimation (10_depth_estimation_modelled.py)
for the SFINCS-based alternative. Both write the same unified output file
(river_network_depth_estimated.gpkg) and the same unified burned-river-DEM
filenames (river_burned_dem.tif, river_burned_dem_sfincs_grid.tif): the
modelled alternative produces these directly from its own per-cell
calibrated bed every round, and this sibling rule produces them directly
too (from its own rivdph, via the same burn_river_channel excavation, not
hydromt's own gdf_zb/burn_river_rect), so rule 13 (production) never needs
to care which depth_method ran -- it just imports whichever sibling
produced these files.

The estuarine blend never runs independently of the power-law depth step,
so both live in this single script.
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import Affine

from src.domain import load_domain
from src.estuarine_depth import (
    compute_estuarine_depths,
    enforce_mouth_depth_monotonic,
    load_nienhuis,
    match_basin_to_delta,
)
from src.log import setup_logging
from src.plots import (
    plot_hydraulic_relations_with_estuarine,
    plot_river_depth,
    plot_river_network_width_discharge,
)
from src.profiling import ScriptProfiler
from src.river_burn import build_channel_mask_regular, burn_river_channel, constrain_to_coarse_channel_mask
from src.river_network import compute_hydraulic_depth
from src.river_preburn import compute_river_bed_points

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
compute_hydraulic_depth = profiler.wrap(compute_hydraulic_depth)
load_nienhuis            = profiler.wrap(load_nienhuis)
match_basin_to_delta     = profiler.wrap(match_basin_to_delta)
compute_estuarine_depths = profiler.wrap(compute_estuarine_depths)
compute_river_bed_points = profiler.wrap(compute_river_bed_points)
burn_river_channel       = profiler.wrap(burn_river_channel)

# ── domain ────────────────────────────────────────────────────────────────────

wgs84_bounds, domain_crs, domain_poly = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)
log.info(f"Domain WGS84 bounds: {wgs84_bounds}, CRS: {domain_crs}")

log.info(f"Hydraulic geometry: depth = {snakemake.params.hg_c} * Q^{snakemake.params.hg_f}")

# ── load clean network (discharge already propagated by rule 08) ──────────────

rivers = gpd.read_file(snakemake.input.clean_river_network)
log.info(f"Loaded {len(rivers)} reaches (bankfull_discharge_acc pre-computed)")

# Seed reach IDs stored by rule 08 as the 'is_seed' flag
seed_reach_ids: set[str] = set(
    rivers.loc[rivers["is_seed"].astype(bool), "reach_id"].astype(str)
) if "is_seed" in rivers.columns else set()
log.info(f"Seed reaches: {len(seed_reach_ids)}")

# ── power-law hydraulic depth ─────────────────────────────────────────────────

rivers_out = rivers.copy()
rivers_out["rivdph"] = compute_hydraulic_depth(
    rivers_out["bankfull_discharge_acc"].values,
    c=snakemake.params.hg_c,
    f=snakemake.params.hg_f,
)
if len(rivers_out) > 0:
    log.info(
        f"Hydraulic depth: min={rivers_out['rivdph'].min():.3f} m, "
        f"max={rivers_out['rivdph'].max():.3f} m, "
        f"median={rivers_out['rivdph'].median():.3f} m"
    )

# ── estuarine depth model (only when enabled) ─────────────────────────────────

estuarine_enabled = bool(snakemake.params.estuarine_enabled)
log.info(f"Estuarine depth model: {'ENABLED' if estuarine_enabled else 'DISABLED'}")

L_e_m = None
if estuarine_enabled:
    nienhuis_df = load_nienhuis(snakemake.input.nienhuis)
    delta_params = match_basin_to_delta(
        delta_polygon_path=snakemake.input.delta_polygon,
        nienhuis_df=nienhuis_df,
        max_dist_km=snakemake.params.max_match_dist_km,
    )

    if delta_params is None:
        log.warning(
            f"No Nienhuis delta found within {snakemake.params.max_match_dist_km:.0f} km "
            "of basin bbox — retaining power-law depths for all reaches"
        )
        rivers_out["rivdph_estuarine"] = False
        rivers_out["rivdph_powerlaw"] = rivers_out["rivdph"]
        rivers_out["rivdph_blend_alpha"] = float("nan")
    else:
        log.info(
            f"Matched Nienhuis delta: {delta_params.get('name', delta_params['id'])} "
            f"(L_e={float(delta_params['L_e'])/1000:.1f} km, P={float(delta_params['P']):.3g} m³)"
        )
        L_e_m = float(delta_params["L_e"])
        rivers_out = compute_estuarine_depths(
            rivers=rivers_out,
            delta_params=delta_params,
            obrien_C=snakemake.params.obrien_C,
            obrien_alpha=snakemake.params.obrien_alpha,
            convergence_ratio_k=snakemake.params.convergence_ratio_k,
            blend_fraction=snakemake.params.blend_fraction,
            min_depth_m=snakemake.params.min_depth_m,
        )

# ── mouth depth monotonicity ───────────────────────────────────────────────────
# Applied unconditionally (not just when the estuarine model ran): a mouth
# shouldn't be shallower than its own power-law estimate or the reach
# feeding it, whether that shallowing came from the estuarine model's
# O'Brien-relation estimate (and/or its min_depth_m floor) or -- in the
# disabled/no-delta-match case -- would just be a plain power-law
# discontinuity at the outlet.
rivers_out = enforce_mouth_depth_monotonic(rivers_out)

# ── write unified river_network_depth_estimated.gpkg ──────────────────────────

Path(snakemake.output.depth_estimated_river_network).parent.mkdir(parents=True, exist_ok=True)
rivers_out.to_file(snakemake.output.depth_estimated_river_network, driver="GPKG")
log.info(f"Written: {snakemake.output.depth_estimated_river_network}")

# ── burned river DEM (native + coarse grid) ──────────────────────────────
# Burns directly (compute_river_bed_points -> burn_river_channel), never
# hydromt's own gdf_zb/burn_river_rect (produces a wavy, non-monotonic bed)
# -- the only river-bed mechanism this codebase trusts. Always against
# elevation_conditioned (matching rule modelled_depth_estimation's own
# reference exactly).
elevation_conditioned_path = Path(snakemake.input.elevation_conditioned)
zbed_gdf = compute_river_bed_points(
    rivers=rivers_out, elevation_path=elevation_conditioned_path, depth_column="rivdph",
)
log.info(f"River bed anchor points: {len(zbed_gdf)}")

with rasterio.open(elevation_conditioned_path) as _src:
    utm_crs = _src.crs
    native_resolution_m = abs(_src.transform.a)

NODATA = np.float32(-9999.0)

with open(snakemake.input.sfincs_grid) as f:
    grid_def = json.load(f)
sfincs_grid_transform = Affine(*grid_def["transform"])
sfincs_grid_shape = (grid_def["height"], grid_def["width"])

rivers_utm = rivers_out.to_crs(utm_crs)
channel_mask = build_channel_mask_regular(rivers_utm, "width", sfincs_grid_shape, sfincs_grid_transform)
log.info(f"Shared channel_mask (SFINCS-grid resolution): {int(channel_mask.sum()):,} cell(s)")

burned_arr, burned_transform, _nd, stats = burn_river_channel(
    rivers=rivers_out, zbed_anchors=zbed_gdf,
    natural_dem_path=elevation_conditioned_path,
    utm_crs=utm_crs, resolution_m=native_resolution_m,
)
# Constrain to the coarse channel_mask: the native burn above rasterizes
# each reach's own buf_poly independently, which can extend a handful of
# excavated pixels beyond the coarse weir-protection corridor (see
# constrain_to_coarse_channel_mask's own docstring) -- guarantees the
# native footprint is always a subset of channel_mask, same as the coarse
# burn below already is by construction.
burned_arr = constrain_to_coarse_channel_mask(
    burned_arr, burned_transform, utm_crs, channel_mask, sfincs_grid_transform,
)
river_burned_dem_path = Path(snakemake.output.river_burned_dem)
river_burned_dem_path.parent.mkdir(parents=True, exist_ok=True)
with rasterio.open(
    river_burned_dem_path, "w", driver="GTiff", dtype="float32",
    width=burned_arr.shape[1], height=burned_arr.shape[0],
    count=1, crs=utm_crs, transform=burned_transform,
    nodata=float(NODATA), compress="deflate", tiled=True,
) as dst:
    dst.write(np.where(np.isnan(burned_arr), NODATA, burned_arr).astype(np.float32), 1)
log.info(
    f"Written: {river_burned_dem_path} ({stats['n_reaches_burned']} reach(es) burned, "
    f"{stats['n_reaches_skipped']} skipped, {stats['n_pixels_burned']:,} pixel(s))"
)

burned_arr_sfincs, transform_sfincs, _nd_sfincs, stats_sfincs = burn_river_channel(
    rivers=rivers_out, zbed_anchors=zbed_gdf,
    natural_dem_path=elevation_conditioned_path,
    utm_crs=utm_crs, resolution_m=grid_def["resolution"],
    out_transform=sfincs_grid_transform, out_shape=sfincs_grid_shape,
    channel_mask=channel_mask,
)
river_burned_dem_sfincs_grid_path = Path(snakemake.output.river_burned_dem_sfincs_grid)
river_burned_dem_sfincs_grid_path.parent.mkdir(parents=True, exist_ok=True)
with rasterio.open(
    river_burned_dem_sfincs_grid_path, "w", driver="GTiff", dtype="float32",
    width=sfincs_grid_shape[1], height=sfincs_grid_shape[0],
    count=1, crs=utm_crs, transform=transform_sfincs,
    nodata=float(NODATA), compress="deflate",
) as dst:
    dst.write(np.where(np.isnan(burned_arr_sfincs), NODATA, burned_arr_sfincs).astype(np.float32), 1)
log.info(
    f"Written: {river_burned_dem_sfincs_grid_path} ({stats_sfincs['n_reaches_burned']} reach(es) burned, "
    f"{stats_sfincs['n_reaches_skipped']} skipped, {stats_sfincs['n_pixels_burned']:,} pixel(s) "
    f"@ {grid_def['resolution']:.2f} m)"
)

# ── plots ─────────────────────────────────────────────────────────────────────

rivers_wgs = (
    rivers_out.to_crs("EPSG:4326")
    if rivers_out.crs is not None and rivers_out.crs.to_epsg() != 4326
    else rivers_out
)
plot_river_depth(
    rivers_wgs=rivers_wgs,
    bbox_poly=domain_poly,
    osm_land_path=snakemake.input.land_polygons,
    output_path=snakemake.output.plot_river_depth,
)
plot_river_network_width_discharge(
    rivers_wgs=rivers_wgs,
    bbox_poly=domain_poly,
    osm_land_path=snakemake.input.land_polygons,
    seed_reach_ids=seed_reach_ids,
    output_path=snakemake.output.plot_river_network_width_discharge,
)
Path(snakemake.output.plot_hydraulic_relations).parent.mkdir(parents=True, exist_ok=True)
plot_hydraulic_relations_with_estuarine(
    rivers_wgs=rivers_wgs,
    output_path=snakemake.output.plot_hydraulic_relations,
    L_e_m=L_e_m,
)
log.info(f"Plot written: {snakemake.output.plot_hydraulic_relations}")

profiler.stop()
log.info("Done")
