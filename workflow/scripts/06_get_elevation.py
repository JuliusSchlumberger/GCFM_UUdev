"""Merge FathomDEM and GEBCO into elevation_merged.tif and put land use on its grid; run by rule get_elevation.

1.  FathomDEM tiles to the UTM working grid, coverage-weighted resampling.
1b. Land use resampled onto that grid by mode (landuse.tif); FathomDEM set to
    NaN on its sea cells (land use 200):
    a terrestrial DEM reports spurious shallow "elevation" over water, so GEBCO
    fills these, including sea-connected creeks, estuaries and lagoons.
2.  FathomDEM EGM2008 -> GOCO06s: add N_EGM2008 - N_GOCO06s (pyshtools). Mandatory:
    GEBCO/COAST-RP/MDT share the GOCO06s frame. No MDT on the DEM (sea only).
3.  GEBCO local MSL -> GOCO06s by ADDING the MDT (HYBRID-CNES-CLS2022; MDT = mean
    sea surface above the geoid, H_GOCO06s = H_MSL + MDT), IDW-extrapolated over
    land so nearshore pixels are not NaN-poisoned.
3b. GEBCO clamped between terrain.gebco_max_depth_m and terrain.gebco_min_depth_m
    below local MSL (the MDT surface); the shallow limit on sea cells only.
4.  FathomDEM where valid, GEBCO elsewhere, linear blend over terrain.coast_blend_m
    seaward of the coast in open water only; FathomDEM pixels never change.
5.  Write elevation_merged.tif (nodata -9999).
6.  Diagnostic maps: elevation, land use, geoid offset, MDT, both datum-correction deltas.

zsini.tif is NOT produced here: rule modelled_depth_estimation (11) writes
its sea cells (zsini_sea_cells_on_grid.tif) and rule build_sfincs_skeleton
turns them into zsini.tif.
"""

import json
import math
import os
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import geometry_mask as rio_geom_mask
from rasterio.fill import fillnodata
from rasterio.mask import mask as rio_mask
from rasterio.transform import from_origin as _rast_from_origin
from shapely.geometry import box as _box

from src.domain import load_domain
from src.landuse import LANDUSE_SEA_CODE, warp_landuse_to_grid
from src.log import setup_logging
from src.profiling import ScriptProfiler
from src.raster import (
    FLOAT_NODATA,
    _tile_intersects,
    blend_coast,
    compute_geoid_offset_arr,
    find_fathomdem_tiles,
    latlon_dims,
    merge_tiled_raster,
    merge_tiles,
    open_water_mask,
    reproject_nan_aware,
)
from src.surge import load_mdt

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
find_fathomdem_tiles = profiler.wrap(find_fathomdem_tiles)
merge_tiled_raster = profiler.wrap(merge_tiled_raster)

# params
domain_meta_path = Path(snakemake.input.spec_basins_meta)
topo_tiles_dir = snakemake.input.global_topography_tiles
bathymetry_path = Path(snakemake.input.global_bathymetry)
landuse_source_path = Path(snakemake.input.landuse_source)
land_polygons_path = Path(snakemake.input.land_polygons)
goco_path = Path(snakemake.input.goco06s_gfc)
egm_path = Path(snakemake.input.egm2008_gfc)
mdt_path = snakemake.input.mdt

out_elev_path = Path(snakemake.output.elevation_merged)
ANALYSIS = hasattr(snakemake.output, "plot_geoid_offset")

mdt_variable = "mdt"  # sole variable in data_catalogue's mdt_cnes_cls22 source
mdt_load_margin_deg = float(snakemake.params.mdt_load_margin_deg)
gebco_max_depth_m = float(snakemake.params.gebco_max_depth_m)
gebco_min_depth_m = float(snakemake.params.gebco_min_depth_m)
coast_blend_m = float(snakemake.params.coast_blend_m)
if not 0 <= gebco_min_depth_m < gebco_max_depth_m:
    raise ValueError(
        f"terrain.gebco_min_depth_m ({gebco_min_depth_m}) must be >= 0 and below "
        f"terrain.gebco_max_depth_m ({gebco_max_depth_m})"
    )

log.info("Topography source: FathomDEM (mandatory EGM2008 → GOCO06s datum correction)")

# domain
wgs84_bounds, domain_crs_str, domain_poly = load_domain(
    domain_meta_path, snakemake.input.domain_gpkg
)
utm_crs = CRS.from_string(domain_crs_str)
log.info(f"Domain WGS84 bounds: {wgs84_bounds}, UTM CRS: {utm_crs.to_string()}")

# 1. Merge FathomDEM tiles
tiles = find_fathomdem_tiles(topo_tiles_dir, wgs84_bounds)
if not tiles:
    raise FileNotFoundError(
        "No FathomDEM tiles found for domain. Check the data catalogue path."
    )
log.info(f"Found {len(tiles)} FathomDEM tile(s)")

# Working resolution from FathomDEM's native 1 arcsecond, converted with the
# north-south arc length (111 320 m/deg), which barely varies with latitude,
# so every basin gets the same ~30 m pixel.
with rasterio.open(tiles[0]) as _tile_src:
    native_deg_res = abs(_tile_src.res[1])
work_res_m = native_deg_res * 111_320.0
log.info(
    f"Working resolution auto-derived from FathomDEM native pixel size "
    f"({native_deg_res:.6f} deg): {work_res_m:.2f} m"
)

# UTM grid from the domain's own bounds: calculate_default_transform would put
# the origin several metres off the UTM domain corner, misaligning DEM and model.
with open(domain_meta_path) as _f:
    _dm = json.load(_f)
_b = _dm["bounds"]  # xmin/ymin/xmax/ymax in domain UTM CRS
w_utm = math.ceil((_b["xmax"] - _b["xmin"]) / work_res_m)
h_utm = math.ceil((_b["ymax"] - _b["ymin"]) / work_res_m)
transform_utm = _rast_from_origin(_b["xmin"], _b["ymax"], work_res_m, work_res_m)
log.info(
    f"UTM output grid: {w_utm}×{h_utm} px @ {work_res_m:.2f} m, "
    f"origin ({_b['xmin']:.1f}, {_b['ymax']:.1f}) [{domain_crs_str}]"
)

# domain-polygon mask (UTM working grid)
# fillnodata cannot tell outside the delta polygon from a real gap, so this
# mask re-clamps after every fill.
domain_poly_utm = gpd.GeoSeries([domain_poly], crs="EPSG:4326").to_crs(utm_crs).iloc[0]
outside_domain = rio_geom_mask(
    [domain_poly_utm.__geo_interface__],
    out_shape=(h_utm, w_utm),
    transform=transform_utm,
    all_touched=True,
)

dem_meta = dict(
    driver="GTiff",
    dtype="float32",
    width=w_utm,
    height=h_utm,
    count=1,
    crs=utm_crs,
    transform=transform_utm,
    nodata=FLOAT_NODATA,
    compress="deflate",
    tiled=True,
)

with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tf:
    tmp_topo = tf.name

try:
    merge_tiled_raster(tiles, wgs84_bounds, tmp_topo)

    with rasterio.open(tmp_topo) as src:
        topo_nodata = src.nodata
        # Mask before reprojection: tile edges (meridians/parallels) curve in
        # UTM and would show as NaN strips.
        _nd = float(topo_nodata) if topo_nodata is not None else FLOAT_NODATA
        _masked, _masked_transform = rio_mask(
            src,
            [domain_poly.__geo_interface__],
            crop=False,
            nodata=_nd,
            all_touched=True,
        )
        topo_arr = _masked[0].astype(np.float32)
        topo_arr[topo_arr == np.float32(_nd)] = np.nan
        # FathomDEM tiles store elevation as int32 centimeters (raster tag
        # VERTICAL_UNITS=centimeters), not meters.
        topo_arr = topo_arr / 100.0
        topo_src_crs = src.crs

    # Coverage-weighted: a pixel is NaN only when its whole receptive field is,
    # so plain bilinear does not erode data at tile gaps / the polygon edge.
    topo_utm = reproject_nan_aware(
        topo_arr,
        _masked_transform,
        topo_src_crs,
        (h_utm, w_utm),
        transform_utm,
        utm_crs,
        resampling=Resampling.bilinear,
    )
finally:
    os.unlink(tmp_topo)

log.info(
    f"FathomDEM: {(~np.isnan(topo_utm)).sum():,} valid px "
    f"({100 * (~np.isnan(topo_utm)).mean():.1f} %)"
)

# 1b. Land use on this grid (landuse.tif); FathomDEM to NaN on its sea cells (200).
# Mode, not nearest: ESA WorldCover (10 m) is finer than this grid (~30 m), so
# nearest would keep one arbitrary source pixel per cell; with 100 m LC100 it
# is upsampling, where both agree. Streamed in row blocks (a 10 m source window
# reaches 2.0 Gpx on the largest delta).
histogram = warp_landuse_to_grid(
    landuse_source_path, dem_meta, snakemake.output.spec_landuse
)
_total = sum(histogram.values())
log.info(
    "landuse.tif class histogram (% of grid): "
    + ", ".join(f"{c}={100 * n / _total:.2f}" for c, n in sorted(histogram.items()))
)
log.info(f"Written: {snakemake.output.spec_landuse}")
with rasterio.open(snakemake.output.spec_landuse) as _lu_src:
    sea_mask = _lu_src.read(1) == LANDUSE_SEA_CODE
n_sea_clipped = int((sea_mask & ~np.isnan(topo_utm)).sum())
topo_utm[sea_mask] = np.nan
log.info(
    f"Sea clip (land use {LANDUSE_SEA_CODE}): {n_sea_clipped:,} FathomDEM px reset to NaN "
    f"({(~np.isnan(topo_utm)).sum():,} valid px remaining) — "
    "GEBCO will fill these at the step 4 merge"
)

# 2. Vertical datum correction — EGM2008 → GOCO06s (mandatory)
topo_egm2008 = topo_utm.copy()  # snapshot before correction for the delta plot
valid_topo = ~np.isnan(topo_utm)

log.info(
    f"Computing EGM2008→GOCO06s geoid offset from {goco_path.name} / {egm_path.name} …"
)
offset_arr, offset_transform, offset_crs = compute_geoid_offset_arr(goco_path, egm_path)
offset_on_dem = reproject_nan_aware(
    offset_arr.astype(np.float32),
    offset_transform,
    offset_crs,
    (h_utm, w_utm),
    transform_utm,
    utm_crs,
)
log.info(
    f"Geoid offset over domain: "
    f"[{offset_on_dem[valid_topo].min():.4f}, {offset_on_dem[valid_topo].max():.4f}] m"
)
topo_utm[valid_topo] += offset_on_dem[valid_topo]
del offset_on_dem

if ANALYSIS:
    from src.plots import plot_geoid_offset

    plot_geoid_offset(
        offset_arr=offset_arr,
        offset_transform=offset_transform,
        wgs84_bounds=wgs84_bounds,
        land_polygons_path=str(land_polygons_path),
        output_path=snakemake.output.plot_geoid_offset,
    )
    log.info(f"Plot written: {snakemake.output.plot_geoid_offset}")
del offset_arr

delta_correction = np.where(valid_topo, topo_utm - topo_egm2008, np.nan).astype(
    np.float32
)
del topo_egm2008
log.info(
    f"DEM datum correction complete (EGM2008 → GOCO06s) — Δ: "
    f"mean={np.nanmean(delta_correction):.4f} m, "
    f"std={np.nanstd(delta_correction):.4f} m, "
    f"range=[{np.nanmin(delta_correction):.4f}, {np.nanmax(delta_correction):.4f}] m"
)

# 3. GEBCO on UTM working grid
log.info(f"Loading GEBCO from {bathymetry_path}…")

if bathymetry_path.is_dir():
    bbox_geom = _box(*wgs84_bounds)
    candidates = [
        str(fp)
        for fp in sorted(bathymetry_path.glob("*.tif"))
        if _tile_intersects(fp, bbox_geom)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No GEBCO tiles found overlapping domain in {bathymetry_path}"
        )
    log.info(f"  {len(candidates)} GEBCO tile(s) overlap domain")
    gebco_data, gebco_transform, _gebco_meta = merge_tiles(candidates, wgs84_bounds)
    gebco_nodata = _gebco_meta["nodata"]
    gebco_crs = _gebco_meta["crs"]
    gebco_arr = gebco_data[0].astype(np.float32)
    if gebco_nodata is not None:
        gebco_arr[gebco_arr == np.float32(gebco_nodata)] = np.nan
    _outside = rio_geom_mask(
        [domain_poly.__geo_interface__],
        out_shape=gebco_arr.shape,
        transform=gebco_transform,
        all_touched=True,
    )
    gebco_arr[_outside] = np.nan
    gebco_utm = reproject_nan_aware(
        gebco_arr,
        gebco_transform,
        gebco_crs,
        (h_utm, w_utm),
        transform_utm,
        utm_crs,
        resampling=Resampling.bilinear,
    )
else:
    with rasterio.open(bathymetry_path) as src:
        _nd_g = float(src.nodata) if src.nodata is not None else FLOAT_NODATA
        _g_masked, _g_transform = rio_mask(
            src,
            [domain_poly.__geo_interface__],
            crop=False,
            nodata=_nd_g,
            all_touched=True,
        )
        gebco_arr = _g_masked[0].astype(np.float32)
        gebco_arr[gebco_arr == np.float32(_nd_g)] = np.nan
        gebco_utm = reproject_nan_aware(
            gebco_arr,
            _g_transform,
            src.crs,
            (h_utm, w_utm),
            transform_utm,
            utm_crs,
            resampling=Resampling.bilinear,
        )

log.info(f"GEBCO valid: {(~np.isnan(gebco_utm)).sum():,} px")

# 3. GEBCO datum correction (add MDT: MSL → GOCO06s)
# H_GOCO06s = H_MSL + MDT (HYBRID-CNES-CLS22), as rule 07d does for COAST-RP.
# Applied to every valid GEBCO pixel: the merge may fall back to GEBCO on land.
gebco_before_mdt = gebco_utm.copy()  # snapshot before correction for the delta plot
valid_gebco = ~np.isnan(gebco_utm)

log.info(f"Loading MDT (variable='{mdt_variable}') from {mdt_path} …")
mdt_clip = load_mdt(
    mdt_path, mdt_variable, bbox=wgs84_bounds, margin_deg=mdt_load_margin_deg
)
lat_dim, lon_dim = latlon_dims(mdt_clip)
# load_mdt returns ascending latitude; flip to north-up.
mdt_np = mdt_clip.transpose(lat_dim, lon_dim).values[::-1].astype(np.float32)
mdt_lats = mdt_clip[lat_dim].values[::-1]
mdt_lons = mdt_clip[lon_dim].values

dlat = float(np.abs(mdt_lats[0] - mdt_lats[1]))
dlon = float(np.abs(mdt_lons[0] - mdt_lons[1]))
mdt_transform_wgs = _rast_from_origin(
    float(mdt_lons[0]) - dlon / 2,
    float(mdt_lats[0]) + dlat / 2,
    dlon,
    dlat,
)

if ANALYSIS:
    from src.plots import plot_mdt_ocean

    plot_mdt_ocean(
        mdt_np=mdt_np,
        mdt_transform=mdt_transform_wgs,
        wgs84_bounds=wgs84_bounds,
        land_polygons_path=str(land_polygons_path),
        output_path=snakemake.output.plot_mdt_ocean,
    )
    log.info(f"Plot written: {snakemake.output.plot_mdt_ocean}")

# Extrapolate MDT over land (IDW + light smoothing) before resampling.
# Without this, adding the MDT only affects pixels that happen to overlap
# an ocean MDT cell, leaving land-side GEBCO gap-fill pixels unconnected from
# sea-level information.
n_ocean = int((~np.isnan(mdt_np)).sum())
log.info(
    f"MDT clip: {n_ocean}/{mdt_np.size} ocean pixels valid; "
    f"extrapolating over land (IDW, search radius = {max(mdt_np.shape) * 2} px) …"
)
mdt_filled = mdt_np.copy()
mdt_valid_mask = (~np.isnan(mdt_np)).astype(np.uint8)  # 1=ocean, 0=land
if n_ocean > 0:
    fillnodata(
        mdt_filled,
        mask=mdt_valid_mask,
        max_search_distance=float(max(mdt_np.shape) * 2),
        smoothing_iterations=1,  # mild Laplacian smoothing ≈ ArcGIS smooth factor 0.5
    )
else:
    log.warning(
        "MDT has no valid ocean pixels in the extended domain — skipping the MDT correction"
    )

mdt_on_gebco = reproject_nan_aware(
    mdt_filled,
    mdt_transform_wgs,
    "EPSG:4326",
    (h_utm, w_utm),
    transform_utm,
    utm_crs,
)
log.info(
    f"MDT over GEBCO domain: "
    f"mean={float(np.nanmean(mdt_on_gebco)):.4f} m, "
    f"range=[{float(np.nanmin(mdt_on_gebco)):.4f}, {float(np.nanmax(mdt_on_gebco)):.4f}] m"
)
gebco_utm[valid_gebco] += mdt_on_gebco[valid_gebco]
# Local mean sea level in the GOCO06s frame -- the reference of the clamp below.
msl_utm = mdt_on_gebco
del mdt_np, mdt_filled, mdt_on_gebco

gebco_delta_correction = np.where(
    valid_gebco, gebco_utm - gebco_before_mdt, np.nan
).astype(np.float32)
del gebco_before_mdt
log.info(
    f"GEBCO datum correction complete (raw → GOCO06s via MDT) — Δ: "
    f"mean={np.nanmean(gebco_delta_correction):.4f} m, "
    f"std={np.nanstd(gebco_delta_correction):.4f} m, "
    f"range=[{np.nanmin(gebco_delta_correction):.4f}, {np.nanmax(gebco_delta_correction):.4f}] m"
)

# 3b. Clamp GEBCO depth, relative to local mean sea level
# Depths below MSL (msl_utm), after datum correction, in GOCO06s metres.
# Deepest: SFINCS's CFL time step (dt ~ dx/sqrt(g*h)) is set by the single
# deepest active cell, and offshore forcing is prescribed at the boundary, so
# deep water only slows the run.
# Shallowest: GEBCO (~450 m) is often too shallow or above sea level at the
# coast, leaving dry "land" in the sea; creeks and estuaries hold the
# surrounding land, so they come out exactly this deep. Sea cells only: GEBCO
# also fills rare land gaps, which must not go below sea level.
_deepest = msl_utm - gebco_max_depth_m
_shallowest = msl_utm - gebco_min_depth_m
_too_deep = gebco_utm < _deepest
_too_shallow = sea_mask & (gebco_utm > _shallowest)
gebco_utm = np.where(_too_deep, _deepest, gebco_utm)
gebco_utm = np.where(_too_shallow, _shallowest, gebco_utm)
log.info(
    f"GEBCO depth clamp (below MSL): {int(_too_deep.sum()):,} px deeper than "
    f"{gebco_max_depth_m:g} m reset to it; {int(_too_shallow.sum()):,} sea px shallower than "
    f"{gebco_min_depth_m:g} m reset to it"
)
del msl_utm, _deepest, _shallowest, _too_deep, _too_shallow

# 4. Merge: FathomDEM wherever valid, GEBCO everywhere else
# Blend in open water only: in sea narrower than twice the band (creeks,
# estuaries) it would reach in from both banks and fill the channel back up
# towards bank level, so those keep the hard step.
_topo_valid = ~np.isnan(topo_utm)
merged = np.where(_topo_valid, topo_utm, gebco_utm)
_open_water = open_water_mask(~_topo_valid, work_res_m, coast_blend_m)
merged, _n_blended = blend_coast(
    merged,
    _topo_valid,
    gebco_utm,
    pixel_size_m=work_res_m,
    blend_m=coast_blend_m,
    where=_open_water,
)
log.info(
    f"Coast blend: {_n_blended:,} px of open water within {coast_blend_m:g} m of the "
    f"coastline interpolated between FathomDEM (coast) and GEBCO; "
    f"{int((sea_mask & ~_open_water).sum()):,} px of narrow sea (< {2 * coast_blend_m:g} m wide) left unblended"
)
del _open_water
_remaining = np.isnan(merged)
if _remaining.any():
    # Rare: GEBCO is globally complete. `_remaining` also holds the whole
    # outside-polygon area, which can be most of the raster, so the search is
    # bounded small (an unbounded one takes hours on a large basin); genuine
    # gaps are small and near real data. The outside is re-masked next anyway.
    _GAP_FILL_MAX_SEARCH_PX = 100.0
    log.info(
        f"Hard-merge gap-fill: {int(_remaining.sum()):,} px need filling "
        f"(incl. outside-domain area), max_search_distance="
        f"{_GAP_FILL_MAX_SEARCH_PX:g} px"
    )
    fillnodata(merged, mask=~_remaining, max_search_distance=_GAP_FILL_MAX_SEARCH_PX)
# fillnodata's mask only distinguishes "has data" from "no data" -- with no
# notion of "outside domain_poly", so even the 100 px max_search_distance
# would otherwise extrapolate valid elevation all the way out to the
# rectangular bbox edges. Re-clamp to NaN outside the polygon regardless of how
# far the fill reached.
merged[outside_domain] = np.nan
log.info(
    f"Merge (FathomDEM valid={_topo_valid.sum():,} px, GEBCO fallback "
    f"elsewhere, blended at the coast): min={np.nanmin(merged):.2f} m, max={np.nanmax(merged):.2f} m, "
    f"nan={np.isnan(merged).sum():,} px"
)
del _remaining

# 5. Write outputs
out_elev_path.parent.mkdir(parents=True, exist_ok=True)

with rasterio.open(out_elev_path, "w", **dem_meta) as dst:
    dst.write(np.where(np.isnan(merged), FLOAT_NODATA, merged).astype(np.float32), 1)
log.info(f"Written: {out_elev_path}")

# 6. Diagnostic plots
if ANALYSIS:
    from src.plots import plot_datum_correction_delta

    plot_datum_correction_delta(
        delta=delta_correction,
        wgs84_bounds=wgs84_bounds,
        land_polygons_path=str(land_polygons_path),
        output_path=snakemake.output.plot_datum_fathomdem,
        title="Vertical datum correction — FathomDEM\n(EGM2008 → GOCO06s geoid)",
        colorbar_label="Δ elevation (m)  [GOCO06s − EGM2008]",
        note=(
            "Note: FathomDEM and the underlying land-use map (grey background) are not fully aligned.\n"
            "In areas without FathomDEM coverage, GEBCO data are used (see 06_datum_gebco_correction.png)."
        ),
    )
    log.info(f"Plot written: {snakemake.output.plot_datum_fathomdem}")

    plot_datum_correction_delta(
        delta=gebco_delta_correction,
        wgs84_bounds=wgs84_bounds,
        land_polygons_path=str(land_polygons_path),
        output_path=snakemake.output.plot_datum_gebco,
        title="Vertical datum correction — GEBCO\n(local MSL → GOCO06s: + MDT_CNES-CLS22)",
        colorbar_label="Δ elevation (m)  [GOCO06s − raw GEBCO]",
    )
    log.info(f"Plot written: {snakemake.output.plot_datum_gebco}")

profiler.stop()
log.info("Done")
