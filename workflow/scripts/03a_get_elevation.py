"""
03a_get_elevation.py — Build the merged elevation product for a basin.

Processing chain
----------------
1.  Merge coastal DEM tiles for the domain (DiluviumDEM or DeltaDTM, per config).
1b. [DiluviumDEM only, if datum_correction.enabled] EGM2008 → local MSL:
      (a) add geoid offset  N_EGM2008 − N_GOCO06s   (pyshtools synthesis)
      (b) subtract MDT extrapolated over land with IDW  (HYBRID-CNES-CLS2022)
2.  Clip GEBCO to domain, resample to working UTM grid.
3.  Rasterise land polygons; fill land NaN with impassable-barrier value.
4.  Hard-boundary merge: coastal DEM on land, GEBCO in ocean (no gradient).
5.  Write elevation_merged.tif and zsini.tif.
6.  Diagnostic elevation map.  If datum correction ran, also writes
    01b_datum_correction.png (Δ MSL−EGM2008 over the domain).

Supported coastal DEMs (topography.source in config)
------------------------------------------------------
DiluviumDEM  — 1°×1° tiles, EGM2008 vertical datum, clip at 80 m.
DeltaDTM     — 1°×1° tiles, already MSL-corrected (GOCO06s + MDT), clip at 10 m.

Optional gradient blend (topography.blend.enabled: true)
---------------------------------------------------------
When enabled the script also computes a distance-weighted gradient blend and
writes elevation_gradient.tif + a two-subplot diagnostic PNG alongside the
regular outputs.  These files are NOT tracked as Snakemake outputs; they are
produced as side effects for visual comparison only.  All downstream pipeline
steps use elevation_merged.tif (the hard-boundary product).

Datum correction (topography.datum_correction.enabled)
------------------------------------------------------
When enabled for DiluviumDEM: follows Seeger & Minderhoud (2026) — geoid
offset (EGM2008 → GOCO06s, pyshtools) then MDT subtraction.  MDT is
extrapolated over land via inverse-distance weighting before resampling;
this is the key step that makes the correction spatially continuous across
the coast rather than ocean-only.  Requires pyshtools + boule.
DeltaDTM tiles already carry the MSL correction (suffix _GOCO06s_MDT).
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
from rasterio.features import geometry_mask as rio_geom_mask, rasterize as rio_rasterize
from rasterio.fill import fillnodata
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge as rio_merge_mem
from rasterio.transform import from_origin as _rast_from_origin
from rasterio.warp import reproject as rio_reproject
from shapely.geometry import box as _box

from src.domain import load_domain
from src.log import setup_logging
from src.plots import plot_elevation_merged, plot_elevation_blending, plot_zsini
from src.profiling import ScriptProfiler
from src.raster import (
    _tile_intersects,
    compute_geoid_offset_arr,
    find_diluviumdem_tiles,
    find_deltadtm_tiles,
    merge_tiled_raster,
    resample_to_utm_array,
)

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
find_diluviumdem_tiles = profiler.wrap(find_diluviumdem_tiles)
find_deltadtm_tiles    = profiler.wrap(find_deltadtm_tiles)
merge_tiled_raster     = profiler.wrap(merge_tiled_raster)

# ── params ────────────────────────────────────────────────────────────────────
domain_meta_path   = Path(snakemake.input.spec_basins_meta)
topo_tiles_dir     = snakemake.input.global_topography_tiles
bathymetry_path    = Path(snakemake.input.global_bathymetry)
land_polygons_path = Path(snakemake.input.land_polygons)

out_elev_path  = Path(snakemake.output.elevation_merged)
out_zsini_path = Path(snakemake.output.zsini)
out_plot_path  = snakemake.output.plot_elevation
out_plot_zsini_path = snakemake.output.plot_zsini

# Derived paths for optional blend outputs (not Snakemake-tracked outputs)
out_gradient_path      = out_elev_path.parent / "elevation_gradient.tif"
out_plot_gradient_path = str(out_elev_path.parent.parent / "plots" / "01b_elevation_gradient.png")

work_res_m        = snakemake.params.work_res_m
blend_buffer_m    = snakemake.params.blend_buffer_m
clip_elevation_m  = float(snakemake.params.clip_elevation_m)
land_barrier_val  = float(snakemake.params.land_barrier_val)
blend_enabled     = bool(snakemake.params.blend_enabled)
topo_source       = snakemake.params.topo_source   # "DiluviumDEM" or "DeltaDTM"
_dc_cfg           = bool(snakemake.params.datum_correction_enabled)
datum_correction_enabled = _dc_cfg and topo_source == "DiluviumDEM"  # DeltaDTM is already MSL-corrected

buffer_px = blend_buffer_m / work_res_m

log.info(f"Topography source: {topo_source}  (clip ≥ {clip_elevation_m} m → NaN)")
log.info(f"Blend enabled: {blend_enabled}")

# ── domain ────────────────────────────────────────────────────────────────────
wgs84_bounds, domain_crs_str, domain_poly = load_domain(
    domain_meta_path, snakemake.input.domain_gpkg
)
utm_crs = CRS.from_string(domain_crs_str)
log.info(f"Domain WGS84 bounds: {wgs84_bounds}, UTM CRS: {utm_crs.to_string()}")

# Build the UTM output grid directly from the domain's own bounds so that
# the elevation/bathymetry rasters are pixel-perfect with the SFINCS model
# grid.  calculate_default_transform back-projects the WGS84 source extent
# and produces an origin that can be several metres off the UTM domain
# corner, causing sub-pixel misalignment between the DEM and the model grid.
with open(domain_meta_path) as _f:
    _dm = json.load(_f)
_b = _dm["bounds"]  # xmin/ymin/xmax/ymax in domain UTM CRS
w_utm = math.ceil((_b["xmax"] - _b["xmin"]) / work_res_m)
h_utm = math.ceil((_b["ymax"] - _b["ymin"]) / work_res_m)
transform_utm = _rast_from_origin(_b["xmin"], _b["ymax"], work_res_m, work_res_m)
log.info(
    f"UTM output grid: {w_utm}×{h_utm} px @ {work_res_m} m, "
    f"origin ({_b['xmin']:.1f}, {_b['ymax']:.1f}) [{domain_crs_str}]"
)

# ── 1. Merge coastal DEM tiles ─────────────────────────────────────────────────
_find_tiles = find_diluviumdem_tiles if topo_source == "DiluviumDEM" else find_deltadtm_tiles
tiles = _find_tiles(topo_tiles_dir, wgs84_bounds)
if not tiles:
    raise FileNotFoundError(
        f"No {topo_source} tiles found for domain. "
        f"Check topography.source in config and data catalogue path."
    )
log.info(f"Found {len(tiles)} {topo_source} tile(s)")

with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tf:
    tmp_topo = tf.name

try:
    merge_tiled_raster(tiles, wgs84_bounds, tmp_topo)

    with rasterio.open(tmp_topo) as src:
        topo_nodata = src.nodata
        # Clip merged WGS84 tiles to the domain polygon before reprojection.
        # Tile boundaries are meridians/parallels which become curves in UTM;
        # masking outside the domain first prevents those curved edges from
        # appearing as NaN strips in the final UTM raster.
        _nd = float(topo_nodata) if topo_nodata is not None else -9999.0
        _masked, _masked_transform = rio_mask(
            src, [domain_poly.__geo_interface__],
            crop=False, nodata=_nd, all_touched=True,
        )
        topo_arr = _masked[0].astype(np.float32)
        topo_arr[topo_arr == np.float32(_nd)] = np.nan
        topo_src_crs = src.crs

    topo_utm = np.full((h_utm, w_utm), np.nan, dtype=np.float32)
    rio_reproject(
        source=topo_arr, destination=topo_utm,
        src_transform=_masked_transform, src_crs=topo_src_crs,
        dst_transform=transform_utm, dst_crs=utm_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan, dst_nodata=np.nan,
    )
finally:
    os.unlink(tmp_topo)

topo_utm[topo_utm >= clip_elevation_m] = np.nan

dem_meta = dict(
    driver="GTiff", dtype="float32",
    width=w_utm, height=h_utm, count=1,
    crs=utm_crs, transform=transform_utm,
    nodata=-9999.0, compress="deflate", tiled=True,
)

# ── 1b. Vertical datum correction (DiluviumDEM only) ──────────────────────────
# Converts DiluviumDEM from EGM2008 to local MSL via:
#   Step 1: + geoid offset  (EGM2008 → GOCO06s)
#   Step 2: − MDT           (GOCO06s → local MSL)
# MDT is ocean-only in the raw product; we extrapolate it over land with IDW
# before resampling, following Seeger & Minderhoud (2026).
delta_correction = None  # populated below when enabled; used for the diagnostic plot

if datum_correction_enabled:
    import xarray as xr

    goco_path           = Path(snakemake.input.goco06s_gfc)
    egm_path            = Path(snakemake.input.egm2008_gfc)
    mdt_path            = snakemake.input.mdt
    mdt_variable        = snakemake.params.mdt_variable
    mdt_load_margin_deg = float(snakemake.params.mdt_load_margin_deg)

    log.info("Datum correction enabled (EGM2008 → GOCO06s → local MSL)")
    topo_egm2008 = topo_utm.copy()   # snapshot before correction for the delta plot
    valid_topo   = ~np.isnan(topo_utm)

    # Step 1: + geoid offset (EGM2008 → GOCO06s)
    log.info(f"Computing EGM2008→GOCO06s geoid offset from {goco_path.name} / {egm_path.name} …")
    offset_arr, offset_transform, offset_crs = compute_geoid_offset_arr(goco_path, egm_path)
    offset_on_dem = resample_to_utm_array(offset_arr, offset_transform, offset_crs, dem_meta)
    log.info(
        f"Geoid offset over domain: "
        f"[{offset_on_dem[valid_topo].min():.4f}, {offset_on_dem[valid_topo].max():.4f}] m"
    )
    topo_utm[valid_topo] += offset_on_dem[valid_topo]
    del offset_on_dem

    from src.plots import plot_geoid_offset
    _out_geoid = str(out_elev_path.parent.parent / "plots" / "01b_geoid_offset.png")
    plot_geoid_offset(
        offset_arr       = offset_arr,
        offset_transform = offset_transform,
        wgs84_bounds     = wgs84_bounds,
        osm_land_path    = str(land_polygons_path),
        output_path      = _out_geoid,
    )
    log.info(f"Plot written: {_out_geoid}")
    del offset_arr

    # Step 2: − MDT (GOCO06s → local MSL)
    # MDT is NaN over land in the raw dataset — must extrapolate first
    log.info(f"Loading MDT (variable='{mdt_variable}') from {mdt_path} …")
    lon_min, lat_min, lon_max, lat_max = wgs84_bounds
    margin = mdt_load_margin_deg

    with xr.open_dataset(mdt_path) as mdt_ds:
        mdt_da = mdt_ds[mdt_variable]
        lat_dim = next(d for d in mdt_da.dims if "lat" in d.lower())
        lon_dim = next(d for d in mdt_da.dims if "lon" in d.lower())
        # Drop any extra dimensions (e.g. time)
        for extra in [d for d in mdt_da.dims if d not in (lat_dim, lon_dim)]:
            mdt_da = mdt_da.isel({extra: 0})
        # Remap lon from 0…360 to −180…180 if needed
        if float(mdt_da[lon_dim].max()) > 180:
            mdt_da = mdt_da.assign_coords(
                {lon_dim: xr.where(mdt_da[lon_dim] > 180,
                                   mdt_da[lon_dim] - 360, mdt_da[lon_dim])}
            ).sortby(lon_dim)
        mdt_clip = mdt_da.sel({
            lat_dim: slice(lat_min - margin, lat_max + margin),
            lon_dim: slice(lon_min - margin, lon_max + margin),
        })
        mdt_np   = mdt_clip.values.astype(np.float32)
        mdt_lats = mdt_clip[lat_dim].values
        mdt_lons = mdt_clip[lon_dim].values

    dlat = float(np.abs(mdt_lats[0] - mdt_lats[1]))
    dlon = float(np.abs(mdt_lons[0] - mdt_lons[1]))
    if mdt_lats[0] < mdt_lats[-1]:   # ensure north-up
        mdt_np   = mdt_np[::-1]
        mdt_lats = mdt_lats[::-1]
    mdt_transform_wgs = _rast_from_origin(
        float(mdt_lons[0]) - dlon / 2,
        float(mdt_lats[0]) + dlat / 2,
        dlon, dlat,
    )

    from src.plots import plot_mdt_ocean
    _out_mdt = str(out_elev_path.parent.parent / "plots" / "01b_mdt_ocean.png")
    plot_mdt_ocean(
        mdt_np        = mdt_np,
        mdt_transform = mdt_transform_wgs,
        wgs84_bounds  = wgs84_bounds,
        osm_land_path = str(land_polygons_path),
        output_path   = _out_mdt,
    )
    log.info(f"Plot written: {_out_mdt}")

    # Extrapolate MDT over land (IDW + light smoothing) before resampling.
    # Without this, the subtraction only affects pixels that happen to overlap
    # an ocean MDT cell, leaving most of the land unconnected from the sea level
    # information — the key fix vs. naive bilinear resampling of the raw MDT.
    n_ocean = int((~np.isnan(mdt_np)).sum())
    log.info(
        f"MDT clip: {n_ocean}/{mdt_np.size} ocean pixels valid; "
        f"extrapolating over land (IDW, search radius = {max(mdt_np.shape) * 2} px) …"
    )
    mdt_filled     = mdt_np.copy()
    mdt_valid_mask = (~np.isnan(mdt_np)).astype(np.uint8)   # 1=ocean, 0=land
    if n_ocean > 0:
        fillnodata(
            mdt_filled,
            mask=mdt_valid_mask,
            max_search_distance=float(max(mdt_np.shape) * 2),
            smoothing_iterations=1,   # mild Laplacian smoothing ≈ ArcGIS smooth factor 0.5
        )
    else:
        log.warning("MDT has no valid ocean pixels in the extended domain — skipping subtraction")

    mdt_on_dem = resample_to_utm_array(mdt_filled, mdt_transform_wgs, "EPSG:4326", dem_meta)
    log.info(
        f"MDT over DEM domain: "
        f"mean={float(np.nanmean(mdt_on_dem)):.4f} m, "
        f"range=[{float(np.nanmin(mdt_on_dem)):.4f}, {float(np.nanmax(mdt_on_dem)):.4f}] m"
    )
    topo_utm[valid_topo] -= mdt_on_dem[valid_topo]
    del mdt_np, mdt_filled, mdt_on_dem

    # Diagnostic delta for the correction plot
    delta_correction = np.where(valid_topo, topo_utm - topo_egm2008, np.nan).astype(np.float32)
    del topo_egm2008
    log.info(
        f"Datum correction complete — Δ MSL correction: "
        f"mean={np.nanmean(delta_correction):.4f} m, "
        f"std={np.nanstd(delta_correction):.4f} m, "
        f"range=[{np.nanmin(delta_correction):.4f}, {np.nanmax(delta_correction):.4f}] m"
    )

log.info(
    f"{topo_source}: {(~np.isnan(topo_utm)).sum():,} valid px "
    f"({100 * (~np.isnan(topo_utm)).mean():.1f} %)"
)

# ── 2. GEBCO on UTM working grid ──────────────────────────────────────────────
log.info(f"Loading GEBCO from {bathymetry_path}…")
gebco_utm = np.full((h_utm, w_utm), np.nan, dtype=np.float32)

if bathymetry_path.is_dir():
    bbox_geom  = _box(*wgs84_bounds)
    candidates = [str(fp) for fp in sorted(bathymetry_path.glob("*.tif"))
                  if _tile_intersects(fp, bbox_geom)]
    if not candidates:
        raise FileNotFoundError(
            f"No GEBCO tiles found overlapping domain in {bathymetry_path}"
        )
    log.info(f"  {len(candidates)} GEBCO tile(s) overlap domain")
    open_ds = [rasterio.open(p) for p in candidates]
    try:
        gebco_data, gebco_transform = rio_merge_mem(open_ds, bounds=wgs84_bounds)
        gebco_nodata = open_ds[0].nodata
        gebco_crs    = open_ds[0].crs
    finally:
        for ds in open_ds:
            ds.close()
    gebco_arr = gebco_data[0].astype(np.float32)
    if gebco_nodata is not None:
        gebco_arr[gebco_arr == np.float32(gebco_nodata)] = np.nan
    _outside = rio_geom_mask(
        [domain_poly.__geo_interface__],
        out_shape=gebco_arr.shape, transform=gebco_transform, all_touched=True,
    )
    gebco_arr[_outside] = np.nan
    rio_reproject(
        source=gebco_arr, destination=gebco_utm,
        src_transform=gebco_transform, src_crs=gebco_crs,
        dst_transform=transform_utm, dst_crs=utm_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan, dst_nodata=np.nan,
    )
else:
    with rasterio.open(bathymetry_path) as src:
        _nd_g = float(src.nodata) if src.nodata is not None else -9999.0
        _g_masked, _g_transform = rio_mask(
            src, [domain_poly.__geo_interface__],
            crop=False, nodata=_nd_g, all_touched=True,
        )
        gebco_arr = _g_masked[0].astype(np.float32)
        gebco_arr[gebco_arr == np.float32(_nd_g)] = np.nan
        rio_reproject(
            source=gebco_arr, destination=gebco_utm,
            src_transform=_g_transform, src_crs=src.crs,
            dst_transform=transform_utm, dst_crs=utm_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan, dst_nodata=np.nan,
        )

log.info(f"GEBCO valid: {(~np.isnan(gebco_utm)).sum():,} px")
gebco_original = gebco_utm.copy()

# ── 3. Land-polygon mask ──────────────────────────────────────────────────────
land_gdf = gpd.read_file(land_polygons_path).to_crs(utm_crs)
if land_gdf.empty:
    land_mask = np.zeros((h_utm, w_utm), dtype=bool)
    log.warning("No land polygons — using pure GEBCO")
else:
    land_mask = rio_rasterize(
        shapes=[(geom, 1) for geom in land_gdf.geometry if geom is not None],
        out_shape=(h_utm, w_utm), transform=transform_utm,
        fill=0, dtype=np.uint8, all_touched=False,
    ).astype(bool)
log.info(f"Land mask: {land_mask.sum():,} / {land_mask.size:,} px "
         f"({100*land_mask.mean():.1f} %)")

# ── Fill land NaN with impassable-barrier value ───────────────────────────────
# Two types of NaN can exist inside the land polygon at this point:
#   (a) Cells that were ≥ clip_elevation_m and clamped to NaN.
#   (b) Genuine gaps where no DEM tile exists.
# Both must be filled so GEBCO cannot replace them with low/negative bathymetric
# values that would allow water to route through high-elevation barriers.
_land_nan = land_mask & np.isnan(topo_utm)
if _land_nan.any():
    topo_utm[_land_nan] = np.float32(land_barrier_val)
    log.info(
        f"Filled {_land_nan.sum():,} land NaN px with {land_barrier_val:.0e} m "
        f"(clamp artefacts or missing tiles → impassable barrier)"
    )
del _land_nan

# ── 4. Hard-boundary merge ────────────────────────────────────────────────────
# Coastal DEM on land, GEBCO in ocean — hard cutoff, no gradient transition.
# This is elevation_merged.tif: the product used by all downstream rules.
_topo_valid = ~np.isnan(topo_utm)
merged_hard = gebco_original.copy()
merged_hard[land_mask & _topo_valid] = topo_utm[land_mask & _topo_valid]
_remaining = np.isnan(merged_hard)
if _remaining.any():
    fillnodata(merged_hard, mask=~_remaining,
               max_search_distance=float(max(merged_hard.shape)))
log.info(
    f"Hard-boundary merge: min={np.nanmin(merged_hard):.2f} m, "
    f"max={np.nanmax(merged_hard):.2f} m, "
    f"nan={np.isnan(merged_hard).sum():,} px"
)
del _topo_valid, _remaining

# ── 5. Optional gradient blend ────────────────────────────────────────────────
merged = None
blend_weight_plot = None
if blend_enabled:
    from scipy.ndimage import distance_transform_edt

    log.info(f"Blend buffer: {blend_buffer_m} m = {buffer_px:.1f} px at {work_res_m} m/px")

    dist_f64 = distance_transform_edt(~land_mask)
    buffered_land_mask = dist_f64 <= buffer_px

    blend_weight = dist_f64.astype(np.float32, copy=False)
    del dist_f64
    buf_f32 = np.float32(buffer_px)
    blend_weight /= buf_f32
    blend_weight *= np.float32(-1.0)
    blend_weight += np.float32(1.0)
    np.clip(blend_weight, np.float32(0.0), np.float32(1.0), out=blend_weight)
    blend_weight_plot = blend_weight.copy()

    topo_blend = topo_utm.copy()
    topo_blend[~land_mask] = np.nan
    topo_valid = ~np.isnan(topo_blend)

    gebco_blend = gebco_utm.copy()
    gebco_blend[buffered_land_mask] = np.nan
    gebco_valid = ~np.isnan(gebco_blend)

    fillnodata(topo_blend, mask=topo_valid, max_search_distance=buffer_px + 1.0)
    beyond_buffer = np.isnan(topo_blend) & ~topo_valid
    topo_blend[beyond_buffer] = np.float32(0.0)

    fillnodata(gebco_blend, mask=gebco_valid, max_search_distance=buffer_px + 1.0)
    gebco_reach = ~np.isnan(gebco_blend)
    gebco_blend[~gebco_reach] = np.float32(0.0)

    merged = np.empty((h_utm, w_utm), dtype=np.float32)
    np.multiply(blend_weight, topo_blend, out=merged)
    del topo_blend
    np.subtract(np.float32(1.0), blend_weight, out=blend_weight)
    np.multiply(blend_weight, gebco_blend, out=blend_weight)
    np.add(merged, blend_weight, out=merged)
    del blend_weight, gebco_blend

    # `topo_blend`/`gebco_blend` placeholder fills (0.0, used only so the blend
    # arithmetic stays finite where one source has no data) must not leak into
    # the output with nonzero weight — otherwise pixels with no relevant
    # elevation data render as if elevation ≈ 0 instead of nodata.
    no_data_px = (
        (beyond_buffer & (blend_weight_plot > 0))
        | (~gebco_reach & (blend_weight_plot < 1))
    )
    merged[no_data_px] = np.nan
    del beyond_buffer, gebco_reach, no_data_px

    remaining = np.isnan(merged)
    if remaining.any():
        log.info(f"Gradient blend final fill: {remaining.sum():,} domain-edge NaN px")
        fillnodata(merged, mask=~remaining, max_search_distance=float(max(merged.shape)))

    log.info(
        f"Gradient blend: min={np.nanmin(merged):.2f} m, "
        f"max={np.nanmax(merged):.2f} m, "
        f"nan={np.isnan(merged).sum():,} px"
    )

# ── 6. Write outputs ──────────────────────────────────────────────────────────
NODATA = np.float32(-9999.0)
out_elev_path.parent.mkdir(parents=True, exist_ok=True)

with rasterio.open(out_elev_path, "w", **dem_meta) as dst:
    dst.write(np.where(np.isnan(merged_hard), NODATA, merged_hard).astype(np.float32), 1)
log.info(f"Written: {out_elev_path}")

if blend_enabled and merged is not None:
    out_gradient_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_gradient_path, "w", **dem_meta) as dst:
        dst.write(np.where(np.isnan(merged), NODATA, merged).astype(np.float32), 1)
    log.info(f"Written (gradient): {out_gradient_path}")

zsini = np.where(land_mask, NODATA, np.float32(0.0))
zsini[np.isnan(merged_hard)] = NODATA
with rasterio.open(out_zsini_path, "w", **dem_meta) as dst:
    dst.write(zsini.astype(np.float32), 1)
log.info(f"Written: {out_zsini_path}")

# ── 7. Diagnostic plots ───────────────────────────────────────────────────────
plot_elevation_merged(
    merged_path=str(out_elev_path),
    bbox_poly=_box(*wgs84_bounds),
    osm_land_path=str(land_polygons_path),
    output_path=str(out_plot_path),
    title_str=f"Merged elevation ({topo_source} on land, GEBCO offshore)",
    clip_elevation_m=clip_elevation_m,
)
log.info(f"Plot written: {out_plot_path}")

plot_zsini(
    zsini_path=str(out_zsini_path),
    bbox_poly=_box(*wgs84_bounds),
    osm_land_path=str(land_polygons_path),
    output_path=str(out_plot_zsini_path),
)
log.info(f"Plot written: {out_plot_zsini_path}")

if blend_enabled and merged is not None and blend_weight_plot is not None:
    plot_elevation_blending(
        blend_weight=blend_weight_plot,
        merged=merged,
        gebco_original=gebco_original,
        land_mask=land_mask,
        utm_crs_str=domain_crs_str,
        transform=transform_utm,
        osm_land_path=str(land_polygons_path),
        output_path=out_plot_gradient_path,
    )
    log.info(f"Plot written: {out_plot_gradient_path}")

if datum_correction_enabled and delta_correction is not None:
    from src.plots import plot_datum_correction_delta
    out_plot_datum_path = str(out_elev_path.parent.parent / "plots" / "01b_datum_correction.png")
    plot_datum_correction_delta(
        delta         = delta_correction,
        utm_crs_str   = domain_crs_str,
        wgs84_bounds  = wgs84_bounds,
        osm_land_path = str(land_polygons_path),
        output_path   = out_plot_datum_path,
    )
    log.info(f"Plot written: {out_plot_datum_path}")

profiler.stop()
log.info("Done")
