"""
05a_get_elevation.py — Build the merged elevation product for a basin.

Processing chain
----------------
1.  Merge FathomDEM tiles for the domain, reprojected to the UTM working grid
    with a coverage-weighted ("nan-aware") resampling that avoids eroding real
    data near nodata edges (tile gaps, domain-polygon boundary).
2.  Vertical datum correction, EGM2008 → GOCO06s: add the geoid offset
    N_EGM2008 − N_GOCO06s (pyshtools synthesis) to every valid FathomDEM pixel.
    Mandatory — FathomDEM's native EGM2008 datum must not be blended with
    GEBCO/COAST-RP/MDT, which all share the GOCO06s frame. No further
    correction (e.g. MDT subtraction) is applied to the DEM itself: MDT is
    only meaningful at sea, and the merge below only ever needs land-side DEM
    values corrected to the same geoid GEBCO is re-referenced to.
3.  Clip GEBCO to the domain, resample to the working UTM grid, and
    re-reference it to GOCO06s by subtracting the MDT (HYBRID-CNES-CLS2022) —
    extrapolated over land via inverse-distance weighting before resampling,
    so the subtraction doesn't NaN-poison nearshore GEBCO pixels whose
    receptive field straddles a land-side nodata cell.
4.  Hard merge: FathomDEM wherever it has valid data, GEBCO everywhere else
    (no land-polygon mask, no gradient blend — FathomDEM's own nodata pattern
    is what decides which product a pixel gets). A final defensive
    fillnodata covers the rare case where neither source has data for a
    pixel inside the domain.
5.  Write elevation_merged.tif.
6.  Diagnostic elevation map, plus (always) the geoid-offset, MDT, and
    datum-correction delta maps for both the DEM and GEBCO corrections.

zsini.tif is NOT produced here — it needs the land/water-body distinction
(OSM land polygons + landuse), which rule get_landuse (05b) already computes
for its own reprojection step, so the full zsini is built there instead of
splitting the land-mask computation across two rules.
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
from rasterio.merge import merge as rio_merge_mem
from rasterio.transform import from_origin as _rast_from_origin
from shapely.geometry import box as _box

from src.domain import load_domain
from src.log import setup_logging
from src.plots import plot_elevation_merged
from src.profiling import ScriptProfiler
from src.raster import (
    _tile_intersects,
    compute_geoid_offset_arr,
    find_fathomdem_tiles,
    merge_tiled_raster,
    reproject_nan_aware,
)

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
find_fathomdem_tiles = profiler.wrap(find_fathomdem_tiles)
merge_tiled_raster    = profiler.wrap(merge_tiled_raster)

# ── params ────────────────────────────────────────────────────────────────────
domain_meta_path   = Path(snakemake.input.spec_basins_meta)
topo_tiles_dir     = snakemake.input.global_topography_tiles
bathymetry_path    = Path(snakemake.input.global_bathymetry)
land_polygons_path = Path(snakemake.input.land_polygons)
goco_path           = Path(snakemake.input.goco06s_gfc)
egm_path             = Path(snakemake.input.egm2008_gfc)
mdt_path             = snakemake.input.mdt

out_elev_path  = Path(snakemake.output.elevation_merged)
out_plot_path  = snakemake.output.plot_elevation
# Directory for the extra diagnostic plots below (geoid offset, MDT ocean,
# datum-correction deltas) — derived from the tracked plot_elevation output
# rather than out_elev_path's own directory, since those two no longer share
# a parent (elevation_merged.tif lives under inputs/domain/, plots live
# under visuals/input_data/).
plots_dir = Path(out_plot_path).parent

mdt_variable         = "mdt"  # sole variable in data_catalogue's mdt_cnes_cls22 source
mdt_load_margin_deg  = float(snakemake.params.mdt_load_margin_deg)

log.info("Topography source: FathomDEM (mandatory EGM2008 → GOCO06s datum correction)")

# ── domain ────────────────────────────────────────────────────────────────────
wgs84_bounds, domain_crs_str, domain_poly = load_domain(
    domain_meta_path, snakemake.input.domain_gpkg
)
utm_crs = CRS.from_string(domain_crs_str)
log.info(f"Domain WGS84 bounds: {wgs84_bounds}, UTM CRS: {utm_crs.to_string()}")

# ── 1. Merge FathomDEM tiles ──────────────────────────────────────────────────
tiles = find_fathomdem_tiles(topo_tiles_dir, wgs84_bounds)
if not tiles:
    raise FileNotFoundError(
        "No FathomDEM tiles found for domain. Check the data catalogue path."
    )
log.info(f"Found {len(tiles)} FathomDEM tile(s)")

# Working resolution: auto-derived from FathomDEM's own native pixel size
# (fixed at 1 arcsecond globally, see data_catalogue.yml) rather than a
# separately configured value. Converted to metres via the north-south
# (latitude) WGS84 arc-length constant -- this is what "1 arcsecond ~= 30 m"
# already refers to, and (unlike the east-west/longitude direction, which
# shrinks toward the poles) barely varies with latitude, so every basin gets
# a consistent, predictable pixel size regardless of location.
with rasterio.open(tiles[0]) as _tile_src:
    native_deg_res = abs(_tile_src.res[1])
work_res_m = native_deg_res * 111_320.0
log.info(
    f"Working resolution auto-derived from FathomDEM native pixel size "
    f"({native_deg_res:.6f} deg): {work_res_m:.2f} m"
)

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
    f"UTM output grid: {w_utm}×{h_utm} px @ {work_res_m:.2f} m, "
    f"origin ({_b['xmin']:.1f}, {_b['ymax']:.1f}) [{domain_crs_str}]"
)

# ── domain-polygon mask (UTM working grid) ────────────────────────────────────
# topo_utm/gebco_utm are clipped to domain_poly (the actual delta polygon, not
# its bounding box) below -- fillnodata() has no notion of "outside the study
# area" vs. "a real internal gap", so this mask is needed to re-clamp after
# every fillnodata call.
domain_poly_utm = gpd.GeoSeries([domain_poly], crs="EPSG:4326").to_crs(utm_crs).iloc[0]
outside_domain = rio_geom_mask(
    [domain_poly_utm.__geo_interface__],
    out_shape=(h_utm, w_utm), transform=transform_utm, all_touched=True,
)

dem_meta = dict(
    driver="GTiff", dtype="float32",
    width=w_utm, height=h_utm, count=1,
    crs=utm_crs, transform=transform_utm,
    nodata=-9999.0, compress="deflate", tiled=True,
)

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
        # FathomDEM tiles store elevation as int32 centimeters (raster tag
        # VERTICAL_UNITS=centimeters), not meters.
        topo_arr = topo_arr / 100.0
        topo_src_crs = src.crs

    # Coverage-weighted resampling: a destination pixel only goes to NaN when
    # its entire receptive field is nodata, instead of plain bilinear eroding
    # real elevation right at tile gaps / the domain-polygon boundary.
    topo_utm = reproject_nan_aware(
        topo_arr, _masked_transform, topo_src_crs,
        (h_utm, w_utm), transform_utm, utm_crs,
        resampling=Resampling.bilinear,
    )
finally:
    os.unlink(tmp_topo)

log.info(
    f"FathomDEM: {(~np.isnan(topo_utm)).sum():,} valid px "
    f"({100 * (~np.isnan(topo_utm)).mean():.1f} %)"
)

# ── 2. Vertical datum correction — EGM2008 → GOCO06s (mandatory) ─────────────
topo_egm2008 = topo_utm.copy()   # snapshot before correction for the delta plot
valid_topo = ~np.isnan(topo_utm)

log.info(f"Computing EGM2008→GOCO06s geoid offset from {goco_path.name} / {egm_path.name} …")
offset_arr, offset_transform, offset_crs = compute_geoid_offset_arr(goco_path, egm_path)
offset_on_dem = reproject_nan_aware(
    offset_arr.astype(np.float32), offset_transform, offset_crs,
    (h_utm, w_utm), transform_utm, utm_crs,
)
log.info(
    f"Geoid offset over domain: "
    f"[{offset_on_dem[valid_topo].min():.4f}, {offset_on_dem[valid_topo].max():.4f}] m"
)
topo_utm[valid_topo] += offset_on_dem[valid_topo]
del offset_on_dem

from src.plots import plot_geoid_offset
_out_geoid = str(plots_dir / "05a_elevation_geoid_offset.png")
plot_geoid_offset(
    offset_arr       = offset_arr,
    offset_transform = offset_transform,
    wgs84_bounds     = wgs84_bounds,
    osm_land_path    = str(land_polygons_path),
    output_path      = _out_geoid,
)
log.info(f"Plot written: {_out_geoid}")
del offset_arr

delta_correction = np.where(valid_topo, topo_utm - topo_egm2008, np.nan).astype(np.float32)
del topo_egm2008
log.info(
    f"DEM datum correction complete (EGM2008 → GOCO06s) — Δ: "
    f"mean={np.nanmean(delta_correction):.4f} m, "
    f"std={np.nanstd(delta_correction):.4f} m, "
    f"range=[{np.nanmin(delta_correction):.4f}, {np.nanmax(delta_correction):.4f}] m"
)

# ── 3. GEBCO on UTM working grid ──────────────────────────────────────────────
log.info(f"Loading GEBCO from {bathymetry_path}…")

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
    # Coverage-weighted resampling -- same rationale as the DEM in step 1.
    gebco_utm = reproject_nan_aware(
        gebco_arr, gebco_transform, gebco_crs,
        (h_utm, w_utm), transform_utm, utm_crs,
        resampling=Resampling.bilinear,
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
        gebco_utm = reproject_nan_aware(
            gebco_arr, _g_transform, src.crs,
            (h_utm, w_utm), transform_utm, utm_crs,
            resampling=Resampling.bilinear,
        )

log.info(f"GEBCO valid: {(~np.isnan(gebco_utm)).sum():,} px")

# ── 3b. Vertical datum correction for GEBCO (subtract MDT → GOCO06s) ─────────
# Re-references GEBCO to the same GOCO06s geoid as FathomDEM (step 2) by
# subtracting the MDT (HYBRID-CNES-CLS22) -- extrapolated over land via
# inverse-distance weighting before resampling, so the subtraction doesn't
# NaN-poison nearshore GEBCO pixels whose receptive field straddles a
# land-side nodata cell. Applied to every valid GEBCO pixel (not just at
# sea) since the hard merge below may also fall back to GEBCO on land where
# FathomDEM has no data.
import xarray as xr

gebco_before_mdt = gebco_utm.copy()   # snapshot before correction for the delta plot
valid_gebco = ~np.isnan(gebco_utm)

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
_out_mdt = str(plots_dir / "05a_elevation_mdt_ocean.png")
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
# an ocean MDT cell, leaving land-side GEBCO gap-fill pixels unconnected from
# the sea level information — the key fix vs. naive bilinear resampling of
# the raw MDT.
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

mdt_on_gebco = reproject_nan_aware(
    mdt_filled, mdt_transform_wgs, "EPSG:4326",
    (h_utm, w_utm), transform_utm, utm_crs,
)
log.info(
    f"MDT over GEBCO domain: "
    f"mean={float(np.nanmean(mdt_on_gebco)):.4f} m, "
    f"range=[{float(np.nanmin(mdt_on_gebco)):.4f}, {float(np.nanmax(mdt_on_gebco)):.4f}] m"
)
gebco_utm[valid_gebco] -= mdt_on_gebco[valid_gebco]
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

# ── 4. Hard merge: FathomDEM wherever valid, GEBCO everywhere else ───────────
# No land-polygon mask, no gradient blend -- FathomDEM's own nodata pattern
# (not an independent land/sea classification) decides which source a pixel
# gets.
_topo_valid = ~np.isnan(topo_utm)
merged = np.where(_topo_valid, topo_utm, gebco_utm)
_remaining = np.isnan(merged)
if _remaining.any():
    # Defensive fallback: neither FathomDEM nor GEBCO had data for these
    # pixels (should be rare -- GEBCO is globally complete).
    fillnodata(merged, mask=~_remaining,
               max_search_distance=float(max(merged.shape)))
# fillnodata's mask only distinguishes "has data" from "no data" -- with no
# notion of "outside domain_poly", an effectively unlimited max_search_distance
# would otherwise extrapolate valid elevation all the way out to the
# rectangular bbox edges. Re-clamp to NaN outside the polygon regardless of how
# far the fill reached.
merged[outside_domain] = np.nan
log.info(
    f"Hard merge (FathomDEM valid={_topo_valid.sum():,} px, GEBCO fallback "
    f"elsewhere): min={np.nanmin(merged):.2f} m, max={np.nanmax(merged):.2f} m, "
    f"nan={np.isnan(merged).sum():,} px"
)
del _remaining

# ── 5. Write outputs ──────────────────────────────────────────────────────────
NODATA = np.float32(-9999.0)
out_elev_path.parent.mkdir(parents=True, exist_ok=True)

with rasterio.open(out_elev_path, "w", **dem_meta) as dst:
    dst.write(np.where(np.isnan(merged), NODATA, merged).astype(np.float32), 1)
log.info(f"Written: {out_elev_path}")

# ── 6. Diagnostic plots ───────────────────────────────────────────────────────
plot_elevation_merged(
    merged_path=str(out_elev_path),
    bbox_poly=_box(*wgs84_bounds),
    osm_land_path=str(land_polygons_path),
    output_path=str(out_plot_path),
    title_str="Merged elevation (FathomDEM where available, GEBCO elsewhere)",
)
log.info(f"Plot written: {out_plot_path}")

from src.plots import plot_datum_correction_delta

out_plot_datum_path = str(plots_dir / "05a_elevation_datum_correction.png")
plot_datum_correction_delta(
    delta          = delta_correction,
    utm_crs_str    = domain_crs_str,
    wgs84_bounds   = wgs84_bounds,
    osm_land_path  = str(land_polygons_path),
    output_path    = out_plot_datum_path,
    title          = "Vertical datum correction — FathomDEM\n(EGM2008 → GOCO06s geoid)",
    colorbar_label = "Δ elevation (m)  [GOCO06s − EGM2008]",
)
log.info(f"Plot written: {out_plot_datum_path}")

out_plot_gebco_datum_path = str(plots_dir / "05a_elevation_gebco_correction.png")
plot_datum_correction_delta(
    delta          = gebco_delta_correction,
    utm_crs_str    = domain_crs_str,
    wgs84_bounds   = wgs84_bounds,
    osm_land_path  = str(land_polygons_path),
    output_path    = out_plot_gebco_datum_path,
    title          = "Vertical datum correction — GEBCO\n(raw → GOCO06s via MDT_CNES-CLS22 subtraction)",
    colorbar_label = "Δ elevation (m)  [GOCO06s − raw GEBCO]",
)
log.info(f"Plot written: {out_plot_gebco_datum_path}")

profiler.stop()
log.info("Done")
