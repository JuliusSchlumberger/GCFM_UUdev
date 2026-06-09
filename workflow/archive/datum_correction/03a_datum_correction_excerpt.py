# Archived: 2026-06-05
# Source: workflow/scripts/03a_get_elevation.py
#
# These code blocks performed the vertical datum re-referencing of DiluviumDEM:
#   EGM2008 → GOCO06s geoid shift  +  MDT_CNES-CLS22 subtraction  → local MSL
#
# Reason removed: DiluviumDEM in its native EGM2008 datum is close enough for
# coastal flood modelling at the scales used in this pipeline.  The correction
# added computational cost (pyshtools, xarray, large input files) without a
# demonstrable improvement in model performance.
#
# Companion archives in this directory:
#   compute_geoid_offset_arr.py  — src/raster.py function (the pyshtools synthesis)
#   plot_datum_correction_delta.py — src/plots.py function (diagnostic map)
#
# To restore: re-integrate the blocks below into 03a_get_elevation.py at the
# positions indicated by the section comments, and restore the inputs/params
# in workflow/rules/03_static_model_data.smk (see git history).

# ── additional inputs needed (add back to rule + script params) ───────────────
# goco_path          = Path(snakemake.input.goco06s_gfc)
# egm_path           = Path(snakemake.input.egm2008_gfc)
# mdt_path           = snakemake.input.mdt
# mdt_variable        = snakemake.params.mdt_variable
# mdt_load_margin_deg = float(snakemake.params.mdt_load_margin_deg)
# out_plot_datum_path = snakemake.output.plot_datum_correction

# ── additional imports needed ─────────────────────────────────────────────────
import xarray as xr  # noqa: F401  (for MDT loading)
from rasterio.transform import from_origin  # noqa: F401  (for MDT transform)
from rasterio.warp import reproject as _rp, Resampling as _RS  # noqa: F401
from src.raster import compute_geoid_offset_arr  # noqa: F401  (archived; see companion file)
from src.plots import plot_datum_correction_delta  # noqa: F401  (archived; see companion file)


# ── helper: resample any WGS84 array onto the UTM dem_meta grid ───────────────
# (defined inline in 03a; only used for geoid offset and MDT)
def _resample_to_grid(src_arr, src_transform, src_crs, dst_meta):
    dest = np.empty((dst_meta["height"], dst_meta["width"]), dtype=np.float32)
    _rp(
        source=src_arr,
        destination=dest,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_meta["transform"],
        dst_crs=dst_meta["crs"],
        resampling=_RS.bilinear,
    )
    return dest


# ── INSERTION POINT: after DiluviumDEM is reprojected to diluvium_utm ─────────
# Save EGM2008-referenced values before datum correction — used for the delta plot.
diluvium_egm2008 = diluvium_utm.copy()

# Step 1: + geoid offset (EGM2008 → GOCO06s)
log.info("Computing EGM2008→GOCO06s geoid offset (pyshtools)…")
offset_arr, offset_transform, offset_crs = compute_geoid_offset_arr(goco_path, egm_path)
offset_on_dem = _resample_to_grid(offset_arr, offset_transform, offset_crs, dem_meta)
valid_dil = ~np.isnan(diluvium_utm)
diluvium_utm[valid_dil] += offset_on_dem[valid_dil]
log.info(
    f"Geoid offset over domain: "
    f"[{offset_on_dem[valid_dil].min():.4f}, {offset_on_dem[valid_dil].max():.4f}] m"
)
del offset_arr, offset_on_dem

# Step 2: − MDT (GOCO06s → MSL)
log.info(f"Subtracting MDT (variable='{mdt_variable}') from {mdt_path}")
lon_min, lat_min, lon_max, lat_max = wgs84_bounds
with xr.open_dataset(mdt_path) as mdt_ds:
    mdt_da = mdt_ds[mdt_variable]
    lat_dim = next(d for d in mdt_da.dims if "lat" in d.lower())
    lon_dim = next(d for d in mdt_da.dims if "lon" in d.lower())
    for extra in [d for d in mdt_da.dims if d not in (lat_dim, lon_dim)]:
        mdt_da = mdt_da.isel({extra: 0})
    if float(mdt_da[lon_dim].max()) > 180:
        mdt_da = mdt_da.assign_coords(
            {
                lon_dim: xr.where(
                    mdt_da[lon_dim] > 180, mdt_da[lon_dim] - 360, mdt_da[lon_dim]
                )
            }
        ).sortby(lon_dim)
    margin = mdt_load_margin_deg
    mdt_clip = mdt_da.sel(
        {
            lat_dim: slice(lat_min - margin, lat_max + margin),
            lon_dim: slice(lon_min - margin, lon_max + margin),
        }
    )
    mdt_np = mdt_clip.values.astype(np.float32)
    mdt_lats = mdt_clip[lat_dim].values
    mdt_lons = mdt_clip[lon_dim].values
dlat = float(np.abs(mdt_lats[0] - mdt_lats[1]))
dlon = float(np.abs(mdt_lons[0] - mdt_lons[1]))
if mdt_lats[0] < mdt_lats[-1]:
    mdt_np = mdt_np[::-1]
    mdt_lats = mdt_lats[::-1]
mdt_transform = from_origin(
    float(mdt_lons[0]) - dlon / 2, float(mdt_lats[0]) + dlat / 2, dlon, dlat
)
mdt_on_dem = _resample_to_grid(mdt_np, mdt_transform, "EPSG:4326", dem_meta)
mdt_valid = ~np.isnan(mdt_on_dem)
diluvium_utm[valid_dil & mdt_valid] -= mdt_on_dem[valid_dil & mdt_valid]
del mdt_np, mdt_on_dem

log.info(
    f"DiluviumDEM after corrections: {valid_dil.sum():,} valid px "
    f"({100 * valid_dil.mean():.1f} %)"
)

# ── INSERTION POINT: after land NaN fill, before hard-boundary merge ──────────
# Compute datum correction delta for diagnostic plot.
# delta = (MSL elevation) - (EGM2008 elevation), restricted to land pixels
# where DiluviumDEM had valid data.
delta_correction = np.where(
    land_mask & ~np.isnan(diluvium_egm2008),
    diluvium_utm - diluvium_egm2008,
    np.nan,
).astype(np.float32)
del diluvium_egm2008
log.info(
    f"Datum correction delta: mean={np.nanmean(delta_correction):.4f} m, "
    f"std={np.nanstd(delta_correction):.4f} m, "
    f"range=[{np.nanmin(delta_correction):.4f}, {np.nanmax(delta_correction):.4f}] m"
)

# ── INSERTION POINT: in the diagnostic-plots section ─────────────────────────
plot_datum_correction_delta(
    delta=delta_correction,
    utm_crs_str=domain_crs_str,
    wgs84_bounds=wgs84_bounds,
    osm_land_path=str(land_polygons_path),
    output_path=str(out_plot_datum_path),
)
log.info(f"Plot written: {out_plot_datum_path}")
