"""Identify the existing flood protection standard of a delta (FLOPROS x WRI geounits).

Returns the dominant (largest-area) geounit's riverine and coastal design
return periods; the boundary-forcings rule subtracts the matching
discharge/water level, since that protection is not represented in the DEM
or any other input.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import shapely.vectorized
import xarray as xr
from shapely.geometry import Polygon

log = logging.getLogger(__name__)


def load_flopros_table(path: str) -> pd.DataFrame:
    """Load the FLOPROS-x-geogunit_107 protection-standard table.

    Indexed by geounit id (FID_Aque, matching the wri_geogunit_107 raster's
    pixel values); 'Riverine'/'Coastal' are design return periods (years),
    NaN where no formal standard is known (the common case, not an error).
    """
    df = pd.read_excel(path, usecols=["FID_Aque", "ISO", "Riverine", "Coastal"])
    df = df.set_index("FID_Aque")
    return df


def load_geogunit_iso_lookup(path: str) -> pd.Series:
    """Load the geounit-id -> ISO3 lookup (two unnamed columns: index, ISO)."""
    df = pd.read_excel(path, header=None, names=["geounit_id", "ISO"])
    return df.set_index("geounit_id")["ISO"]


def _dominant_geounit_pixel_counts(
    delta_polygon_wgs84: Polygon,
    geogunit_raster_path: str,
) -> tuple[dict[int, int], int]:
    lon_min, lat_min, lon_max, lat_max = delta_polygon_wgs84.bounds
    with xr.open_dataset(geogunit_raster_path) as ds:
        # lat/lon are both ascending in wri_geogunit_107, so a plain slice() works.
        sel = ds["Geogunits"].sel(
            lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
        )
        arr = sel.values.astype(np.float64)
        lon_vals = sel["lon"].values
        lat_vals = sel["lat"].values

    if arr.size == 0:
        return {}, 0

    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    # xarray CF-decodes the raster's _FillValue (-9999) to NaN, so isfinite excludes fill pixels.
    inside = shapely.vectorized.contains(delta_polygon_wgs84, lon_grid, lat_grid)
    valid = inside & np.isfinite(arr)

    if not valid.any():
        return {}, 0

    ids, counts = np.unique(arr[valid].astype(np.int64), return_counts=True)
    return dict(zip(ids.tolist(), counts.tolist())), int(valid.sum())


def _resolve_rp(
    own_value: float,
    other_value: float,
    default_rp_yr: float,
    max_rp_yr: float,
) -> tuple[float, str]:
    if pd.notna(own_value):
        rp, source = float(own_value), "flopros"
    elif pd.notna(other_value):
        rp, source = float(other_value), "flopros_other_hazard_fallback"
    else:
        rp, source = float(default_rp_yr), "default"
    return min(rp, float(max_rp_yr)), source


def identify_dominant_protection(
    delta_polygon_wgs84: Polygon,
    geogunit_raster_path: str,
    flopros_df: pd.DataFrame,
    iso_lookup: pd.Series,
    default_rp_yr: float = 2.0,
    max_rp_yr: float = 1000.0,
) -> dict:
    """Resolve riverine/coastal design return periods of the dominant geounit in a delta.

    Dominant = largest pixel count inside the polygon (EPSG:4326). A missing
    FLOPROS value falls back to the other hazard's, then to default_rp_yr;
    both are capped at max_rp_yr. With no valid pixels, both use default_rp_yr.
    Returns a flat JSON-serialisable dict; 'riverine_rp_yr'/'coastal_rp_yr'
    are always populated.
    """
    counts, n_valid = _dominant_geounit_pixel_counts(
        delta_polygon_wgs84, geogunit_raster_path
    )

    if not counts:
        log.warning(
            "identify_dominant_protection: no valid geounit pixels found inside the "
            "delta polygon -- falling back to default_rp_yr for both hazards"
        )
        return {
            "dominant_geounit_id": None,
            "dominant_iso": None,
            "pixel_count": 0,
            "pixel_fraction": 0.0,
            "n_geounits_in_delta": 0,
            "riverine_rp_raw": None,
            "coastal_rp_raw": None,
            "riverine_rp_yr": float(default_rp_yr),
            "coastal_rp_yr": float(default_rp_yr),
            "riverine_source": "default",
            "coastal_source": "default",
        }

    dominant_id = max(counts, key=counts.get)
    dominant_count = counts[dominant_id]

    row = flopros_df.loc[dominant_id] if dominant_id in flopros_df.index else None
    riverine_raw = (
        float(row["Riverine"])
        if row is not None and pd.notna(row["Riverine"])
        else None
    )
    coastal_raw = (
        float(row["Coastal"]) if row is not None and pd.notna(row["Coastal"]) else None
    )

    riverine_rp, riverine_source = _resolve_rp(
        riverine_raw, coastal_raw, default_rp_yr, max_rp_yr
    )
    coastal_rp, coastal_source = _resolve_rp(
        coastal_raw, riverine_raw, default_rp_yr, max_rp_yr
    )

    dominant_iso = iso_lookup.get(dominant_id)
    dominant_iso = str(dominant_iso) if pd.notna(dominant_iso) else None

    log.info(
        f"identify_dominant_protection: dominant geounit {dominant_id} ({dominant_iso}), "
        f"{dominant_count}/{n_valid} px ({100 * dominant_count / n_valid:.1f}%) of "
        f"{len(counts)} geounit(s) in delta polygon -- "
        f"riverine RP={riverine_rp:.1f} yr ({riverine_source}), "
        f"coastal RP={coastal_rp:.1f} yr ({coastal_source})"
    )

    return {
        "dominant_geounit_id": int(dominant_id),
        "dominant_iso": dominant_iso,
        "pixel_count": int(dominant_count),
        "pixel_fraction": float(dominant_count / n_valid),
        "n_geounits_in_delta": len(counts),
        "riverine_rp_raw": riverine_raw,
        "coastal_rp_raw": coastal_raw,
        "riverine_rp_yr": riverine_rp,
        "coastal_rp_yr": coastal_rp,
        "riverine_source": riverine_source,
        "coastal_source": coastal_source,
    }
