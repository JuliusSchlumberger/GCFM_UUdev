"""Existing flood-protection-standard identification (FLOPROS x WRI geounits).

Identifies the dominant (largest-area) administrative unit's design flood
protection return period inside a delta polygon, separately for riverine and
coastal hazards, so that the boundary-forcings rule can subtract the
corresponding discharge/water-level from the forcing timeseries -- water
below an already-existing protection standard never reaches the model
domain in practice, and that standard is not represented in the DEM or any
other input this pipeline otherwise uses.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import shapely.vectorized
import xarray as xr
from shapely.geometry import Polygon

log = logging.getLogger(__name__)

_HAZARDS = ("Riverine", "Coastal")


def load_flopros_table(path: str) -> pd.DataFrame:
    """
    Load the FLOPROS-x-geogunit_107 protection-standard table.

    Returns a DataFrame indexed by geounit id (FID_Aque, matching the
    wri_geogunit_107 raster's pixel values) with 'Riverine'/'Coastal'
    columns (design return period, years; NaN where no formal standard is
    known for that hazard in that unit -- the common case, not an error).
    """
    df = pd.read_excel(path, usecols=["FID_Aque", "ISO", "Riverine", "Coastal"])
    df = df.set_index("FID_Aque")
    return df


def load_geogunit_iso_lookup(path: str) -> pd.Series:
    """
    Load the geounit-id -> ISO3 lookup (two unnamed columns: index, ISO).

    Returns a Series indexed by geounit id, values = ISO3 country code.
    """
    df = pd.read_excel(path, header=None, names=["geounit_id", "ISO"])
    return df.set_index("geounit_id")["ISO"]


def _dominant_geounit_pixel_counts(
    delta_polygon_wgs84: Polygon,
    geogunit_raster_path: str,
) -> tuple[dict[int, int], int]:
    """
    Count pixels per geounit id inside ``delta_polygon_wgs84``, via a small
    windowed read of the global geounit raster clipped to the polygon's own
    bbox (never loads the full global array -- xarray's netCDF backend only
    reads the sliced window from disk).

    Returns (counts_by_geounit_id, n_valid_pixels). Both empty/zero if the
    polygon's window contains no valid (non-fill-value) pixels.
    """
    lon_min, lat_min, lon_max, lat_max = delta_polygon_wgs84.bounds
    with xr.open_dataset(geogunit_raster_path) as ds:
        # lat/lon are both ascending in wri_geogunit_107 -- a plain slice()
        # works without the descending-coordinate handling load_mdt() needs.
        sel = ds["Geogunits"].sel(
            lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
        )
        arr = sel.values.astype(np.float64)
        lon_vals = sel["lon"].values
        lat_vals = sel["lat"].values

    if arr.size == 0:
        return {}, 0

    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    # xarray CF-decodes the raster's _FillValue (-9999) to NaN on read, so
    # np.isfinite already excludes fill-value pixels -- no separate nodata check needed.
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
    """
    Apply the missing-value fallback chain (own hazard -> other hazard ->
    default) and clamp to max_rp_yr. Returns (resolved_rp, source_label).
    """
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
    """
    Identify the dominant (largest pixel count) protection geounit inside
    ``delta_polygon_wgs84`` and resolve its riverine/coastal design return
    periods, applying the missing-value fallback chain and RP cap.

    Args:
        delta_polygon_wgs84: Delta polygon (EPSG:4326), e.g. from rule
                              split_delta_polygons's delta_polygon.gpkg.
        geogunit_raster_path: Path to the wri_geogunit_107 raster.
        flopros_df:           Output of load_flopros_table().
        iso_lookup:           Output of load_geogunit_iso_lookup().
        default_rp_yr:        Assumed protection RP when neither hazard has
                               a FLOPROS value for the dominant unit.
        max_rp_yr:            Cap applied to the resolved RP for both
                               hazards.

    Returns:
        Flat dict (JSON-serialisable) with the dominant geounit id/ISO,
        pixel counts, and resolved 'riverine_rp_yr'/'coastal_rp_yr' (always
        populated) plus the raw FLOPROS values and fallback source labels.
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
