"""Surge boundary forcing: station selection, time series construction, dataset assembly."""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import xarray as xr

log = logging.getLogger(__name__)


# ── time series primitives ────────────────────────────────────────────────────


def build_time_axis(
    lead_days: float,
    period_hr: float,
    dt_hr: float,
    total_hr: float | None = None,
) -> np.ndarray:
    """
    Build an evenly-spaced time axis covering a lead-in period and one wave cycle.

    The axis represents hours since simulation start and is shared by both the
    surge and river forcing datasets.

    Args:
        lead_days:  Constant lead-in duration before the wave begins (days).
        period_hr:  Duration of the wave cycle (hours).
        dt_hr:      Time step size (hours).
        total_hr:   Optional override for the total axis length (hours).  When
                    provided the axis extends to ``total_hr`` instead of the
                    default ``lead_days * 24 + period_hr``.  Used to pad the
                    shorter of the two forcing time series so both span the same
                    duration; ``sinusoidal_wave`` holds the baseline value for
                    any time steps beyond the wave's own period.

    Returns:
        1-D float array of time values in hours.
    """
    if total_hr is None:
        total_hr = lead_days * 24.0 + period_hr
    n_steps = round(total_hr / dt_hr)
    return np.arange(n_steps + 1, dtype=float) * dt_hr


def sinusoidal_wave(
    baseline: float,
    peak: float,
    times: np.ndarray,
    lead_days: float,
    period_hr: float,
) -> np.ndarray:
    """
    Generate a synthetic forcing time series: constant lead-in then one cosine wave.

    The wave rises from `baseline` to `peak` and returns to `baseline` over one
    period, modelled as a raised cosine (half-period of a full cosine).  This
    represents a stylised storm surge or flood event.

    Args:
        baseline:   Constant value during the lead-in and at wave start/end.
        peak:       Maximum value at the wave crest.
        times:      Time axis from build_time_axis() (hours since simulation start).
        lead_days:  Lead-in duration before wave onset (days).
        period_hr:  Wave period (hours).

    Returns:
        1-D array of forcing values, same length as `times`.
    """
    t_wave_start = lead_days * 24.0
    values = np.full(len(times), baseline)
    mask = (times >= t_wave_start) & (times <= t_wave_start + period_hr)
    t_local = times[mask] - t_wave_start
    values[mask] = baseline + (peak - baseline) * 0.5 * (
        1.0 - np.cos(2.0 * np.pi * t_local / period_hr)
    )
    return values


# ── station selection ─────────────────────────────────────────────────────────


def load_coastrp_stations(
    nc_path: str,
    return_period: int,
) -> gpd.GeoDataFrame:
    """
    Load CoastRP surge stations from a NetCDF file as a GeoDataFrame.

    Args:
        nc_path:       Path to the CoastRP NetCDF file.
        return_period: Return period (years) to extract; the variable
                       ``storm_tide_rp_{return_period:04d}`` is read.

    Returns:
        GeoDataFrame with Point geometries (EPSG:4326) and an 'rp_level' column
        containing the return-period storm-tide level (m).
    """
    import xarray as xr

    rp_var = f"storm_tide_rp_{return_period:04d}"
    with xr.open_dataset(nc_path) as ds:
        lons = ds["station_x_coordinate"].values
        lats = ds["station_y_coordinate"].values
        rp_vals = ds[rp_var].values
    # Some CoastRP stations carry NaN/fill-value coordinates; points built from
    # them produce NaN geometries that make shapely.distance() emit
    # "invalid value encountered in distance" warnings downstream.
    valid = np.isfinite(lons) & np.isfinite(lats)
    if not valid.all():
        log.debug(
            f"Dropping {int((~valid).sum())} surge station(s) with invalid coordinates"
        )
    return gpd.GeoDataFrame(
        {"rp_level": rp_vals[valid]},
        geometry=gpd.points_from_xy(lons[valid], lats[valid]),
        crs="EPSG:4326",
    )


def compute_distances_to_bbox(
    stations: gpd.GeoDataFrame,
    bbox_utm: gpd.GeoDataFrame,
    domain_crs: str,
) -> gpd.GeoDataFrame:
    """
    Compute the metric distance from each station to the domain bbox boundary.

    Args:
        stations:   GeoDataFrame of surge stations (any CRS).
        bbox_utm:   Single-row GeoDataFrame of the domain bbox in the UTM CRS.
        domain_crs: UTM CRS string used for metric distance computation.

    Returns:
        Copy of `stations` with an additional 'dist_m' column (metres).
    """
    boundary = bbox_utm.geometry.iloc[0].exterior
    stations_utm = stations.to_crs(domain_crs)
    result = stations.copy()
    result["dist_m"] = stations_utm.geometry.distance(boundary).values
    return result


def select_nearest_stations(
    stations: gpd.GeoDataFrame,
    min_stations: int,
    max_stations: int,
    search_radii_km: list[float],
    dedupe_radius_km: float,
    domain_crs: str,
) -> gpd.GeoDataFrame:
    """
    Return the closest stations within an expanding search radius, deduplicated
    and capped at `max_stations`.

    1. Iterates through `search_radii_km` until at least `min_stations` stations
       are found within the current radius.  Falls back to the `min_stations`
       nearest stations if no radius is sufficient.
    2. Drops near-duplicates: among any cluster of stations mutually within
       `dedupe_radius_km` of each other, keeps only the one closest to the
       domain boundary (processing candidates in ascending `dist_m` order
       guarantees that the first station kept in a cluster is the closest one).
    3. If more than `max_stations` remain, keeps the `max_stations` closest
       to the boundary.

    Args:
        stations:         GeoDataFrame with a 'dist_m' column and Point
                          geometry (from compute_distances_to_bbox()).
        min_stations:     Minimum number of stations to include.
        max_stations:     Maximum number of stations to include.
        search_radii_km:  Ordered list of search radii to try (km).
        dedupe_radius_km: Stations within this distance (km) of an
                          already-kept, closer station are dropped as
                          near-duplicates.
        domain_crs:       UTM CRS string used for metric distance computation
                          between stations.

    Returns:
        Filtered copy of `stations`, ordered by ascending distance to the
        domain boundary.
    """
    for radius_km in search_radii_km:
        radius_m = radius_km * 1000.0
        candidates = stations[stations["dist_m"] <= radius_m]
        if len(candidates) >= min_stations:
            log.info(
                f"Found {len(candidates)} surge stations within {radius_km:.0f} km"
            )
            break
    else:
        candidates = stations.nsmallest(min_stations, "dist_m")
        log.warning(
            f"Could not reach {min_stations} stations within "
            f"{search_radii_km[-1]:.0f} km; using {len(candidates)} closest stations"
        )

    candidates = candidates.sort_values("dist_m")
    candidates_utm = candidates.to_crs(domain_crs)

    dedupe_radius_m = dedupe_radius_km * 1000.0
    kept_geoms = []
    keep_mask = []
    for geom in candidates_utm.geometry:
        is_duplicate = any(
            geom.distance(kept) <= dedupe_radius_m for kept in kept_geoms
        )
        keep_mask.append(not is_duplicate)
        if not is_duplicate:
            kept_geoms.append(geom)

    selected = candidates[keep_mask]
    n_dropped = len(candidates) - len(selected)
    if n_dropped:
        log.info(
            f"Dropped {n_dropped} near-duplicate station(s) within "
            f"{dedupe_radius_km:.1f} km of a closer station"
        )

    if len(selected) > max_stations:
        log.info(f"Capping at {max_stations} closest stations (had {len(selected)})")
        selected = selected.nsmallest(max_stations, "dist_m")

    return selected.copy()


# ── dataset assembly ──────────────────────────────────────────────────────────


def build_surge_dataset(
    stations: gpd.GeoDataFrame,
    times: np.ndarray,
    lead_days: float,
    period_hr: float,
    return_period: int,
) -> xr.Dataset:
    """
    Assemble the surge forcing xr.Dataset with synthetic sinusoidal time series.

    Each station receives a half-cosine wave rising from 0 m to its RP water
    level and back, after a constant lead-in.

    Args:
        stations:      GeoDataFrame with 'rp_level' and 'dist_m' columns and
                       Point geometries in EPSG:4326.
        times:         Shared time axis from build_time_axis() (hours).
        lead_days:     Lead-in duration before wave onset (days).
        period_hr:     Wave period (hours).
        return_period: Return period label written to the 'rp_level' metadata.

    Returns:
        xr.Dataset with dimensions (station, time) and coordinates
        longitude / latitude / time.
    """
    surge_matrix = np.stack(
        [
            sinusoidal_wave(0.0, float(row["rp_level"]), times, lead_days, period_hr)
            for _, row in stations.iterrows()
        ]
    )
    return xr.Dataset(
        {
            "water_level": (
                ["station", "time"],
                surge_matrix,
                {"units": "m", "long_name": "storm tide water level"},
            ),
            "rp_level": (
                ["station"],
                stations["rp_level"].values,
                {"units": "m", "long_name": f"RP{return_period} storm tide level"},
            ),
            "distance_m": (
                ["station"],
                stations["dist_m"].values,
                {"units": "m", "long_name": "distance from domain boundary"},
            ),
        },
        coords={
            "longitude": (
                ["station"],
                stations.geometry.x.values,
                {"units": "degrees_east"},
            ),
            "latitude": (
                ["station"],
                stations.geometry.y.values,
                {"units": "degrees_north"},
            ),
            "time": (["time"], times, {"units": "hours since simulation start"}),
        },
    )
