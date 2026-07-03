"""Surge boundary forcing: station selection, time series construction, dataset assembly."""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)

# Fixed return periods tabulated in COAST-RP (storm_tide_rp_{rp:04d} variables).
_COASTRP_RPS = (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000)


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


# ── vertical-reference correction & SLR fingerprint ─────────────────────────────
# The MDT-based vertical correction and SLR-fingerprint scaling below adapt the
# methodology developed by Natalia Aleksandrova (notebooks
# 01_retrieve_MDT_correction.ipynb, 02_get_SLR_fingerprint.ipynb,
# 03_combine_wl_data_scenarios.ipynb) for direct use in this pipeline.


def _nearest_valid_grid(
    da: xr.DataArray,
    lon_dim: str,
    lon: float,
    lat_dim: str,
    lat: float,
    fallback_deg: float,
) -> float:
    """
    Look up the value of a 2-D lat/lon grid nearest (lon, lat), falling back
    to the nearest non-NaN cell within +/-fallback_deg if the nearest cell
    itself is NaN.

    Ports the ``find_nearest_valid`` helper from Natalia Aleksandrova's
    MDT-correction notebook (credited there to
    https://github.com/pydata/xarray/issues/644). `da` must have ascending
    lat/lon coordinates (see load_mdt).
    """
    val = float(da.sel({lon_dim: lon, lat_dim: lat}, method="nearest").values)
    if not np.isnan(val):
        return val

    window = da.sel(
        {
            lon_dim: slice(lon - fallback_deg, lon + fallback_deg),
            lat_dim: slice(lat - fallback_deg, lat + fallback_deg),
        }
    )
    if window.size == 0:
        return np.nan

    values = window.values
    valid = ~np.isnan(values)
    if not valid.any():
        return np.nan

    lons2d, lats2d = np.meshgrid(window[lon_dim].values, window[lat_dim].values)
    dist2 = (lons2d - lon) ** 2 + (lats2d - lat) ** 2
    dist2 = np.where(valid, dist2, np.inf)
    idx = np.unravel_index(np.argmin(dist2), dist2.shape)
    return float(values[idx])


def load_mdt(mdt_path: str, mdt_variable: str = "mdt") -> xr.DataArray:
    """
    Load the AVISO MDT_CNES-CLS22 'mean dynamic topography' field as a 2-D
    lat/lon DataArray with ascending coordinates (any extra dimensions, e.g.
    'time', are dropped by selecting their first index; longitudes are
    remapped from 0..360 to -180..180 if needed).
    """
    with xr.open_dataset(mdt_path) as ds:
        da = ds[mdt_variable].load()

    lat_dim = next(d for d in da.dims if "lat" in d.lower())
    lon_dim = next(d for d in da.dims if "lon" in d.lower())
    for extra in [d for d in da.dims if d not in (lat_dim, lon_dim)]:
        da = da.isel({extra: 0})

    if float(da[lon_dim].max()) > 180:
        da = da.assign_coords(
            {lon_dim: xr.where(da[lon_dim] > 180, da[lon_dim] - 360, da[lon_dim])}
        )
    return da.sortby([lat_dim, lon_dim])


def apply_mdt_correction(
    stations: gpd.GeoDataFrame,
    mdt_da: xr.DataArray,
    fallback_deg: float = 3.0,
) -> gpd.GeoDataFrame:
    """
    Look up the AVISO MDT_CNES-CLS22 value nearest each station and record it
    alongside a geoid-referenced 'rp_level'.

    Adds columns:
        rp_level_raw: copy of the original 'rp_level' (local-MSL-referenced,
                       per the COAST-RP source documentation).
        mdt:          MDT value at the station (m; NaN if no valid cell was
                       found within +/-fallback_deg).
        rp_level:     rp_level_raw - mdt (local MSL -> GOCO06s geoid), matching
                       the sign convention used to re-reference GEBCO to
                       GOCO06s (gebco -= mdt) in 05a_get_elevation.py. The
                       caller decides whether to keep this or revert to
                       rp_level_raw based on vertical_correction.enabled.
    """
    lat_dim = next(d for d in mdt_da.dims if "lat" in d.lower())
    lon_dim = next(d for d in mdt_da.dims if "lon" in d.lower())

    result = stations.copy()
    result["rp_level_raw"] = result["rp_level"]
    result["mdt"] = [
        _nearest_valid_grid(mdt_da, lon_dim, geom.x, lat_dim, geom.y, fallback_deg)
        for geom in result.geometry
    ]
    result["rp_level"] = result["rp_level_raw"] - result["mdt"]
    return result


def load_slr_fingerprint(
    slr_root: str,
    ssp_scenario: str,
    confidence_level: str,
    year: int,
    quantile: float,
) -> xr.Dataset:
    """
    Load the IPCC AR6 sea-level-change field for one SSP/confidence/year/quantile.

    Returns a Dataset with 'lat', 'lon', and 'sea_level_change' (m) over the
    combined gauge + 1deg x 1deg grid 'locations' dimension.
    """
    import os

    path = os.path.join(
        slr_root,
        f"{confidence_level}_confidence",
        ssp_scenario,
        f"total_{ssp_scenario}_{confidence_level}_confidence_values.nc",
    )
    with xr.open_dataset(path) as ds:
        sel = ds.sel(years=year, quantiles=quantile)
        return xr.Dataset(
            {"sea_level_change": sel["sea_level_change"] / 1000.0},  # mm -> m
            coords={"lat": sel["lat"], "lon": sel["lon"]},
        ).load()


def compute_global_mean_slr(slr_ds: xr.Dataset) -> float:
    """
    Global-mean SLR (m) over valid 1deg x 1deg grid cells — the scaling
    reference ('mean_ori' in the source notebook) used to derive each
    station's fingerprint.
    """
    lat = slr_ds["lat"].values
    lon = slr_ds["lon"].values
    on_grid = np.isclose(lat % 1.0, 0.0) & np.isclose(lon % 1.0, 0.0)
    return float(np.nanmean(slr_ds["sea_level_change"].values[on_grid]))


def _nearest_valid_location(
    lons: np.ndarray,
    lats: np.ndarray,
    vals: np.ndarray,
    lon: float,
    lat: float,
    fallback_deg: float,
) -> float:
    """
    Return the value at the nearest (lon, lat) location with a non-NaN value,
    searching outward up to +/-fallback_deg. Generalises
    _nearest_valid_grid() to the AR6 dataset's irregular combined
    gauge + grid 'locations' array.
    """
    dist2 = (lons - lon) ** 2 + (lats - lat) ** 2
    max_dist2 = fallback_deg**2
    for idx in np.argsort(dist2):
        if dist2[idx] > max_dist2:
            break
        if not np.isnan(vals[idx]):
            return float(vals[idx])
    return np.nan


def apply_slr_fingerprint(
    stations: gpd.GeoDataFrame,
    slr_ds: xr.Dataset,
    global_mean_slr: float,
    slr_m: float,
    fallback_deg: float = 3.0,
) -> gpd.GeoDataFrame:
    """
    Scale the AR6 SLR fingerprint (local / global-mean SLR) by slr_m and add
    it to each station's 'rp_level'.

    Adds columns:
        slr_fingerprint: local SLR / global_mean_slr at the nearest AR6
                         location (1.0 — i.e. the uniform global value — if
                         no valid location was found within +/-fallback_deg).
        slr_m:           slr_fingerprint * slr_m (m), added to 'rp_level'.
    """
    lons = slr_ds["lon"].values
    lats = slr_ds["lat"].values
    vals = slr_ds["sea_level_change"].values

    result = stations.copy()
    fingerprints = []
    for geom in result.geometry:
        local_slr = _nearest_valid_location(
            lons, lats, vals, geom.x, geom.y, fallback_deg
        )
        fingerprints.append(1.0 if np.isnan(local_slr) else local_slr / global_mean_slr)
    result["slr_fingerprint"] = fingerprints
    result["slr_m"] = result["slr_fingerprint"] * slr_m
    result["rp_level"] = result["rp_level"] + result["slr_m"]
    return result


# ── station selection ─────────────────────────────────────────────────────────


def load_coastrp_stations(
    nc_path: str,
    return_period: int,
) -> gpd.GeoDataFrame:
    """
    Load CoastRP surge stations from a NetCDF file as a GeoDataFrame.

    Also loads every fixed return period tabulated in COAST-RP (_COASTRP_RPS)
    into 'rp_raw_{rp:04d}' columns, alongside the configured 'rp_level' --
    these extra columns are cheap (same already-open file) and let
    interpolate_protection_level() look up an arbitrary protection return
    period later, for whichever stations survive selection/dedup (those
    functions only filter rows, they don't drop unknown columns).

    Args:
        nc_path:       Path to the CoastRP NetCDF file.
        return_period: Return period (years) to extract; the variable
                       ``storm_tide_rp_{return_period:04d}`` is read into
                       'rp_level'.

    Returns:
        GeoDataFrame with Point geometries (EPSG:4326), an 'rp_level' column
        containing the configured return-period storm-tide level (m), and
        one 'rp_raw_{rp:04d}' column per entry in _COASTRP_RPS (raw, i.e.
        before any vertical/SLR correction).
    """
    import xarray as xr

    rp_var = f"storm_tide_rp_{return_period:04d}"
    with xr.open_dataset(nc_path) as ds:
        lons = ds["station_x_coordinate"].values
        lats = ds["station_y_coordinate"].values
        rp_vals = ds[rp_var].values
        raw_rp_vals = {rp: ds[f"storm_tide_rp_{rp:04d}"].values for rp in _COASTRP_RPS}
    # Some CoastRP stations carry NaN/fill-value coordinates; points built from
    # them produce NaN geometries that make shapely.distance() emit
    # "invalid value encountered in distance" warnings downstream.
    valid = np.isfinite(lons) & np.isfinite(lats)
    if not valid.all():
        log.debug(
            f"Dropping {int((~valid).sum())} surge station(s) with invalid coordinates"
        )
    data = {"rp_level": rp_vals[valid]}
    for rp, vals in raw_rp_vals.items():
        data[f"rp_raw_{rp:04d}"] = vals[valid]
    return gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(lons[valid], lats[valid]),
        crs="EPSG:4326",
    )


def interpolate_protection_level(
    stations: gpd.GeoDataFrame,
    target_rp_yr: float,
) -> pd.Series:
    """
    Linearly interpolate each station's raw storm-tide level at an arbitrary
    return period, between the two bracketing fixed RPs tabulated in
    COAST-RP (_COASTRP_RPS; rp_raw_{rp:04d} columns from
    load_coastrp_stations()).

    target_rp_yr is clamped to [min(_COASTRP_RPS), max(_COASTRP_RPS)] --
    callers needing a wider cap (e.g. top-level protection_levels.
    max_rp_yr) should apply it before calling this, but COAST-RP itself
    cannot extrapolate past its own tabulated range regardless.

    Args:
        stations:      GeoDataFrame with 'rp_raw_{rp:04d}' columns for every
                       rp in _COASTRP_RPS (from load_coastrp_stations()).
        target_rp_yr:  Return period (years) to interpolate at.

    Returns:
        pd.Series (same index as ``stations``) of the raw (uncorrected)
        interpolated storm-tide level (m).
    """
    rp = float(np.clip(target_rp_yr, min(_COASTRP_RPS), max(_COASTRP_RPS)))

    lo_rp = max(r for r in _COASTRP_RPS if r <= rp)
    hi_rp = min(r for r in _COASTRP_RPS if r >= rp)

    lo_vals = stations[f"rp_raw_{lo_rp:04d}"].astype(float)
    if lo_rp == hi_rp:
        return lo_vals.copy()

    hi_vals = stations[f"rp_raw_{hi_rp:04d}"].astype(float)
    frac = (rp - lo_rp) / (hi_rp - lo_rp)
    return lo_vals + frac * (hi_vals - lo_vals)


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
    baseline_m: float = 0.0,
    station_baselines: np.ndarray | None = None,
) -> xr.Dataset:
    """
    Assemble the surge forcing xr.Dataset with synthetic sinusoidal time series.

    Each station receives a half-cosine wave rising from its lead-period
    baseline to its MDT-corrected RP water level (``rp_level``) and back.

    Datum note: GEBCO is re-referenced to GOCO06s by subtracting MDT in rule
    03a, so local MSL maps to −MDT in model coordinates.  When
    ``station_baselines`` is provided each station uses its own local MSL
    (−mdt_i + slr_m_i) as the wave baseline, so the surge amplitude equals
    exactly ``rp_level_raw`` (the COAST-RP storm-tide above calm water)
    regardless of how MDT varies spatially across the selected stations.
    ``baseline_m`` (the mean of those per-station values) is still stored in
    the dataset so rule 13 can initialise sea cells at the same vertical
    reference (zsini_baseline.tif).

    Args:
        stations:          GeoDataFrame with 'rp_level' and 'dist_m' columns and
                           Point geometries in EPSG:4326.
        times:             Shared time axis from build_time_axis() (hours).
        lead_days:         Lead-in duration before wave onset (days).
        period_hr:         Wave period (hours).
        return_period:     Return period label written to the 'rp_level' metadata.
        baseline_m:        Mean vertical correction applied to rp_level (m).
                           Equals mean(−MDT + SLR) across selected stations.
                           Stored in the dataset so rule 13 can initialise sea
                           cells (zsini_baseline.tif).  Defaults to 0.0.
        station_baselines: Per-station lead-period flat values (m), length
                           equal to ``len(stations)``.  Each entry is the
                           station's own local MSL in model coordinates
                           (= rp_level_i − rp_level_raw_i = −mdt_i + slr_m_i).
                           When None, ``baseline_m`` is used for all stations.

    Returns:
        xr.Dataset with dimensions (station, time) and coordinates
        longitude / latitude / time.
    """
    if station_baselines is not None:
        surge_matrix = np.stack(
            [
                sinusoidal_wave(
                    float(station_baselines[i]),
                    float(row["rp_level"]),
                    times,
                    lead_days,
                    period_hr,
                )
                for i, (_, row) in enumerate(stations.iterrows())
            ]
        )
    else:
        surge_matrix = np.stack(
            [
                sinusoidal_wave(
                    baseline_m, float(row["rp_level"]), times, lead_days, period_hr
                )
                for _, row in stations.iterrows()
            ]
        )
    ds = xr.Dataset(
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

    ds["baseline_m"] = (
        [],
        float(baseline_m),
        {
            "units": "m",
            "long_name": (
                "Mean vertical correction applied as lead-period baseline "
                "(mean(−MDT + SLR) across selected stations = calm sea level "
                "in model coordinates). Read by rule 13 to initialise sea cells "
                "via zsini_baseline.tif.  Equals 0.0 when both corrections are off."
            ),
        },
    )

    if station_baselines is not None:
        ds["station_baseline"] = (
            ["station"],
            station_baselines,
            {
                "units": "m",
                "long_name": (
                    "Per-station lead-period flat value (−mdt_i + slr_m_i). "
                    "Local MSL for each station in model coordinates. "
                    "Ensures surge amplitude = rp_level_raw per station."
                ),
            },
        )

    # Optional provenance from the MDT vertical correction and SLR fingerprint
    # (src.surge.apply_mdt_correction / apply_slr_fingerprint), if present.
    extra_station_vars = {
        "rp_level_raw": {
            "units": "m",
            "long_name": "RP storm tide level before vertical/SLR correction",
        },
        "mdt": {
            "units": "m",
            "long_name": "MDT correction applied (local MSL -> geoid)",
        },
        "slr_fingerprint": {
            "units": "1",
            "long_name": "AR6 SLR fingerprint (local / global-mean SLR)",
        },
        "slr_m": {"units": "m", "long_name": "SLR contribution applied to rp_level"},
    }
    for col, attrs in extra_station_vars.items():
        if col in stations.columns:
            ds[col] = (["station"], stations[col].values, attrs)

    return ds
