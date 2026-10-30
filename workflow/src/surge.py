"""Surge boundary forcing: station selection, time series construction, dataset assembly."""

from __future__ import annotations

import json
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from src.raster import latlon_dims

log = logging.getLogger(__name__)

# Fixed return periods tabulated in COAST-RP (storm_tide_rp_{rp:04d} variables).
_COASTRP_RPS = (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000)


def build_time_axis(
    lead_days: float,
    period_hr: float,
    dt_hr: float,
    total_hr: float | None = None,
) -> np.ndarray:
    """Build an evenly spaced time axis (hours since simulation start) for lead-in + one wave.

    Shared by the surge and river forcing. total_hr overrides the default
    length lead_days * 24 + period_hr, to pad the shorter forcing to the
    other's duration.
    """
    if total_hr is None:
        total_hr = lead_days * 24.0 + period_hr
    n_steps = round(total_hr / dt_hr)
    return np.arange(n_steps + 1, dtype=float) * dt_hr


# Every absolute coastal water level and every weir crest is kept to the
# centimetre (2 decimals): levels rounded to nearest, crests rounded UP.
LEVEL_DECIMALS = 2
# Smallest crest step; an exceedance below it is rounding, not overtopping.
CREST_PRECISION_M = 10.0**-LEVEL_DECIMALS


def round_level(values) -> np.ndarray:
    """Round a coastal water level to the nearest centimetre.

    Used for every absolute coastal level (calm sea, storm tide at an RP);
    weir crests are rounded UP instead (ceil_crest). NaN passes through;
    -0.0 is returned as 0.0.
    """
    return np.round(np.asarray(values, dtype=float), LEVEL_DECIMALS) + 0.0


def ceil_crest(values) -> np.ndarray:
    """Round a weir crest UP to the next centimetre, never below its source value.

    sfincs.weir is written at the same precision (src.sfincs_run.write_weir_file),
    so a crest is stored exactly. The inner np.round strips float noise so a
    value on the grid (2.75 stored as 2.7500000000000004) is not bumped 1 cm.
    NaN passes through; -0.0 is returned as 0.0.
    """
    steps = np.ceil(np.round(np.asarray(values, dtype=float) / CREST_PRECISION_M, 6))
    return np.round(steps * CREST_PRECISION_M, LEVEL_DECIMALS) + 0.0


def sinusoidal_wave(
    baseline: float,
    peak: float,
    times: np.ndarray,
    lead_days: float,
    period_hr: float,
) -> np.ndarray:
    """Return a constant lead-in followed by one raised-cosine wave from baseline to peak.

    times in hours since simulation start; the value holds baseline outside
    the wave period.
    """
    t_wave_start = lead_days * 24.0
    values = np.full(len(times), baseline)
    mask = (times >= t_wave_start) & (times <= t_wave_start + period_hr)
    t_local = times[mask] - t_wave_start
    values[mask] = baseline + (peak - baseline) * 0.5 * (
        1.0 - np.cos(2.0 * np.pi * t_local / period_hr)
    )
    return values


def extract_hydrograph_components(
    hydrograph_path: str,
    station_lons: np.ndarray,
    station_lats: np.ndarray,
    window_hr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read per-station average tide and per-RP surge shapes from storm_tide_hydrographs.nc.

    Tide has high water at t=0; shapes are normalised (peak=1 at t=0), cut to
    +-window_hr, matched to stations by nearest coordinates. Levels are
    relative to a fixed local MSL (COAST-RP datum); MDT/SLR are added at build
    time. Returns (hg_time_hr (n_t,), tide (n_station, n_t), shape
    (n_station, n_rp, n_t), shape_rps_yr (n_rp,), match_km (n_station,)).
    Raises ValueError if the file was built with another window_hr.
    """
    with xr.open_dataset(hydrograph_path) as ds:
        hx = ds["longitude"].values
        hy = ds["latitude"].values
        t = ds["time_hr"].values
        win = np.abs(t) <= window_hr + 1e-9
        tide_all = ds["average_tide"].values[:, win]
        shape_all = ds["surge_shape"].values[:, :, win]
        shape_rps_yr = ds["rp_yr"].values.astype(float)
    # The shapes are derived FOR their window (events are cut to it), so a
    # file built with another window is not the same shapes cut shorter.
    if abs(t[0] + window_hr) > 1e-6 or abs(t[-1] - window_hr) > 1e-6:
        raise ValueError(
            f"hydrograph file spans {t[0]:.1f}..{t[-1]:.1f} h but this basin's event window is "
            f"+-{window_hr:g} h -- it was built with another window_hr; rerun rule storm_tide_hydrographs"
        )

    n = len(station_lons)
    tide = np.full((n, int(win.sum())), np.nan)
    shape = np.full((n, len(shape_rps_yr), int(win.sum())), np.nan)
    match_km = np.full(n, np.nan)
    for i, (lon, lat) in enumerate(zip(station_lons, station_lats)):
        d = np.hypot((hx - lon) * np.cos(np.radians(lat)), hy - lat) * 111.0
        j = int(np.nanargmin(d))
        tide[i], shape[i], match_km[i] = tide_all[j], shape_all[j], d[j]
    return t[win], tide, shape, shape_rps_yr, match_km


def surge_shape_at_rp(surge_ds: xr.Dataset, rp_yr: float) -> np.ndarray:
    """Return the normalised surge shape (n_station, n_hg_time) at an arbitrary RP.

    Linear in RP between the stored COAST-RP periods (hg_rp_yr), clamped to the
    tabulated range; the result stays normalised (peak = 1 at t = 0).
    """
    shape = surge_ds["hg_surge_shape"].values
    rps_yr = surge_ds["hg_rp_yr"].values.astype(float)
    clamped_rp_yr = float(np.clip(float(rp_yr), rps_yr.min(), rps_yr.max()))
    hi = int(np.searchsorted(rps_yr, clamped_rp_yr))
    if np.isclose(rps_yr[hi], clamped_rp_yr):
        return shape[:, hi]
    lo = hi - 1
    frac = (clamped_rp_yr - rps_yr[lo]) / (rps_yr[hi] - rps_yr[lo])
    return (1.0 - frac) * shape[:, lo] + frac * shape[:, hi]


def periodic_tide(rel_hr: np.ndarray, hg_t: np.ndarray, tide: np.ndarray) -> np.ndarray:
    """Return the average tide at rel_hr, repeated with its own period outside the stored window.

    The period is the lag of the tide's highest autocorrelation between 11 and
    26 h (12.42 h semidiurnal, 24.84 h diurnal or mixed); beyond either end of
    the window the last (first) period is repeated, so the tide carries on
    through a long river flood instead of holding its edge value.
    """
    rel_hr = np.asarray(rel_hr, dtype=float)
    dt = float(np.median(np.diff(hg_t)))
    lags = np.arange(int(np.ceil(11.0 / dt)), int(np.floor(26.0 / dt)) + 1)
    lags = lags[lags < len(tide) - 1]
    if lags.size == 0:
        return np.interp(rel_hr, hg_t, tide)
    a = tide - tide.mean()
    corr = [np.dot(a[:-k], a[k:]) / max(np.dot(a, a), 1e-12) for k in lags]
    period = float(lags[int(np.argmax(corr))] * dt)
    t0, t1 = float(hg_t[0]), float(hg_t[-1])
    folded = np.where(
        rel_hr > t1,
        t1 - period + np.mod(rel_hr - t1, period),
        np.where(rel_hr < t0, t0 + np.mod(rel_hr - t0, period), rel_hr),
    )
    return np.interp(folded, hg_t, tide)


def storm_tide_event(
    surge_ds: xr.Dataset,
    rel_hr: np.ndarray,
    peak_level: np.ndarray | None,
    slr_m: float = 0.0,
    rp_yr: float | None = None,
) -> np.ndarray:
    """Build the storm-tide hydrograph (n_station, len(rel_hr)) in m.

    rel_hr = 0 is the surge peak on tidal high water. Per station:

      tide  = average tide + station calm level (MDT, + SLR x fingerprint)
      event = tide + A x surge_shape_at_rp(rp_yr), A = peak_level - tide high
              water (>= 0), so the event peaks at peak_level exactly.

    Shared by the production boundary (rule 14) and the crest calibration
    (rule 11), so crests are calibrated against the wave production sends.
    peak_level: absolute level per station (same slr_m), or None for tide
    only; rp_yr is required with it. Outside the stored window the tide
    repeats with its own period (periodic_tide) and the surge is 0.
    """
    hg_t = surge_ds["hg_time_hr"].values
    tide = surge_ds["hg_tide_m"].values
    base = _station_calm_levels_unrounded(surge_ds, slr_m)
    event = (
        np.stack([periodic_tide(rel_hr, hg_t, tide[i]) for i in range(len(base))])
        + base[:, None]
    )
    if peak_level is not None:
        if rp_yr is None:
            raise ValueError(
                "storm_tide_event: rp_yr is required with peak_level (selects the surge shape)"
            )
        shape = surge_shape_at_rp(surge_ds, rp_yr)
        high_water = tide.max(axis=1) + base
        amp = np.maximum(np.asarray(peak_level, dtype=float) - high_water, 0.0)
        surge = np.stack(
            [
                np.interp(rel_hr, hg_t, shape[i], left=0.0, right=0.0)
                for i in range(len(base))
            ]
        )
        event = event + amp[:, None] * surge
    return event


def _hydrograph_surge_matrix(
    surge_ds: xr.Dataset, rp_yr: float | str, slr_m: float
) -> np.ndarray:
    # Window t=0 sits at attrs["hg_peak_hr"]. Up to the
    # event start (the restart, lead_days) the boundary holds the calm level
    # the spin-up ends at; over attrs["hg_ramp_hr"] from there it blends into
    # the event (tide, and the surge in its window) with the half-cosine
    # weight, so the event starts from the restart without a step or a kink.
    times = surge_ds["time"].values
    peak_hr = float(surge_ds.attrs["hg_peak_hr"])
    ramp_hr = float(surge_ds.attrs.get("hg_ramp_hr", 0.0))
    is_tide = isinstance(rp_yr, str) and rp_yr.strip().lower() == "tide"
    level = None if is_tide else lookup_storm_tide_at_rp(surge_ds, rp_yr, slr_m=slr_m)
    event = storm_tide_event(
        surge_ds,
        times - peak_hr,
        level,
        slr_m=slr_m,
        rp_yr=None if is_tide else float(rp_yr),
    )

    start_hr = float(surge_ds.attrs["lead_days"]) * 24.0
    s = (
        np.clip((times - start_hr) / ramp_hr, 0.0, 1.0)
        if ramp_hr > 0
        else (times >= start_hr).astype(float)
    )
    w = 0.5 * (1.0 - np.cos(np.pi * s))
    calm = calm_sea_levels(surge_ds, slr_m)
    return (1.0 - w) * calm[:, None] + w * event


def _station_calm_levels_unrounded(surge_ds: xr.Dataset, slr_m: float) -> np.ndarray:
    # station_calm_sea_level_m (MDT-only; scalar calm_sea_level_m if absent) + slr_m x slr_fingerprint.
    baselines = (
        surge_ds["station_calm_sea_level_m"].values
        if "station_calm_sea_level_m" in surge_ds
        else np.full(
            surge_ds.sizes["station"], float(surge_ds["calm_sea_level_m"].values)
        )
    )
    if slr_m != 0.0 and "slr_fingerprint" in surge_ds:
        baselines = baselines + surge_ds["slr_fingerprint"].values * slr_m
    return baselines


def calm_sea_levels(surge_ds: xr.Dataset, slr_m: float = 0.0) -> np.ndarray:
    """Return the per-station calm-sea (no surge) lead-in level (m), rounded to 1 cm."""
    return round_level(_station_calm_levels_unrounded(surge_ds, slr_m))


def read_calm_sea_level_m(surge_ds: xr.Dataset) -> float:
    """Read surge_forcing.nc's calm_sea_level_m, rounded to 1 cm; 0.0 if the field is absent.

    calm_sea_level_m is the basin-mean local MSL in model coordinates (MDT-only),
    the level rule 11's calibration and rule 12a's skeleton zsini hold the sea at.
    """
    if "calm_sea_level_m" not in surge_ds:
        return 0.0
    return float(round_level(float(surge_ds["calm_sea_level_m"].values)))


def lookup_storm_tide_at_rp(
    surge_ds: xr.Dataset, rp_yr: float | None, slr_m: float = 0.0
) -> np.ndarray:
    """Return the absolute per-station storm-tide level (m) at a tabulated COAST-RP rp_yr.

    Exact match against the tabulated RPs (ValueError otherwise), plus the
    MDT-only station_calm_sea_level_m and slr_m (target global-mean SLR, m) x
    slr_fingerprint, then rounded to 1 cm. SLR is applied here, not stored in
    surge_forcing.nc, so that file is independent of slr_m. rp_yr=None gives
    the calm-sea level.
    """
    if rp_yr is None:
        return calm_sea_levels(surge_ds, slr_m)  # flat: no storm surge
    baselines = _station_calm_levels_unrounded(surge_ds, slr_m)
    rps_yr = surge_ds["rp_yr"].values
    idx = np.nonzero(np.isclose(rps_yr, float(rp_yr)))[0]
    if idx.size == 0:
        raise ValueError(
            f"surge RP {rp_yr} not tabulated in COAST-RP ({[int(r) for r in rps_yr]})"
        )
    return round_level(surge_ds["storm_tide_rp_table"].values[:, idx[0]] + baselines)


def storm_tide_at_rp_interpolated(
    surge_ds: xr.Dataset, rp_yr: float, slr_m: float = 0.0
) -> np.ndarray:
    """Return the per-station storm-tide level (m) at an arbitrary RP, linear in RP.

    Clamped to the tabulated range (as interpolate_protection_level); same
    vertical terms and rounding as lookup_storm_tide_at_rp, which it equals
    at a tabulated RP.
    """
    rps_yr = surge_ds["rp_yr"].values.astype(float)
    clamped_rp_yr = float(np.clip(rp_yr, rps_yr.min(), rps_yr.max()))
    raw = np.array(
        [
            np.interp(clamped_rp_yr, rps_yr, row)
            for row in surge_ds["storm_tide_rp_table"].values
        ]
    )
    return round_level(raw + _station_calm_levels_unrounded(surge_ds, slr_m))


def _ramp_in_slr(
    matrix: np.ndarray,
    surge_ds: xr.Dataset,
    slr_m: float,
    slr_ramp_hr: float,
    spinup_slr_m: float = 0.0,
) -> np.ndarray:
    # The shared spin-up runs at spinup_slr_m; a boundary jumping to another
    # SLR level at the event start sends in a front that overtops the coastal
    # weir. The offset is the difference of ROUNDED calm levels, so before the
    # event the boundary equals the spin-up's calm level exactly.
    if slr_m == spinup_slr_m or slr_ramp_hr <= 0.0 or "slr_fingerprint" not in surge_ds:
        return matrix
    times = surge_ds["time"].values.astype(float)
    start_hr = float(surge_ds.attrs["lead_days"]) * 24.0
    peak_hr = float(
        surge_ds.attrs.get(
            "hg_peak_hr", start_hr + float(surge_ds.attrs["period_hr"]) / 2.0
        )
    )
    if start_hr + slr_ramp_hr > peak_hr:
        raise ValueError(
            f"slr_ramp_hr={slr_ramp_hr:g} h runs past the surge peak "
            f"({peak_hr - start_hr:g} h after the event start) -- the design peak would "
            "miss part of its SLR; shorten sfincs.boundary_setup.slr_ramp_hr"
        )
    slr_offset_m = calm_sea_levels(surge_ds, slr_m) - calm_sea_levels(
        surge_ds, spinup_slr_m
    )
    w = np.clip((times - start_hr) / slr_ramp_hr, 0.0, 1.0)
    return matrix - (1.0 - w)[None, :] * slr_offset_m[:, None]


def build_design_surge_matrix(
    surge_ds: xr.Dataset,
    rp_yr: int | None | str,
    slr_m: float = 0.0,
    slr_ramp_hr: float = 0.0,
    spinup_slr_m: float = 0.0,
) -> np.ndarray:
    """Rebuild the (n_station, n_time) water-level boundary (m) at rp_yr.

    Surge counterpart of river_forcing.build_design_discharge_matrix; a new
    surge RP reruns rule 14 only, never rule 07d. None: flat calm sea. With
    hydrograph components in surge_forcing.nc, an RP gives average tide +
    surge peaking on tidal high water and "tide" (case-insensitive) the tide
    alone; without them an RP is a half-cosine from calm level to RP level
    and "tide" raises ValueError. Lead-in and peak levels are rounded to 1 cm;
    the tide signal is not.
    slr_m: target global-mean SLR (m) x slr_fingerprint, added to baseline,
    peak and tide; 0.0 reproduces the MDT-only fields. slr_ramp_hr > 0:
    the SLR term is spinup_slr_m (the spin-up's) until the event start and
    goes linearly to slr_m over slr_ramp_hr.
    The protection_level is deliberately NOT subtracted: it already floors
    the weir crest, and subtracting it here would double-count FLOPROS and
    assume a uniform wall along the whole coast.
    """
    times = surge_ds["time"].values

    if rp_yr is not None and "hg_tide_m" in surge_ds:
        return _ramp_in_slr(
            _hydrograph_surge_matrix(surge_ds, rp_yr, slr_m),
            surge_ds,
            slr_m,
            slr_ramp_hr,
            spinup_slr_m,
        )

    if isinstance(rp_yr, str) and rp_yr.strip().lower() == "tide":
        raise ValueError(
            "surge_rp_yr 'Tide' needs the storm-tide hydrographs (tide signal) in "
            "surge_forcing.nc -- enable boundary_forcings.surge.hydrograph and rerun rule 07d"
        )

    baselines = calm_sea_levels(surge_ds, slr_m)
    storm_tide_at_rp_m = lookup_storm_tide_at_rp(surge_ds, rp_yr, slr_m=slr_m)
    wave = np.stack(
        [
            sinusoidal_wave(
                float(b),
                float(lvl),
                times,
                float(surge_ds.attrs["lead_days"]),
                float(surge_ds.attrs["period_hr"]),
            )
            for b, lvl in zip(baselines, storm_tide_at_rp_m)
        ]
    )
    return _ramp_in_slr(wave, surge_ds, slr_m, slr_ramp_hr, spinup_slr_m)


# The MDT vertical correction and SLR fingerprint below adapt the method of
# Natalia Aleksandrova (notebooks 01_retrieve_MDT_correction.ipynb,
# 02_get_SLR_fingerprint.ipynb, 03_combine_wl_data_scenarios.ipynb).


def _nearest_valid_grid(
    da: xr.DataArray,
    lon_dim: str,
    lon: float,
    lat_dim: str,
    lat: float,
    fallback_deg: float,
) -> float:
    # Port of find_nearest_valid from the MDT-correction notebook (credited to
    # https://github.com/pydata/xarray/issues/644). Needs ascending lat/lon.
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


def load_mdt(
    mdt_path: str,
    mdt_variable: str = "mdt",
    bbox: tuple[float, float, float, float] | None = None,
    margin_deg: float = 0.0,
) -> xr.DataArray:
    """Load AVISO MDT_CNES-CLS22 mean dynamic topography as a 2-D ascending lat/lon array.

    Extra dimensions (e.g. time) take their first index; longitudes are
    remapped from 0..360 to -180..180 if needed. ``bbox`` (lon_min, lat_min,
    lon_max, lat_max; degrees) clips to that box widened by ``margin_deg``.
    """
    with xr.open_dataset(mdt_path) as ds:
        da = ds[mdt_variable]
        lat_dim, lon_dim = latlon_dims(da)
        for extra in [d for d in da.dims if d not in (lat_dim, lon_dim)]:
            da = da.isel({extra: 0})

        if float(da[lon_dim].max()) > 180:
            da = da.assign_coords(
                {lon_dim: xr.where(da[lon_dim] > 180, da[lon_dim] - 360, da[lon_dim])}
            )
        da = da.sortby([lat_dim, lon_dim])
        if bbox is not None:
            lon_min, lat_min, lon_max, lat_max = bbox
            da = da.sel(
                {
                    lat_dim: slice(lat_min - margin_deg, lat_max + margin_deg),
                    lon_dim: slice(lon_min - margin_deg, lon_max + margin_deg),
                }
            )
        return da.load()


def apply_mdt_correction(
    stations: gpd.GeoDataFrame,
    mdt_da: xr.DataArray,
    fallback_deg: float = 3.0,
) -> gpd.GeoDataFrame:
    """Re-reference station 'storm_tide_at_rp_m' from local MSL to the GOCO06s geoid with AVISO MDT.

    Adds storm_tide_at_rp_raw_m (original, local-MSL per COAST-RP), mdt (m; NaN if no
    valid cell within +/-fallback_deg) and storm_tide_at_rp_m = storm_tide_at_rp_raw_m + mdt,
    matching GEBCO's re-referencing (gebco += mdt) in 06_get_elevation.py.
    Sign: MDT is the mean sea surface height ABOVE the geoid (CF
    mean_dynamic_topography; e.g. +0.6 m Sargasso Sea, -1.5 m Southern
    Ocean), so H_GOCO06s = H_MSL + MDT.
    """
    lat_dim, lon_dim = latlon_dims(mdt_da)

    result = stations.copy()
    result["storm_tide_at_rp_raw_m"] = result["storm_tide_at_rp_m"]
    result["mdt"] = [
        _nearest_valid_grid(mdt_da, lon_dim, geom.x, lat_dim, geom.y, fallback_deg)
        for geom in result.geometry
    ]
    result["storm_tide_at_rp_m"] = result["storm_tide_at_rp_raw_m"] + result["mdt"]
    return result


def load_slr_fingerprint(
    slr_root: str,
    ssp_scenario: str,
    confidence_level: str,
    year: int,
    quantile: float,
) -> xr.Dataset:
    """Load the IPCC AR6 sea-level change (m) for one SSP/confidence/year/quantile.

    Over the combined gauge + 1deg x 1deg grid 'locations' dimension.
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
    """Return the global-mean SLR (m) over valid 1deg grid cells ('mean_ori' in the notebook)."""
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
    global_mean_slr_m: float,
    fallback_deg: float = 3.0,
) -> gpd.GeoDataFrame:
    """Add 'slr_fingerprint' = local AR6 SLR / global-mean SLR (dimensionless) per station.

    1.0 if no valid AR6 location lies within +/-fallback_deg. Not scaled by
    slr_m, so surge_forcing.nc (and the basin-level calibration and skeleton)
    stays independent of slr_m; fingerprint * slr_m is applied when the
    scenario boundary is built.
    """
    lons = slr_ds["lon"].values
    lats = slr_ds["lat"].values
    vals = slr_ds["sea_level_change"].values

    result = stations.copy()
    fingerprints = []
    for geom in result.geometry:
        local_slr_m = _nearest_valid_location(
            lons, lats, vals, geom.x, geom.y, fallback_deg
        )
        fingerprints.append(
            1.0 if np.isnan(local_slr_m) else local_slr_m / global_mean_slr_m
        )
    result["slr_fingerprint"] = fingerprints
    return result


def preview_station_rp_yr(surge_rp_yr: int | str | None) -> int:
    """Return the COAST-RP RP (yr) that fills rule 07d's per-station preview columns.

    A numeric surge_rp_yr is returned unchanged. 'Tide' and None have no
    storm-tide RP, so the smallest tabulated RP keeps storm_tide_at_rp_m
    defined; the production boundary (rule 14) never reads that column.
    """
    if surge_rp_yr is None or isinstance(surge_rp_yr, str):
        return _COASTRP_RPS[0]
    return surge_rp_yr


def load_coastrp_stations(
    nc_path: str,
    rp_yr: int,
) -> gpd.GeoDataFrame:
    """Load COAST-RP surge stations (EPSG:4326) with storm-tide levels (m).

    'storm_tide_at_rp_m' holds storm_tide_rp_{rp_yr:04d}; 'rp_raw_{rp:04d}'
    columns hold every tabulated RP (raw, before vertical/SLR correction) for
    interpolate_protection_level. Stations with non-finite coordinates are dropped.
    """
    import xarray as xr

    rp_var = f"storm_tide_rp_{rp_yr:04d}"
    with xr.open_dataset(nc_path) as ds:
        lons = ds["station_x_coordinate"].values
        lats = ds["station_y_coordinate"].values
        rp_vals = ds[rp_var].values
        raw_rp_vals = {
            tabulated_rp_yr: ds[f"storm_tide_rp_{tabulated_rp_yr:04d}"].values
            for tabulated_rp_yr in _COASTRP_RPS
        }
    # Some CoastRP stations carry NaN/fill-value coordinates, which make
    # shapely.distance() warn downstream.
    valid = np.isfinite(lons) & np.isfinite(lats)
    if not valid.all():
        log.debug(
            f"Dropping {int((~valid).sum())} surge station(s) with invalid coordinates"
        )
    data = {"storm_tide_at_rp_m": rp_vals[valid]}
    for tabulated_rp_yr, vals in raw_rp_vals.items():
        data[f"rp_raw_{tabulated_rp_yr:04d}"] = vals[valid]
    return gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(lons[valid], lats[valid]),
        crs="EPSG:4326",
    )


def interpolate_protection_level(
    stations: gpd.GeoDataFrame,
    target_rp_yr: float,
) -> pd.Series:
    """Interpolate each station's raw storm-tide level (m) linearly in RP.

    Between the bracketing COAST-RP RPs (rp_raw_{rp:04d} columns);
    target_rp_yr is clamped to [1, 1000] -- COAST-RP cannot extrapolate.
    """
    clamped_rp_yr = float(np.clip(target_rp_yr, min(_COASTRP_RPS), max(_COASTRP_RPS)))

    lo_rp_yr = max(r for r in _COASTRP_RPS if r <= clamped_rp_yr)
    hi_rp_yr = min(r for r in _COASTRP_RPS if r >= clamped_rp_yr)

    lo_vals = stations[f"rp_raw_{lo_rp_yr:04d}"].astype(float)
    if lo_rp_yr == hi_rp_yr:
        return lo_vals.copy()

    hi_vals = stations[f"rp_raw_{hi_rp_yr:04d}"].astype(float)
    frac = (clamped_rp_yr - lo_rp_yr) / (hi_rp_yr - lo_rp_yr)
    return lo_vals + frac * (hi_vals - lo_vals)


def compute_distances_to_bbox(
    stations: gpd.GeoDataFrame,
    bbox_utm: gpd.GeoDataFrame,
    domain_crs: str,
) -> gpd.GeoDataFrame:
    """Add 'dist_m', the metric distance from each station to the domain bbox boundary."""
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
    """Select the closest stations within an expanding radius, deduplicated and capped.

    1. The first radius in search_radii_km holding >= min_stations wins;
       otherwise the min_stations nearest are used.
    2. A station within dedupe_radius_km of an already-kept, closer station
       is dropped.
    3. At most max_stations closest to the boundary are kept.
    Result is ordered by ascending distance to the domain boundary.
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


def select_surge_stations(
    nc_path: str,
    domain_utm: gpd.GeoDataFrame,
    domain_crs: str,
    min_stations: int,
    max_stations: int,
    search_radii_km: list[float],
    dedupe_radius_km: float,
    rp_yr: int = _COASTRP_RPS[0],
) -> gpd.GeoDataFrame:
    """Select a basin's COAST-RP surge stations by distance to the domain.

    Run by rule select_surge_stations (07a); rule get_boundary_forcings (07d)
    reloads its choice with load_selected_stations. rp_yr only sets
    the 'storm_tide_at_rp_m' column; it does not affect the selection.
    """
    stations = load_coastrp_stations(nc_path, rp_yr)
    log.info(f"CoastRP stations loaded: {len(stations)}")
    stations = compute_distances_to_bbox(stations, domain_utm, domain_crs)
    return select_nearest_stations(
        stations,
        min_stations,
        max_stations,
        search_radii_km,
        dedupe_radius_km,
        domain_crs,
    )


def load_selected_stations(
    nc_path: str,
    domain_utm: gpd.GeoDataFrame,
    domain_crs: str,
    stations_json: str,
    rp_yr: int = _COASTRP_RPS[0],
) -> gpd.GeoDataFrame:
    """Reload the COAST-RP stations listed in rule 07a's surge_stations.json, in its order.

    Stations are matched on lon/lat rounded to 6 decimals, as 07a writes them;
    raises ValueError when a listed station matches none or several stations.
    """
    stations = load_coastrp_stations(nc_path, rp_yr)
    log.info(f"CoastRP stations loaded: {len(stations)}")
    stations = compute_distances_to_bbox(stations, domain_utm, domain_crs)
    by_lonlat: dict[tuple[float, float], list] = {}
    for label, geom in zip(stations.index, stations.geometry):
        key = (round(float(geom.x), 6), round(float(geom.y), 6))
        by_lonlat.setdefault(key, []).append(label)
    with open(stations_json) as f:
        listed = json.load(f)["stations"]
    labels = []
    for s in listed:
        matches = by_lonlat.get((s["lon"], s["lat"]), [])
        if len(matches) != 1:
            raise ValueError(
                f"{stations_json}: station ({s['lon']}, {s['lat']}) matches "
                f"{len(matches)} COAST-RP station(s) in {nc_path}, expected 1"
            )
        labels.append(matches[0])
    return stations.loc[labels].copy()


def build_surge_dataset(
    stations: gpd.GeoDataFrame,
    times: np.ndarray,
    lead_days: float,
    period_hr: float,
    rp_yr: int,
    calm_sea_level_m: float = 0.0,
    station_calm_sea_level_m: np.ndarray | None = None,
) -> xr.Dataset:
    """Assemble surge_forcing.nc: half-cosine waves from lead-in baseline to storm_tide_at_rp_m per station.

    All levels are MDT-only; SLR is not baked in, only the dimensionless
    slr_fingerprint is stored. Datum: MSL-referenced data (GEBCO, COAST-RP)
    is re-referenced to GOCO06s by ADDING MDT, so local MSL maps to +MDT.
    station_calm_sea_level_m (m, one per station): own local MSL = storm_tide_at_rp_m_i -
    storm_tide_at_rp_raw_m_i = +mdt_i, so the surge amplitude is exactly storm_tide_at_rp_raw_m;
    None uses calm_sea_level_m for all. calm_sea_level_m = mean(+MDT) over stations, the
    SLR-independent level rule 11 and rule 12a use for zsini.tif.
    """
    if station_calm_sea_level_m is not None:
        surge_matrix = np.stack(
            [
                sinusoidal_wave(
                    float(station_calm_sea_level_m[i]),
                    float(row["storm_tide_at_rp_m"]),
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
                    calm_sea_level_m,
                    float(row["storm_tide_at_rp_m"]),
                    times,
                    lead_days,
                    period_hr,
                )
                for _, row in stations.iterrows()
            ]
        )
    ds = xr.Dataset(
        {
            "water_level": (
                ["station", "time"],
                surge_matrix,
                {
                    "units": "m",
                    "long_name": (
                        "storm tide water level (MDT-only, diagnostic preview -- "
                        "does NOT include SLR; the real production boundary "
                        "adds slr_fingerprint*slr_m at build time, see "
                        "build_design_surge_matrix)"
                    ),
                },
            ),
            "storm_tide_at_rp_m": (
                ["station"],
                stations["storm_tide_at_rp_m"].values,
                {"units": "m", "long_name": f"RP{rp_yr} storm tide level"},
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

    ds["calm_sea_level_m"] = (
        [],
        float(calm_sea_level_m),
        {
            "units": "m",
            "long_name": (
                "Mean vertical correction applied as lead-period baseline "
                "(mean(+MDT) across selected stations = calm sea level in "
                "model coordinates, MDT-only -- SLR deliberately excluded, "
                "see apply_slr_fingerprint). Read by rule 11's calibration "
                "and rule 12a's skeleton to initialise sea cells (zsini.tif) "
                "at an slr_m-independent reference. Equals 0.0 when MDT "
                "correction is off."
            ),
        },
    )

    if station_calm_sea_level_m is not None:
        ds["station_calm_sea_level_m"] = (
            ["station"],
            station_calm_sea_level_m,
            {
                "units": "m",
                "long_name": (
                    "Per-station lead-period flat value (+mdt_i, MDT-only). "
                    "Local MSL for each station in model coordinates. "
                    "Ensures surge amplitude = storm_tide_at_rp_raw_m per station. "
                    "SLR is NOT included here -- see slr_fingerprint and "
                    "build_design_surge_matrix's own slr_m argument."
                ),
            },
        )

    # Optional provenance from apply_mdt_correction / apply_slr_fingerprint.
    # No slr_m-scaled term is stored: surge_forcing.nc is independent of slr_m.
    extra_station_vars = {
        "storm_tide_at_rp_raw_m": {
            "units": "m",
            "long_name": "RP storm tide level before vertical (MDT) correction",
        },
        "mdt": {
            "units": "m",
            "long_name": "MDT correction applied (local MSL -> geoid)",
        },
        "slr_fingerprint": {
            "units": "1",
            "long_name": (
                "AR6 SLR fingerprint (local / global-mean SLR) -- dimensionless, "
                "multiply by a target slr_m (m) at build time to get the actual "
                "per-station SLR contribution (see build_design_surge_matrix)"
            ),
        },
    }
    for col, attrs in extra_station_vars.items():
        if col in stations.columns:
            ds[col] = (["station"], stations[col].values, attrs)

    # Full raw COAST-RP table lets rule 14 rebuild the wave at any tabulated RP
    # without rerunning rule 07d (as river_forcing.nc's discharge_rp_table).
    rp_cols = [
        c
        for c in (f"rp_raw_{tabulated_rp_yr:04d}" for tabulated_rp_yr in _COASTRP_RPS)
        if c in stations.columns
    ]
    if len(rp_cols) == len(_COASTRP_RPS):
        ds["storm_tide_rp_table"] = (
            ["station", "rp_yr"],
            np.column_stack([stations[c].values for c in rp_cols]),
            {
                "units": "m",
                "long_name": "raw COAST-RP storm-tide level per tabulated RP",
            },
        )
        ds = ds.assign_coords(
            rp_yr=(
                "rp_yr",
                np.asarray(_COASTRP_RPS, dtype=float),
                {
                    "units": "yr",
                    "long_name": "return period axis for storm_tide_rp_table",
                },
            )
        )
    ds.attrs["lead_days"] = float(lead_days)
    ds.attrs["period_hr"] = float(period_hr)

    return ds
