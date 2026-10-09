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


# Every absolute coastal water level and every weir crest is kept to the
# centimetre (2 decimals): levels rounded to nearest, crests rounded UP.
LEVEL_DECIMALS = 2


def round_level(values) -> np.ndarray:
    """
    Round a coastal water level to the nearest centimetre.

    THE rounding for every absolute coastal water level: calm sea
    (calm_sea_levels / read_baseline_m) and storm tide at an RP
    (lookup_storm_tide_at_rp, storm_tide_at_rp_interpolated). Weir crests
    are kept on the same centimetre grid but rounded UP -- see ceil_crest.

    Until 2026-10-09 levels and crests alike were rounded UP to the next
    0.1 m, because hydromt_sfincs writes weir crests at 0.1 m: that added
    0-10 cm of unphysical level to the forcing and to the crests, and made
    return periods a few centimetres apart indistinguishable.

    NaN passes through; -0.0 is returned as 0.0.
    """
    return np.round(np.asarray(values, dtype=float), LEVEL_DECIMALS) + 0.0


def ceil_crest(values) -> np.ndarray:
    """
    Round a weir crest UP to the next centimetre -- never down, so a crest
    is never below the value it was computed from (a simulated water level,
    a protection level). Used for every weir crest
    (src.protection_weir.build_coastal_protection_weir, rule 10);
    sfincs.weir is written at the same precision
    (src.sfincs_run.write_weir_file), so a crest is stored exactly.

    The inner np.round strips float noise so a value already on the grid
    (2.75, stored as 2.7500000000000004) isn't bumped a whole centimetre.
    NaN passes through; -0.0 is returned as 0.0.
    """
    step = 10.0**-LEVEL_DECIMALS
    steps = np.ceil(np.round(np.asarray(values, dtype=float) / step, 6))
    return np.round(steps * step, LEVEL_DECIMALS) + 0.0


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


def extract_hydrograph_components(
    hydrograph_path: str,
    station_lons: np.ndarray,
    station_lats: np.ndarray,
    window_hr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-station storm-tide hydrograph components from the basin's own
    storm_tide_hydrographs.nc (rule storm_tide_hydrographs, src.
    storm_tide_hydrograph): the average tide signal (high water at t=0) and
    the normalised surge hydrograph PER RETURN PERIOD (peak=1 at t=0), cut
    to -window_hr..+window_hr, at each station's own hydrograph location
    (the same COAST-RP stations, matched by coordinates). Levels are
    relative to a fixed local MSL, same datum as COAST-RP -- MDT/SLR are
    added at build time.

    Returns:
        (hg_time_hr, tide, shape, shape_rps, match_km): hg_time_hr (n_t,)
        hours relative to the peak, shared; tide (n_station, n_t); shape
        (n_station, n_rp, n_t); shape_rps (n_rp,) return periods the shapes
        belong to; match_km (n_station,) distance to the matched location.
    """
    with xr.open_dataset(hydrograph_path) as ds:
        hx = ds["longitude"].values
        hy = ds["latitude"].values
        t = ds["time_hr"].values
        win = np.abs(t) <= window_hr + 1e-9
        tide_all = ds["average_tide"].values[:, win]
        shape_all = ds["surge_shape"].values[:, :, win]
        shape_rps = ds["return_period"].values.astype(float)
    # The shapes are derived FOR their window (events are cut to it), so a
    # file built with another window is not the same shapes cut shorter.
    if abs(t[0] + window_hr) > 1e-6 or abs(t[-1] - window_hr) > 1e-6:
        raise ValueError(
            f"hydrograph file spans {t[0]:.1f}..{t[-1]:.1f} h but this basin's event window is "
            f"+-{window_hr:g} h -- it was built with another window_hr; rerun rule storm_tide_hydrographs"
        )

    n = len(station_lons)
    tide = np.full((n, int(win.sum())), np.nan)
    shape = np.full((n, len(shape_rps), int(win.sum())), np.nan)
    match_km = np.full(n, np.nan)
    for i, (lon, lat) in enumerate(zip(station_lons, station_lats)):
        d = np.hypot((hx - lon) * np.cos(np.radians(lat)), hy - lat) * 111.0
        j = int(np.nanargmin(d))
        tide[i], shape[i], match_km[i] = tide_all[j], shape_all[j], d[j]
    return t[win], tide, shape, shape_rps, match_km


def surge_shape_at_rp(surge_ds: xr.Dataset, rp_yr: float) -> np.ndarray:
    """
    surge_forcing.nc's normalised surge shape at an arbitrary return period:
    (n_station, n_hg_time). The shapes are stored per COAST-RP return period
    (hg_rp); between two of them the shape is interpolated linearly in RP
    (the rule storm_tide_at_rp_interpolated uses for the level itself),
    clamped to the tabulated range. A normalised shape stays normalised
    (peak = 1 at t = 0) under that interpolation.
    """
    shape = surge_ds["hg_surge_shape"].values
    rps = surge_ds["hg_rp"].values.astype(float)
    rp = float(np.clip(float(rp_yr), rps.min(), rps.max()))
    hi = int(np.searchsorted(rps, rp))
    if np.isclose(rps[hi], rp):
        return shape[:, hi]
    lo = hi - 1
    frac = (rp - rps[lo]) / (rps[hi] - rps[lo])
    return (1.0 - frac) * shape[:, lo] + frac * shape[:, hi]


def storm_tide_event(
    surge_ds: xr.Dataset,
    rel_hr: np.ndarray,
    peak_level: np.ndarray | None,
    slr_m: float = 0.0,
    rp_yr: float | None = None,
) -> np.ndarray:
    """
    The storm-tide hydrograph itself, on hours relative to its own peak
    (rel_hr = 0 is the surge peak on tidal high water), from surge_forcing.nc's
    hg_tide_m/hg_surge_shape. Per station:

      tide  = average tide + station calm level (MDT, + SLR x fingerprint)
      event = tide + A x surge_shape(rp_yr), A = peak_level - tide high
              water (>= 0), so the event peaks at peak_level exactly. The
              surge shape is the one derived for that return period
              (surge_shape_at_rp).

    THE single builder of that wave: production's boundary
    (_hydrograph_surge_matrix, rule 13) and the dike-crest calibration's
    protection-level storm tide (rule 10, rounds 1-2) both call it, each on
    its own time axis, so the crests are calibrated against the wave
    production sends.

    peak_level: (n_station,) absolute water level at the peak (as
        lookup_storm_tide_at_rp / storm_tide_at_rp_interpolated return it,
        same slr_m), or None for the tide alone.
    rp_yr: return period peak_level belongs to -- selects the surge shape;
        required with peak_level.

    Returns:
        (n_station, len(rel_hr)) np.ndarray, water level (m). Outside the
        stored window the tide holds its edge value and the surge is 0.
    """
    hg_t = surge_ds["hg_time_hr"].values
    tide = surge_ds["hg_tide_m"].values
    base = _station_calm_levels_unrounded(surge_ds, slr_m)
    event = (
        np.stack([np.interp(rel_hr, hg_t, tide[i]) for i in range(len(base))])
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
    surge_ds: xr.Dataset, design_rp_yr: float | str, slr_m: float
) -> np.ndarray:
    """
    build_design_surge_matrix's storm-tide-hydrograph branch (surge_forcing.nc
    carries hg_tide_m/hg_surge_shape, rule 07 with
    boundary_forcings.surge.hydrograph.enabled). Per station, on the model
    time axis with the window's t=0 at attrs["hg_peak_hr"]:

      tide  = average tide + station calm level (MDT, + SLR x fingerprint)
      "tide": tide only
      RP:   tide + A x surge_shape(RP), A = RP level - tide high water, so the
            peak (surge on high water) equals lookup_storm_tide_at_rp exactly
            (rounded to the centimetre like every coastal level); A >= 0

    Before the window the boundary holds the calm lead-in level (as the
    spin-up ends); over the window's first attrs["hg_ramp_hours"] it blends
    linearly from that level into the series (surge is still 0 there), so
    the event starts from the restart without a step at the boundary.
    """
    times = surge_ds["time"].values
    peak_hr = float(surge_ds.attrs["hg_peak_hr"])
    ramp_hr = float(surge_ds.attrs.get("hg_ramp_hours", 0.0))
    hg_t = surge_ds["hg_time_hr"].values
    is_tide = isinstance(design_rp_yr, str) and design_rp_yr.strip().lower() == "tide"
    level = (
        None
        if is_tide
        else lookup_storm_tide_at_rp(surge_ds, design_rp_yr, slr_m=slr_m)
    )
    event = storm_tide_event(
        surge_ds,
        times - peak_hr,
        level,
        slr_m=slr_m,
        rp_yr=None if is_tide else float(design_rp_yr),
    )

    start_hr = peak_hr + hg_t[0]
    w = (
        np.clip((times - start_hr) / ramp_hr, 0.0, 1.0)
        if ramp_hr > 0
        else (times >= start_hr).astype(float)
    )
    calm = calm_sea_levels(surge_ds, slr_m)
    return (1.0 - w) * calm[:, None] + w * event


def _station_calm_levels_unrounded(surge_ds: xr.Dataset, slr_m: float) -> np.ndarray:
    """Per-station calm-sea level before rounding: rule 07's own
    station_baseline (MDT-only; the scalar baseline_m for every station if
    absent) plus the optional SLR term (slr_m x slr_fingerprint).
    """
    baselines = (
        surge_ds["station_baseline"].values
        if "station_baseline" in surge_ds
        else np.full(surge_ds.sizes["station"], float(surge_ds["baseline_m"].values))
    )
    if slr_m != 0.0 and "slr_fingerprint" in surge_ds:
        baselines = baselines + surge_ds["slr_fingerprint"].values * slr_m
    return baselines


def calm_sea_levels(surge_ds: xr.Dataset, slr_m: float = 0.0) -> np.ndarray:
    """
    Per-station calm-sea (tide-only, no storm surge) water level -- the
    event forcing's own lead-in level, rounded to the centimetre
    (round_level).

    Returns:
        (n_station,) np.ndarray, water level (m).
    """
    return round_level(_station_calm_levels_unrounded(surge_ds, slr_m))


def read_baseline_m(surge_ds: xr.Dataset) -> float:
    """
    surge_forcing.nc's scalar baseline_m (basin-mean local MSL in model
    coordinates, MDT-only -- see 07_get_boundary_forcings.py), rounded to
    the centimetre (round_level); 0.0 if the file predates the field.
    THE single reader for every consumer that starts/holds the sea at this
    level (rule 10's calibration boundary + zsini, rule 13's skeleton
    zsini).
    """
    if "baseline_m" not in surge_ds:
        return 0.0
    return float(round_level(float(surge_ds["baseline_m"].values)))


def lookup_storm_tide_at_rp(
    surge_ds: xr.Dataset, rp_yr: float | None, slr_m: float = 0.0
) -> np.ndarray:
    """
    Per-station storm-tide water level at ``rp_yr`` from surge_forcing.nc's
    stored storm_tide_rp_table (exact match against COAST-RP's tabulated
    RPs — no interpolation), vertically corrected with rule 07's own
    station_baseline (MDT-only — see 07_get_boundary_forcings.py) plus an
    OPTIONAL SLR contribution applied here, not baked into surge_forcing.nc:
    ``slr_m`` (a target global-mean SLR, m) is scaled by each station's own
    ``slr_fingerprint`` (src.surge.apply_slr_fingerprint) and added on top.
    This keeps surge_forcing.nc itself completely independent of the slr_m
    target -- callers building a real production/spin-up boundary (13_build_
    sfincs.py, 14_run_spinup.py) pass their own slr_m; callers that only need
    the MDT-only reference (rule 10's calibration, rule 13's skeleton zsini)
    simply don't pass it (default 0.0).

    rp_yr=None means mean coastal conditions: each station's own baseline
    (tide-only / calm sea, no storm surge), plus the same optional SLR term.

    The returned levels are ABSOLUTE water levels, rounded to the centimetre
    (round_level) after the MDT/SLR terms are added.

    Returns:
        (n_station,) np.ndarray, water level (m).
    """
    if rp_yr is None:
        return calm_sea_levels(surge_ds, slr_m)  # flat: no storm surge
    baselines = _station_calm_levels_unrounded(surge_ds, slr_m)
    table_rps = surge_ds["table_rp"].values
    idx = np.nonzero(np.isclose(table_rps, float(rp_yr)))[0]
    if idx.size == 0:
        raise ValueError(
            f"surge RP {rp_yr} not tabulated in COAST-RP "
            f"({[int(r) for r in table_rps]})"
        )
    return round_level(surge_ds["storm_tide_rp_table"].values[:, idx[0]] + baselines)


def storm_tide_at_rp_interpolated(
    surge_ds: xr.Dataset, rp_yr: float, slr_m: float = 0.0
) -> np.ndarray:
    """
    Per-station storm-tide water level at an ARBITRARY return period (e.g.
    the FLOPROS coastal protection RP, which COAST-RP doesn't tabulate) --
    linear in RP between the two bracketing tabulated RPs, the same rule
    interpolate_protection_level uses for the coastal crest itself; clamped
    to the tabulated range. Then the same vertical terms as
    lookup_storm_tide_at_rp (which it equals at a tabulated RP), rounded to
    the centimetre.

    Returns:
        (n_station,) np.ndarray, water level (m).
    """
    table_rps = surge_ds["table_rp"].values.astype(float)
    rp = float(np.clip(rp_yr, table_rps.min(), table_rps.max()))
    raw = np.array(
        [
            np.interp(rp, table_rps, row)
            for row in surge_ds["storm_tide_rp_table"].values
        ]
    )
    return round_level(raw + _station_calm_levels_unrounded(surge_ds, slr_m))


def build_design_surge_matrix(
    surge_ds: xr.Dataset,
    design_rp_yr: int | None | str,
    slr_m: float = 0.0,
) -> np.ndarray:
    """
    Rebuild the per-station water-level timeseries at ``design_rp_yr`` from
    surge_forcing.nc — the surge counterpart to
    river_forcing.build_design_discharge_matrix.  Changing the surge RP only
    re-runs the build (rule 13), never rule 07.

    design_rp_yr=None means mean coastal conditions: zero surge amplitude, a
    flat timeseries at each station's own baseline (calm sea, no tide).

    When surge_forcing.nc carries storm-tide hydrograph components
    (hg_tide_m/hg_surge_shape, boundary_forcings.surge.hydrograph.enabled),
    an RP and the string "tide" (case-insensitive) are built by
    _hydrograph_surge_matrix: average tide + surge peaking on tidal high
    water over the event window, tide only for "tide". Without them an RP
    falls back to a half-cosine wave (sinusoidal_wave) from the calm level to
    the RP level, and "tide" is unavailable.

    Both the lead-in level (calm_sea_levels) and the peak
    (lookup_storm_tide_at_rp) are rounded to the centimetre. The tide
    itself is NOT rounded -- it's a genuinely varying signal, not a single
    absolute crest/lead-in level being compared against a weir.

    slr_m: target global-mean SLR (m), scaled by each station's own
        'slr_fingerprint' and added to the lead-period baseline, the
        RP-level peak (see lookup_storm_tide_at_rp) and the tide. Applied
        HERE, not baked into surge_forcing.nc -- 0.0 (default) reproduces
        surge_forcing.nc's own MDT-only fields exactly, so calibration/
        skeleton callers that never pass slr_m are fully insulated from the
        slr_m config value.

    The stored protection_level (if rule 07 wrote one) is deliberately NOT
    subtracted here -- see 07_get_boundary_forcings.py's own design comment
    for why: a weir is a real, SFINCS-modelled barrier, whereas subtracting
    a scalar directly from the boundary assumes the whole coast sits behind
    a uniform wall that isn't actually there, silently suppressing the flood
    signal at any land below the corrected level. protection_level already
    floors the weir crest in rule 13 (coastal_protection_crest_m); applying
    it again here would double-count the same FLOPROS standard through two
    independent mechanisms.

    Returns:
        (n_station, n_time) np.ndarray, water level (m).
    """
    times = surge_ds["time"].values

    if design_rp_yr is not None and "hg_tide_m" in surge_ds:
        return _hydrograph_surge_matrix(surge_ds, design_rp_yr, slr_m)

    if isinstance(design_rp_yr, str) and design_rp_yr.strip().lower() == "tide":
        raise ValueError(
            "surge_rp 'Tide' needs the storm-tide hydrographs (tide signal) in "
            "surge_forcing.nc -- enable boundary_forcings.surge.hydrograph and rerun rule 07"
        )

    baselines = calm_sea_levels(surge_ds, slr_m)
    rp_level = lookup_storm_tide_at_rp(surge_ds, design_rp_yr, slr_m=slr_m)
    wave = np.stack(
        [
            sinusoidal_wave(
                float(b),
                float(lvl),
                times,
                float(surge_ds.attrs["lead_days"]),
                float(surge_ds.attrs["period_hr"]),
            )
            for b, lvl in zip(baselines, rp_level)
        ]
    )
    return wave


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
        rp_level:     rp_level_raw + mdt (local MSL -> GOCO06s geoid), matching
                       the re-referencing of GEBCO to GOCO06s (gebco += mdt)
                       in 05a_get_elevation.py.

    Sign: MDT is the height of the mean sea surface ABOVE the geoid (CF
    standard_name mean_dynamic_topography; e.g. +0.6 m in the Sargasso Sea,
    -1.5 m in the Southern Ocean), so H_GOCO06s = H_MSL + MDT. Before
    2026-09-10 this subtracted MDT -- the GOCO06s -> MSL direction, carried
    over from an earlier pipeline that converted the DEM to local MSL (see
    workflow/archive/datum_correction/) -- putting every MSL-referenced level
    2*MDT off relative to the GOCO06s DEM.
    """
    lat_dim = next(d for d in mdt_da.dims if "lat" in d.lower())
    lon_dim = next(d for d in mdt_da.dims if "lon" in d.lower())

    result = stations.copy()
    result["rp_level_raw"] = result["rp_level"]
    result["mdt"] = [
        _nearest_valid_grid(mdt_da, lon_dim, geom.x, lat_dim, geom.y, fallback_deg)
        for geom in result.geometry
    ]
    result["rp_level"] = result["rp_level_raw"] + result["mdt"]
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
    fallback_deg: float = 3.0,
) -> gpd.GeoDataFrame:
    """
    Compute each station's AR6 SLR fingerprint (local / global-mean SLR) —
    a dimensionless ratio, independent of any target global-mean SLR value.

    Deliberately does NOT scale by a target slr_m or touch 'rp_level': the
    fingerprint depends only on the reference distribution (ssp_scenario,
    confidence_level, year, quantile), so storing just the ratio here keeps
    surge_forcing.nc (rule 07, basin-level, feeds the expensive weir/depth
    calibration and skeleton build) completely independent of the numeric
    slr_m target. The actual `fingerprint * slr_m` correction is applied
    downstream, at the point where a scenario's real production boundary
    forcing is built (see build_design_surge_matrix/lookup_storm_tide_at_rp's
    own `slr_m` argument, applied in 13_build_sfincs.py/14_run_spinup.py) --
    so changing slr_m only reruns the cheap per-scenario forcing build and
    event run, never the basin-level calibration/skeleton.

    Adds column:
        slr_fingerprint: local SLR / global_mean_slr at the nearest AR6
                         location (1.0 — i.e. the uniform global value — if
                         no valid location was found within +/-fallback_deg).
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
    callers needing a wider cap (e.g. top-level flopros_range.max_rp_yr)
    should apply it before calling this, but COAST-RP itself cannot
    extrapolate past its own tabulated range regardless.

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


def select_surge_stations(
    nc_path: str,
    domain_utm: gpd.GeoDataFrame,
    domain_crs: str,
    min_stations: int,
    max_stations: int,
    search_radii_km: list[float],
    dedupe_radius_km: float,
    return_period: int = _COASTRP_RPS[0],
) -> gpd.GeoDataFrame:
    """
    THE selection of a basin's surge stations: COAST-RP stations loaded
    (load_coastrp_stations), ranked by distance to the domain
    (compute_distances_to_bbox) and picked by select_nearest_stations.

    One function so rule select_surge_stations (07a, which decides what rule
    extract_gtsm_series extracts) and rule get_boundary_forcings (07) select
    the identical stations from the same config values. return_period only
    sets which level lands in the 'rp_level' column; it does not affect the
    selection.
    """
    stations = load_coastrp_stations(nc_path, return_period)
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
    Both ``rp_level``/``baseline_m``/``station_baselines`` are MDT-only here
    -- SLR is deliberately NOT baked in at this stage (see
    src.surge.apply_slr_fingerprint's own docstring): only the dimensionless
    ``slr_fingerprint`` is carried in the dataset (added separately below),
    and the actual target-scaled SLR correction is applied later, at the
    point a real production/spin-up boundary is built
    (lookup_storm_tide_at_rp/build_design_surge_matrix's own ``slr_m``
    argument).

    Datum note: MSL-referenced data (GEBCO in rule 05a, COAST-RP here) is
    re-referenced to GOCO06s by ADDING MDT, so local MSL maps to +MDT in
    model coordinates.  When
    ``station_baselines`` is provided each station uses its own local MSL
    (+mdt_i) as the wave baseline, so the surge amplitude equals
    exactly ``rp_level_raw`` (the COAST-RP storm-tide above calm water)
    regardless of how MDT varies spatially across the selected stations.
    ``baseline_m`` (the mean of those per-station values) is still stored in
    the dataset so rule 10's calibration and rule 13's skeleton can
    initialise sea cells (zsini.tif) at a stable, SLR-independent reference.

    Args:
        stations:          GeoDataFrame with 'rp_level' and 'dist_m' columns and
                           Point geometries in EPSG:4326.
        times:             Shared time axis from build_time_axis() (hours).
        lead_days:         Lead-in duration before wave onset (days).
        period_hr:         Wave period (hours).
        return_period:     Return period label written to the 'rp_level' metadata.
        baseline_m:        Mean vertical correction applied to rp_level (m).
                           Equals mean(+MDT) across selected stations (MDT-only,
                           SLR-independent by design -- see this function's own
                           docstring). Stored in the dataset so rule 10's
                           calibration and rule 13's skeleton can initialise sea
                           cells (zsini.tif).  Defaults to 0.0.
        station_baselines: Per-station lead-period flat values (m), length
                           equal to ``len(stations)``.  Each entry is the
                           station's own local MSL in model coordinates
                           (= rp_level_i − rp_level_raw_i = +mdt_i, MDT-only).
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
                "(mean(+MDT) across selected stations = calm sea level in "
                "model coordinates, MDT-only -- SLR deliberately excluded, "
                "see apply_slr_fingerprint). Read by rule 10's calibration "
                "and rule 13's skeleton to initialise sea cells (zsini.tif) "
                "at an slr_m-independent reference. Equals 0.0 when MDT "
                "correction is off."
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
                    "Per-station lead-period flat value (+mdt_i, MDT-only). "
                    "Local MSL for each station in model coordinates. "
                    "Ensures surge amplitude = rp_level_raw per station. "
                    "SLR is NOT included here -- see slr_fingerprint and "
                    "build_design_surge_matrix's own slr_m argument."
                ),
            },
        )

    # Optional provenance from the MDT vertical correction and SLR fingerprint
    # (src.surge.apply_mdt_correction / apply_slr_fingerprint), if present.
    # Note: no 'slr_m' column here anymore -- the fingerprint is dimensionless
    # and target-independent by design; the actual slr_m-scaled contribution
    # is computed on demand by lookup_storm_tide_at_rp/build_design_surge_matrix,
    # never stored in this file (that's the whole point: surge_forcing.nc stays
    # identical regardless of the slr_m config value).
    extra_station_vars = {
        "rp_level_raw": {
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

    # Full COAST-RP table (raw storm-tide at every tabulated RP) —
    # lets rule 13 rebuild the wave at any tabulated RP without re-running rule 07,
    # mirroring river_forcing.nc's discharge_rp_table.
    rp_cols = [
        c for c in (f"rp_raw_{rp:04d}" for rp in _COASTRP_RPS) if c in stations.columns
    ]
    if len(rp_cols) == len(_COASTRP_RPS):
        ds["storm_tide_rp_table"] = (
            ["station", "table_rp"],
            np.column_stack([stations[c].values for c in rp_cols]),
            {
                "units": "m",
                "long_name": "raw COAST-RP storm-tide level per tabulated RP",
            },
        )
        ds = ds.assign_coords(table_rp=np.asarray(_COASTRP_RPS, dtype=float))
    ds.attrs["lead_days"] = float(lead_days)
    ds.attrs["period_hr"] = float(period_hr)

    return ds
