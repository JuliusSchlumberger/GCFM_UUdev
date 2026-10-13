"""
storm_tide_hydrograph.py -- stylised storm-tide hydrograph components from
the hourly GTSM reanalysis (total water level + surge residual) at one
station: the average tide and a normalised surge shape PER RETURN PERIOD.

Method: HGRAPHER (Dullaart et al. 2023, NHESS 23:1847,
https://doi.org/10.5194/nhess-23-1847-2023), as ported to hourly data in
tests/KL_gtsm_storm_tide.py, with one change -- the surge shape is no longer
one average over the largest surge peaks, applied to every return period:

  tide    = total water level - surge residual, after removing the slowly
            varying mean sea level (1-yr running mean), both interpolated
            hourly -> 10 min
  average tide: mean of all 24 h 50 min cycles, anchored on the daily
            higher high water (or lower low water, whichever is larger)
  events  = peaks at least 72 h apart of the TOTAL water level
            (match_on="water_level") or of the surge residual
            (match_on="surge"); each event's surge curve is centred on its
            surge maximum (for water-level peaks: the maximum within
            SURGE_PEAK_SEARCH_HR of the peak) and normalised by it
  shape for a return level: the events whose peak lies within a window
            around the target -- the return level itself for
            "water_level", the surge it needs (return level - average high
            water) for "surge". With fewer than MIN_EVENTS in the window the
            MIN_EVENTS nearest events are taken; a target above every
            recorded peak takes the MIN_EVENTS largest. Per limb, the mean
            time those events spend above each fraction 0..1 of their own
            surge peak (HGRAPHER's duration averaging), over
            +-half_window_hr.

Levels are relative to a fixed local mean sea level, the datum of COAST-RP.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.signal as ss

log = logging.getLogger(__name__)

STEPS_PER_HR = 6  # 10-min, as HGRAPHER
LUNAR_STEPS = 149  # 24 h 50 min
LEVELS = np.round(np.arange(0, 1.0001, 0.005), 3)
EVENT_SEPARATION_HR = 72
SURGE_PEAK_SEARCH_HR = 12.5  # one tidal cycle either side of the water-level peak
MSL_WINDOW_HR = 8766  # 1 year


def fill_short_gaps(series: pd.Series, max_gap_hr: int = 3) -> pd.Series | None:
    """Linear fill of gaps up to max_gap_hr; None if longer gaps remain."""
    filled = series.interpolate(limit=max_gap_hr, limit_area="inside")
    return None if filled.isna().any() else filled


def remove_mean_sea_level(twl: pd.Series, surge: pd.Series) -> pd.Series:
    """Total water level with the slowly varying mean sea level removed
    (1-yr centred running mean of twl - surge), so tide and peaks refer to a
    fixed mean sea level like COAST-RP.
    """
    msl = (
        (twl - surge)
        .rolling(MSL_WINDOW_HR, center=True, min_periods=MSL_WINDOW_HR // 2)
        .mean()
    )
    return twl - msl


def to_10min(series: pd.Series) -> np.ndarray:
    return series.resample("10min").interpolate("cubic").values


def average_tide_signal(
    tide10: np.ndarray, half_window_hr: float
) -> tuple[np.ndarray, int]:
    """Average tide on -half_window_hr..+half_window_hr (10-min), high water at t=0.

    Returns:
        (signal (2 * half_window_hr * 6 + 1,), number of tidal cycles averaged).
    """
    use_min = abs(np.quantile(tide10, 0.99)) < abs(np.quantile(tide10, 0.01))
    arg = np.argmin if use_min else np.argmax
    anchors = [int(arg(tide10[:150]))]
    while True:
        c = anchors[-1] + LUNAR_STEPS
        if c + 25 > len(tide10):
            break
        anchors.append(c - 24 + int(arg(tide10[c - 24 : c + 25])))
    cycles = np.array(
        [tide10[a : a + LUNAR_STEPS] for a in anchors if a + LUNAR_STEPS <= len(tide10)]
    )
    mean_cycle = cycles.mean(axis=0)
    half = int(round(half_window_hr * STEPS_PER_HR))
    n_tiles = 2 * (half // LUNAR_STEPS + 2) + 1
    i_max = int(mean_cycle.argmax()) + (n_tiles // 2) * LUNAR_STEPS
    return np.tile(mean_cycle, n_tiles)[i_max - half : i_max + half + 1], len(cycles)


def storm_events(
    twl10: np.ndarray, surge10: np.ndarray, half_window_hr: float, match_on: str
) -> dict:
    """Every peak (>= 72 h apart) of the total water level (match_on=
    "water_level") or of the surge residual ("surge") that has a positive
    surge, and the time its surge spends above each fraction of its own peak.

    Returns dict of arrays over events:
        twl_peak_m, surge_peak_m (the event's highest total water level /
        surge within SURGE_PEAK_SEARCH_HR), index (10-min position of the
        surge peak), dur_before / dur_after (n_events, len(LEVELS)): 10-min
        steps above each level on the rising / falling limb within
        half_window_hr.
    """
    if match_on not in ("water_level", "surge"):
        raise ValueError(f"match_on must be 'water_level' or 'surge', got {match_on!r}")
    half = int(round(half_window_hr * STEPS_PER_HR))
    search = int(round(SURGE_PEAK_SEARCH_HR * STEPS_PER_HR))
    series = twl10 if match_on == "water_level" else surge10
    peaks, _ = ss.find_peaks(series, distance=EVENT_SEPARATION_HR * STEPS_PER_HR)
    out = {
        k: []
        for k in ("twl_peak_m", "surge_peak_m", "index", "dur_before", "dur_after")
    }
    for p in peaks:
        lo, hi = max(p - search, 0), min(p + search + 1, len(surge10))
        s = p if match_on == "surge" else lo + int(np.argmax(surge10[lo:hi]))
        peak = surge10[s]
        if peak <= 0 or s - half < 0 or s + half >= len(surge10):
            continue
        before = surge10[s - half : s + 1] / peak
        after = surge10[s : s + half + 1] / peak
        neg_b, neg_a = np.where(before < 0)[0], np.where(after < 0)[0]
        before = before[neg_b[-1] :] if neg_b.size else before[1:]
        after = after[: neg_a[0]] if neg_a.size else after[:half]
        out["twl_peak_m"].append(twl10[lo:hi].max())
        out["surge_peak_m"].append(peak)
        out["index"].append(s)
        out["dur_before"].append((before[:, None] > LEVELS).sum(axis=0))
        out["dur_after"].append((after[:, None] > LEVELS).sum(axis=0))
    return {k: np.asarray(v) for k, v in out.items()}


def select_events_for_level(
    peaks: np.ndarray,
    target: float,
    min_events: int,
    window_m: float,
) -> tuple[np.ndarray, float, str]:
    """Events whose peak is within +-window_m of `target`.

    With fewer than min_events inside, the window is widened to the
    min_events nearest events ("widened"). A target above every recorded
    peak takes the min_events largest events instead ("above_record").

    Returns:
        (indices into peaks, half-window finally used (m), "window" |
        "widened" | "above_record").
    """
    n = min(min_events, len(peaks))
    if target > peaks.max():
        idx = np.argsort(peaks)[-n:]
        return idx, float(target - peaks[idx].min()), "above_record"
    dist = np.abs(peaks - target)
    idx = np.flatnonzero(dist <= window_m)
    if len(idx) >= n:
        return idx, float(window_m), "window"
    idx = np.argsort(dist)[:n]
    return idx, float(dist[idx].max()), "widened"


def surge_shape(
    dur_before: np.ndarray, dur_after: np.ndarray, half_window_hr: float
) -> np.ndarray:
    """Normalised surge (peak = 1 at t = 0) on -half_window_hr..+half_window_hr
    (10-min) from the mean per-level durations of the two limbs.
    """
    half = int(round(half_window_hr * STEPS_PER_HR))
    steps = np.arange(-half, half + 1)
    rise = np.interp(np.abs(steps), dur_before[::-1], LEVELS[::-1], right=0.0)
    fall = np.interp(np.abs(steps), dur_after[::-1], LEVELS[::-1], right=0.0)
    return np.where(steps <= 0, rise, fall)


def station_hydrograph_components(
    twl: pd.Series,
    surge: pd.Series,
    return_levels: np.ndarray,
    half_window_hr: float,
    min_events: int,
    window_m: float,
    match_on: str = "water_level",
) -> dict | None:
    """Average tide + one surge shape per return level for one station.

    Args:
        twl, surge:     hourly total water level and surge residual (m).
        return_levels:  storm-tide level per return period (m, local MSL --
                        COAST-RP), in the order the shapes are returned.
        half_window_hr: half-length of the tide/shape signals (h).
        min_events, window_m: see select_events_for_level.
        match_on:       "water_level" -- events matched by their peak total
                        water level against the return level; "surge" -- by
                        their surge peak against return level - average high
                        water.

    Returns:
        None if the series has gaps longer than 3 h; else a dict with
        time_hr (n_t,), tide (n_t,), shape (n_rp, n_t), and per return
        period n_events, window_m, mode, surge_peak_m (median surge peak of
        the events used), dur50_hr (hours the shape stays above 0.5); plus
        n_tidal_cycles, n_candidate_events, max_peak_m (highest recorded
        peak of the matched quantity), high_water_m.
    """
    twl, surge = fill_short_gaps(twl), fill_short_gaps(surge)
    if twl is None or surge is None:
        return None
    twl = remove_mean_sea_level(twl, surge)
    twl10, surge10 = to_10min(twl), to_10min(surge)
    tide, n_cycles = average_tide_signal(twl10 - surge10, half_window_hr)
    events = storm_events(twl10, surge10, half_window_hr, match_on)
    high_water = float(tide.max())
    peaks = (
        events["twl_peak_m"] if match_on == "water_level" else events["surge_peak_m"]
    )
    half = int(round(half_window_hr * STEPS_PER_HR))
    n_rp = len(return_levels)
    out = {
        "time_hr": np.arange(-half, half + 1) / STEPS_PER_HR,
        "tide": tide,
        "shape": np.full((n_rp, 2 * half + 1), np.nan),
        "n_events": np.zeros(n_rp, dtype=int),
        "window_m": np.full(n_rp, np.nan),
        "mode": np.empty(n_rp, dtype=object),
        "surge_peak_m": np.full(n_rp, np.nan),
        "dur50_hr": np.full(n_rp, np.nan),
        "n_tidal_cycles": n_cycles,
        "n_candidate_events": len(peaks),
        "max_peak_m": float(peaks.max()),
        "high_water_m": high_water,
    }
    for j, level in enumerate(return_levels):
        target = (
            float(level) if match_on == "water_level" else float(level) - high_water
        )
        idx, w, mode = select_events_for_level(peaks, target, min_events, window_m)
        shape = surge_shape(
            events["dur_before"][idx].mean(axis=0),
            events["dur_after"][idx].mean(axis=0),
            half_window_hr,
        )
        out["shape"][j] = shape
        out["n_events"][j], out["window_m"][j], out["mode"][j] = len(idx), w, mode
        out["surge_peak_m"][j] = float(np.median(events["surge_peak_m"][idx]))
        out["dur50_hr"][j] = float((shape > 0.5).sum()) / STEPS_PER_HR
    return out
