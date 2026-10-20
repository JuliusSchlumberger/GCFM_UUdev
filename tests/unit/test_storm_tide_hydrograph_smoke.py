"""Smoke tests for the HGRAPHER tide and surge-shape components in src.storm_tide_hydrograph."""

import numpy as np
import pandas as pd
import pytest
from src.storm_tide_hydrograph import (
    STEPS_PER_HR,
    select_events_for_level,
    station_hydrograph_components,
)

_HALF_WINDOW_HR = 24.0


def _hourly_station(n_days: int = 400, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=n_days * 24, freq="h")
    t_hr = np.arange(idx.size, dtype=float)
    tide = 1.0 * np.cos(2.0 * np.pi * t_hr / 12.42) + 0.3 * np.cos(
        2.0 * np.pi * t_hr / 12.0
    )
    surge = rng.normal(0.0, 0.02, idx.size)
    for centre in np.arange(100, idx.size - 100, 24 * 6):
        amp = rng.uniform(0.2, 1.5)
        surge += amp * np.exp(-0.5 * ((t_hr - centre) / 8.0) ** 2)
    return pd.Series(tide + surge, index=idx), pd.Series(surge, index=idx)


@pytest.fixture(scope="module")
def components():
    twl, surge = _hourly_station()
    return station_hydrograph_components(
        twl,
        surge,
        np.array([1.8, 2.2, 3.5]),
        _HALF_WINDOW_HR,
        min_events=5,
        window_m=0.2,
    )


def test_components_span_window_with_one_shape_per_return_level(components):
    n_t = int(2 * _HALF_WINDOW_HR * STEPS_PER_HR + 1)
    assert components["time_hr"][0] == -_HALF_WINDOW_HR
    assert components["time_hr"][-1] == _HALF_WINDOW_HR
    assert components["tide"].shape == (n_t,)
    assert components["shape"].shape == (3, n_t)
    assert np.all(components["n_events"] >= 5)


def test_surge_shape_peaks_at_one_on_t0_and_stays_in_unit_interval(components):
    shape = components["shape"]
    t0 = int(np.flatnonzero(components["time_hr"] == 0.0)[0])
    assert np.all(np.isfinite(shape))
    assert np.all(shape >= 0.0) and np.all(shape <= 1.0)
    np.testing.assert_allclose(shape[:, t0], 1.0, atol=0.01)
    assert np.all(components["dur50_hr"] > 0.0)


def test_average_tide_has_high_water_at_t0(components):
    tide = components["tide"]
    t0 = int(np.flatnonzero(components["time_hr"] == 0.0)[0])
    assert tide[t0] == pytest.approx(tide.max())
    assert components["high_water_m"] == pytest.approx(tide.max())
    assert components["n_tidal_cycles"] > 300


def test_return_level_above_record_uses_largest_events(components):
    assert components["mode"][-1] == "above_record"
    assert components["max_peak_m"] < 3.5


def test_long_gap_returns_none():
    twl, surge = _hourly_station(n_days=60)
    twl.iloc[100:110] = np.nan
    assert (
        station_hydrograph_components(twl, surge, np.array([2.0]), 24.0, 5, 0.2) is None
    )


def test_select_events_widens_when_window_too_sparse():
    peaks = np.array([1.0, 1.1, 1.5, 2.0, 2.6])
    idx, w, mode = select_events_for_level(peaks, 1.05, min_events=2, window_m=0.1)
    assert mode == "window" and w == 0.1 and set(idx) == {0, 1}
    idx, w, mode = select_events_for_level(peaks, 2.3, min_events=2, window_m=0.1)
    assert mode == "widened" and len(idx) == 2 and w >= 0.1
    idx, _, mode = select_events_for_level(peaks, 3.0, min_events=2, window_m=0.1)
    assert mode == "above_record" and set(idx) == {3, 4}
