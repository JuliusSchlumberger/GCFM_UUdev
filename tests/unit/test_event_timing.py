"""Tests for the peak travel times and the river tail of src.event_timing."""

import numpy as np
import pytest
import xarray as xr
from src.event_timing import (
    G_MS2,
    event_start_discharge,
    event_window,
    scenario_sea_level_m,
    sea_travel_hr,
)


def _flood():
    t = np.arange(0.0, 400.0)
    return t, 100.0 + 900.0 * np.exp(-(((t - 200.0) / 40.0) ** 2))


def test_event_window_starts_and_ends_where_the_flow_crosses_q_start():
    t, q = _flood()
    start, end = event_window(t, q, 200.0, 550.0, 168.0, 100.0)
    half = 40.0 * np.sqrt(np.log(2.0))  # q = 550 at 200 +- half
    assert start == pytest.approx(200.0 - half, abs=0.5)
    assert end == float(np.ceil(200.0 + half))


def test_event_window_is_capped_before_and_after_the_peak():
    t = np.arange(0.0, 400.0)
    start, end = event_window(t, np.full(t.size, 900.0), 200.0, 500.0, 50.0, 100.0)
    assert (start, end) == (150.0, 300.0)


def test_q_start_is_the_protection_discharge_unless_the_event_stays_below_it():
    assert event_start_discharge(100.0, 1100.0, 400.0, 0.5) == 400.0
    assert (
        event_start_discharge(100.0, 1100.0, 1000.0, 0.5) == 1000.0
    )  # above halfway too
    assert event_start_discharge(100.0, 500.0, 800.0, 0.5) == 300.0
    assert event_start_discharge(100.0, 500.0, None, 0.5) == 300.0


def test_sea_travel_is_distance_over_shallow_water_speed():
    bed = np.full((3, 50), -10.0)
    mask = np.ones(bed.shape, dtype=int)
    mask[:, 0] = 2
    hours, km = sea_travel_hr(bed, mask, 0.0, (1, 49), 100.0)
    assert km == 4.9
    assert np.isclose(hours, 4900.0 / np.sqrt(G_MS2 * 10.0) / 3600.0)


def test_sea_travel_snaps_a_dry_mouth_to_the_nearest_wet_cell():
    bed = np.full((3, 20), -5.0)
    bed[1, 19] = 2.0
    mask = np.ones(bed.shape, dtype=int)
    mask[:, 0] = 2
    hours, _ = sea_travel_hr(bed, mask, 0.0, (1, 19), 100.0)
    assert np.isfinite(hours) and hours > 0


def _surge_ds():
    return xr.Dataset(
        {
            "station_calm_sea_level_m": ("station", [0.1, 0.3]),
            "slr_fingerprint": ("station", [1.0, 1.2]),
            "storm_tide_rp_table": (("station", "rp"), [[1.0, 2.0], [1.4, 2.2]]),
        },
        coords={"rp_yr": ("rp", [10.0, 100.0])},
    )


def test_surge_scenario_sea_is_the_storm_tide_peak_and_others_the_calm_sea():
    ds = _surge_ds()
    calm = scenario_sea_level_m(ds, "Tide", 0.5)
    assert calm == pytest.approx(np.mean([0.1 + 0.5, 0.3 + 0.6]))
    storm = scenario_sea_level_m(ds, 100, 0.5)
    assert storm == pytest.approx(np.mean([2.0 + 0.6, 2.2 + 0.9]))
    assert scenario_sea_level_m(ds, None, 0.0) < storm
