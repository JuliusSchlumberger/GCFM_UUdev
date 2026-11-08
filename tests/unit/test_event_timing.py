"""Tests for the peak travel times and the river tail of src.event_timing."""

import numpy as np
import pytest
import xarray as xr
from src.event_timing import G_MS2, falling_limb_hr, scenario_sea_level_m, sea_travel_hr


def test_falling_limb_ends_at_the_first_hour_below_bankfull():
    t = np.arange(0.0, 400.0)
    q = 1000.0 * np.exp(-(((t - 100.0) / 40.0) ** 2))
    fall = falling_limb_hr(t, q, 100.0, 500.0, 200.0)
    assert fall == float(np.ceil(40.0 * np.sqrt(np.log(2.0))))


def test_falling_limb_is_capped_when_the_flow_stays_above_bankfull():
    t = np.arange(0.0, 400.0)
    assert falling_limb_hr(t, np.full(t.size, 900.0), 100.0, 500.0, 200.0) == 200.0


def test_rising_limb_below_bankfull_does_not_end_the_event():
    t = np.arange(0.0, 300.0)
    q = np.where(t < 100.0, 100.0, 1000.0)
    assert falling_limb_hr(t, q, 100.0, 500.0, 150.0) == 150.0


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
