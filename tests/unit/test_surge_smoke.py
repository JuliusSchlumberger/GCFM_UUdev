"""Smoke tests for storm-tide levels and design boundary series in src.surge."""

import numpy as np
import pytest
import xarray as xr
from src.surge import (
    _COASTRP_RPS,
    build_design_surge_matrix,
    build_time_axis,
    calm_sea_levels,
    ceil_crest,
    lookup_storm_tide_at_rp,
    read_baseline_m,
    sinusoidal_wave,
    storm_tide_at_rp_interpolated,
    storm_tide_event,
)

_LEAD_DAYS = 1.0
_PERIOD_HR = 24.0


def _surge_ds(with_hydrograph: bool = False) -> xr.Dataset:
    rps = np.array(_COASTRP_RPS, dtype=float)
    table = np.stack([1.0 + 0.3 * np.log(rps), 1.5 + 0.4 * np.log(rps)])
    times = build_time_axis(_LEAD_DAYS, _PERIOD_HR, 0.5)
    ds = xr.Dataset(
        {
            "storm_tide_rp_table": (("station", "table_rp"), table),
            "station_baseline": ("station", np.array([0.123, -0.051])),
            "baseline_m": 0.036,
            "slr_fingerprint": ("station", np.array([1.0, 0.8])),
        },
        coords={"table_rp": rps, "time": times},
        attrs={"lead_days": _LEAD_DAYS, "period_hr": _PERIOD_HR},
    )
    if with_hydrograph:
        hg_t = np.arange(-6.0, 6.01, 0.5)
        tide = 0.5 * np.cos(2.0 * np.pi * hg_t / 12.42)
        shape = np.clip(1.0 - np.abs(hg_t) / 6.0, 0.0, 1.0)
        hg_rps = np.array([1.0, 10.0, 100.0])
        ds["hg_tide_m"] = (("station", "hg_time"), np.stack([tide, tide]))
        ds["hg_surge_shape"] = (
            ("station", "hg_rp", "hg_time"),
            np.broadcast_to(shape, (2, hg_rps.size, hg_t.size)).copy(),
        )
        ds = ds.assign_coords(hg_time_hr=("hg_time", hg_t), hg_rp=hg_rps)
        ds.attrs.update(hg_peak_hr=_LEAD_DAYS * 24.0 + 12.0, hg_ramp_hours=2.0)
    return ds


def test_storm_tide_level_increases_with_rp_yr():
    ds = _surge_ds()
    rp2, rp10, rp100 = (lookup_storm_tide_at_rp(ds, rp) for rp in (2, 10, 100))
    assert np.all(rp100 >= rp10) and np.all(rp10 >= rp2)
    assert np.all(rp2 > calm_sea_levels(ds))


def test_lookup_rejects_untabulated_rp():
    with pytest.raises(ValueError, match="not tabulated"):
        lookup_storm_tide_at_rp(_surge_ds(), 30)


def test_interpolated_level_lies_between_neighbours_and_matches_table():
    ds = _surge_ds()
    lo, hi = lookup_storm_tide_at_rp(ds, 25), lookup_storm_tide_at_rp(ds, 50)
    mid = storm_tide_at_rp_interpolated(ds, 30.0)
    assert np.all(mid >= lo) and np.all(mid <= hi)
    np.testing.assert_allclose(storm_tide_at_rp_interpolated(ds, 50.0), hi)
    np.testing.assert_allclose(
        storm_tide_at_rp_interpolated(ds, 1e6), lookup_storm_tide_at_rp(ds, 1000)
    )


def test_levels_are_on_centimetre_grid_and_crest_never_below_input():
    ds = _surge_ds()
    levels = lookup_storm_tide_at_rp(ds, 100, slr_m=0.37)
    np.testing.assert_allclose(levels * 100.0, np.round(levels * 100.0), atol=1e-9)
    assert read_baseline_m(ds) == pytest.approx(0.04)
    raw = np.array([2.75, 2.751, -0.004, np.nan])
    crest = ceil_crest(raw)
    assert np.all(crest[:3] >= raw[:3])
    assert np.all(crest[:3] - raw[:3] < 0.01)
    assert np.isnan(crest[3])


def test_slr_raises_every_station_by_its_fingerprint():
    ds = _surge_ds()
    diff = lookup_storm_tide_at_rp(ds, 10, slr_m=1.0) - lookup_storm_tide_at_rp(ds, 10)
    np.testing.assert_allclose(diff, ds["slr_fingerprint"].values, atol=0.011)


def test_sinusoidal_wave_peaks_at_design_level_and_spans_window():
    times = build_time_axis(_LEAD_DAYS, _PERIOD_HR, 1.0)
    wave = sinusoidal_wave(0.2, 3.0, times, _LEAD_DAYS, _PERIOD_HR)
    assert times[0] == 0.0 and times[-1] == pytest.approx(_LEAD_DAYS * 24 + _PERIOD_HR)
    assert wave.shape == times.shape
    assert wave.max() == pytest.approx(3.0)
    assert wave.min() == pytest.approx(0.2)
    assert wave[0] == pytest.approx(0.2) and wave[-1] == pytest.approx(0.2)


def test_design_matrix_without_hydrograph_peaks_at_rp_level():
    ds = _surge_ds()
    m = build_design_surge_matrix(ds, 100)
    assert m.shape == (2, ds.sizes["time"])
    np.testing.assert_allclose(m.max(axis=1), lookup_storm_tide_at_rp(ds, 100))
    np.testing.assert_allclose(m[:, 0], calm_sea_levels(ds))
    flat = build_design_surge_matrix(ds, None)
    np.testing.assert_allclose(flat, calm_sea_levels(ds)[:, None] * np.ones_like(m))
    with pytest.raises(ValueError, match="Tide"):
        build_design_surge_matrix(ds, "tide")


def test_design_matrix_with_hydrograph_peaks_at_rp_level_from_calm_start():
    ds = _surge_ds(with_hydrograph=True)
    m = build_design_surge_matrix(ds, 10)
    level = lookup_storm_tide_at_rp(ds, 10)
    np.testing.assert_allclose(m.max(axis=1), level, atol=1e-9)
    np.testing.assert_allclose(m[:, 0], calm_sea_levels(ds))
    tide_only = build_design_surge_matrix(ds, "Tide")
    assert np.all(tide_only.max(axis=1) < level)


def test_storm_tide_event_hits_peak_and_holds_tide_outside_window():
    ds = _surge_ds(with_hydrograph=True)
    rel_hr = np.linspace(-20.0, 20.0, 81)
    level = storm_tide_at_rp_interpolated(ds, 30.0)
    event = storm_tide_event(ds, rel_hr, level, rp_yr=30.0)
    tide = storm_tide_event(ds, rel_hr, None)
    assert event.shape == (2, rel_hr.size)
    np.testing.assert_allclose(event[:, rel_hr == 0.0].ravel(), level, atol=1e-9)
    assert np.all(event >= tide - 1e-12)
    outside = np.abs(rel_hr) > 6.0
    np.testing.assert_allclose(event[:, outside], tide[:, outside])
    with pytest.raises(ValueError, match="rp_yr is required"):
        storm_tide_event(ds, rel_hr, level)
