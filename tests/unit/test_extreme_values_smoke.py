"""Smoke tests for the discharge EVA and GRDC bias correction in src.extreme_values."""

import numpy as np
import pandas as pd
import pytest
import src.extreme_values as ev
from src.extreme_values import (
    STANDARD_RETURN_PERIODS_YR,
    EVAFitError,
    EVAResult,
    analyse_cell,
    bias_correct_discharge,
    empirical_quantile_map,
    estimate_decorrelation_days,
    eva_active_flags,
    gpd_return_value_table,
)

_EVA_CFG = {
    "min_days": 3650,
    "min_years_high_confidence": 10.0,
    "fit_method": "Lmoments",
    "bankfull_rp_yr": 2.0,
    "preview_rp_yr": 100.0,
    "ci_alpha": 0.95,
    "n_boot": 200,
    "decorr_cut": 1.0 / np.e,
    "decorr_max": 30,
    "deseasonalize_for_decorr": True,
    "threshold_start_pct": 90.0,
    "threshold_step_pct": 1.0,
    "threshold_max_pct": 99.5,
    "threshold_min_pct": 50.0,
    "threshold_max_iter": 30,
    "peaks_per_year_min": 1.0,
    "peaks_per_year_max": 5.0,
}
_BIAS_CFG = {"enabled": True, "min_overlap_days": 730, "tail_percentile": 90.0}


def _daily_discharge(n_years: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    times = pd.date_range("1990-01-01", periods=int(n_years * 365.25), freq="D")
    doy = times.dayofyear.values
    season = 1000.0 + 600.0 * np.sin(2.0 * np.pi * doy / 365.25)
    noise = np.empty(times.size)
    noise[0] = 0.0
    for i in range(1, times.size):
        noise[i] = 0.8 * noise[i - 1] + rng.normal(0.0, 0.25)
    return times.values, season * np.exp(noise)


def test_gpd_return_value_table_increases_with_rp_yr():
    table = gpd_return_value_table(500.0, 200.0, 0.1, 2.0)
    assert table.shape == STANDARD_RETURN_PERIODS_YR.shape
    assert np.all(np.isfinite(table))
    assert np.all(np.diff(table) >= 0.0)
    assert table[0] >= 500.0


def test_gpd_return_value_table_is_nan_for_failed_fit():
    table = gpd_return_value_table(500.0, np.nan, 0.1, 2.0, np.array([2.0, 100.0]))
    assert table.shape == (2,)
    assert np.isnan(table).all()


def test_quantile_map_is_monotonic_non_negative_and_extrapolates_tail():
    rng = np.random.default_rng(1)
    src = rng.gamma(2.0, 100.0, 2000)
    tgt = 1.3 * src + 20.0
    full = np.concatenate([src, [src.max() * 2.0]])
    out = empirical_quantile_map(src, tgt, full)
    assert out.shape == full.shape
    assert np.all(out >= 0.0)
    order = np.argsort(full)
    assert np.all(np.diff(out[order]) >= -1e-9)
    assert out[-1] > tgt.max()


def test_bias_correction_disabled_returns_input_unchanged():
    times, q = _daily_discharge(3)
    out, diag = bias_correct_discharge(
        times, q, times, q * 1.2, {**_BIAS_CFG, "enabled": False}
    )
    assert diag is None
    np.testing.assert_array_equal(out, q)


def test_bias_correction_moves_glofas_towards_gauge():
    times, q = _daily_discharge(4)
    gauge = 1.25 * q
    out, diag = bias_correct_discharge(
        times, q, times, gauge, {**_BIAS_CFG, "min_overlap_days": 365}
    )
    assert diag is not None
    assert out.shape == q.shape
    assert abs(np.median(out) - np.median(gauge)) < abs(np.median(q) - np.median(gauge))


def test_bias_correction_handles_gaps_duplicates_and_unsorted_days():
    times, q = _daily_discharge(4)
    gauge = 1.25 * q
    q_gappy = q.copy()
    q_gappy[10:40] = np.nan
    order = np.r_[np.arange(times.size)[::-1], [5]]  # reversed, one day twice
    out, diag = bias_correct_discharge(
        np.asarray(times)[order],
        q_gappy[order],
        times,
        gauge,
        {**_BIAS_CFG, "min_overlap_days": 365},
    )
    assert diag is not None
    assert out.shape == order.shape
    assert np.isnan(out[np.isnan(q_gappy[order])]).all()
    assert diag["corrected_overlap"].shape == diag["grdc_overlap"].shape
    assert diag["correlation_full_corrected"] > 0.99


def test_decorrelation_days_within_bounds():
    times, q = _daily_discharge(3)
    lag = estimate_decorrelation_days(pd.Series(q, index=times), 1.0 / np.e, 30)
    assert 1 <= lag <= 30


@pytest.fixture(scope="module")
def eva_result():
    times, q = _daily_discharge(12)
    return analyse_cell(times, q, _EVA_CFG, label="synthetic", protection_rp_yr=20.0)


def test_analyse_cell_rp100_exceeds_rp2_with_ordered_ci(eva_result):
    res = eva_result
    assert res.ok
    assert res.q_rp100 > res.q_rp2 > 0.0
    for lo, hi in (res.q_rp2_ci, res.q_rp100_ci):
        assert np.isfinite(lo) and np.isfinite(hi)
        assert lo <= hi
    assert res.pot_n_peaks > 0
    assert res.amax_n_blocks >= 11


def test_protection_rp_level_lies_between_rp2_and_rp100(eva_result):
    assert eva_result.q_rp2 < eva_result.q_protection < eva_result.q_rp100


def test_protection_level_equals_discharge_rp_table_at_that_rp(eva_result):
    res = eva_result
    table = gpd_return_value_table(
        res.pot_threshold,
        res.pot_scale,
        res.pot_shape,
        res.pot_peaks_per_year,
        np.array([20.0]),
    )
    assert res.q_protection == pytest.approx(table[0], rel=1e-12)


def test_analyse_cell_requires_n_boot():
    times, q = _daily_discharge(2)
    cfg = {k: v for k, v in _EVA_CFG.items() if k != "n_boot"}
    with pytest.raises(KeyError, match="n_boot"):
        analyse_cell(times, q, cfg)


def test_analyse_cell_short_record_is_skipped_with_nan_levels():
    times, q = _daily_discharge(2)
    q[::7] = np.nan
    res = analyse_cell(times, q, _EVA_CFG)
    assert not res.ok
    assert np.isnan(res.q_rp2) and np.isnan(res.q_rp100)
    assert res.n_valid_days == int(np.isfinite(q).sum())
    assert res.messages
    assert res.skip_reason == "record too short"


def test_valid_eva_is_active(eva_result):
    assert eva_active_flags([("synthetic", eva_result)]) == [True]


def test_forced_pot_failure_is_not_ok_and_fails_the_crossing(monkeypatch):
    def _fail(*_args, **_kwargs):
        raise ValueError("forced POT failure")

    monkeypatch.setattr(ev, "estimate_decorrelation_days", _fail)
    times, q = _daily_discharge(12)
    res = analyse_cell(times, q, _EVA_CFG, label="forced")
    assert np.isfinite(res.q_rp2)
    assert np.isnan(res.q_rp100)
    assert not res.ok and not res.skip_reason
    with pytest.raises(EVAFitError, match="crossing0.*forced POT failure"):
        eva_active_flags([("crossing0", res)])


def test_missing_fit_dependency_propagates(monkeypatch):
    def _missing(*_args, **_kwargs):
        raise ImportError("lmoments3 not installed")

    monkeypatch.setattr(ev.EVA, "fit_model", _missing)
    times, q = _daily_discharge(12)
    with pytest.raises(ImportError):
        analyse_cell(times, q, _EVA_CFG)


def test_threshold_search_exhaustion_raises():
    times, q = _daily_discharge(12)
    cfg = {
        **_EVA_CFG,
        "peaks_per_year_min": 200.0,
        "peaks_per_year_max": 300.0,
        "threshold_start_pct": 60.0,
        "threshold_min_pct": 58.0,
    }
    with pytest.raises(EVAFitError, match="threshold search"):
        analyse_cell(times, q, cfg, label="exhausted")


def test_skipped_crossing_is_inactive_next_to_an_active_one(eva_result):
    skipped = EVAResult(skip_reason="record too short")
    assert eva_active_flags([("a", eva_result), ("b", skipped)]) == [True, False]


def test_all_crossings_skipped_raises():
    skipped = EVAResult(skip_reason="record too short")
    with pytest.raises(EVAFitError, match="None of 1"):
        eva_active_flags([("a", skipped)])


def test_no_analysed_crossing_is_not_an_error():
    assert eva_active_flags([]) == []


def test_analyse_cell_missing_config_key_names_its_path():
    times, q = _daily_discharge(12)
    cfg = {k: v for k, v in _EVA_CFG.items() if k != "bankfull_rp_yr"}
    with pytest.raises(
        KeyError, match=r"boundary_forcings\.river\.eva\.bankfull_rp_yr"
    ):
        analyse_cell(times, q, cfg)


def test_bias_diagnostics_roundtrip_keeps_arrays_scalars_and_times(tmp_path):
    from src.extreme_values import read_bias_diagnostics, write_bias_diagnostics

    times = pd.date_range("2000-01-01", periods=4, freq="D").values
    diag = {
        "glofas_overlap": np.array([1.0, 2.0, 3.0, 4.0]),
        "tail_mask": np.array([False, True, True, False]),
        "glofas_overlap_times": times,
        "tail_percentile": 90.0,
        "grdc_station_id": 6123,
        "correlation_raw": None,
    }
    path = tmp_path / "bias.npz"
    write_bias_diagnostics(
        path,
        [
            {
                "label": "crossing0_cell(1.000,2.000)",
                "i_lat": 3,
                "i_lon": 4,
                "diagnostics": diag,
            }
        ],
    )
    (cell,) = read_bias_diagnostics(path)
    assert (cell["label"], cell["i_lat"], cell["i_lon"]) == (
        "crossing0_cell(1.000,2.000)",
        3,
        4,
    )
    d = cell["diagnostics"]
    assert np.array_equal(d["tail_mask"], diag["tail_mask"])
    assert np.array_equal(d["glofas_overlap_times"], times)
    assert d["tail_percentile"] == 90.0 and d["grdc_station_id"] == 6123
    assert np.isnan(d["correlation_raw"])


def test_bias_diagnostics_without_cells_reads_empty(tmp_path):
    from src.extreme_values import read_bias_diagnostics, write_bias_diagnostics

    write_bias_diagnostics(tmp_path / "b.npz", [])
    assert read_bias_diagnostics(tmp_path / "b.npz") == []
