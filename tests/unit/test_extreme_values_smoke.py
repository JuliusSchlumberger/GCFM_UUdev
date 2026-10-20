"""Smoke tests for the discharge EVA and GRDC bias correction in src.extreme_values."""

import numpy as np
import pandas as pd
import pytest
from src.extreme_values import (
    STANDARD_RETURN_PERIODS_YR,
    analyse_cell,
    bias_correct_discharge,
    empirical_quantile_map,
    estimate_decorrelation_days,
    gpd_return_value_table,
)

_EVA_CFG = {"min_days": 3650, "min_years_high_confidence": 10.0}


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
    out, diag = bias_correct_discharge(times, q, times, q * 1.2, {"enabled": False})
    assert diag is None
    np.testing.assert_array_equal(out, q)


def test_bias_correction_moves_glofas_towards_gauge():
    times, q = _daily_discharge(4)
    gauge = 1.25 * q
    out, diag = bias_correct_discharge(
        times, q, times, gauge, {"min_overlap_days": 365}
    )
    assert diag is not None
    assert out.shape == q.shape
    assert abs(np.median(out) - np.median(gauge)) < abs(np.median(q) - np.median(gauge))


def test_decorrelation_days_within_bounds():
    times, q = _daily_discharge(3)
    lag = estimate_decorrelation_days(pd.Series(q, index=times), 1.0 / np.e, 30)
    assert 1 <= lag <= 30


@pytest.fixture(scope="module")
def eva_result():
    times, q = _daily_discharge(12)
    return analyse_cell(times, q, _EVA_CFG, label="synthetic", protection_rp=20.0)


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


def test_analyse_cell_short_record_is_skipped_with_nan_levels():
    times, q = _daily_discharge(2)
    q[::7] = np.nan
    res = analyse_cell(times, q, _EVA_CFG)
    assert not res.ok
    assert np.isnan(res.q_rp2) and np.isnan(res.q_rp100)
    assert res.n_valid_days == int(np.isfinite(q).sum())
    assert res.messages
