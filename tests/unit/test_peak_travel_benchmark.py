"""Smoke tests for the settle time in analysis/benchmark_peak_travel.py."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))

from benchmark_peak_travel import settle_hr  # noqa: E402


def test_settle_is_when_a_delayed_ramp_comes_within_tol_of_its_end():
    t = np.arange(0.0, 21.0)
    level = np.clip((t - 4.0) / 6.0, 0.0, 1.0)
    assert np.isclose(settle_hr(t, level, 0.05), 9.7)


def test_settle_sees_a_late_second_rise_after_an_early_one():
    t = np.arange(0.0, 21.0)
    sea_then_river = (
        np.clip(t / 2.0, 0.0, 1.0) * 0.5 + np.clip((t - 12.0) / 2.0, 0.0, 1.0) * 0.2
    )
    assert np.isclose(settle_hr(t, sea_then_river, 0.05), 13.5)


def test_settle_is_nan_without_a_change():
    t = np.arange(0.0, 5.0)
    assert np.isnan(settle_hr(t, np.full(5, 1.0), 0.05))
    assert np.isnan(settle_hr(t, np.array([0.0, 0.5, 1.0, 1.0, np.nan]), 0.05))
