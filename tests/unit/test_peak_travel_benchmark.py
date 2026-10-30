"""Smoke tests for the arrival time in analysis/benchmark_peak_travel.py."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))

from benchmark_peak_travel import arrival_hr  # noqa: E402


def test_arrival_is_the_half_rise_time_of_a_delayed_ramp():
    t = np.arange(0.0, 21.0)
    level = np.clip((t - 4.0) / 6.0, 0.0, 1.0)
    assert np.isclose(arrival_hr(t, level), 7.0)
    assert np.isclose(arrival_hr(t, level, frac=0.9), 9.4)


def test_arrival_is_nan_without_a_rise():
    t = np.arange(0.0, 5.0)
    assert np.isnan(arrival_hr(t, np.full(5, 1.0)))
    assert np.isnan(arrival_hr(t, np.array([np.nan, 0.0, 0.5, 1.0, 1.0])))
