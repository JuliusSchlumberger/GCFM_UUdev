"""Smoke tests for the 1D peak travel estimate in analysis/estimate_peak_travel.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))

from estimate_peak_travel import G_MS2, lag_and_window, river_leg, sea_leg  # noqa: E402
from src.gvf_solver import normal_depth  # noqa: E402

SLOPE, WIDTH_M, Q_M3S, N = 1e-4, 200.0, 500.0, 0.03
SPACING_M = 500.0


def _profile(n_nodes=101):
    """Uniform channel to the sea at distance 0, upstream first."""
    x = np.arange(n_nodes)[::-1] * SPACING_M
    h_n = normal_depth(Q_M3S, WIDTH_M, N, SLOPE)
    bed = SLOPE * x - h_n
    return pd.DataFrame(
        {
            "node": np.arange(n_nodes),
            "reach_id": "a",
            "x": x,
            "y": 0.0,
            "dist_from_mouth_m": x,
            "bed_m": bed,
            "width_m": WIDTH_M,
            "q_design_m3s": Q_M3S,
        }
    )


def test_fast_regime_slow_are_ordered_and_grow_downstream():
    leg = river_leg(_profile(), np.full(101, Q_M3S), 0.0, 0.0, N)
    t = {k: leg[f"t_{k}_hr"].to_numpy() for k in ("fast", "regime", "slow")}
    assert all(np.all(np.diff(v) > 0) for v in t.values())
    assert t["fast"][-1] < t["regime"][-1] < t["slow"][-1]
    assert leg["drowned"].iloc[-1] and not leg["drowned"].iloc[0]


def test_drowned_channel_regime_equals_fast():
    prof = _profile().assign(bed_m=-10.0)
    leg = river_leg(prof, np.full(101, Q_M3S), 0.0, 0.0, N)
    assert np.allclose(leg["t_regime_hr"], leg["t_fast_hr"])


def test_sea_leg_matches_gravity_wave_over_uniform_depth():
    depth_m, dx_m = 10.0, 100.0
    bed = np.full((5, 40), -depth_m)
    mask = np.ones(bed.shape, int)
    mask[:, 0] = 2
    t_hr, km = sea_leg(bed, mask, 0.0, (2, 39), dx_m)
    assert np.isclose(t_hr, 39 * dx_m / np.sqrt(G_MS2 * depth_m) / 3600.0, rtol=1e-6)
    assert np.isclose(km, 3.9)


def test_sea_leg_snaps_a_dry_mouth_to_the_nearest_wet_cell():
    bed = np.full((5, 40), -10.0)
    bed[2, 39] = 5.0
    mask = np.ones(bed.shape, int)
    mask[:, 0] = 2
    t_hr, _ = sea_leg(bed, mask, 0.0, (2, 39), 100.0)
    assert np.isfinite(t_hr)


def test_lag_material_only_beyond_tau95_and_window_follows_it():
    short = lag_and_window(
        3.0, 0.5, tau95_hr=10.0, tau50_hr=30.0, before_hr=60.0, after_hr=60.0
    )
    assert not short["lag_material"] and short["window_ok"]
    long = lag_and_window(
        40.0, 0.5, tau95_hr=10.0, tau50_hr=30.0, before_hr=60.0, after_hr=60.0
    )
    assert long["lag_material"] and np.isclose(long["lag_hr"], 39.5)
    assert not long["window_ok"]  # shifted limb needs 39.5 + 30 h before the surge peak
