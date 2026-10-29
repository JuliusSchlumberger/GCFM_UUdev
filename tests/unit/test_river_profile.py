"""Smoke tests for the walled peak travel time in workflow/src/river_profile.py."""

import numpy as np
import pandas as pd
from src.gvf_solver import normal_depth
from src.river_profile import G_MS2, walled_travel_time

SLOPE, WIDTH_M, Q_M3S, N = 1e-4, 200.0, 500.0, 0.03
SPACING_M, N_NODES = 500.0, 101


def _uniform_channel():
    """Uniform channel at normal depth to the sea at distance 0, upstream first."""
    x = np.arange(N_NODES)[::-1] * SPACING_M
    h_n = normal_depth(Q_M3S, WIDTH_M, N, SLOPE)
    return pd.DataFrame(
        {"dist_from_mouth_m": x, "bed_m": SLOPE * x - h_n, "width_m": WIDTH_M}
    ), h_n


def test_kinematic_time_matches_five_thirds_velocity_at_normal_depth():
    prof, h_n = _uniform_channel()
    t = walled_travel_time(prof, np.full(N_NODES, Q_M3S), 0.0, N, forcing_hr=6.0)
    u_n = Q_M3S / (WIDTH_M * h_n)
    length_m = prof["dist_from_mouth_m"].iloc[0]
    assert np.isclose(
        t["t_kinematic_hr"].iloc[-1], length_m / (5.0 / 3.0 * u_n) / 3600.0, rtol=0.02
    )
    assert np.isclose(
        t["t_gravity_hr"].iloc[-1],
        length_m / (u_n + np.sqrt(G_MS2 * h_n)) / 3600.0,
        rtol=0.02,
    )


def test_times_grow_downstream_and_gravity_is_the_fast_bound():
    prof, _ = _uniform_channel()
    t = walled_travel_time(prof, np.full(N_NODES, Q_M3S), 0.0, N, forcing_hr=6.0)
    assert t["t_kinematic_hr"].iloc[0] == 0.0
    assert np.all(np.diff(t["t_kinematic_hr"]) > 0)
    assert np.all(t["t_gravity_hr"].iloc[1:] < t["t_kinematic_hr"].iloc[1:])


def test_higher_sea_level_slows_the_kinematic_wave():
    prof, _ = _uniform_channel()
    calm = walled_travel_time(prof, np.full(N_NODES, Q_M3S), 0.0, N, forcing_hr=6.0)[
        "t_kinematic_hr"
    ].iloc[-1]
    raised = walled_travel_time(prof, np.full(N_NODES, Q_M3S), 3.0, N, forcing_hr=6.0)[
        "t_kinematic_hr"
    ].iloc[-1]
    assert raised > calm


def test_wave_time_takes_gravity_speed_only_where_friction_is_slower_than_the_forcing():
    prof, h_n = _uniform_channel()
    q = np.full(N_NODES, Q_M3S)
    u_n = Q_M3S / (WIDTH_M * h_n)
    r = G_MS2 * N**2 * u_n / h_n ** (4.0 / 3.0)
    for r_t, regime, speed in (
        (0.1, "inertial", "gravity"),
        (1.0, "transition", "kinematic"),
        (10.0, "frictional", "kinematic"),
    ):
        t = walled_travel_time(prof, q, 0.0, N, forcing_hr=r_t / r / 3600.0)
        assert set(t["regime"]) == {regime}
        np.testing.assert_allclose(t["friction_rT"], r_t, rtol=0.05)
        np.testing.assert_allclose(t["t_wave_hr"], t[f"t_{speed}_hr"])
