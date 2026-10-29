"""Smoke tests for the mainstem profile helpers in analysis/_upstream_profile.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))

from _upstream_profile import (  # noqa: E402
    forward_levels,
    mainstem_path,
    mainstem_profile,
    sample_sfincs_at_nodes,
    scaled_discharge,
    surface_mismatch,
)
from src.gvf_solver import normal_depth  # noqa: E402

SLOPE, WIDTH_M, Q_M3S, N = 1e-4, 200.0, 500.0, 0.03
SPACING_M = 500.0


def _channel(n_nodes=201, split_at=None):
    """Straight channel along x running to the sea at x=0: reaches a (upstream), b, c (mouth)."""
    x = np.arange(n_nodes)[::-1] * SPACING_M
    reach = np.where(x > 2 * x.max() / 3, "a", np.where(x > x.max() / 3, "b", "c"))
    q = np.full(n_nodes, Q_M3S)
    if split_at is not None:
        q[x <= split_at] = 0.6 * Q_M3S
    h_n = normal_depth(Q_M3S, WIDTH_M, N, SLOPE)
    bed = SLOPE * x - h_n
    return pd.DataFrame(
        {
            "node": np.arange(n_nodes),
            "reach_id": reach,
            "x": x,
            "y": np.zeros(n_nodes),
            "dist_to_sea_m": x,
            "width_m": WIDTH_M,
            "slope": SLOPE,
            "bed_m": bed,
            "level_m": bed + h_n,
            "fathom_level_m": bed + h_n,
            "informative": True,
            "bathymetry": False,
            "q_m3s": q,
        }
    ).sample(frac=1.0, random_state=0)  # the table is not in path order


DOWNSTREAM = {"a": ["b"], "b": ["c", "side"], "c": [], "side": []}


def test_mainstem_path_takes_largest_discharge_and_stops_at_network_end():
    assert mainstem_path(DOWNSTREAM, "a", {"a": 9, "b": 9, "c": 6, "side": 3}) == [
        "a",
        "b",
        "c",
    ]
    assert mainstem_path(DOWNSTREAM, "a", {"a": 9, "b": 9, "c": 3, "side": 6}) == [
        "a",
        "b",
        "side",
    ]


def test_mainstem_path_raises_on_cycle():
    with pytest.raises(ValueError, match="cycle"):
        mainstem_path({"a": ["b"], "b": ["a"]}, "a", {"a": 1, "b": 1})


def test_profile_runs_upstream_first_with_distance_from_mouth():
    prof = mainstem_profile(_channel(), DOWNSTREAM, "a")
    assert prof["reach_id"].iloc[0] == "a" and prof["reach_id"].iloc[-1] == "c"
    assert np.all(np.diff(prof["dist_from_mouth_m"]) < 0)
    np.testing.assert_allclose(prof["dist_from_mouth_m"], prof["dist_to_sea_m"])


def test_profile_rejects_reach_without_nodes():
    with pytest.raises(ValueError, match="no nodes"):
        mainstem_profile(_channel(), DOWNSTREAM, "missing")


def test_scaled_discharge_keeps_bifurcation_shares():
    prof = mainstem_profile(_channel(split_at=10_000.0), DOWNSTREAM, "a")
    q = scaled_discharge(prof, 100.0)
    assert q[0] == pytest.approx(100.0)
    assert q[-1] == pytest.approx(60.0)


def test_forward_levels_at_normal_depth_sea_level_reproduce_uniform_flow():
    prof = mainstem_profile(_channel(), DOWNSTREAM, "a")
    eta = forward_levels(
        prof, Q_M3S, sea_level_m=float(prof["level_m"].iloc[-1]), manning_n=N
    )
    np.testing.assert_allclose(eta, prof["level_m"], atol=0.01)


def test_surface_mismatch_zero_at_design_discharge_and_negative_below_it():
    prof = mainstem_profile(_channel(), DOWNSTREAM, "a")
    sea = float(prof["level_m"].iloc[-1])
    out = surface_mismatch(
        prof, {"Q_mean": 0.3 * Q_M3S, "Q_bf": Q_M3S}, sea, N
    ).set_index("discharge")
    assert out.loc["Q_bf", "max_abs_m"] < 0.01
    assert out.loc["Q_bf", "share_within_tol"] == 1.0
    assert out.loc["Q_mean", "at_forcing_point_m"] < -0.1


def test_raised_sea_level_perturbation_is_positive_and_decays_upstream():
    prof = mainstem_profile(_channel(), DOWNSTREAM, "a")
    calm = forward_levels(prof, Q_M3S, float(prof["level_m"].iloc[-1]), N)
    surge = forward_levels(prof, Q_M3S, float(prof["level_m"].iloc[-1]) + 1.0, N)
    d = (surge - calm)[::-1]  # mouth first
    assert d[0] == pytest.approx(1.0, abs=1e-6)
    assert np.all(d > -1e-9) and np.all(np.diff(d) < 1e-9)
    assert d[-1] < 0.01


def test_sample_sfincs_at_nodes_picks_nearest_active_cell(tmp_path):
    xc, yc = np.meshgrid(
        np.arange(0.0, 1000.0, 100.0) + 50.0, np.arange(-500.0, 500.0, 100.0) + 50.0
    )
    msk = np.ones(xc.shape, dtype=np.int8)
    msk[:, 0] = 0
    zs = np.stack([xc / 1000.0, xc / 1000.0 + 1.0])
    xr.Dataset(
        {
            "zs": (("time", "n", "m"), zs),
            "msk": (("n", "m"), msk),
            "x": (("n", "m"), xc),
            "y": (("n", "m"), yc),
        },
        coords={"time": [0.0, 3600.0]},
    ).to_netcdf(tmp_path / "sfincs_map.nc")
    prof = pd.DataFrame({"node": [7, 8], "x": [460.0, 10.0], "y": [0.0, 0.0]})
    out = sample_sfincs_at_nodes(tmp_path / "sfincs_map.nc", prof)
    assert out.dims == ("time", "node")
    np.testing.assert_allclose(
        out.isel(time=0).values, [0.45, 0.15]
    )  # x=10 is inactive col 0 -> col 1
    np.testing.assert_allclose(
        out["cell_distance_m"].values, [np.hypot(10.0, 50.0), np.hypot(140.0, 50.0)]
    )
