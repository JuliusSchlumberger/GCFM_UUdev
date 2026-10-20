"""Unit tests for round 0's steady-state helpers and the sfincs.inp editor."""

import numpy as np

from src.river_depth_calibration import (
    compute_end_of_run_zs,
    is_steady,
    steady_state_metrics,
)
from src.sfincs_run import parse_sfincs_inp, set_sfincs_inp_keys


def test_still_rising_run_is_not_steady():
    t = np.arange(25)[:, None]
    zs = np.hstack([0.1 * t, 0.2 * t])  # rising 0.1 / 0.2 m per hour
    m = steady_state_metrics(zs, window_steps=3)
    assert m["median"] > 0.2
    assert m["frac_rising_1cm"] == 1.0
    assert not is_steady(m, 0.01, 0.05)


def test_settled_run_is_steady_and_dry_points_are_ignored():
    zs = np.full((25, 3), 1.0)
    zs[:, 2] = np.nan  # never wetted
    zs[-1, 0] += 0.004
    m = steady_state_metrics(zs)
    assert m["median"] < 0.01 and m["p95"] < 0.05
    assert is_steady(m, 0.01, 0.05)


def test_too_short_run_gives_nan_and_not_steady():
    m = steady_state_metrics(np.zeros((2, 4)), window_steps=3)
    assert np.isnan(m["median"])
    assert not is_steady(m, 0.01, 0.05)


def test_end_of_run_level_is_mean_of_last_two_outputs_not_the_peak():
    zs = np.array([[0.0], [5.0], [2.0], [2.2]])  # transient peak of 5 m early on
    assert compute_end_of_run_zs(zs)[0] == np.float64(2.1)


def test_set_sfincs_inp_keys_replaces_adds_and_removes(tmp_path):
    inp = tmp_path / "sfincs.inp"
    inp.write_text(
        "tstart               = 20000101 000000\ninifile              = sfincs.ini\nmmax = 10\n"
    )
    set_sfincs_inp_keys(
        inp, {"tstart": "20000102 000000", "inifile": None, "rstfile": "a.rst"}
    )
    cfg = parse_sfincs_inp(inp)
    assert cfg["tstart"] == "20000102 000000"
    assert "inifile" not in cfg
    assert cfg["rstfile"] == "a.rst"
    assert cfg["mmax"] == "10"
