"""Tests for the forcing smoothness check of src.sfincs_run."""

import numpy as np
import pytest
from src.sfincs_run import check_forcing_smoothness, discharge_kinks, water_level_jumps

HOURS = np.arange(0.0, 48.0)
CHECKS = {
    "discharge_kink_ratio_max": 0.6,
    "water_level_jump_ratio_max": 3.0,
    "water_level_jump_min_m": 0.05,
}


def _cosine_ramp(start, end, ramp_hr):
    s = np.clip(HOURS / ramp_hr, 0.0, 1.0)
    return start + (end - start) * 0.5 * (1.0 - np.cos(np.pi * s))


def test_half_cosine_discharge_ramp_and_constant_flow_pass():
    q = np.column_stack([_cosine_ramp(100.0, 900.0, 6.0), np.full(HOURS.size, 50.0)])
    assert discharge_kinks(HOURS, q, 0.6) == []


@pytest.mark.parametrize(
    "q",
    [
        np.where(HOURS >= 10, 900.0, 100.0),
        100.0 + 800.0 * np.clip(HOURS / 6.0, 0.0, 1.0),
    ],
    ids=["step", "straight_ramp_corner"],
)
def test_discharge_step_or_ramp_corner_is_a_kink(q):
    lines = discharge_kinks(HOURS, q[:, None], 0.6)
    assert len(lines) == 1 and "kink ratio 1.00" in lines[0]


def test_tide_passes_and_a_step_on_a_calm_sea_is_a_jump():
    tide = 1.0 * np.sin(2 * np.pi * HOURS / 12.42)
    calm = 0.02 * np.sin(2 * np.pi * HOURS / 12.42) + 0.3 * (HOURS >= 20)
    lines = water_level_jumps(HOURS, np.column_stack([tide, calm]), 3.0, 0.05)
    assert len(lines) == 1 and lines[0].startswith("water-level point 2")
    assert "from 19 to 20 h" in lines[0]


def test_small_steps_are_never_jumps():
    zs = 0.04 * (HOURS >= 20)
    assert water_level_jumps(HOURS, zs[:, None], 3.0, 0.05) == []


def _write_run(tmp_path, q, zs):
    t_s = HOURS * 3600.0
    np.savetxt(tmp_path / "sfincs.dis", np.column_stack([t_s, q]))
    np.savetxt(tmp_path / "sfincs.bzs", np.column_stack([t_s, zs]))
    (tmp_path / "sfincs.inp").write_text("disfile = sfincs.dis\nbzsfile = sfincs.bzs\n")


def test_check_reads_the_runs_forcing_files_and_warns_on_a_step(tmp_path, caplog):
    _write_run(tmp_path, _cosine_ramp(100.0, 900.0, 6.0), _cosine_ramp(0.1, 0.4, 6.0))
    assert check_forcing_smoothness(tmp_path, CHECKS) == []
    _write_run(tmp_path, np.where(HOURS >= 10, 900.0, 100.0), np.full(HOURS.size, 0.1))
    problems = check_forcing_smoothness(tmp_path, CHECKS)
    assert len(problems) == 1 and problems[0].startswith(
        "discharge point 1: kink ratio"
    )
    assert "not smooth" in caplog.text


def test_two_point_forcing_is_checked_as_sfincs_interpolates_it(tmp_path):
    np.savetxt(
        tmp_path / "sfincs.bzs", np.array([[0.0, 0.1], [3600.0, 0.4], [86400.0, 0.4]])
    )
    (tmp_path / "sfincs.inp").write_text("bzsfile = sfincs.bzs\n")
    (problem,) = check_forcing_smoothness(tmp_path, CHECKS)
    assert "step +0.300 m from 0 to 1 h" in problem
