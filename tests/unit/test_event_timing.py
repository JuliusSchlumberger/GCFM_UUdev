"""Smoke tests for the late-compartment helpers in analysis/check_event_timing.py."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))

from check_event_timing import _compartment_table, _label_compartments  # noqa: E402

CELL_AREA_M2 = 100.0 * 100.0


def _grid():
    """Channel in row 0, a filling 4x5 compartment next to it, a full one away from it, a 2-cell speck."""
    zb = np.zeros((10, 12))
    wet0 = np.zeros(zb.shape, bool)
    wet0[0, :] = True
    late = np.zeros(zb.shape, bool)
    late[1:5, 0:5] = True
    late[7:10, 8:12] = True
    late[8, 2:4] = True
    return zb, wet0, late


def _frames(zb, labels, t_hr):
    """Compartment 1 rises 0.1 m/h throughout; compartment 2 fills in the first 10 h, then holds."""
    for t in t_hr:
        zs = zb.copy()
        zs[labels == 1] = 0.1 * t
        zs[labels == 2] = 0.1 * min(t, 10.0)
        yield zs


def test_small_patches_are_dropped():
    _, _, late = _grid()
    labels, n = _label_compartments(late, min_cells=10)
    assert n == 2
    assert labels[8, 2] == 0


def test_filling_compartment_has_inflow_at_run_end():
    zb, wet0, late = _grid()
    labels, n = _label_compartments(late, min_cells=10)
    t_hr = np.arange(0.0, 25.0)
    comp = _compartment_table(
        labels, n, zb, wet0, _frames(zb, labels, t_hr), t_hr, 12.0, CELL_AREA_M2
    ).set_index("compartment")
    filling, full = comp.loc[labels[2, 2]], comp.loc[labels[8, 9]]
    assert filling["inflow_last6h_m3s"] > 0
    assert np.isclose(filling["mean_rise_last6h_cm_hr"], 10.0)
    assert 0 < filling["share_gained_after_surge"] < 1
    assert full["inflow_last6h_m3s"] == 0
    assert full["share_gained_after_surge"] == 0
    assert filling["borders_initial_water"] and not full["borders_initial_water"]
