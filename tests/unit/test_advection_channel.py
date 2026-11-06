"""Tests for the synthetic trench of analysis/advection_channel.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "workflow"))

import advection_channel as ac  # noqa: E402

DX_M = 100.0
LENGTH_M = 2000.0
NR_SUBGRID_PIXELS = 10


def test_normal_depth_satisfies_manning():
    q, w, n, s = 3000.0, 100.0, 0.03, 1e-4
    h = ac.normal_depth_m(q, w, n, s)
    assert w * h ** (5 / 3) * np.sqrt(s) / n == pytest.approx(q, rel=1e-9)


@pytest.mark.parametrize("layout", ac.LAYOUTS)
def test_centreline_path_length_covers_requested_length(layout):
    _, s = ac.centreline(layout, LENGTH_M)
    assert s[0] == 0.0
    assert s[-1] == pytest.approx(LENGTH_M, abs=2 * ac.CENTRELINE_STEP_M)
    assert np.all(np.diff(s) > 0)


def test_aligned_trench_fills_exactly_one_cell_row():
    r = ac.channel_rasters("aligned", DX_M, NR_SUBGRID_PIXELS, LENGTH_M)
    rows = np.unique(np.where(r["trench"])[0])
    assert len(rows) == ac.TRENCH_WIDTH_M / r["px_m"]
    assert len(np.unique(rows // NR_SUBGRID_PIXELS)) == 1


@pytest.mark.parametrize("layout", ac.LAYOUTS)
def test_trench_bed_falls_at_bed_slope_and_walls_stand_above(layout):
    r = ac.channel_rasters(layout, DX_M, NR_SUBGRID_PIXELS, LENGTH_M)
    z = r["z"]
    bed, wall = z[r["trench"]], z[~r["trench"]]
    span = ac.BED_SLOPE * (r["s"][-1] - r["s"][0])
    assert np.ptp(bed) == pytest.approx(span, rel=0.05)
    assert wall.min() > bed.min() + ac.WALL_HEIGHT_M - span


def _steady_errors(root, layout, advection):
    from _basin_inputs import load_config
    from _common import site_path
    from src.sfincs_run import run_sfincs_subprocess

    r = ac.build_case(
        root, layout, 3000.0, advection, min(ac.ALPHAS), load_config(), 8000.0
    )
    run_sfincs_subprocess(site_path("sfincs_exe"), root, 1800, ac.log, n_threads=4)
    err = ac.level_errors(root, r)
    assert (err["zs_end_m"] - err["zs_prev_m"]).abs().max() < 0.05
    return err["error_m"]


@pytest.mark.needs_sfincs
def test_advection_raises_meander_level_but_not_aligned(tmp_path):
    aligned = {
        a: _steady_errors(tmp_path / f"aligned{a}", "aligned", a) for a in (0, 1)
    }
    meander = {
        a: _steady_errors(tmp_path / f"meander{a}", "meander", a) for a in (0, 1)
    }
    assert abs(aligned[1].median() - aligned[0].median()) < 0.05
    assert meander[1].median() - meander[0].median() > 2.0
    assert meander[0].median() < 2.0
