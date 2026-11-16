"""Smoke tests for the analysis suite scripts on synthetic inputs."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "analysis"))
sys.path.insert(0, str(REPO_ROOT / "workflow"))

import depth_sensitivity  # noqa: E402
import gvf_setup  # noqa: E402
import travel_time  # noqa: E402


def test_half_rise_of_a_delayed_ramp_is_its_midpoint():
    t = np.arange(0.0, 100.0, 1.0)
    q = np.clip((t - 20.0) / 40.0, 0.0, 1.0) * 1000.0 + 100.0
    assert travel_time.half_rise_hr(t, q) == 40.0


def test_half_rise_skips_the_filter_edges():
    t = np.arange(0.0, 50.0, 1.0)
    q = np.r_[np.full(5, np.nan), np.linspace(0.0, 10.0, 45)]
    assert 25.0 <= travel_time.half_rise_hr(t, q) <= 30.0


def test_froude_classes_put_the_thresholds_on_bounds():
    _cmap, norm, bounds = gvf_setup._fr_cmap(0.3, 0.5)
    assert 0.3 in bounds and 0.5 in bounds
    assert np.all(np.diff(bounds) > 0)
    assert norm(0.4) == norm(0.45) != norm(0.55)


def test_network_lines_give_one_segment_per_node_pair():
    t = pd.DataFrame(
        {
            "reach_id": ["a"] * 3 + ["b"] * 2,
            "dist_to_sea_m": [2.0, 1.0, 0.0, 1.0, 0.0],
            "x": [0.0, 1.0, 2.0, 5.0, 6.0],
            "y": np.zeros(5),
            "fr": [0.1, 0.2, 0.3, 0.4, 0.6],
        }
    )
    segs, vals = gvf_setup._network_lines(t, "fr")
    assert len(segs) == 3
    assert np.allclose(vals, [0.15, 0.25, 0.5])


def test_concept_figure_renders(tmp_path):
    path = tmp_path / "concept.png"
    gvf_setup.fig_concept(path)
    assert path.stat().st_size > 10_000
    assert "5. Flood model" in path.with_suffix(".txt").read_text(encoding="utf-8")


def test_depth_figure_renders_from_a_summary(tmp_path):
    rows = []
    for basin, scen in (("1", "coast"), ("1", "river"), ("2", "coast")):
        for scale, f in ((1.0, 1.0), (0.7, 0.9), (1.3, 1.2)):
            row = {"basin_id": basin, "scenario": scen, "depth_scale": scale}
            for key in depth_sensitivity.METRICS:
                row[key] = 10.0 * f
                row[f"{key}_rel"] = f
            rows.append(row)
    path = tmp_path / "depth.png"
    depth_sensitivity.figure(pd.DataFrame(rows), path)
    assert path.stat().st_size > 10_000
