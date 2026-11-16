"""Tests for the observation-point placement of src.observation_points."""

import numpy as np
import pandas as pd
import pytest
from rasterio.transform import Affine
from src.observation_points import downstream_reaches, river_points, sea_points


def test_downstream_reaches_follow_every_bifurcation_arm():
    adjacency = {"1": ["2"], "2": ["3", "4"], "3": ["5"], "4": ["5"], "9": ["5"]}
    assert downstream_reaches(adjacency, ["1"], 10) == {"1", "2", "3", "4", "5"}


def test_river_points_are_one_node_per_spacing_on_each_reach():
    nodes = pd.DataFrame(
        {
            "reach_id": ["1"] * 9 + ["7"] * 3,
            "x": np.arange(12.0),
            "y": np.zeros(12),
            "dist_to_sea_m": list(np.arange(9) * 500.0) + [0.0, 500.0, 1000.0],
        }
    )
    pts = river_points(nodes, {"1"}, 1000.0)
    assert list(pts["dist_from_mouth_m"]) == [0.0, 1000.0, 2000.0, 3000.0, 4000.0]
    assert set(pts["kind"]) == {"river"}


def test_sea_points_run_from_the_mouth_to_the_boundary():
    bed = np.full((3, 30), -5.0)
    mask = np.ones(bed.shape, dtype=int)
    mask[:, -1] = 2  # water-level boundary on the right
    tr = Affine(100.0, 0.0, 0.0, 0.0, -100.0, 300.0)
    pts = sea_points(bed, mask, tr, 0.0, [("1", 50.0, 150.0)], 1000.0)
    assert list(pts["dist_from_mouth_m"]) == pytest.approx([1000.0, 2000.0, 2900.0])
    assert pts["x"].iloc[-1] == pytest.approx(2950.0)  # last point on the boundary cell
    assert set(pts["kind"]) == {"sea"}
