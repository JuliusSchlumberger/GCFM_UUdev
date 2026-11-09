"""Tests for the alpha check helpers of src.sfincs_run."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString
from src.sfincs_run import level_reversal_m, river_cells_on_map, selected_alpha


def test_steady_rise_is_not_a_reversal():
    zs = np.cumsum(np.full((50, 3), 1.5), axis=0)
    assert np.all(level_reversal_m(zs) == 0.0)


def test_up_down_swing_counts_with_the_smaller_step():
    zs = np.array([[0.0], [3.0], [1.0], [1.0]])
    assert level_reversal_m(zs)[0] == pytest.approx(2.0)


def test_level_turning_non_finite_is_infinite_reversal():
    zs = np.array([[0.0, np.nan], [1.0, np.nan], [np.nan, np.nan]])
    rev = level_reversal_m(zs)
    assert np.isinf(rev[0]) and rev[1] == 0.0


def test_river_cells_follow_the_line_only():
    x, y = np.meshgrid(np.arange(10) * 100.0 + 50.0, np.arange(5) * 100.0 + 50.0)
    rivers = gpd.GeoDataFrame(geometry=[LineString([(0.0, 250.0), (1000.0, 250.0)])])
    n, m = river_cells_on_map(x, y, rivers, 50.0)
    assert set(n.tolist()) == {2}
    assert sorted(m.tolist()) == list(range(10))


def test_selected_alpha_reads_the_stable_row(tmp_path):
    path = tmp_path / "alpha_check.csv"
    pd.DataFrame({"alpha": [0.5, 0.35], "selected": [False, True]}).to_csv(
        path, index=False
    )
    assert selected_alpha(path) == 0.35


def test_selected_alpha_fails_without_a_stable_row(tmp_path):
    path = tmp_path / "alpha_check.csv"
    pd.DataFrame({"alpha": [0.5], "selected": [False]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        selected_alpha(path)
