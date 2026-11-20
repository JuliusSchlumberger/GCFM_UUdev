"""Tests for the alpha check helpers of src.sfincs_run."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString
from src.sfincs_run import (
    UNSTABLE_MAP_NAME,
    level_reversal_m,
    river_cells_on_map,
    stop_failed_run,
)


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


def test_failed_run_keeps_its_map_and_names_the_remedy(tmp_path):
    (tmp_path / "sfincs_map.nc").write_bytes(b"map")
    with pytest.raises(RuntimeError, match="UNSTABLE.*Map kept.*alpha_overrides"):
        stop_failed_run(tmp_path, "UNSTABLE: swing.", "set sfincs.alpha_overrides")
    assert not (tmp_path / "sfincs_map.nc").exists()
    assert (tmp_path / UNSTABLE_MAP_NAME).read_bytes() == b"map"
