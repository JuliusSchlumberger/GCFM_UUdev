"""Tests for the run stability marking: swing check on a map, notes, metrics columns, stamps."""

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from shapely.geometry import LineString  # noqa: E402
from src.plots import stamp_unstable  # noqa: E402
from src.sfincs_run import (  # noqa: E402
    river_network_stability,
    stability_metrics,
    stability_note,
)


def _run_dir(tmp_path, swing_m):
    """Write a 5 x 10 hourly sfincs_map.nc whose river row (n = 2) swings by swing_m."""
    x, y = np.meshgrid(np.arange(10) * 100.0 + 50.0, np.arange(5) * 100.0 + 50.0)
    zs = np.zeros((24, 5, 10))
    zs[:, 2, 4] = np.where(np.arange(24) % 2, swing_m, 0.0)
    ds = xr.Dataset(
        {"zs": (("time", "n", "m"), zs), "x": (("n", "m"), x), "y": (("n", "m"), y)},
        coords={"time": pd.date_range("2000-01-01", periods=24, freq="h")},
    )
    ds.to_netcdf(tmp_path / "sfincs_map.nc")
    (tmp_path / "sfincs.inp").write_text("epsg = 32631\nalpha = 0.5\n")
    rivers = gpd.GeoDataFrame(
        geometry=[LineString([(0.0, 250.0), (1000.0, 250.0)])], crs="EPSG:32631"
    )
    return tmp_path, rivers


def test_swinging_river_cell_marks_the_run_unstable(tmp_path):
    run, rivers = _run_dir(tmp_path, 3.0)
    s = river_network_stability(run, rivers, 1.0)
    assert s["unstable"] and s["cells_over"] == 1 and s["alpha"] == 0.5
    assert s["max_swing_m"] == pytest.approx(3.0)
    assert (s["x"], s["y"]) == (450.0, 250.0)


def test_quiet_river_is_stable_and_has_no_note(tmp_path):
    run, rivers = _run_dir(tmp_path, 0.1)
    s = river_network_stability(run, rivers, 1.0)
    assert not s["unstable"]
    assert stability_note(s | {"spinup_unstable": False}) is None


def test_unstable_spinup_taints_the_event_metrics(tmp_path):
    path = tmp_path / "stability.csv"
    pd.DataFrame(
        [
            {
                "alpha": 0.25,
                "max_swing_m": 0.1,
                "cells_over": 0,
                "cells": 9,
                "x": 0,
                "y": 0,
                "unstable": False,
                "spinup_unstable": True,
            }
        ]
    ).to_csv(path, index=False)
    m = stability_metrics(path)
    assert m["unstable"] and m["spinup_unstable"]
    assert "SPIN-UP" in stability_note(pd.read_csv(path).iloc[0].to_dict())


def test_stamp_keeps_the_figure_size(tmp_path):
    png = tmp_path / "fig.png"
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.plot([0, 1], [0, 1])
    fig.savefig(png, dpi=100)
    plt.close(fig)
    before = plt.imread(str(png)).shape[:2]
    stamp_unstable(png, "UNSTABLE: test")
    assert plt.imread(str(png)).shape[:2] == before
