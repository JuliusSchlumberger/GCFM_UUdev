"""Smoke tests for the failure modes of src.plots writers."""

import errno

import numpy as np
import pytest
import src.plots as plots
import xarray as xr
from affine import Affine


class _FakeFigure:
    def __init__(self, errors: list[OSError]):
        self.errors = list(errors)
        self.calls = 0

    def savefig(self, path, **_kwargs):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        path.write_bytes(b"png")


@pytest.fixture
def no_wait(monkeypatch):
    monkeypatch.setattr(plots, "_SAVE_RETRY_WAIT_S", 0.0)


def test_save_figure_retries_windows_file_lock(tmp_path, no_wait):
    fig = _FakeFigure([OSError(errno.EINVAL, "Invalid argument")])
    out = tmp_path / "fig.png"
    plots.save_figure(fig, out, close=False)
    assert fig.calls == 2
    assert out.exists()


def test_save_figure_reraises_non_lock_oserror_immediately(tmp_path, no_wait):
    fig = _FakeFigure([OSError(errno.ENOSPC, "No space left on device")])
    with pytest.raises(OSError) as excinfo:
        plots.save_figure(fig, tmp_path / "fig.png", close=False)
    assert excinfo.value.errno == errno.ENOSPC
    assert fig.calls == 1


def test_animate_flood_progression_without_time_dim_raises(tmp_path):
    da_h = xr.DataArray(np.zeros((2, 2)), dims=("y", "x"))
    out = tmp_path / "anim.mp4"
    with pytest.raises(ValueError, match="no 'time' dimension"):
        plots.animate_flood_progression(da_h, None, "", "", str(out))
    assert not out.exists()


def test_crest_gap_map_without_crest_cells_raises(tmp_path):
    out = tmp_path / "crest_gap.png"
    with pytest.raises(ValueError, match="no finite cell"):
        plots.plot_crest_gap_map(
            np.full((3, 3), np.nan),
            Affine.identity(),
            "",
            None,
            None,
            np.array([]),
            np.array([]),
            str(out),
        )
    assert not out.exists()


def _round_profile():
    import pandas as pd

    along = np.array([0.0, 500.0, 1000.0])
    return pd.DataFrame(
        {
            "along_path_m": along,
            "dem_m": [2.0, 1.0, 0.0],
            "river_depth_m": [1.5, 1.0, 0.0],
            "weir_crest_m": [3.0, 2.0, np.nan],
            "zs_m": [2.5, 1.5, 0.2],
        }
    )


def test_calibration_round_profiles_read_river_depth_m(tmp_path):
    subplots, combined = tmp_path / "sub.png", tmp_path / "comb.png"
    profiles = {i: _round_profile() for i in range(2)}
    plots.plot_calibration_round_profiles(
        "0", "1", profiles, 1, str(subplots), str(combined)
    )
    assert subplots.exists() and combined.exists()


def _subplot_legend_texts(monkeypatch, tmp_path, **kwargs) -> list[list[str]]:
    figs = []
    monkeypatch.setattr(plots, "save_figure", lambda fig, *_a, **_k: figs.append(fig))
    profiles = {i: _round_profile() for i in range(2)}
    plots.plot_calibration_round_profiles(
        "0",
        "1",
        profiles,
        1,
        str(tmp_path / "s.png"),
        str(tmp_path / "c.png"),
        **kwargs,
    )
    return [
        [t.get_text() for t in ax.get_legend().get_texts()] if ax.get_legend() else []
        for ax in figs[0].axes
    ]


def test_round_profile_legend_says_end_of_run_for_round0(monkeypatch, tmp_path):
    labels = ["water level, end of run", "water level, period max"]
    round0, round1 = _subplot_legend_texts(
        monkeypatch, tmp_path, water_level_labels=labels
    )
    assert "water level, end of run" in round0
    assert "water level, period max" in round1


def test_round_profile_legend_defaults_to_period_max(monkeypatch, tmp_path):
    round0, round1 = _subplot_legend_texts(monkeypatch, tmp_path)
    assert "water level, period max" in round0
    assert round1 == []
