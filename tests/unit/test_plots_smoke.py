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


def test_crest_profiles_one_panel_per_seed_with_dem_bed_and_crest(
    monkeypatch, tmp_path
):
    figs = []
    monkeypatch.setattr(plots, "save_figure", lambda fig, *_a, **_k: figs.append(fig))
    plots.plot_crest_profiles(
        "0", {"1": _round_profile(), "2": _round_profile()}, str(tmp_path / "p.png")
    )
    axes = figs[0].axes
    assert len(axes) == 2
    for ax in axes:
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["DEM", "river bed", "dike crest"]
        bed = ax.get_lines()[1].get_ydata()
        assert np.allclose(bed, [0.5, 0.0, 0.0])  # dem_m - river_depth_m


def test_crest_profiles_without_seed_still_writes(tmp_path):
    out = tmp_path / "p.png"
    plots.plot_crest_profiles("0", {}, str(out))
    assert out.exists()


def test_run_forcing_reads_bzs_and_dis_and_marks_tstart(monkeypatch, tmp_path):
    (tmp_path / "sfincs.inp").write_text(
        "tref = 20000101 000000\ntstart = 20000101 020000\n"
        "bzsfile = sfincs.bzs\ndisfile = sfincs.dis\n"
    )
    np.savetxt(
        tmp_path / "sfincs.bzs", [[0, 0.1, 0.2], [3600, 0.5, 0.6], [7200, 0.3, 0.4]]
    )
    np.savetxt(tmp_path / "sfincs.dis", [[0, 10.0], [7200, 50.0]])
    figs = []
    monkeypatch.setattr(plots, "save_figure", lambda fig, *_a, **_k: figs.append(fig))
    plots.plot_run_forcing(tmp_path, tmp_path / "f.png", "0", "default")
    ax_wl, ax_q = figs[0].axes
    assert len(ax_wl.get_lines()) == 2 + 1  # two boundary points + tstart line
    assert len(ax_q.get_lines()) == 1 + 1
    assert "default" in figs[0]._suptitle.get_text()
