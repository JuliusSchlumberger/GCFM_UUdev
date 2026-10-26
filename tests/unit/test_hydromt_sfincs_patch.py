"""Unit tests for the subgrid workarounds in src.hydromt_sfincs_patch."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from src import hydromt_sfincs_patch
from src.hydromt_sfincs_patch import (
    ReusableSubgrid,
    _Float32ZerosNumpy,
    patch_subgrid_write,
)

subgrid_module = pytest.importorskip("hydromt_sfincs.components.grid.subgrid")


def test_zeros_defaults_to_float32_and_respects_explicit_dtype():
    shim = _Float32ZerosNumpy(np)
    assert shim.zeros((2, 3)).dtype == np.float32
    assert shim.zeros(4, dtype=np.int64).dtype == np.int64
    assert shim.where is np.where


def test_write_netcdf_sees_float32_zeros_and_restores_numpy(monkeypatch):
    table = subgrid_module.SfincsSubgridTable
    seen = {}

    def fake_upstream(self, filename=None):
        seen["dtype"] = subgrid_module.np.zeros(3).dtype
        raise RuntimeError("upstream failure")

    monkeypatch.setattr(table, "write_netcdf", fake_upstream)
    monkeypatch.setattr(
        hydromt_sfincs_patch, "version", lambda _: hydromt_sfincs_patch.PATCHED_VERSION
    )
    assert patch_subgrid_write()
    with pytest.raises(RuntimeError):
        table.write_netcdf(object())
    assert seen["dtype"] == np.float32
    assert subgrid_module.np is np


def test_write_netcdf_compresses_losslessly_and_restores_xarray(monkeypatch, tmp_path):
    table = subgrid_module.SfincsSubgridTable
    values = np.linspace(0.0, 1.0, 50, dtype=np.float32)

    def fake_upstream(self, filename=None):
        ds = subgrid_module.xr.Dataset()
        ds["z_zmin"] = (("np",), values)
        ds.to_netcdf(filename)

    monkeypatch.setattr(table, "write_netcdf", fake_upstream)
    monkeypatch.setattr(
        hydromt_sfincs_patch, "version", lambda _: hydromt_sfincs_patch.PATCHED_VERSION
    )
    assert patch_subgrid_write()
    path = tmp_path / "sfincs_subgrid.nc"
    table.write_netcdf(object(), path)
    with xr.open_dataset(path) as back:
        assert back["z_zmin"].encoding["zlib"]
        np.testing.assert_array_equal(back["z_zmin"].values, values)
    assert subgrid_module.xr is xr


def test_patch_is_idempotent(monkeypatch):
    table = subgrid_module.SfincsSubgridTable
    monkeypatch.setattr(table, "write_netcdf", lambda self, filename=None: None)
    monkeypatch.setattr(
        hydromt_sfincs_patch, "version", lambda _: hydromt_sfincs_patch.PATCHED_VERSION
    )
    patch_subgrid_write()
    first = table.write_netcdf
    patch_subgrid_write()
    assert table.write_netcdf is first


def test_other_versions_are_left_unpatched(monkeypatch):
    table = subgrid_module.SfincsSubgridTable
    upstream = lambda self, filename=None: None  # noqa: E731
    monkeypatch.setattr(table, "write_netcdf", upstream)
    monkeypatch.setattr(hydromt_sfincs_patch, "version", lambda _: "9.9.9")
    assert not patch_subgrid_write()
    assert table.write_netcdf is upstream


class _StubSubgrid:
    def __init__(self, model):
        self.model = model
        self.data = None
        self.n_builds = 0

    def create(self, **kwargs):
        self.n_builds += 1
        self.data = object()
        out = Path(self.model.root.path) / "subgrid"
        out.mkdir(parents=True, exist_ok=True)
        (out / "dep_subgrid.tif").write_bytes(b"dep%d" % self.n_builds)
        (out / "manning_subgrid.tif").write_bytes(b"man%d" % self.n_builds)


class _StubModel:
    def __init__(self, root):
        mask = np.ones((3, 4), dtype=np.uint8)
        self.grid = SimpleNamespace(
            mask=SimpleNamespace(
                values=mask,
                shape=mask.shape,
                raster=SimpleNamespace(transform=(10.0, 0.0, 0.0, 0.0, 10.0, 0.0)),
            )
        )
        self.root = SimpleNamespace(path=str(root))
        self.subgrid = _StubSubgrid(self)


_KWARGS = dict(
    elevation_list=[{"elevation": "dem"}],
    nr_levels=5,
    write_dep_tif=True,
    write_man_tif=True,
)


def _setup(tmp_path):
    source = tmp_path / "dem.tif"
    source.write_bytes(b"elevation")
    sf = _StubModel(tmp_path / "round0")
    return sf, ReusableSubgrid(sf), source


def test_unchanged_inputs_reuse_table_and_copy_tifs_to_new_root(tmp_path):
    sf, builder, source = _setup(tmp_path)
    assert builder.create([source], **_KWARGS)
    table = sf.subgrid.data
    sf.root.path = str(tmp_path / "round1")
    assert not builder.create([source], **_KWARGS)
    assert sf.subgrid.n_builds == 1
    assert sf.subgrid.data is table
    for name in ("dep_subgrid.tif", "manning_subgrid.tif"):
        assert (tmp_path / "round1" / "subgrid" / name).read_bytes() == (
            tmp_path / "round0" / "subgrid" / name
        ).read_bytes()


def test_changed_input_file_kwargs_or_mask_rebuild(tmp_path):
    sf, builder, source = _setup(tmp_path)
    builder.create([source], **_KWARGS)
    source.write_bytes(b"burned elevation")
    assert builder.create([source], **_KWARGS)
    assert builder.create([source], **{**_KWARGS, "nr_levels": 6})
    sf.grid.mask.values[0, 0] = 0
    assert builder.create([source], **{**_KWARGS, "nr_levels": 6})
    assert sf.subgrid.n_builds == 4


def test_table_replaced_outside_the_builder_is_rebuilt(tmp_path):
    sf, builder, source = _setup(tmp_path)
    builder.create([source], **_KWARGS)
    sf.subgrid.data = object()
    assert builder.create([source], **_KWARGS)
