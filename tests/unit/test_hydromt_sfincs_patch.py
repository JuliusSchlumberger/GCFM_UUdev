"""Unit tests for the float32 subgrid-write patch in src.hydromt_sfincs_patch."""

import numpy as np
import pytest
from src import hydromt_sfincs_patch
from src.hydromt_sfincs_patch import _Float32ZerosNumpy, use_float32_subgrid_write

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
    assert use_float32_subgrid_write()
    with pytest.raises(RuntimeError):
        table.write_netcdf(object())
    assert seen["dtype"] == np.float32
    assert subgrid_module.np is np


def test_patch_is_idempotent(monkeypatch):
    table = subgrid_module.SfincsSubgridTable
    monkeypatch.setattr(table, "write_netcdf", lambda self, filename=None: None)
    monkeypatch.setattr(
        hydromt_sfincs_patch, "version", lambda _: hydromt_sfincs_patch.PATCHED_VERSION
    )
    use_float32_subgrid_write()
    first = table.write_netcdf
    use_float32_subgrid_write()
    assert table.write_netcdf is first


def test_other_versions_are_left_unpatched(monkeypatch):
    table = subgrid_module.SfincsSubgridTable
    upstream = lambda self, filename=None: None  # noqa: E731
    monkeypatch.setattr(table, "write_netcdf", upstream)
    monkeypatch.setattr(hydromt_sfincs_patch, "version", lambda _: "9.9.9")
    assert not use_float32_subgrid_write()
    assert table.write_netcdf is upstream
