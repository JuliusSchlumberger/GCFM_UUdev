"""Workarounds for the pinned hydromt_sfincs subgrid table: float32 zlib write, reuse."""

from __future__ import annotations

import functools
import hashlib
import logging
import shutil
from importlib.metadata import version
from pathlib import Path

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)

PATCHED_VERSION = "2.0.0rc2"
# Lossless; SFINCS v2.3.0 reads it with identical results and about 40% less disk.
SUBGRID_ZLIB_LEVEL = 4


class _Float32ZerosNumpy:
    """numpy stand-in whose ``zeros`` defaults to float32; everything else is forwarded."""

    def __init__(self, numpy_module):
        self._np = numpy_module

    def zeros(self, shape, dtype=np.float32, **kwargs):
        return self._np.zeros(shape, dtype=dtype, **kwargs)

    def __getattr__(self, name):
        return getattr(self._np, name)


class _ZlibDataset(xr.Dataset):
    """Dataset whose ``to_netcdf`` zlib-compresses every data variable by default."""

    __slots__ = ()

    def to_netcdf(self, *args, **kwargs):
        kwargs.setdefault(
            "encoding",
            {
                v: {"zlib": True, "complevel": SUBGRID_ZLIB_LEVEL}
                for v in self.data_vars
            },
        )
        return super().to_netcdf(*args, **kwargs)


class _ZlibXarray:
    """xarray stand-in whose ``Dataset`` is a _ZlibDataset; everything else is forwarded."""

    def __init__(self, xarray_module):
        self._xr = xarray_module
        self.Dataset = _ZlibDataset

    def __getattr__(self, name):
        return getattr(self._xr, name)


def patch_subgrid_write() -> bool:
    """Make the subgrid netCDF writer use float32 buffers and zlib compression.

    hydromt_sfincs 2.0.0rc2 ``SfincsSubgridTable.write_netcdf`` copies float32
    subgrid arrays into ``np.zeros`` buffers without a dtype (float64), doubling
    peak memory; on ~1e6 active cells this raises an out-of-memory error. The
    written values are identical; ``z_level`` and ``uv_*`` are stored as float32
    instead of float64, and every variable zlib-compressed. Applied only to the
    pinned version, since a later release may change the method. Returns
    whether the fix is active.
    """
    from hydromt_sfincs.components.grid import subgrid as subgrid_module

    table = subgrid_module.SfincsSubgridTable
    if getattr(table.write_netcdf, "_float32_patch", False):
        return True
    installed = version("hydromt_sfincs")
    if installed != PATCHED_VERSION:
        log.warning(
            "hydromt_sfincs %s is not %s: float32 subgrid-write fix not applied; "
            "check whether the release fixes write_netcdf's float64 buffers",
            installed,
            PATCHED_VERSION,
        )
        return False

    upstream_write = table.write_netcdf

    @functools.wraps(upstream_write)
    def write_netcdf(self, filename=None):
        real_np, real_xr = subgrid_module.np, subgrid_module.xr
        subgrid_module.np = _Float32ZerosNumpy(real_np)
        subgrid_module.xr = _ZlibXarray(real_xr)
        try:
            return upstream_write(self, filename)
        finally:
            subgrid_module.np, subgrid_module.xr = real_np, real_xr

    write_netcdf._float32_patch = True
    table.write_netcdf = write_netcdf
    log.info(
        "hydromt_sfincs %s: subgrid netCDF written float32, zlib-compressed", installed
    )
    return True


class ReusableSubgrid:
    """Build ``sf``'s subgrid table, reusing the last build when no input changed.

    ``create`` compares its keyword arguments (by ``repr``), the grid mask and
    the size and mtime of ``input_paths`` with the previous call. When all match
    and ``sf.subgrid`` still holds that table, the build is skipped: the table
    stays in memory for the next ``sf.write`` and the ``subgrid/`` geotiffs are
    copied to the current model root, where ``create`` would rewrite them
    identically.
    """

    _TIFS = {"write_dep_tif": "dep_subgrid.tif", "write_man_tif": "manning_subgrid.tif"}

    def __init__(self, sf):
        self._sf = sf
        self._key = None
        self._table = None
        self._tif_dir: Path | None = None

    def _input_key(self, input_paths: list[Path], create_kwargs: dict) -> tuple:
        mask = self._sf.grid.mask
        mask_digest = hashlib.sha256(np.ascontiguousarray(mask.values).tobytes())
        mask_digest.update(repr((mask.shape, tuple(mask.raster.transform))).encode())
        files = []
        for path in input_paths:
            stat = Path(path).stat()
            files.append((str(path), stat.st_size, stat.st_mtime_ns))
        return (
            repr(sorted(create_kwargs.items())),
            mask_digest.hexdigest(),
            tuple(files),
        )

    def create(self, input_paths: list[Path], **create_kwargs) -> bool:
        """Build the subgrid table with ``create_kwargs`` unless it repeats the last build.

        ``input_paths`` are the files read by the catalog sources named in
        ``create_kwargs``. Returns whether the table was built.
        """
        key = self._input_key(input_paths, create_kwargs)
        tif_dir = Path(self._sf.root.path) / "subgrid"
        if key == self._key and self._sf.subgrid.data is self._table:
            for flag, name in self._TIFS.items():
                if create_kwargs.get(flag) and tif_dir != self._tif_dir:
                    tif_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(self._tif_dir / name, tif_dir / name)
            log.info(
                "Subgrid inputs unchanged: table and geotiffs reused from %s",
                self._tif_dir,
            )
            return False
        self._sf.subgrid.create(**create_kwargs)
        self._key, self._table, self._tif_dir = key, self._sf.subgrid.data, tif_dir
        return True
