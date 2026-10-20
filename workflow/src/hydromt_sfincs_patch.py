"""Runtime fix for a memory bug in the pinned hydromt_sfincs; call before writing a model."""

from __future__ import annotations

import functools
import logging
from importlib.metadata import version

import numpy as np

log = logging.getLogger(__name__)

PATCHED_VERSION = "2.0.0rc2"


class _Float32ZerosNumpy:
    """numpy stand-in whose ``zeros`` defaults to float32; everything else is forwarded."""

    def __init__(self, numpy_module):
        self._np = numpy_module

    def zeros(self, shape, dtype=np.float32, **kwargs):
        return self._np.zeros(shape, dtype=dtype, **kwargs)

    def __getattr__(self, name):
        return getattr(self._np, name)


def use_float32_subgrid_write() -> bool:
    """Make the subgrid netCDF writer allocate float32 buffers instead of float64.

    hydromt_sfincs 2.0.0rc2 ``SfincsSubgridTable.write_netcdf`` copies float32
    subgrid arrays into ``np.zeros`` buffers without a dtype (float64), doubling
    peak memory; on ~1e6 active cells this raises an out-of-memory error. The
    written values are identical; ``z_level`` and ``uv_*`` are stored as float32
    instead of float64. Applied only to the pinned version, since a later
    release may change the method. Returns whether the fix is active.
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
        real_np = subgrid_module.np
        subgrid_module.np = _Float32ZerosNumpy(real_np)
        try:
            return upstream_write(self, filename)
        finally:
            subgrid_module.np = real_np

    write_netcdf._float32_patch = True
    table.write_netcdf = write_netcdf
    log.info(
        "hydromt_sfincs %s: subgrid netCDF written with float32 buffers", installed
    )
    return True
