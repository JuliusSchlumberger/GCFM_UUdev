"""Read and patch the water level in a SFINCS binary restart (.rst) file.

The format is undocumented; this layout was reverse-engineered and validated on
basin 2433835 (byte-exact round-trip; zs matches mod.grid.mask in sfincs.ind
order on all 263,907 cells, zs-zb >= 0 everywhere, ~46% of cells dry).

Fortran unformatted sequential records, each `[int32 length][data][int32 length]`
(both length markers must match):
    0: 1 int32   (unidentified header/flag)
    1: N float32 (zs, water surface elevation, one value per ACTIVE cell,
                  in sfincs.ind's active-cell order)
    2: ~2N float32 (unidentified, likely flux/velocity state)
    3: small float32 array (unidentified)
Only record 1 (zs) is modified; every other record is copied byte-for-byte.

sfincs.ind: a header int32 (active-cell count) followed by that many int32
1-based FORTRAN (column-major) flat indices into the (nmax, mmax) grid:
`np.unravel_index(ind - 1, (nmax, mmax), order="F")` gives each active cell's
(row, col), in zs order. mod.grid.mask has the same (nmax, mmax) grid.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def _read_records(path: Path) -> list[bytes]:
    """Parse a Fortran unformatted sequential file into its raw data records."""
    with open(path, "rb") as f:
        data = f.read()

    records: list[bytes] = []
    pos = 0
    while pos < len(data):
        if pos + 4 > len(data):
            raise ValueError(f"{path}: truncated record header at byte {pos}")
        (rec_len,) = struct.unpack_from("<i", data, pos)
        start = pos + 4
        end = start + rec_len
        if end + 4 > len(data):
            raise ValueError(
                f"{path}: record at byte {pos} claims length {rec_len}, exceeds file size"
            )
        (rec_len2,) = struct.unpack_from("<i", data, end)
        if rec_len2 != rec_len:
            raise ValueError(
                f"{path}: record at byte {pos} has mismatched length markers "
                f"({rec_len} != {rec_len2}) -- not a valid Fortran unformatted file, "
                f"or this format assumption is wrong for this file"
            )
        records.append(data[start:end])
        pos = end + 4
    return records


def _write_records(path: Path, records: list[bytes]) -> None:
    out = bytearray()
    for rec in records:
        n = len(rec)
        out += struct.pack("<i", n) + rec + struct.pack("<i", n)
    with open(path, "wb") as f:
        f.write(bytes(out))


def read_ind(ind_path: Path) -> np.ndarray:
    """0-based (row, col) for each active cell, in sfincs.ind's own order.

    Returns:
        (rows, cols): two (n_active,) int arrays.
    """
    ind_raw = np.fromfile(ind_path, dtype="<i4")
    n_active = int(ind_raw[0])
    if len(ind_raw) - 1 != n_active:
        raise ValueError(
            f"{ind_path}: header count {n_active} does not match "
            f"{len(ind_raw) - 1} index entries actually present"
        )
    return ind_raw[1:] - 1  # 1-based Fortran -> 0-based flat index


def patch_restart_zs(
    src_rst_path: Path,
    dst_rst_path: Path,
    ind_path: Path,
    grid_shape: tuple[int, int],
    active_cell_mask: np.ndarray,
    dz: np.ndarray | float,
) -> int:
    """Write a copy of `src_rst_path` with zs shifted by `dz` where the mask is True.

    Everything else is byte-identical; `src_rst_path` is never modified in place.
    `ind_path` must belong to the same basin/skeleton grid as `src_rst_path`;
    `grid_shape` is (nmax, mmax) of the coarse SFINCS grid (= mod.grid.mask.shape).
    `active_cell_mask` is an (nmax, mmax) bool array, True where zs is shifted.
    `dz` is a scalar or (nmax, mmax) array in metres added to zs (negative to
    follow a DEM lowering). Returns the number of active cells patched.
    """
    ind0 = read_ind(ind_path)  # 0-based flat index, Fortran order, per active cell
    nmax, mmax = grid_shape
    if active_cell_mask.shape != grid_shape:
        raise ValueError(
            f"active_cell_mask shape {active_cell_mask.shape} != grid_shape {grid_shape}"
        )
    rows, cols = np.unravel_index(ind0, (nmax, mmax), order="F")

    records = _read_records(src_rst_path)
    if len(records) < 2:
        raise ValueError(f"{src_rst_path}: expected >=2 records, found {len(records)}")

    n_active = ind0.size
    zs = np.frombuffer(records[1], dtype="<f4", count=n_active).copy()
    if zs.size * 4 != len(records[1]):
        raise ValueError(
            f"{src_rst_path}: record[1] size ({len(records[1])} bytes) does not match "
            f"{n_active} active cells (sfincs.ind) at float32 -- refusing to patch a "
            f"file that doesn't match this basin's own grid"
        )

    select = active_cell_mask[rows, cols]  # (n_active,) bool, per zs entry
    n_patched = int(select.sum())
    if n_patched > 0:
        dz_per_cell = dz if np.isscalar(dz) else np.asarray(dz)[rows, cols]
        zs = zs.astype(np.float64)
        zs[select] = zs[select] + (dz_per_cell[select] if not np.isscalar(dz) else dz)
        zs = zs.astype(np.float32)

    records[1] = zs.tobytes()
    dst_rst_path = Path(dst_rst_path)
    dst_rst_path.parent.mkdir(parents=True, exist_ok=True)
    _write_records(dst_rst_path, records)
    return n_patched
