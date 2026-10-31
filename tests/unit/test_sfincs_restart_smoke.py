"""Smoke tests for the Fortran restart-file patching in src.sfincs_restart."""

import struct

import numpy as np
import pytest
from src.sfincs_restart import patch_restart_zs, read_ind

SHAPE = (4, 5)


def _record(payload: bytes) -> bytes:
    return struct.pack("<i", len(payload)) + payload + struct.pack("<i", len(payload))


def _fixture(tmp_path):
    active = np.ones(SHAPE, dtype=bool)
    active[0, 0] = False
    flat = np.flatnonzero(active.ravel(order="F")) + 1  # 1-based Fortran order
    ind_path = tmp_path / "sfincs.ind"
    np.concatenate([[flat.size], flat]).astype("<i4").tofile(ind_path)

    zs = np.linspace(0.0, 1.9, flat.size).astype("<f4")
    records = [
        np.array([7], dtype="<i4").tobytes(),
        zs.tobytes(),
        np.arange(2 * flat.size, dtype="<f4").tobytes(),
        np.array([1.5, 2.5], dtype="<f4").tobytes(),
    ]
    rst_path = tmp_path / "sfincs.rst"
    rst_path.write_bytes(b"".join(_record(r) for r in records))
    return ind_path, rst_path, flat, zs, records


def test_read_ind_returns_zero_based_flat_indices(tmp_path):
    ind_path, _, flat, _, _ = _fixture(tmp_path)
    ind0 = read_ind(ind_path)
    np.testing.assert_array_equal(ind0, flat - 1)
    assert ind0.min() >= 0


def test_patch_shifts_only_selected_zs_and_copies_other_records(tmp_path):
    ind_path, rst_path, flat, zs, records = _fixture(tmp_path)
    select = np.zeros(SHAPE, dtype=bool)
    select[2:, 3:] = True
    dst = tmp_path / "out" / "patched.rst"
    n = patch_restart_zs(rst_path, dst, ind_path, SHAPE, select, dz=-0.5)
    assert n == 4

    data = dst.read_bytes()
    pos, out_records = 0, []
    while pos < len(data):
        (length,) = struct.unpack_from("<i", data, pos)
        out_records.append(data[pos + 4 : pos + 4 + length])
        pos += length + 8
    assert len(out_records) == len(records)
    for i in (0, 2, 3):
        assert out_records[i] == records[i]

    rows, cols = np.unravel_index(flat - 1, SHAPE, order="F")
    hit = select[rows, cols]
    new_zs = np.frombuffer(out_records[1], dtype="<f4")
    np.testing.assert_array_equal(new_zs[~hit], zs[~hit])
    np.testing.assert_allclose(new_zs[hit], zs[hit] - 0.5, atol=1e-6)


def test_empty_selection_round_trips_byte_identical(tmp_path):
    ind_path, rst_path, *_ = _fixture(tmp_path)
    dst = tmp_path / "copy.rst"
    n = patch_restart_zs(
        rst_path, dst, ind_path, SHAPE, np.zeros(SHAPE, dtype=bool), dz=1.0
    )
    assert n == 0
    assert dst.read_bytes() == rst_path.read_bytes()


def test_mismatched_record_markers_are_rejected(tmp_path):
    ind_path, rst_path, *_ = _fixture(tmp_path)
    bad = bytearray(rst_path.read_bytes())
    bad[4 + 4 : 4 + 4 + 4] = struct.pack("<i", 99)  # trailing marker of record 0
    rst_path.write_bytes(bytes(bad))
    with pytest.raises(ValueError, match="mismatched length markers"):
        patch_restart_zs(
            rst_path, tmp_path / "x.rst", ind_path, SHAPE, np.ones(SHAPE, bool), 0.1
        )
