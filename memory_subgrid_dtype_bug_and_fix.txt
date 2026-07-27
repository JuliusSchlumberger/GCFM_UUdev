"""
hydromt_sfincs_subgrid_dtype_bugfix.py -- write-up + minimal reproduction of
a memory bug found in hydromt_sfincs.components.grid.subgrid.SubgridTable.
write_netcdf(), for reporting back to the hydromt_sfincs maintainers.

Patched file (local hydromt_sfincs clone): C:/Users/Schlu005/repos/
hydromt_sfincs/hydromt_sfincs/components/grid/subgrid.py -- see that repo's
`git diff` for the exact change already applied there.

Where we hit this
------------------
While benchmarking SFINCS build time across grid resolutions for a large
delta basin (~1e6 active z-points at 300 m resolution), model builds started
failing with numpy.core._exceptions._ArrayMemoryError partway through
sf.write() -> subgrid.write_netcdf(), e.g.:

    numpy._core._exceptions._ArrayMemoryError: Unable to allocate 2.78 GiB
    for an array with shape (30, 12438241) and data type float64

The bug
-------
Every subgrid array built in SubgridTable.create() is explicitly allocated
as float32 (z_zmin, z_zmax, z_volmax, z_level, u_zmin/u_zmax/u_havg/u_nrep/
u_pwet/u_ffit/u_navg, and the v_* equivalents -- confirmed by reading every
np.full(..., dtype=np.float32) call in create(), lines ~139-174 and
~738-770 as of this writing).

write_netcdf() (lines ~382-449) re-derives active-cell-only versions of
these same arrays before writing them to the output .nc file, via three
np.zeros(...) calls that do NOT specify a dtype:

    z_level = np.zeros((nr_levels, nr_z_points))                  # line ~416
    uv_var  = np.zeros(nr_uv_points)                               # line ~424
    uv_var  = np.zeros((nr_levels, nr_uv_points))                  # line ~435

np.zeros() defaults to float64, so each of these silently allocates twice
the memory actually needed -- for data that is float32-precision on both
sides of the assignment (the right-hand side, ds["u_" + var].values, is
always float32). The extra precision is never used: the float32 source
values are simply upcast into a float64 buffer and later written to netCDF,
gaining nothing. For large domains at fine resolution this routinely adds
1-3 GiB of avoidable peak memory at exactly the step where our builds were
already memory-constrained, and was the direct cause of the crash quoted
above (nr_uv_points=12,438,241 * 30 levels * 8 bytes = 2.78 GiB; at float32
this becomes 1.39 GiB).

The fix
-------
Add dtype=np.float32 to all three allocations, matching create()'s own
dtype convention. This is a pure memory-efficiency fix with zero effect on
numerical results (see verification below): every value written is already
float32-precision before this function ever runs.

Status: applied directly in our local hydromt_sfincs clone (see the file
path above). This script documents and demonstrates the fix for reporting
upstream (GitHub issue / PR against hydromt_sfincs).
"""

import numpy as np


def demonstrate_memory_difference() -> None:
    """Reproduce the bug's memory impact at the exact shape that crashed,
    without needing a full SFINCS model -- just the same allocation pattern
    write_netcdf() uses.
    """
    nr_levels = 30
    nr_uv_points = 12_438_241  # from the actual crash (basin 4267691, R=120 m)

    before = np.zeros((nr_levels, nr_uv_points))  # buggy: defaults to float64
    after = np.zeros((nr_levels, nr_uv_points), dtype=np.float32)  # fixed

    before_gib = before.nbytes / 1024**3
    after_gib = after.nbytes / 1024**3
    print(f"shape={before.shape}")
    print(f"  before fix: dtype={before.dtype}, {before_gib:.2f} GiB")
    print(f"  after fix:  dtype={after.dtype}, {after_gib:.2f} GiB")
    print(
        f"  saved:      {before_gib - after_gib:.2f} GiB ({(1 - after_gib / before_gib):.0%})"
    )

    del before, after


def verify_no_precision_loss() -> None:
    """Confirm the fix doesn't change written values: the source data
    (float32, exactly as create() produces it) survives a float32 buffer
    round-trip bit-for-bit, since it was never higher precision to begin
    with -- the float64 buffer only ever held upcast float32 values.
    """
    rng = np.random.default_rng(0)
    source_f32 = rng.random(10_000).astype(np.float32)

    buffer_f64 = np.zeros(10_000)
    buffer_f64[:] = source_f32
    roundtrip_via_f64 = buffer_f64.astype(np.float32)

    buffer_f32 = np.zeros(10_000, dtype=np.float32)
    buffer_f32[:] = source_f32
    roundtrip_via_f32 = buffer_f32

    identical = np.array_equal(roundtrip_via_f64, roundtrip_via_f32)
    print(f"values identical after fix: {identical}")
    assert identical, "fix must not alter written values"


if __name__ == "__main__":
    demonstrate_memory_difference()
    print()
    verify_no_precision_loss()
