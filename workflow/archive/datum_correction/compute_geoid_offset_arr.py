# Archived: 2026-06-05
# Source: workflow/src/raster.py  (function compute_geoid_offset_arr)
#
# Computes the geoid height offset N_EGM2008 − N_GOCO06s from ICGEM .gfc files
# using pyshtools + boule.  Was called in 03a_get_elevation.py as Step 1 of the
# EGM2008 → local MSL datum re-referencing.
#
# Dependencies (not installed by default):
#   conda install -c conda-forge pyshtools boule
#
# See also:
#   03a_datum_correction_excerpt.py — the call sites and surrounding script blocks
#   plot_datum_correction_delta.py  — diagnostic plot function

import numpy as np


def compute_geoid_offset_arr(
    goco_path,
    egm_path,
):
    """
    Compute the geoid height offset N_EGM2008 − N_GOCO06s from ICGEM .gfc files.

    EGM2008 is truncated to GOCO06s's maximum degree (≈ 300) before synthesis so
    both grids share the same spectral bandwidth — differencing at mismatched
    degrees would produce artefacts from EGM2008's high-frequency content that
    GOCO06s simply does not represent.

    Args:
        goco_path:  Path to GOCO06s.gfc
        egm_path:   Path to EGM2008.gfc (or any truncated version)

    Returns:
        offset_arr:  float32 ndarray, shape (nlat, nlon), global, north-up,
                     longitudes in −180…180 convention.
        transform:   rasterio Affine transform for the array.
        crs:         "EPSG:4326"
    """
    try:
        import pyshtools as pysh
    except ImportError:
        raise ImportError(
            "pyshtools is required for geoid height computation.\n"
            "Install with: conda install -c conda-forge pyshtools"
        )
    from rasterio.transform import from_origin

    goco = pysh.SHGravCoeffs.from_file(str(goco_path), format="icgem")
    egm = pysh.SHGravCoeffs.from_file(str(egm_path), format="icgem")

    lmax = goco.lmax
    egm_trunc = egm.pad(lmax)

    try:
        import boule as _boule

        wgs84 = _boule.WGS84
    except ImportError:
        raise ImportError(
            "boule is required by pyshtools for ellipsoid definitions.\n"
            "Install with: conda install -c conda-forge boule"
        )

    grid_goco = goco.geoid(ellipsoid=wgs84, lmax=lmax)
    grid_egm = egm_trunc.geoid(ellipsoid=wgs84, lmax=lmax)

    da_goco = grid_goco.to_xarray()
    da_egm = grid_egm.to_xarray()
    offset = (da_egm.values - da_goco.values).astype(np.float32)

    lat_dim = next(d for d in da_goco.dims if "lat" in d.lower())
    lon_dim = next(d for d in da_goco.dims if "lon" in d.lower())
    lats = da_goco[lat_dim].values
    lons = da_goco[lon_dim].values
    half = len(lons) // 2
    offset = np.roll(offset, -half, axis=1)
    lons = np.concatenate([lons[half:] - 360.0, lons[:half]])

    dlat = float(np.abs(lats[0] - lats[1]))
    dlon = float(lons[1] - lons[0])
    transform = from_origin(
        west=float(lons[0]) - dlon / 2,
        north=float(lats[0]) + dlat / 2,
        xsize=dlon,
        ysize=dlat,
    )
    return offset, transform, "EPSG:4326"
