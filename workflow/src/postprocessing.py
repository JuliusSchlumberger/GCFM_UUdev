"""
Postprocessing of SFINCS scenario run outputs (sfincs_map.nc / sfincs_his.nc)
for output analysis and visualisation.

Each ``compute_*`` function loads a run directory's map output via
``SfincsModel`` (so spatial dimensions are HydroMT-aware), derives the
quantity needed for one type of analysis, and returns an ``xr.DataArray``
ready to be handed to the matching ``plot_*`` / ``animate_*`` function in
``src.plots``.  ``postprocess_sfincs_output`` dispatches to these by keyword.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from hydromt_sfincs import SfincsModel
from hydromt_sfincs import utils as sfincs_utils

# Copernicus LC100 land-use codes that represent water bodies (see _LC_NAMES
# in src.plots): 80 = "Inland water", 200 = "Sea".
WATER_LANDUSE_CODES: tuple[int, ...] = (80, 200)


def load_sfincs_output(run_dir: str | Path) -> SfincsModel:
    """Load a SFINCS run's map output via SfincsModel (HydroMT-aware spatial dims)."""
    mod = SfincsModel(root=str(run_dir), mode="r")
    mod.output.read()
    return mod


def get_bed_level(
    mod: SfincsModel,
    sfincs_root: str | Path,
    include_subgrid: bool = True,
) -> xr.DataArray | None:
    """
    Bed level (dep) used to convert water levels to inundation depths.

    Prefers ``subgrid/dep_subgrid.tif`` — the resolution SFINCS uses internally
    for subgrid runs — and falls back to the coarser model-grid ``zb`` written
    to the run output.  Returns None when neither is available.
    """
    dep_subgrid_path = Path(sfincs_root) / "subgrid" / "dep_subgrid.tif"
    if include_subgrid and dep_subgrid_path.exists():
        return mod.data_catalog.get_rasterdataset(str(dep_subgrid_path))
    if "zb" in mod.output.data:
        return mod.output.data["zb"].squeeze()
    return None


def compute_max_inundation(
    run_dir: str | Path,
    sfincs_root: str | Path,
    landuse_path: str | Path,
    hmin: float = 0.0,
    include_subgrid: bool = True,
) -> tuple[xr.DataArray, xr.DataArray] | tuple[None, None]:
    """
    Max inundation depth (zsmax − dep) for a SFINCS run, downscaled to the
    (sub)grid resolution and masked to the land domain.

    Takes the max of ``zsmax`` over the ``timemax`` dimension, determines the
    bed level via ``get_bed_level`` (subgrid-aware), derives the flood depth
    via ``hydromt_sfincs.utils.downscale_floodmap``, and masks both the flood
    depth and the bed-level reference grid to pixels where the land-use raster
    is in ``WATER_LANDUSE_CODES`` (80 = "Inland water", 200 = "Sea") — so all
    water bodies are excluded from both the flooded count and the land-domain
    denominator.

    Returns:
        (da_hmax, da_dep): the downscaled, land-masked flood depth and the
        bed-level raster used as the land-domain reference grid — its non-null
        pixel count gives the total number of land-domain pixels, e.g. for a
        flooded-area fraction (see 10_sanity_checks.py).  Both are None when
        ``zsmax`` or the bed level is unavailable.
    """
    mod = load_sfincs_output(run_dir)
    if "zsmax" not in mod.output.data:
        return None, None

    da_zsmax = mod.output.data["zsmax"]
    if "timemax" in da_zsmax.dims:
        da_zsmax = da_zsmax.max(dim="timemax")

    da_dep = get_bed_level(mod, sfincs_root, include_subgrid)
    if da_dep is None:
        return None, None

    da_hmax = sfincs_utils.downscale_floodmap(zsmax=da_zsmax, dep=da_dep, hmin=hmin)

    da_lu = mod.data_catalog.get_rasterdataset(str(landuse_path))
    # Compute eagerly: da_lu is dask-backed, and a dask-backed mask would force
    # da_dep/da_hmax to become lazy too via .where() below — breaking the
    # .item() calls in 10_sanity_checks.py ("'item' is not yet a valid method
    # on dask arrays").
    da_lu_grid = da_lu.raster.reproject_like(da_dep, method="nearest").compute()
    water_mask = da_lu_grid.isin(list(WATER_LANDUSE_CODES))

    da_dep = da_dep.where(~water_mask).compute()
    da_hmax = da_hmax.where(~water_mask).compute()
    return da_hmax, da_dep


def compute_flood_progression(
    run_dir: str | Path,
    landuse_path: str | Path,
    water_landuse_codes: tuple[int, ...] = WATER_LANDUSE_CODES,
) -> xr.DataArray | None:
    """
    Instantaneous land-surface inundation depth time series for an animation
    of flood progression.

    Loads the instantaneous water level ``zs`` (one frame per ``dtmapout``)
    and the static bed level ``zb`` from the run's map output, derives the
    depth ``h = max(zs - zb, 0)``, and masks out water bodies using the
    land-use raster reprojected onto the model grid — so only land-surface
    inundation remains.

    Returns None when ``zs`` (the full time-series map output, as opposed to
    just the ``zsmax`` envelope) is not present in the run output.
    """
    mod = load_sfincs_output(run_dir)
    if "zs" not in mod.output.data or "zb" not in mod.output.data:
        return None

    da_zs = mod.output.data["zs"]
    da_zb = mod.output.data["zb"].squeeze()
    da_h = (da_zs - da_zb).clip(min=0.0)
    da_h.name = "h"

    da_lu = mod.data_catalog.get_rasterdataset(str(landuse_path))
    da_lu_grid = da_lu.raster.reproject_like(da_h, method="nearest")
    water_mask = da_lu_grid.isin(list(water_landuse_codes))

    return da_h.where(~water_mask)


_ANALYSES = {
    "max_inundation": compute_max_inundation,
    "flood_progression": compute_flood_progression,
}


def postprocess_sfincs_output(
    run_dir: str | Path,
    analysis: str,
    **kwargs,
):
    """
    Run the postprocessing required for a named SFINCS output analysis.

    Args:
        run_dir:  Path to the SFINCS run directory containing ``sfincs_map.nc``
                  / ``sfincs_his.nc`` (e.g. ``<sfincs_root>/spinup`` or a
                  scenario subdirectory).
        analysis: One of ``"max_inundation"`` or ``"flood_progression"``.
        **kwargs: Forwarded to the matching ``compute_*`` function — see their
                  docstrings for the required dataset paths, options, and
                  return values (``compute_max_inundation`` returns a
                  ``(da_hmax, da_dep)`` tuple, ``compute_flood_progression``
                  returns a single ``xr.DataArray``).

    Returns:
        The result of the matching ``compute_*`` function, ready to be passed
        to ``plot_max_inundation_map`` / ``animate_flood_progression`` in
        ``src.plots``.
    """
    try:
        compute_fn = _ANALYSES[analysis]
    except KeyError:
        raise ValueError(
            f"Unknown analysis {analysis!r}; expected one of {sorted(_ANALYSES)}"
        ) from None
    return compute_fn(run_dir, **kwargs)
