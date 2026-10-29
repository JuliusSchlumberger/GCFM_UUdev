"""Define the SFINCS model grid once per basin and share it across rules.

The grid is an axis-aligned regular lattice. Rule build_sfincs_grid (04) writes
``{basin_id}_sfincs_grid.json`` (``build_grid_def``); later rules read it
(``load_grid_def``) and build every SfincsModel from it (``create_model_grid``).

Conventions (as in hydromt_sfincs ``sf.grid.data``):

- ``transform = translation(x0, y0) * scale(dx, dy)``, ``dy`` POSITIVE: row 0
  is the southern side, so row = SFINCS n index, col = SFINCS m index.
- The cell centre of (row, col) is ``transform * (col + 0.5, row + 0.5)``.
- ``rotation`` is always 0; it is stored because SFINCS requires it.
"""

import json
import math
from pathlib import Path

import geopandas as gpd
from affine import Affine
from shapely.geometry import Polygon


def cell_size_m(transform) -> float:
    """Return the pixel size along the column axis (m); valid for any transform orientation."""
    return float(math.hypot(transform.a, transform.d))


def cell_area_m2(transform) -> float:
    """Return the pixel area (m2); valid for any transform orientation."""
    return float(abs(transform.a * transform.e - transform.b * transform.d))


def footprint(transform, shape) -> Polygon:
    """Return the true outline of a raster (shape = (rows, cols)) in its own CRS."""
    rows, cols = shape
    return Polygon(
        [transform * c for c in [(0, 0), (cols, 0), (cols, rows), (0, rows)]]
    )


def build_grid_def(domain: gpd.GeoDataFrame, resolution_m: float) -> dict:
    """Return the JSON-serialisable grid definition with square cells of resolution_m.

    Holds the SFINCS grid parameters (x0, y0, dx, dy, mmax, nmax, rotation,
    epsg) plus the equivalent affine six-tuple and raster shape.
    """
    # Same call and y-ascending flip as SfincsModel.grid.create_from_region(),
    # so the grid matches hydromt_sfincs' own grid.
    from hydromt.model.processes.grid import create_grid_from_region

    ds = create_grid_from_region(
        region={"geom": domain},
        res=resolution_m,
        crs="utm",
        region_crs=4326,
        rotated=False,
        add_mask=False,
        align=True,
    )
    if ds.raster.res[1] < 0:
        ds = ds.raster.flipud()
    t = ds.raster.transform
    x0, y0 = t.c, t.f
    nmax, mmax = ds.raster.shape
    crs = ds.raster.crs
    transform = Affine.translation(x0, y0) * Affine.scale(resolution_m, resolution_m)
    return {
        "resolution": float(resolution_m),
        "crs": crs.to_string(),
        "epsg": int(crs.to_epsg()),
        "x0": float(x0),
        "y0": float(y0),
        "dx": float(resolution_m),
        "dy": float(resolution_m),
        "mmax": int(mmax),
        "nmax": int(nmax),
        "rotation": 0.0,
        "height": int(nmax),
        "width": int(mmax),
        # affine six-tuple (a, b, c, d, e, f) -- Affine(*transform) rebuilds it.
        "transform": list(transform)[:6],
    }


def load_grid_def(path: str | Path) -> dict:
    """Read ``{basin_id}_sfincs_grid.json`` with transform as Affine.

    Adds ``shape`` as (rows, cols) = (nmax, mmax).
    """
    with open(path) as fh:
        grid_def = json.load(fh)
    grid_def["transform"] = Affine(*grid_def["transform"])
    grid_def["shape"] = (int(grid_def["nmax"]), int(grid_def["mmax"]))
    return grid_def


def create_model_grid(sf, grid_def: dict) -> None:
    """Create sf's regular grid from the shared grid definition.

    Raises ValueError if the resulting transform differs from the definition.
    """
    sf.grid.create(
        x0=grid_def["x0"],
        y0=grid_def["y0"],
        dx=grid_def["dx"],
        dy=grid_def["dy"],
        nmax=grid_def["nmax"],
        mmax=grid_def["mmax"],
        rotation=grid_def["rotation"],
        epsg=grid_def["epsg"],
    )
    t = sf.grid.data.raster.transform
    if not t.almost_equals(grid_def["transform"], precision=1e-6):
        raise ValueError(
            f"SfincsModel grid transform {tuple(t)[:6]} differs from the shared "
            f"grid definition {tuple(grid_def['transform'])[:6]}"
        )
