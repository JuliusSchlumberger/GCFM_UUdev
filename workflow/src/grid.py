"""Define the SFINCS model grid once per basin and share it across rules.

The grid is a regular lattice, axis-aligned or rotated (``sfincs.grid.rotated``).
Rule determine_model_domain (02) fits the frame (``fit_grid_frame``), rule
build_sfincs_grid (08c) writes ``{basin_id}_sfincs_grid.json``
(``build_grid_def``); later rules read it (``load_grid_def``) and build every
SfincsModel from it (``create_model_grid``).

Conventions (both orientations; as in hydromt_sfincs ``sf.grid.data``):

- ``transform = translation(x0, y0) * rotation(rotation) * scale(dx, dy)``,
  ``dy`` POSITIVE: row 0 is the side through the origin (the southern side
  of an unrotated grid), so row = SFINCS n index, col = SFINCS m index.
- ``rotation`` is counter-clockwise, degrees, about (x0, y0).
- The cell centre of (row, col) is ``transform * (col + 0.5, row + 0.5)``.

With a rotated transform the pixel size is NOT ``transform.a`` and the
pixel area is NOT ``a * e`` (those are dx*cos(rotation) and
dx*dy*cos^2(rotation)) -- use ``cell_size_m`` / ``cell_area_m2`` below,
which are correct for any orientation.
"""

import json
import math
from pathlib import Path

import geopandas as gpd
from affine import Affine
from shapely.geometry import Polygon

# Rounding hydromt(_sfincs) itself applies when fitting a rotated grid
# (create_from_region's own dec_origin / dec_rotation defaults).
_DEC_ORIGIN = 0
_DEC_ROTATION = 3


def cell_size_m(transform) -> float:
    """Return the pixel size along the column axis (m) of a possibly rotated transform."""
    return float(math.hypot(transform.a, transform.d))


def cell_area_m2(transform) -> float:
    """Return the pixel area (m2) of a possibly rotated transform."""
    return float(abs(transform.a * transform.e - transform.b * transform.d))


def footprint(transform, shape) -> Polygon:
    """Return the true outline of a raster (shape = (rows, cols)) in its own CRS."""
    rows, cols = shape
    return Polygon(
        [transform * c for c in [(0, 0), (cols, 0), (cols, rows), (0, rows)]]
    )


def fit_grid_frame(domain_utm: gpd.GeoDataFrame, rotated: bool) -> dict:
    """Return the rectangle the model grid lives in, fitted to the UTM domain polygon.

    rotated=True: the minimum rotated rectangle, as hydromt_sfincs
    ``create_from_region(rotated=True)``, fitted at 1 m so side lengths are
    whole metres and independent of resolution. rotated=False: nothing is
    stored; rule 08c fits the axis-aligned grid to the polygon bounds.
    """
    if not rotated:
        return {"rotated": False}
    from hydromt.model.processes.grid import rotated_grid

    x0, y0, length_x, length_y, rotation = rotated_grid(
        domain_utm.union_all(), 1.0, dec_origin=_DEC_ORIGIN, dec_rotation=_DEC_ROTATION
    )
    return {
        "rotated": True,
        "x0": float(x0),
        "y0": float(y0),
        # 0-360, as hydromt_sfincs itself writes it to sfincs.inp
        "rotation": round(float(rotation) % 360.0, _DEC_ROTATION),
        "length_x_m": int(length_x),
        "length_y_m": int(length_y),
    }


def frame_footprint(frame: dict) -> Polygon:
    """Return the outline of a rotated grid frame in the domain's UTM CRS."""
    transform = Affine.translation(frame["x0"], frame["y0"]) * Affine.rotation(
        frame["rotation"]
    )
    return footprint(transform, (frame["length_y_m"], frame["length_x_m"]))


def build_grid_def(domain: gpd.GeoDataFrame, frame: dict, resolution: float) -> dict:
    """Return the JSON-serialisable grid definition with cells of resolution m.

    Holds the SFINCS grid parameters (x0, y0, dx, dy, mmax, nmax, rotation,
    epsg) plus the equivalent affine six-tuple and raster shape.
    """
    if frame.get("rotated"):
        x0, y0, rotation = frame["x0"], frame["y0"], frame["rotation"]
        mmax = int(math.ceil(frame["length_x_m"] / resolution))
        nmax = int(math.ceil(frame["length_y_m"] / resolution))
        crs = domain.crs
    else:
        # Same call and y-ascending flip as SfincsModel.grid.create_from_region(),
        # so the grid matches hydromt_sfincs' own unrotated grid.
        from hydromt.model.processes.grid import create_grid_from_region

        ds = create_grid_from_region(
            region={"geom": domain},
            res=resolution,
            crs="utm",
            region_crs=4326,
            rotated=False,
            add_mask=False,
            align=True,
        )
        if ds.raster.res[1] < 0:
            ds = ds.raster.flipud()
        t = ds.raster.transform
        x0, y0, rotation = t.c, t.f, 0.0
        nmax, mmax = ds.raster.shape
        crs = ds.raster.crs
    transform = (
        Affine.translation(x0, y0)
        * Affine.rotation(rotation)
        * Affine.scale(resolution, resolution)
    )
    return {
        "rotated": bool(frame.get("rotated")),
        "resolution": float(resolution),
        "crs": crs.to_string(),
        "epsg": int(crs.to_epsg()),
        "x0": float(x0),
        "y0": float(y0),
        "dx": float(resolution),
        "dy": float(resolution),
        "mmax": int(mmax),
        "nmax": int(nmax),
        "rotation": float(rotation),
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
