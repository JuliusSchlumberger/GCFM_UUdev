"""The SFINCS model grid: one definition per basin, shared by every rule.

The grid is a regular lattice, either axis-aligned or ROTATED
(``sfincs.grid.rotated`` in config.yml). Rule determine_model_domain (02)
fits the rectangle the grid lives in (``fit_grid_frame``), rule
build_sfincs_grid (08c) subdivides it into cells (``build_grid_def``) and
writes ``{basin_id}_sfincs_grid.json``; every later rule reads that file
(``load_grid_def``) and every SfincsModel is created from it
(``create_model_grid``) -- nothing refits the grid on its own.

Conventions (identical for both orientations, and identical to what
hydromt_sfincs keeps in ``sf.grid.data``):

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
    """Pixel size along the column axis (m) of any affine transform, rotated or not."""
    return float(math.hypot(transform.a, transform.d))


def cell_area_m2(transform) -> float:
    """Pixel area (m2) of any affine transform, rotated or not."""
    return float(abs(transform.a * transform.e - transform.b * transform.d))


def footprint(transform, shape) -> Polygon:
    """True outline of a raster (``shape`` = (rows, cols)) as a polygon in its own CRS."""
    rows, cols = shape
    return Polygon(
        [transform * c for c in [(0, 0), (cols, 0), (cols, rows), (0, rows)]]
    )


def fit_grid_frame(domain_utm: gpd.GeoDataFrame, rotated: bool) -> dict:
    """Rectangle the model grid lives in, fitted to the domain polygon (UTM).

    ``rotated=True``: the minimum rotated rectangle around the polygon --
    the same fit hydromt_sfincs' ``create_from_region(rotated=True)`` does,
    run at 1 m so the side lengths come out in whole metres and the
    rectangle does not depend on the grid resolution. ``rotated=False``:
    no rectangle is stored; rule 08c fits the axis-aligned grid to the
    polygon's own bounds, snapped to the resolution.
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
    """Outline of a rotated grid frame (``fit_grid_frame``) in the domain's UTM CRS."""
    transform = Affine.translation(frame["x0"], frame["y0"]) * Affine.rotation(
        frame["rotation"]
    )
    return footprint(transform, (frame["length_y_m"], frame["length_x_m"]))


def build_grid_def(domain: gpd.GeoDataFrame, frame: dict, resolution: float) -> dict:
    """Subdivide a basin's grid frame into cells of ``resolution`` m.

    Returns the JSON-serialisable grid definition written by rule 08c: the
    SFINCS grid parameters (x0, y0, dx, dy, mmax, nmax, rotation, epsg) plus
    the equivalent affine six-tuple and raster shape.
    """
    if frame.get("rotated"):
        x0, y0, rotation = frame["x0"], frame["y0"], frame["rotation"]
        mmax = int(math.ceil(frame["length_x_m"] / resolution))
        nmax = int(math.ceil(frame["length_y_m"] / resolution))
        crs = domain.crs
    else:
        # Same call (and the same y-ascending flip) SfincsModel.grid.
        # create_from_region() makes, so an unrotated grid is exactly the
        # one this pipeline built before the grid definition was shared.
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
    """Read ``{basin_id}_sfincs_grid.json``; ``transform`` comes back as an
    Affine and ``shape`` as (rows, cols) = (nmax, mmax).
    """
    with open(path) as fh:
        grid_def = json.load(fh)
    grid_def["transform"] = Affine(*grid_def["transform"])
    grid_def["shape"] = (int(grid_def["nmax"]), int(grid_def["mmax"]))
    return grid_def


def create_model_grid(sf, grid_def: dict) -> None:
    """Create ``sf``'s regular grid from the shared grid definition."""
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
