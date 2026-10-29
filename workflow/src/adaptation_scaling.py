"""Resolve {"from_forcing": ...} strategy params from the scenario's baseline run.

Entry point: resolve_forcing_params. Markers are left by strategy_def_for()
(00_common.smk) and resolved from the baseline run and the skeleton weir:

  max_coastal_water_level  scale x max of the baseline sfincs.bzs (all points, all times)
  max_discharge           scale x max of the baseline sfincs.dis (all sources, all times)
  max_river_water_level    scale x max baseline zsmax in the grid cells along the
                          measure's own `locations` polyline (first zsmax output
                          step dropped: it holds the restart start-up shock,
                          e.g. 2.33 m at hour 0 in basin 2433835/coast_500)
  current_protection      {"current": True, "fill_crest": <m>}, never scaled;
                          the levee measure keeps each matched baseline weir
                          segment's crest and gives fill_crest (the highest
                          matched crest) to stretches with no baseline levee

Not resolvable in the Snakefile: the baseline files do not exist when the DAG is built.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
import xarray as xr
from src.sfincs_run import parse_sfincs_inp, read_weir_file

# Same tolerance as merge_weir_preserve_unmatched (src/protection_weir.py):
# baseline weir segments within this distance of the new polyline are the
# same physical levee (only reformatting/reprojection drift apart).
MAX_MATCH_DISTANCE_M = 2.0


def _max_of_timeseries_file(path: Path, what: str) -> float:
    """Max over every column but the first (time) of a SFINCS .bzs/.dis file."""
    if not path.exists():
        raise FileNotFoundError(
            f"{what} needs {path}, which this scenario's baseline run does not have"
        )
    data = np.loadtxt(path, ndmin=2)
    return float(data[:, 1:].max())


def read_polyline(locations: Path | gpd.GeoDataFrame, crs) -> gpd.GeoDataFrame:
    """Return polyline ``locations`` as single LineStrings in ``crs``, minus empty geometries.

    Empty geometries are rows whose geometry was deleted in QGIS but kept as
    attributes. Raises ValueError when ``locations`` has no CRS or holds a
    geometry that is not a (Multi)LineString.
    """
    if isinstance(locations, gpd.GeoDataFrame):
        gdf, name = locations, "polyline"
    else:
        gdf, name = gpd.read_file(locations), str(locations)
    if gdf.crs is None:
        raise ValueError(f"{name} has no CRS")
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    gdf = gdf.explode(index_parts=True).reset_index(drop=True)
    if not gdf.geometry.type.eq("LineString").all():
        raise ValueError("Weirs must be of type LineString.")
    return gdf[["geometry"]].to_crs(crs)


def _current_protection(
    locations: Path, skeleton_weir: Path, crs, where: str, log
) -> dict:
    weir = read_weir_file(skeleton_weir)
    line = read_polyline(locations, crs).geometry.union_all()
    # Segment MIDPOINT vs the polyline, as merge_weir_preserve_unmatched does --
    # a segment merely touching the polyline at one endpoint is not "on" it.
    midpoints = shapely.line_interpolate_point(
        weir.geometry.values, 0.5, normalized=True
    )
    matched = shapely.distance(midpoints, line) <= MAX_MATCH_DISTANCE_M
    if not matched.any():
        raise ValueError(
            f"{where}: current_protection found no segment of {skeleton_weir} within "
            f"{MAX_MATCH_DISTANCE_M} m of {locations} -- wrong polyline or CRS?"
        )
    crests = weir["elevation"].to_numpy()[matched]
    covered = shapely.union_all(
        [g.buffer(MAX_MATCH_DISTANCE_M) for g in weir.geometry.values[matched]]
    )
    unmatched_length = line.difference(covered).length
    fill_crest = float(crests.max())
    log.info(
        f"{where} = current_protection: {int(matched.sum())} baseline weir segments on "
        f"{locations.name}, crest min/median/max {crests.min():.2f}/{np.median(crests):.2f}/"
        f"{fill_crest:.2f} m; {unmatched_length:.0f} m of polyline without a baseline "
        f"levee gets fill_crest {fill_crest:.2f} m"
    )
    return {"current": True, "fill_crest": fill_crest}


def _max_river_water_level(
    locations: Path,
    map_nc: Path,
    dx_m: float,
    crs,
    where: str,
    log,
    buffer_cells: int = 1,
) -> float:
    line = read_polyline(locations, crs).geometry.union_all()
    with xr.open_dataset(map_nc) as ds:
        zsmax = ds["zsmax"]
        tdim = next(d for d in zsmax.dims if "time" in d)
        if zsmax.sizes[tdim] > 1:
            zsmax = zsmax.isel({tdim: slice(1, None)})  # drop restart start-up shock
        zsmax = zsmax.max(tdim).values  # NaN = never wet
        x, y = ds["x"].values, ds["y"].values  # cell centres
    near = shapely.dwithin(
        shapely.points(x.ravel(), y.ravel()), line, (buffer_cells + 0.5) * dx_m
    )
    vals = zsmax.ravel()[near]
    if not np.isfinite(vals).any():
        raise ValueError(
            f"{where}: no wet cell within {buffer_cells} cell(s) of {locations}"
        )
    k = int(np.nanargmax(vals))
    log.info(
        f"{where}: river zsmax along {locations.name}: max {vals[k]:.3f} m at "
        f"({x.ravel()[near][k]:.0f}, {y.ravel()[near][k]:.0f}), p99 {np.nanpercentile(vals, 99):.3f} m, "
        f"p50 {np.nanpercentile(vals, 50):.3f} m over {int(np.isfinite(vals).sum())} wet cells"
    )
    return float(vals[k])


def resolve_forcing_params(
    strategy_def: dict,
    *,
    baseline_sfincs_root: str | Path,
    baseline_map_nc: str | Path,
    skeleton_weir: str | Path,
    adaptation_root: str | Path,
    log: logging.Logger,
) -> dict:
    """Return a copy of `strategy_def` with every {"from_forcing": ...}
    marker replaced by its value for this scenario (see module docstring).
    """
    baseline_sfincs_root = Path(baseline_sfincs_root)
    inp = parse_sfincs_inp(baseline_sfincs_root / "sfincs.inp")
    crs = f"EPSG:{int(float(inp['epsg']))}"
    dx_m = float(inp["dx"])
    cache: dict[tuple, float] = {}

    out = copy.deepcopy(strategy_def)
    for measure, params in out["measures"].items():
        for param, value in params.items():
            if not (isinstance(value, dict) and "from_forcing" in value):
                continue
            kind, where = value["from_forcing"], f"{measure}.{param}"
            locations = (
                Path(adaptation_root) / params["locations"]
                if "locations" in params
                else None
            )

            if kind == "current_protection":
                params[param] = _current_protection(
                    locations, Path(skeleton_weir), crs, where, log
                )
                continue

            key = (kind, str(locations))
            if key not in cache:
                if kind == "max_coastal_water_level":
                    cache[key] = _max_of_timeseries_file(
                        baseline_sfincs_root / "sfincs.bzs", where
                    )
                elif kind == "max_discharge":
                    cache[key] = _max_of_timeseries_file(
                        baseline_sfincs_root / "sfincs.dis", where
                    )
                elif kind == "max_river_water_level":
                    cache[key] = _max_river_water_level(
                        locations, Path(baseline_map_nc), dx_m, crs, where, log
                    )
                else:
                    raise ValueError(f"{where}: unknown from_forcing kind {kind!r}")
            base, scale = cache[key], float(value["scale"])
            params[param] = round(
                scale * base, 4
            )  # 0.4 x 1.02 -> 0.408, not 0.40800000000000003
            log.info(f"{where} = {kind} {base:.3f} x {scale:g} = {params[param]:.3f}")
    return out
