"""
adaptation_scaling.py -- Resolve the run-time-derived strategy params that
rule 00_common.smk's strategy_def_for() leaves as {"from_forcing": ...}
markers (see adaptation_strategies.yml's own header), using this scenario's
own baseline run and the basin's skeleton weir:

  max_coastal_waterlevel  scale x max of the baseline sfincs.bzs (all points, all times)
  max_discharge           scale x max of the baseline sfincs.dis (all sources, all times)
  max_river_waterlevel    scale x max baseline zsmax in the grid cells along the
                          measure's own `locations` polyline (first zsmax output
                          step dropped -- it holds the restart start-up shock,
                          e.g. 2.33 m at hour 0 in basin 2433835/coast_500)
  current_protection      {"current": True, "fill_crest": <m>} -- never scaled;
                          the levee measure keeps each matched baseline weir
                          segment's own crest and gives fill_crest (the highest
                          matched crest) to stretches with no baseline levee

A coastal_levee/river_levee `elevation` resolved from max_coastal_waterlevel
or max_river_waterlevel also gets a sibling `raise_by` param = scale x the
largest shortfall of today's levee on the measure's polyline (target water
level - existing crest, over the matched baseline weir segments; target =
max_coastal_waterlevel everywhere for the coast, each segment's own local
zsmax for the river). The preprocessing levee raises every existing segment
by raise_by, so the new crest follows today's profile and is never lowered;
`elevation` itself stays the scalar design crest that the postprocessing
levee compares against the max water level, and that new stretches without a
baseline levee (e.g. a river-mouth closure) get.

These can't be resolved in the Snakefile itself: the baseline forcing/map
files don't exist yet when the DAG is built.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
import xarray as xr
from shapely.geometry import LineString

from src.sfincs_run import parse_sfincs_inp

# Same tolerance as merge_weir_preserve_unmatched (src/protection_weir.py):
# baseline weir segments within this distance of the new polyline are the
# same physical levee (only reformatting/reprojection drift apart).
MAX_MATCH_DISTANCE_M = 2.0

# measures whose `elevation` also gets a `raise_by` (see module docstring)
LEVEE_MEASURES = ("coastal_levee", "river_levee")


# Peak boundary value
def _max_of_timeseries_file(path: Path, what: str) -> float:
    """Max over every column but the first (time) of a SFINCS .bzs/.dis file."""
    if not path.exists():
        raise FileNotFoundError(
            f"{what} needs {path}, which this scenario's baseline run does not have"
        )
    data = np.loadtxt(path, ndmin=2)
    return float(data[:, 1:].max())


# Parse sfincs weirs
def read_weir_file(path: str | Path) -> gpd.GeoDataFrame:
    """ASCII sfincs.weir -> one row per segment, with `elevation` = the
    segment's highest vertex crest (every segment this repo writes has the
    same crest on both vertices). No CRS -- the caller sets it.
    """
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    geoms, crests, i = [], [], 0
    while i < len(lines):
        n_pts = int(lines[i + 1].split()[0])
        pts = np.array(
            [[float(v) for v in ln.split()] for ln in lines[i + 2 : i + 2 + n_pts]]
        )
        geoms.append(LineString(pts[:, :2]))
        crests.append(float(pts[:, 2].max()))
        i += 2 + n_pts
    return gpd.GeoDataFrame({"elevation": crests}, geometry=geoms)


# load a measures location
def _read_polyline(locations: Path, crs) -> shapely.Geometry:
    gdf = gpd.read_file(locations)
    if gdf.crs is None:
        raise ValueError(f"{locations} has no CRS")
    return gdf.to_crs(crs).geometry.union_all()


# baseline weir segments lying on a measure's polyline
def _matched_weir_segments(
    locations: Path, skeleton_weir: Path, crs, where: str
) -> tuple[gpd.GeoDataFrame, shapely.Geometry]:
    weir = read_weir_file(skeleton_weir)
    line = _read_polyline(locations, crs)
    # Segment MIDPOINT vs the polyline, as merge_weir_preserve_unmatched does --
    # a segment merely touching the polyline at one endpoint is not "on" it.
    midpoints = shapely.line_interpolate_point(
        weir.geometry.values, 0.5, normalized=True
    )
    matched = shapely.distance(midpoints, line) <= MAX_MATCH_DISTANCE_M
    if not matched.any():
        raise ValueError(
            f"{where}: found no segment of {skeleton_weir} within "
            f"{MAX_MATCH_DISTANCE_M} m of {locations} -- wrong polyline or CRS?"
        )
    return weir[matched].reset_index(drop=True), line


# checks how high existing levee is
def _current_protection(
    locations: Path, skeleton_weir: Path, crs, where: str, log
) -> dict:
    segs, line = _matched_weir_segments(locations, skeleton_weir, crs, where)
    crests = segs["elevation"].to_numpy()
    covered = shapely.union_all(
        [g.buffer(MAX_MATCH_DISTANCE_M) for g in segs.geometry.values]
    )
    unmatched_length = line.difference(covered).length
    fill_crest = float(crests.max())
    log.info(
        f"{where} = current_protection: {len(segs)} baseline weir segments on "
        f"{locations.name}, crest min/median/max {crests.min():.2f}/{np.median(crests):.2f}/"
        f"{fill_crest:.2f} m; {unmatched_length:.0f} m of polyline without a baseline "
        f"levee gets fill_crest {fill_crest:.2f} m"
    )
    return {"current": True, "fill_crest": fill_crest}


# peak water level along the river (zsmax)
def _max_river_waterlevel(
    locations: Path,
    map_nc: Path,
    dx: float,
    crs,
    where: str,
    log,
    buffer_cells: int = 1,
) -> float:
    line = _read_polyline(locations, crs)
    zsmax, x, y = _read_zsmax(map_nc)
    near = shapely.dwithin(shapely.points(x, y), line, (buffer_cells + 0.5) * dx)
    vals = zsmax[near]
    if not np.isfinite(vals).any():
        raise ValueError(
            f"{where}: no wet cell within {buffer_cells} cell(s) of {locations}"
        )
    k = int(np.nanargmax(vals))
    log.info(
        f"{where}: river zsmax along {locations.name}: max {vals[k]:.3f} m at "
        f"({x[near][k]:.0f}, {y[near][k]:.0f}), p99 {np.nanpercentile(vals, 99):.3f} m, "
        f"p50 {np.nanpercentile(vals, 50):.3f} m over {int(np.isfinite(vals).sum())} wet cells"
    )
    return float(vals[k])


# baseline max water level per grid cell, flattened, with cell centres
def _read_zsmax(map_nc: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(map_nc) as ds:
        zsmax = ds["zsmax"]
        tdim = next(d for d in zsmax.dims if "time" in d)
        if zsmax.sizes[tdim] > 1:
            zsmax = zsmax.isel({tdim: slice(1, None)})  # drop restart start-up shock
        zsmax = zsmax.max(tdim).values  # NaN = never wet
        x, y = ds["x"].values, ds["y"].values  # cell centres (rotated grid OK)
    return zsmax.ravel(), x.ravel(), y.ravel()


# how far today's levee on the polyline falls short of the target water level
def _levee_shortfall(
    kind: str,
    base: float,
    locations: Path,
    skeleton_weir: Path,
    map_nc: Path,
    dx: float,
    crs,
    where: str,
    log,
    buffer_cells: int = 1,
) -> float:
    """Largest (target - existing crest) over the baseline weir segments on
    `locations`, floored at 0. Target: `base` (the max coastal water level)
    for every segment, or for the river each segment's own max zsmax in the
    cells within buffer_cells of it (same window as _max_river_waterlevel);
    a segment with no wet cell nearby has no target and is skipped.
    """
    segs, _ = _matched_weir_segments(locations, skeleton_weir, crs, where)
    crests = segs["elevation"].to_numpy()
    if kind == "max_coastal_waterlevel":
        target = np.full(len(segs), base)
    else:
        zsmax, x, y = _read_zsmax(map_nc)
        tree = shapely.STRtree(shapely.points(x, y))
        target = np.full(len(segs), np.nan)
        for i, geom in enumerate(segs.geometry.values):
            idx = tree.query(
                geom, predicate="dwithin", distance=(buffer_cells + 0.5) * dx
            )
            if len(idx) and np.isfinite(zsmax[idx]).any():
                target[i] = np.nanmax(zsmax[idx])
    short = target - crests
    if not np.isfinite(short).any():
        raise ValueError(
            f"{where}: no baseline levee segment on {locations} has a target water level"
        )
    k = int(np.nanargmax(short))
    full = max(float(short[k]), 0.0)
    log.info(
        f"{where}: {int(np.isfinite(short).sum())} baseline levee segments on {locations.name}, "
        f"crest min/median/max {crests.min():.2f}/{np.median(crests):.2f}/{crests.max():.2f} m; "
        f"largest shortfall {short[k]:.3f} m (target {target[k]:.3f} m vs crest {crests[k]:.2f} m), "
        f"{int((short > 0).sum())} segments below target"
    )
    return full


# the entry point. Where the code determines whether it sends water level or max discharge
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
    dx = float(inp["dx"])
    cache: dict[tuple, float] = {}

    out = copy.deepcopy(strategy_def)
    for measure, params in out["measures"].items():
        # list(): raise_by is added to params inside this loop
        for param, value in list(params.items()):
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
                if kind == "max_coastal_waterlevel":
                    cache[key] = _max_of_timeseries_file(
                        baseline_sfincs_root / "sfincs.bzs", where
                    )
                elif kind == "max_discharge":
                    cache[key] = _max_of_timeseries_file(
                        baseline_sfincs_root / "sfincs.dis", where
                    )
                elif kind == "max_river_waterlevel":
                    cache[key] = _max_river_waterlevel(
                        locations, Path(baseline_map_nc), dx, crs, where, log
                    )
                else:
                    raise ValueError(f"{where}: unknown from_forcing kind {kind!r}")
            # apply the scale
            base, scale = cache[key], float(value["scale"])
            params[param] = round(
                scale * base, 4
            )  # 0.4 x 1.02 -> 0.408, not 0.40800000000000003
            log.info(f"{where} = {kind} {base:.3f} x {scale:g} = {params[param]:.3f}")

            # levees: uniform raise of today's levee (see module docstring)
            if (
                measure in LEVEE_MEASURES
                and param == "elevation"
                and kind
                in (
                    "max_coastal_waterlevel",
                    "max_river_waterlevel",
                )
            ):
                rkey = ("raise", kind, str(locations))
                if rkey not in cache:
                    cache[rkey] = _levee_shortfall(
                        kind,
                        base,
                        locations,
                        Path(skeleton_weir),
                        Path(baseline_map_nc),
                        dx,
                        crs,
                        where,
                        log,
                    )
                params["raise_by"] = round(scale * cache[rkey], 4)
                log.info(
                    f"{measure}.raise_by = largest shortfall {cache[rkey]:.3f} x {scale:g} "
                    f"= {params['raise_by']:.3f} m"
                )
    return out
