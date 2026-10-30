"""River water surface from FathomDEM, its slope, and the DEM conditioned to it.

Entry points: :func:`build_water_surface` and :func:`condition_channels_to_surface`
(rule river_surface, 10); :func:`read_water_surface` (rule 11's river bed);
:func:`river_cells_at_surface` (initial water level, rules 11 and 12a).

FathomDEM's channels hold a flattened water surface: flat within a water
body, dropping in steps between them, with stray peaks (bridges, bars, bank
pixels) and dips. The surface is built on the node network
(src.gvf_target.build_node_network), which rule 11 rebuilds identically:

1. median of the water pixels (land use == ``landuse.water_code``) along the
   centreline within ``median_half_window_m`` of each node -- drops peaks and
   dips alike. Islands and banks the centreline crosses are skipped, so their
   nodes take the level of the water around them;
2. sea mouths pinned at ``calm_sea_level_m`` and every node held at or above
   it: FathomDEM flattens open water to the EGM2008 geoid, below the model's
   sea (geoid + MDT), and no steady river level lies below its sea. A mouth is
   a network end that is not a delta outflow and is sea class, has no level,
   or lies within ``mouth_level_tol_m`` of the sea;
3. non-increasing downstream (network isotonic regression), smoothed over
   ``smoothing_m``, and made non-increasing again, the mouths held fixed;
4. slope: the same fit smoothed over ``slope_smoothing_m`` instead, its fall
   per metre along each node's largest-Q path (src.gvf_target.main_path_slope),
   not floored. FathomDEM's surface on low-gradient rivers is a staircase of
   flat treads, so a slope over a few km is near zero on the treads and steep
   at the risers; the level itself keeps the short smoothing, as a long one
   rides above a concave profile. SWOT reach slopes are a check only (reported
   per reach).

Structures (weirs, locks, dams) are detected where the reach level drops by
more than ``step_threshold_m`` (or ``structure_min_drop_m`` at a SWORD
obstruction reach) and reported; the surface is smoothed across them, as they
open in floods. Rises of that size are reported too.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from scipy.spatial import cKDTree
from shapely.geometry import Point
from src.gvf_target import (
    NODE_INVERT,
    GvfNetwork,
    _fill_from_neighbours,
    build_node_network,
    classify_nodes,
    detect_structure_edges,
    enforce_non_increasing,
    main_path_slope,
    orient_downstream,
    smooth_on_network,
)
from src.io import require_keys
from src.river_network import (
    _sample_line_cells,
    build_downstream_adjacency,
    clamped_profile,
    line_window_cells,
    normalize_reach_id,
)

log = logging.getLogger(__name__)

LANDUSE_SEA_CODE = 200
SURFACE_LAYER = "nodes"
STRUCTURE_LAYER = "structures"
SURFACE_KEYS = (
    "node_spacing_m",
    "median_half_window_m",
    "smoothing_m",
    "slope_smoothing_m",
    "step_threshold_m",
    "structure_min_drop_m",
    "use_sword_obstructions",
    "structures",
    "mouth_level_tol_m",
    "water_body",
)
SWOT_RATIO_SPAN = 2.0  # a surface slope within this factor of SWOT counts as agreeing


@dataclass
class WaterSurface:
    """Node network with the raw (median) and smoothed water level and slope per node."""

    net: GvfNetwork
    level_m: np.ndarray
    level_raw_m: np.ndarray
    slope: np.ndarray  # m/m along the largest-Q path, not floored
    rising_edges: list[tuple[str, str, float]]
    samples: dict[
        str, tuple[np.ndarray, np.ndarray]
    ]  # {reach: (chainage_m, elevation_m)} of the pixels the medians use
    skipped: dict[
        str, tuple[np.ndarray, np.ndarray]
    ]  # {reach: (chainage_m, elevation_m)} of the other centreline pixels
    structures: set[tuple[str, str]]  # detected, reported only
    mouths: np.ndarray  # node ids pinned at the sea level
    node_class: np.ndarray
    dem_m: np.ndarray
    width_m: np.ndarray
    swot: pd.DataFrame  # per reach: swot_slope, surface_slope, ratio


def sample_centrelines(
    lines: dict[str, object],
    dem_path: str | Path,
    landuse_path: str | Path,
    water_code: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {reach: (chainage_m, elevation_m, is_water)} of the pixels along each line.

    is_water marks the pixels the water surface is built from; the others
    (islands, banks, sea) are returned for the rule 10 figure.

    The DEM and the land-use raster must share one grid (rule 06 writes
    landuse.tif on elevation_merged.tif's grid); lines are in that CRS.
    """
    with rasterio.open(dem_path) as dem_src, rasterio.open(landuse_path) as lu_src:
        if dem_src.transform != lu_src.transform or dem_src.shape != lu_src.shape:
            raise ValueError(f"{landuse_path} is not on the grid of {dem_path}")
        dem = dem_src.read(1).astype(np.float64)
        nodata = dem_src.nodata
        water = lu_src.read(1) == water_code
        transform = dem_src.transform
    if nodata is not None:
        dem[dem == nodata] = np.nan
    step_m = float(np.hypot(transform.a, transform.d))
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for rid, line in lines.items():
        cells = _sample_line_cells(line, transform, dem.shape, step_m)
        if not cells:
            out[rid] = (np.array([]), np.array([]), np.array([], dtype=bool))
            continue
        rr = np.array([c["row"] for c in cells])
        cc = np.array([c["col"] for c in cells])
        along = np.array([c["along_m"] for c in cells])
        keep = np.isfinite(dem[rr, cc])
        out[rid] = (along[keep], dem[rr, cc][keep], water[rr, cc][keep])
    return out


def node_medians(
    net: GvfNetwork,
    samples: dict[str, tuple[np.ndarray, np.ndarray]],
    half_window_m: float,
) -> np.ndarray:
    """Return per node the median sample within ``half_window_m`` along the network.

    The window runs on into the up- and downstream reaches (not across a cut),
    so a reach shorter than the window -- e.g. a lock whose FathomDEM pixels
    are walls, not water -- is outvoted by the river around it. A node shared
    by several reaches gets the mean of their medians; nodes without samples
    (sea, no data) are filled from their graph neighbours.
    """
    upstream = net.upstream()
    empty = (np.array([]), np.array([]))

    def linked(u: str, d: str) -> bool:
        return (u, d) not in net.structure_edges

    def window_samples(r: str) -> tuple[np.ndarray, np.ndarray]:
        pos, val = [samples.get(r, empty)[0]], [samples.get(r, empty)[1]]
        # Upstream: a reach ending at position `end` (relative to r's start).
        stack, seen = [(u, 0.0) for u in upstream[r] if linked(u, r)], {r}
        while stack:
            u, end = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            a, v = samples.get(u, empty)
            start = end - net.reach_length[u]
            pos.append(a + start)
            val.append(v)
            if start > -half_window_m:
                stack += [(w, start) for w in upstream[u] if linked(w, u)]
        # Downstream: a reach starting at position `start`.
        stack, seen = (
            [(d, net.reach_length[r]) for d in net.downstream[r] if linked(r, d)],
            {r},
        )
        while stack:
            d, start = stack.pop()
            if d in seen:
                continue
            seen.add(d)
            a, v = samples.get(d, empty)
            pos.append(a + start)
            val.append(v)
            end = start + net.reach_length[d]
            if end < net.reach_length[r] + half_window_m:
                stack += [(w, end) for w in net.downstream[d] if linked(d, w)]
        return np.concatenate(pos), np.concatenate(val)

    acc = np.zeros(net.n_nodes)
    cnt = np.zeros(net.n_nodes)
    for r in net.reach_ids:
        along, value = window_samples(r)
        if along.size == 0:
            continue
        for nid, ch in zip(net.reach_nodes[r], net.reach_chainage[r]):
            near = np.abs(along - ch) <= half_window_m
            if near.any():
                acc[nid] += float(np.median(value[near]))
                cnt[nid] += 1.0
    raw = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return _fill_from_neighbours(raw, net.node_edges())


def _split_samples(raw: dict) -> tuple[dict, dict]:
    used, skipped = {}, {}
    for r, t in raw.items():
        along, value = np.asarray(t[0]), np.asarray(t[1])
        use = (
            np.asarray(t[2], dtype=bool)
            if len(t) > 2
            else np.ones(along.size, dtype=bool)
        )
        used[r] = (along[use], value[use])
        skipped[r] = (along[~use], value[~use])
    return used, skipped


def _reach_levels(net: GvfNetwork, node_level: np.ndarray) -> dict[str, float]:
    return {
        r: float(np.nanmedian(node_level[net.reach_nodes[r]])) for r in net.reach_ids
    }


def node_widths_from_reaches(net: GvfNetwork, rivers: gpd.GeoDataFrame) -> np.ndarray:
    """Return per node the mean width of the reaches that share it."""
    reach_width = dict(
        zip(
            rivers["reach_id"].map(normalize_reach_id),
            pd.to_numeric(rivers["width"], errors="coerce").astype(float),
        )
    )
    acc = np.zeros(net.n_nodes)
    cnt = np.zeros(net.n_nodes)
    for r in net.reach_ids:
        np.add.at(acc, net.reach_nodes[r], reach_width.get(r, np.nan))
        np.add.at(cnt, net.reach_nodes[r], 1.0)
    return acc / np.maximum(cnt, 1)


def _reach_flag(rivers: gpd.GeoDataFrame, column: str) -> set[str]:
    if column not in rivers.columns:
        return set()
    return {
        normalize_reach_id(r)
        for r, f in zip(rivers["reach_id"], rivers[column])
        if bool(f)
    }


def _reach_discharge(rivers: gpd.GeoDataFrame) -> dict[str, float]:
    # Routed bankfull discharge (rule 08) picks the main arm; width where it is missing.
    col = (
        "bankfull_discharge_acc_m3s"
        if "bankfull_discharge_acc_m3s" in rivers.columns
        else "width"
    )
    return dict(
        zip(
            rivers["reach_id"].map(normalize_reach_id),
            pd.to_numeric(rivers[col], errors="coerce").fillna(0.0).astype(float),
        )
    )


def sea_mouths(
    net: GvfNetwork,
    level_raw_m: np.ndarray,
    node_class: np.ndarray,
    outflow_reaches: set[str],
    sea_level_m: float,
    tol_m: float,
) -> np.ndarray:
    """Return the network ends that discharge to the sea.

    A downstream end that is not a delta outflow and is sea class, has no
    level, or lies within ``tol_m`` of ``sea_level_m``.
    """
    ends = []
    for r in net.reach_ids:
        if net.downstream[r] or r in outflow_reaches:
            continue
        i = int(net.reach_nodes[r][-1])
        if (
            node_class[i] != NODE_INVERT
            or not np.isfinite(level_raw_m[i])
            or level_raw_m[i] <= sea_level_m + tol_m
        ):
            ends.append(i)
    return np.array(sorted(set(ends)), dtype=int)


def swot_check(
    net: GvfNetwork, rivers: gpd.GeoDataFrame, level_m: np.ndarray
) -> pd.DataFrame:
    """Return per reach with a reliable positive SWOT slope the surface slope and their ratio."""
    if not {"slope_obs_reliable", "slope_obs_p50"} <= set(rivers.columns):
        return pd.DataFrame(
            columns=["reach_id", "swot_slope", "surface_slope", "ratio"]
        )
    rel = pd.to_numeric(rivers["slope_obs_reliable"], errors="coerce")
    p50 = pd.to_numeric(rivers["slope_obs_p50"], errors="coerce")
    rows = []
    for r, s, ok in zip(rivers["reach_id"].map(normalize_reach_id), p50, rel):
        if ok != 1 or not np.isfinite(s) or s <= 0 or r not in net.reach_nodes:
            continue
        ids, ch = net.reach_nodes[r], net.reach_chainage[r]
        surf = float(level_m[ids[0]] - level_m[ids[-1]]) / max(float(ch[-1]), 1.0)
        rows.append(
            {
                "reach_id": r,
                "swot_slope": float(s),
                "surface_slope": surf,
                "ratio": surf / float(s),
            }
        )
    return pd.DataFrame(
        rows, columns=["reach_id", "swot_slope", "surface_slope", "ratio"]
    )


def build_water_surface(
    rivers: gpd.GeoDataFrame,
    sample,
    cfg: dict,
    sea_level_m: float | None = None,
    sample_nodes=None,
    gebco_floor_m: float = -np.inf,
) -> tuple[WaterSurface, gpd.GeoDataFrame]:
    """Build the water surface and its slope on the node network.

    Args:
        rivers: Cleaned network in the DEM's metric CRS ('reach_id',
            'rch_id_dn', 'width', optionally 'obstr_type', 'is_delta_outflow',
            'bankfull_discharge_acc_m3s', 'slope_obs_p50', 'slope_obs_reliable').
        sample: Callable {reach: LineString} -> {reach: (chainage_m,
            elevation_m[, use])}, e.g. :func:`sample_centrelines` bound to the
            rasters; ``use`` marks the pixels the medians take (default all).
        cfg: ``river_processing.water_surface``.
        sea_level_m: ``calm_sea_level_m`` in the model datum; the mouths are
            pinned at it and no node lies below it. None: no pinning (tests).
        sample_nodes: Callable (x, y, width_m) -> (dem_m, sea_fraction) per
            node (:func:`node_sampler`); None: every node is a river node.
        gebco_floor_m: Lowest level of the clamped GEBCO bathymetry.

    Returns:
        The surface and the reaches with downstream-running lines.
    """
    require_keys(cfg, SURFACE_KEYS, "river_processing.water_surface")
    wb = cfg["water_body"]
    require_keys(
        wb,
        ("min_width_m", "sea_class_fraction"),
        "river_processing.water_surface.water_body",
    )
    downstream = build_downstream_adjacency(rivers)
    rivers, _ = orient_downstream(rivers, downstream)
    spacing = float(cfg["node_spacing_m"])
    half = float(cfg["median_half_window_m"])

    net = build_node_network(rivers, downstream, spacing)
    samples, skipped = _split_samples(sample(net.reach_line))
    raw = node_medians(net, samples, half)
    structures, rising = detect_structure_edges(
        rivers,
        downstream,
        _reach_levels(net, raw),
        step_threshold_m=float(cfg["step_threshold_m"]),
        structure_min_drop_m=float(cfg["structure_min_drop_m"]),
        manual=cfg["structures"] or [],
        use_sword_obstructions=bool(cfg["use_sword_obstructions"]),
    )
    for u, d, rise in rising:
        log.warning(f"Water surface: level rises {rise:.2f} m from reach {u} to {d}")
    log.info(
        f"Water surface: {len(structures)} structure junction(s) and {len(rising)} rising junction(s), "
        "smoothed across"
    )

    width = node_widths_from_reaches(net, rivers)
    if sample_nodes is None:
        dem, sea_fraction = np.full(net.n_nodes, np.nan), np.zeros(net.n_nodes)
    else:
        dem, sea_fraction = (
            np.asarray(v, float) for v in sample_nodes(net.node_x, net.node_y, width)
        )
    node_class = classify_nodes(
        width,
        sea_fraction,
        dem,
        gebco_floor_m,
        min_width_m=float(wb["min_width_m"]),
        sea_class_fraction=float(wb["sea_class_fraction"]),
    )

    unset = ~np.isfinite(raw)
    if unset.any() and np.isfinite(raw).any():
        fill = float(np.nanmin(raw)) if sea_level_m is None else float(sea_level_m)
        raw[unset] = fill
        log.warning(
            f"Water surface: {int(unset.sum())} node(s) without water pixels set to {fill:.2f} m"
        )

    fixed = np.zeros(net.n_nodes, dtype=bool)
    mouths = np.array([], dtype=int)
    target = raw.copy()
    if sea_level_m is not None:
        mouths = sea_mouths(
            net,
            raw,
            node_class,
            _reach_flag(rivers, "is_delta_outflow"),
            float(sea_level_m),
            float(cfg["mouth_level_tol_m"]),
        )
        if not mouths.size:
            raise ValueError(
                "water surface: no sea mouth in the network -- nothing pins the level"
            )
        fixed[mouths] = True
        below = int((target < sea_level_m).sum())
        target = np.maximum(target, sea_level_m)
        target[mouths] = sea_level_m
        log.info(
            f"Water surface: {mouths.size} sea mouth(s) pinned at {sea_level_m:+.2f} m; "
            f"{below} of {net.n_nodes} node median(s) below it raised to it"
        )
    edges = net.node_edges()
    isotonic = enforce_non_increasing(net, target, fixed=fixed)

    def smoothed(length_m: float) -> np.ndarray:
        out = smooth_on_network(isotonic, edges, length_m, spacing, fixed=fixed)
        out = enforce_non_increasing(net, out, fixed=fixed)
        return out if sea_level_m is None else np.maximum(out, sea_level_m)

    level = smoothed(float(cfg["smoothing_m"]))
    slope_level = smoothed(float(cfg["slope_smoothing_m"]))
    slope = main_path_slope(net, slope_level, _reach_discharge(rivers))
    swot = swot_check(net, rivers, slope_level)
    log.info(
        f"Water surface: {net.n_nodes} nodes on {len(net.reach_ids)} reaches, "
        f"level {np.nanmin(level):.2f}..{np.nanmax(level):.2f} m, "
        f"|smoothed - median| p50/p90 {np.nanpercentile(np.abs(level - raw), 50):.2f}/"
        f"{np.nanpercentile(np.abs(level - raw), 90):.2f} m; slope p5/50/95 "
        f"{np.round(np.percentile(slope, [5, 50, 95]), 7).tolist()}; classes "
        f"{pd.Series(node_class).value_counts().to_dict()}"
    )
    if len(swot):
        agree = np.mean(
            (swot["ratio"] >= 1 / SWOT_RATIO_SPAN) & (swot["ratio"] <= SWOT_RATIO_SPAN)
        )
        log.info(
            f"Water surface SWOT check: {len(swot)} reach(es) with a reliable SWOT slope, surface/SWOT "
            f"median {swot['ratio'].median():.2f}, p10/p90 {swot['ratio'].quantile(0.1):.2f}/"
            f"{swot['ratio'].quantile(0.9):.2f}, {agree:.0%} within x{SWOT_RATIO_SPAN:g}"
        )
        for row in swot.itertuples():
            log.info(
                f"  SWOT reach {row.reach_id}: SWOT {row.swot_slope:.2e}, surface {row.surface_slope:.2e}, "
                f"ratio {row.ratio:.2f}"
            )
    return WaterSurface(
        net=net,
        level_m=level,
        level_raw_m=raw,
        slope=slope,
        rising_edges=rising,
        samples=samples,
        skipped=skipped,
        structures=set(structures),
        mouths=mouths,
        node_class=node_class,
        dem_m=dem,
        width_m=width,
        swot=swot,
    ), rivers


def node_sampler(dem_path: str | Path, landuse_path: str | Path, crs):
    """Return (x, y, width_m) -> (DEM, sea-class fraction of a disc of the channel width) per node."""
    from pyproj import Transformer

    offs = np.linspace(-0.5, 0.5, 5)
    disc = [(a, b) for a in offs for b in offs if a * a + b * b <= 0.25]

    def sample(x, y, width_m):
        with rasterio.open(dem_path) as src:
            to_src = Transformer.from_crs(crs, src.crs, always_xy=True)
            xs, ys = to_src.transform(np.asarray(x), np.asarray(y))
            dem = np.array([v[0] for v in src.sample(zip(xs, ys))], dtype=float)
            if src.nodata is not None:
                dem[dem == src.nodata] = np.nan
        sea = np.zeros(len(x))
        with rasterio.open(landuse_path) as src:
            to_src = Transformer.from_crs(crs, src.crs, always_xy=True)
            for k, (xk, yk, wk) in enumerate(zip(x, y, width_m)):
                wk = float(wk) if np.isfinite(wk) else 0.0
                px, py = to_src.transform(
                    [xk + a * wk for a, _ in disc], [yk + b * wk for _, b in disc]
                )
                sea[k] = np.mean(
                    np.array([v[0] for v in src.sample(zip(px, py))])
                    == LANDUSE_SEA_CODE
                )
        return dem, sea

    return sample


def surface_table(surface: WaterSurface, crs) -> gpd.GeoDataFrame:
    """Return one row per (reach, node) with the levels, slope and node attributes."""
    net = surface.net
    mouth = np.zeros(net.n_nodes, dtype=bool)
    mouth[surface.mouths] = True
    swot = surface.swot.set_index("reach_id")
    rows = [
        {
            "reach_id": r,
            "node_id": int(nid),
            "chainage_m": float(ch),
            "level_m": float(surface.level_m[nid]),
            "level_raw_m": float(surface.level_raw_m[nid]),
            "slope": float(surface.slope[nid]),
            "node_class": str(surface.node_class[nid]),
            "dem_m": float(surface.dem_m[nid]),
            "width_m": float(surface.width_m[nid]),
            "is_mouth": bool(mouth[nid]),
            "swot_slope": float(swot["swot_slope"].get(r, np.nan)),
            "swot_ratio": float(swot["ratio"].get(r, np.nan)),
            "geometry": Point(float(net.node_x[nid]), float(net.node_y[nid])),
        }
        for r in net.reach_ids
        for nid, ch in zip(net.reach_nodes[r], net.reach_chainage[r])
    ]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


def write_water_surface(surface: WaterSurface, path: str | Path, crs) -> None:
    """Write the surface table and the detected structure edges as two GPKG layers."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    surface_table(surface, crs).to_file(path, layer=SURFACE_LAYER, driver="GPKG")
    pyogrio.write_dataframe(
        pd.DataFrame(
            sorted(surface.structures), columns=["reach_up", "reach_dn"], dtype=str
        ),
        path,
        layer=STRUCTURE_LAYER,
        driver="GPKG",
    )


def read_water_surface(
    path: str | Path,
) -> tuple[gpd.GeoDataFrame, set[tuple[str, str]]]:
    """Return the per-(reach, node) surface table and the detected structure edges of rule 10."""
    nodes = gpd.read_file(path, layer=SURFACE_LAYER)
    nodes["reach_id"] = nodes["reach_id"].map(normalize_reach_id)
    st = pyogrio.read_dataframe(path, layer=STRUCTURE_LAYER, read_geometry=False)
    return nodes, {
        (normalize_reach_id(u), normalize_reach_id(d))
        for u, d in zip(st["reach_up"], st["reach_dn"])
    }


def condition_channels_to_surface(
    surface: WaterSurface,
    rivers: gpd.GeoDataFrame,
    elevation_path: str | Path,
    landuse_path: str | Path,
    output_path: str | Path,
    width_column: str = "width",
) -> dict[str, int]:
    """Write the DEM with every land pixel of each channel buffer set to the surface.

    Buffers are ``width / 2`` around the downstream-running reach lines; the
    level is interpolated by chainage along the reach. Sea pixels (GEBCO
    bathymetry) and nodata are left unchanged; where buffers overlap the
    lower level wins. A reach without a positive width sets its centreline
    pixels only.
    """
    with rasterio.open(elevation_path) as src, rasterio.open(landuse_path) as lu_src:
        dem = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform, nodata = src.transform, src.nodata
        sea = lu_src.read(1) == LANDUSE_SEA_CODE
    valid = ~sea & (
        ~np.isclose(dem, np.float32(nodata)) if nodata is not None else True
    )
    out = dem.copy()
    touched = np.zeros(dem.shape, dtype=bool)
    width = dict(
        zip(
            rivers["reach_id"].map(normalize_reach_id),
            pd.to_numeric(rivers[width_column], errors="coerce"),
        )
    )
    step_m = float(np.hypot(transform.a, transform.d))
    net = surface.net
    n_set = 0
    for r in net.reach_ids:
        line = net.reach_line[r]
        interp = clamped_profile(
            net.reach_chainage[r], surface.level_m[net.reach_nodes[r]]
        )
        w = width.get(r, np.nan)
        if np.isfinite(w) and w > 0:
            cells = line_window_cells(
                line.buffer(float(w) / 2.0),
                line,
                transform,
                dem.shape,
                along_step_m=step_m,
            )
            if cells is None:
                continue
            rr, cc, along = cells
        else:
            cl = _sample_line_cells(line, transform, dem.shape, step_m)
            if not cl:
                continue
            rr = np.array([c["row"] for c in cl])
            cc = np.array([c["col"] for c in cl])
            along = np.array([c["along_m"] for c in cl])
        keep = valid[rr, cc]
        rr, cc = rr[keep], cc[keep]
        target = np.asarray(interp(np.asarray(along)[keep]), dtype=np.float32)
        finite = np.isfinite(target)
        rr, cc, target = rr[finite], cc[finite], target[finite]
        new = np.where(touched[rr, cc], np.minimum(out[rr, cc], target), target)
        out[rr, cc] = new
        touched[rr, cc] = True
        n_set += int(keep.sum())
    changed = touched & ~np.isclose(out, dem)
    log.info(
        f"Channels conditioned to the water surface: {int(touched.sum()):,} land pixel(s) in channel buffers, "
        f"{int(changed.sum()):,} changed (median |change| {np.median(np.abs(out[changed] - dem[changed])) if changed.any() else 0:.2f} m)"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out, 1)
    return {
        "pixels_in_buffers": int(touched.sum()),
        "pixels_changed": int(changed.sum()),
        "pixel_visits": n_set,
    }


def river_cells_at_surface(
    ini: np.ndarray,
    nodes: gpd.GeoDataFrame,
    channel_mask: np.ndarray,
    transform,
) -> tuple[np.ndarray, int]:
    """Return the initial water level with dry channel cells set to their nearest node's ``level_m``.

    Rules 11 and 12a pass the steady mean-discharge level on rule 11's bed.
    Cells ``ini`` already holds (the sea) keep their value; ``nodes`` must be in
    the grid's CRS.
    """
    rows, cols = np.nonzero(channel_mask & ~np.isfinite(ini))
    out = ini.copy()
    if rows.size == 0 or nodes.empty:
        return out, 0
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    tree = cKDTree(np.column_stack([nodes.geometry.x, nodes.geometry.y]))
    _, nearest = tree.query(np.column_stack([xs, ys]))
    out[rows, cols] = nodes["level_m"].to_numpy(dtype=np.float32)[nearest]
    return out, int(rows.size)


def raise_wet_cells_to(
    ini: np.ndarray, wet_mask: np.ndarray, level_m: float
) -> tuple[np.ndarray, int]:
    """Return the initial water level with wet_mask cells below level_m raised to it.

    NaN cells (dry, no initial level) stay NaN.
    """
    raise_ = wet_mask & np.isfinite(ini) & (ini < level_m)
    out = ini.copy()
    out[raise_] = np.float32(level_m)
    return out, int(raise_.sum())
