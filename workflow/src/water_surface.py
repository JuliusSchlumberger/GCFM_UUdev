"""Smoothed river water surface from FathomDEM, and the DEM conditioned to it.

Entry points: :func:`build_water_surface` and :func:`condition_channels_to_surface`
(rule enforce_river_monotonicity, 09); :func:`read_water_surface` (rule 10, GVF
target).

FathomDEM's channels hold a flattened water surface: flat within a water
body, dropping in steps between them, with stray peaks (bridges, bars, bank
pixels) and dips. The surface is built on the GVF node network
(src.gvf_target.build_node_network):

1. median of the water pixels (land use == ``landuse.water_code``) along the
   centreline within ``median_half_window_m`` of each node -- drops peaks and
   dips alike. Islands and banks the centreline crosses are skipped, so their
   nodes take the level of the water around them;
2. structures detected where the reach level drops by more than
   ``step_threshold_m`` (or ``structure_min_drop_m`` at a SWORD obstruction
   reach; src.gvf_target.detect_structure_edges). They split the network only
   with ``split_at_structures``; by default the surface is smoothed across
   them, because weirs and locks open in floods and a mean-flow level drop
   would otherwise become a step in the flood bed. Rises of that size are
   inconsistent data and are cut (``cut_rising_edges``);
3. non-increasing downstream (network isotonic regression), smoothed over
   ``smoothing_m`` (never across a cut), and made non-increasing again.

At network scale this surface has the same gradient as SWOT's median levels
(1248635: 1.71 vs 1.80 cm/km).
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
from shapely.geometry import Point
from src.gvf_target import (
    GvfNetwork,
    _fill_from_neighbours,
    build_node_network,
    detect_structure_edges,
    enforce_non_increasing,
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
    "step_threshold_m",
    "structure_min_drop_m",
    "use_sword_obstructions",
    "split_at_structures",
    "cut_rising_edges",
    "structures",
)


@dataclass
class WaterSurface:
    """Node network with the raw (median) and smoothed water level per node (m)."""

    net: GvfNetwork
    level_m: np.ndarray
    level_raw_m: np.ndarray
    rising_edges: list[tuple[str, str, float]]
    samples: dict[
        str, tuple[np.ndarray, np.ndarray]
    ]  # {reach: (chainage_m, elevation_m)} of the pixels the medians use
    skipped: dict[
        str, tuple[np.ndarray, np.ndarray]
    ]  # {reach: (chainage_m, elevation_m)} of the other centreline pixels
    structures: set[
        tuple[str, str]
    ]  # detected; network split only if split_at_structures


def sample_centrelines(
    lines: dict[str, object],
    dem_path: str | Path,
    landuse_path: str | Path,
    water_code: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {reach: (chainage_m, elevation_m, is_water)} of the pixels along each line.

    is_water marks the pixels the water surface is built from; the others
    (islands, banks, sea) are returned for the rule 09 figure.

    The DEM and the land-use raster must share one grid (rule 05b writes
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


def build_water_surface(
    rivers: gpd.GeoDataFrame, sample, cfg: dict, sea_level_m: float | None = None
) -> tuple[WaterSurface, gpd.GeoDataFrame]:
    """Build the smoothed water surface on the node network.

    Args:
        rivers: Cleaned network in the DEM's metric CRS ('reach_id',
            'rch_id_dn', optionally 'obstr_type').
        sample: Callable {reach: LineString} -> {reach: (chainage_m,
            elevation_m[, use])}, e.g. :func:`sample_centrelines` bound to the
            rasters; ``use`` marks the pixels the medians take (default all).
        cfg: ``river_processing.water_surface``.
        sea_level_m: Level of nodes without any land pixel (all-sea reaches cut
            off by a structure), in the model datum; None: the lowest node level.

    Returns:
        The surface and the reaches with downstream-running lines.
    """
    require_keys(cfg, SURFACE_KEYS, "river_processing.water_surface")
    downstream = build_downstream_adjacency(rivers)
    rivers, _ = orient_downstream(rivers, downstream)
    spacing = float(cfg["node_spacing_m"])
    half = float(cfg["median_half_window_m"])

    provisional = build_node_network(rivers, downstream, spacing)
    samples, skipped = _split_samples(sample(provisional.reach_line))
    structures, rising = detect_structure_edges(
        rivers,
        downstream,
        _reach_levels(provisional, node_medians(provisional, samples, half)),
        step_threshold_m=float(cfg["step_threshold_m"]),
        structure_min_drop_m=float(cfg["structure_min_drop_m"]),
        manual=cfg["structures"] or [],
        use_sword_obstructions=bool(cfg["use_sword_obstructions"]),
    )
    for u, d, rise in rising:
        log.warning(f"Water surface: level rises {rise:.2f} m from reach {u} to {d}")
    split = set(structures) if bool(cfg["split_at_structures"]) else set()
    if rising and bool(cfg["cut_rising_edges"]):
        split |= {(u, d) for u, d, _ in rising}
    log.info(
        f"Water surface: {len(structures)} structure junction(s) "
        f"({'split' if cfg['split_at_structures'] else 'smoothed across'}), {len(rising)} rising"
    )

    net = build_node_network(rivers, downstream, spacing, split)
    raw = node_medians(net, samples, half)
    unset = ~np.isfinite(raw)
    if unset.any() and np.isfinite(raw).any():
        fill = float(np.nanmin(raw)) if sea_level_m is None else float(sea_level_m)
        raw[unset] = fill
        log.warning(
            f"Water surface: {int(unset.sum())} node(s) without water pixels set to {fill:.2f} m"
        )
    edges = net.node_edges()
    level = enforce_non_increasing(net, raw)
    level = smooth_on_network(level, edges, float(cfg["smoothing_m"]), spacing)
    level = enforce_non_increasing(net, level)
    log.info(
        f"Water surface: {net.n_nodes} nodes on {len(net.reach_ids)} reaches, "
        f"level {np.nanmin(level):.2f}..{np.nanmax(level):.2f} m, "
        f"|smoothed - median| p50/p90 {np.nanpercentile(np.abs(level - raw), 50):.2f}/"
        f"{np.nanpercentile(np.abs(level - raw), 90):.2f} m"
    )
    return WaterSurface(
        net=net,
        level_m=level,
        level_raw_m=raw,
        rising_edges=rising,
        samples=samples,
        skipped=skipped,
        structures=set(structures),
    ), rivers


def surface_table(surface: WaterSurface, crs) -> gpd.GeoDataFrame:
    """Return one row per (reach, node): reach_id, node_id, chainage_m, level_m, level_raw_m."""
    net = surface.net
    rows = [
        {
            "reach_id": r,
            "node_id": int(nid),
            "chainage_m": float(ch),
            "level_m": float(surface.level_m[nid]),
            "level_raw_m": float(surface.level_raw_m[nid]),
            "geometry": Point(float(net.node_x[nid]), float(net.node_y[nid])),
        }
        for r in net.reach_ids
        for nid, ch in zip(net.reach_nodes[r], net.reach_chainage[r])
    ]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


def write_water_surface(surface: WaterSurface, path: str | Path, crs) -> None:
    """Write the surface table and the structure edges as two GPKG layers."""
    net = surface.net
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    surface_table(surface, crs).to_file(path, layer=SURFACE_LAYER, driver="GPKG")
    pyogrio.write_dataframe(
        pd.DataFrame(
            sorted(net.structure_edges), columns=["reach_up", "reach_dn"], dtype=str
        ),
        path,
        layer=STRUCTURE_LAYER,
        driver="GPKG",
    )


def read_water_surface(
    path: str | Path,
) -> tuple[gpd.GeoDataFrame, set[tuple[str, str]]]:
    """Return the per-node surface table and the structure edges written by rule 09."""
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
