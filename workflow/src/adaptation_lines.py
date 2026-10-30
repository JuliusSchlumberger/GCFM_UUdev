"""Derive adaptation-measure lines (levees, mouth closure, pump) from the baseline weir.

Entry point: rule build_adaptation_lines (12c), once per basin after the
skeleton; config/adaptation_strategies.yml references its geojsons as
`locations: auto/<name>`. The lines follow the skeleton's baseline weir, which
runs along the grid's cell edges, and src.protection_weir.
merge_weir_preserve_unmatched only replaces baseline segments within 2 m of a
`locations` line.

Steps
-----
1. Label every baseline weir segment by the water it faces (label_weir_kinds):
   "coast" when a grid cell on either side is open sea (landuse 200, plus
   sea-linked wetlands when those are left unprotected), else "river" when
   one is river channel, burned without the network's seaward tail
   (centrelines_off_sea) so SWORD running on past the coastline does not
   turn coastal weir into river. A segment with land on both sides faces a
   small land island the weir builder merged into the water (fewer than
   min_component_cells cells) and takes the kind of the water that island
   borders; failing that, the kind of the nearest water cell.
2. Close each river mouth (mouth_closures): for every SWORD outlet reach
   (n_rch_dn == 0), the closure is the narrowest straight crossing through
   the reach's upstream end (where the channel meets the sea) that reaches
   land on both sides, within max_tilt_deg of perpendicular to the river,
   snapped onto the grid's cell edges and extended ext_cells into land so
   it joins the coastal weir.
3. With the baseline weir and the closures as barriers, sea-classified
   cells no longer connected to the open sea form the estuary; coast
   segments facing it become "estuary" (sheltered: they stay at baseline
   under protect-closed). The same connectivity check confirms the river
   behind every closure is cut off from the open sea.
4. The pump line per mouth is the river centreline from pump_upstream_m
   upstream of the closure to the outlet reach's seaward end, oriented
   inland -> sea (hydromt_sfincs keeps only its two end points: first =
   intake, last = outlet).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from scipy.ndimage import distance_transform_edt, generate_binary_structure
from scipy.ndimage import label as ndimage_label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, substring, unary_union
from src.adaptation_strategies import ADAPTATION_LINE_NAMES
from src.io import require_keys
from src.landuse import LANDUSE_SEA_CODE
from src.plots import save_figure
from src.protection_weir import GridArrays, ocean_linked_wetland_mask
from src.river_burn import build_channel_mask_regular

log = logging.getLogger(__name__)

KIND_COAST = "coast"
KIND_RIVER = "river"
KIND_ESTUARY = "estuary"
KIND_COLOURS = {KIND_COAST: "black", KIND_ESTUARY: "0.6", KIND_RIVER: "purple"}

_STRUCT8 = generate_binary_structure(2, 2)
_NEIGHBOURS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _edge_cells(seg: LineString, inv_transform) -> tuple[tuple, tuple]:
    """`seg` is a 2-point segment along a cell edge, as the weir builder writes them."""
    (x0, y0), (x1, y1) = seg.coords[0][:2], seg.coords[-1][:2]
    c0, r0 = (round(v) for v in inv_transform * (x0, y0))
    c1, r1 = (round(v) for v in inv_transform * (x1, y1))
    if r0 == r1:  # along a row line: cells above and below
        c = min(c0, c1)
        return (r0 - 1, c), (r0, c)
    r = min(r0, r1)  # along a column line: cells left and right
    return (r, c0 - 1), (r, c0)


def _edge_key(a: tuple, b: tuple) -> tuple:
    return (a, b) if a <= b else (b, a)


def _in_grid(rc: tuple, shape: tuple) -> bool:
    return 0 <= rc[0] < shape[0] and 0 <= rc[1] < shape[1]


def _components(valid: np.ndarray, blocked: set) -> np.ndarray:
    """4-neighbour links only (SFINCS exchanges water across shared cell edges);
    never across a blocked (weir) edge. Label -1 = invalid cell.
    """
    h, w = valid.shape
    idx = np.arange(h * w).reshape(h, w)
    rows_i, rows_j = [], []
    for dr, dc in ((1, 0), (0, 1)):
        ok = valid[: h - dr, : w - dc] & valid[dr:, dc:]
        for r, c in zip(*np.where(ok)):
            if _edge_key((r, c), (r + dr, c + dc)) in blocked:
                continue
            rows_i.append(idx[r, c])
            rows_j.append(idx[r + dr, c + dc])
    graph = coo_matrix(
        (np.ones(len(rows_i), dtype=np.int8), (rows_i, rows_j)), shape=(h * w, h * w)
    )
    _, labels = connected_components(graph, directed=False)
    labels = labels.reshape(h, w)
    labels[~valid] = -1
    return labels


def _weir_edges(weir_gdf: gpd.GeoDataFrame, transform) -> set:
    inv = ~transform
    return {_edge_key(*_edge_cells(g, inv)) for g in weir_gdf.geometry}


def _open_sea_label(labels: np.ndarray, seaward: np.ndarray, valid: np.ndarray) -> int:
    """The component holding the most sea cells = the open sea."""
    return int(np.argmax(np.bincount(labels[seaward & valid] + 1))) - 1


def protected_land(
    weir_gdf: gpd.GeoDataFrame, seaward: np.ndarray, valid: np.ndarray, transform
) -> np.ndarray:
    """Return the cells the baseline weir keeps apart from the open sea.

    This is land in the hydraulic sense the closure has to reach: a
    land-classified cell with no weir between it and the sea (a small merged
    island, or a dropped tiny dikering) floods from the sea and does not count.
    """
    labels = _components(valid, _weir_edges(weir_gdf, transform))
    return valid & (labels != _open_sea_label(labels, seaward, valid))


def centrelines_off_sea(
    rivers: gpd.GeoDataFrame, seaward: np.ndarray, transform
) -> gpd.GeoDataFrame:
    """Return the river network without its seaward tail.

    From every outlet reach (n_rch_dn == 0) the downstream-end part running
    through sea cells is dropped; a reach lying wholly in the sea passes the cut
    on to its upstream reaches. SWORD often carries a river on past the land-use
    coastline (ghost reaches, a channel hugging the shore), and burning that tail
    as channel would label the coastal weir beside it "river". Sea cells
    mid-river (land use classing part of a wide channel as sea) are left alone.
    """
    inv = ~transform
    step = float(np.hypot(transform.a, transform.d)) / 4
    ids = rivers["reach_id"].astype("int64").to_numpy()
    by_id = dict(zip(ids, range(len(rivers))))

    def _line(geom) -> LineString:
        if geom.geom_type == "MultiLineString":
            geom = linemerge(geom)
        if geom.geom_type != "LineString":
            geom = max(geom.geoms, key=lambda g: g.length)
        return geom

    def _in_sea(p) -> bool:
        c, r = inv * (p.x, p.y)
        rc = (int(np.floor(r)), int(np.floor(c)))
        return _in_grid(rc, seaward.shape) and bool(seaward[rc])

    geoms = list(rivers.geometry)
    stack = [(rid, None) for rid in ids[(rivers["n_rch_dn"] == 0).to_numpy()]]
    done = set()
    while stack:
        rid, dn_line = stack.pop()
        if rid in done or rid not in by_id:
            continue
        done.add(rid)
        i = by_id[rid]
        line = _line(rivers.geometry.iloc[i])
        ups = [u for u in _reach_ids(rivers["rch_id_up"].iloc[i]) if u in by_id]
        start, end = Point(line.coords[0]), Point(line.coords[-1])
        if dn_line is not None:  # downstream end = the one touching the reach below
            flip = start.distance(dn_line) < end.distance(dn_line)
        elif ups:  # outlet: downstream end = the one away from the reaches above
            up = unary_union([rivers.geometry.iloc[by_id[u]] for u in ups])
            flip = end.distance(up) < start.distance(up)
        else:
            flip = False
        if flip:
            line = LineString(line.coords[::-1])
        keep = line.length
        while keep > 0 and _in_sea(line.interpolate(keep)):
            keep -= step
        if keep >= line.length:
            continue
        if keep <= 0:
            geoms[i] = None
            stack.extend((u, line) for u in ups)
        else:
            geoms[i] = substring(line, 0.0, keep)
    out = rivers.copy()
    out["geometry"] = geoms
    return out[out.geometry.notna()]


def label_weir_kinds(
    weir_gdf: gpd.GeoDataFrame,
    seaward: np.ndarray,
    channel: np.ndarray,
    valid: np.ndarray,
    transform,
    min_component_cells: int,
) -> tuple[list, dict]:
    """Label each baseline weir segment "coast"/"river" (module docstring, step 1).

    Returns (kinds, counts).
    """
    inv = ~transform
    shape = seaward.shape
    water = seaward | channel
    land_labels, n_land = ndimage_label(valid & ~water, structure=_STRUCT8)
    land_sizes = np.bincount(land_labels.ravel(), minlength=n_land + 1)
    island_kind: dict[int, str | None] = {}

    def _island_kind(lab: int) -> str | None:
        # majority of the island's 4-neighbours that are sea vs channel; ties -> coast
        if lab not in island_kind:
            cells = np.argwhere(land_labels == lab)
            n_sea = n_chan = 0
            for r, c in cells:
                for dr, dc in _NEIGHBOURS4:
                    rr, cc = r + dr, c + dc
                    if _in_grid((rr, cc), shape):
                        n_sea += bool(seaward[rr, cc])
                        n_chan += bool(channel[rr, cc] and not seaward[rr, cc])
            island_kind[lab] = (
                None
                if n_sea == n_chan == 0
                else KIND_COAST
                if n_sea >= n_chan
                else KIND_RIVER
            )
        return island_kind[lab]

    # Land on both sides away from a small island (e.g. a shoreline weir whose
    # outer cell is a land strip the sea-clipped channel mask no longer covers):
    # the kind of the nearest water cell.
    dist_sea = distance_transform_edt(~seaward) if seaward.any() else None
    dist_chan = distance_transform_edt(~channel) if channel.any() else None

    def _nearest_water_kind(sides: list) -> str | None:
        d_sea = min(dist_sea[rc] for rc in sides) if dist_sea is not None else np.inf
        d_chan = min(dist_chan[rc] for rc in sides) if dist_chan is not None else np.inf
        if np.isinf(d_sea) and np.isinf(d_chan):
            return None
        return KIND_COAST if d_sea <= d_chan else KIND_RIVER

    kinds, n_island, n_nearest, n_unresolved = [], 0, 0, 0
    for seg in weir_gdf.geometry:
        sides = [rc for rc in _edge_cells(seg, inv) if _in_grid(rc, shape)]
        if any(seaward[rc] for rc in sides):
            kinds.append(KIND_COAST)
        elif any(channel[rc] for rc in sides):
            kinds.append(KIND_RIVER)
        else:
            n_island += 1
            small = [
                land_labels[rc]
                for rc in sides
                if land_labels[rc] > 0
                and land_sizes[land_labels[rc]] < min_component_cells
            ]
            kind = _island_kind(small[0]) if small else None
            if kind is None:
                kind = _nearest_water_kind(sides)
                n_nearest += kind is not None
            if kind is None:
                n_unresolved += 1
                kind = KIND_RIVER
            kinds.append(kind)
    counts = {
        KIND_COAST: kinds.count(KIND_COAST),
        KIND_RIVER: kinds.count(KIND_RIVER),
        "facing_small_island": n_island,
        "by_nearest_water": n_nearest,
        "unresolved_as_river": n_unresolved,
    }
    return kinds, counts


def _reach_ids(value) -> list[int]:
    """SWORD rch_id_up/rch_id_dn as stored in the gpkg (e.g. "[21604100013]")."""
    return [int(x) for x in re.findall(r"\d+", str(value)) if int(x) != 0]


def _outlet_centreline(
    rivers: gpd.GeoDataFrame, outlet
) -> tuple[LineString, Point, float]:
    """Return (upstream + outlet reach line oriented inland -> sea, anchor, anchor distance).

    The anchor is the outlet reach's upstream end.
    """
    up = rivers[
        rivers["reach_id"].astype("int64").isin(_reach_ids(outlet["rch_id_up"]))
    ]
    if up.empty:
        raise ValueError(f"outlet reach {outlet['reach_id']} has no upstream reach")
    up_line = unary_union(list(up.geometry))
    if up_line.geom_type == "MultiLineString":
        up_line = linemerge(up_line)
    if (
        up_line.geom_type != "LineString"
    ):  # confluence right at the mouth: longest branch
        up_line = max(up_line.geoms, key=lambda g: g.length)
    out_line = outlet.geometry
    if out_line.geom_type == "MultiLineString":
        out_line = linemerge(out_line)
    if Point(out_line.coords[-1]).distance(up_line) < Point(
        out_line.coords[0]
    ).distance(up_line):
        out_line = LineString(out_line.coords[::-1])
    anchor = Point(out_line.coords[0])
    if Point(up_line.coords[0]).distance(anchor) < Point(up_line.coords[-1]).distance(
        anchor
    ):
        up_line = LineString(up_line.coords[::-1])
    full = LineString(list(up_line.coords) + list(out_line.coords)[1:])
    return full, anchor, up_line.length


def _first_land_point(
    anchor, direction, land, inv_transform, max_len, step_m
) -> Point | None:
    """Sampled every step_m, so grazing a land cell's corner still counts."""
    shape = land.shape
    for s in np.arange(0.0, max_len, step_m):
        x, y = anchor.x + direction[0] * s, anchor.y + direction[1] * s
        c, r = inv_transform * (x, y)
        rc = (int(np.floor(r)), int(np.floor(c)))
        if not _in_grid(rc, shape):
            return None
        if land[rc]:
            return Point(x, y)
    return None


def _crossing_edges(p0, p1, transform, shape, ext_m) -> tuple[list[tuple], float]:
    """Return grid edges whose 4-neighbour cells straddle the line p0-p1 (extended by ext_m).

    Returns ([(cell_a, cell_b, along-line edge midpoint position)], segment
    length); a position < 0 or > length lies in the extension.
    """
    h, w = shape
    rows, cols = np.mgrid[0:h, 0:w]
    cx, cy = transform * (cols + 0.5, rows + 0.5)
    u = np.array(p1) - np.array(p0)
    length = float(np.hypot(*u))
    u = u / length
    side = (cx - p0[0]) * u[1] - (cy - p0[1]) * u[0]
    along = (cx - p0[0]) * u[0] + (cy - p0[1]) * u[1]
    out = []
    for dr, dc in ((1, 0), (0, 1)):
        s1, s2 = side[: h - dr, : w - dc], side[dr:, dc:]
        am = 0.5 * (along[: h - dr, : w - dc] + along[dr:, dc:])
        hit = (np.sign(s1) != np.sign(s2)) & (am >= -ext_m) & (am <= length + ext_m)
        for r, c in zip(*np.where(hit)):
            out.append(((int(r), int(c)), (int(r + dr), int(c + dc)), float(am[r, c])))
    return out, length


def _edge_line(a: tuple, b: tuple, transform) -> LineString:
    (r, c), (r2, c2) = a, b
    if r2 != r:  # vertical neighbours: shared edge along the row line max(r, r2)
        rl = max(r, r2)
        return LineString([transform * (c, rl), transform * (c + 1, rl)])
    cl = max(c, c2)
    return LineString([transform * (cl, r), transform * (cl, r + 1)])


CLOSURE_KEYS = (
    "direction_length_m",
    "max_tilt_deg",
    "max_half_length_m",
    "ext_cells",
    "pump_upstream_m",
)


def mouth_closures(
    rivers: gpd.GeoDataFrame,
    land: np.ndarray,
    transform,
    *,
    direction_length_m: float,
    max_tilt_deg: float,
    max_half_length_m: float,
    ext_cells: int,
    pump_upstream_m: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Build closure edges and pump line per SWORD outlet reach (module docstring, steps 2, 4).

    `land` = protected_land(). Returns (closure_rows, pump_rows, report);
    closure rows carry cell_a/cell_b for the connectivity check.
    """
    inv = ~transform
    cell = float(np.hypot(transform.a, transform.d))
    closures, pumps, report = [], [], []
    for _, outlet in rivers[rivers["n_rch_dn"] == 0].iterrows():
        reach_id = int(outlet["reach_id"])
        full, anchor, d_anchor = _outlet_centreline(rivers, outlet)
        p_up = full.interpolate(max(d_anchor - direction_length_m, 0.0))
        t = np.array([anchor.x - p_up.x, anchor.y - p_up.y])
        t /= np.hypot(*t)
        perp = np.arctan2(-t[1], t[0])  # bearing (from north) of the left-hand normal
        best = None
        for off in np.radians(np.arange(-max_tilt_deg, max_tilt_deg + 0.5, 1.0)):
            v = np.array([np.sin(perp + off), np.cos(perp + off)])
            ends = [
                _first_land_point(
                    anchor, s * v, land, inv, max_half_length_m, cell / 50
                )
                for s in (1, -1)
            ]
            if any(e is None for e in ends):
                continue
            total = anchor.distance(ends[0]) + anchor.distance(ends[1])
            if best is None or total < best[0]:
                best = (total, ends, float(np.degrees(off)))
        if best is None:
            log.warning(
                f"Outlet reach {reach_id}: no straight crossing within +-{max_tilt_deg:.0f} deg of "
                f"perpendicular reaches land on both sides within {max_half_length_m:.0f} m -- "
                "no closure or pump written for this mouth"
            )
            report.append({"reach_id": reach_id, "closed": False})
            continue
        _, ends, tilt = best
        p0, p1 = ends[1].coords[0], ends[0].coords[0]
        edges, length = _crossing_edges(p0, p1, transform, land.shape, ext_cells * cell)
        n_kept = 0
        for a, b, pos in edges:
            if not (_in_grid(a, land.shape) and _in_grid(b, land.shape)):
                continue
            in_extension = pos < 0 or pos > length
            if in_extension and not land[a] and not land[b]:
                continue  # extension reaching past the land back into open water
            closures.append(
                {
                    "reach_id": reach_id,
                    "cell_a": a,
                    "cell_b": b,
                    "geometry": _edge_line(a, b, transform),
                }
            )
            n_kept += 1
        pump = substring(full, max(d_anchor - pump_upstream_m, 0.0), full.length)
        pumps.append({"reach_id": reach_id, "geometry": pump})
        report.append(
            {
                "reach_id": reach_id,
                "closed": True,
                "anchor": [round(anchor.x, 1), round(anchor.y, 1)],
                "crossing_m": round(length, 1),
                "tilt_deg_from_perpendicular": round(tilt, 1),
                "closure_edges": n_kept,
                "pump_length_m": round(pump.length, 1),
                "river_check_point": list(
                    full.interpolate(max(d_anchor - 3 * cell, 0.0)).coords[0]
                ),
            }
        )
        log.info(
            f"Outlet reach {reach_id}: closure {length:.0f} m across the mouth at "
            f"({anchor.x:.0f}, {anchor.y:.0f}), {tilt:+.0f} deg from perpendicular, "
            f"{n_kept} grid edge(s); pump line {pump.length:.0f} m"
        )
    return closures, pumps, report


def estuary_and_seal(
    weir_gdf: gpd.GeoDataFrame,
    closures: list[dict],
    report: list[dict],
    seaward: np.ndarray,
    valid: np.ndarray,
    transform,
) -> np.ndarray:
    """Return sea cells cut off from the open sea by weir + closures (module docstring, step 3).

    Side effect: adds "sealed" to every closed report entry (whether the river
    check point behind it is cut off from the open sea).
    """
    inv = ~transform
    blocked = _weir_edges(weir_gdf, transform)
    with_closure = blocked | {_edge_key(c["cell_a"], c["cell_b"]) for c in closures}
    lab = _components(valid, with_closure)
    sea = _open_sea_label(lab, seaward, valid)
    lab_before = _components(valid, blocked)
    sea_before = _open_sea_label(lab_before, seaward, valid)
    # only what the closures cut off -- not sea pockets the baseline weir already isolates
    estuary = seaward & (lab != sea) & (lab_before == sea_before)
    for entry in report:
        if not entry.get("closed"):
            continue
        c, r = (int(np.floor(v)) for v in inv * tuple(entry["river_check_point"]))
        entry["sealed"] = bool(_in_grid((r, c), lab.shape) and lab[r, c] != sea)
        if not entry["sealed"]:
            log.warning(
                f"Outlet reach {entry['reach_id']}: the river behind the closure is still "
                "connected to the open sea -- the closure does not seal this mouth"
            )
    return estuary


def relabel_estuary(weir_gdf, kinds, estuary, transform) -> list:
    """Relabel coast segments bordering an estuary cell as estuary."""
    inv = ~transform
    out = []
    for seg, kind in zip(weir_gdf.geometry, kinds):
        sides = [rc for rc in _edge_cells(seg, inv) if _in_grid(rc, estuary.shape)]
        out.append(
            KIND_ESTUARY
            if kind == KIND_COAST and any(estuary[rc] for rc in sides)
            else kind
        )
    return out


@dataclass(frozen=True)
class AdaptationLines:
    """Labelled baseline weir, mouth closures, pump lines and per-mouth report."""

    weir: gpd.GeoDataFrame
    closure: gpd.GeoDataFrame
    pump: gpd.GeoDataFrame
    seaward: np.ndarray
    estuary: np.ndarray
    kind_counts: dict
    report: list[dict]


def build_adaptation_lines(
    weir_gdf: gpd.GeoDataFrame,
    landuse: np.ndarray,
    valid: np.ndarray,
    transform,
    rivers: gpd.GeoDataFrame,
    *,
    min_component_cells: int,
    unprotected_ocean_wetlands: bool,
    closure: dict,
) -> AdaptationLines:
    """Label the baseline weir, close every river mouth and draw its pump line.

    `weir_gdf` holds the model's 2-point cell-edge weir segments and `rivers`
    the clean SWORD network (width, reach_id, n_rch_dn, rch_id_up), both in
    the CRS of `landuse`/`valid`, which lie on the SFINCS grid (`transform`).
    `min_component_cells`/`unprotected_ocean_wetlands` are rule 11's weir-builder
    settings; `closure` is config `adaptation.lines` (mouth_closures settings).
    """
    require_keys(closure, CLOSURE_KEYS, "adaptation.lines")
    crs = weir_gdf.crs
    seaward = landuse == LANDUSE_SEA_CODE
    if unprotected_ocean_wetlands:
        grid = GridArrays(
            crs,
            np.zeros(landuse.shape, np.float32),
            valid,
            float(np.hypot(transform.a, transform.d)),
            transform,
        )
        seaward = seaward | ocean_linked_wetland_mask(landuse, seaward, grid)
    channel = build_channel_mask_regular(
        centrelines_off_sea(rivers, seaward, transform),
        "width",
        landuse.shape,
        transform,
    )

    kinds, kind_counts = label_weir_kinds(
        weir_gdf, seaward, channel, valid, transform, min_component_cells
    )
    log.info(f"Baseline weir labels: {kind_counts}")
    land = protected_land(weir_gdf, seaward, valid, transform)
    closures, pumps, report = mouth_closures(rivers, land, transform, **closure)
    estuary = estuary_and_seal(weir_gdf, closures, report, seaward, valid, transform)
    kinds = relabel_estuary(weir_gdf, kinds, estuary, transform)
    weir = weir_gdf.assign(kind=kinds)
    log.info(
        f"After closure: {pd.Series(kinds).value_counts().to_dict()}, "
        f"{int(estuary.sum())} sea cell(s) cut off as estuary"
    )

    closure = (
        gpd.GeoDataFrame(
            [
                {
                    "reach_id": c["reach_id"],
                    "kind": "closure",
                    "geometry": c["geometry"],
                }
                for c in closures
            ],
            geometry="geometry",
            crs=crs,
        )
        if closures
        else gpd.GeoDataFrame({"reach_id": [], "kind": []}, geometry=[], crs=crs)
    )
    pump = (
        gpd.GeoDataFrame(pumps, geometry="geometry", crs=crs)
        if pumps
        else gpd.GeoDataFrame({"reach_id": []}, geometry=[], crs=crs)
    )
    return AdaptationLines(weir, closure, pump, seaward, estuary, kind_counts, report)


def measure_line_layers(lines: AdaptationLines) -> dict[str, gpd.GeoDataFrame]:
    """Return the measure line files, keyed by ADAPTATION_LINE_NAMES."""
    weir = lines.weir[["kind", "geometry"]]
    layers = {
        "coastal_protection_weir_PC": pd.concat(
            [weir[weir["kind"] == KIND_COAST], lines.closure[["kind", "geometry"]]],
            ignore_index=True,
        ),
        "coastal_protection_weir_PO": weir[weir["kind"] == KIND_COAST],
        "river_protection_weir_PO": weir[weir["kind"].isin([KIND_RIVER, KIND_ESTUARY])],
        "pump_river_mouth": lines.pump,
    }
    return {
        name: gpd.GeoDataFrame(layers[name], geometry="geometry", crs=lines.weir.crs)
        for name in ADAPTATION_LINE_NAMES
    }


def unsealed_mouths(report: list[dict]) -> list[int]:
    """Return the outlet reaches whose closure leaves the river open to the sea."""
    return [r["reach_id"] for r in report if r.get("closed") and not r.get("sealed")]


def require_sealed(report: list[dict]) -> None:
    """Raise RuntimeError if a mouth closure does not seal its river."""
    unsealed = unsealed_mouths(report)
    if unsealed:
        raise RuntimeError(
            f"mouth closure does not seal outlet reach(es) {unsealed}: the river "
            "behind the closure is still connected to the open sea, so "
            "coastal_protection_weir_PC would leave the mouth open; check "
            "adaptation_lines.png (snakemake --keep-incomplete keeps it)"
        )


def build_info(
    lines: AdaptationLines, basin_id: str, built: str, source_weir: str, grid_def: dict
) -> dict:
    """Return the build_info.json content: source grid, counts, per-mouth report."""
    return {
        "basin_id": basin_id,
        "built": built,
        "source_weir": source_weir,
        "grid": {
            k: grid_def[k]
            for k in ("x0", "y0", "dx", "dy", "mmax", "nmax", "rotation")
            if k in grid_def
        },
        "baseline_weir_kinds": lines.weir["kind"].value_counts().to_dict(),
        "facing_small_island": lines.kind_counts["facing_small_island"],
        "by_nearest_water": lines.kind_counts["by_nearest_water"],
        "unresolved_as_river": lines.kind_counts["unresolved_as_river"],
        "estuary_cells": int(lines.estuary.sum()),
        "mouths": [
            {k: v for k, v in r.items() if k != "river_check_point"}
            for r in lines.report
        ],
    }


def plot_adaptation_lines(path, lines: AdaptationLines, transform, basin_id) -> None:
    """Plot the weir by kind with closures and pumps: overview + zoom per mouth."""
    weir, closure, pump = lines.weir, lines.closure, lines.pump
    mouths = [r for r in lines.report if r.get("closed")]
    fig, axes = plt.subplots(
        1, 1 + len(mouths), figsize=(7.5 * (1 + len(mouths)), 7.5), squeeze=False
    )
    axes = axes[0]
    for kind, colour in KIND_COLOURS.items():
        part = weir[weir["kind"] == kind]
        if not part.empty:
            part.plot(ax=axes[0], color=colour, lw=1)
    for ax in axes:
        if not closure.empty:
            closure.plot(ax=ax, color="red", lw=3)
        if not pump.empty:
            pump.plot(ax=ax, color="blue", lw=2.5)
        ax.set_aspect("equal")
    axes[0].set_title(f"{basin_id}: baseline weir by kind, closure(s), pump(s)")
    h, w = lines.seaward.shape
    for ax, entry in zip(axes[1:], mouths):
        x, y = entry["anchor"]
        half = max(1.5 * entry["crossing_m"], 1000.0)
        for r in range(h):
            for c in range(w):
                if not lines.seaward[r, c]:
                    continue
                corners = [
                    transform * (c, r),
                    transform * (c + 1, r),
                    transform * (c + 1, r + 1),
                    transform * (c, r + 1),
                ]
                if (
                    abs(corners[0][0] - x) < half + 200
                    and abs(corners[0][1] - y) < half + 200
                ):
                    ax.add_patch(
                        MplPolygon(
                            corners,
                            fc="#fdd0a2" if lines.estuary[r, c] else "#c6dbef",
                            ec="none",
                        )
                    )
        for kind, colour in KIND_COLOURS.items():
            part = weir[weir["kind"] == kind]
            if not part.empty:
                part.plot(ax=ax, color=colour, lw=2 if kind == KIND_COAST else 1.5)
        closure.plot(ax=ax, color="red", lw=3)
        pump.plot(ax=ax, color="blue", lw=2.5)
        ax.set_xlim(x - half, x + half)
        ax.set_ylim(y - half, y + half)
        ax.set_title(
            f"mouth of reach {entry['reach_id']}: {entry['crossing_m']:.0f} m crossing, "
            f"{entry['tilt_deg_from_perpendicular']:+.0f} deg, sealed={entry['sealed']}"
        )
    axes[-1].legend(
        handles=[
            Line2D([], [], color="black", lw=2, label="coast (raised: PC + PO)"),
            Line2D(
                [],
                [],
                color="0.6",
                lw=1.5,
                label="estuary bank (PC: baseline, PO: river_levee)",
            ),
            Line2D([], [], color="purple", lw=1.5, label="river bank (river_levee)"),
            Line2D([], [], color="red", lw=3, label="mouth closure (PC)"),
            Line2D([], [], color="blue", lw=2.5, label="pump line (intake -> outlet)"),
            Line2D([], [], color="#fdd0a2", lw=8, label="sea cells cut off = estuary"),
        ],
        loc="upper left",
        fontsize=8,
    )
    fig.tight_layout()
    save_figure(fig, path, dpi=120)
