"""Target water-surface profile for the gradually-varied-flow (GVF) bed estimate.

Builds the node network the GVF solver (src.gvf_solver) runs on, plus the
target level, discharge, width, slope scale and node class at every node.
Implements WP1 of the GVF plan; the method follows Neal et al. (2021),
*Estimating river channel bathymetry in large scale flood inundation models*,
WRR 57, e2020WR028301.

Node network:

- Nodes every ``node_spacing_m`` along each reach, both ends included.
  Reach geometries must run downstream; :func:`orient_downstream` checks this
  against the topology and reverses lines that run upstream.
- Reach ends that meet at a junction share ONE node, so the target, the bed
  and the simulated level each have a single value there. Ends on either side
  of a structure (a weir, lock or dam) are kept as separate nodes, so the
  level can drop across it.

Target level (``target_source: sword_wse``):

- SWORD's per-reach ``wse`` (EGM2008, 0.1 m steps) plus the geoid offset
  N_EGM2008 - N_GOCO06s at the reach, i.e. the model datum (GOCO06s).
- Placed at each reach's midpoint and interpolated by along-network distance
  to the nodes (towards the up- and downstream neighbours' midpoints).
- Smoothed over ``target_smoothing_m`` (removes the 0.1 m quantisation), never
  across a structure.
- Made non-increasing downstream (per-reach isotonic regression, repeated
  until junctions agree, then a final clip).

Structures are only placed at reach junctions: the edge from reach u to its
downstream reach d is a structure if the level drops by more than
``step_threshold_m``, if u or d is a SWORD obstruction reach (``obstr_type``
1-4) and the level drops by at least ``structure_min_drop_m``, or if u is in
the manual ``structures`` list.

An edge where the level RISES downstream by more than ``step_threshold_m``
is inconsistent data (1248635: the IJssel's SWORD level is 1.1 m above the
Nederrijn reach it branches from). With ``cut_rising_edges`` (default) it is
cut like a structure, so the rise stays local; otherwise the monotone
conditioning would lift every upstream node to the higher level.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import linemerge

from src.river_network import normalize_reach_id

log = logging.getLogger(__name__)

NODE_INVERT = "invert"
NODE_FIXED_BED = "fixed_bed"
NODE_WATER_BODY = "water_body"
NODE_NO_FLOW = "no_flow"  # set per reach by src.gvf_solver (no discharge)

# SWORD obstr_type: 1 dam, 2 lock, 3 low-permeability dam, 4 waterfall.
SWORD_OBSTRUCTION_TYPES = (1, 2, 3, 4)


@dataclass
class GvfNetwork:
    """Node network of the GVF solve.

    Attributes:
        reach_ids: Reach ids in topological order (upstream first).
        downstream: {reach_id: downstream reach ids in the network}.
        structure_edges: (upstream, downstream) reach pairs with a structure
            between them; their end nodes are not shared.
        reach_nodes: {reach_id: node ids in downstream order}.
        reach_chainage: {reach_id: chainage (m) of those nodes from the
            reach's upstream end}.
        reach_length: {reach_id: geometry length (m)}.
        node_x, node_y: Node coordinates (CRS of the input reaches).
        owner: Per node, the most downstream reach containing it (reverse
            topological order); used to pick one width and discharge for
            per-node diagnostics.
        reach_slots: {reach_id: bed-slot ids of its nodes}. The water level is
            one value per node, but the BED is one value per reach and node
            ("slot"): reaches meeting at a junction keep their own bed there,
            as real junctions are often discordant (a small branch's bed sits
            above the main channel's).
        reach_line: {reach_id: the reach's LineString, running downstream}.
    """

    reach_ids: list[str]
    downstream: dict[str, list[str]]
    structure_edges: set[tuple[str, str]]
    reach_nodes: dict[str, np.ndarray]
    reach_chainage: dict[str, np.ndarray]
    reach_length: dict[str, float]
    node_x: np.ndarray
    node_y: np.ndarray
    owner: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    reach_slots: dict[str, np.ndarray] = field(default_factory=dict)
    reach_line: dict[str, LineString] = field(default_factory=dict)

    @property
    def n_slots(self) -> int:
        """Return the number of bed slots (sum of nodes over reaches)."""
        return sum(len(v) for v in self.reach_slots.values())

    def slot_node(self) -> np.ndarray:
        """Return the node id of every bed slot."""
        out = np.empty(self.n_slots, dtype=int)
        for r in self.reach_ids:
            out[self.reach_slots[r]] = self.reach_nodes[r]
        return out

    def slot_reach(self) -> np.ndarray:
        """Return the reach id of every bed slot."""
        out = np.empty(self.n_slots, dtype=object)
        for r in self.reach_ids:
            out[self.reach_slots[r]] = r
        return out

    @property
    def n_nodes(self) -> int:
        """Return the number of distinct nodes."""
        return len(self.node_x)

    def upstream(self) -> dict[str, list[str]]:
        """Return {reach_id: upstream reach ids in the network}."""
        up: dict[str, list[str]] = {r: [] for r in self.reach_ids}
        for r in self.reach_ids:
            for d in self.downstream[r]:
                up[d].append(r)
        return up

    def node_edges(self) -> list[tuple[int, int, float]]:
        """Return directed (upstream node, downstream node, distance m) edges."""
        edges = []
        for r in self.reach_ids:
            ids, ch = self.reach_nodes[r], self.reach_chainage[r]
            for k in range(len(ids) - 1):
                edges.append((int(ids[k]), int(ids[k + 1]), float(ch[k + 1] - ch[k])))
        return edges


# ── topology and geometry ─────────────────────────────────────────────────


def topological_order(
    reach_ids: list[str], downstream: dict[str, list[str]]
) -> list[str]:
    """Return reach ids upstream-first (Kahn); reaches on a cycle go last."""
    indeg = {r: 0 for r in reach_ids}
    for r in reach_ids:
        for d in downstream.get(r, []):
            indeg[d] += 1
    queue = deque(r for r in reach_ids if indeg[r] == 0)
    order: list[str] = []
    while queue:
        r = queue.popleft()
        order.append(r)
        for d in downstream.get(r, []):
            indeg[d] -= 1
            if indeg[d] == 0:
                queue.append(d)
    if len(order) < len(reach_ids):
        rest = [r for r in reach_ids if r not in set(order)]
        log.warning(
            f"GVF network has a directed cycle through {len(rest)} reaches: {rest[:10]}"
        )
        order += rest
    return order


def _line(geom) -> LineString:
    if geom.geom_type == "MultiLineString":
        merged = linemerge(geom)
        if merged.geom_type == "MultiLineString":
            return max(merged.geoms, key=lambda g: g.length)
        return merged
    return geom


def orient_downstream(
    rivers: gpd.GeoDataFrame, downstream: dict[str, list[str]]
) -> tuple[gpd.GeoDataFrame, list[str]]:
    """Reverse reach lines that run upstream.

    A line runs downstream if its last vertex is closer than its first to the
    downstream neighbours' lines (or, for a network end, if its first vertex
    is closer than its last to the upstream neighbours' lines). Reaches
    without neighbours are left as they are.

    Returns:
        The reaches with downstream-running LineStrings, and the reversed ids.
    """
    rivers = rivers.copy()
    lines = {
        normalize_reach_id(r): _line(g)
        for r, g in zip(rivers["reach_id"], rivers.geometry)
    }
    upstream: dict[str, list[str]] = defaultdict(list)
    for r, dns in downstream.items():
        for d in dns:
            upstream[d].append(r)

    reversed_ids = []
    new_geoms = []
    for rid in rivers["reach_id"].map(normalize_reach_id):
        line = lines[rid]
        first, last = Point(line.coords[0]), Point(line.coords[-1])
        dns = [lines[d] for d in downstream.get(rid, []) if d in lines]
        ups = [lines[u] for u in upstream.get(rid, []) if u in lines]
        flip = False
        if dns:
            flip = min(g.distance(first) for g in dns) < min(
                g.distance(last) for g in dns
            )
        elif ups:
            flip = min(g.distance(last) for g in ups) < min(
                g.distance(first) for g in ups
            )
        if flip:
            line = LineString(list(line.coords)[::-1])
            reversed_ids.append(rid)
        new_geoms.append(line)
    rivers = rivers.set_geometry(
        gpd.GeoSeries(new_geoms, index=rivers.index, crs=rivers.crs)
    )
    if reversed_ids:
        log.warning(
            f"{len(reversed_ids)} reach line(s) ran upstream and were reversed: {reversed_ids[:10]}"
        )
    return rivers, reversed_ids


class _UnionFind:
    def __init__(self):
        self.parent: dict = {}

    def find(self, a):
        self.parent.setdefault(a, a)
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_node_network(
    rivers: gpd.GeoDataFrame,
    downstream: dict[str, list[str]],
    node_spacing_m: float,
    structure_edges: set[tuple[str, str]] | None = None,
) -> GvfNetwork:
    """Place nodes along every reach and merge reach ends at junctions.

    Args:
        rivers: Reaches in a metric CRS, lines running downstream
            (:func:`orient_downstream`), with 'reach_id'.
        downstream: In-network downstream adjacency
            (src.river_network.build_downstream_adjacency).
        node_spacing_m: Target node spacing; each reach gets
            ``ceil(length / spacing)`` equal segments, at least one.
        structure_edges: (upstream, downstream) reach pairs whose end nodes
            are kept separate.

    Returns:
        The node network.
    """
    structure_edges = set(structure_edges or ())
    rids = [normalize_reach_id(r) for r in rivers["reach_id"]]
    lines = dict(zip(rids, (_line(g) for g in rivers.geometry)))
    downstream = {r: [d for d in downstream.get(r, []) if d in lines] for r in rids}
    order = topological_order(rids, downstream)

    # Reach ends that meet (no structure between them) become one node.
    uf = _UnionFind()
    for r in rids:
        uf.find((r, "up"))
        uf.find((r, "dn"))
        for d in downstream[r]:
            if (r, d) not in structure_edges:
                uf.union((r, "dn"), (d, "up"))

    end_id: dict = {}
    end_xy: dict[int, list[tuple[float, float]]] = defaultdict(list)
    xs: list[float] = []
    ys: list[float] = []

    def _end_node(key, xy) -> int:
        root = uf.find(key)
        if root not in end_id:
            end_id[root] = len(xs)
            xs.append(0.0)
            ys.append(0.0)
        nid = end_id[root]
        end_xy[nid].append(xy)
        return nid

    reach_nodes: dict[str, np.ndarray] = {}
    reach_chainage: dict[str, np.ndarray] = {}
    reach_length: dict[str, float] = {}
    for r in order:
        line = lines[r]
        length = float(line.length)
        n_seg = max(1, int(np.ceil(length / node_spacing_m)))
        chain = np.linspace(0.0, length, n_seg + 1)
        ids = np.empty(n_seg + 1, dtype=int)
        ids[0] = _end_node((r, "up"), line.coords[0][:2])
        ids[-1] = _end_node((r, "dn"), line.coords[-1][:2])
        for k in range(1, n_seg):
            pt = line.interpolate(chain[k])
            ids[k] = len(xs)
            xs.append(pt.x)
            ys.append(pt.y)
        reach_nodes[r], reach_chainage[r], reach_length[r] = ids, chain, length

    node_x, node_y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    for nid, pts in end_xy.items():
        node_x[nid] = np.mean([p[0] for p in pts])
        node_y[nid] = np.mean([p[1] for p in pts])

    # Owner: the first reach in REVERSE topological order containing the node.
    owner = np.full(len(node_x), None, dtype=object)
    for r in reversed(order):
        ids = reach_nodes[r]
        unowned = ids[owner[ids] == None]  # noqa: E711 (object array)
        owner[unowned] = r

    reach_slots: dict[str, np.ndarray] = {}
    k = 0
    for r in order:
        reach_slots[r] = np.arange(k, k + len(reach_nodes[r]))
        k += len(reach_nodes[r])

    return GvfNetwork(
        reach_ids=order,
        downstream=downstream,
        structure_edges=structure_edges,
        reach_nodes=reach_nodes,
        reach_chainage=reach_chainage,
        reach_length=reach_length,
        node_x=node_x,
        node_y=node_y,
        owner=owner,
        reach_slots=reach_slots,
        reach_line=lines,
    )


# ── target levels ─────────────────────────────────────────────────────────


def reach_target_levels(
    rivers: gpd.GeoDataFrame,
    datum_offset_m: np.ndarray | float = 0.0,
    wse_column: str = "wse",
    min_valid_m: float = -1000.0,
) -> dict[str, float]:
    """Per-reach target level in the model datum.

    Args:
        rivers: Reaches with ``wse_column`` (SWORD: m above EGM2008).
        datum_offset_m: N_EGM2008 - N_GOCO06s per reach (or one value), added
            to ``wse`` -- the same correction rule 05a applies to FathomDEM.
        wse_column: Column with the reach water-surface elevation.
        min_valid_m: Values below it (SWORD's -9999 fill) count as missing.

    Returns:
        {reach_id: level (m, GOCO06s)}; missing values are NaN.
    """
    wse = pd.to_numeric(rivers[wse_column], errors="coerce").to_numpy(dtype=float)
    wse = np.where(wse < min_valid_m, np.nan, wse)
    level = wse + np.broadcast_to(np.asarray(datum_offset_m, dtype=float), wse.shape)
    return {normalize_reach_id(r): float(v) for r, v in zip(rivers["reach_id"], level)}


def detect_structure_edges(
    rivers: gpd.GeoDataFrame,
    downstream: dict[str, list[str]],
    reach_level: dict[str, float],
    step_threshold_m: float = 0.5,
    structure_min_drop_m: float = 0.2,
    manual: list | tuple = (),
    use_sword_obstructions: bool = True,
) -> tuple[set[tuple[str, str]], list[tuple[str, str, float]]]:
    """Find junctions that carry a weir, lock or dam, and rising levels.

    Args:
        rivers: Reaches with 'reach_id' and, optionally, 'obstr_type'.
        downstream: In-network downstream adjacency.
        reach_level: Per-reach target level (:func:`reach_target_levels`).
        step_threshold_m: Any drop larger than this is a structure.
        structure_min_drop_m: Smallest drop that counts as a structure at a
            SWORD obstruction reach.
        manual: Reach ids with a structure at their downstream end (all their
            downstream edges), from config.
        use_sword_obstructions: Use SWORD ``obstr_type``.

    Returns:
        The structure edges, and the edges where the level RISES downstream
        by more than ``step_threshold_m`` as (upstream, downstream, rise).
    """
    obstructed: set[str] = set()
    if use_sword_obstructions and "obstr_type" in rivers.columns:
        ot = pd.to_numeric(rivers["obstr_type"], errors="coerce").fillna(0).astype(int)
        obstructed = {
            normalize_reach_id(r)
            for r, t in zip(rivers["reach_id"], ot)
            if t in SWORD_OBSTRUCTION_TYPES
        }
    manual_ids = {normalize_reach_id(r) for r in manual}

    edges: set[tuple[str, str]] = set()
    rising: list[tuple[str, str, float]] = []
    for u, dns in downstream.items():
        for d in dns:
            drop = reach_level.get(u, np.nan) - reach_level.get(d, np.nan)
            if u in manual_ids:
                edges.add((u, d))
            elif np.isfinite(drop):
                if drop > step_threshold_m:
                    edges.add((u, d))
                elif (
                    u in obstructed or d in obstructed
                ) and drop >= structure_min_drop_m - 1e-9:
                    edges.add((u, d))
                elif -drop > step_threshold_m:
                    rising.append((u, d, float(-drop)))
    return edges, rising


def interpolate_node_targets(
    net: GvfNetwork, reach_level: dict[str, float]
) -> np.ndarray:
    """Interpolate per-reach levels onto the nodes by along-network distance.

    Each reach's level sits at its midpoint. Towards each end the level runs
    linearly to the mean level of the neighbours on that side, placed at their
    mean half-length beyond the end; neighbours across a structure are
    ignored. A node shared by several reaches gets the mean of their values.
    Nodes left without a value are filled from their graph neighbours.
    """
    up = net.upstream()
    acc = np.zeros(net.n_nodes)
    cnt = np.zeros(net.n_nodes)
    for r in net.reach_ids:
        length = net.reach_length[r]
        pts: list[tuple[float, float]] = []
        ups = [
            u
            for u in up[r]
            if (u, r) not in net.structure_edges
            and np.isfinite(reach_level.get(u, np.nan))
        ]
        if ups:
            pts.append(
                (
                    -np.mean([net.reach_length[u] for u in ups]) / 2,
                    np.mean([reach_level[u] for u in ups]),
                )
            )
        if np.isfinite(reach_level.get(r, np.nan)):
            pts.append((length / 2, reach_level[r]))
        dns = [
            d
            for d in net.downstream[r]
            if (r, d) not in net.structure_edges
            and np.isfinite(reach_level.get(d, np.nan))
        ]
        if dns:
            pts.append(
                (
                    length + np.mean([net.reach_length[d] for d in dns]) / 2,
                    np.mean([reach_level[d] for d in dns]),
                )
            )
        if not pts:
            continue
        px, pv = zip(*sorted(pts))
        ids = net.reach_nodes[r]
        np.add.at(acc, ids, np.interp(net.reach_chainage[r], px, pv))
        np.add.at(cnt, ids, 1.0)
    target = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return _fill_from_neighbours(target, net.node_edges())


def _neighbours(n_nodes: int, edges: list[tuple[int, int, float]]) -> list[list[int]]:
    nb: list[list[int]] = [[] for _ in range(n_nodes)]
    for a, b, _ in edges:
        if a != b:
            nb[a].append(b)
            nb[b].append(a)
    return nb


def _fill_from_neighbours(
    values: np.ndarray, edges: list[tuple[int, int, float]]
) -> np.ndarray:
    values = values.copy()
    nb = _neighbours(len(values), edges)
    for _ in range(len(values)):
        missing = np.flatnonzero(~np.isfinite(values))
        if missing.size == 0:
            break
        new = values.copy()
        for i in missing:
            vals = [values[j] for j in nb[i] if np.isfinite(values[j])]
            if vals:
                new[i] = float(np.mean(vals))
        if np.array_equal(np.isfinite(new), np.isfinite(values)):
            break
        values = new
    return values


def smooth_on_network(
    values: np.ndarray,
    edges: list[tuple[int, int, float]],
    window_m: float,
    spacing_m: float,
) -> np.ndarray:
    """Smooth node values along the network, like a running mean of ``window_m``.

    Repeated neighbour averaging (``x <- x/2 + mean(neighbours)/2``); each pass
    adds a variance of spacing^2 / 2 on a chain, so ``window^2 / (6 spacing^2)``
    passes match the variance of a ``window_m`` running mean. Edges are the
    node network's, so nothing is smoothed across a structure.
    """
    if window_m <= 0 or len(values) == 0:
        return values.copy()
    n_pass = max(1, int(round(window_m**2 / (6.0 * spacing_m**2))))
    nb = _neighbours(len(values), edges)
    v = values.astype(float).copy()
    for _ in range(n_pass):
        mean_nb = np.array([np.mean(v[j]) if j else v[i] for i, j in enumerate(nb)])
        v = 0.5 * v + 0.5 * mean_nb
    return v


def node_topological_order(net: GvfNetwork) -> list[int]:
    """Return node ids upstream-first along the directed node edges."""
    edges = net.node_edges()
    succ: list[list[int]] = [[] for _ in range(net.n_nodes)]
    indeg = np.zeros(net.n_nodes, dtype=int)
    for a, b, _ in edges:
        succ[a].append(b)
        indeg[b] += 1
    queue = deque(np.flatnonzero(indeg == 0).tolist())
    order: list[int] = []
    while queue:
        a = queue.popleft()
        order.append(a)
        for b in succ[a]:
            indeg[b] -= 1
            if indeg[b] == 0:
                queue.append(b)
    seen = set(order)
    order += [i for i in range(net.n_nodes) if i not in seen]
    return order


def enforce_non_increasing(
    net: GvfNetwork, values: np.ndarray, tol_m: float = 1e-4, max_sweeps: int = 20000
) -> np.ndarray:
    """Make node values non-increasing downstream with one value per node.

    Weighted L2 isotonic regression on the whole node graph: the values
    closest to the input (weights = length of channel each node stands for)
    with ``v[a] >= v[b]`` on every node edge a -> b. Solved exactly by
    Hildreth's dual coordinate ascent, one constraint per edge, so loops
    (bifurcate-then-rejoin) and junctions are handled together rather than
    reach by reach. A last pass lifts any node still below its downstream
    neighbour (only the solver tolerance remains there).
    """
    v = values.astype(float).copy()
    edges = net.node_edges()
    if not edges:
        return v
    a = np.array([e[0] for e in edges])
    b = np.array([e[1] for e in edges])
    length = np.array([e[2] for e in edges])
    w = np.zeros(net.n_nodes)
    np.add.at(w, a, length / 2)
    np.add.at(w, b, length / 2)
    inv_w = 1.0 / np.maximum(w, 1.0)
    denom = inv_w[a] + inv_w[b]
    lam = np.zeros(len(edges))
    for _ in range(max_sweeps):
        worst = 0.0
        for k in range(len(edges)):
            ia, ib = a[k], b[k]
            g = v[ib] - v[ia]  # > 0: violation
            delta = max(-lam[k], g / denom[k])
            if delta != 0.0:
                lam[k] += delta
                v[ia] += delta * inv_w[ia]
                v[ib] -= delta * inv_w[ib]
            worst = max(worst, g)
        if worst < tol_m:
            break
    else:
        log.warning(
            f"GVF target: isotonic regression stopped after {max_sweeps} sweeps"
        )

    succ: list[list[int]] = [[] for _ in range(net.n_nodes)]
    for ia, ib in zip(a, b):
        succ[ia].append(ib)
    before = v.copy()
    for i in reversed(node_topological_order(net)):
        if succ[i]:
            v[i] = max(v[i], max(v[j] for j in succ[i]))
    lift = float(np.nanmax(v - before))
    if lift > 0.05:
        log.warning(f"GVF target: final monotone clip lifted a node by {lift:.2f} m")
    return v


def node_slope(net: GvfNetwork, values: np.ndarray) -> np.ndarray:
    """Downstream fall of ``values`` per metre at each node (central where possible)."""
    fall = np.zeros(net.n_nodes)
    dist = np.zeros(net.n_nodes)
    for a, b, d in net.node_edges():
        if d <= 0:
            continue
        for i in (a, b):
            fall[i] += values[a] - values[b]
            dist[i] += d
    return np.where(dist > 0, fall / np.maximum(dist, 1e-9), 0.0)


def classify_nodes(
    width_m: np.ndarray,
    sea_fraction: np.ndarray,
    dem_m: np.ndarray,
    gebco_floor_m: float,
    min_width_m: float = 1000.0,
    sea_class_fraction: float = 0.5,
) -> np.ndarray:
    """Class per node: water body, fixed bed or invert.

    - ``water_body``: wider than ``min_width_m`` and either mostly sea class
      (> ``sea_class_fraction``) or with the DEM at the GEBCO floor (bed from
      GEBCO's clamp, within 0.1 m of ``gebco_floor_m``). Level prescribed from
      the target, bed from the DEM.
    - ``fixed_bed``: other nodes that are mostly sea class. Bed from the DEM,
      forward solve only.
    - ``invert``: everything else; the bed is estimated.
    """
    sea = np.nan_to_num(sea_fraction, nan=0.0) > sea_class_fraction
    at_floor = np.nan_to_num(dem_m, nan=np.inf) <= gebco_floor_m + 0.1
    wide = np.nan_to_num(width_m, nan=0.0) > min_width_m
    has_dem = np.isfinite(dem_m)
    cls = np.full(len(width_m), NODE_INVERT, dtype=object)
    cls[sea & has_dem] = NODE_FIXED_BED
    cls[wide & (sea | at_floor) & has_dem] = NODE_WATER_BODY
    return cls


# ── main entry ────────────────────────────────────────────────────────────


@dataclass
class GvfTargets:
    """Node network plus per-node and per-reach inputs for src.gvf_solver."""

    net: GvfNetwork
    eta_target: np.ndarray
    eta_raw: np.ndarray
    s_scale: np.ndarray
    node_class: np.ndarray
    dem: np.ndarray
    reach_q: dict[str, float]
    reach_width: dict[str, float]
    reach_level: dict[str, float]
    rising_edges: list[tuple[str, str, float]]

    def to_frame(self) -> pd.DataFrame:
        """One row per (reach, node) in downstream order: the WP1 node table."""
        rows = []
        net = self.net
        for r in net.reach_ids:
            for nid, ch in zip(net.reach_nodes[r], net.reach_chainage[r]):
                rows.append(
                    {
                        "reach_id": r,
                        "node_id": int(nid),
                        "chainage_m": float(ch),
                        "x": float(net.node_x[nid]),
                        "y": float(net.node_y[nid]),
                        "width": self.reach_width[r],
                        "q": self.reach_q[r],
                        "eta_raw": float(self.eta_raw[nid]),
                        "eta_target": float(self.eta_target[nid]),
                        "s_scale": float(self.s_scale[nid]),
                        "node_class": self.node_class[nid],
                        "owner": net.owner[nid],
                    }
                )
        return pd.DataFrame(rows)


def build_gvf_targets(
    rivers: gpd.GeoDataFrame,
    reach_q: dict[str, float],
    datum_offset_m: np.ndarray | float,
    sample_sea_fraction,
    sample_dem,
    gebco_floor_m: float,
    cfg: dict,
) -> GvfTargets:
    """Build the GVF node network and its target profile (WP1).

    Args:
        rivers: Cleaned network in a metric CRS, with 'reach_id', 'rch_id_dn',
            'width', 'wse' and optionally 'obstr_type'.
        reach_q: {reach_id: discharge (m3/s)} paired with ``wse``.
        datum_offset_m: Per-reach (same order as ``rivers``) or scalar
            N_EGM2008 - N_GOCO06s.
        sample_sea_fraction: Callable (x, y, width) -> sea-class fraction of
            the cross-section at each node.
        sample_dem: Callable (x, y) -> merged (unconditioned) DEM at each node,
            NaN outside it.
        gebco_floor_m: Lowest level of the clamped GEBCO bathymetry
            (``baseline_m - terrain.gebco_max_depth_m``).
        cfg: The ``river_depth_modelling.gvf`` config block.

    Returns:
        The targets for src.gvf_solver.
    """
    from src.river_network import build_downstream_adjacency

    downstream = build_downstream_adjacency(rivers)
    rivers, _ = orient_downstream(rivers, downstream)
    reach_level = reach_target_levels(rivers, datum_offset_m)
    structures, rising = detect_structure_edges(
        rivers,
        downstream,
        reach_level,
        step_threshold_m=float(cfg.get("step_threshold_m", 0.5)),
        structure_min_drop_m=float(cfg.get("structure_min_drop_m", 0.2)),
        manual=cfg.get("structures") or [],
        use_sword_obstructions=bool(cfg.get("use_sword_obstructions", True)),
    )
    for u, d, rise in rising:
        log.warning(
            f"GVF target: level rises {rise:.2f} m from reach {u} to {d} (canal or SWORD error?)"
        )
    log.info(
        f"GVF target: {len(structures)} structure junction(s): {sorted(structures)[:20]}"
    )
    if rising and bool(cfg.get("cut_rising_edges", True)):
        structures = structures | {(u, d) for u, d, _ in rising}
        log.info(f"GVF target: {len(rising)} rising junction(s) cut like structures")

    spacing = float(cfg.get("node_spacing_m", 500.0))
    net = build_node_network(rivers, downstream, spacing, structures)
    edges = net.node_edges()

    eta_raw = interpolate_node_targets(net, reach_level)
    eta = smooth_on_network(
        eta_raw, edges, float(cfg.get("target_smoothing_m", 5000.0)), spacing
    )
    eta = enforce_non_increasing(net, eta)

    slope = smooth_on_network(
        node_slope(net, eta),
        edges,
        float(cfg.get("target_smoothing_m", 5000.0)),
        spacing,
    )
    s_scale = np.maximum(slope, float(cfg.get("slope_floor", 1.0e-5)))

    rid_norm = [normalize_reach_id(r) for r in rivers["reach_id"]]
    reach_width = dict(
        zip(rid_norm, pd.to_numeric(rivers["width"], errors="coerce").astype(float))
    )
    node_width = np.zeros(net.n_nodes)
    node_cnt = np.zeros(net.n_nodes)
    for r in net.reach_ids:
        np.add.at(node_width, net.reach_nodes[r], reach_width[r])
        np.add.at(node_cnt, net.reach_nodes[r], 1.0)
    node_width /= np.maximum(node_cnt, 1)

    dem = np.asarray(sample_dem(net.node_x, net.node_y), dtype=float)
    sea_fraction = np.asarray(
        sample_sea_fraction(net.node_x, net.node_y, node_width), dtype=float
    )
    wb = cfg.get("water_body") or {}
    node_class = classify_nodes(
        node_width,
        sea_fraction,
        dem,
        gebco_floor_m,
        min_width_m=float(wb.get("min_width_m", 1000.0)),
        sea_class_fraction=float(wb.get("sea_class_fraction", 0.5)),
    )
    counts = pd.Series(node_class).value_counts().to_dict()
    log.info(
        f"GVF target: {net.n_nodes} nodes on {len(net.reach_ids)} reaches; classes {counts}"
    )

    return GvfTargets(
        net=net,
        eta_target=eta,
        eta_raw=eta_raw,
        s_scale=s_scale,
        node_class=node_class,
        dem=dem,
        reach_q={r: float(reach_q.get(r, 0.0)) for r in net.reach_ids},
        reach_width={r: float(reach_width[r]) for r in net.reach_ids},
        reach_level=reach_level,
        rising_edges=rising,
    )
