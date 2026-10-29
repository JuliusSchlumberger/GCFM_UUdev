"""Node network along the river reaches: placement, structures, smoothing and slope.

Rule 10 (src.water_surface) and rule 11 (src.river_bed) build the same network
from the same reaches and spacing, so node ids are shared.

- Nodes every ``node_spacing_m`` along each reach, both ends included.
  Reach geometries must run downstream; :func:`orient_downstream` checks this
  against the topology and reverses lines that run upstream.
- Reach ends that meet at a junction share ONE node, so the level and the bed
  each have a single value there. Ends on either side of a structure edge
  passed to :func:`build_node_network` are kept as separate nodes.

Structures are only placed at reach junctions: the edge from reach u to its
downstream reach d is a structure if the level drops by more than
``step_threshold_m``, if u or d is a SWORD obstruction reach (``obstr_type``
1-4) and the level drops by at least ``structure_min_drop_m``, or if u is in
the manual ``structures`` list. An edge where the level RISES downstream by
more than ``step_threshold_m`` is inconsistent data and is reported.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import linemerge
from src.river_network import kahn_order, normalize_reach_id

log = logging.getLogger(__name__)

NODE_INVERT = "invert"
NODE_FIXED_BED = "fixed_bed"
NODE_WATER_BODY = "water_body"

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


def topological_order(
    reach_ids: list[str], downstream: dict[str, list[str]]
) -> list[str]:
    """Return reach ids upstream-first (Kahn); reaches on a cycle go last."""
    order, rest = kahn_order(reach_ids, lambda r: downstream.get(r, []))
    if rest:
        log.warning(
            f"GVF network has a directed cycle through {len(rest)} reaches: {rest[:10]}"
        )
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
    for reach_id in rivers["reach_id"].map(normalize_reach_id):
        line = lines[reach_id]
        first, last = Point(line.coords[0]), Point(line.coords[-1])
        dns = [lines[d] for d in downstream.get(reach_id, []) if d in lines]
        ups = [lines[u] for u in upstream.get(reach_id, []) if u in lines]
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
            reversed_ids.append(reach_id)
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
    reach_ids = [normalize_reach_id(r) for r in rivers["reach_id"]]
    lines = dict(zip(reach_ids, (_line(g) for g in rivers.geometry)))
    downstream = {
        r: [d for d in downstream.get(r, []) if d in lines] for r in reach_ids
    }
    order = topological_order(reach_ids, downstream)

    # Reach ends that meet (no structure between them) become one node.
    uf = _UnionFind()
    for r in reach_ids:
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
        reach_level: Per-reach level (median of its nodes' levels).
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
    fixed: np.ndarray | None = None,
) -> np.ndarray:
    """Smooth node values along the network, like a running mean of ``window_m``.

    Repeated neighbour averaging (``x <- x/2 + mean(neighbours)/2``); each pass
    adds a variance of spacing^2 / 2 on a chain, so ``window^2 / (6 spacing^2)``
    passes match the variance of a ``window_m`` running mean. Edges are the
    node network's, so nothing is smoothed across a structure. A node with a
    single neighbour (network end, either side of a structure) is mirrored
    (ghost value 2 v_i - v_j), so it keeps its value and a linear trend runs
    through to the ends instead of flattening there. ``fixed`` nodes keep
    their input value.
    """
    if window_m <= 0 or len(values) == 0:
        return values.copy()
    n_pass = max(1, int(round(window_m**2 / (6.0 * spacing_m**2))))
    nb = _neighbours(len(values), edges)
    v = values.astype(float).copy()
    keep = np.zeros(len(v), dtype=bool) if fixed is None else np.asarray(fixed, bool)
    for _ in range(n_pass):
        mean_nb = np.array(
            [np.mean(v[j]) if len(j) > 1 else v[i] for i, j in enumerate(nb)]
        )
        v = np.where(keep, values, 0.5 * v + 0.5 * mean_nb)
    return v


def node_topological_order(net: GvfNetwork) -> list[int]:
    """Return node ids upstream-first along the directed node edges."""
    succ: list[list[int]] = [[] for _ in range(net.n_nodes)]
    for a, b, _ in net.node_edges():
        succ[a].append(b)
    order, _ = kahn_order(range(net.n_nodes), succ.__getitem__)
    return order


def _edge_colouring(a: np.ndarray, b: np.ndarray, n_nodes: int) -> list[np.ndarray]:
    """Greedy proper edge colouring: batches of edge indices with no shared node."""
    used: list[set[int]] = [set() for _ in range(n_nodes)]
    colour = np.empty(len(a), dtype=int)
    for k, (ia, ib) in enumerate(zip(a, b)):
        c = 0
        while c in used[ia] or c in used[ib]:
            c += 1
        colour[k] = c
        used[ia].add(c)
        used[ib].add(c)
    return [np.flatnonzero(colour == c) for c in range(int(colour.max()) + 1)]


def enforce_non_increasing(
    net: GvfNetwork,
    values: np.ndarray,
    tol_m: float = 1e-3,
    max_sweeps: int = 20000,
    fixed: np.ndarray | None = None,
) -> np.ndarray:
    """Make node values non-increasing downstream with one value per node.

    Weighted L2 isotonic regression on the whole node graph: the values
    closest to the input (weights = length of channel each node stands for)
    with ``v[a] >= v[b]`` on every node edge a -> b. Solved exactly by
    Hildreth's dual coordinate ascent, one constraint per edge, so loops
    (bifurcate-then-rejoin) and junctions are handled together rather than
    reach by reach. ``fixed`` nodes have infinite weight and keep their
    value. A last pass lifts any node still below its downstream
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
    if fixed is not None:
        inv_w[np.asarray(fixed, bool)] = 0.0
    denom = np.maximum(inv_w[a] + inv_w[b], 1e-12)
    lam = np.zeros(len(edges))
    batches = _edge_colouring(a, b, net.n_nodes)
    for _ in range(max_sweeps):
        # Edges in one batch share no node, so updating them together equals
        # updating them one by one.
        for k in batches:
            ia, ib = a[k], b[k]
            delta = np.maximum(-lam[k], (v[ib] - v[ia]) / denom[k])
            lam[k] += delta
            v[ia] += delta * inv_w[ia]
            v[ib] -= delta * inv_w[ib]
        if np.max(v[b] - v[a]) < tol_m:
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
        if succ[i] and not (fixed is not None and fixed[i]):
            v[i] = max(v[i], max(v[j] for j in succ[i]))
    lift = float(np.nanmax(v - before))
    if lift > 0.05:
        log.warning(f"GVF target: final monotone clip lifted a node by {lift:.2f} m")
    return v


def node_slope(net: GvfNetwork, values: np.ndarray) -> np.ndarray:
    """Return the downstream fall of ``values`` per metre at each node (central where possible)."""
    fall = np.zeros(net.n_nodes)
    dist = np.zeros(net.n_nodes)
    for a, b, d in net.node_edges():
        if d <= 0:
            continue
        for i in (a, b):
            fall[i] += values[a] - values[b]
            dist[i] += d
    return np.where(dist > 0, fall / np.maximum(dist, 1e-9), 0.0)


def main_path_slope(
    net: GvfNetwork, values: np.ndarray, reach_q: dict[str, float]
) -> np.ndarray:
    """Return the downstream fall of ``values`` per metre at each node along its largest-Q path.

    Inside a reach this is the central difference. At a junction the edges
    taken are those of the up- and downstream reaches with the largest
    discharge, so a small arm does not set the slope of the main channel.
    """
    best_dn: dict[int, tuple[float, int, float]] = {}
    best_up: dict[int, tuple[float, int, float]] = {}
    for r in net.reach_ids:
        q = float(reach_q.get(r, 0.0))
        ids, ch = net.reach_nodes[r], net.reach_chainage[r]
        for k in range(len(ids) - 1):
            a, b, d = int(ids[k]), int(ids[k + 1]), float(ch[k + 1] - ch[k])
            if d <= 0:
                continue
            if a not in best_dn or q > best_dn[a][0]:
                best_dn[a] = (q, b, d)
            if b not in best_up or q > best_up[b][0]:
                best_up[b] = (q, a, d)
    fall = np.zeros(net.n_nodes)
    dist = np.zeros(net.n_nodes)
    for i, (_, b, d) in best_dn.items():
        fall[i] += values[i] - values[b]
        dist[i] += d
    for i, (_, a, d) in best_up.items():
        fall[i] += values[a] - values[i]
        dist[i] += d
    return np.where(dist > 0, fall / np.maximum(dist, 1e-9), 0.0)


def main_reaches(net: GvfNetwork, reach_q: dict[str, float]) -> set[str]:
    """Return the reaches on the largest-Q path from every network head to its end."""
    up = net.upstream()
    main: set[str] = set()
    for head in (r for r in net.reach_ids if not up[r]):
        r = head
        while r not in main:
            main.add(r)
            dns = net.downstream[r]
            if not dns:
                break
            r = max(dns, key=lambda d: float(reach_q.get(d, 0.0)))
    return main


def classify_nodes(
    width_m: np.ndarray,
    sea_fraction: np.ndarray,
    dem_m: np.ndarray,
    gebco_floor_m: float,
    min_width_m: float = 1000.0,
    sea_class_fraction: float = 0.5,
) -> np.ndarray:
    """Return the class per node: water body, fixed bed or invert.

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
