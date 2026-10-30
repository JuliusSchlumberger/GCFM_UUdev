"""River bed below rule 10's water surface: Manning depth at a design discharge and the surface slope.

Entry point: :func:`estimate_river_bed` (rule modelled_depth_estimation, round 0).

At the low Froude numbers of these rivers the water-surface slope equals the
friction slope, also in backwater, so the local depth follows from Manning
with the surface slope; the surface places the bed.

1. Surface and slope: rule 10's node table (src.water_surface), on the same
   node network: the smoothed FathomDEM level, mouths pinned at
   ``calm_sea_level_m``, and its slope along each node's largest-Q path.
2. Depth: Manning normal depth of a rectangular channel (the section of the
   GVF solver) at ``max(slope, slope_floor)``: normal depth diverges on a flat
   surface, so the floor bounds it there while the surface keeps its own
   slope. Q is the GloFAS design discharge (mean or bankfull, rule 11) routed
   through the network, W the SWORD node width scaled so each reach's median
   matches the pipeline reach width (manual corrections included), as a
   running median and mean along the network (``width_smoothing_m``) that run
   across reach boundaries. Reaches with Q <= ``q_min_m3s`` take h = c W^b fitted to
   the others.
3. Bed = surface - h. Wide sea-class water bodies (where the DEM is GEBCO
   bathymetry, not a water surface) keep the DEM bed unless it sits at GEBCO's
   shallow clamp; river beds within ``bathymetry_handover_m`` of such a node
   blend linearly into it.
4. Check: a steady GVF solve on the bed from the surface at the network ends
   (src.gvf_solver.forward_network), at the design discharge, minus the
   surface. The depth uses the long-smoothed slope and the floor while the
   surface keeps the short smoothing, so the two differ by design: the
   deviation shows where the bed would carry the design discharge above or
   below the smoothed FathomDEM level. Logged for all river nodes and the
   largest-Q paths, and mapped by rule 11.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import substring
from src.gvf_solver import fit_width_depth, forward_network, normal_depth
from src.gvf_target import (
    NODE_WATER_BODY,
    GvfNetwork,
    build_node_network,
    main_reaches,
    orient_downstream,
    smooth_on_network,
)
from src.io import require_keys
from src.river_network import build_downstream_adjacency, normalize_reach_id

log = logging.getLogger(__name__)

CFG_KEYS = (
    "design_discharge",
    "slope_floor",
    "width_smoothing_m",
    "bathymetry_handover_m",
    "q_min_m3s",
)
FORWARD_BANDS_M = (0.5, 1.0)  # forward-check deviation bands reported
_ROW_CHUNK = 500


@dataclass
class RiverBed:
    """Per-node results on the node network (node order of ``net``)."""

    net: GvfNetwork
    fathom_level_m: np.ndarray
    dist_to_sea_m: np.ndarray
    slope: np.ndarray
    depth_slope: np.ndarray
    width_m: np.ndarray
    q_m3s: np.ndarray
    depth_m: np.ndarray
    level_m: np.ndarray
    bed_m: np.ndarray
    bathymetry: np.ndarray
    mouths: np.ndarray
    main_path: np.ndarray
    forward_level_m: np.ndarray
    mean_level_m: np.ndarray
    sea_level_m: float

    @property
    def informative(self) -> np.ndarray:
        """Mark nodes whose FathomDEM level is above the sea (below it shows the geoid, not a level)."""
        return np.isfinite(self.fathom_level_m) & (
            self.fathom_level_m > self.sea_level_m
        )

    @property
    def on_floor(self) -> np.ndarray:
        """Nodes whose depth uses the slope floor, not the surface slope."""
        return self.depth_slope > self.slope

    def node_table(self) -> pd.DataFrame:
        net = self.net
        return pd.DataFrame(
            {
                "node": np.arange(net.n_nodes),
                "reach_id": np.asarray(net.owner),
                "x": net.node_x,
                "y": net.node_y,
                "dist_to_sea_m": self.dist_to_sea_m,
                "fathom_level_m": self.fathom_level_m,
                "informative": self.informative,
                "slope": self.slope,
                "depth_slope": self.depth_slope,
                "on_floor": self.on_floor,
                "width_m": self.width_m,
                "q_m3s": self.q_m3s,
                "depth_m": self.depth_m,
                "level_m": self.level_m,
                "bed_m": self.bed_m,
                "bathymetry": self.bathymetry,
                "main_path": self.main_path,
                "forward_level_m": self.forward_level_m,
                "mean_level_m": self.mean_level_m,
            }
        )


def _edges(net: GvfNetwork):
    e = [(a, b, d) for a, b, d in net.node_edges() if d > 0]
    return (
        np.array([x[0] for x in e], int),
        np.array([x[1] for x in e], int),
        np.array([x[2] for x in e], float),
    )


def _neighbours(
    graph: csr_matrix, radius_m: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per node, (node ids, network distance) within radius_m, both directions."""
    n = graph.shape[0]
    out = []
    for start in range(0, n, _ROW_CHUNK):
        dist = dijkstra(
            graph,
            directed=False,
            indices=np.arange(start, min(start + _ROW_CHUNK, n)),
            limit=radius_m,
        )
        for row in dist:
            j = np.flatnonzero(np.isfinite(row))
            out.append((j, row[j]))
    return out


def node_widths(
    net,
    reach_width: dict[str, float],
    sword_nodes: pd.DataFrame,
    smoothing_m: float,
    spacing_m: float = 500.0,
) -> np.ndarray:
    """SWORD node width per node, scaled to the reach width, smoothed along the network.

    sword_nodes: columns reach_id, x, y (model CRS), width. Nodes of a reach
    without SWORD nodes take the reach width. A running median over
    ``smoothing_m`` drops single-node spikes, a running mean over the same
    length (src.gvf_target.smooth_on_network) spreads the steps the per-reach
    scaling leaves at reach boundaries; both run across reach boundaries.
    """
    owner = np.asarray(net.owner)
    width = np.array([reach_width.get(o, np.nan) for o in owner], dtype=float)
    sn = sword_nodes[sword_nodes["width"] > 0].copy()
    sn["rid"] = sn["reach_id"].map(normalize_reach_id)
    for r, g in sn.groupby("rid"):
        order = np.array([i for i in net.reach_nodes.get(r, []) if owner[i] == r], int)
        if not order.size:
            continue
        _, idx = cKDTree(np.c_[g["x"], g["y"]]).query(
            np.c_[net.node_x[order], net.node_y[order]]
        )
        scale = (
            reach_width[r] / float(np.median(g["width"]))
            if reach_width.get(r, 0) > 0
            else 1.0
        )
        width[order] = g["width"].to_numpy(float)[idx] * scale
    if smoothing_m <= 0:
        return width
    a, b, d = _edges(net)
    n = net.n_nodes
    graph = csr_matrix((np.r_[d, d], (np.r_[a, b], np.r_[b, a])), shape=(n, n))
    median = np.array(
        [np.nanmedian(width[j]) for j, _ in _neighbours(graph, smoothing_m / 2.0)]
    )
    return smooth_on_network(median, net.node_edges(), smoothing_m, spacing_m)


def manning_depth(q, w, n_manning: float, s) -> np.ndarray:
    """Normal depth (m) of a rectangular channel, hydraulic radius w h / (w + 2 h)."""
    q, w, s = np.broadcast_arrays(
        np.maximum(np.asarray(q, float), 0.0),
        np.maximum(np.asarray(w, float), 1.0),
        np.asarray(s, float),
    )
    h = [
        normal_depth(a, b, n_manning, c)
        for a, b, c in zip(q.ravel(), w.ravel(), s.ravel())
    ]
    return np.asarray(h, float).reshape(q.shape)


def surface_on_network(net: GvfNetwork, surface_nodes: pd.DataFrame) -> pd.DataFrame:
    """Return rule 10's node table indexed by node id, checked against ``net``.

    Rule 10 and rule 11 build the network from the same reaches and spacing,
    so node ids match; a mismatch means rule 10 ran on another network.
    """
    t = surface_nodes.drop_duplicates("node_id").set_index("node_id").sort_index()
    if len(t) != net.n_nodes or not np.array_equal(
        t.index.to_numpy(), np.arange(net.n_nodes)
    ):
        raise ValueError(
            f"rule 10's node table has {len(t)} nodes, the network {net.n_nodes}: rerun rule 10"
        )
    off = np.hypot(
        t.geometry.x.to_numpy() - net.node_x, t.geometry.y.to_numpy() - net.node_y
    )
    if off.max() > 1.0:
        raise ValueError(
            f"rule 10's node table is off the network by up to {off.max():.1f} m: rerun rule 10"
        )
    return t


def estimate_river_bed(
    rivers_m: gpd.GeoDataFrame,
    reach_q: dict[str, float],
    surface_nodes: gpd.GeoDataFrame,
    sword_nodes: pd.DataFrame,
    sea_level_m: float,
    gebco_shallow_m: float,
    manning_n: float,
    surface_cfg: dict,
    cfg: dict,
    reach_q_mean: dict[str, float] | None = None,
) -> RiverBed:
    """Estimate the river bed on the node network.

    rivers_m: cleaned network in the model CRS with 'reach_id', 'rch_id_dn',
    'width'. surface_nodes: rule 10's node table (src.water_surface). sword_nodes:
    SWORD nodes in the model CRS. gebco_shallow_m: shallowest clamp of the
    GEBCO bathymetry (rule 06). reach_q_mean: the routed mean discharge; its
    steady level on the bed (``mean_level_m``, NaN without it) is the initial
    river level of rules 11 and 12a.
    """
    require_keys(cfg, CFG_KEYS, "river_processing.river_depth_modelling.bed")
    downstream = build_downstream_adjacency(rivers_m)
    rivers_o, _ = orient_downstream(rivers_m, downstream)
    spacing = float(surface_cfg["node_spacing_m"])
    net = build_node_network(rivers_o, downstream, spacing)
    t = surface_on_network(net, surface_nodes)
    level = t["level_m"].to_numpy(float)
    slope = t["slope"].to_numpy(float)
    node_class = t["node_class"].to_numpy(object)
    dem = t["dem_m"].to_numpy(float)
    mouths = np.flatnonzero(t["is_mouth"].to_numpy(bool))
    a, b, d = _edges(net)
    n = net.n_nodes

    upstream_graph = csr_matrix((d, (b, a)), shape=(n, n))
    dist_to_sea = dijkstra(upstream_graph, directed=True, indices=mouths, min_only=True)
    if not np.isfinite(dist_to_sea).all():
        ends = sorted(set(range(n)) - set(a.tolist()))
        dist_to_sea = np.where(
            np.isfinite(dist_to_sea),
            dist_to_sea,
            dijkstra(upstream_graph, directed=True, indices=ends, min_only=True),
        )

    floor = float(cfg["slope_floor"])
    depth_slope = np.maximum(slope, floor)
    reach_width = dict(
        zip(
            rivers_o["reach_id"].map(normalize_reach_id),
            pd.to_numeric(rivers_o["width"], errors="coerce").astype(float),
        )
    )
    width = node_widths(
        net, reach_width, sword_nodes, float(cfg["width_smoothing_m"]), spacing
    )
    q = np.array([reach_q.get(o, 0.0) for o in net.owner], float)
    depth = manning_depth(q, width, manning_n, depth_slope)
    flowing = q > float(cfg["q_min_m3s"])
    if (~flowing).any():
        c, bexp = (
            fit_width_depth(width[flowing], depth[flowing])
            if flowing.any()
            else (1.0, 0.0)
        )
        depth = np.where(flowing, depth, c * width**bexp)
        log.info(
            f"River bed: {int((~flowing).sum())} node(s) without discharge, depth = {c:.3f} * width^{bexp:.2f}"
        )

    bed = level - depth
    bathy = (
        (node_class == NODE_WATER_BODY)
        & np.isfinite(dem)
        & (dem < gebco_shallow_m - 0.1)
    )
    handover = float(cfg["bathymetry_handover_m"])
    if bathy.any():
        graph = csr_matrix((np.r_[d, d], (np.r_[a, b], np.r_[b, a])), shape=(n, n))
        dist_b, _pred, src = dijkstra(
            graph,
            directed=False,
            indices=np.flatnonzero(bathy),
            min_only=True,
            return_predecessors=True,
            limit=handover,
        )
        bed = np.where(bathy, dem, bed)
        blend = ~bathy & np.isfinite(dist_b) & (src >= 0)
        w = np.clip(dist_b[blend] / handover, 0.0, 1.0)
        bed[blend] = (1.0 - w) * dem[src[blend]] + w * bed[blend]

    reach_q_net = {r: float(reach_q.get(r, 0.0)) for r in net.reach_ids}
    main = np.zeros(n, dtype=bool)
    for r in main_reaches(net, reach_q_net):
        main[np.asarray(net.reach_nodes[r])] = True
    ends = {
        int(net.reach_nodes[r][-1]): float(level[net.reach_nodes[r][-1]])
        for r in net.reach_ids
        if not net.downstream[r]
    }
    forward = forward_network(net, bed, width, reach_q_net, manning_n, ends)
    mean_level = (
        forward_network(
            net,
            bed,
            width,
            {r: float(reach_q_mean.get(r, 0.0)) for r in net.reach_ids},
            manning_n,
            ends,
        )
        if reach_q_mean is not None
        else np.full(n, np.nan)
    )

    result = RiverBed(
        net=net,
        fathom_level_m=t["level_raw_m"].to_numpy(float),
        dist_to_sea_m=dist_to_sea,
        slope=slope,
        depth_slope=depth_slope,
        width_m=width,
        q_m3s=q,
        depth_m=depth,
        level_m=level,
        bed_m=bed,
        bathymetry=bathy,
        mouths=mouths,
        main_path=main,
        forward_level_m=forward,
        mean_level_m=mean_level,
        sea_level_m=float(sea_level_m),
    )
    _log_summary(result)
    return result


DESIGN_DISCHARGE_VAR = {
    "mean": "mean_discharge_m3s",
    "bankfull": "bankfull_discharge_m3s",
}


def estimate_river_bed_from_files(
    rivers: gpd.GeoDataFrame,
    river_forcing_path,
    water_surface_path,
    sword_paths,
    wgs84_bounds: tuple[float, float, float, float],
    crs,
    sea_level_m: float,
    gebco_min_depth_m: float,
    manning_n: float,
    surface_cfg: dict,
    cfg: dict,
) -> tuple[RiverBed, np.ndarray, dict[str, float], np.ndarray]:
    """Run :func:`estimate_river_bed` on a basin's rule 08-10 outputs (rule 11 round 0).

    rivers: rule 08's cleaned network (any CRS). The design discharge
    (``cfg['design_discharge']``: mean or bankfull GloFAS discharge at the seed
    crossings) and the mean discharge are routed through the network as the
    calibration discharge is. Returns the bed, the design discharge per reach of
    ``rivers``, its seeds and the mean discharge per reach.
    """
    import xarray as xr
    from pyproj import Transformer
    from src.river_forcing import seed_discharge_by_reach
    from src.river_network import accumulate_discharge, outflow_crossing_mask
    from src.sword import read_sword_nodes
    from src.water_surface import read_water_surface

    design = str(cfg["design_discharge"])
    if design not in DESIGN_DISCHARGE_VAR:
        raise ValueError(
            f"river_depth_modelling.bed.design_discharge must be one of {sorted(DESIGN_DISCHARGE_VAR)}, got {design!r}"
        )
    excluded = (
        {
            normalize_reach_id(r)
            for r, f in zip(rivers["reach_id"], rivers["zero_discharge_share"])
            if bool(f)
        }
        if "zero_discharge_share" in rivers.columns
        else set()
    )
    with xr.open_dataset(river_forcing_path, decode_times=False) as rds:
        active = rds["has_glofas"].values.astype(bool) & ~outflow_crossing_mask(
            rds, rivers
        )
        seed_q = seed_discharge_by_reach(
            rds, rds[DESIGN_DISCHARGE_VAR[design]].values[active], active
        )
        seed_q_mean = seed_discharge_by_reach(
            rds, rds["mean_discharge_m3s"].values[active], active
        )
    adjacency = build_downstream_adjacency(rivers)
    q_design = accumulate_discharge(rivers, seed_q, adjacency, excluded_arms=excluded)
    q_mean = accumulate_discharge(
        rivers, seed_q_mean, adjacency, excluded_arms=excluded
    )
    reach_ids = rivers["reach_id"].map(normalize_reach_id)

    surface_nodes, _structures = read_water_surface(
        water_surface_path
    )  # structures are reported only
    if surface_nodes.crs != crs:
        raise ValueError(
            f"water surface CRS {surface_nodes.crs} differs from the model CRS {crs}"
        )
    sword = read_sword_nodes(sword_paths, wgs84_bounds, rivers["reach_id"])
    sx, sy = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
        sword["x"].to_numpy(), sword["y"].to_numpy()
    )
    bed = estimate_river_bed(
        rivers.to_crs(crs),
        dict(zip(reach_ids, np.asarray(q_design, dtype=float))),
        surface_nodes,
        sword.assign(x=sx, y=sy),
        sea_level_m=sea_level_m,
        gebco_shallow_m=sea_level_m - gebco_min_depth_m,
        manning_n=manning_n,
        surface_cfg=surface_cfg,
        cfg=cfg,
        reach_q_mean=dict(zip(reach_ids, np.asarray(q_mean, dtype=float))),
    )
    return (
        bed,
        np.asarray(q_design, dtype=float),
        seed_q,
        np.asarray(q_mean, dtype=float),
    )


def forward_deviation_summary(rb: RiverBed) -> dict[str, float]:
    """Forward level minus the surface on flowing, non-bathymetry nodes: all, and the largest-Q paths."""
    dev = rb.forward_level_m - rb.level_m
    river = (rb.q_m3s > 0) & ~rb.bathymetry
    out = {}
    for k, m in {"river": river, "main": river & rb.main_path}.items():
        d = dev[m]
        out[f"{k}_nodes"] = float(m.sum())
        for name, q in (("p5", 5), ("median", 50), ("p95", 95)):
            out[f"{k}_{name}_m"] = float(np.percentile(d, q)) if d.size else np.nan
        for band in FORWARD_BANDS_M:
            out[f"{k}_within_{band:g}m"] = (
                float(np.mean(np.abs(d) <= band)) if d.size else np.nan
            )
    return out


def _log_summary(rb: RiverBed) -> None:
    fwd = forward_deviation_summary(rb)
    drift = (rb.level_m - rb.fathom_level_m)[rb.informative]
    log.info(
        f"River bed: {rb.net.n_nodes} nodes, {len(rb.mouths)} sea mouth(s); surface slope p5/50/95 "
        f"{np.round(np.percentile(rb.slope, [5, 50, 95]), 7).tolist()}; depth on the slope floor at "
        f"{rb.on_floor.mean():.0%} of nodes; depth p50/95/max "
        f"{np.round([np.median(rb.depth_m), np.percentile(rb.depth_m, 95), rb.depth_m.max()], 2).tolist()} m; "
        f"surface - FathomDEM on {int(rb.informative.sum())} informative node(s) median "
        f"{np.median(drift) if drift.size else np.nan:+.2f} m; {int(rb.bathymetry.sum())} GEBCO bathymetry node(s); "
        f"sea level {rb.sea_level_m:+.2f} m"
    )
    for k, name in (("river", "all river nodes"), ("main", "largest-Q paths")):
        bands = ", ".join(
            f"{fwd[f'{k}_within_{b:g}m']:.0%} within {b:g} m" for b in FORWARD_BANDS_M
        )
        log.info(
            f"River bed forward check (steady GVF on the bed at the design discharge minus the surface), "
            f"{name} ({fwd[f'{k}_nodes']:.0f}): p5/median/p95 {fwd[f'{k}_p5_m']:+.2f}/"
            f"{fwd[f'{k}_median_m']:+.2f}/{fwd[f'{k}_p95_m']:+.2f} m; {bands}"
        )


def values_at_points(
    net: GvfNetwork, per_node: np.ndarray, reach_ids, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Interpolate a per-node quantity onto points along their own reach (NaN off the network)."""
    out = np.full(len(x), np.nan)
    norm = np.array([normalize_reach_id(r) for r in reach_ids], dtype=object)
    for r in set(norm):
        if r not in net.reach_line:
            continue
        idx = np.flatnonzero(norm == r)
        line = net.reach_line[r]
        ch = np.array([line.project(Point(float(x[i]), float(y[i]))) for i in idx])
        out[idx] = np.interp(
            ch, net.reach_chainage[r], per_node[np.asarray(net.reach_nodes[r])]
        )
    return out


def depth_segments(rb: RiverBed, crs=None) -> gpd.GeoDataFrame:
    """Return reach pieces between consecutive nodes with their depth, width and forward-check deviation."""
    net = rb.net
    depth = rb.level_m - rb.bed_m
    deviation = rb.forward_level_m - rb.level_m
    floor = rb.on_floor
    rows = []
    for r in net.reach_ids:
        line, ch, nodes = (
            net.reach_line[r],
            net.reach_chainage[r],
            np.asarray(net.reach_nodes[r]),
        )
        for k in range(len(ch) - 1):
            piece = substring(line, float(ch[k]), float(ch[k + 1]))
            if not isinstance(piece, LineString) or piece.length == 0:
                continue
            rows.append(
                {
                    "reach_id": r,
                    "depth": float(np.mean(depth[nodes[k : k + 2]])),
                    "width": float(np.mean(rb.width_m[nodes[k : k + 2]])),
                    "forward_minus_surface_m": float(
                        np.mean(deviation[nodes[k : k + 2]])
                    ),
                    "fallback": bool(floor[nodes[k]]),
                    "geometry": piece,
                }
            )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
