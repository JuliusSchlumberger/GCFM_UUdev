"""River network graph operations: adjacency, BFS connectivity, flow accumulation, depth."""

from __future__ import annotations

import logging
import re
from collections import deque

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import LineString, Point

from src.geometry import pick_utm_crs

log = logging.getLogger(__name__)


# Minimum distance (m) an own anchor must already be from a reach's start/end
# before a borrowed neighbour junction value is skipped as redundant (used by
# src.river_burn.burn_river_channel/_smoothed_weir_crest_profiles --
# everywhere a per-point profile is extended with a synthetic
# junction-boundary anchor).
_BOUNDARY_BLEND_EPS_M = 1.0


def _junction_value(
    reach_data: dict[str, tuple[object, np.ndarray, np.ndarray]],
    up_rid: str,
    dn_rid: str,
) -> float | None:
    """
    Shared value at the junction between two adjacent reaches: the average
    of the upstream reach's own anchor nearest ITS downstream end and the
    downstream reach's own anchor nearest ITS upstream start.

    Using this SAME averaged value as the synthetic boundary anchor on BOTH
    sides of the junction (rather than each side simply adopting the other
    side's raw value) is what actually removes the step there -- both
    reaches' interpolation then passes through an identical value at the
    shared coordinate. Adopting the other side's raw value outright would
    just swap which side has the mismatch instead of removing it.

    Generic over whatever per-point quantity `reach_data` carries (river bed
    elevation in src.river_burn.burn_river_channel, calibrated weir crest in
    src.river_burn's smoothed-crest profiles) -- reach_data maps reach_id
    to (line, along_sorted, value_sorted).
    """
    up_entry = reach_data.get(up_rid)
    dn_entry = reach_data.get(dn_rid)
    if up_entry is None or dn_entry is None:
        return None
    _, up_along, up_vals = up_entry
    _, dn_along, dn_vals = dn_entry
    if len(up_along) == 0 or len(dn_along) == 0:
        return None
    up_val = float(up_vals[np.argmax(up_along)])
    dn_val = float(dn_vals[np.argmin(dn_along)])
    return (up_val + dn_val) / 2.0


def normalize_reach_id(x) -> str | None:
    """Normalize a SWORD reach_id value (int/float/str) to a canonical string, or None if NA."""
    if pd.isna(x):
        return None
    try:
        return str(int(float(x)))
    except (ValueError, TypeError):
        s = str(x).strip()
        return s if s else None


def _as_linestring(geom):
    """Return a simple LineString from geom, extracting a sub-geometry if needed."""

    if hasattr(geom, "geoms"):
        # Multi-part: pick the longest component
        parts = [g for g in geom.geoms if hasattr(g, "coords")]
        if not parts:
            return None
        return max(parts, key=lambda g: g.length)
    return geom


def collect_downstream_main_paths(
    rivers: gpd.GeoDataFrame,
    seed_reach_ids: set[str],
) -> set[str]:
    """
    Collect all reaches reachable downstream from seed reaches by BFS over
    the rch_id_dn adjacency.

    Delegates parsing to build_downstream_adjacency(), then follows all
    downstream links (including bifurcations) with a simple breadth-first
    search.  The returned set contains every retained reach; the complement
    within the original GeoDataFrame gives the discarded reaches for plotting.

    Args:
        rivers:          River network GeoDataFrame with 'reach_id' and
                         'rch_id_dn'.
        seed_reach_ids:  Reach IDs to start from (boundary forcing crossings).

    Returns:
        Set of reach_id strings reachable downstream from any seed,
        including the seeds themselves.
    """
    adjacency = build_downstream_adjacency(rivers)

    visited: set[str] = set()
    queue: deque[str] = deque(seed_reach_ids)
    while queue:
        rid = queue.popleft()
        if rid in visited:
            continue
        visited.add(rid)
        for dn_id in adjacency.get(rid, []):
            if dn_id not in visited:
                queue.append(dn_id)

    log.info(
        f"collect_downstream_main_paths: {len(visited)} reachable reaches "
        f"from {len(seed_reach_ids)} seeds"
    )
    return visited


def build_downstream_adjacency(
    rivers: gpd.GeoDataFrame,
) -> dict[str, list[str]]:
    """
    Parse the 'rch_id_dn' column to build a downstream adjacency map.

    'rch_id_dn' is stored as a Python-list-repr string, e.g.
    "[id1, id2]" (SWORD v17c's convention; splitting on comma-or-whitespace
    below is deliberately also robust to a plain whitespace-separated
    string, e.g. "id1 id2", in case a future source uses that convention
    instead).
    Only reach IDs that exist within the provided GeoDataFrame are retained;
    references outside the domain are silently dropped.

    Args:
        rivers: River network GeoDataFrame with 'reach_id' and 'rch_id_dn'.

    Returns:
        Dict mapping each reach_id string to a list of downstream reach_id
        strings that exist within the domain.
    """

    all_ids = {normalize_reach_id(x) for x in rivers["reach_id"]} - {None}
    adjacency: dict[str, list[str]] = {}
    for rid_raw, dn_raw in zip(rivers["reach_id"], rivers["rch_id_dn"]):
        rid = normalize_reach_id(rid_raw)
        if rid is None:
            continue
        dn_ids: list[str] = []
        if pd.notna(dn_raw):
            s = str(dn_raw).strip().strip("[]")
            if s:
                for token in re.split(r"[,\s]+", s):
                    normed = normalize_reach_id(token.strip())
                    if normed and normed in all_ids:
                        dn_ids.append(normed)
        adjacency[rid] = dn_ids
    return adjacency


def fix_tjunction_tails(
    rivers: gpd.GeoDataFrame,
    reach_id_col: str = "reach_id",
    snap_tolerance_m: float = 0.1,
    endpoint_tolerance_m: float = 2.0,
    max_search_radius_m: float = 500.0,
) -> gpd.GeoDataFrame:
    """
    Fix T-junction topology where a downstream reach B starts on the interior
    of upstream reach A rather than at A's true endpoint.

    This is a systematic artifact in SWORD: at bifurcations, one downstream
    branch (B) has its start vertex recorded at 98-99% along A's geometry
    rather than at A's true endpoint.  The tail between the T-junction and
    A's true endpoint (typically 18-111 m) becomes an orphan fragment in
    shapely.union_all/linemerge, fragmenting the merged-line connectivity
    that burn_river_rect relies on for bed-level interpolation.

    Fix per T-junction:
    1. Find C: the reach whose start lies nearest to A's true endpoint
       (the other downstream branch at the real bifurcation node).
    2. Extract the tail: the last segment of A from the T-junction to
       A's true endpoint.
    3. Trim A at the T-junction: A's new endpoint = B's start = C's
       new start, making A→B and A→C both proper end-to-end connections.
    4. Prepend the tail to C so C absorbs it; B is unchanged.

    Should be called on the raw (unfiltered) river network before any
    downstream filtering, since A, B, and C may not all be in the
    reachable set.

    Args:
        rivers:               River network GeoDataFrame (any CRS).
        reach_id_col:         Column holding the reach identifier.
        snap_tolerance_m:     Max distance (m) for B.start to be considered
                              "on" A's line (detected T-junction).
        endpoint_tolerance_m: Min distance (m) from A's true endpoint
                              required for B.start to count as mid-reach
                              rather than a legitimate endpoint coincidence.
        max_search_radius_m:  Max distance (m) within which C's start must
                              lie from A's true endpoint; if no reach is
                              found within this radius the tail is dropped.

    Returns:
        Copy of ``rivers`` (original CRS) with A reaches trimmed and C
        reaches extended.  Geometry is unchanged for B and for any reach
        not involved in a T-junction.
    """
    from shapely.ops import substring as _substring

    metric_crs = pick_utm_crs(rivers) if rivers.crs.is_geographic else rivers.crs
    rivers_m = rivers.to_crs(metric_crs)

    rid_to_idx: dict[str, int] = {}
    rid_to_geom: dict[str, LineString] = {}
    for idx, row in rivers_m.iterrows():
        rid = normalize_reach_id(row.get(reach_id_col))
        if rid is None:
            continue
        geom = _as_linestring(row.geometry)
        if geom is not None and geom.length > 0:
            rid_to_idx[rid] = idx
            rid_to_geom[rid] = geom

    # ── detect T-junctions ────────────────────────────────────────────────
    junctions: list[tuple[str, str, float]] = []  # (rid_a, rid_b, junction_along_m)
    for rid_b, geom_b in rid_to_geom.items():
        start_b = Point(geom_b.coords[0])
        for rid_a, geom_a in rid_to_geom.items():
            if rid_a == rid_b:
                continue
            if start_b.distance(geom_a) > snap_tolerance_m:
                continue
            if start_b.distance(Point(geom_a.coords[-1])) <= endpoint_tolerance_m:
                continue
            if start_b.distance(Point(geom_a.coords[0])) <= endpoint_tolerance_m:
                continue
            junctions.append((rid_a, rid_b, geom_a.project(start_b)))

    if not junctions:
        log.info("fix_tjunction_tails: no T-junctions found")
        return rivers

    log.info(f"fix_tjunction_tails: {len(junctions)} T-junction(s) found")

    # ── apply repairs (work on metric-CRS geometries) ─────────────────────
    new_geom_m: dict[str, LineString] = {}  # only modified reaches

    for rid_a, rid_b, junction_along in junctions:
        geom_a = new_geom_m.get(rid_a, rid_to_geom[rid_a])
        true_end_a = Point(geom_a.coords[-1])

        tail = _substring(geom_a, junction_along, geom_a.length)
        trimmed_a = _substring(geom_a, 0.0, junction_along)
        if (
            trimmed_a is None
            or trimmed_a.is_empty
            or trimmed_a.geom_type != "LineString"
        ):
            log.warning(f"fix_tjunction_tails: could not trim A={rid_a}; skipping")
            continue
        new_geom_m[rid_a] = trimmed_a

        # Find C: reach with start nearest to A's original true endpoint
        best_rid_c, best_dist_c = None, float("inf")
        for rid_c, geom_c in rid_to_geom.items():
            if rid_c in (rid_a, rid_b):
                continue
            d = true_end_a.distance(Point(geom_c.coords[0]))
            if d < best_dist_c:
                best_dist_c, best_rid_c = d, rid_c

        if best_rid_c is None or best_dist_c > max_search_radius_m:
            log.warning(
                f"fix_tjunction_tails: no reach C found within {max_search_radius_m:.0f}m "
                f"of A={rid_a} endpoint (best={best_dist_c:.1f}m) — tail discarded"
            )
            continue

        geom_c = new_geom_m.get(best_rid_c, rid_to_geom[best_rid_c])
        tail_coords = list(tail.coords)
        c_coords = list(geom_c.coords)
        merged = tail_coords + (
            c_coords[1:] if tail_coords[-1] == c_coords[0] else c_coords
        )
        if len(merged) >= 2:
            new_geom_m[best_rid_c] = LineString(merged)

        log.info(
            f"  A={rid_a} trimmed at {junction_along:.1f}m; "
            f"tail ({tail.length:.1f}m) prepended to C={best_rid_c}"
        )

    # ── convert modified geometries back to original CRS ──────────────────
    if not new_geom_m:
        return rivers

    modified_gdf = gpd.GeoDataFrame(
        {"rid": list(new_geom_m.keys()), "geometry": list(new_geom_m.values())},
        crs=metric_crs,
    ).to_crs(rivers.crs)

    out = rivers.copy()
    for _, row in modified_gdf.iterrows():
        idx = rid_to_idx.get(row["rid"])
        if idx is not None:
            out.at[idx, "geometry"] = row["geometry"]

    return out


def outward_seed_reaches(
    rivers: gpd.GeoDataFrame, seed_ids, adjacency: dict[str, list[str]]
) -> set[str]:
    """
    Seed reaches that SWORD routes OUT of the network rather than into it.

    A seed is where a GloFAS boundary crossing injects river discharge. A
    real inflow seed has no upstream reach inside the network (the network
    is built downstream from the seeds). One that does is fed by the delta
    itself and carries water out through the crossing -- e.g. basin
    1248635's Drecht crossing, on a branch running from the Hollandsche
    IJssel north out of the domain. Its crossing is an outflow, not an
    inflow.

    Args:
        rivers:    Network with 'reach_id'.
        seed_ids:  Seed reach ids (any iterable).
        adjacency: Downstream adjacency (build_downstream_adjacency).

    Returns:
        Normalised reach ids of the seeds that have in-network upstream reaches.
    """
    ids = {normalize_reach_id(r) for r in rivers["reach_id"]} - {None}
    has_upstream = {
        d for u, dns in adjacency.items() if u in ids for d in dns if d in ids
    }
    return {normalize_reach_id(s) for s in seed_ids} & has_upstream


def outflow_crossing_mask(river_ds, rivers: gpd.GeoDataFrame) -> np.ndarray:
    """
    Boolean mask over river_forcing.nc's crossings: True where the crossing's
    reach is a seed that flows OUT of the network (rule 08's
    'seed_flows_out', see outward_seed_reaches) -- its discharge must not be
    injected. All False when the column or 'inside_reach_id' is absent.
    """
    n = int(river_ds["has_glofas"].size)
    if "seed_flows_out" not in rivers.columns or "inside_reach_id" not in river_ds:
        return np.zeros(n, dtype=bool)
    flagged = {
        normalize_reach_id(r)
        for r, f in zip(rivers["reach_id"], rivers["seed_flows_out"])
        if bool(f)
    } - {None}
    return np.array(
        [normalize_reach_id(r) in flagged for r in river_ds["inside_reach_id"].values],
        dtype=bool,
    )


def canal_excluded_arms(
    rivers: gpd.GeoDataFrame,
    adjacency: dict[str, list[str]],
    force_exclude=(),
    force_include=(),
    auto: bool = True,
) -> set[str]:
    """
    Bifurcation arms that get no share of the upstream discharge: canals.

    With ``auto``, an arm of a bifurcation is excluded when it -- or any
    reach on its chain down to the next junction -- is a SWORD canal
    (``lakeflag == 2``) AND at least one other arm of the same bifurcation
    is not: where every arm is canal-flagged there is no river to prefer,
    and the plain width split is kept. Basin 1248635: the Maas-Waal Canal
    (Meuse -> Waal), the Lek -> Amsterdam-Rijnkanaal -> Vecht offtake, and
    the Hollandsche IJssel -> Drecht branch.

    Locks (SWORD ``obstr_type == 2``) are deliberately NOT used: SWORD
    marks the weir-and-lock complexes on the main canalised rivers (Meuse at
    Grave and Lith, Nederrijn-Lek) as locks too, so a lock does not tell a
    canal from a canalised river. A canal that SWORD does not flag (1248635:
    the Amsterdam-Rijnkanaal leaving the Waal at Tiel, reach 23261000641)
    needs ``force_exclude``.

    ``force_exclude`` / ``force_include`` (reach ids) override the automatic
    choice; a forced exclusion applies wherever that reach is an arm.

    Returns:
        Normalised reach ids of the excluded arms.
    """
    ids = {normalize_reach_id(r) for r in rivers["reach_id"]} - {None}
    attrs = rivers.assign(
        _rid=[normalize_reach_id(r) for r in rivers["reach_id"]]
    ).set_index("_rid")
    lakeflag = (
        attrs["lakeflag"]
        if "lakeflag" in attrs.columns
        else pd.Series(0, index=attrs.index)
    )
    down = {u: [d for d in dns if d in ids] for u, dns in adjacency.items() if u in ids}
    n_up: dict[str, int] = {r: 0 for r in ids}
    for dns in down.values():
        for d in dns:
            n_up[d] += 1

    def _is_canal(r: str) -> bool:
        return int(lakeflag.get(r, 0)) == 2

    def _arm_is_canal(arm: str) -> bool:
        node, seen = arm, set()
        while node is not None and node not in seen:
            seen.add(node)
            if _is_canal(node):
                return True
            nxt = down.get(node, [])
            node = nxt[0] if len(nxt) == 1 and n_up.get(nxt[0], 0) == 1 else None
        return False

    forced_in = {normalize_reach_id(r) for r in force_include}
    forced_out = {normalize_reach_id(r) for r in force_exclude}
    excluded: set[str] = set()
    for u, dns in down.items():
        if len(dns) < 2:
            continue
        canal_arms = {a for a in dns if auto and _arm_is_canal(a)} - forced_in
        if canal_arms == set(dns):
            canal_arms = set()
        for arm in dns:
            if arm in forced_out or arm in canal_arms:
                excluded.add(arm)
    excluded |= {r for r in forced_out if r in ids}
    if excluded:
        log.info(f"Canal arms with zero discharge share: {sorted(excluded)}")
    return excluded


def accumulate_discharge(
    rivers: gpd.GeoDataFrame,
    seed_q: dict[str, float],
    adjacency: dict[str, list[str]],
    excluded_arms: set[str] | None = None,
) -> np.ndarray:
    """
    Width-weighted downstream flow accumulation, one pass in topological order.

    Every reach receives the discharge of its upstream reaches: summed at a
    confluence, and at a bifurcation each upstream reach's discharge is split
    between its downstream reaches in proportion to their ``width``. A seed
    reach adds its own boundary discharge ONCE, on top of whatever flows in
    from upstream reaches inside the network -- so a seed that also has
    in-network upstream reaches (e.g. a tributary crossing whose reach is fed
    by a delta branch too) gets inflow + seed value.

    Until 2026-10-12 this was an iterative scheme (``n_iterations`` hops) that
    held seeds fixed and then added their inflow again on every iteration:
    a seed with upstream reaches gained that inflow once per iteration
    (basin 1248635's Drecht seed: 245 m3/s at the crossing, 103,693 m3/s
    after 100 iterations). The result no longer depends on any iteration
    count.

    The network is expected to be a directed acyclic graph (bifurcations
    that rejoin are fine; a directed cycle is not). Reaches on a directed
    cycle are logged and processed last, receiving only what had reached
    them from outside the cycle.

    Args:
        rivers:    Cleaned river network GeoDataFrame with 'reach_id' and
                   'width'. Width is assumed valid (non-missing, non-zero) --
                   callers run this after remove_reaches_with_missing_width/
                   normalize_channel_widths. A bifurcation whose downstream
                   widths sum to zero is split equally.
        seed_q:    {reach_id_str: discharge} from the boundary-forcing
                   snapping step (multiple crossings on one reach pre-summed).
        adjacency: Downstream adjacency from build_downstream_adjacency().
        excluded_arms: Reach ids that get no share when they are an arm of a
                   bifurcation (canals, see canal_excluded_arms); the other
                   arms split the discharge by width. If every arm of a
                   bifurcation is excluded, the plain width split is used.

    Returns:
        1-D array of accumulated discharge (m3 s-1) in the order of
        ``rivers``; 0.0 for reaches not downstream of any seed.
    """
    rids = [normalize_reach_id(r) for r in rivers["reach_id"]]
    if len(rids) == 0:
        log.warning(
            "accumulate_discharge: river network is empty; returning zero-length array"
        )
        return np.zeros(0)
    excluded = {normalize_reach_id(r) for r in (excluded_arms or set())}
    widths = rivers["width"].to_numpy(dtype=float)
    reach_to_idx = {r: i for i, r in enumerate(rids) if r is not None}

    down: dict[int, list[int]] = {
        i: [reach_to_idx[d] for d in adjacency.get(rid, []) if d in reach_to_idx]
        for i, rid in enumerate(rids)
    }
    n_up = np.zeros(len(rids), dtype=int)
    for dns in down.values():
        for d in dns:
            n_up[d] += 1

    q = np.zeros(len(rids))
    for rid, val in seed_q.items():
        rid = normalize_reach_id(rid)
        if rid in reach_to_idx:
            q[reach_to_idx[rid]] += float(val)
    seeds_with_inflow = [
        rid
        for rid in seed_q
        if normalize_reach_id(rid) in reach_to_idx
        and n_up[reach_to_idx[normalize_reach_id(rid)]] > 0
    ]

    # Kahn's algorithm: a reach is passed on only once all its upstream
    # reaches have delivered into it.
    remaining = n_up.copy()
    queue = deque(i for i in range(len(rids)) if remaining[i] == 0)
    order: list[int] = []
    while queue:
        i = queue.popleft()
        order.append(i)
        for d in down[i]:
            remaining[d] -= 1
            if remaining[d] == 0:
                queue.append(d)
    on_cycle = [i for i in range(len(rids)) if remaining[i] > 0]
    if on_cycle:
        log.warning(
            f"accumulate_discharge: {len(on_cycle)} reach(es) on a directed cycle "
            f"(e.g. {[rids[i] for i in on_cycle[:5]]}) -- processed last, with only "
            "the inflow from outside the cycle"
        )
        order.extend(on_cycle)

    inflow = np.zeros(len(rids))
    for i in order:
        dns = down[i]
        if not dns or q[i] <= 0:
            continue
        w_dn = widths[dns].copy()
        if len(dns) > 1 and excluded:
            keep = np.array([rids[d] not in excluded for d in dns])
            if keep.any():
                w_dn = np.where(keep, w_dn, 0.0)
                if w_dn.sum() <= 0:
                    w_dn = keep.astype(float)
        total_w = w_dn.sum()
        shares = w_dn / total_w if total_w > 0 else np.full(len(dns), 1.0 / len(dns))
        for d, share in zip(dns, shares):
            q[d] += q[i] * share
            inflow[d] += q[i] * share

    for rid in seeds_with_inflow:
        idx = reach_to_idx[normalize_reach_id(rid)]
        log.warning(
            f"accumulate_discharge: seed reach {rid} also has upstream reaches in the "
            f"network -- its own boundary discharge {float(seed_q[rid]):.1f} m3/s is added "
            f"to {inflow[idx]:.1f} m3/s of in-network inflow (total {q[idx]:.1f} m3/s); "
            "check that the forcing at that crossing does not already include that flow"
        )
    log.info(
        f"Flow accumulation: {int((q > 0).sum())}/{len(rids)} reach(es) with Q > 0, "
        f"{len([r for r in seed_q if normalize_reach_id(r) in reach_to_idx])} seed reach(es), "
        f"max Q = {q.max():.2f} m3 s-1"
    )
    return q


def identify_delta_outflow_points(
    rivers: gpd.GeoDataFrame,
    delta_polygon,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Identify river reaches that cross the delta polygon's OUTLINE (the
    boundary ring, not its filled interior) but are neither the seed
    (inflow boundary point) nor a mouth (network-terminal outflow point)
    there, and mark their outline-crossing location as a hydrodynamic
    outflow boundary point -- a place where flow genuinely exits the
    modelled river network into the delta plain (e.g. a distributary
    channel not itself flagged as a network-terminal "mouth"), rather
    than a data artefact to discard.

    Reaches entirely inside or entirely outside the delta polygon never
    qualify, regardless of 'is_seed'/mouth status -- only a reach whose
    geometry actually crosses the boundary line is a candidate at all.
    Checking against the filled polygon interior instead would flag nearly
    every reach in a delta network, which is not the intent.

    A qualifying reach -- crosses the delta outline, 'is_seed' is False,
    isn't a mouth (no downstream neighbour in-network), AND isn't a
    bifurcation (a reach with >1 downstream neighbour is always a valid,
    ordinary network junction, never treated as an outflow location) --
    is KEPT UNCHANGED in the network (no removal, no rch_id_dn rewriting),
    flagged via a new 'is_delta_outflow' column, and its exact intersection
    point(s) with the delta outline are collected into a separate output
    GeoDataFrame for use as an SFINCS outflow boundary (mask=3). Must run
    AFTER 'is_seed' has been assigned.

    Args:
        rivers:        River network with 'reach_id', 'rch_id_dn', 'is_seed'
                       columns and a geometry column.
        delta_polygon: Shapely geometry or GeoDataFrame of the delta
                       boundary (any CRS -- reprojected to match ``rivers``
                       internally). Only its outline (.boundary) is used.

    Returns:
        (rivers, outflow_points): ``rivers`` is the input GeoDataFrame with
        an added boolean 'is_delta_outflow' column (otherwise unchanged);
        ``outflow_points`` is a GeoDataFrame (columns: 'reach_id',
        geometry -- Point, same CRS as ``rivers``) with one row per
        outline-crossing point (a reach whose geometry crosses the outline
        more than once contributes more than one row). Empty (but
        correctly-shaped) when no reach qualifies.
    """
    if hasattr(delta_polygon, "crs"):
        if delta_polygon.crs is not None and delta_polygon.crs != rivers.crs:
            delta_polygon = delta_polygon.to_crs(rivers.crs)
        delta_geom = delta_polygon.geometry.union_all()
    else:
        delta_geom = delta_polygon
    delta_outline = delta_geom.boundary

    downstream_adj = build_downstream_adjacency(rivers)
    mouth_ids = {rid for rid, dns in downstream_adj.items() if not dns}

    rivers = rivers.copy()
    rids = rivers["reach_id"].apply(normalize_reach_id)
    is_seed = (
        rivers["is_seed"].fillna(False).astype(bool)
        if "is_seed" in rivers.columns
        else False
    )
    crosses_outline = rivers.geometry.intersects(delta_outline)

    is_bifurcation = rids.map(lambda rid: len(downstream_adj.get(rid, [])) > 1)
    qualifying_mask = (
        crosses_outline.to_numpy()
        & ~np.asarray(is_seed)
        & ~rids.isin(mouth_ids)
        & ~is_bifurcation.to_numpy()
    )
    rivers["is_delta_outflow"] = qualifying_mask

    empty_points = gpd.GeoDataFrame(
        {"reach_id": pd.Series(dtype=str)}, geometry=[], crs=rivers.crs
    )
    if not qualifying_mask.any():
        log.info("identify_delta_outflow_points: no qualifying reach(es) found")
        return rivers, empty_points

    records = []
    for rid, geom in zip(rids[qualifying_mask], rivers.geometry[qualifying_mask]):
        crossing = geom.intersection(delta_outline)
        points = list(crossing.geoms) if hasattr(crossing, "geoms") else [crossing]
        for pt in points:
            if pt.is_empty:
                continue
            # A tangential intersection can degenerate to a LineString/point
            # cluster rather than a clean Point -- reduce to its centroid so
            # the output is always a single Point per crossing.
            pt = pt if isinstance(pt, Point) else pt.centroid
            records.append({"reach_id": rid, "geometry": pt})

    outflow_points = gpd.GeoDataFrame(records, crs=rivers.crs)
    log.info(
        f"identify_delta_outflow_points: {int(qualifying_mask.sum())} reach(es) "
        f"flagged 'is_delta_outflow' (cross delta outline, not seed, not "
        f"mouth, not a bifurcation) -> {len(outflow_points)} outflow point(s)"
    )
    return rivers, outflow_points


def remove_reaches_with_missing_width(
    rivers: gpd.GeoDataFrame,
    nodata_value: float = -9999.0,
) -> gpd.GeoDataFrame:
    """
    Handle SWORD reaches with missing (nodata_value) 'width'/'max_width'.

    Must run BEFORE fix_width_max_width_order/clip_anomalous_max_width:
    fix_width_max_width_order's swap condition (max_width < width) fires
    unconditionally whenever max_width is the nodata sentinel (nodata_value
    < any real width), silently moving the valid value into max_width and
    leaving 'width' corrupted with the sentinel instead.

    * Exactly one of the two is missing: the reach's channel geometry is
      still known from the other value, so both columns are set to it.
    * Both are missing (no channel-width information at all): the reach is
      removed, along with the unbranched chain of neighbouring reaches
      extending downstream to the next confluence (a reach with >1 upstream
      neighbour) and upstream to the next bifurcation (a reach with >1
      downstream neighbour) -- a missing width usually reflects a broader
      data gap (e.g. a masked/obscured river segment in the SWORD source
      imagery) affecting the whole unbranched stretch, not just one reach.

    Args:
        rivers:       River network with 'reach_id', 'rch_id_dn', 'width',
                      and 'max_width' columns.
        nodata_value: SWORD's missing-value sentinel.

    Returns:
        Copy of ``rivers`` with the fixed/removed reaches applied.
    """
    rivers = rivers.copy()
    width = rivers["width"].to_numpy(dtype=float)
    max_width = rivers["max_width"].to_numpy(dtype=float)
    width_missing = np.isclose(width, nodata_value)
    max_width_missing = np.isclose(max_width, nodata_value)

    only_width_missing = width_missing & ~max_width_missing
    only_max_missing = ~width_missing & max_width_missing
    rivers.loc[only_width_missing, "width"] = rivers.loc[
        only_width_missing, "max_width"
    ]
    rivers.loc[only_max_missing, "max_width"] = rivers.loc[only_max_missing, "width"]
    if only_width_missing.any() or only_max_missing.any():
        log.info(
            f"remove_reaches_with_missing_width: filled {int(only_width_missing.sum())} "
            f"width={nodata_value:g} and {int(only_max_missing.sum())} "
            f"max_width={nodata_value:g} reach(es) from the other (valid) column"
        )

    both_missing = width_missing & max_width_missing
    bad_ids = {
        normalize_reach_id(rid)
        for rid in rivers.loc[both_missing, "reach_id"]
        if normalize_reach_id(rid) is not None
    }
    if not bad_ids:
        return rivers

    downstream_adj = build_downstream_adjacency(rivers)
    upstream_adj: dict[str, list[str]] = {rid: [] for rid in downstream_adj}
    for rid, dns in downstream_adj.items():
        for dn in dns:
            upstream_adj.setdefault(dn, []).append(rid)

    to_remove: set[str] = set()
    for rid in bad_ids:
        to_remove.add(rid)
        # downstream: stop before a confluence (next reach fed by >1 upstream)
        cur = rid
        while len(downstream_adj.get(cur, [])) == 1:
            nxt = downstream_adj[cur][0]
            if len(upstream_adj.get(nxt, [])) != 1 or nxt in to_remove:
                break
            to_remove.add(nxt)
            cur = nxt
        # upstream: stop before a bifurcation (prev reach splits into >1 downstream)
        cur = rid
        while len(upstream_adj.get(cur, [])) == 1:
            prev = upstream_adj[cur][0]
            if len(downstream_adj.get(prev, [])) != 1 or prev in to_remove:
                break
            to_remove.add(prev)
            cur = prev

    rivers = rivers[
        ~rivers["reach_id"].apply(normalize_reach_id).isin(to_remove)
    ].copy()
    log.warning(
        f"remove_reaches_with_missing_width: removed {len(to_remove)} reach(es) with no "
        f"width/max_width data at all ({len(bad_ids)} directly affected, "
        f"{len(to_remove) - len(bad_ids)} more in the same unbranched chain)"
    )
    return rivers


def enforce_mouth_width_monotonic(
    rivers: gpd.GeoDataFrame,
    width_column: str = "width",
) -> gpd.GeoDataFrame:
    """
    Ensure each river mouth's width is at least as large as its upstream
    neighbour's -- channels don't narrow immediately before the outlet;
    where SWORD's own attributes show otherwise (e.g. a mouth obscured by
    tidal flats/vegetation in the source imagery), treat it as an artifact
    and raise the mouth's width to match.

    Mouths are reaches with no downstream neighbour in-network (same
    definition used elsewhere, e.g. 11_test_upstream_boundary.py). A mouth
    with multiple upstream neighbours (a confluence right at the outlet) is
    compared against the WIDEST of them.

    Args:
        rivers:       River network with 'reach_id', 'rch_id_dn', and
                      width_column columns.
        width_column: Canonical width column to enforce (default 'width').
                      'max_width' is raised alongside if it would otherwise
                      fall below the new width.

    Returns:
        Copy of ``rivers`` with mouth widths raised where needed.
    """
    downstream_adj = build_downstream_adjacency(rivers)
    upstream_adj: dict[str, list[str]] = {rid: [] for rid in downstream_adj}
    for rid, dns in downstream_adj.items():
        for dn in dns:
            upstream_adj.setdefault(dn, []).append(rid)

    rivers = rivers.copy()
    rids = rivers["reach_id"].apply(normalize_reach_id)
    width_by_rid = dict(zip(rids, rivers[width_column]))

    mouth_ids = [rid for rid, dns in downstream_adj.items() if not dns]
    n_raised = 0
    for rid in mouth_ids:
        ups = upstream_adj.get(rid, [])
        upstream_width = max(
            (
                width_by_rid[u]
                for u in ups
                if u in width_by_rid and pd.notna(width_by_rid[u])
            ),
            default=None,
        )
        if upstream_width is None:
            continue
        mouth_mask = rids == rid
        mouth_width = rivers.loc[mouth_mask, width_column].iloc[0]
        if pd.notna(mouth_width) and mouth_width < upstream_width:
            rivers.loc[mouth_mask, width_column] = upstream_width
            if "max_width" in rivers.columns:
                mw = rivers.loc[mouth_mask, "max_width"].iloc[0]
                if pd.isna(mw) or mw < upstream_width:
                    rivers.loc[mouth_mask, "max_width"] = upstream_width
            n_raised += 1

    if n_raised:
        log.info(
            f"enforce_mouth_width_monotonic: raised width for {n_raised} mouth "
            f"reach(es) to match their (wider) upstream neighbour"
        )
    return rivers


# SWORD v17c 'max_width' value that marks "no value" rather than a measured
# width: 2,343 reaches worldwide carry exactly 30 m (not on SWORD's ~42 m
# width lattice), 2,164 of them with a real 'width' far above it (Waal:
# 389 m) -- swapping those turned 30 m into the channel width.
MAX_WIDTH_FILL_M = 30.0


def fix_width_max_width_order(
    rivers: gpd.GeoDataFrame, max_width_fill_m: float | None = MAX_WIDTH_FILL_M
) -> gpd.GeoDataFrame:
    """
    Swap 'width' and 'max_width' wherever max_width < width.

    'max_width' (a reach's maximum channel width) should never be smaller
    than 'width' (its average channel width) — where the SWORD attributes
    violate that, the two values are swapped rather than discarded. A
    max_width equal to ``max_width_fill_m`` (SWORD's 30 m fill value, see
    MAX_WIDTH_FILL_M) is not a measurement and never triggers the swap: the
    reach keeps its own 'width'. ``None`` disables that check.

    Adds a 'width_source' column: "sword_width" (kept), "sword_max_width"
    (swapped in), or "sword_width, max_width fill ignored".

    Args:
        rivers: River network with 'width' and 'max_width' columns. Rows
                where either value is missing (NaN) are left unchanged,
                since NaN comparisons are never True.
        max_width_fill_m: max_width value treated as "no value".

    Returns:
        Copy of ``rivers`` with 'width'/'max_width' swapped wherever they
        were inverted; unchanged otherwise.
    """
    rivers = rivers.copy()
    if "width_source" not in rivers.columns:
        rivers["width_source"] = "sword_width"
    is_fill = (
        np.isclose(rivers["max_width"].astype(float), float(max_width_fill_m))
        if max_width_fill_m is not None
        else np.zeros(len(rivers), dtype=bool)
    )
    inverted = rivers["max_width"] < rivers["width"]
    fill_ignored = inverted & is_fill
    invalid = inverted & ~is_fill
    if fill_ignored.any():
        rivers.loc[fill_ignored, "width_source"] = "sword_width, max_width fill ignored"
        log.warning(
            f"max_width == {max_width_fill_m:g} m (SWORD fill value) below width on "
            f"{int(fill_ignored.sum())}/{len(rivers)} reach(es) -- width kept, not swapped"
        )
    n_invalid = int(invalid.sum())
    if n_invalid == 0:
        return rivers

    width_vals = rivers.loc[invalid, "width"].copy()
    rivers.loc[invalid, "width"] = rivers.loc[invalid, "max_width"]
    rivers.loc[invalid, "max_width"] = width_vals
    rivers.loc[invalid, "width_source"] = "sword_max_width"
    log.warning(
        f"width/max_width swapped for {n_invalid}/{len(rivers)} reach(es) (max_width < width)"
    )
    return rivers


def select_width_column(
    rivers: gpd.GeoDataFrame, width_column: str = "width"
) -> gpd.GeoDataFrame:
    """
    Overwrite 'width' with the values of ``width_column``, so every
    downstream consumer that reads 'width' (discharge propagation, hydraulic
    depth, boundary-forcing grid-visibility, SFINCS rivwth) picks up a
    single, consistently-chosen representative channel width, rather than
    each reading 'width' or 'max_width' independently.

    Args:
        rivers:       River network with 'width' and 'max_width' columns.
        width_column: Which column's values become the new 'width' --
                      "width" (no-op) or "max_width".

    Returns:
        Copy of ``rivers`` with 'width' set to ``rivers[width_column]``. The
        original 'max_width' column is left untouched (still available for
        diagnostics, e.g. tests/plot_width_max_width_heatmap.py).
    """
    if width_column not in ("width", "max_width"):
        raise ValueError(
            f"width_column must be 'width' or 'max_width', got {width_column!r}"
        )
    if width_column == "width":
        return rivers
    rivers = rivers.copy()
    rivers["width"] = rivers["max_width"]
    log.info(
        f"select_width_column: 'width' set to 'max_width' values for all {len(rivers)} reach(es)"
    )
    return rivers


def normalize_channel_widths(
    rivers: gpd.GeoDataFrame,
    width_column: str = "width",
    max_width_fill_m: float | None = MAX_WIDTH_FILL_M,
) -> gpd.GeoDataFrame:
    """
    Fix SWORD's width/max_width inconsistencies, then select which one
    becomes the canonical 'width' used by every width-dependent step in the
    pipeline.

    Bundles, in order:
      1. fix_width_max_width_order -- swap where max_width < width.
      2. select_width_column       -- overwrite 'width' with width_column's
                                      (now-corrected) values.

    Must be applied independently wherever a river network is read from its
    raw source -- currently rule clean_river_network and rule
    boundary_forcings both read river_network.gpkg directly. Each call is
    deterministic given the same input, so there's no risk of the two
    diverging.

    Args:
        rivers:       River network with 'width' and 'max_width' columns.
        width_column: Which (corrected) column becomes 'width' -- "width" or
                      "max_width". Passed through to select_width_column.
        max_width_fill_m: SWORD's max_width fill value, never swapped in
                      (fix_width_max_width_order).

    Returns:
        Copy of ``rivers`` with 'width'/'max_width' fixed and 'width' set to
        the configured choice.
    """
    rivers = fix_width_max_width_order(rivers, max_width_fill_m=max_width_fill_m)
    rivers = select_width_column(rivers, width_column=width_column)
    return rivers


def _sample_line_cells(
    line: LineString, transform, shape: tuple[int, int], step_m: float
) -> list[dict]:
    """
    Sample `line` at `step_m` intervals (always including both endpoints),
    snap each sample to its containing raster cell, and de-duplicate
    consecutive repeats.

    Returns a list of dicts {'row', 'col', 'along_m', 'point'} in
    downstream order, one entry per distinct cell touched.
    """
    length = line.length
    n = 1 if length == 0 else max(2, int(np.ceil(length / step_m)) + 1)
    distances = np.linspace(0.0, length, n)
    nrows, ncols = shape

    cells: list[dict] = []
    last_rc = None
    for d in distances:
        pt = line.interpolate(d)
        row, col = rasterio.transform.rowcol(transform, pt.x, pt.y)
        if not (0 <= row < nrows and 0 <= col < ncols):
            continue
        rc = (row, col)
        if rc == last_rc:
            continue
        cells.append({"row": row, "col": col, "along_m": float(d), "point": pt})
        last_rc = rc
    return cells


def trace_widest_path(rivers: gpd.GeoDataFrame, seed: str) -> list[str]:
    """
    From `seed`, walk downstream; at a bifurcation always continue onto the
    candidate with the largest 'width'. Stops at a true outlet (no
    downstream neighbour).

    Deliberately NOT SWORD's own main_path_id/is_mainstem_edge
    tagging (by request) -- that SWORD-provided tagging
    can be unreliable, whereas width is a directly observed quantity
    already used everywhere else a bifurcation choice matters (e.g.
    accumulate_discharge's own width-proportional split).

    Args:
        rivers: River network with 'reach_id', 'rch_id_dn', 'width'.
        seed:   Starting reach_id (normalized string).

    Returns:
        Ordered list of reach_id strings from the seed (inclusive) to
        wherever the walk ends.
    """
    adjacency = build_downstream_adjacency(rivers)
    width_by_rid = {
        normalize_reach_id(rid): w
        for rid, w in zip(rivers["reach_id"], rivers["width"])
        if normalize_reach_id(rid) is not None
    }
    path = [seed]
    seen = {seed}
    current = seed
    while True:
        candidates = adjacency.get(current, [])
        if not candidates:
            break
        if len(candidates) == 1:
            nxt = candidates[0]
        else:
            nxt = max(candidates, key=lambda c: width_by_rid.get(c, -np.inf))
        if nxt in seen:
            break  # defensive cycle guard
        path.append(nxt)
        seen.add(nxt)
        current = nxt
    return path


def compute_seed_path_offsets(
    rivers: gpd.GeoDataFrame,
    path_rids: list[str],
) -> dict[str, float]:
    """
    Cumulative distance (m) from a traced seed path's start (the seed
    reach's own upstream end) to each reach's upstream end, summing actual
    reach lengths along the path.

    Deliberately independent of the network's 'dist_out' attribute, which
    can be locally inconsistent between adjacent reaches in this dataset --
    using it directly as a path-distance axis produces spurious gaps/jumps
    between reaches. Geometric reach length is used instead, which is exact
    by construction.

    Args:
        rivers:    River network with 'reach_id' and geometry (any CRS —
                   reprojected internally to a metric CRS to measure each
                   reach's length).
        path_rids: Ordered reach_id list from trace_widest_path()
                   (seed reach first, walking downstream).

    Returns:
        Dict mapping each reach_id in path_rids -> cumulative start offset
        (m). Reaches missing from ``rivers`` or with degenerate geometry
        contribute zero length to the running total.
    """
    metric_crs = pick_utm_crs(rivers) if rivers.crs.is_geographic else rivers.crs
    rivers_m = rivers.to_crs(metric_crs)
    length_by_rid: dict[str, float] = {}
    for row in rivers_m.itertuples(index=False):
        rid = normalize_reach_id(getattr(row, "reach_id"))
        if rid is None:
            continue
        line = _as_linestring(row.geometry)
        length_by_rid[rid] = line.length if line is not None else 0.0

    offsets: dict[str, float] = {}
    cumulative = 0.0
    for rid in path_rids:
        offsets[rid] = cumulative
        cumulative += length_by_rid.get(rid, 0.0)
    return offsets
