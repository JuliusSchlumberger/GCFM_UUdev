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


# Distance (m) from a reach end within which an own anchor makes a borrowed
# junction value redundant (src.river_burn.burn_river_channel/
# _smoothed_weir_crest_profiles).
_BOUNDARY_BLEND_EPS_M = 1.0


def _junction_value(
    reach_data: dict[str, tuple[object, np.ndarray, np.ndarray]],
    up_rid: str,
    dn_rid: str,
) -> float | None:
    """Mean of the upstream reach's last anchor and the downstream reach's first.

    Used on both sides of the junction so their profiles meet; adopting the
    other side's raw value would only move the step. reach_data maps reach_id
    to (line, along_sorted, value_sorted) of any quantity (river bed in
    src.river_burn.burn_river_channel, calibrated weir crest in src.river_burn's
    smoothed-crest profiles).
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
    """Return a LineString; for a multi-part geometry, its longest part."""

    if hasattr(geom, "geoms"):
        parts = [g for g in geom.geoms if hasattr(g, "coords")]
        if not parts:
            return None
        return max(parts, key=lambda g: g.length)
    return geom


def collect_downstream_main_paths(
    rivers: gpd.GeoDataFrame,
    seed_reach_ids: set[str],
) -> set[str]:
    """Return all reach ids downstream of the seeds (seeds included), all branches."""
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
    """Return {reach_id: downstream reach ids} parsed from 'rch_id_dn'.

    rch_id_dn is a list-repr string per SWORD v17c (e.g. "[id1, id2]"); a
    whitespace-separated string is also accepted. Ids not in `rivers` are
    dropped silently.
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
    """Fix T-junctions where downstream reach B starts inside upstream reach A.

    A SWORD artifact: at bifurcations one branch B starts at 98-99% along A,
    leaving an orphan tail (typically 18-111 m) that fragments the merged-line
    connectivity burn_river_rect relies on. Per T-junction: trim A at B's
    start and prepend the tail to C, the reach starting nearest A's true end
    (within max_search_radius_m, else the tail is dropped). Run on the raw
    network before filtering, as A, B and C may not all be reachable.

    Args:
        snap_tolerance_m: Max distance (m) for B's start to lie on A.
        endpoint_tolerance_m: Min distance (m) from A's ends for B's start to
            count as mid-reach.
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
    """Return seed reaches that have in-network upstream reaches (outflow crossings).

    A real inflow seed has no upstream reach in the network; one that does is
    fed by the delta and carries water out (basin 1248635: the Drecht
    crossing, from the Hollandsche IJssel north out of the domain).
    """
    ids = {normalize_reach_id(r) for r in rivers["reach_id"]} - {None}
    has_upstream = {
        d for u, dns in adjacency.items() if u in ids for d in dns if d in ids
    }
    return {normalize_reach_id(s) for s in seed_ids} & has_upstream


def outflow_crossing_mask(river_ds, rivers: gpd.GeoDataFrame) -> np.ndarray:
    """Return a mask over river_forcing.nc crossings whose seed flows out of the network.

    Uses rule 08's 'seed_flows_out'; such discharge must not be injected. All
    False when that column or 'inside_reach_id' is absent.
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
    """Return bifurcation arms that get no share of the upstream discharge (canals).

    With ``auto``, an arm is excluded when it or any reach on its chain to the
    next junction is a SWORD canal (``lakeflag == 2``) and another arm is not;
    if every arm is a canal the width split is kept. Basin 1248635: the
    Maas-Waal Canal, the Lek -> Amsterdam-Rijnkanaal -> Vecht offtake, the
    Hollandsche IJssel -> Drecht branch.

    Locks (``obstr_type == 2``) are not used: SWORD also marks the weir-lock
    complexes on canalised rivers (Meuse at Grave and Lith, Nederrijn-Lek). An
    unflagged canal (1248635: Amsterdam-Rijnkanaal at Tiel, reach 23261000641)
    needs ``force_exclude``; ``force_exclude``/``force_include`` override the
    automatic choice wherever that reach is an arm.
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
    """Return accumulated discharge (m3 s-1) per reach, width-split at bifurcations.

    One pass in topological order: summed at confluences, split by ``width``
    at bifurcations (equally if widths sum to zero). A seed adds its boundary
    discharge once on top of any in-network inflow. Width must be valid (run
    after remove_reaches_with_missing_width/normalize_channel_widths).
    Reaches on a directed cycle are logged and processed last, with only the
    inflow from outside the cycle. excluded_arms get no share at a
    bifurcation unless every arm is excluded. 0.0 for reaches not downstream
    of any seed.
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

    # Kahn: pass a reach on only once all upstream reaches have delivered.
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
    """Flag reaches leaving the network across the delta outline as outflow points.

    A reach qualifies if it crosses the delta polygon's outline (not interior:
    that would flag nearly every delta reach) and is not a seed, a mouth or a
    bifurcation. It stays in the network, flagged 'is_delta_outflow'; its
    outline intersection points (one row per crossing, CRS of ``rivers``) are
    returned for an SFINCS outflow boundary (mask=3). Run after 'is_seed' is
    set. delta_polygon may be a geometry or GeoDataFrame in any CRS.

    Returns:
        (rivers with 'is_delta_outflow', outflow_points with 'reach_id').
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
            # A tangential intersection can be a LineString: use its centroid.
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
    """Fill or remove SWORD reaches with missing (nodata_value) 'width'/'max_width'.

    Run before fix_width_max_width_order/clip_anomalous_max_width, whose swap
    would otherwise move the sentinel into 'width'. One value missing: both
    columns take the other. Both missing: the reach is removed with its
    unbranched chain down to the next confluence and up to the next
    bifurcation, as a missing width usually marks a wider data gap (e.g.
    obscured imagery).
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
        cur = rid
        while len(downstream_adj.get(cur, [])) == 1:
            nxt = downstream_adj[cur][0]
            if len(upstream_adj.get(nxt, [])) != 1 or nxt in to_remove:
                break
            to_remove.add(nxt)
            cur = nxt
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
    """Raise each mouth's width to its widest upstream neighbour's where narrower.

    A narrower mouth is treated as a SWORD artifact (e.g. obscured by tidal
    flats). Mouths have no in-network downstream neighbour (as in
    11_test_upstream_boundary.py). 'max_width' is raised too if it would
    fall below the new width.
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


# SWORD v17c 'max_width' fill value, not a measurement: 2,343 reaches carry
# exactly 30 m (off SWORD's ~42 m width lattice), 2,164 with a real 'width'
# far above it (Waal: 389 m); swapping would make 30 m the channel width.
MAX_WIDTH_FILL_M = 30.0


def fix_width_max_width_order(
    rivers: gpd.GeoDataFrame, max_width_fill_m: float | None = MAX_WIDTH_FILL_M
) -> gpd.GeoDataFrame:
    """Swap 'width' (mean) and 'max_width' wherever max_width < width.

    A max_width equal to ``max_width_fill_m`` (SWORD's 30 m fill) never
    triggers the swap; ``None`` disables that check. NaN rows are unchanged.
    Adds 'width_source': "sword_width", "sword_max_width" (swapped in) or
    "sword_width, max_width fill ignored".
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
    """Overwrite 'width' with ``width_column`` ("width": no-op, or "max_width").

    Every consumer then reads one channel width from 'width' (discharge split,
    hydraulic depth, grid visibility, SFINCS rivwth). 'max_width' is kept
    (e.g. for tests/plot_width_max_width_heatmap.py).
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
    """Apply fix_width_max_width_order, then select_width_column.

    Apply wherever river_network.gpkg is read raw (rule clean_river_network and
    rule boundary_forcings); it is deterministic, so both agree.
    """
    rivers = fix_width_max_width_order(rivers, max_width_fill_m=max_width_fill_m)
    rivers = select_width_column(rivers, width_column=width_column)
    return rivers


def _sample_line_cells(
    line: LineString, transform, shape: tuple[int, int], step_m: float
) -> list[dict]:
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
    """Return reach ids from `seed` downstream, taking the widest arm at bifurcations.

    Width, not SWORD's main_path_id/is_mainstem_edge, which can be unreliable.
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
            break
        path.append(nxt)
        seen.add(nxt)
        current = nxt
    return path


def compute_seed_path_offsets(
    rivers: gpd.GeoDataFrame,
    path_rids: list[str],
) -> dict[str, float]:
    """Return {reach_id: distance (m) from the path start to the reach's upstream end}.

    Sums geometric reach lengths (metric CRS) rather than 'dist_out', which
    jumps between adjacent reaches. Missing or degenerate reaches add zero.
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
