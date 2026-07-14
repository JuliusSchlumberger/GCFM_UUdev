"""River network graph operations: adjacency, BFS connectivity, flow accumulation, depth."""

from __future__ import annotations

import logging
import re
from collections import deque
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.features import geometry_mask
from rasterio.windows import Window, intersect
from shapely.geometry import LineString, Point

from src.geometry import pick_utm_crs

log = logging.getLogger(__name__)


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


def trace_seed_mainstem_paths(rivers: gpd.GeoDataFrame) -> dict[str, list[str]]:
    """
    For each 'is_seed' reach (an active river boundary-forcing point), trace
    its downstream mainstem run through the network.

    Starting at the seed, repeatedly steps onto its current reach's
    downstream neighbour. Where a reach has more than one in-domain
    downstream neighbour (a bifurcation), only a candidate that shares the
    current reach's 'main_path_id' *and* is flagged 'is_mainstem_edge' is
    eligible to continue the run -- i.e. the same SWORD river arm is followed
    through every bifurcation it crosses (confluences don't create a choice
    here; they only add upstream contributions, which doesn't affect a
    downstream walk). The run stops the moment no candidate satisfies both
    conditions, or there is no downstream neighbour at all (a true outlet).
    Where more than one candidate qualifies, one is picked arbitrarily
    (sorted by reach_id, for determinism).

    Args:
        rivers: River network with 'reach_id', 'rch_id_dn', 'main_path_id',
                'is_mainstem_edge', 'is_seed' columns.

    Returns:
        Dict mapping each seed reach_id (str) to the ordered list of
        reach_id strings from the seed (inclusive) to wherever its mainstem
        run ends.
    """
    required = {"reach_id", "rch_id_dn", "main_path_id", "is_mainstem_edge", "is_seed"}
    missing = required - set(rivers.columns)
    if missing:
        raise ValueError(f"rivers is missing required column(s): {sorted(missing)}")

    adjacency = build_downstream_adjacency(rivers)
    main_path_id: dict[str, object] = {}
    is_mainstem: dict[str, bool] = {}
    seeds: list[str] = []
    for rid_raw, mpid, mainstem_flag, seed_flag in zip(
        rivers["reach_id"],
        rivers["main_path_id"],
        rivers["is_mainstem_edge"],
        rivers["is_seed"],
    ):
        rid = normalize_reach_id(rid_raw)
        if rid is None:
            continue
        main_path_id[rid] = mpid if not pd.isna(mpid) else None
        is_mainstem[rid] = bool(mainstem_flag) if not pd.isna(mainstem_flag) else False
        if not pd.isna(seed_flag) and bool(seed_flag):
            seeds.append(rid)

    paths: dict[str, list[str]] = {}
    for seed in seeds:
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
                eligible = sorted(
                    c
                    for c in candidates
                    if main_path_id.get(c) == main_path_id.get(current)
                    and is_mainstem.get(c)
                )
                if not eligible:
                    break
                nxt = eligible[0]
            if nxt in seen:
                break  # defensive cycle guard
            path.append(nxt)
            seen.add(nxt)
            current = nxt
        paths[seed] = path

    if paths:
        lengths = [len(p) for p in paths.values()]
        log.info(
            f"trace_seed_mainstem_paths: traced {len(paths)} seed path(s), "
            f"{min(lengths)}-{max(lengths)} reach(es) each"
        )
    else:
        log.warning("trace_seed_mainstem_paths: no 'is_seed' reach found")
    return paths


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


def accumulate_discharge(
    rivers: gpd.GeoDataFrame,
    seed_q: dict[str, float],
    adjacency: dict[str, list[str]],
    n_iterations: int,
    min_width_m: float,
) -> np.ndarray:
    """
    Iterative width-weighted downstream flow accumulation.

    Starting from seed reaches whose discharge is known from the boundary
    forcing, propagates discharge one hop downstream per iteration. At
    bifurcations the upstream discharge is split proportional to each
    receiving arm's width. At confluences contributions from all upstream
    reaches accumulate via addition.

    Args:
        rivers:       Cleaned river network GeoDataFrame with 'reach_id',
                      'width', and geometry columns.
        seed_q:       Dict {reach_id_str: bankfull_discharge} from the boundary
                      forcing snapping step.  Multiple crossings that snap to the
                      same reach are pre-summed.
        adjacency:    Downstream adjacency dict from build_downstream_adjacency().
        n_iterations: Number of propagation steps (100 covers chains of ≤100 hops).
        min_width_m:  Minimum channel width used as a floor to prevent division
                      by zero.

    Returns:
        1-D NumPy array of accumulated discharge (m³ s⁻¹) indexed in the same
        order as `rivers`.  Reaches not downstream of any seed retain 0.0.
    """
    rids = rivers["reach_id"].astype(str).values
    if len(rids) == 0:
        log.warning(
            "accumulate_discharge: river network is empty; returning zero-length array"
        )
        return np.zeros(0)
    widths = np.maximum(
        rivers["width"].fillna(min_width_m).to_numpy(dtype=float),
        min_width_m,
    )
    reach_to_idx = {r: i for i, r in enumerate(rids)}

    adj_idx: dict[int, list[int]] = {
        i: [reach_to_idx[d] for d in adjacency.get(rid, []) if d in reach_to_idx]
        for i, rid in enumerate(rids)
    }

    q = np.zeros(len(rids))
    seed_mask = np.zeros(len(rids), dtype=bool)
    for rid, val in seed_q.items():
        if rid in reach_to_idx:
            idx = reach_to_idx[rid]
            q[idx] = val
            seed_mask[idx] = True

    log.info(
        f"Starting flow accumulation: {seed_mask.sum()} seed reaches, "
        f"{n_iterations} iterations"
    )

    for _ in range(n_iterations):
        next_q = np.zeros(len(rids))
        next_q[seed_mask] = q[seed_mask]  # preserve fixed boundary inputs

        for i in np.where(q > 0)[0]:
            dn = adj_idx.get(i, [])
            if not dn:
                continue
            w_dn = widths[dn]
            total_w = w_dn.sum()
            shares = w_dn / total_w if total_w > 0 else np.ones(len(dn)) / len(dn)
            for j, d in enumerate(dn):
                next_q[d] += q[i] * shares[j]

        q = next_q

    if len(q) > 0:
        log.info(
            f"Flow accumulation complete: {(q > 0).sum()} reaches with Q > 0, "
            f"max Q = {q.max():.2f} m³ s⁻¹"
        )
    else:
        log.info("Flow accumulation complete: 0 reaches (empty network)")
    return q


def compute_hydraulic_depth(
    q_acc: np.ndarray,
    widths: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    Compute bankfull hydraulic depth from the Leopold–Maddock hydraulic geometry.

    The combined at-a-station and downstream hydraulic geometry relation gives:

        cross_section_area = alpha · Q^beta
        depth = cross_section_area / width = alpha · Q^beta / width

    where alpha = a · c and beta = b + f, with (a, b) the at-a-station width
    exponents and (c, f) the at-a-station depth exponents (Leopold & Maddock,
    1953).

    Args:
        q_acc:  Accumulated discharge array (m³ s⁻¹), same length as widths.
        widths: Observed channel widths (m); already floored at min_width_m
                by the caller.
        alpha:  Combined coefficient (a · c).
        beta:   Combined exponent (b + f).

    Returns:
        Array of hydraulic depths (m), same shape as q_acc.
    """
    # TODO: update to consider coastal influence. For now, the same formula is applied to all reaches regardless of proximity to the coast, which may lead to overestimation of depth in tidally influenced reaches where the hydraulic geometry may differ from the inland river regime.
    depth = alpha * np.power(q_acc, beta) / widths
    return depth


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
    geometry actually crosses the boundary line is a candidate at all
    (confirmed for basin 4267691: only 4/312 raw reaches cross the outline,
    3 of which are exactly its 3 known seed reaches; checking against the
    filled polygon instead flagged 127/147 reaches -- almost the entire
    network -- which was not the intent).

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
    leaving 'width' corrupted with the sentinel instead -- confirmed to
    already be happening (basin 1248635: 10 reaches with max_width=-9999
    and a valid width in the raw network end up with width=-9999 after that
    swap runs).

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
    definition used elsewhere, e.g. 12_test_upstream_boundary.py). A mouth
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


def fix_width_max_width_order(rivers: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Swap 'width' and 'max_width' wherever max_width < width.

    'max_width' (a reach's maximum channel width) should never be smaller
    than 'width' (its average channel width) — where the SWORD attributes
    violate that, the two values are swapped rather than discarded.

    Args:
        rivers: River network with 'width' and 'max_width' columns. Rows
                where either value is missing (NaN) are left unchanged,
                since NaN comparisons are never True.

    Returns:
        Copy of ``rivers`` with 'width'/'max_width' swapped wherever they
        were inverted; unchanged otherwise.
    """
    invalid = rivers["max_width"] < rivers["width"]
    n_invalid = int(invalid.sum())
    if n_invalid == 0:
        return rivers

    rivers = rivers.copy()
    width_vals = rivers.loc[invalid, "width"].copy()
    rivers.loc[invalid, "width"] = rivers.loc[invalid, "max_width"]
    rivers.loc[invalid, "max_width"] = width_vals
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
    depth, boundary-forcing grid-visibility, quadtree refinement buffer,
    SFINCS rivwth) picks up a single, consistently-chosen representative
    channel width, rather than each reading 'width' or 'max_width' independently.

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

    Returns:
        Copy of ``rivers`` with 'width'/'max_width' fixed and 'width' set to
        the configured choice.
    """
    rivers = fix_width_max_width_order(rivers)
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


def sample_dem_near_river(
    rivers: gpd.GeoDataFrame,
    dem_path: str | Path,
    width_column: str = "max_width",
) -> pd.DataFrame:
    """
    Sample every valid DEM pixel within ``width_column``/2 of each reach's
    centerline, recording its elevation and approximate along-network
    distance from the river mouth.

    Distance from mouth is anchored on the SWORD 'dist_out' attribute (m;
    distance from a reach's upstream point to the network outlet) and
    interpolated linearly within each reach via the pixel's projected
    position along the centerline (``shapely.line_locate_point``, measured
    from the line's start = the reach's upstream end) — exact at reach
    boundaries, approximate within a reach (assumes 'dist_out' deltas track
    reach length; at bifurcations the chosen distributary's 'dist_out' is
    used as-is, so branches downstream of a split are not forced consistent
    with each other).

    Args:
        rivers:       River network with 'reach_id', 'dist_out', and
                      ``width_column`` columns (any CRS — reprojected
                      internally to the DEM's CRS).
        dem_path:     Path to a merged elevation raster (e.g.
                      elevation_merged.tif).
        width_column: Column giving the full channel width (m); reaches with
                      a missing/non-positive value are skipped.

    Returns:
        DataFrame with one row per sampled pixel: 'reach_id',
        'distance_from_mouth_m', 'along_m' (distance from the reach's own
        upstream end, measured along its own centerline -- independent of
        'dist_out', see trace_seed_mainstem_paths/compute_seed_path_offsets
        for why that matters), 'elevation_m', 'lateral_offset_m' (unsigned
        distance from the centerline, m).
    """
    if "dist_out" not in rivers.columns:
        raise ValueError(
            "rivers is missing the SWORD 'dist_out' column needed for distance-from-mouth"
        )

    with rasterio.open(dem_path) as dem_src:
        dem_crs = dem_src.crs
        nodata = dem_src.nodata
        rivers_proj = rivers.to_crs(dem_crs) if rivers.crs != dem_crs else rivers

        records: list[dict] = []
        n_skipped_width = 0
        n_skipped_empty = 0
        for _, row in rivers_proj.iterrows():
            line = _as_linestring(row.geometry)
            half_width = row[width_column]
            if (
                line is None
                or line.length == 0
                or pd.isna(half_width)
                or half_width <= 0
                or pd.isna(row["dist_out"])
            ):
                n_skipped_width += 1
                continue
            half_width = float(half_width) / 2.0

            buffer_poly = line.buffer(half_width)
            window = dem_src.window(*buffer_poly.bounds).round_offsets().round_lengths()
            full_window = Window(0, 0, dem_src.width, dem_src.height)
            if not intersect(window, full_window):
                n_skipped_empty += 1
                continue
            # Clip to the raster's own extent -- a reach buffer that overhangs the edge
            # (e.g. near the domain boundary) would otherwise leave window_transform()
            # anchored to the *unclipped* (possibly negative) offset while .read() quietly
            # returns the smaller, clipped array, misaligning every sampled coordinate.
            window = window.intersection(full_window)
            if window.width <= 0 or window.height <= 0:
                n_skipped_empty += 1
                continue
            arr = dem_src.read(1, window=window)
            transform = dem_src.window_transform(window)

            inside = geometry_mask(
                [buffer_poly], out_shape=arr.shape, transform=transform, invert=True
            )
            valid = inside & np.isfinite(arr)
            if nodata is not None:
                valid &= arr != nodata
            if not valid.any():
                n_skipped_empty += 1
                continue

            rows_idx, cols_idx = np.where(valid)
            xs, ys = rasterio.transform.xy(transform, rows_idx, cols_idx)
            points = shapely.points(np.asarray(xs), np.asarray(ys))

            along = shapely.line_locate_point(line, points)
            lateral = shapely.distance(line, points)
            dist_from_mouth = float(row["dist_out"]) - along

            reach_id = normalize_reach_id(row.get("reach_id"))
            records.extend(
                {
                    "reach_id": reach_id,
                    "distance_from_mouth_m": float(d),
                    "along_m": float(a),
                    "elevation_m": float(e),
                    "lateral_offset_m": float(lat),
                }
                for d, a, e, lat in zip(
                    dist_from_mouth, along, arr[rows_idx, cols_idx], lateral
                )
            )

    log.info(
        f"sample_dem_near_river: {len(records)} pixel sample(s) from "
        f"{len(rivers_proj) - n_skipped_width} reach(es) "
        f"({n_skipped_width} skipped: no/zero {width_column} or dist_out; "
        f"{n_skipped_empty} skipped: no valid DEM pixel in buffer)"
    )
    return pd.DataFrame.from_records(records)


def compute_river_elevation_profile(rivers: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    River elevation profile against distance from the river mouth, built by
    chaining the 'slope' (m/km) attribute upstream through the network from
    the mouth, rather than trusting each reach's own 'wse' independently.

    Each reach's downstream-point elevation is inherited from its
    already-resolved downstream neighbour's upstream-point elevation (so the
    profile is continuous by construction across reach boundaries), and its
    own upstream-point elevation = that inherited base + slope/1000 * reach
    length. Only the network's root reach(es) (no in-domain downstream
    neighbour — the true outlet(s) within this clipped network) seed their
    downstream elevation from their own 'wse' attribute; every other reach's
    'wse' is ignored in favour of the chained value. This avoids the small
    per-reach 'wse' inconsistencies that otherwise produce spurious jumps
    where reaches connect (each reach's 'wse'/'slope' are independent
    SWORD-reported estimates, not guaranteed to agree exactly at shared
    nodes).

    At a bifurcation (a reach with more than one in-domain downstream
    neighbour), the upstream reach is only resolved once *all* of its
    downstream branches have themselves resolved; it then inherits from
    whichever branch shares its own 'main_path_id' (SWORD's own
    upstream-to-downstream river-arm grouping — unambiguous at nearly every
    bifurcation, since exactly one branch normally continues the same arm).
    If more than one branch shares that main_path_id, or none does, the
    branch flagged 'is_mainstem_edge' wins instead. This mirrors SWORD's own
    mainstem-vs-distributary classification rather than picking arbitrarily
    by BFS resolution order.

    The branch(es) *not* chosen were each independently chained from their
    own far-downstream root, so their implied elevation at the bifurcation
    point need not agree with the chosen branch's. Each discarded branch's
    entire already-resolved upstream subtree (its own tributaries included)
    is shifted by the constant offset needed to match the chosen branch at
    the junction — preserving that branch's own slope-derived shape while
    removing the discontinuity, rather than leaving it at the first reach of
    the side-arm. Nested bifurcations are corrected outermost (most
    downstream) first, so each sees any already-corrected upstream anchor.

    Args:
        rivers: River network with 'reach_id', 'rch_id_dn', 'dist_out',
                'wse', 'slope', 'main_path_id', 'is_mainstem_edge' columns
                (any CRS — reprojected internally to a metric CRS to measure
                each reach's length).

    Returns:
        DataFrame with one row per reach endpoint (two per reach: upstream
        and downstream), columns 'reach_id', 'distance_from_mouth_m',
        'elevation_m', sorted by distance_from_mouth_m for direct line
        plotting. Reaches without a path to a root reach, or missing
        'dist_out'/'slope', are excluded.
    """
    required = {
        "reach_id",
        "rch_id_dn",
        "dist_out",
        "wse",
        "slope",
        "main_path_id",
        "is_mainstem_edge",
    }
    missing = required - set(rivers.columns)
    if missing:
        raise ValueError(f"rivers is missing required column(s): {sorted(missing)}")

    metric_crs = pick_utm_crs(rivers) if rivers.crs.is_geographic else rivers.crs
    rivers_m = rivers.to_crs(metric_crs)

    rids = [normalize_reach_id(x) for x in rivers_m["reach_id"]]
    adjacency = build_downstream_adjacency(rivers_m)
    upstream_of: dict[str, list[str]] = {}
    for rid, dn_list in adjacency.items():
        for dn in dn_list:
            upstream_of.setdefault(dn, []).append(rid)

    lengths: dict[str, float] = {}
    rises: dict[
        str, float
    ] = {}  # slope-implied elevation gain along the reach, downstream -> upstream
    wse: dict[str, float] = {}
    dist_out: dict[str, float] = {}
    main_path_id: dict[str, object] = {}
    is_mainstem: dict[str, bool] = {}
    for rid, row in zip(rids, rivers_m.itertuples(index=False)):
        if rid is None:
            continue
        line = _as_linestring(row.geometry)
        if (
            line is None
            or line.length == 0
            or pd.isna(row.slope)
            or pd.isna(row.dist_out)
        ):
            continue
        lengths[rid] = line.length
        rises[rid] = float(row.slope) / 1000.0 * line.length
        dist_out[rid] = float(row.dist_out)
        if not pd.isna(row.wse):
            wse[rid] = float(row.wse)
        if not pd.isna(row.main_path_id):
            main_path_id[rid] = row.main_path_id
        is_mainstem[rid] = (
            bool(row.is_mainstem_edge) if not pd.isna(row.is_mainstem_edge) else False
        )

    # Number of *resolvable* downstream branches per reach (i.e. excluding
    # downstream neighbours that themselves lack slope/dist_out and can
    # therefore never resolve) -- used to detect genuine bifurcations and to
    # avoid ever waiting on a branch that will never arrive.
    out_degree: dict[str, int] = {
        rid: sum(1 for dn in dn_list if dn in lengths)
        for rid, dn_list in adjacency.items()
    }

    def _select_downstream_branch(up_rid: str, resolved: dict[str, float]) -> str:
        """Pick which resolved downstream branch up_rid's elevation chain continues into."""
        candidates = list(resolved)  # dict insertion order = order branches resolved in
        same_path = [
            d for d in candidates if main_path_id.get(d) == main_path_id.get(up_rid)
        ]
        pool = same_path if same_path else candidates
        mainstem = [d for d in pool if is_mainstem.get(d)]
        return (mainstem or pool)[0]

    elev_dn: dict[str, float] = {}
    elev_up: dict[str, float] = {}
    pending: dict[
        str, dict[str, float]
    ] = {}  # up_rid -> {resolved downstream rid: its elev_up}
    queue: deque[str] = deque()
    for rid in lengths:
        if not adjacency.get(rid):  # no in-domain downstream neighbour -> network root
            base = wse.get(rid, 0.0) - rises[rid]
            elev_dn[rid] = base
            elev_up[rid] = base + rises[rid]
            queue.append(rid)

    n_roots = len(queue)
    n_bifurcations = 0
    n_ambiguous = 0
    bifurcation_events: list[tuple[str, str, dict[str, float]]] = []
    while queue:
        rid = queue.popleft()
        for up_rid in upstream_of.get(rid, []):
            if up_rid in elev_up or up_rid not in lengths:
                continue
            if out_degree.get(up_rid, 1) <= 1:
                elev_dn[up_rid] = elev_up[rid]
                elev_up[up_rid] = elev_dn[up_rid] + rises[up_rid]
                queue.append(up_rid)
                continue
            branch_results = pending.setdefault(up_rid, {})
            branch_results[rid] = elev_up[rid]
            if len(branch_results) < out_degree[up_rid]:
                continue  # still waiting on other downstream branch(es)
            n_bifurcations += 1
            own_path = main_path_id.get(up_rid)
            if sum(1 for d in branch_results if main_path_id.get(d) == own_path) != 1:
                n_ambiguous += 1
            chosen = _select_downstream_branch(up_rid, branch_results)
            elev_dn[up_rid] = branch_results[chosen]
            elev_up[up_rid] = elev_dn[up_rid] + rises[up_rid]
            bifurcation_events.append((up_rid, chosen, dict(branch_results)))
            queue.append(up_rid)

    if n_bifurcations:
        log.info(
            f"compute_river_elevation_profile: resolved {n_bifurcations} bifurcation(s) "
            f"via main_path_id ({n_ambiguous} needed the is_mainstem_edge tie-break)"
        )

    # Discarded branches were each chained independently from their own far-
    # downstream root, so their elev_up at the bifurcation point need not agree
    # with the chosen branch's value there. Snap each discarded branch's entire
    # already-resolved upstream subtree (its own tributaries included) by the
    # constant offset needed to match the chosen branch at the junction --
    # this preserves each branch's own (real, slope-derived) internal shape
    # while removing the standing discontinuity, rather than just relocating
    # it to the first reach of the side-arm. Processed most-downstream-first
    # (ascending dist_out) so a nested bifurcation's anchor already reflects
    # any outer correction by the time it's used.
    #
    # In a braided/anastomosing reach (parallel channels rejoining), the same
    # discarded node can be a branch candidate for more than one bifurcation.
    # The offset must therefore be computed against its *current* (possibly
    # already shifted by an earlier, more-downstream correction) elev_up --
    # not the value captured in branch_results back in the main pass -- or
    # an already-applied shift gets compounded on top of itself instead of
    # superseded.
    bifurcation_events.sort(key=lambda ev: dist_out[ev[0]])
    n_shifted = 0
    for up_rid, chosen, branch_results in bifurcation_events:
        anchor = elev_dn[up_rid]
        for branch_rid in branch_results:
            if branch_rid == chosen or branch_rid not in elev_up:
                continue
            offset = anchor - elev_up[branch_rid]
            if offset == 0.0:
                continue
            stack = [branch_rid]
            seen: set[str] = set()
            while stack:
                node = stack.pop()
                if node in seen or node not in elev_dn:
                    continue
                seen.add(node)
                elev_dn[node] += offset
                elev_up[node] += offset
                stack.extend(upstream_of.get(node, []))
            n_shifted += len(seen)

    if n_shifted:
        log.info(
            f"compute_river_elevation_profile: shifted {n_shifted} reach(es) in discarded "
            f"bifurcation branch(es) to match the chosen branch at each junction"
        )

    rows: list[dict] = []
    for rid in lengths:
        if rid not in elev_dn:
            continue
        rows.append(
            {
                "reach_id": rid,
                "distance_from_mouth_m": dist_out[rid] - lengths[rid],
                "elevation_m": elev_dn[rid],
            }
        )
        rows.append(
            {
                "reach_id": rid,
                "distance_from_mouth_m": dist_out[rid],
                "elevation_m": elev_up[rid],
            }
        )

    log.info(
        f"compute_river_elevation_profile: {len(elev_dn)}/{len(lengths)} reach(es) resolved "
        f"via chained slope integration from {n_roots} network root(s)"
    )
    profile = pd.DataFrame(rows)
    if profile.empty:
        return profile
    return profile.sort_values("distance_from_mouth_m").reset_index(drop=True)


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
        path_rids: Ordered reach_id list from trace_seed_mainstem_paths()
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
