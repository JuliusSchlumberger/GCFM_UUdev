"""River network graph operations: adjacency, BFS connectivity, flow accumulation, depth."""

from __future__ import annotations

import logging
import math
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


def _end_direction_vec(
    line,
    from_start: bool,
    simplify_tol: float = 1e-3,
) -> np.ndarray | None:
    """Return a unit vector at one end of a (simplified) LineString.

    from_start=True  → vector pointing forward from the first coord (downstream arm).
    from_start=False → vector pointing backward from the last coord (upstream reach).

    Both vectors point away from the junction, so a straight-through reach
    produces antiparallel vectors with dot-product = –1  (angle = 180°).
    Returns None if fewer than 2 distinct coords remain after simplification.
    """
    line = _as_linestring(line)
    if line is None:
        return None
    simp = line.simplify(simplify_tol, preserve_topology=False)
    simp = _as_linestring(simp) or line
    coords = list(simp.coords)
    if len(coords) < 2:
        coords = list(line.coords)
    if len(coords) < 2:
        return None
    if from_start:
        ax, ay = coords[0][:2]
        bx, by = coords[1][:2]
        vec = np.array([bx - ax, by - ay], dtype=float)
    else:
        ax, ay = coords[-2][:2]
        bx, by = coords[-1][:2]
        vec = np.array([ax - bx, ay - by], dtype=float)  # back from junction
    norm = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
    return vec / norm if norm > 0 else None


def _angle_factor(angle_deg: float, min_factor: float = 0.1) -> float:
    """Map junction angle to a discharge-weighting factor.

    angle_deg = 180 → straight continuation → 1.0
    angle_deg =  90 → perpendicular turn   → min_factor
    angle_deg <  90 → acute / U-turn       → min_factor
    """
    if angle_deg >= 90.0:
        return min_factor + (1.0 - min_factor) * (angle_deg - 90.0) / 90.0
    return min_factor


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

    'rch_id_dn' is stored as a stringified list "[id1, id2, ...]".
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
                for token in s.split(","):
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
    Iterative width-and-angle-weighted downstream flow accumulation.

    Starting from seed reaches whose discharge is known from the boundary
    forcing, propagates discharge one hop downstream per iteration.  At
    bifurcations the upstream discharge is split in proportion to
    ``width × angle_factor`` for each receiving arm.  The angle_factor is 1.0
    for a straight continuation (180°), scales linearly to 0.1 at 90°, and
    stays at 0.1 for more acute deflections.  At confluences contributions
    from all upstream reaches accumulate via addition.

    Angle factors are computed once from the simplified reach geometries before
    the iteration loop.

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

    # Pre-compute angle correction factors for bifurcations (len(dn) > 1).
    geoms = rivers.geometry
    angle_factors: dict[int, np.ndarray] = {}
    for i, rid in enumerate(rids):
        dn = adj_idx.get(i, [])
        if len(dn) <= 1:
            continue
        up_vec = _end_direction_vec(geoms.iloc[i], from_start=False)
        af: list[float] = []
        for j in dn:
            dn_vec = _end_direction_vec(geoms.iloc[j], from_start=True)
            if up_vec is None or dn_vec is None:
                af.append(1.0)
            else:
                cos_a = float(np.clip(np.dot(up_vec, dn_vec), -1.0, 1.0))
                af.append(_angle_factor(math.degrees(math.acos(cos_a))))
        angle_factors[i] = np.array(af, dtype=float)

    if angle_factors:
        all_af = np.concatenate(list(angle_factors.values()))
        log.info(
            f"Angle correction pre-computed for {len(angle_factors)} bifurcations "
            f"({len(all_af)} arms); "
            f"mean factor = {all_af.mean():.3f}, "
            f"min = {all_af.min():.3f}, max = {all_af.max():.3f}"
        )

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
            if i in angle_factors:
                combined = w_dn * angle_factors[i]
                total = combined.sum()
                shares = combined / total if total > 0 else np.ones(len(dn)) / len(dn)
            else:
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


def clip_anomalous_max_width(
    rivers: gpd.GeoDataFrame, max_ratio: float = 10.0
) -> gpd.GeoDataFrame:
    """
    Clip 'max_width' down to max_ratio * 'width' wherever it exceeds that.

    Some SWORD reaches report a 'max_width' tens to hundreds of times their
    own 'width' (e.g. 504x in one observed case: width=63 m, max_width=
    31,770 m) -- far beyond plausible single-reach channel variability, so
    treated as erroneous rather than real. Left uncapped, this blows up
    anything buffering by max_width/2 (quadtree river refinement) into
    multi-kilometre patches far from the actual channel, rather than a
    buffer that tracks the centerline.

    Args:
        rivers:    River network with 'width' and 'max_width' columns. Rows
                   where 'width' is missing/non-positive are left unchanged
                   (nothing to clip relative to).
        max_ratio: Maximum allowed max_width / width ratio before clipping.

    Returns:
        Copy of ``rivers`` with 'max_width' clipped wherever it exceeded
        max_ratio * width; unchanged otherwise.
    """
    width = rivers["width"]
    max_width = rivers["max_width"]
    anomalous = (width > 0) & (max_width > max_ratio * width)
    n_anomalous = int(anomalous.sum())
    if n_anomalous == 0:
        return rivers

    worst_ratio = float((max_width[anomalous] / width[anomalous]).max())
    rivers = rivers.copy()
    rivers.loc[anomalous, "max_width"] = max_ratio * width[anomalous]
    log.warning(
        f"max_width clipped for {n_anomalous}/{len(rivers)} reach(es) "
        f"(max_width > {max_ratio:.0f}x width; worst observed ratio was {worst_ratio:.0f}x)"
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
    max_ratio: float = 10.0,
) -> gpd.GeoDataFrame:
    """
    Fix SWORD's width/max_width inconsistencies, then select which one
    becomes the canonical 'width' used by every width-dependent step in the
    pipeline.

    Bundles, in order:
      1. fix_width_max_width_order -- swap where max_width < width.
      2. clip_anomalous_max_width  -- clip max_width to max_ratio * width
                                      where it's an outlier.
      3. select_width_column       -- overwrite 'width' with width_column's
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
        max_ratio:    Passed through to clip_anomalous_max_width.

    Returns:
        Copy of ``rivers`` with 'width'/'max_width' fixed and 'width' set to
        the configured choice.
    """
    rivers = fix_width_max_width_order(rivers)
    rivers = clip_anomalous_max_width(rivers, max_ratio=max_ratio)
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
