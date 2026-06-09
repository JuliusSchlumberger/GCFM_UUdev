"""River network graph operations: adjacency, BFS connectivity, flow accumulation, depth."""

from __future__ import annotations

import logging
import math
from collections import deque

import geopandas as gpd
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


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

    def _norm(x) -> str | None:
        if pd.isna(x):
            return None
        try:
            return str(int(float(x)))
        except Exception:
            s = str(x).strip()
            return s if s else None

    all_ids = {_norm(x) for x in rivers["reach_id"]} - {None}
    adjacency: dict[str, list[str]] = {}
    for rid_raw, dn_raw in zip(rivers["reach_id"], rivers["rch_id_dn"]):
        rid = _norm(rid_raw)
        if rid is None:
            continue
        dn_ids: list[str] = []
        if pd.notna(dn_raw):
            s = str(dn_raw).strip().strip("[]")
            if s:
                for token in s.split(","):
                    normed = _norm(token.strip())
                    if normed and normed in all_ids:
                        dn_ids.append(normed)
        adjacency[rid] = dn_ids
    return adjacency


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
