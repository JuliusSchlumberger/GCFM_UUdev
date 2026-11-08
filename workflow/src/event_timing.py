"""Peak travel times to the main mouth and the event-run length they set (rules event_timing, build_sfincs)."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

G_MS2 = 9.81
MIN_SEA_DEPTH_M = 0.1


def gravity_wave_travel(
    depth_m: np.ndarray,
    wet: np.ndarray,
    start: tuple[int, int],
    targets: np.ndarray,
    dx_m: float,
) -> tuple[float, float]:
    """Return (hours, km) of the fastest wet 4-neighbour path from start to any target cell, at sqrt(g h)."""
    hours, path = _fastest_wet_path(depth_m, wet, start, targets, dx_m)
    if path is None:
        return np.nan, np.nan
    return hours, (len(path) - 1) * dx_m / 1e3


def sea_path(
    bed_m: np.ndarray,
    mask: np.ndarray,
    sea_level_m: float,
    mouth_rc: tuple[int, int],
    dx_m: float,
) -> np.ndarray:
    """Return the (row, col) cells of sea_travel_hr's path, from the mouth to the water-level boundary.

    Empty (0, 2) when no wet path reaches the boundary.
    """
    depth, wet, mouth_rc = _sea_wet_cells(bed_m, mask, sea_level_m, mouth_rc)
    _, path = _fastest_wet_path(depth, wet, mouth_rc, mask == 2, dx_m)
    return np.empty((0, 2), dtype=int) if path is None else np.asarray(path, dtype=int)


def _fastest_wet_path(depth_m, wet, start, targets, dx_m):
    idx = -np.ones(wet.shape, dtype=np.int64)
    idx[wet] = np.arange(int(wet.sum()))
    rows, cols, weights = [], [], []
    for dr, dc in ((0, 1), (1, 0)):
        a = wet[: wet.shape[0] - dr, : wet.shape[1] - dc] & wet[dr:, dc:]
        r, c = np.nonzero(a)
        h = 0.5 * (depth_m[r, c] + depth_m[r + dr, c + dc])
        rows.append(idx[r, c])
        cols.append(idx[r + dr, c + dc])
        weights.append(dx_m / np.sqrt(G_MS2 * h))
    rows, cols, weights = (
        np.concatenate(rows),
        np.concatenate(cols),
        np.concatenate(weights),
    )
    n = int(wet.sum())
    graph = coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
    if idx[start] < 0:
        return np.nan, None
    t_s, pred = dijkstra(
        graph, directed=False, indices=int(idx[start]), return_predecessors=True
    )
    tgt = idx[targets & wet]
    tgt = tgt[np.isfinite(t_s[tgt])]
    if not tgt.size:
        return np.nan, None
    best = int(tgt[np.argmin(t_s[tgt])])
    rc = np.argwhere(wet)  # row-major, the order of idx
    nodes, node = [best], best
    while node != idx[start] and node >= 0:
        node = pred[node]
        nodes.append(node)
    return float(t_s[best] / 3600.0), [tuple(rc[n]) for n in reversed(nodes)]


def _sea_wet_cells(bed_m, mask, sea_level_m, mouth_rc):
    depth = sea_level_m - bed_m
    wet = (mask > 0) & np.isfinite(bed_m) & (depth > MIN_SEA_DEPTH_M)
    if not wet.any():
        raise ValueError("sea_travel_hr: no wet cell at the sea level")
    if not wet[mouth_rc]:
        r, c = np.nonzero(wet)
        k = int(np.argmin((r - mouth_rc[0]) ** 2 + (c - mouth_rc[1]) ** 2))
        mouth_rc = (int(r[k]), int(c[k]))
    return depth, wet, mouth_rc


def sea_travel_hr(
    bed_m: np.ndarray,
    mask: np.ndarray,
    sea_level_m: float,
    mouth_rc: tuple[int, int],
    dx_m: float,
) -> tuple[float, float]:
    """Return (hours, km) of the surge peak from the water-level boundary (mask 2) to the mouth cell.

    Shallow-water wave speed sqrt(g h) over the bed at ``sea_level_m``. The
    mouth snaps to the nearest wet cell when its own cell is dry.
    """
    depth, wet, mouth_rc = _sea_wet_cells(bed_m, mask, sea_level_m, mouth_rc)
    return gravity_wave_travel(depth, wet, mouth_rc, mask == 2, dx_m)


def falling_limb_hr(
    t_hr: np.ndarray,
    q_m3s: np.ndarray,
    peak_hr: float,
    q_bankfull_m3s: float,
    max_hr: float,
) -> float:
    """Return the hours from the peak until the discharge falls below bankfull, at most ``max_hr``."""
    after = (t_hr > peak_hr) & (t_hr <= peak_hr + max_hr)
    below = np.flatnonzero(after & (q_m3s < q_bankfull_m3s))
    return float(t_hr[below[0]] - peak_hr) if below.size else float(max_hr)


def scenario_sea_level_m(surge_ds, surge_rp_yr, slr_m: float) -> float:
    """Return a scenario's sea level at the mouth for the travel times (m), mean over the stations.

    The storm-tide peak level at a numeric surge_rp_yr, else the calm level;
    both with slr_m x fingerprint (src.surge).
    """
    from src.surge import calm_sea_levels, lookup_storm_tide_at_rp

    if surge_rp_yr is not None and not isinstance(surge_rp_yr, str):
        return float(
            np.mean(lookup_storm_tide_at_rp(surge_ds, float(surge_rp_yr), slr_m=slr_m))
        )
    return float(np.mean(calm_sea_levels(surge_ds, slr_m)))
