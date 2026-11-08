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
        return np.nan, np.nan
    t_s, pred = dijkstra(
        graph, directed=False, indices=int(idx[start]), return_predecessors=True
    )
    tgt = idx[targets & wet]
    tgt = tgt[np.isfinite(t_s[tgt])]
    if not tgt.size:
        return np.nan, np.nan
    best = int(tgt[np.argmin(t_s[tgt])])
    steps, node = 0, best
    while node != idx[start] and node >= 0:
        node = pred[node]
        steps += 1
    return float(t_s[best] / 3600.0), steps * dx_m / 1e3


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
    depth = sea_level_m - bed_m
    wet = (mask > 0) & np.isfinite(bed_m) & (depth > MIN_SEA_DEPTH_M)
    if not wet.any():
        raise ValueError("sea_travel_hr: no wet cell at the sea level")
    if not wet[mouth_rc]:
        r, c = np.nonzero(wet)
        k = int(np.argmin((r - mouth_rc[0]) ** 2 + (c - mouth_rc[1]) ** 2))
        mouth_rc = (int(r[k]), int(c[k]))
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
