"""Steady gradually-varied-flow (GVF) forward solve on a reach and a river network.

Forward model, rectangular channel of width w, discharge Q, Manning n:

    dh/dx = (S0 - Sf) / (1 - Fr^2)
    Sf    = n^2 Q^2 / (A^2 R^(4/3)),   A = w h,  R = w h / (w + 2 h)
    Fr^2  = Q^2 / (g w^2 h^3)

Subcritical flow is integrated upstream from a downstream level with RK4
(substeps <= ``max_step_m``), bed and width linear between nodes. Where the
Froude number would reach ``froude_cap`` the depth is held at the depth with
that Froude number, and the node is flagged.

Network (:func:`forward_network`): reaches are solved downstream-first. A
reach's downstream level is the mean level its flowing downstream reaches
start at, or the given level at a network end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

G = 9.81


def friction_slope(q, w, h, n):
    """Manning friction slope of a rectangular channel."""
    h = np.maximum(h, 1e-6)
    area = w * h
    radius = area / (w + 2.0 * h)
    return n**2 * q**2 / (area**2 * radius ** (4.0 / 3.0))


def froude(q, w, h):
    """Froude number Q / (w h sqrt(g h))."""
    h = np.maximum(h, 1e-6)
    return q / (w * h * np.sqrt(G * h))


def critical_depth(q: float, w: float, fr: float = 1.0) -> float:
    """Depth at which the Froude number equals ``fr`` (1: critical depth)."""
    if q <= 0:
        return 0.0
    return float((q**2 / (G * w**2 * fr**2)) ** (1.0 / 3.0))


def normal_depth(q: float, w: float, n: float, s: float) -> float:
    """Depth at which the Manning friction slope equals ``s``."""
    if q <= 0:
        return 0.0
    s = max(s, 1e-12)
    f = lambda h: friction_slope(q, w, h, n) - s  # noqa: E731
    lo, hi = 1e-4, 1.0
    while f(hi) > 0:
        hi *= 2.0
        if hi > 1e5:
            return hi
    return float(brentq(f, lo, hi, xtol=1e-6))


@dataclass
class ReachSolve:
    """Forward-solve result of one reach (arrays over its nodes)."""

    eta: np.ndarray
    h: np.ndarray
    capped: np.ndarray


def integrate_reach(
    chainage: np.ndarray,
    z: np.ndarray,
    w: float,
    q: float,
    n: float,
    eta_dn: float,
    eta_fixed: np.ndarray | None = None,
    froude_cap: float = 0.95,
    max_step_m: float = 50.0,
    h_dry_m: float = 0.01,
) -> ReachSolve:
    """Integrate the steady water level upstream along one reach.

    Args:
        chainage: Node chainage (m), increasing downstream.
        z: Bed level at the nodes.
        w: Channel width (m).
        q: Discharge (m3/s).
        n: Manning n.
        eta_dn: Water level at the last (downstream) node.
        eta_fixed: Per node, a prescribed level (water body) or NaN.
        froude_cap: Largest Froude number allowed.
        max_step_m: Largest RK4 substep.
        h_dry_m: Smallest depth kept for Q = 0 (still water).

    Returns:
        Level, depth and the capped flag per node.
    """
    m = len(chainage)
    h_cap = max(critical_depth(q, w, froude_cap), h_dry_m)
    eta = np.empty(m)
    h = np.empty(m)
    capped = np.zeros(m, dtype=bool)
    fixed = np.full(m, np.nan) if eta_fixed is None else eta_fixed
    q, w, n = float(q), float(w), float(n)
    n2q2, q2_gw2, min_denom = n * n * q * q, q * q / (G * w * w), 1.0 - froude_cap**2

    def _start(i, level):
        depth = level - z[i]
        if depth < h_cap:
            capped[i] = q > 0
            depth = h_cap
        h[i] = depth
        eta[i] = z[i] + depth

    _start(m - 1, fixed[m - 1] if np.isfinite(fixed[m - 1]) else eta_dn)
    for i in range(m - 2, -1, -1):
        dx = chainage[i + 1] - chainage[i]
        if np.isfinite(fixed[i]):
            _start(i, fixed[i])
            continue
        s0 = (z[i] - z[i + 1]) / dx if dx > 0 else 0.0
        n_sub = max(1, int(np.ceil(dx / max_step_m)))
        ds = dx / n_sub
        hh = h[i + 1]

        def dhds(depth):
            # Plain-float form of friction_slope / froude: this is the inner loop.
            area = w * depth
            sf = n2q2 / (area * area * (area / (w + 2.0 * depth)) ** (4.0 / 3.0))
            fr2 = q2_gw2 / (depth * depth * depth)
            return -(s0 - sf) / max(1.0 - fr2, min_denom)

        hit = False
        for _ in range(n_sub):
            k1 = dhds(hh)
            k2 = dhds(max(hh + 0.5 * ds * k1, h_cap))
            k3 = dhds(max(hh + 0.5 * ds * k2, h_cap))
            k4 = dhds(max(hh + ds * k3, h_cap))
            hh = hh + ds * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            if hh < h_cap:
                hh, hit = h_cap, q > 0
        h[i], capped[i] = hh, hit
        eta[i] = z[i] + hh
    return ReachSolve(eta=eta, h=h, capped=capped)


def fit_width_depth(width_m: np.ndarray, depth_m: np.ndarray) -> tuple[float, float]:
    """Fit h = c w^b (log-log least squares), b kept within [0, 1].

    Falls back to b = 0, c = median depth with fewer than 3 points.
    """
    ok = np.isfinite(width_m) & np.isfinite(depth_m) & (width_m > 0) & (depth_m > 0)
    if ok.sum() < 3:
        return (float(np.median(depth_m[ok])) if ok.any() else 1.0), 0.0
    b = float(
        np.clip(np.polyfit(np.log(width_m[ok]), np.log(depth_m[ok]), 1)[0], 0.0, 1.0)
    )
    a = float(np.median(np.log(depth_m[ok]) - b * np.log(width_m[ok])))
    return float(np.exp(a)), b


def forward_network(
    net,
    bed_m: np.ndarray,
    width_m: np.ndarray,
    reach_q: dict[str, float],
    n: float,
    end_levels: dict[int, float],
    froude_cap: float = 0.95,
    return_capped: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return the steady water level per node of ``net`` (src.gvf_target.GvfNetwork).

    Bed and width are per node; the width of a segment is the mean of its two
    nodes. ``end_levels`` gives the level at every downstream network end.
    Reaches without discharge take the level at their downstream end.
    return_capped: also return a bool per node, True where the depth was held
    at the Froude-cap depth (the upstream node of a capped segment).
    """
    level = np.full(net.n_nodes, np.nan)
    start: dict[str, float] = {}
    per_reach: dict[str, np.ndarray] = {}
    capped_reach: dict[str, np.ndarray] = {}
    for r in reversed(net.reach_ids):
        ids = np.asarray(net.reach_nodes[r])
        ch = net.reach_chainage[r]
        q = float(reach_q.get(r, 0.0))
        dns = [d for d in net.downstream[r] if d in start]
        flowing = [d for d in dns if reach_q.get(d, 0.0) > 0] or dns
        if flowing:
            eta = float(np.mean([start[d] for d in flowing]))
        elif int(ids[-1]) in end_levels:
            eta = float(end_levels[int(ids[-1])])
        else:
            raise KeyError(
                f"forward_network: no level for the downstream end of reach {r}"
            )
        eta_r = np.empty(len(ids))
        cap_r = np.zeros(len(ids), dtype=bool)
        eta_r[-1] = max(eta, bed_m[ids[-1]])
        for k in range(len(ids) - 2, -1, -1):
            if q <= 0:
                eta_r[k] = eta_r[k + 1]
                continue
            seg = integrate_reach(
                np.array([0.0, ch[k + 1] - ch[k]]),
                bed_m[ids[k : k + 2]],
                0.5 * (width_m[ids[k]] + width_m[ids[k + 1]]),
                q,
                n,
                eta_r[k + 1],
                froude_cap=froude_cap,
            )
            eta_r[k] = seg.eta[0]
            cap_r[k] = bool(seg.capped[0])
            cap_r[k + 1] |= bool(seg.capped[-1])
        start[r] = float(eta_r[0])
        per_reach[r] = eta_r
        capped_reach[r] = cap_r
    # A shared node takes a flowing reach's level over a dry one's.
    capped = np.zeros(net.n_nodes, dtype=bool)
    for r in sorted(per_reach, key=lambda r: reach_q.get(r, 0.0) > 0):
        level[np.asarray(net.reach_nodes[r])] = per_reach[r]
        capped[np.asarray(net.reach_nodes[r])] = capped_reach[r]
    return (level, capped) if return_capped else level
