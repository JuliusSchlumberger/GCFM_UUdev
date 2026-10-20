"""Steady gradually-varied-flow (GVF) solve and bed inversion on a river network.

After Neal et al. (2021), *Estimating river channel bathymetry in large scale
flood inundation models*, WRR 57, e2020WR028301.

Forward model, rectangular channel of width w, discharge Q, Manning n:

    dh/dx = (S0 - Sf) / (1 - Fr^2)
    Sf    = n^2 Q^2 / (A^2 R^(4/3)),   A = w h,  R = w h / (w + 2 h)
    Fr^2  = Q^2 / (g w^2 h^3)

Subcritical flow is integrated upstream from a downstream level with RK4
(substeps <= ``max_step_m``), bed and width linear between nodes. Where the
Froude number would reach ``froude_cap`` the depth is held at the depth with
that Froude number, and the node is flagged.

Network: reaches are solved downstream-first (reverse topological order).
A reach's downstream level is the simulated level at its shared end node
(mean over its downstream reaches at a bifurcation; the spread is logged as
the junction mismatch), the target level above a structure, ``sea_level_m``
at a mouth, and the target level at any other network end (SWORD's level is
trusted there; normal depth only if the target is missing). Water-body nodes
keep the target level.

Inverse: the bed z at every ``invert`` node so that the simulated level
matches the target:

1. Start from the Manning normal depth on the target slope.
2. Nudging over the whole network, ``z += alpha (eta_target - eta_sim)``.
3. Per reach, downstream reaches first, bounded least squares
   (``scipy.optimize.least_squares``, trust-region reflective) on
   ``[eta_sim - eta_target ; rho (S_bed / s_scale - 1) ; w_p (h - h_p) / h_p]``.
   The second block regularises the bed slope S_bed towards the target slope;
   it is zero on a uniform reach, so unlike a plain ``rho grad(z)`` it does
   not pull the bed flat. It only acts between two inverted beds: a step into
   a bed from the DEM (fixed bed, water body, often GEBCO) is real. The third
   is a weak prior on the depth h below the target, towards the Manning
   normal depth h_p on the target slope: where
   the target is (nearly) flat, as in a tidal delta whose SWORD levels sit in
   0.1 m steps near sea level, the level says almost nothing about the depth
   and the fit would otherwise run to ``h_max``.

Reaches with no discharge (``q <= q_min_m3s``: tidal side channels, canals
given no share at a bifurcation) cannot be inverted: without flow the level
does not depend on the bed. Their depth comes from a width-depth relation
h = c w^b fitted to the inverted reaches (class ``no_flow``), and their level
is ignored at a junction that also has a flowing reach.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.sparse import lil_matrix
from src.gvf_target import NODE_INVERT, NODE_NO_FLOW, NODE_WATER_BODY, GvfTargets

log = logging.getLogger(__name__)

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


@dataclass
class NetworkSolve:
    """Forward-solve result on the node network."""

    eta: np.ndarray  # per node (mean over the reaches that share it)
    h: np.ndarray  # per bed slot: node level minus slot bed
    capped: np.ndarray
    junction_mismatch: np.ndarray  # per node, spread of the reaches' levels
    reach: dict[str, ReachSolve]


class GvfProblem:
    """Inputs and boundary rules shared by the forward and inverse solves.

    Beds are per bed slot (``net.reach_slots``: one per reach and node), water
    levels per node.
    """

    def __init__(
        self,
        targets: GvfTargets,
        n: float,
        sea_level_m: float,
        mouth_reaches: set[str],
        froude_cap: float = 0.95,
        max_step_m: float = 50.0,
        q_min_m3s: float = 1.0,
    ):
        """Set up the problem.

        Args:
            targets: Output of src.gvf_target.build_gvf_targets.
            n: Channel Manning n (the same value is written into SFINCS).
            sea_level_m: Mean sea level in the model datum (``baseline_m``).
            mouth_reaches: Network ends that discharge to the sea; other ends
                keep their target level.
            froude_cap: Largest Froude number allowed.
            max_step_m: Largest RK4 substep.
            q_min_m3s: Reaches with at most this discharge are ``no_flow``.
        """
        self.t = targets
        self.net = targets.net
        self.n = float(n)
        self.sea_level_m = float(sea_level_m)
        self.mouth_reaches = set(mouth_reaches)
        self.froude_cap = froude_cap
        self.max_step_m = max_step_m
        self.order_dn_first = list(reversed(self.net.reach_ids))
        self.eta_fixed = np.where(
            targets.node_class == NODE_WATER_BODY, targets.eta_target, np.nan
        )
        self.slot_node = self.net.slot_node()
        self.slot_reach = self.net.slot_reach()
        self.slot_class = targets.node_class[self.slot_node].copy()
        self.flowing = {r: targets.reach_q[r] > q_min_m3s for r in self.net.reach_ids}
        for r, flows in self.flowing.items():
            if not flows:
                self.slot_class[self.net.reach_slots[r]] = NODE_NO_FLOW

    def downstream_level(
        self, r: str, z: np.ndarray, solved: dict[str, ReachSolve]
    ) -> float:
        """Return the water level at the last node of reach ``r``."""
        net, t = self.net, self.t
        end = net.reach_nodes[r][-1]
        linked = [
            d
            for d in net.downstream[r]
            if (r, d) not in net.structure_edges and d in solved
        ]
        if any(self.flowing[d] for d in linked):
            linked = [d for d in linked if self.flowing[d]]
        if linked:
            return float(np.mean([solved[d].eta[0] for d in linked]))
        if net.downstream[r]:  # only structures downstream: level above them = target
            return float(t.eta_target[end])
        if r in self.mouth_reaches:
            return self.sea_level_m
        if np.isfinite(t.eta_target[end]):
            return float(t.eta_target[end])
        h = normal_depth(t.reach_q[r], t.reach_width[r], self.n, float(t.s_scale[end]))
        return float(z[net.reach_slots[r][-1]] + h)

    def solve_reach(self, r: str, z: np.ndarray, eta_dn: float) -> ReachSolve:
        """Forward-solve one reach for slot beds ``z`` and level ``eta_dn``."""
        ids = self.net.reach_nodes[r]
        return integrate_reach(
            self.net.reach_chainage[r],
            z[self.net.reach_slots[r]],
            self.t.reach_width[r],
            self.t.reach_q[r],
            self.n,
            eta_dn,
            eta_fixed=self.eta_fixed[ids],
            froude_cap=self.froude_cap,
            max_step_m=self.max_step_m,
        )

    def forward(self, z: np.ndarray) -> NetworkSolve:
        """Forward-solve the whole network for slot beds ``z``."""
        net = self.net
        solved: dict[str, ReachSolve] = {}
        for r in self.order_dn_first:
            solved[r] = self.solve_reach(r, z, self.downstream_level(r, z, solved))
        acc = np.zeros(net.n_nodes)
        cnt = np.zeros(net.n_nodes)
        lo = np.full(net.n_nodes, np.inf)
        hi = np.full(net.n_nodes, -np.inf)
        capped = np.zeros(net.n_nodes, dtype=bool)
        # Node level and junction mismatch over the flowing reaches only; a
        # no-flow reach's still-water level counts only where nothing flows.
        has_flow = np.zeros(net.n_nodes, dtype=bool)
        for r in solved:
            if self.flowing[r]:
                has_flow[net.reach_nodes[r]] = True
        for r, s in solved.items():
            ids = net.reach_nodes[r]
            use = np.full(len(ids), True) if self.flowing[r] else ~has_flow[ids]
            ids, e = ids[use], s.eta[use]
            np.add.at(acc, ids, e)
            np.add.at(cnt, ids, 1.0)
            np.minimum.at(lo, ids, e)
            np.maximum.at(hi, ids, e)
            capped[ids] |= s.capped[use]
        eta = acc / np.maximum(cnt, 1)
        mismatch = np.where(np.isfinite(hi), hi - lo, 0.0)
        return NetworkSolve(eta, eta[self.slot_node] - z, capped, mismatch, solved)

    def slot_eta(self, solve: NetworkSolve) -> np.ndarray:
        """Return per slot the level its own reach simulated there."""
        out = np.empty(self.net.n_slots)
        for r, s in solve.reach.items():
            out[self.net.reach_slots[r]] = s.eta
        return out


@dataclass
class GvfResult:
    """Inverted bed (per slot) and diagnostics."""

    z: np.ndarray
    solve: NetworkSolve
    bound_hit: np.ndarray  # per slot
    residual: np.ndarray  # per slot, NaN where not inverted
    reach_rmse: dict[str, float]


def initial_bed(problem: GvfProblem, h_min: float, h_max: float) -> np.ndarray:
    """Slot beds from the Manning normal depth on the target slope (DEM on fixed nodes)."""
    t = problem.t
    node, reach = problem.slot_node, problem.slot_reach
    z = np.empty(len(node))
    for k in range(len(node)):
        h = normal_depth(
            t.reach_q[reach[k]],
            t.reach_width[reach[k]],
            problem.n,
            float(t.s_scale[node[k]]),
        )
        z[k] = t.eta_target[node[k]] - np.clip(h, h_min, h_max)
    fixed = (problem.slot_class != NODE_INVERT) & np.isfinite(t.dem[node])
    z[fixed] = t.dem[node][fixed]
    return z


def _reach_rmse(problem: GvfProblem, solve: NetworkSolve) -> dict[str, float]:
    out = {}
    t = problem.t
    for r, s in solve.reach.items():
        ids = problem.net.reach_nodes[r]
        m = problem.slot_class[problem.net.reach_slots[r]] == NODE_INVERT
        out[r] = (
            float(np.sqrt(np.mean((s.eta[m] - t.eta_target[ids][m]) ** 2)))
            if m.any()
            else 0.0
        )
    return out


def prior_depth(problem: GvfProblem, h_min: float, h_max: float) -> np.ndarray:
    """Per slot, the Manning normal depth on the target slope, within bounds."""
    t = problem.t
    node, reach = problem.slot_node, problem.slot_reach
    h = np.array(
        [
            normal_depth(
                t.reach_q[reach[k]],
                t.reach_width[reach[k]],
                problem.n,
                float(t.s_scale[node[k]]),
            )
            for k in range(len(node))
        ]
    )
    return np.clip(h, h_min, h_max)


def fit_width_depth(width: np.ndarray, depth: np.ndarray) -> tuple[float, float]:
    """Fit h = c w^b (log-log least squares), b kept within [0, 1].

    Falls back to b = 0, c = median depth with fewer than 3 points.
    """
    ok = np.isfinite(width) & np.isfinite(depth) & (width > 0) & (depth > 0)
    if ok.sum() < 3:
        return (float(np.median(depth[ok])) if ok.any() else 1.0), 0.0
    b = float(np.clip(np.polyfit(np.log(width[ok]), np.log(depth[ok]), 1)[0], 0.0, 1.0))
    a = float(np.median(np.log(depth[ok]) - b * np.log(width[ok])))
    return float(np.exp(a)), b


def invert_bed(
    problem: GvfProblem,
    h_min: float = 0.5,
    h_max: float = 30.0,
    nudging_alpha: float = 1.0,
    nudging_passes: int = 3,
    lsq_apply: str = "flagged",
    rmse_tol_m: float = 0.10,
    rho_m: float = 0.05,
    prior_weight_m: float = 0.05,
    lsq_max_nfev: int = 200,
) -> GvfResult:
    """Estimate the bed at the ``invert`` slots (nudging, then least squares).

    Args:
        problem: The network problem.
        h_min, h_max: Depth bounds below the target level.
        nudging_alpha: Nudging gain.
        nudging_passes: Number of nudging passes (0: none).
        lsq_apply: ``"all"``, ``"flagged"`` (reaches whose RMSE after
            nudging exceeds ``rmse_tol_m``) or ``"none"``.
        rmse_tol_m: Reach RMSE above which least squares runs ("flagged").
        rho_m: Bed-slope regularisation weight (m); 0 disables it.
        prior_weight_m: Weight (m) of the depth prior; 0 disables it.
        lsq_max_nfev: Function evaluations per reach.

    Returns:
        Bed per slot, final forward solve and diagnostics.
    """
    t = problem.t
    node = problem.slot_node
    inv = problem.slot_class == NODE_INVERT
    lb = t.eta_target[node] - h_max
    ub = t.eta_target[node] - h_min
    z = initial_bed(problem, h_min, h_max)
    h_prior = prior_depth(problem, h_min, h_max)

    for _ in range(int(nudging_passes)):
        eta_s = problem.slot_eta(problem.forward(z))
        z[inv] = np.clip(
            z[inv] + nudging_alpha * (t.eta_target[node][inv] - eta_s[inv]),
            lb[inv],
            ub[inv],
        )

    sol = problem.forward(z)
    rmse = _reach_rmse(problem, sol)
    if lsq_apply != "none":
        solved: dict[str, ReachSolve] = {}
        for r in problem.order_dn_first:
            eta_dn = problem.downstream_level(r, z, solved)
            if lsq_apply == "all" or rmse[r] > rmse_tol_m:
                z = _lsq_reach(
                    problem,
                    r,
                    z,
                    eta_dn,
                    lb,
                    ub,
                    rho_m,
                    prior_weight_m,
                    h_prior,
                    lsq_max_nfev,
                )
            solved[r] = problem.solve_reach(r, z, eta_dn)
        sol = problem.forward(z)

    bound_hit = inv & ((z <= lb + 1e-3) | (z >= ub - 1e-3))

    # No-flow reaches: depth from the width-depth relation of the inverted ones.
    no_flow = problem.slot_class == NODE_NO_FLOW
    if no_flow.any():
        depth = t.eta_target[node] - z
        good = inv & ~bound_hit
        per_reach = [
            (t.reach_width[r], float(np.median(depth[sl][good[sl]])))
            for r, sl in problem.net.reach_slots.items()
            if good[sl].any()
        ]
        c, b = (
            fit_width_depth(*map(np.array, zip(*per_reach)))
            if per_reach
            else (1.0, 0.0)
        )
        width = np.array([t.reach_width[r] for r in problem.slot_reach])
        z[no_flow] = t.eta_target[node][no_flow] - np.clip(
            c * width[no_flow] ** b, h_min, h_max
        )
        log.info(
            f"GVF: {int(no_flow.sum())} no-flow slot(s), depth = {c:.3f} * width^{b:.2f}"
        )
        sol = problem.forward(z)

    residual = np.where(inv, problem.slot_eta(sol) - t.eta_target[node], np.nan)
    return GvfResult(
        z=z,
        solve=sol,
        bound_hit=bound_hit,
        residual=residual,
        reach_rmse=_reach_rmse(problem, sol),
    )


def _lsq_reach(
    problem, r, z, eta_dn, lb, ub, rho_m, prior_weight_m, h_prior, max_nfev
) -> np.ndarray:
    t, net = problem.t, problem.net
    ids = net.reach_nodes[r]
    slots = net.reach_slots[r]
    cls = problem.slot_class[slots]
    free = np.flatnonzero(cls == NODE_INVERT)
    if free.size == 0:
        return z
    # The last node's level is the boundary value; it carries no information.
    obs = free[free < len(ids) - 1]
    target = t.eta_target[ids]
    s_scale = t.s_scale[ids]
    dx = np.diff(net.reach_chainage[r])
    # Bed-slope regularisation only between two inverted beds.
    is_free = cls == NODE_INVERT
    seg = (
        np.flatnonzero(is_free[:-1] & is_free[1:])
        if rho_m > 0
        else np.array([], dtype=int)
    )
    n_reg = len(seg)
    n_pri = free.size if prior_weight_m > 0 else 0
    hp = h_prior[slots[free]]

    def residuals(x):
        zz = z.copy()
        zz[slots[free]] = x
        s = problem.solve_reach(r, zz, eta_dn)
        res = s.eta[obs] - target[obs]
        if n_reg:
            zr = zz[slots]
            s_bed = (zr[seg] - zr[seg + 1]) / np.maximum(dx[seg], 1e-6)
            s_ref = 0.5 * (s_scale[seg] + s_scale[seg + 1])
            res = np.concatenate([res, rho_m * (s_bed / s_ref - 1.0)])
        if n_pri:
            res = np.concatenate([res, prior_weight_m * ((target[free] - x) - hp) / hp])
        return res

    # Level at node j depends on the beds at j and downstream of it.
    sparsity = lil_matrix((len(obs) + n_reg + n_pri, free.size), dtype=int)
    for a, j in enumerate(obs):
        sparsity[a, free >= j] = 1
    for a, k in enumerate(seg):
        sparsity[len(obs) + a, (free == k) | (free == k + 1)] = 1
    for k in range(n_pri):
        sparsity[len(obs) + n_reg + k, k] = 1

    lo, hi = lb[slots[free]], ub[slots[free]]
    x0 = np.clip(z[slots[free]], lo + 1e-6, hi - 1e-6)
    fit = least_squares(
        residuals,
        x0,
        bounds=(lo, hi),
        method="trf",
        jac_sparsity=sparsity,
        max_nfev=max_nfev,
    )
    z = z.copy()
    z[slots[free]] = fit.x
    return z


def wave_class(
    s0: np.ndarray, h: np.ndarray, fr: np.ndarray, length_m: float = 1000.0
) -> tuple[np.ndarray, np.ndarray]:
    """Kinematic wave number k = S0 L / (h Fr^2) and Vieira (1983) class.

    kinematic: k > 10 and Fr > 0.5; diffusive: k > 3 and Fr < 0.5;
    otherwise shallow-water.
    """
    k = s0 * length_m / np.maximum(h * fr**2, 1e-12)
    cls = np.full(len(k), "shallow_water", dtype=object)
    cls[(k > 3) & (fr < 0.5)] = "diffusive"
    cls[(k > 10) & (fr > 0.5)] = "kinematic"
    return k, cls


def slot_diagnostics(problem: GvfProblem, result: GvfResult) -> pd.DataFrame:
    """Return one diagnostics row per reach and node, in downstream order."""
    t, net = problem.t, problem.net
    node, reach = problem.slot_node, problem.slot_reach
    chain = np.empty(net.n_slots)
    for r in net.reach_ids:
        chain[net.reach_slots[r]] = net.reach_chainage[r]
    q = np.array([t.reach_q[r] for r in reach])
    w = np.array([t.reach_width[r] for r in reach])
    eta_slot = problem.slot_eta(result.solve)
    h = eta_slot - result.z
    fr = froude(q, w, h)
    k, wclass = wave_class(t.s_scale[node], h, fr)
    return pd.DataFrame(
        {
            "reach_id": reach,
            "node_id": node,
            "chainage_m": chain,
            "x": net.node_x[node],
            "y": net.node_y[node],
            "width": w,
            "q": q,
            "n": problem.n,
            "node_class": problem.slot_class,
            "eta_target": t.eta_target[node],
            "eta_raw": t.eta_raw[node],
            "s_scale": t.s_scale[node],
            "eta_sim": eta_slot,
            "residual": result.residual,
            "z_bed": result.z,
            "h": h,
            "froude": fr,
            "k_wave": k,
            "wave_class": wclass,
            "flag_capped": result.solve.capped[node],
            "flag_bound": result.bound_hit,
            "flag_slope_floor": t.s_scale[node] <= problem_slope_floor(t) + 1e-15,
            "junction_mismatch_m": result.solve.junction_mismatch[node],
        }
    )


def problem_slope_floor(t: GvfTargets) -> float:
    """Smallest slope scale in the targets (the configured floor where hit)."""
    return float(np.min(t.s_scale)) if len(t.s_scale) else 0.0
