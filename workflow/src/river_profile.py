"""Mainstem profiles on rule 11's river-bed node table: the path, its steady levels and the walled peak travel time."""

import numpy as np
import pandas as pd

G_MS2 = 9.81

PROFILE_COLUMNS = [
    "node",
    "reach_id",
    "x",
    "y",
    "dist_from_mouth_m",
    "dist_to_sea_m",
    "width_m",
    "slope",
    "bed_m",
    "level_m",
    "fathom_level_m",
    "informative",
    "bathymetry",
    "q_design_m3s",
]


def mainstem_path(
    downstream: dict[str, list[str]], start: str, reach_q: dict[str, float]
) -> list[str]:
    """Return the reaches from ``start`` to its network end, taking the largest discharge at bifurcations."""
    path = [start]
    while True:
        nxt = [d for d in downstream.get(path[-1], []) if d in reach_q]
        if not nxt:
            return path
        d = max(nxt, key=lambda r: reach_q[r])
        if d in path:
            raise ValueError(f"cycle in the network at reach {d} (path {path})")
        path.append(d)


def mainstem_profile(
    nodes: pd.DataFrame,
    downstream: dict[str, list[str]],
    forcing_reach: str,
    reach_q: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Return the node table along a forcing reach's mainstem, upstream first.

    ``nodes`` is the river-bed node table (``reach_id`` as normalised strings).
    ``dist_from_mouth_m`` is the distance along the node polyline from the path's
    last node; nodes are 500 m apart, so it misses meanders shorter than that.
    ``reach_q``: discharge per reach that picks the path at bifurcations; default
    the node table's. Pass the network's own discharges where a short reach may
    own no node (its junction nodes belong to the reaches around it): the table
    then lacks it and the path takes the other arm, even one carrying nothing
    (1248635: the Rhine into a zero-discharge canal arm, travel time infinite).
    """
    nodes = nodes.assign(reach_id=nodes["reach_id"].astype(str))
    if forcing_reach not in set(nodes["reach_id"]):
        raise ValueError(
            f"forcing reach {forcing_reach} has no nodes in the river-bed table"
        )
    if reach_q is None:
        reach_q = nodes.groupby("reach_id")["q_m3s"].mean().to_dict()
    path = mainstem_path(downstream, forcing_reach, reach_q)
    order = {r: k for k, r in enumerate(path)}
    prof = nodes[nodes["reach_id"].isin(order)].copy()
    prof["_k"] = prof["reach_id"].map(order)
    prof = prof.sort_values(["_k", "dist_to_sea_m"], ascending=[True, False]).drop(
        columns="_k"
    )
    step = np.hypot(np.diff(prof["x"].to_numpy()), np.diff(prof["y"].to_numpy()))
    prof["dist_from_mouth_m"] = np.append(np.cumsum(step[::-1])[::-1], 0.0)
    prof = prof.rename(columns={"q_m3s": "q_design_m3s"})
    return prof[PROFILE_COLUMNS].reset_index(drop=True)


def scaled_discharge(profile: pd.DataFrame, q_forcing_m3s: float) -> np.ndarray:
    """Return per node the discharge for ``q_forcing_m3s`` entering at the forcing point.

    Scales the routed design discharge, so bifurcation shares are kept; exact
    when the forcing point is the path's only inflow.
    """
    q = profile["q_design_m3s"].to_numpy(float)
    return q * (q_forcing_m3s / q[0])


def forward_levels(
    profile: pd.DataFrame,
    q_m3s: np.ndarray,
    sea_level_m: float,
    manning_n: float,
    froude_cap: float = 0.95,
) -> np.ndarray:
    """Return the steady water level per profile node for discharge ``q_m3s`` and a sea level at the mouth.

    Rectangular section, node width averaged per segment, Manning friction,
    subcritical (Froude capped), RK4 as in rule 11's GVF solver.
    """
    from src.gvf_solver import integrate_reach

    z = profile["bed_m"].to_numpy(float)
    w = profile["width_m"].to_numpy(float)
    x = profile["dist_from_mouth_m"].to_numpy(float)
    q = np.broadcast_to(np.asarray(q_m3s, float), z.shape)
    m = len(z)
    eta = np.empty(m)
    last = integrate_reach(
        np.array([0.0]),
        z[-1:],
        w[-1],
        q[-1],
        manning_n,
        sea_level_m,
        froude_cap=froude_cap,
    )
    eta[-1] = last.eta[0]
    for i in range(m - 2, -1, -1):
        seg = integrate_reach(
            np.array([0.0, x[i] - x[i + 1]]),
            np.array([z[i], z[i + 1]]),
            0.5 * (w[i] + w[i + 1]),
            q[i],
            manning_n,
            eta[i + 1],
            froude_cap=froude_cap,
        )
        eta[i] = seg.eta[0]
    return eta


def walled_travel_time(
    profile: pd.DataFrame,
    q_m3s: np.ndarray,
    sea_level_m: float,
    manning_n: float,
    forcing_hr: float,
    regime_rT: tuple[float, float] = (0.3, 3.0),
    min_depth_m: float = 0.1,
) -> pd.DataFrame:
    """Return per node the steady walled hydraulics and the peak travel time (h) from the first node.

    Walled: the whole discharge stays in the channel (no overtopping, no
    floodplain storage). ``t_kinematic_hr`` sums dx / (5/3 u), the in-bank
    flood-wave speed of a wide Manning channel (Kleitz-Seddon);
    ``t_gravity_hr`` sums dx / (u + sqrt(g h)), the fast bound. ``t_wave_hr``,
    the estimate, takes per node the speed of its regime, set by the linearised
    friction rate r = g n^2 u / h^(4/3) (1/s) against the time ``forcing_hr``
    over which the forcing changes (the ramp): r T below ``regime_rT[0]``,
    friction acts slower than the forcing changes and the change travels as a
    gravity wave (wide, deep, slow reaches: 1248635 Meuse, u 0.015 m/s at 10 m,
    r T 0.15); from ``regime_rT[0]`` on the kinematic speed, which is also the
    slower one in the transition below ``regime_rT[1]`` (1248635 Waal, r T 12).
    In inertial reaches the kinematic speed alone runs to weeks (1248635 Meuse:
    516 h over 80 km).
    """
    eta = forward_levels(profile, q_m3s, sea_level_m, manning_n)
    z = profile["bed_m"].to_numpy(float)
    w = profile["width_m"].to_numpy(float)
    q = np.broadcast_to(np.asarray(q_m3s, float), z.shape)
    h = np.maximum(eta - z, min_depth_m)
    u = q / (w * h)
    r_t = G_MS2 * manning_n**2 * np.abs(u) / h ** (4.0 / 3.0) * forcing_hr * 3600.0
    dx = -np.diff(profile["dist_from_mouth_m"].to_numpy(float))
    out = pd.DataFrame(
        {
            "dist_from_mouth_m": profile["dist_from_mouth_m"].to_numpy(float),
            "level_m": eta,
            "depth_m": h,
            "velocity_ms": u,
            "friction_rT": r_t,
            "regime": np.select(
                [r_t < regime_rT[0], r_t < regime_rT[1]],
                ["inertial", "transition"],
                "frictional",
            ),
        }
    )
    kinematic = 5.0 / 3.0 * u
    gravity = u + np.sqrt(G_MS2 * h)
    with np.errstate(divide="ignore"):
        for name, c in (
            ("kinematic", kinematic),
            ("gravity", gravity),
            ("wave", np.where(r_t < regime_rT[0], gravity, kinematic)),
        ):
            out[f"t_{name}_hr"] = (
                np.concatenate([[0.0], np.cumsum(dx / (0.5 * (c[:-1] + c[1:])))])
                / 3600.0
            )
    return out
