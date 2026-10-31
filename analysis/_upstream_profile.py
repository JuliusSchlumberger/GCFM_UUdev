"""Mainstem profiles from a forcing point to the sea for the upstream boundary and river-bed checks.

The profile helpers (path, node profile, steady levels) live in src.river_profile,
shared with rule 11; this module adds the influence helpers, which read a
sea-level perturbation along such a path.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_WORKFLOW = str(Path(__file__).resolve().parents[1] / "workflow")
if _WORKFLOW not in sys.path:
    sys.path.insert(0, _WORKFLOW)

from src.river_profile import (  # noqa: E402, F401
    PROFILE_COLUMNS,
    forward_levels,
    mainstem_path,
    mainstem_profile,
    scaled_discharge,
)


def surface_mismatch(
    profile: pd.DataFrame,
    discharges_m3s: dict[str, float],
    sea_level_m: float,
    manning_n: float,
    tol_m: float = 0.1,
) -> pd.DataFrame:
    """Return per discharge how far the forward level at ``sea_level_m`` departs from rule 11's ``level_m``.

    ``discharges_m3s`` maps a label (e.g. Q_mean, Q_bf) to the discharge at the
    forcing point. The bed is cut for rule 11's design discharge, so that pair
    should reproduce ``level_m``; the others show how far the river departs
    from it at other flows.
    """
    level = profile["level_m"].to_numpy(float)
    x_km = profile["dist_from_mouth_m"].to_numpy(float) / 1000.0
    rows = []
    for label, q in discharges_m3s.items():
        d = (
            forward_levels(
                profile, scaled_discharge(profile, q), sea_level_m, manning_n
            )
            - level
        )
        k = int(np.argmax(np.abs(d)))
        rows.append(
            {
                "discharge": label,
                "q_forcing_m3s": q,
                "mean_m": d.mean(),
                "p10_m": np.percentile(d, 10),
                "p90_m": np.percentile(d, 90),
                "max_abs_m": abs(d[k]),
                "max_abs_at_km": x_km[k],
                "at_forcing_point_m": d[0],
                "share_within_tol": float(np.mean(np.abs(d) <= tol_m)),
            }
        )
    return pd.DataFrame(rows)


def path_nodes(net, reaches: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return the node ids along consecutive reaches of ``net``, upstream first, and their distance (m) to the path's end.

    Distances follow the reach chainage; across a junction whose end nodes are
    not shared (a structure edge) the straight gap is added.
    """
    ids: list[int] = []
    seg: list[float] = []
    for r in reaches:
        nodes = [int(v) for v in net.reach_nodes[r]]
        ch = np.asarray(net.reach_chainage[r], float)
        for k, nd in enumerate(nodes):
            if ids and nd == ids[-1]:
                continue
            if not ids:
                seg.append(0.0)
            elif k == 0:
                seg.append(
                    float(
                        np.hypot(
                            net.node_x[nd] - net.node_x[ids[-1]],
                            net.node_y[nd] - net.node_y[ids[-1]],
                        )
                    )
                )
            else:
                seg.append(float(ch[k] - ch[k - 1]))
            ids.append(nd)
    along = np.cumsum(seg)
    return np.asarray(ids, dtype=int), along[-1] - along


def influence_length(
    dist_m: np.ndarray, delta_m: np.ndarray, threshold_m: float
) -> float:
    """Return the distance from the mouth where ``delta_m`` first falls below ``threshold_m`` going upstream.

    inf when it never does on the path.
    """
    order = np.argsort(dist_m)
    below = np.flatnonzero(delta_m[order] < threshold_m)
    return float(dist_m[order][below[0]]) if below.size else np.inf


def efolding_length(dist_m: np.ndarray, delta_m: np.ndarray, x_max_m: float) -> float:
    """Return L of ``delta = delta_mouth * exp(-x / L)`` fitted (least squares in log) over x < ``x_max_m``."""
    order = np.argsort(dist_m)
    x, d = dist_m[order], delta_m[order]
    if d[0] <= 0:
        return np.nan
    m = (x > 0) & (x < x_max_m) & (d > 0)
    if m.sum() < 2:
        return np.nan
    log_ratio = np.log(d[m] / d[0])
    denom = float(np.sum(x[m] * log_ratio))
    return -float(np.sum(x[m] ** 2)) / denom if denom < 0 else np.inf


def first_above(dist_m: np.ndarray, level_m: np.ndarray, threshold_m: float) -> float:
    """Return the distance from the mouth of the first node, going upstream, whose level exceeds ``threshold_m``."""
    order = np.argsort(dist_m)
    above = np.flatnonzero(level_m[order] > threshold_m)
    return float(dist_m[order][above[0]]) if above.size else np.inf


def boundary_verdict(
    x_f_m: float, delta_f_m: float, x_inf_m: float, level_tol_m: float, margin_m: float
) -> tuple[bool, str]:
    """Return whether a forcing point passes, and the failing criteria ('level', 'distance') or 'pass'."""
    failed = [
        name
        for name, bad in (
            ("level", delta_f_m >= level_tol_m),
            ("distance", x_f_m < x_inf_m + margin_m),
        )
        if bad
    ]
    return (not failed), ("+".join(failed) or "pass")


def high_gradient_rule(
    level_above_msl_m: float, elevation_m: float, marginal_m: float = 1.0
) -> str:
    """Return 'pass', 'marginal' or 'fail' for DE2's rule: the river surface at least ``elevation_m`` above MSL."""
    if level_above_msl_m >= elevation_m + marginal_m:
        return "pass"
    return "marginal" if level_above_msl_m >= elevation_m - marginal_m else "fail"


def routed_discharge(rivers, river_forcing_path, variable: str) -> dict[str, float]:
    """Return per reach the GloFAS ``variable`` at the seed crossings routed through the network, as rule 11 routes it."""
    import xarray as xr
    from src.river_forcing import seed_discharge_by_reach
    from src.river_network import (
        accumulate_discharge,
        build_downstream_adjacency,
        normalize_reach_id,
        outflow_crossing_mask,
    )

    excluded = (
        {
            normalize_reach_id(r)
            for r, f in zip(rivers["reach_id"], rivers["zero_discharge_share"])
            if bool(f)
        }
        if "zero_discharge_share" in rivers.columns
        else set()
    )
    with xr.open_dataset(river_forcing_path, decode_times=False) as ds:
        active = ds["has_glofas"].values.astype(bool) & ~outflow_crossing_mask(
            ds, rivers
        )
        seed = seed_discharge_by_reach(ds, ds[variable].values[active], active)
    q = accumulate_discharge(
        rivers, seed, build_downstream_adjacency(rivers), excluded_arms=excluded
    )
    return dict(
        zip(rivers["reach_id"].map(normalize_reach_id), np.asarray(q, dtype=float))
    )


def sample_sfincs_at_nodes(
    map_path: str | Path, profile: pd.DataFrame, var: str = "zs"
):
    """Return ``var`` of a SFINCS map file at the active cell nearest each profile node.

    The result has dims (time, node) and a ``cell_distance_m`` coordinate.
    """
    import xarray as xr
    from scipy.spatial import cKDTree

    with xr.open_dataset(map_path, decode_times=False) as ds:
        active = ds["msk"].values > 0
        xs, ys = ds["x"].values[active], ds["y"].values[active]
        n_idx, m_idx = np.nonzero(active)
        dist, k = cKDTree(np.column_stack([xs, ys])).query(
            profile[["x", "y"]].to_numpy(float)
        )
        # pointwise indexing of the lazy (compressed) array takes hours on large grids
        da = ds[var].load()
        tdim = [d for d in da.dims if d not in ("n", "m")][0]
        out = da.isel(
            n=xr.DataArray(n_idx[k], dims="node"), m=xr.DataArray(m_idx[k], dims="node")
        )
    return out.rename({tdim: "time"}).assign_coords(
        node=profile["node"].to_numpy(), cell_distance_m=("node", dist)
    )
