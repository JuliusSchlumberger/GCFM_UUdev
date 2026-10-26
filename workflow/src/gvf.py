"""Run the GVF river-bed estimate for rule 10 (``river_depth_modelling.method: gvf``).

Ties src.gvf_target and src.gvf_solver to the pipeline. The target is rule
09's smoothed FathomDEM water surface (src.water_surface), paired with the
GloFAS mean discharge. values_at_points yields ``zbed_anchors`` for
src.river_burn.burn_river_channel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import substring
from src.gvf_solver import GvfProblem, GvfResult, invert_bed, slot_diagnostics
from src.gvf_target import (
    NODE_FIXED_BED,
    NODE_INVERT,
    NODE_NO_FLOW,
    NODE_WATER_BODY,
    GvfTargets,
    build_gvf_targets,
)
from src.io import require_keys
from src.river_network import kahn_order, normalize_reach_id

log = logging.getLogger(__name__)


def mean_seed_discharge(river_ds, active: np.ndarray) -> dict[str, float]:
    """Return the GloFAS mean discharge per seed reach ({reach_id: m3/s}).

    active: crossings that inject discharge (``has_glofas`` and not a seed that
    flows out, see src.river_network.outflow_crossing_mask).
    """
    from src.river_forcing import seed_discharge_by_reach

    if "inside_reach_id" not in river_ds:
        raise KeyError(
            "river_forcing dataset has no 'inside_reach_id' (written by rule 07)"
        )
    return seed_discharge_by_reach(
        river_ds, river_ds["mean_discharge_m3s"].values[active], active
    )


@dataclass
class GvfRun:
    """Hold everything rule 10 needs from a GVF solve."""

    targets: GvfTargets
    problem: GvfProblem
    result: GvfResult
    mouths: set[str]


def run_gvf(
    rivers_m: gpd.GeoDataFrame,
    reach_q: dict[str, float],
    surface_nodes,
    structure_edges: set[tuple[str, str]],
    sample_sea_fraction,
    sample_dem,
    sea_level_m: float,
    gebco_floor_m: float,
    outflow_reaches: set[str],
    manning_n: float,
    surface_cfg: dict,
    cfg: dict,
) -> GvfRun:
    """Build the GVF target and invert the bed.

    Args:
        rivers_m: Cleaned network in the model CRS (metric).
        surface_nodes, structure_edges: rule 09's water surface
            (src.water_surface.read_water_surface).
        sample_sea_fraction: Callable (x, y, width) -> sea-class fraction.
        sample_dem: Callable (x, y) -> merged (unconditioned) DEM.
        sea_level_m: Mean sea level in the model datum (``calm_sea_level_m``).
        gebco_floor_m: Lowest level of the clamped GEBCO bathymetry.
        outflow_reaches: Reaches flagged ``is_delta_outflow``; never a sea mouth.
        manning_n: Channel Manning n (``river_processing.channel_manning_n``).
        surface_cfg: ``river_processing.water_surface``.
        cfg: The ``river_depth_modelling.gvf`` config block.

    A network end is a sea mouth unless it is a free outflow, and only if its
    last node is a water body or fixed bed (sea) or its target is within
    ``mouth_level_tol_m`` of ``sea_level_m``; other ends keep their target
    level.
    """
    where = "river_processing.river_depth_modelling.gvf"
    require_keys(
        cfg,
        (
            "mouth_level_tol_m",
            "froude_cap",
            "q_min_m3s",
            "h_min_m",
            "h_max_m",
            "nudging",
            "least_squares",
        ),
        where,
    )
    nud = cfg["nudging"]
    ls = cfg["least_squares"]
    require_keys(nud, ("alpha", "passes"), f"{where}.nudging")
    require_keys(
        ls, ("apply", "rmse_tol_m", "rho_m", "prior_weight_m"), f"{where}.least_squares"
    )
    targets = build_gvf_targets(
        rivers_m,
        reach_q,
        surface_nodes,
        structure_edges,
        sample_sea_fraction,
        sample_dem,
        gebco_floor_m,
        surface_cfg,
        cfg,
    )
    net = targets.net
    tol = float(cfg["mouth_level_tol_m"])
    mouths: set[str] = set()
    for r in net.reach_ids:
        if net.downstream[r]:
            continue
        end = net.reach_nodes[r][-1]
        sea_like = (
            targets.node_class[end] != NODE_INVERT
            or targets.eta_target[end] <= sea_level_m + tol
        )
        is_mouth = r not in outflow_reaches and bool(sea_like)
        if is_mouth:
            mouths.add(r)
        log.info(
            f"GVF network end {r}: target {targets.eta_target[end]:+.2f} m, class {targets.node_class[end]}, "
            f"{'free outflow' if r in outflow_reaches else ''} -> {'sea mouth' if is_mouth else 'target level'}"
        )

    problem = GvfProblem(
        targets,
        manning_n=float(manning_n),
        sea_level_m=sea_level_m,
        mouth_reaches=mouths,
        froude_cap=float(cfg["froude_cap"]),
        q_min_m3s=float(cfg["q_min_m3s"]),
    )
    result = invert_bed(
        problem,
        h_min_m=float(cfg["h_min_m"]),
        h_max_m=float(cfg["h_max_m"]),
        nudging_alpha=float(nud["alpha"]),
        nudging_passes=int(nud["passes"]),
        lsq_apply=str(ls["apply"]),
        rmse_tol_m=float(ls["rmse_tol_m"]),
        rho_m=float(ls["rho_m"]),
        prior_weight_m=float(ls["prior_weight_m"]),
    )
    diag = slot_diagnostics(problem, result)
    inv = diag["node_class"] == NODE_INVERT
    log.info(
        f"GVF: {len(diag)} bed slot(s) on {len(net.reach_ids)} reach(es); classes "
        f"{diag['node_class'].value_counts().to_dict()}; inverted depth p10/50/90 "
        f"{np.round(np.nanpercentile(diag.loc[inv, 'h'], [10, 50, 90]), 2).tolist() if inv.any() else '-'} m; "
        f"level RMSE {np.sqrt(np.nanmean(diag.loc[inv, 'residual'] ** 2)) if inv.any() else np.nan:.3f} m; "
        f"bound hits {int(diag['flag_bound'].sum())}; Froude-capped {int(diag['flag_capped'].sum())}; "
        f"junction mismatch max {diag['junction_mismatch_m'].max():.3f} m"
    )
    return GvfRun(targets=targets, problem=problem, result=result, mouths=mouths)


def values_at_points(
    run: GvfRun, per_slot: np.ndarray, reach_ids, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Interpolate a per-slot quantity onto points along their own reach.

    Each point is projected onto its reach's (downstream-running) line and
    the reach's slot values are interpolated linearly by chainage, held
    constant beyond the end nodes. Points on reaches outside the GVF network
    get NaN.
    """
    net = run.targets.net
    out = np.full(len(x), np.nan)
    norm_reach_ids = np.array([normalize_reach_id(r) for r in reach_ids], dtype=object)
    for r in set(norm_reach_ids):
        if r not in net.reach_line:
            continue
        idx = np.flatnonzero(norm_reach_ids == r)
        line = net.reach_line[r]
        ch = np.array([line.project(Point(float(x[i]), float(y[i]))) for i in idx])
        out[idx] = np.interp(ch, net.reach_chainage[r], per_slot[net.reach_slots[r]])
    return out


def carry_bed_to_sea(run: GvfRun) -> np.ndarray:
    """Return the slot beds with every sea slot no higher than the river bed above it.

    Fixed-bed and water-body slots (sea class, src.gvf_target.classify_nodes)
    take their bed from the DEM, i.e. GEBCO's coarse coastal bathymetry, which
    leaves a sill where a river channel enters the sea (2433835: -10.4 m ->
    -2.7 m over 0.5 km). Each such slot takes min(DEM bed, the bed of the
    nearest estimated slot upstream), carried on downstream through the
    network; at a confluence the deepest incoming bed is carried. The GVF
    solve itself is unchanged.
    """
    net = run.targets.net
    z = np.asarray(run.result.z, dtype=float).copy()
    cls = run.problem.slot_class
    sea = np.isin(cls, (NODE_FIXED_BED, NODE_WATER_BODY))
    upstream = net.upstream()
    order, _ = kahn_order(net.reach_ids, lambda r: net.downstream[r])
    carried: dict[str, float] = {}
    for r in order:
        incoming = [
            carried[u] for u in upstream[r] if np.isfinite(carried.get(u, np.nan))
        ]
        bed = min(incoming) if incoming else np.nan
        for slot in net.reach_slots[r]:
            if sea[slot]:
                if np.isfinite(bed) and np.isfinite(z[slot]):
                    z[slot] = min(z[slot], bed)
            elif np.isfinite(z[slot]):
                bed = z[slot]
        carried[r] = bed
    n = int((z < run.result.z - 1e-9).sum())
    if n:
        log.info(f"GVF bed carried into {n} sea slot(s) below their GEBCO bed")
    return z


def slot_levels(run: GvfRun) -> np.ndarray:
    """Return per slot the simulated level of its own reach."""
    return run.problem.slot_eta(run.result.solve)


def depth_segments(run: GvfRun, crs=None) -> gpd.GeoDataFrame:
    """Return reach pieces between consecutive GVF nodes, for the river depth map.

    Each piece is cut from the reach's own line (bends kept) and carries the
    mean water depth of its two end slots (``depth``: simulated level minus
    bed at the mean discharge), the reach width used in the solve
    (``width``) and ``fallback`` where the depth comes from the width-depth
    fit (no-flow reach) instead of the hydraulics.
    """
    net = run.targets.net
    h = slot_levels(run) - run.result.z
    no_flow = run.problem.slot_class == NODE_NO_FLOW
    rows = []
    for r in net.reach_ids:
        line = net.reach_line[r]
        ch = net.reach_chainage[r]
        sl = net.reach_slots[r]
        for k in range(len(ch) - 1):
            piece = substring(line, float(ch[k]), float(ch[k + 1]))
            if not isinstance(piece, LineString) or piece.length == 0:
                continue
            rows.append(
                {
                    "reach_id": r,
                    "depth": float(np.mean(h[sl[k : k + 2]])),
                    "width": run.targets.reach_width[r],
                    "fallback": bool(no_flow[sl[k]] or no_flow[sl[k + 1]]),
                    "geometry": piece,
                }
            )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
