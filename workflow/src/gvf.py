"""Pipeline driver for the GVF river-bed estimate (``river_depth_modelling.method: gvf``).

Ties src.gvf_target (WP1) and src.gvf_solver (WP2) to the pipeline's inputs
and outputs for rule 10 (WP3):

- :func:`mean_seed_discharge`: GloFAS ``mean_discharge`` per crossing, snapped
  to seed reaches the same way the calibration discharge is. SWORD ``wse`` is
  paired with the mean discharge (the user's choice, 2026-10-12): SWORD v17c
  levels are long-term means of SWOT/altimetry observations.
- :func:`geoid_offset_at`: N_EGM2008 - N_GOCO06s at points, to put SWORD's
  EGM2008 ``wse`` in the model datum, as rule 05a does for FathomDEM.
- :func:`run_gvf`: target, network ends, inversion.
- :func:`values_at_points`: interpolate per-slot beds (or levels) onto the
  centreline cells by along-reach distance -> ``zbed_anchors`` for
  src.river_burn.burn_river_channel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import LineString, Point
from shapely.ops import substring

from src.gvf_solver import GvfProblem, GvfResult, invert_bed, slot_diagnostics
from src.gvf_target import NODE_INVERT, NODE_NO_FLOW, GvfTargets, build_gvf_targets
from src.river_network import normalize_reach_id

log = logging.getLogger(__name__)


def mean_seed_discharge(river_ds, active: np.ndarray) -> dict[str, float]:
    """Return the GloFAS mean discharge per seed reach ({reach_id: m3/s}).

    Args:
        river_ds: Opened river_forcing.nc.
        active: Crossings that inject discharge (``has_glofas`` and not a seed
            that flows out, see src.river_network.outflow_crossing_mask).
    """
    from src.river_forcing import snap_crossings_to_reaches

    crossings = gpd.GeoDataFrame(
        {
            "bankfull_q": river_ds["mean_discharge"].values[active],
            "inside_reach_id": river_ds["inside_reach_id"].values[active],
        },
        geometry=gpd.points_from_xy(
            river_ds["longitude"].values[active], river_ds["latitude"].values[active]
        ),
        crs="EPSG:4326",
    )
    return snap_crossings_to_reaches(crossings)


def geoid_offset_at(
    goco_path, egm_path, lon: np.ndarray, lat: np.ndarray
) -> np.ndarray:
    """N_EGM2008 - N_GOCO06s (m) at WGS84 points (nearest cell of the global field)."""
    from src.raster import compute_geoid_offset_arr

    off, transform, _ = compute_geoid_offset_arr(goco_path, egm_path)
    rows, cols = rasterio.transform.rowcol(transform, np.asarray(lon), np.asarray(lat))
    rows = np.clip(np.asarray(rows), 0, off.shape[0] - 1)
    cols = np.clip(np.asarray(cols), 0, off.shape[1] - 1)
    return off[rows, cols].astype(float)


@dataclass
class GvfRun:
    """Everything rule 10 needs from a GVF solve."""

    targets: GvfTargets
    problem: GvfProblem
    result: GvfResult
    mouths: set[str]


def run_gvf(
    rivers_m: gpd.GeoDataFrame,
    reach_q: dict[str, float],
    datum_offset_m: np.ndarray,
    sample_sea_fraction,
    sample_dem,
    sea_level_m: float,
    gebco_floor_m: float,
    outflow_reaches: set[str],
    cfg: dict,
) -> GvfRun:
    """Build the GVF target and invert the bed.

    Args:
        rivers_m: Cleaned network in the model CRS (metric).
        reach_q: {reach_id: discharge paired with the target}.
        datum_offset_m: Per reach (``rivers_m`` order), N_EGM2008 - N_GOCO06s.
        sample_sea_fraction: Callable (x, y, width) -> sea-class fraction.
        sample_dem: Callable (x, y) -> merged (unconditioned) DEM.
        sea_level_m: Mean sea level in the model datum (``baseline_m``).
        gebco_floor_m: Lowest level of the clamped GEBCO bathymetry.
        outflow_reaches: Reaches flagged ``is_delta_outflow`` (free outflow
            through the delta outline: never a sea mouth).
        cfg: The ``river_depth_modelling.gvf`` config block.

    A network end is a sea mouth unless it is a free outflow, and only if its
    last node is a water body or fixed bed (sea) or its target is within
    ``mouth_level_tol_m`` of ``sea_level_m``; other ends keep their target
    level.
    """
    targets = build_gvf_targets(
        rivers_m,
        reach_q,
        datum_offset_m,
        sample_sea_fraction,
        sample_dem,
        gebco_floor_m,
        cfg,
    )
    net = targets.net
    tol = float(cfg.get("mouth_level_tol_m", 0.5))
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
        n=float(cfg["manning_channel"]),
        sea_level_m=sea_level_m,
        mouth_reaches=mouths,
        froude_cap=float(cfg.get("froude_cap", 0.95)),
        q_min_m3s=float(cfg.get("q_min_m3s", 1.0)),
    )
    ls = cfg.get("least_squares") or {}
    nud = cfg.get("nudging") or {}
    result = invert_bed(
        problem,
        h_min=float(cfg.get("h_min_m", 0.5)),
        h_max=float(cfg.get("h_max_m", 30.0)),
        nudging_alpha=float(nud.get("alpha", 1.0)),
        nudging_passes=int(nud.get("passes", 3)),
        lsq_apply=str(ls.get("apply", "all")),
        rmse_tol_m=float(ls.get("rmse_tol_m", 0.10)),
        rho_m=float(ls.get("rho_m", 0.05)),
        prior_weight_m=float(ls.get("prior_weight_m", 0.05)),
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
    rids = np.array([normalize_reach_id(r) for r in reach_ids], dtype=object)
    for r in set(rids):
        if r not in net.reach_line:
            continue
        idx = np.flatnonzero(rids == r)
        line = net.reach_line[r]
        ch = np.array([line.project(Point(float(x[i]), float(y[i]))) for i in idx])
        out[idx] = np.interp(ch, net.reach_chainage[r], per_slot[net.reach_slots[r]])
    return out


def slot_levels(run: GvfRun) -> np.ndarray:
    """Per slot, the simulated level of its own reach."""
    return run.problem.slot_eta(run.result.solve)


def depth_segments(run: GvfRun, crs=None) -> gpd.GeoDataFrame:
    """Reach pieces between consecutive GVF nodes, for the river depth map.

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
