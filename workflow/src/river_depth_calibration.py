"""SFINCS-based river channel depth calibration helpers (rule modelled_depth_estimation).

A minimal SFINCS model, river and coast confined by 1000 m walls, is forced
with the steady protection-level RP discharge (bankfull as fallback) on the
un-excavated elevation_conditioned.tif. Water piles up above ground and a
fraction of the rise is excavated
(river_processing.river_depth_modelling.excavation_fraction):

    rise   = period_max_water_level - DEM_at_cell
    river_depth_m = excavation_fraction * max(rise, 0)

A second confined run on the excavated channel sets the weir crests; a third
verifies them (see 10_depth_estimation_modelled.py).
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from src.river_forcing import interpolate_discharge_at_rp, seed_discharge_by_reach
from src.river_network import (
    compute_seed_path_offsets,
    normalize_reach_id,
    trace_widest_path,
)

log = logging.getLogger(__name__)


def build_calibration_seed_discharge(
    river_ds,
    protection_rp_yr: float | None,
) -> dict[str, float]:
    """Return the steady calibration discharge per seed reach ({reach_id: m3/s}).

    Protection-level RP discharge (log-RP interpolated from
    discharge_rp_table), or bankfull discharge when protection_rp_yr is None.
    The 'bankfull_discharge_m3s' column passed to snap_crossings_to_reaches holds this
    calibration discharge, not literal bankfull.
    """
    active = river_ds["has_glofas"].values.astype(bool)
    bankfull_discharge_m3s = river_ds["bankfull_discharge_m3s"].values[active]

    if protection_rp_yr is not None and "discharge_rp_table" in river_ds:
        table = river_ds["discharge_rp_table"].values[active]
        rps_yr = river_ds["rp_yr"].values
        calibration_q = interpolate_discharge_at_rp(table, rps_yr, protection_rp_yr)
        log.info(
            f"Calibration discharge: protection level RP={protection_rp_yr:.1f} yr "
            f"(log-RP interpolated from discharge_rp_table)"
        )
    else:
        calibration_q = bankfull_discharge_m3s
        log.info(
            "Calibration discharge: bankfull_discharge_m3s (no protection level RP)"
        )

    return seed_discharge_by_reach(river_ds, calibration_q, active)


def steady_state_metrics(zs: np.ndarray, window_steps: int = 3) -> dict[str, float]:
    """Return how much the water level still changes at the end of a run.

    zs is (n_time, n_points) at a regular output interval; window_steps counts
    output intervals (3 hourly outputs = the last 3 h). Returns median, p95 of
    |zs[-1] - zs[-1 - window_steps]| and frac_rising_1cm over points wet at
    both times; all NaN when the run has too few outputs.
    """
    if zs.shape[0] <= window_steps:
        return {"median": np.nan, "p95": np.nan, "frac_rising_1cm": np.nan}
    d = zs[-1] - zs[-1 - window_steps]
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"median": np.nan, "p95": np.nan, "frac_rising_1cm": np.nan}
    return {
        "median": float(np.median(np.abs(d))),
        "p95": float(np.percentile(np.abs(d), 95)),
        "frac_rising_1cm": float(np.mean(d > 0.01)),
    }


def is_steady(metrics: dict[str, float], median_tol_m: float, p95_tol_m: float) -> bool:
    """Return True when both the median and the p95 change are within tolerance."""
    return bool(
        np.isfinite(metrics["median"])
        and metrics["median"] < median_tol_m
        and metrics["p95"] < p95_tol_m
    )


def compute_end_of_run_zs(zs: np.ndarray, n_last: int = 2) -> np.ndarray:
    """Return per point the NaN-aware mean of the last ``n_last`` outputs.

    Steady-state alternative to compute_period_max_zs, which also picks up a
    run that is still filling or a start-up transient.
    """
    tail = zs[-max(1, int(n_last)) :]
    with np.errstate(all="ignore"):
        return np.nanmean(tail, axis=0)


def compute_period_max_zs(zs: np.ndarray) -> np.ndarray:
    """Return per point the maximum water level over the whole run (zs: n_time x n_points).

    A persistent oscillation at a seed cell never meets a flatness tolerance
    (check_convergence), but the peak is what depth/crest are sized against.
    nanmax because SFINCS map output is NaN while a cell is dry (e.g. at t=0);
    NaN only for a point never wetted. Slice zs for a trailing-window peak.
    """
    return np.nanmax(zs, axis=0)


def gather_calibration_round_profile(
    calib_root: Path,
    rivers_utm: gpd.GeoDataFrame,
    seed: str,
    n_rounds: int,
) -> tuple[list[str], dict[int, pd.DataFrame]]:
    """Return per-round calibration state along one seed's widest path.

    Reads round{i}/calibration_state.csv under calib_root for i in
    0..n_rounds (dem, river_depth_m, weir_crest, zs). zs is the period-max water
    level, except in round 0: the level the excavation used (end of run, period
    max or GVF). weir_crest is NaN in round 0 (1000 m walls). rivers_utm may be in any CRS.

    Returns:
        (path_rids, {round_idx: rows on the path with 'along_path_m', sorted}).
    """
    path_reach_ids = trace_widest_path(rivers_utm, seed)
    offsets = compute_seed_path_offsets(rivers_utm, path_reach_ids)

    profiles: dict[int, pd.DataFrame] = {}
    for round_idx in range(n_rounds + 1):
        csv_path = calib_root / f"round{round_idx}" / "calibration_state.csv"
        df = pd.read_csv(csv_path)
        df["reach_id"] = df["reach_id"].apply(normalize_reach_id)
        on_path = df[df["reach_id"].isin(path_reach_ids)].copy()
        on_path["along_path_m"] = on_path["reach_id"].map(offsets) + on_path["along_m"]
        profiles[round_idx] = on_path.sort_values("along_path_m").reset_index(drop=True)
    return path_reach_ids, profiles


def compute_excavation_depth(
    period_max_zs: np.ndarray, dem_at_cell: np.ndarray, excavation_fraction: float
) -> np.ndarray:
    """Return excavation depth (m) = excavation_fraction * max(zs - DEM, 0).

    From a confined, un-excavated run; the rest of the rise is left to the
    weir, measured from a later run. period_max_zs and dem_at_cell share a
    datum. The rise is clipped at 0: with subgrid the water level can sit
    below the coarse DEM (subgrid bed is the lowest sub-pixel), which must not
    raise the bed. NaN where never wetted.
    """
    rise = period_max_zs - dem_at_cell
    return excavation_fraction * np.maximum(rise, 0.0)


def cell_depth_segments(
    cell_gdf: gpd.GeoDataFrame, depth: np.ndarray, width_by_reach: dict[str, float]
) -> gpd.GeoDataFrame:
    """Return segments between consecutive centreline cells, for the river depth map.

    cell_gdf comes from src.river_burn.build_centerline_cells_regular; depth is
    per cell in its order. Each segment's 'depth' is the mean of its two cells.
    """
    from shapely.geometry import LineString

    rows = []
    reach_id_arr = cell_gdf["reach_id"].map(normalize_reach_id).to_numpy()
    along = cell_gdf["along_m"].to_numpy(dtype=float)
    xs, ys = cell_gdf["x"].to_numpy(dtype=float), cell_gdf["y"].to_numpy(dtype=float)
    for reach_id in pd.unique(reach_id_arr):
        pos = np.flatnonzero(reach_id_arr == reach_id)
        pos = pos[np.argsort(along[pos])]
        for a, b in zip(pos[:-1], pos[1:]):
            rows.append(
                {
                    "reach_id": reach_id,
                    "depth": float(np.nanmean(depth[[a, b]])),
                    "width": float(width_by_reach.get(reach_id, np.nan)),
                    "geometry": LineString([(xs[a], ys[a]), (xs[b], ys[b])]),
                }
            )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=cell_gdf.crs)
