"""Helpers for rule modelled_depth_estimation's SFINCS crest calibration.

The river bed comes from src.river_bed; a confined SFINCS run on that channel
at the steady protection-level RP discharge sets the weir crests, a second
verifies them (see 11_depth_estimation_modelled.py).
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from src.geometry import pick_utm_crs
from src.gvf_target import orient_downstream
from src.river_forcing import interpolate_discharge_at_rp, seed_discharge_by_reach
from src.river_network import (
    build_downstream_adjacency,
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


def compute_period_max_zs(zs: np.ndarray) -> np.ndarray:
    """Return per point the maximum water level over the whole run (zs: n_time x n_points).

    A persistent oscillation at a seed cell never meets a flatness tolerance,
    but the peak is what the crests are sized against.
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
    0..n_rounds (dem_m, river_depth_m, weir_crest_m, zs_m). zs_m is the period-max
    water level, except in round 0: the river-bed level. weir_crest_m is NaN
    in round 0. rivers_utm may be in any CRS. The state's along_m runs along
    each reach line as stored; a line that runs upstream (none on rule 02's
    network) is flipped, so the path runs downstream throughout.

    Returns:
        (path_rids, {round_idx: rows on the path with 'along_path_m', sorted}).
    """
    path_reach_ids = trace_widest_path(rivers_utm, seed)
    offsets = compute_seed_path_offsets(rivers_utm, path_reach_ids)
    rivers_m = (
        rivers_utm.to_crs(pick_utm_crs(rivers_utm))
        if rivers_utm.crs.is_geographic
        else rivers_utm
    )
    _, flipped = orient_downstream(rivers_m, build_downstream_adjacency(rivers_m))
    # Same lengths as compute_seed_path_offsets.
    lengths = {
        normalize_reach_id(r): float(g.length)
        for r, g in zip(rivers_m["reach_id"], rivers_m.geometry)
    }

    profiles: dict[int, pd.DataFrame] = {}
    for round_idx in range(n_rounds + 1):
        csv_path = calib_root / f"round{round_idx}" / "calibration_state.csv"
        df = pd.read_csv(csv_path)
        df["reach_id"] = df["reach_id"].apply(normalize_reach_id)
        on_path = df[df["reach_id"].isin(path_reach_ids)].copy()
        along = on_path["along_m"].to_numpy(dtype=float)
        flip = on_path["reach_id"].isin(flipped).to_numpy()
        reach_len = on_path["reach_id"].map(lengths).to_numpy(dtype=float)
        along = np.where(flip, reach_len - along, along)
        on_path["along_path_m"] = on_path["reach_id"].map(offsets).to_numpy() + along
        profiles[round_idx] = on_path.sort_values("along_path_m").reset_index(drop=True)
    return path_reach_ids, profiles
