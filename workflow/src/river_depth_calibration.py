"""
river_depth_calibration.py -- SFINCS-based river channel depth calibration
(rule modelled_depth_estimation, river_processing.depth_method ==
"modelled"), an alternative to the empirical hydraulic-geometry depth
estimate (rule empirical_depth_estimation).

Design: run a minimal SFINCS model with the river network (and coast)
confined by artificially high (1000 m) walls, forced with a steady
discharge equal to the basin's protection-level return period (bankfull as
a fallback). On the *un-excavated* elevation_conditioned.tif, confining the
design discharge to channel width forces water to pile up above natural
ground level; a fraction of that rise becomes channel excavation
(river_processing.river_depth_modelling.excavation_fraction):

    rise   = period_max_water_level - DEM_at_cell
    rivdph = excavation_fraction * max(rise, 0)

A second confined run on the excavated channel then gives the water level
the production weir crests are set to, and a third run verifies them --
see 10_depth_estimation_modelled.py's own module docstring for the full
three-round design.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from src.river_forcing import interpolate_discharge_at_rp, snap_crossings_to_reaches
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
    """
    Per-crossing steady calibration discharge -- protection-level RP
    (log-RP interpolated from discharge_rp_table) when available, bankfull
    discharge as the fallback -- snapped to seed reach IDs.

    Reuses src.river_forcing.snap_crossings_to_reaches unmodified: builds a
    crossings GeoDataFrame with the same 'bankfull_q'/'inside_reach_id'
    columns that function expects (holding calibration-discharge values,
    not literal bankfull, despite the column name), rather than duplicating
    its keep-largest-per-reach dedup logic.

    Args:
        river_ds: Opened river_forcing.nc (xr.Dataset).
        protection_rp_yr: Basin's assigned riverine protection-level return
            period (years), or None to use bankfull discharge everywhere
            (no valid RP assigned by rule get_protection_levels).

    Returns:
        Dict {reach_id_str: calibration_discharge} -- same shape as
        accumulate_discharge's seed_q argument (src.river_network).
    """
    active = river_ds["has_glofas"].values.astype(bool)
    bankfull_q = river_ds["bankfull_discharge"].values[active]

    if protection_rp_yr is not None and "discharge_rp_table" in river_ds:
        table = river_ds["discharge_rp_table"].values[active]
        table_rps = river_ds["return_period"].values
        calibration_q = interpolate_discharge_at_rp(table, table_rps, protection_rp_yr)
        log.info(
            f"Calibration discharge: protection level RP={protection_rp_yr:.1f} yr "
            f"(log-RP interpolated from discharge_rp_table)"
        )
    else:
        calibration_q = bankfull_q
        log.info("Calibration discharge: bankfull_discharge (no protection level RP)")

    inside_ids = (
        river_ds["inside_reach_id"].values[active]
        if "inside_reach_id" in river_ds
        else np.full(active.sum(), "", dtype=object)
    )
    crossings = gpd.GeoDataFrame(
        {"bankfull_q": calibration_q, "inside_reach_id": inside_ids},
        geometry=gpd.points_from_xy(
            river_ds["longitude"].values[active], river_ds["latitude"].values[active]
        ),
        crs="EPSG:4326",
    )
    return snap_crossings_to_reaches(crossings)


def compute_period_max_zs(zs: np.ndarray) -> np.ndarray:
    """
    Per-point maximum simulated water level over the ENTIRE calibration run.

    A fixed-duration steady-discharge run has no reason to "converge" to a
    flat window in the strict sense check_convergence tests for -- a
    persistent, non-damping oscillation at a discharge-injection (seed)
    cell never satisfies a flatness/trend tolerance no matter how long the
    run is extended. The calibration only needs the worst-case (peak)
    forced water level at each cell to size channel depth/crest against,
    and the maximum over the whole run captures that directly, oscillation
    or not, without any tolerance/extension logic. Callers wanting a
    shorter, trailing-window peak instead of the whole run should slice
    `zs`/`times_s` themselves before calling this.

    Uses nan-aware np.nanmax, not a plain np.max: SFINCS's own map output
    carries NaN at a cell for any timestep it's still dry (e.g. universally
    at t=0, before the steady discharge has propagated into the channel
    yet) -- a plain np.max over an axis containing even one NaN returns NaN
    for that whole reduction.

    Args:
        zs: (n_time, n_points) water level array.

    Returns:
        (n_points,) array of each point's own maximum water level over the
        full time series. NaN only for a point that was NaN at every single
        timestep (never wetted at all during the run).
    """
    return np.nanmax(zs, axis=0)


def gather_calibration_round_profile(
    calib_root: Path,
    rivers_utm: gpd.GeoDataFrame,
    seed: str,
    n_rounds: int,
) -> tuple[list[str], dict[int, pd.DataFrame]]:
    """
    Round-by-round per-cell calibration state along one seed-to-mouth path,
    read directly from each round's own calibration_state.csv
    (10_depth_estimation_modelled.py's own real per-cell ground truth --
    dem, rivdph, weir_crest, zs (period-maximum water level)) rather
    than reconstructed from sfincs_map.nc. rivdph is the excavation that
    round actually ran with; weir_crest is NaN while the model was still
    confined by its 1000 m walls (round 0), otherwise the production crest
    derived from the confined, excavated round's own water level.

    Args:
        calib_root: Basin's own preprocessing_inputs/depth_crest_calibration
            directory (round{i}/calibration_state.csv must exist for i in
            0..n_rounds).
        rivers_utm: River network (any CRS accepted; only 'reach_id',
            'rch_id_dn', 'width' are used, via trace_widest_path/
            compute_seed_path_offsets -- reprojection to a metric CRS for
            the offset calculation happens internally there).
        seed:       Seed reach_id (normalized string) to trace from.
        n_rounds:   Number of rounds after round 0 (rounds 0..n_rounds are
            read).

    Returns:
        (path_rids, profiles_by_round) -- path_rids is the ordered list of
        reach_id strings from trace_widest_path; profiles_by_round maps
        round_idx -> DataFrame of that round's own calibration_state.csv
        rows restricted to path_rids, with an added 'along_path_m' column
        (cumulative along-path distance, via compute_seed_path_offsets +
        each cell's own along_m), sorted by along_path_m.
    """
    path_rids = trace_widest_path(rivers_utm, seed)
    offsets = compute_seed_path_offsets(rivers_utm, path_rids)

    profiles: dict[int, pd.DataFrame] = {}
    for round_idx in range(n_rounds + 1):
        csv_path = calib_root / f"round{round_idx}" / "calibration_state.csv"
        df = pd.read_csv(csv_path)
        df["reach_id"] = df["reach_id"].apply(normalize_reach_id)
        on_path = df[df["reach_id"].isin(path_rids)].copy()
        on_path["along_path_m"] = on_path["reach_id"].map(offsets) + on_path["along_m"]
        profiles[round_idx] = on_path.sort_values("along_path_m").reset_index(drop=True)
    return path_rids, profiles


def compute_excavation_depth(
    period_max_zs: np.ndarray, dem_at_cell: np.ndarray, excavation_fraction: float
) -> np.ndarray:
    """
    Channel excavation depth from a confined, UN-excavated calibration run.
    Confining the design discharge to channel width, with no channel yet
    excavated, forces water to pile up ABOVE natural ground level (the
    "rise"); ``excavation_fraction`` of that rise becomes excavation depth.
    The rest is left to the riverbank weir, whose crest is measured
    afterwards from a second confined run on the excavated channel rather
    than derived from this rise.

    The rise is clipped at 0: with subgrid, a cell's water level can sit
    below its own coarse DEM value (SFINCS's subgrid bed is the lowest
    sub-pixel, not the cell mean), which must not turn into a negative
    excavation (a raised bed).

    Args:
        period_max_zs: Per-cell period-max water level (m), same reference
            datum as the DEM.
        dem_at_cell: Per-cell DEM elevation (m) at the same cells.
        excavation_fraction: Fraction of the rise excavated.

    Returns:
        Per-cell excavation depth (m), >= 0 (NaN where period_max_zs is
        NaN, i.e. never wetted).
    """
    rise = period_max_zs - dem_at_cell
    return excavation_fraction * np.maximum(rise, 0.0)
