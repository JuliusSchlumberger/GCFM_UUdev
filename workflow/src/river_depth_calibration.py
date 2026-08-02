"""
river_depth_calibration.py -- SFINCS-based river channel depth calibration
(rule modelled_depth_estimation, river_processing.depth_method ==
"modelled"), an alternative to the empirical hydraulic-geometry depth
estimate (rule empirical_depth_estimation).

Design: run a minimal SFINCS model with the river network confined by
artificially high (1000 m) walls,
forced with a steady discharge equal to the basin's protection-level return
period (bankfull as a fallback), place one observation point per reach at
its own midpoint, and once each point's water level stabilizes, split the
simulated rise above the DEM between real channel excavation and an actual
production-model weir crest (river_processing.river_depth_modelling.
weir_crest_fraction, default 0.2 -- a real river system has both channel
capacity and a levee on the bank, not a single artificially deep channel
with no freeboard structure):

    rise = stabilized_water_level - DEM_at_reach_midpoint
    rivdph_calibrated     = (1 - weir_crest_fraction) * rise
    weir_crest_calibrated = DEM_at_reach_midpoint + weir_crest_fraction * rise

The calibration channel bed is the *un-excavated* elevation_conditioned.tif
(rule 10's output) -- no burning has happened yet, so confining the design
discharge to channel width only forces water to pile up above natural
ground level; the gap between the two (the "rise") is exactly the total
confinement head this discharge needs, split between the two structures
above.

This "round 0" measures every reach in total isolation (each independently
confined by the 1000 m walls, with nothing downstream to push back against
it), so it can't see the backwater effect that couples real, finite-crest
reaches together: a shortfall at one reach raises the head its upstream
neighbour has to overcome, compounding upstream.
10_depth_estimation_modelled.py corrects
for this with river_processing.river_depth_modelling.n_correction_iterations
additional rounds after round 0: each builds a model with the CURRENT best
excavation + REAL per-reach crest (via burn_river_channel/
build_smoothed_weir_crest_regular, the same functions rules 11b/13 use)
instead of an isolated confinement wall, and wherever the resulting water
level still exceeds that round's own crest, the excess is treated exactly
like a new "rise" and split via weir_crest_fraction again. See that
script's own module docstring for the full round-by-round design.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

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


def place_reach_midpoint_observations(
    rivers: gpd.GeoDataFrame, domain_poly=None
) -> gpd.GeoDataFrame:
    """
    One observation point per reach, at 50% of its own length -- the
    per-reach-midpoint analogue of rule 13's per-crossing, evenly-spaced
    downstream observation points. Matches the one-depth-value-per-reach
    granularity already used everywhere else in this pipeline.

    When ``domain_poly`` is given and a reach's midpoint falls outside it
    (the reach only partially overlaps the modelled domain), the
    observation point is placed at whichever ENDPOINT of the reach (start
    or end) falls inside the domain instead -- a real point on the reach's
    own geometry that's actually within the modelled area, rather than an
    out-of-domain midpoint that would otherwise need a large nudge from
    snap_points_into_region and could land at essentially the same spot as
    that reach's discharge source crossing, which is not a location whose
    simulated water level is trustworthy as a depth-calibration measurement
    for that reach. Falls back to the (still out-of-domain) midpoint,
    unchanged, if neither endpoint is inside the domain either --
    snap_points_into_region's own nudge remains the last resort for that
    remaining edge case.

    Args:
        rivers: River network GeoDataFrame (any CRS; caller reprojects to
            the model CRS, matching the rivers/rivers_utm split already
            used in 13_build_sfincs.py).
        domain_poly: Optional domain/region polygon, same CRS as
            ``rivers`` (e.g. sf.region.geometry.union_all()) -- used only
            to prefer an in-domain endpoint over an out-of-domain midpoint.

    Returns:
        GeoDataFrame of Point features (same CRS as ``rivers``), columns
        obs_id (1-indexed) and reach_id.
    """
    records = []
    for i, row in enumerate(rivers.itertuples(index=False), start=1):
        line = row.geometry
        if line is None or line.length == 0:
            continue
        point = line.interpolate(0.5, normalized=True)
        if domain_poly is not None and not domain_poly.contains(point):
            start, end = Point(line.coords[0]), Point(line.coords[-1])
            if domain_poly.contains(start):
                point = start
            elif domain_poly.contains(end):
                point = end
            # else: neither endpoint is inside either -- keep the midpoint;
            # snap_points_into_region's nudge is the fallback for this case.
        records.append(
            {
                "obs_id": i,
                "reach_id": row.reach_id,
                "geometry": point,
            }
        )
    return gpd.GeoDataFrame(records, crs=rivers.crs)


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
    than reconstructed by replaying modelled_depth_estimation's update formulas against
    sfincs_his.nc/sfincs_map.nc. weir_crest is the crest that ACTUALLY
    produced that round's own zs (round 0's own freshly-split estimate, or
    the PRE-update crest for every correction round) -- not the same
    round's post-hoc, freeboard-updated target for the NEXT round's own
    simulation, which this round's own zs was never checked against.

    Args:
        calib_root: Basin's own sfincs_calibration directory (round{i}/
            calibration_state.csv must exist for i in 0..n_rounds).
        rivers_utm: River network (any CRS accepted; only 'reach_id',
            'rch_id_dn', 'width' are used, via trace_widest_path/
            compute_seed_path_offsets -- reprojection to a metric CRS for
            the offset calculation happens internally there).
        seed:       Seed reach_id (normalized string) to trace from.
        n_rounds:   Number of correction rounds after round 0 (rounds
            0..n_rounds are read).

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


def compute_calibrated_depth(
    stabilized_zs: np.ndarray, dem_at_point: np.ndarray, weir_crest_fraction: float
) -> np.ndarray:
    """
    Required channel excavation depth from a stabilized calibration water
    level. Confining the design discharge to channel-width-only, with no
    channel yet excavated, forces water to pile up ABOVE natural ground
    level (the "rise"), so stabilized_zs > DEM at equilibrium -- but not all
    of that rise becomes excavation depth: a ``weir_crest_fraction`` of it
    is instead assigned to an actual production-model weir crest (see
    ``compute_calibrated_weir_crest``), so the two together (channel below
    DEM, weir above DEM) reproduce the full rise, matching how a real river
    system has both channel capacity and a levee, not one artificially deep
    channel with no freeboard structure.

    This slots directly into the existing burn-in mechanism unchanged:
    src.river_preburn.compute_river_bed_points already computes
    rivbed = DEM_conditioned_at_pixel - rivdph for every pixel along each
    reach's centerline.

    Args:
        stabilized_zs: Per-reach stabilized water level (m), same reference
            datum as the DEM.
        dem_at_point: Per-reach DEM elevation (m) at the reach's own
            midpoint (the same location the observation point sits at).
        weir_crest_fraction: Fraction of the rise (stabilized_zs -
            dem_at_point) assigned to the weir crest instead of excavation.

    Returns:
        Per-reach required excavation depth (m).
    """
    rise = stabilized_zs - dem_at_point
    return (1.0 - weir_crest_fraction) * rise


def compute_calibrated_weir_crest(
    stabilized_zs: np.ndarray, dem_at_point: np.ndarray, weir_crest_fraction: float
) -> np.ndarray:
    """
    Required production-model riverbank weir crest elevation (absolute, not
    a depth) from a stabilized calibration water level -- the complement of
    ``compute_calibrated_depth``: ``weir_crest_fraction`` of the rise above
    the DEM becomes an actual weir crest sitting above natural ground,
    rather than additional channel excavation.

    Args:
        stabilized_zs: Per-reach stabilized water level (m), same reference
            datum as the DEM.
        dem_at_point: Per-reach DEM elevation (m) at the reach's own
            midpoint (the same location the observation point sits at).
        weir_crest_fraction: Fraction of the rise (stabilized_zs -
            dem_at_point) assigned to the weir crest.

    Returns:
        Per-reach weir crest elevation (m), same reference datum as the DEM.
    """
    rise = stabilized_zs - dem_at_point
    return dem_at_point + weir_crest_fraction * rise


def sample_model_bed_at_points(dep_da, points: gpd.GeoDataFrame) -> np.ndarray:
    """
    Sample the SFINCS model's OWN bed elevation (sf.grid.data["dep"]) at
    each point -- NOT a raw elevation raster read from disk.

    stabilized_zs (from check_convergence) is the water level SFINCS
    actually computed at each observation point's own grid cell, measured
    against THIS bed elevation array -- SFINCS's physics guarantees water
    level >= bed elevation there. Sampling a separately-read, raw elevation
    file instead (even with a nodata fallback) can legitimately disagree
    with the model's own bed value at that exact cell, since hydromt's own
    elevation.create() step already gap-fills small holes when building
    ``dep`` -- which could produce a dem_at_point higher than
    stabilized_zs, a physically impossible result once both quantities are
    measured from the same source. Sampling dep_da directly instead
    guarantees stabilized_zs >= dem_at_point by construction, for every
    observation point, not just the ones that happen to land on a raw-file
    nodata gap.

    Args:
        dep_da: The model's own bed elevation DataArray
            (sf.grid.data["dep"]) -- regular grid only (uses
            dep_da.raster.transform/.crs).
        points: GeoDataFrame of Point geometries, any CRS.

    Returns:
        Array of sampled bed elevations, same length/order as ``points``.
    """
    transform = dep_da.raster.transform
    crs = dep_da.raster.crs
    dep_arr = dep_da.values
    nrows, ncols = dep_arr.shape
    pts = points.to_crs(crs) if points.crs != crs else points
    inv_transform = ~transform

    values = np.empty(len(pts), dtype=float)
    for i, p in enumerate(pts.geometry):
        col, row = inv_transform * (p.x, p.y)
        row_i = min(max(int(row), 0), nrows - 1)
        col_i = min(max(int(col), 0), ncols - 1)
        values[i] = float(dep_arr[row_i, col_i])
    return values
