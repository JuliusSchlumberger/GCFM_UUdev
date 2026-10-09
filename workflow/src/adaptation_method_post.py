# Applied AFTER the SFINCS run, directly to the flood depth raster.
# No model re-run required.

import shutil
import numpy as np
import rasterio
import rioxarray as rxr
from pathlib import Path
from hydromt_sfincs import SfincsModel

import geopandas as gpd
from shapely.ops import polygonize, unary_union
from rasterio.features import geometry_mask

# Types of flooding attribution classes (used in the attribution mask)
# 1 = river-only
# 2 = coastal-only
# 3 = compound


# Advance
def apply_offshore_barrier(
    flood_map_path: str, scenario_root: str, output_dir: str, elevation: float, **kwargs
) -> str:
    """
    Post-processing offshore barrier rule.
    Logic:
      - Read the max coastal water level from the scenario model.
      - If barrier height > max water level = all coastal flooding removed.
        Remove all flooding in attribution classes 2 and 3 (coastal and compound).
      - If barrier height <= max water level = full risk from coastal flooding.
    """

    out_path = Path(output_dir) / "max_flood_depth.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Get max coastal water level
    mod = SfincsModel(root=scenario_root, mode="r")
    wl = mod.get_component("water_level")
    wl.read()
    max_wl = float(wl.data["bzs"].max())

    # Read flood map raster
    with rasterio.open(flood_map_path) as src:
        flood, prof = src.read(1), src.profile

    # Apply conditional mask
    mask_path = Path(scenario_root).parent / "attribution_mask.tif"
    if not mask_path.exists():
        raise FileNotFoundError(f"Attribution mask not found at {mask_path}")

    if elevation > max_wl:
        with rasterio.open(
            mask_path
        ) as attr_src:  # opens attribution mask which labels each pixel by flood source
            flood = np.where(
                np.isin(attr_src.read(1), [2, 3]), prof["nodata"], flood
            )  # if barrier height exceeds surge, the pixels in 2 are replaced with nodata
        print(
            f"  Barrier sufficient (H={elevation}m > Surge={max_wl:.2f}m) = all coastal flooding removed"
        )
    else:  # if the barrier is overtopped, we keep all flooding
        print(
            f"  Barrier overtopped (H={elevation}m <= Surge={max_wl:.2f}m) = full risk"
        )

    # Save modified raster with updated metadata
    prof.update(dtype="float32")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(flood.astype("float32"), 1)

    return {"method": "flood_map", "out_raster": str(out_path)}


# Grey protect-open
def apply_river_levee(
    flood_map_path: str,
    scenario_root: str,
    output_dir: str,
    elevation: float,
    strategy_measures: dict = None,
    **kwargs,
) -> dict:
    """
    Post-processing river levee rule.
    Logic:
      - Read max river water level (e.g. obs points / dis forcing) from the scenario model.
      - If river_height > max river water level: remove flooding in class 1 (river-only).
      - Class 3 (compound) flooding is only removed if BOTH levees hold, since
        either source alone could cause it -- this function owns that joint
        decision (apply_coastal_levee defers class 3 to here whenever a
        sibling river_levee measure is present, see its own docstring), by
        reading the sibling coastal_levee's own elevation and coastal water
        level directly out of strategy_measures (this strategy's own full
        measures dict, see dispatch_rules). With no sibling coastal_levee in
        the strategy, class 3 is left untouched (still at full risk) rather
        than cleared based on river alone -- compound flooding by definition
        needs both sources to be individually contained.
    """
    out_path = Path(output_dir) / "max_flood_depth.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mod = SfincsModel(root=scenario_root, mode="r")

    # River water level (observation points)
    riv_wl = mod.get_component("output")
    riv_wl.read()
    max_river_wl = float(riv_wl.data["point_zs"].max())

    river_holds = elevation > max_river_wl

    coastal_measure = (strategy_measures or {}).get("coastal_levee")
    if coastal_measure is not None:
        coastal_elevation = float(coastal_measure["elevation"])
        wl = mod.get_component("water_level")
        wl.read()
        max_coastal_wl = float(wl.data["bzs"].max())
        coastal_holds = coastal_elevation > max_coastal_wl
        compound_holds = river_holds and coastal_holds
        print(
            f"  Joint compound check: river {'holds' if river_holds else 'overtopped'} "
            f"(H={elevation}m vs river WL={max_river_wl:.2f}m), coastal "
            f"{'holds' if coastal_holds else 'overtopped'} "
            f"(H={coastal_elevation}m vs surge={max_coastal_wl:.2f}m) "
            f"-> compound flooding {'removed' if compound_holds else 'kept at risk'}"
        )
    else:
        compound_holds = (
            False  # no sibling coastal_levee: compound is never cleared here
        )

    classes_to_clear = [1] + ([3] if compound_holds else [])

    # Read flood map raster
    with rasterio.open(flood_map_path) as src:
        flood, prof = src.read(1), src.profile

    # Apply conditional mask
    mask_path = Path(scenario_root).parent / "attribution_mask.tif"
    if not mask_path.exists():
        raise FileNotFoundError(f"Attribution mask not found at {mask_path}")

    if river_holds:
        with rasterio.open(
            mask_path
        ) as attr_src:  # opens attribution mask which labels each pixel by flood source
            flood = np.where(
                np.isin(attr_src.read(1), classes_to_clear), prof["nodata"], flood
            )
        print(
            f"  Levee sufficient (H={elevation}m > river water level ={max_river_wl:.2f}m) = all river flooding removed"
        )
    else:  # if the levee is overtopped, we keep all flooding
        print(
            f"  Levee overtopped (H={elevation}m <= river water level={max_river_wl:.2f}m) = full risk"
        )

    # Save modified raster with updated metadata
    prof.update(dtype="float32", nodata=prof.get("nodata", np.nan))
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(flood.astype("float32"), 1)

    return {"method": "flood_map", "out_raster": str(out_path)}


# NBS protect-open
def apply_nbs_land_reclamation(
    flood_map_path: str,
    scenario_root: str,
    output_dir: str,
    distance: float,
    attenuation_rate: float,
    attribution_codes: tuple = (2, 3),
    **kwargs,
) -> dict:
    """
    Post-processing created vegetated foreshore ("managed advance").

    The measure is represented by an empirical reduction coefficient applied to
    an existing flood map, without re-running the model. A uniform water-level
    reduction dd = (D / 1000) * r is subtracted from all cells attributed to
    coastal and compound flooding. Because depth is the difference between water
    level and bed elevation, this is equivalent to a uniform lowering of the
    water surface.

    r is a water-level (surge) attenuation rate, NOT a wave attenuation rate:
    wetland/saltmarsh 0.017-0.25 m/km (Wamsley et al. 2010), mangrove
    0.05-0.50 m/km (McIvor et al. 2012).

    Args:
        distance          : Foreshore / vegetation belt width [m]
        attenuation_rate  : Water-level attenuation rate r [m per km of belt width]
        attribution_codes : Attribution-mask codes treated as coastal or compound
    """
    out_path = Path(output_dir) / "max_flood_depth.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Below ~1200 m, van Zelst et al. (2021) report surge levels comparable to
    # bare tidal flats -- a linear r*D extrapolation here is unsupported.
    if distance < 1200:
        print(
            f"  WARNING: belt width {distance:.0f} m is below the 1200 m "
            f"threshold at which surge attenuation is evidenced."
        )

    # Distance [m] -> [km]; r is m per km
    reduction = (distance / 1000) * attenuation_rate
    print(
        f"  Uniform water-level reduction of {reduction:.3f} m "
        f"(D={distance:.0f} m, r={attenuation_rate} m/km)."
    )

    # Read flood map raster
    with rasterio.open(flood_map_path) as src:
        flood, prof = src.read(1).astype("float32"), src.profile
        src_nodata = src.nodata

    # Read attribution mask
    mask_path = Path(scenario_root).parent / "attribution_mask.tif"
    if not mask_path.exists():
        raise FileNotFoundError(f"Attribution mask not found at {mask_path}")
    with rasterio.open(mask_path) as attr_src:
        attr = attr_src.read(1)

    # NaN-safe valid mask: a plain `flood != nodata` comparison is wrong when
    # nodata is NaN, because NaN != NaN evaluates True -- nodata cells would be
    # treated as valid depths and reduced.
    if src_nodata is None or np.isnan(src_nodata):
        valid = np.isfinite(flood)
    else:
        valid = np.isfinite(flood) & (flood != src_nodata)

    is_coastal_flood = np.isin(attr, attribution_codes) & valid

    # Apply the reduction. Cells drained to zero or below take the same "dry"
    # value the baseline map uses, so the two are directly comparable.
    dry = src_nodata if src_nodata is not None else np.nan
    reduced_depth = flood - reduction
    flood = np.where(
        is_coastal_flood,
        np.where(reduced_depth <= 0, dry, reduced_depth),
        flood,
    )

    # Save modified raster
    prof.update(dtype="float32")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(flood.astype("float32"), 1)

    return {"method": "flood_map", "out_raster": str(out_path)}


def apply_water_retention(
    flood_map_path: str,
    output_dir: str,
    storage_fraction: float,
    scenario_root: str,
    baseline_excess_volume: float,
    **kwargs,
) -> dict:
    """
    Post-processing water retention rule.

    Logic:
        - Deducts fraction * baseline_excess_volume from the river+compound
          flood volume (classes 1, 3).
        - Translates the reduction into a SINGLE uniform depth `tau` [m],
          SOLVED (not `target_volume / target_area`) so that the volume
          actually removed -- sum(min(depth_i, tau)) * cell_area, accounting
          for cells whose own depth is shallower than `tau` and so clip to 0
          before giving up their "fair share" -- exactly equals target_volume
          (capped at whatever volume is actually present, if target_volume
          would otherwise exceed it). A naive `target_volume / target_area`
          (the mean depth) systematically UNDER-removes volume whenever
          depths aren't uniform, since shallow cells can't contribute more
          than their own depth -- solving for `tau` directly is what
          guarantees exact volume conservation, and as a consequence also
          guarantees full drainage of every target cell once
          storage_fraction=1 (target_volume equal to everything present).
        - Subtracts `tau` uniformly from those cells in the max flood depth
          map, clipped at 0.

    `baseline_excess_volume` is deliberately a fixed, externally-supplied
    reference -- computed ONCE per basin x scenario by rule attribution_mask
    (18c_attribution_mask.py, via src.postprocessing.compute_excess_volume,
    classes=(1, 3)) and read here from its baseline_excess_volume.json
    output (see 18b_adapt_post.py) -- rather than recomputed from
    `flood_map_path` here. This is the SAME reference number the preprocessing
    sibling (src.adaptation_method_pre.apply_water_retention) reads, so
    storage_fraction means the same thing in both methods. In a chained
    post-processing pipeline (see adaptation.py), `flood_map_path` is the
    OUTPUT of any earlier measure (e.g.
    nbs_land_reclamation's attenuation), so its own excess volume already
    reflects upstream reduction. Sizing storage_fraction against a moving,
    already-reduced volume would make storage_fraction mean different things
    run to run, and would no longer be comparable to the preprocessing
    DEM-lowering method, which is always sized against the same fixed baseline.
    The measure still reads and reduces depths from `flood_map_path` as given,
    so upstream measures' effects are preserved and compounded as before --
    only the volume used to size THIS measure's reduction is pinned.

    NOTE: because depth cannot go negative, the *nominal* volume reduction
    (storage_fraction * baseline_excess_volume) is only NOT fully realized
    when target_volume itself exceeds the volume actually present in the
    target cells (e.g. an upstream chained measure already removed more than
    baseline_excess_volume would suggest, or floating-point/reprojection
    drift between this run's own volume and the externally-computed
    baseline_excess_volume) -- solved `tau` is then capped at draining every
    target cell to 0 rather than solving past that. The realized removed
    volume is reported alongside the nominal target so this can be checked.

    Args:
        flood_map_path : path to the max flood depth raster to reduce (may
                         already reflect earlier measures in the chain)
        output_dir     : where to write the reduced raster
        storage_fraction : fraction (0-1) of baseline_excess_volume to retain/remove.
                         1.0 -> full removal in river+compound cells (every
                         target cell drains to 0), guaranteed by construction
                         (actual removal may be less where the reduction depth
                         exceeds a cell's flood depth).
        scenario_root  : the scenario's sfincs/ model root; attribution_mask.tif
                         lives in its parent folder
        baseline_excess_volume : excess flood volume [m3] from the uncontrolled
                         baseline run, classes (1, 3) (see
                         src.postprocessing.compute_excess_volume;
                         auto-injected by 18b_adapt_post.py, never
                         strategy-configured)

    Returns:
        dict with method and output raster path
    """
    if not (0.0 <= storage_fraction <= 1.0):
        raise ValueError(f"storage_fraction must be in [0, 1], got {storage_fraction}")
    if baseline_excess_volume <= 0:
        raise ValueError(
            f"baseline_excess_volume must be > 0, got {baseline_excess_volume}"
        )

    out_path = Path(output_dir) / "max_flood_depth.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(flood_map_path) as src:
        flood, prof = src.read(1), src.profile
        cell_area = abs(src.transform.a * src.transform.e)

    mask_path = Path(scenario_root).parent / "attribution_mask.tif"
    if not mask_path.exists():
        raise FileNotFoundError(f"Attribution mask not found at {mask_path}")

    with rasterio.open(mask_path) as attr_src:
        attr = attr_src.read(1)

    target_mask = (
        np.isin(attr, [1, 3]) & (flood > 0) & (flood != prof["nodata"])
    )  # river and compound

    target_volume = storage_fraction * baseline_excess_volume
    depths = flood[target_mask].astype(np.float64)
    target_area = depths.size * cell_area
    current_volume = float(depths.sum() * cell_area)

    # Single uniform depth such that sum(min(depth_i, reduction_depth)) *
    # cell_area equals target_volume (capped at the volume present) -- see
    # _uniform_layer_depth; not the naive target_volume/target_area mean,
    # which under-removes whenever depths aren't uniform.
    reduction_depth = _uniform_layer_depth(depths, target_volume, cell_area)
    remaining = np.maximum(depths - reduction_depth, 0.0)

    # Fully-drained cells (remaining == 0) must become NODATA, not a literal
    # 0.0 -- downstream (18b_adapt_post.py's own inundation-ratio plot and
    # compute_risk_metrics' flooded_area_km2) count "flooded" via
    # da_hmax.notnull(), not an actual depth threshold, same convention
    # apply_offshore_barrier/apply_coastal_levee/apply_river_levee already
    # follow (np.where(..., prof["nodata"], flood)). A literal 0.0 stays
    # "valid" and gets counted as still-flooded even though the real depth
    # is zero, silently inflating every area/extent metric back toward the
    # baseline's own flooded footprint.
    flood_new = flood.copy()
    flood_new[target_mask] = np.where(remaining <= 0, prof["nodata"], remaining)
    realized_volume = (
        current_volume - float(remaining.sum() * cell_area) if depths.size else 0.0
    )
    cap_note = (
        " [target_volume exceeds volume present in target cells; capped at full drainage]"
        if realized_volume < target_volume - 1e-6
        else ""
    )

    prof.update(dtype="float32")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(flood_new.astype("float32"), 1)

    print(
        f"  Applied water retention to river+compound flooding: storage_fraction={storage_fraction:.2f}, "
        f"reduction depth={reduction_depth:.4f} m over {target_area / 1e6:.2f} km2 -> "
        f"target={target_volume:.0f} m3, realized={realized_volume:.0f} m3{cap_note} "
        f"(baseline excess volume={baseline_excess_volume:.0f} m3, "
        f"pre-measure volume in this run={current_volume:.0f} m3)."
    )

    return {"method": "flood_map", "out_raster": str(out_path)}


# Protect-closed
def apply_coastal_levee(
    flood_map_path: str,
    scenario_root: str,
    output_dir: str,
    elevation: float,
    strategy_measures: dict = None,
    **kwargs,
) -> str:
    """
    Post-processing coastal levee rule.
    Logic:
      - Read the max coastal water level from the scenario model.
      - If levee height > max water level = all coastal flooding removed.
        Remove all flooding in attribution classes 2 and (usually) 3
        (coastal and compound) -- see strategy_measures note below.
      - If levee height <= max water level = full risk from coastal flooding.

    strategy_measures: this strategy's full measures dict (see dispatch_rules'
    own docstring). When a sibling "river_levee" measure is ALSO part of the
    same strategy (protect_open's own concept -- coast and river are two
    INDEPENDENT open structures), compound flooding requires BOTH to hold, so
    class 3 is left for apply_river_levee to decide jointly instead of being
    cleared here unilaterally. With no sibling river_levee (protect_closed's
    own concept -- the coastal barrier seals the river mouth too, so nothing
    else can independently cause a compound flood once it holds), class 3 is
    cleared here exactly as before.
    """

    out_path = Path(output_dir) / "max_flood_depth.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Get max coastal water level
    mod = SfincsModel(root=scenario_root, mode="r")
    wl = mod.get_component("water_level")
    wl.read()
    max_wl = float(wl.data["bzs"].max())

    # Read flood map raster
    with rasterio.open(flood_map_path) as src:
        flood, prof = src.read(1), src.profile

    # Apply conditional mask
    mask_path = Path(scenario_root).parent / "attribution_mask.tif"
    if not mask_path.exists():
        raise FileNotFoundError(f"Attribution mask not found at {mask_path}")

    has_sibling_river_levee = (
        bool(strategy_measures) and "river_levee" in strategy_measures
    )
    classes_to_clear = [2] if has_sibling_river_levee else [2, 3]

    if elevation > max_wl:
        with rasterio.open(
            mask_path
        ) as attr_src:  # opens attribution mask which labels each pixel by flood source
            flood = np.where(
                np.isin(attr_src.read(1), classes_to_clear), prof["nodata"], flood
            )  # if levee height exceeds surge, the pixels in classes_to_clear are replaced with nodata
        compound_note = (
            " (class 3/compound left for apply_river_levee's own joint check)"
            if has_sibling_river_levee
            else ""
        )
        print(
            f"  Levee sufficient (H={elevation}m > Surge={max_wl:.2f}m) = all coastal flooding removed"
            f"{compound_note}"
        )
    else:  # if the levee is overtopped, we keep all flooding
        print(f"  Levee overtopped (H={elevation}m <= Surge={max_wl:.2f}m) = full risk")

    # Save modified raster with updated metadata
    prof.update(dtype="float32")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(flood.astype("float32"), 1)

    return {"method": "flood_map", "out_raster": str(out_path)}


def _uniform_layer_depth(depths: np.ndarray, volume: float, cell_area: float) -> float:
    """
    Thickness d of the water layer to take off every wet cell so that the
    removed volume equals `volume`: each cell loses min(depth, d), so cells
    shallower than d are emptied and the volume they could not give comes
    from the deeper cells. Same result as repeatedly splitting the remaining
    volume evenly over the cells that still have water.

    Shallowest cell first: removing depth h_k from every cell that is still
    wet costs (water already taken from the shallower cells + h_k x the
    number of cells at least that deep) x cell_area. The first h_k whose
    cost exceeds `volume` is where the layer stops; d then follows from
    sharing the remaining volume over those deeper cells. If `volume`
    exceeds all the water present, d is the deepest depth (all cells -> 0).
    """
    h = np.sort(depths)
    n = len(h)
    if n == 0 or volume <= 0:
        return 0.0
    shallower = np.concatenate(
        [[0.0], np.cumsum(h)[:-1]]
    )  # water in the cells shallower than h_k
    cost = cell_area * (shallower + (n - np.arange(n)) * h)  # volume removed if d = h_k
    k = int(np.searchsorted(cost, volume))
    if k >= n:
        return float(h[-1])
    return float((volume / cell_area - shallower[k]) / (n - k))


def _river_flood_duration_s(scenario_root: str) -> float:
    """
    Time [s] within the event run (tstart..tstop of the scenario's
    sfincs.inp) during which the river discharge (sfincs.dis) is above the
    level it starts the event at, i.e. the level the spin-up leaves it at
    (bankfull for a river RP scenario): the duration of the river flood
    wave. Counted when any discharge point is above its own starting level.
    0 when the run has no discharge forcing or the discharge never rises
    (e.g. river_rp Mean or Null).
    """
    import pandas as pd

    root = Path(scenario_root)
    dis_path = root / "sfincs.dis"
    if not dis_path.exists():
        return 0.0
    inp = {
        line.split("=")[0].strip(): line.split("=", 1)[1].strip()
        for line in open(root / "sfincs.inp")
        if "=" in line
    }
    fmt = "%Y%m%d %H%M%S"
    tref = pd.to_datetime(inp["tref"], format=fmt)
    t0 = (pd.to_datetime(inp["tstart"], format=fmt) - tref).total_seconds()
    t1 = (pd.to_datetime(inp["tstop"], format=fmt) - tref).total_seconds()

    dis = np.atleast_2d(np.loadtxt(dis_path))
    t, q = dis[:, 0], dis[:, 1:]
    # 1-min grid over the event run, linear between the forcing steps (as
    # SFINCS interpolates them); start level = discharge at tstart.
    tt = np.arange(t0, t1, 60.0)
    q_tt = np.column_stack([np.interp(tt, t, q[:, i]) for i in range(q.shape[1])])
    rising = (q_tt > q_tt[0]).any(axis=1)
    return float(np.count_nonzero(rising) * 60.0)


# Advance/ Protect-closed
def apply_pumps(
    flood_map_path: str, scenario_root: str, output_dir: str, discharge: float, **kwargs
) -> dict:
    """
    Post-processing pumps rule.

    Logic:
        - Computes the pumped volume as discharge capacity [m3/s] * flood
          duration [s]: the time within the event run during which the
          river discharge is above the level the spin-up leaves it at
          (_river_flood_duration_s), i.e. the duration of the river flood
          wave. Not the whole simulation: flood water does not drain within
          the run and the run includes calm lead-in/tail hours, so duration
          from the run length or from the flood volume time series would
          cover the whole event. 0 when the discharge never rises (e.g.
          river_rp Mean), so the pumps then remove nothing.
        - Removes that volume from the river and compound cells (attribution
          classes 1, 3) of the max flood depth map as a uniform layer: the
          pumped volume is split evenly over the wet cells, cells without
          enough water are set to 0, and the volume they could not give is
          split evenly over the cells that still have water, until all of it
          is removed (computed directly by _uniform_layer_depth). Every cell
          loses min(depth, d); if the pumped volume exceeds all the water in
          those cells, they all become 0.

    This replaces an earlier binary rule (fully remove river+compound
    flooding if discharge >= peak river discharge, else no removal at all)
    with a graded, volume-based reduction so small/partial pump capacities
    show up as partial relief instead of nothing.

    Args:
        flood_map_path : path to the max flood depth raster to reduce (may
                         already reflect earlier measures in the chain)
        scenario_root  : the scenario's sfincs/ model root (sfincs.inp/.dis for
                         the flood duration); attribution_mask.tif lives in
                         its parent folder
        output_dir     : where to write the reduced raster
        discharge      : pump discharge capacity [m3/s]

    Returns:
        dict with method and output raster path
    """
    out_path = Path(output_dir) / "max_flood_depth.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Flood duration: time the river discharge is above its spin-up level.
    duration_s = _river_flood_duration_s(scenario_root)
    pumped_volume = discharge * duration_s

    with rasterio.open(flood_map_path) as src:
        flood, prof = src.read(1), src.profile
        cell_area = abs(src.transform.a * src.transform.e)

    mask_path = Path(scenario_root).parent / "attribution_mask.tif"
    if not mask_path.exists():
        raise FileNotFoundError(f"Attribution mask not found at {mask_path}")

    with rasterio.open(mask_path) as attr_src:
        attr = attr_src.read(1)

    target_mask = (
        np.isin(attr, [1, 3]) & (flood > 0) & (flood != prof["nodata"])
    )  # river and compound
    depths = flood[target_mask].astype(np.float64)
    target_area = depths.size * cell_area
    reduction_depth = _uniform_layer_depth(depths, pumped_volume, cell_area)
    remaining = np.maximum(depths - reduction_depth, 0.0)

    # Drained cells -> NODATA, not 0.0: downstream metrics count "flooded" via
    # notnull (see apply_water_retention).
    flood_new = flood.copy()
    flood_new[target_mask] = np.where(remaining <= 0, prof["nodata"], remaining)
    realized_volume = float((depths.sum() - remaining.sum()) * cell_area)
    cap_note = (
        " [pumped volume exceeds all river+compound flood water; those cells are now dry]"
        if realized_volume < pumped_volume * (1 - 1e-6)
        else ""
    )

    prof.update(dtype="float32")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(flood_new.astype("float32"), 1)

    print(
        f"  Applied pumps to river+compound flooding: discharge={discharge:.1f} m3/s x "
        f"flood duration (river above spin-up level)={duration_s / 3600:.1f} h -> pumped volume={pumped_volume:.0f} m3, "
        f"reduction depth={reduction_depth:.4f} m over {target_area / 1e6:.2f} km2, "
        f"realized={realized_volume:.0f} m3{cap_note}."
    )

    return {"method": "flood_map", "out_raster": str(out_path)}


def _ring_interior_mask(locations, out_shape, transform, raster_crs):
    """Boolean array: True where a cell falls inside the dike ring `locations`."""
    gdf = (
        locations.copy()
        if isinstance(locations, gpd.GeoDataFrame)
        else gpd.read_file(locations)
    )
    if gdf.crs is None:
        raise ValueError(
            "dike ring `locations` has no CRS; cannot align to the flood raster"
        )
    gdf = gdf.to_crs(raster_crs)

    if set(gdf.geom_type) <= {"Polygon", "MultiPolygon"}:
        polys = list(gdf.geometry)
    else:  # LineString ring -> fill the enclosed area
        polys = list(polygonize(unary_union(gdf.geometry.values)))
        if not polys:
            raise ValueError(
                "dike ring lines don't form a closed loop; snap/close endpoints first"
            )

    return geometry_mask(polys, out_shape=out_shape, transform=transform, invert=True)


# # Accommodate
# def apply_dike_ring(
#     flood_map_path: str,
#     scenario_root: str,
#     output_dir: str,
#     elevation: float,
#     locations: Path,  # was: urban_mask
#     **kwargs,
# ) -> dict:
#     """
#     Post-processing dike ring rule.

#     - Max coastal water level is read from the forcing timeseries (bzs) and
#       compared against the dike crest `elevation` in a binary check.
#     - Cells that are BOTH inside the dike ring (`locations`) AND flooded form the
#       protected set. `locations` now plays the role `urban_mask` used to: the old
#       rule assumed a ring around every urban area; this uses the actual ring.
#     - WL < crest  -> flooding removed in the protected set.
#     - WL >= crest -> full flood risk (no change).
#     """
#     out_path = Path(output_dir) / "max_flood_depth.tif"
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     # Max coastal water level from the forcing timeseries
#     mod = SfincsModel(root=scenario_root, mode="r")
#     wl = mod.get_component("water_level")
#     wl.read()
#     max_wl = float(wl.data["bzs"].max())

#     with rasterio.open(flood_map_path) as src:
#         flood = src.read(1).astype("float32")
#         prof = src.profile
#         nodata = src.nodata
#         inside_ring = _ring_interior_mask(
#             locations, (src.height, src.width), src.transform, src.crs
#         )

#     # "at risk of flooding" = cells that actually hold a flood depth
#     if nodata is not None:
#         flooded = flood != nodata
#     else:
#         flooded = np.isfinite(flood) & (flood > 0)

#     protected = inside_ring & flooded
#     fill = nodata if nodata is not None else 0.0

#     if elevation > max_wl:
#         flood[protected] = fill
#         print(
#             f"  Dike sufficient (H={elevation}m > WL={max_wl:.2f}m) = flooding removed inside ring"
#         )
#     else:
#         print(
#             f"  Dike overtopped (H={elevation}m <= WL={max_wl:.2f}m) = full risk inside ring"
#         )

#     prof.update(dtype="float32")
#     with rasterio.open(out_path, "w", **prof) as dst:
#         dst.write(flood.astype("float32"), 1)

#     return {"method": "flood_map", "out_raster": str(out_path)}


# Accommodate
def apply_urban_raising(
    flood_map_path: str,
    scenario_root: str,
    output_dir: str,
    raise_fraction: float = None,
    freeboard: float = 0.0,
    landuse_path: str = None,
    urban_code: int = 50,
    flood_threshold: float = 0.05,
    locations: Path = None,
    **kwargs,
) -> dict:
    """
    Post-processing urban raising rule.

    Fast, no-rerun approximation of adaptation_method_pre.py's own
    apply_urban_raising (which physically raises the DEM and reruns SFINCS):
    eligible cells are urban AND flooded (optionally restricted to
    `locations`); the deepest-flooded `raise_fraction` of them (same
    selection as apply_retreat) are raised by their own flood depth +
    freeboard, i.e. their flood depth is removed entirely. Without a rerun
    there is no displaced-water effect, so `freeboard` doesn't change the
    result here; it's accepted only so both siblings read the same yml.

    Raised cells are set to nodata (not 0.0) --
    matches apply_dike_ring's own convention above -- since
    compute_risk_metrics (src.postprocessing) tells flooded from dry purely
    via da_hmax.notnull(), not a depth threshold; a plain 0.0 would still
    read as "flooded".

    Args:
        flood_map_path  : (Required) baseline max_flood_depth.tif
        scenario_root   : (Required) scenario model root -- unused here,
                          accepted only for dispatch_rules' uniform call signature
        output_dir      : (Required) directory the adapted raster is written to
        raise_fraction  : (Optional) fraction of flooded urban cells to raise (0-1),
                          deepest flood depth first; if None, all eligible cells are raised
        freeboard       : (Optional) unused here, see above
        landuse_path    : (Required) path to the current landuse raster,
                          used only to determine which cells are urban
        urban_code      : (Optional) land use code marking urban cells (default 50)
        flood_threshold : (Optional) depth [m] above which a cell counts as flooded
        locations       : (Optional) polygon path/gdf; if None (the usual,
                          delta-wide case), all flooded urban cells are eligible

    Returns:
        {"method": "flood_map", "out_raster": path to the adapted max_flood_depth.tif}
    """
    if landuse_path is None:
        raise ValueError("apply_urban_raising needs landuse_path")
    if raise_fraction is not None and not 0.0 <= raise_fraction <= 1.0:
        raise ValueError(f"raise_fraction must be in [0, 1], got {raise_fraction}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "max_flood_depth.tif"

    with rasterio.open(flood_map_path) as src:
        flood = src.read(1).astype("float32")
        prof = src.profile
        nodata = src.nodata
        raster_crs = src.crs
        transform = src.transform
        out_shape = (src.height, src.width)

    da_lulc = rxr.open_rasterio(landuse_path).squeeze(drop=True)
    da_flood = rxr.open_rasterio(flood_map_path, masked=True).squeeze(drop=True)
    da_lulc = da_lulc.raster.reproject_like(da_flood, method="nearest")

    if nodata is not None:
        flooded = (flood != nodata) & (flood > flood_threshold)
    else:
        flooded = np.isfinite(flood) & (flood > flood_threshold)

    eligible = (da_lulc.values == urban_code) & flooded
    if locations is not None:
        eligible &= _ring_interior_mask(locations, out_shape, transform, raster_crs)

    # deepest-flooded urban cells raised first (same selection as apply_retreat)
    if raise_fraction is not None and raise_fraction < 1.0 and eligible.any():
        threshold = np.nanquantile(flood[eligible], 1.0 - raise_fraction)
        raise_mask = eligible & (flood >= threshold)
    else:
        raise_mask = eligible

    fill = nodata if nodata is not None else 0.0
    new_flood = flood.copy()
    new_flood[raise_mask] = fill

    print(
        f"  Urban raising (postprocessing): {int(raise_mask.sum())} of "
        f"{int(eligible.sum())} flooded urban cell(s) raised by their own depth (now dry)."
    )

    prof.update(dtype="float32")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(new_flood.astype("float32"), 1)

    return {"method": "flood_map", "out_raster": str(out_path)}


# Retreat
def apply_retreat(
    flood_map_path: str,
    output_dir: str,
    scenario_root: str,
    landuse_path: str,
    retreat_fraction: float,
    urban_code: int = 50,
    target_code: int = 60,
    flood_threshold: float = 0.05,
    **kwargs,
) -> dict:
    """
    Post-processing managed retreat rule.

    Unlike every other post-processing measure, retreat does not touch the
    hazard itself -- the flood map is copied unchanged. Instead it
    reclassifies the deepest-flooded `retreat_fraction` of eligible
    (urban AND flooded) cells from `urban_code` to `target_code` in a NEW
    landuse raster (mirrors adaptation_method_pre.py's own apply_retreat,
    landuse-only -- no subgrid rebuild, consistent with measures.yml marking
    dep_subgrid preprocessing-only), and returns its path as `landuse_path`.

    Deliberately does NOT write a separate risk_metrics.csv (the ported
    original's approach): 18b_adapt_post.py recomputes every metric fresh,
    ONCE, at the end of the whole measure chain, straight from the final
    flood raster + landuse raster -- a separate metrics file would be
    silently ignored by that final computation (and would also assume the
    OTHER project's own risk_metrics.csv schema, which this repo's
    flood_metrics.csv doesn't match). Returning landuse_path instead lets
    retreat's effect flow through that same final computation, the same way
    every other measure's raster edit already does via out_raster.
    """
    if landuse_path is None:
        raise ValueError("apply_retreat needs landuse_path")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_flood_path = out_dir / "max_flood_depth.tif"
    shutil.copy2(flood_map_path, out_flood_path)

    da_flood = rxr.open_rasterio(flood_map_path, masked=True).squeeze(drop=True)
    da_lulc = rxr.open_rasterio(landuse_path).squeeze(drop=True)
    # Landuse's native resolution may differ from the (subgrid-downscaled)
    # flood map's own grid -- align before any elementwise comparison.
    da_lulc = da_lulc.raster.reproject_like(da_flood, method="nearest")

    flooded = (da_flood > flood_threshold).values
    eligible = (da_lulc.values == urban_code) & flooded

    if 0.0 <= retreat_fraction < 1.0 and eligible.any():
        depth_vals = da_flood.values[eligible]
        depth_cutoff = np.nanquantile(depth_vals, 1.0 - retreat_fraction)
        eligible &= da_flood.values >= depth_cutoff

    lulc_new = da_lulc.values.copy()
    lulc_new[eligible] = target_code

    out_landuse_path = out_dir / "retreat_landuse.tif"
    da_lulc.copy(data=lulc_new).rio.to_raster(str(out_landuse_path))

    print(
        f"  Retreat: {int(eligible.sum())} urban cell(s) reclassified {urban_code} -> {target_code} "
        f"(retreat_fraction={retreat_fraction:.2f})"
    )

    return {
        "method": "flood_map",
        "out_raster": str(out_flood_path),
        "landuse_path": str(out_landuse_path),
    }


# -----------------------------------------------------------------------------
# ----------------------------- Selected measures -----------------------------
# -----------------------------------------------------------------------------

selected_measures = {
    "offshore_barrier": apply_offshore_barrier,
    "nbs_land_reclamation": apply_nbs_land_reclamation,
    "water_retention": apply_water_retention,
    "river_levee": apply_river_levee,
    "coastal_levee": apply_coastal_levee,
    "pumps": apply_pumps,
    "urban_raising": apply_urban_raising,
    # "dike_ring": apply_dike_ring,
    "retreat": apply_retreat,
}


def validate_measure_params(
    measure_type: str, params: dict, method: str, measure_def: dict
) -> None:
    """
    Validate a measure call's params against config/measures.yml's own bounds
    and required_for declarations, before dispatch -- catches a bad strategy
    definition (missing required param, out-of-range value) with a message
    naming the measure/param, instead of a confusing failure deep inside
    hydromt_sfincs.
    """
    param_specs = measure_def.get("params", {})
    for name, spec in param_specs.items():
        required_for = spec.get("required_for", [])
        if method in required_for and name not in params:
            raise ValueError(
                f"Measure '{measure_type}': param '{name}' is required for method "
                f"'{method}' (config/measures.yml) but was not provided."
            )
        if (
            name in params
            and spec.get("min") is not None
            and spec.get("max") is not None
        ):
            value = params[name]
            if isinstance(value, (int, float)) and not (
                spec["min"] <= value <= spec["max"]
            ):
                raise ValueError(
                    f"Measure '{measure_type}': param '{name}'={value} is out of bounds "
                    f"[{spec['min']}, {spec['max']}] (config/measures.yml)."
                )


def dispatch_rules(
    measure_type: str,
    flood_map_path: str,
    scenario_root: str,
    measure_def: dict,
    catalog_path: str,
    output_dir: str,
    landuse_path: str = None,
    method: str = "postprocessing",
    strategy_measures: dict = None,
    **params,
) -> dict:
    """
    Entry point called by adaptation.py for post-processing method.

    Args:
        measure_type   : string matching a key in selected_measures
        flood_map_path : path to the baseline max_flood_depth.tif
        scenario_root  : path to the scenario model root
        measure_def    : full measure definition dict from measures.yml
        catalog_path   : path to data catalog YAML
        output_dir     : directory to write the adapted raster
        landuse_path   : path to the current landuse raster -- only apply_retreat
                         reads this; every other measure ignores it via **kwargs
        strategy_measures : this strategy's own full measures dict (as declared
                         in config/adaptation_strategies.yml), passed through
                         unvalidated (never checked against measure_def's own
                         params -- it describes the STRATEGY, not this one
                         measure). Lets a measure look up a SIBLING measure's
                         own params -- currently only apply_coastal_levee/
                         apply_river_levee, to decide compound (class 3)
                         flooding jointly when both are part of the same
                         strategy instead of each acting on it independently.
        **params       : named parameter values from SimulationConfig (e.g. height=2.0)

    Returns:
        Dictionary containing the method type and path to the adapted flood map raster.
    """
    if measure_type not in selected_measures:
        raise ValueError(
            f"Unknown post-processing measure type '{measure_type}'. "
            f"Available: {list(selected_measures.keys())}"
        )

    validate_measure_params(measure_type, params, method, measure_def)
    return selected_measures[measure_type](
        flood_map_path=flood_map_path,
        scenario_root=scenario_root,
        output_dir=output_dir,
        landuse_path=landuse_path,
        strategy_measures=strategy_measures,
        **params,
    )
