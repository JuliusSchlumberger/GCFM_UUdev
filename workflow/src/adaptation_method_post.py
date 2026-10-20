"""Post-processing adaptation measures applied to a max flood depth raster.

Applied AFTER the SFINCS run; no model rerun. Entry point: dispatch_rules().
"""

import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rioxarray as rxr
from hydromt_sfincs import SfincsModel
from rasterio.features import geometry_mask
from shapely.ops import polygonize, unary_union
from src.grid import cell_area_m2

# Types of flooding attribution classes (used in the attribution mask)
# 1 = river-only
# 2 = coastal-only
# 3 = compound


def _is_current_protection(elevation) -> bool:
    """True for the {"current": True, "fill_crest": ...} form of `current_protection`.

    Current protection never clears a flood class (nor "holds" in the joint
    compound check): the baseline flood map already contains today's levees.
    """
    return isinstance(elevation, dict) and bool(elevation.get("current"))


def _crest_label(elevation) -> str:
    return (
        "current protection" if _is_current_protection(elevation) else f"{elevation}m"
    )


# Advance
def apply_offshore_barrier(
    flood_map_path: str, scenario_root: str, output_dir: str, elevation: float, **kwargs
) -> str:
    """Remove coastal and compound flooding if the barrier tops the max coastal water level.

    elevation >= max coastal water level of the scenario: classes 2 and 3 are
    cleared; otherwise full coastal flood risk remains.
    """

    out_path = Path(output_dir) / "max_flood_depth.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mod = SfincsModel(root=scenario_root, mode="r")
    wl = mod.get_component("water_level")
    wl.read()
    max_wl = float(wl.data["bzs"].max())

    with rasterio.open(flood_map_path) as src:
        flood, prof = src.read(1), src.profile

    mask_path = Path(scenario_root).parent / "attribution_mask.tif"
    if not mask_path.exists():
        raise FileNotFoundError(f"Attribution mask not found at {mask_path}")

    if elevation >= max_wl:
        with rasterio.open(
            mask_path
        ) as attr_src:  # opens attribution mask which labels each pixel by flood source
            flood = np.where(
                np.isin(attr_src.read(1), [2, 3]), prof["nodata"], flood
            )  # if barrier height exceeds surge, the pixels in 2 are replaced with nodata
        print(
            f"  Barrier sufficient (H={elevation}m >= Surge={max_wl:.2f}m) = all coastal flooding removed"
        )
    else:  # if the barrier is overtopped, we keep all flooding
        print(
            f"  Barrier overtopped (H={elevation}m < Surge={max_wl:.2f}m) = full risk"
        )

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
    """Remove river-only flooding (class 1) if the levee tops the max river water level.

    Class 3 (compound) is cleared only if BOTH levees hold, since either source
    alone could cause it: this function owns that joint decision (apply_coastal_levee
    defers class 3 here when a sibling river_levee exists), reading the sibling
    coastal_levee's elevation and coastal water level from strategy_measures.
    With no sibling coastal_levee, class 3 stays at full risk.
    """
    out_path = Path(output_dir) / "max_flood_depth.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mod = SfincsModel(root=scenario_root, mode="r")

    # River water level (observation points)
    riv_wl = mod.get_component("output")
    riv_wl.read()
    max_river_wl = float(riv_wl.data["point_zs"].max())

    river_current = _is_current_protection(elevation)
    river_holds = (not river_current) and elevation >= max_river_wl

    coastal_measure = (strategy_measures or {}).get("coastal_levee")
    if coastal_measure is not None:
        coastal_elevation = coastal_measure["elevation"]
        wl = mod.get_component("water_level")
        wl.read()
        max_coastal_wl = float(wl.data["bzs"].max())
        coastal_holds = (not _is_current_protection(coastal_elevation)) and float(
            coastal_elevation
        ) >= max_coastal_wl
        compound_holds = river_holds and coastal_holds
        print(
            f"  Joint compound check: river {'holds' if river_holds else 'overtopped'} "
            f"(H={_crest_label(elevation)} vs river WL={max_river_wl:.2f}m), coastal "
            f"{'holds' if coastal_holds else 'overtopped'} "
            f"(H={_crest_label(coastal_elevation)} vs surge={max_coastal_wl:.2f}m) "
            f"-> compound flooding {'removed' if compound_holds else 'kept at risk'}"
        )
    else:
        compound_holds = (
            False  # no sibling coastal_levee: compound is never cleared here
        )

    classes_to_clear = [1] + ([3] if compound_holds else [])

    with rasterio.open(flood_map_path) as src:
        flood, prof = src.read(1), src.profile

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
            f"  Levee sufficient (H={elevation}m >= river water level ={max_river_wl:.2f}m) = all river flooding removed"
        )
    elif river_current:  # today's levee is already in the baseline flood map
        print(
            "  River levee at current protection = no change to the baseline flooding"
        )
    else:  # if the levee is overtopped, we keep all flooding
        print(
            f"  Levee overtopped (H={elevation}m < river water level={max_river_wl:.2f}m) = full risk"
        )

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
    """Lower coastal and compound flood depths for a vegetated foreshore (managed advance).

    Empirical, no rerun: a uniform water-level reduction dd = (D / 1000) * r is
    subtracted from all cells attributed to coastal and compound flooding (depth
    is water level minus bed, so this equals a uniform lowering of the surface).
    distance D: belt width [m]. attenuation_rate r: water-level (surge)
    attenuation [m per km of belt width], NOT wave attenuation: wetland/saltmarsh
    0.017-0.25 m/km (Wamsley et al. 2010), mangrove 0.05-0.50 m/km (McIvor et al. 2012).
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

    with rasterio.open(flood_map_path) as src:
        flood, prof = src.read(1).astype("float32"), src.profile
        src_nodata = src.nodata

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
    """Remove storage_fraction * baseline_excess_volume from river and compound flooding.

    The volume (classes 1, 3) is taken off as a SINGLE uniform depth `tau` [m],
    solved so that sum(min(depth_i, tau)) * cell_area equals the target exactly
    (see _uniform_layer_depth), capped at the volume present; the mean depth
    target_volume / target_area would under-remove when depths are not uniform.
    storage_fraction=1 drains every target cell.

    `baseline_excess_volume` [m3] is a fixed reference from rule attribution_mask
    (compute_excess_volume, classes=(1, 3)), auto-injected by 18b_adapt_post.py and
    shared with src.adaptation_method_pre.apply_water_retention, so storage_fraction
    means the same in both methods. In a chained post-processing pipeline (see
    adaptation.py) `flood_map_path` may already reflect earlier measures; depths are
    reduced from it as given, but the volume sizing stays pinned to the baseline.
    The realized removed volume is reported next to the nominal target.
    scenario_root: the scenario's sfincs/ root; attribution_mask.tif is in its parent.
    """
    if not (0.0 <= storage_fraction <= 1.0):
        raise ValueError(f"storage_fraction must be in [0, 1], got {storage_fraction}")
    if baseline_excess_volume < 0:
        raise ValueError(
            f"baseline_excess_volume must be >= 0, got {baseline_excess_volume}"
        )

    out_path = Path(output_dir) / "max_flood_depth.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if baseline_excess_volume == 0:
        # No river/compound flooding (e.g. coastal-only): pass the map through
        # unchanged, as the preprocessing sibling leaves its model unchanged.
        print(
            "  WARNING: baseline_excess_volume is 0 (no river/compound flooding in this "
            "scenario) -- water retention has nothing to store, flood map left unchanged."
        )
        if Path(flood_map_path).resolve() != out_path.resolve():
            shutil.copy2(flood_map_path, out_path)
        return {"method": "flood_map", "out_raster": str(out_path)}

    with rasterio.open(flood_map_path) as src:
        flood, prof = src.read(1), src.profile
        cell_area = cell_area_m2(src.transform)

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

    # Solved uniform depth, not the naive target_volume/target_area mean, which
    # under-removes whenever depths aren't uniform.
    reduction_depth = _uniform_layer_depth(depths, target_volume, cell_area)
    remaining = np.maximum(depths - reduction_depth, 0.0)

    # Fully drained cells must become NODATA, not 0.0: downstream metrics
    # (18b_adapt_post.py's inundation-ratio plot, compute_risk_metrics'
    # flooded_area_km2) count "flooded" via da_hmax.notnull(), so a literal 0.0
    # would still count as flooded.
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
    """Remove coastal (and compound) flooding if the levee tops the max coastal water level.

    elevation >= max coastal water level: class 2 is cleared, and class 3 too
    unless a sibling "river_levee" is in strategy_measures (protect_open: coast and
    river are independent open structures, so apply_river_levee decides class 3
    jointly). Without one (protect_closed: the barrier also seals the river mouth)
    class 3 is cleared here. Otherwise full coastal flood risk remains.
    """

    out_path = Path(output_dir) / "max_flood_depth.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mod = SfincsModel(root=scenario_root, mode="r")
    wl = mod.get_component("water_level")
    wl.read()
    max_wl = float(wl.data["bzs"].max())

    with rasterio.open(flood_map_path) as src:
        flood, prof = src.read(1), src.profile

    mask_path = Path(scenario_root).parent / "attribution_mask.tif"
    if not mask_path.exists():
        raise FileNotFoundError(f"Attribution mask not found at {mask_path}")

    has_sibling_river_levee = (
        bool(strategy_measures) and "river_levee" in strategy_measures
    )
    classes_to_clear = [2] if has_sibling_river_levee else [2, 3]

    if _is_current_protection(
        elevation
    ):  # today's levee is already in the baseline flood map
        print(
            "  Coastal levee at current protection = no change to the baseline flooding"
        )
    elif elevation >= max_wl:
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
            f"  Levee sufficient (H={elevation}m >= Surge={max_wl:.2f}m) = all coastal flooding removed"
            f"{compound_note}"
        )
    else:  # if the levee is overtopped, we keep all flooding
        print(f"  Levee overtopped (H={elevation}m < Surge={max_wl:.2f}m) = full risk")

    prof.update(dtype="float32")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(flood.astype("float32"), 1)

    return {"method": "flood_map", "out_raster": str(out_path)}


def _uniform_layer_depth(depths: np.ndarray, volume: float, cell_area: float) -> float:
    """Return the layer thickness d whose removal (min(depth, d) per cell) equals `volume`.

    Equivalent to repeatedly splitting the remaining volume evenly over the cells
    that still have water. Walks cells shallowest first: removing depth h_k costs
    (water already taken from shallower cells + h_k x cells at least that deep) x
    cell_area; the first h_k whose cost exceeds `volume` bounds d. If `volume`
    exceeds all water present, d is the deepest depth (all cells -> 0).
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
    """Return the time [s] the river discharge is above its event-start level.

    Within tstart..tstop of the scenario's sfincs.inp; the start level is what the
    spin-up leaves (bankfull for a river RP scenario). Counted while any discharge
    point exceeds its own start level. 0 when there is no discharge forcing or it
    never rises (e.g. river_rp Mean or Null).
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
    """Remove the pumped volume from river and compound flooding as a uniform layer.

    Pumped volume = discharge [m3/s] x the river flood-wave duration
    (_river_flood_duration_s), not the whole run: flood water does not drain within
    the run and the run includes calm lead-in/tail hours. 0 when the discharge
    never rises (e.g. river_rp Mean). The volume is removed from classes 1 and 3
    via _uniform_layer_depth, so partial pump capacities give partial relief.
    scenario_root: sfincs/ root (sfincs.inp/.dis); attribution_mask.tif is in its parent.
    """
    out_path = Path(output_dir) / "max_flood_depth.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Flood duration: time the river discharge is above its spin-up level.
    duration_s = _river_flood_duration_s(scenario_root)
    pumped_volume = discharge * duration_s

    with rasterio.open(flood_map_path) as src:
        flood, prof = src.read(1), src.profile
        cell_area = cell_area_m2(src.transform)

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
    """Remove flood depth at the urban cells adaptation_method_pre.apply_urban_raising raises.

    No-rerun approximation: eligible cells are urban AND flooded (optionally inside
    `locations`); the deepest-flooded `raise_fraction` (as apply_retreat) lose
    their whole flood depth. Without a rerun there is no displaced-water effect,
    so `freeboard` is accepted (both siblings read the same yml) but unused;
    scenario_root is unused too (uniform dispatch signature). Raised cells are set
    to nodata, not 0.0: compute_risk_metrics tells flooded from dry via
    da_hmax.notnull(). urban_code: 50 by default.
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
    """Reclassify flooded urban cells to `target_code` in a new landuse raster.

    The flood map is copied unchanged. The deepest-flooded `retreat_fraction` of
    urban AND flooded cells move from `urban_code` to `target_code` (landuse only,
    no subgrid rebuild; dep_subgrid is preprocessing-only in measures.yml), and the
    new raster is returned as `landuse_path`. No risk_metrics.csv is written:
    18b_adapt_post.py computes every metric once, at the end of the measure chain,
    from the final flood and landuse rasters.
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
    """Check params against config/measures.yml bounds and required_for; raise ValueError."""
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
    """Validate params and call the post-processing function for `measure_type`.

    Entry point called by adaptation.py for post-processing method.
    landuse_path: read only by apply_retreat.
    strategy_measures: the strategy's full measures dict
    (config/adaptation_strategies.yml), passed through unvalidated, so
    apply_coastal_levee/apply_river_levee can decide compound (class 3) flooding
    jointly.
    **params: named parameter values from SimulationConfig (e.g. height=2.0)
    Returns {"method": ..., "out_raster": adapted flood map path}.
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
