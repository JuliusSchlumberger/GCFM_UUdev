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
# 4 = baseline/ spinup


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
                np.isin(attr_src.read(1), [2, 3, 4]), prof["nodata"], flood
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
    **kwargs,
) -> str:
    """
    Post-processing coastline extension rule.
    Logic:
        - Use distance and attentuation rate (based on land use of new coastline) to determine the extent of flood depth reduction.
        - Apply to coastal cells only (2)
    """

    out_path = Path(output_dir) / "max_flood_depth.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Translate the built distance into a vertical depth reduction
    # Distance in km * attenuation rate (m/km)
    reduction = (distance / 1000) * attenuation_rate
    print(
        f"  Calculated Attenuation: Results in a flat depth reduction of {reduction:.3f}m."
    )

    # Read flood map raster
    with rasterio.open(flood_map_path) as src:
        flood, prof = src.read(1), src.profile

    # Read attribution mask
    mask_path = Path(scenario_root).parent / "attribution_mask.tif"
    if not mask_path.exists():
        raise FileNotFoundError(f"Attribution mask not found at {mask_path}")

    with rasterio.open(mask_path) as attr_src:
        attr = attr_src.read(1)

    # Filter coastal flooding and compound pixels
    is_coastal_flood = (np.isin(attr, [2, 3, 4])) & (flood != prof["nodata"])

    # Apply the physically derived reduction
    reduced_depth = flood - reduction
    flood = np.where(
        is_coastal_flood,
        np.where(reduced_depth <= 0, prof["nodata"], reduced_depth),
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
          flood volume (classes 1, 3, 4).
        - Translates the reduction into a uniform depth reduction [m], analogous to
          the DEM-lowering computed by the preprocessing equivalent
          (apply_water_retention in preprocessing_adaptation.py): reduction =
          target_volume / target_area.
        - Subtracts that depth uniformly from those cells in the max flood depth
          map, clipped at 0.

    `baseline_excess_volume` is deliberately a fixed, externally-supplied
    reference -- computed ONCE via compute_excess_volume() on the uncontrolled
    baseline run with classes=(1, 3, 4) (see main.py) -- rather than recomputed
    from `flood_map_path` here. In a chained post-processing pipeline (see
    adaptation.py), `flood_map_path` is the OUTPUT of any earlier measure (e.g.
    nbs_land_reclamation's attenuation), so its own excess volume already
    reflects upstream reduction. Sizing storage_fraction against a moving,
    already-reduced volume would make storage_fraction mean different things
    run to run, and would no longer be comparable to the preprocessing
    DEM-lowering method, which is always sized against the same fixed baseline.
    The measure still reads and reduces depths from `flood_map_path` as given,
    so upstream measures' effects are preserved and compounded as before --
    only the volume used to size THIS measure's reduction is pinned.

    NOTE: because depth cannot go negative, the *nominal* volume reduction
    (storage_fraction * baseline_excess_volume) is not guaranteed to be fully
    realized -- cells where the reduction depth exceeds the local flood depth
    are simply clipped to 0, removing less than their nominal share. The
    realized removed volume is reported alongside the nominal target so this
    can be checked.

    Args:
        flood_map_path : path to the max flood depth raster to reduce (may
                         already reflect earlier measures in the chain)
        output_dir     : where to write the reduced raster
        storage_fraction : fraction (0-1) of baseline_excess_volume to retain/remove.
                         1.0 -> nominal full removal in river+compound cells
                         (actual removal may be less where the reduction depth
                         exceeds a cell's flood depth).
        scenario_root  : the scenario's sfincs/ model root; attribution_mask.tif
                         lives in its parent folder
        baseline_excess_volume : excess flood volume [m3] from the uncontrolled
                         baseline run, classes (1, 3, 4) (see compute_excess_volume)

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
        np.isin(attr, [1, 3, 4]) & (flood > 0) & (flood != prof["nodata"])
    )  # river and compound

    target_volume = storage_fraction * baseline_excess_volume
    target_area = np.count_nonzero(target_mask) * cell_area
    reduction_depth = target_volume / target_area if target_area > 0 else 0.0

    flood_new = np.where(target_mask, np.maximum(flood - reduction_depth, 0.0), flood)
    current_volume = np.sum(flood[target_mask] * cell_area)
    realized_volume = current_volume - np.sum(flood_new[target_mask] * cell_area)
    cap_note = (
        " [reduction depth exceeded local depth in some cells; target not fully met]"
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
    classes_to_clear = [2, 4] if has_sibling_river_levee else [2, 3, 4]

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


def apply_pumps(
    flood_map_path: str, scenario_root: str, output_dir: str, discharge: float, **kwargs
) -> dict:
    """
    Post-processing pumps rule.

    Logic:
        - Computes the pumped volume as discharge capacity [m3/s] * event
          duration [s] (tstop - tstart from the scenario model config) --
          the volume the pump could evacuate running continuously for the
          whole modeled event, matching how the preprocessing pump structure
          (mod.drainage_structures.create()) operates: it discharges up to
          `discharge` m3/s whenever the water level is between zmin/zmax,
          not as a single on/off switch at peak discharge.
        - Translates that volume into a uniform depth reduction [m] over the
          river and compound cells (attribution classes 1, 3), the same
          volume -> depth conversion used by apply_water_retention():
          reduction = pumped_volume / target_area.
        - Subtracts that depth uniformly from those cells in the max flood
          depth map, clipped at 0.

    This replaces an earlier binary rule (fully remove river+compound
    flooding if discharge >= peak river discharge, else no removal at all)
    with a graded, volume-based reduction so small/partial pump capacities
    show up as partial relief instead of nothing.

    NOTE: cells where the reduction depth
    exceeds the local flood depth are simply clipped to 0. The realized
    removed volume is reported alongside the nominal pumped volume.

    Args:
        flood_map_path : path to the max flood depth raster to reduce (may
                         already reflect earlier measures in the chain)
        scenario_root  : the scenario's sfincs/ model root (for tstart/tstop);
                         attribution_mask.tif lives in its parent folder
        output_dir     : where to write the reduced raster
        discharge      : pump discharge capacity [m3/s]

    Returns:
        dict with method and output raster path
    """
    out_path = Path(output_dir) / "max_flood_depth.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Event duration: the pump runs continuously over the simulated event,
    # same as the preprocessing drainage structure.
    mod = SfincsModel(root=scenario_root, mode="r")
    tstart, tstop = mod.get_model_time()
    duration_s = (tstop - tstart).total_seconds()
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
    target_area = np.count_nonzero(target_mask) * cell_area
    reduction_depth = pumped_volume / target_area if target_area > 0 else 0.0

    flood_new = np.where(target_mask, np.maximum(flood - reduction_depth, 0.0), flood)
    current_volume = np.sum(flood[target_mask] * cell_area)
    realized_volume = current_volume - np.sum(flood_new[target_mask] * cell_area)
    cap_note = (
        " [reduction depth exceeded local depth in some cells; pumped volume not fully realized]"
        if realized_volume < pumped_volume - 1e-6
        else ""
    )

    prof.update(dtype="float32")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(flood_new.astype("float32"), 1)

    print(
        f"  Applied pumps to river+compound flooding: discharge={discharge:.1f} m3/s x "
        f"duration={duration_s / 86400:.1f} d -> pumped volume={pumped_volume:.0f} m3, "
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


# Accommodate
def apply_dike_ring(
    flood_map_path: str,
    scenario_root: str,
    output_dir: str,
    elevation: float,
    locations: Path,  # was: urban_mask
    **kwargs,
) -> dict:
    """
    Post-processing dike ring rule.

    - Max coastal water level is read from the forcing timeseries (bzs) and
      compared against the dike crest `elevation` in a binary check.
    - Cells that are BOTH inside the dike ring (`locations`) AND flooded form the
      protected set. `locations` now plays the role `urban_mask` used to: the old
      rule assumed a ring around every urban area; this uses the actual ring.
    - WL < crest  -> flooding removed in the protected set.
    - WL >= crest -> full flood risk (no change).
    """
    out_path = Path(output_dir) / "max_flood_depth.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Max coastal water level from the forcing timeseries
    mod = SfincsModel(root=scenario_root, mode="r")
    wl = mod.get_component("water_level")
    wl.read()
    max_wl = float(wl.data["bzs"].max())

    with rasterio.open(flood_map_path) as src:
        flood = src.read(1).astype("float32")
        prof = src.profile
        nodata = src.nodata
        inside_ring = _ring_interior_mask(
            locations, (src.height, src.width), src.transform, src.crs
        )

    # "at risk of flooding" = cells that actually hold a flood depth
    if nodata is not None:
        flooded = flood != nodata
    else:
        flooded = np.isfinite(flood) & (flood > 0)

    protected = inside_ring & flooded
    fill = nodata if nodata is not None else 0.0

    if elevation > max_wl:
        flood[protected] = fill
        print(
            f"  Dike sufficient (H={elevation}m > WL={max_wl:.2f}m) = flooding removed inside ring"
        )
    else:
        print(
            f"  Dike overtopped (H={elevation}m <= WL={max_wl:.2f}m) = full risk inside ring"
        )

    prof.update(dtype="float32")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(flood.astype("float32"), 1)

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
    "dike_ring": apply_dike_ring,
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
