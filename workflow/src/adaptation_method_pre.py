"""
adaptation_measures.py -- Pre-processing adaptation measures: functions that
mutate an open SfincsModel in place, one per measure type declared in
config/measures.yml. Called by 18_adapt_apply.py, once per measure in a
strategy's own config/adaptation_strategies.yml entry, via dispatch_rules().

Ported from the author's separate delta_model project
(preprocessing_adaptation.py) onto this repo's own hydromt_sfincs API
(SfincsWeirs.create / SfincsDrainageStructures.create / SubgridComponent.create
-- confirmed signature-compatible with the ported calls below) and its own
SfincsModel version. The multi-measure-per-strategy chaining behaviour
(current_dep_subgrid forwarding between measures that both rebuild the
subgrid, mirroring the other project's apply_adaptation()) lives in
18_adapt_apply.py, not here, since it's orchestration across measures, not
any single measure's own logic.

nbs_land_reclamation / water_retention are deliberately NOT ported here yet:
water_retention's compute_excess_volume() needs attribution_mask.tif (per-pixel
river/coastal/compound classification), which doesn't exist in this repo --
same prerequisite the postprocessing method needs. See the project plan.
"""

from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from hydromt_sfincs import SfincsModel
from rasterio.features import rasterize

from src.protection_weir import merge_weir_preserve_unmatched


def _infer_nr_subgrid_pixels(mod: SfincsModel, dep: xr.DataArray) -> int:
    """Infer the model's subgrid refinement factor from dep_subgrid and the coarse grid."""
    da_mask = mod.grid.mask
    coarse_res_x, coarse_res_y = (
        np.abs(da_mask.raster.res[0]),
        np.abs(da_mask.raster.res[1]),
    )
    dep_res_x, dep_res_y = (
        np.abs(dep.rio.resolution()[0]),
        np.abs(dep.rio.resolution()[1]),
    )

    if coarse_res_x == 0 or coarse_res_y == 0:
        raise ValueError(
            "Cannot infer nr_subgrid_pixels from a zero-resolution model grid."
        )
    if dep_res_x == 0 or dep_res_y == 0:
        raise ValueError(
            "Cannot infer nr_subgrid_pixels from a zero-resolution dep_subgrid raster."
        )

    ratio_x = coarse_res_x / dep_res_x
    ratio_y = coarse_res_y / dep_res_y
    if not np.isclose(ratio_x, ratio_y, rtol=1e-6, atol=1e-6):
        raise ValueError(
            f"Cannot infer nr_subgrid_pixels because dep_subgrid resolution "
            f"({dep_res_x:.6g}, {dep_res_y:.6g}) does not match model grid resolution "
            f"({coarse_res_x:.6g}, {coarse_res_y:.6g})."
        )

    nr = int(round((ratio_x + ratio_y) / 2.0))
    if nr < 2 or nr % 2 != 0:
        raise ValueError(
            f"Inferred nr_subgrid_pixels={nr} is invalid; expected a positive even integer."
        )
    if not np.isclose(ratio_x, nr, rtol=1e-6, atol=1e-6) or not np.isclose(
        ratio_y, nr, rtol=1e-6, atol=1e-6
    ):
        raise ValueError(
            f"Rounded nr_subgrid_pixels={nr} does not match the actual refinement ratio "
            f"({ratio_x:.6g}, {ratio_y:.6g})."
        )
    return nr


# ── Advance: offshore_barrier ─────────────────────────────────────────────────
def apply_offshore_barrier(
    mod: SfincsModel,
    locations: Path,
    elevation: float,
    par1: float = 0.6,
    dep: Optional[str] = None,
    buffer: Optional[float] = None,
    dz: Optional[float] = None,
    merge: bool = True,
    **kwargs,
) -> SfincsModel:
    """
    Adds offshore barriers as weir line, built from a user-supplied polyline geojson.
    Called before mod.write() and the SFINCS run (pre-processing method).

    Args:
        mod       : (Required) Open SfincsModel object (HydroMT)
        elevation : (Required from measures.yml) Barrier crest elevation [m above datum], assigned to the whole line.
        locations : (Required from measures.yml) Path, data source name, or GeoDataFrame with the weir polyline(s).
        par1      : (Optional) Weir discharge coefficient, default 0.6.
        dep       : (Optional) Alternative elevation raster to sample crest height from.
        buffer    : (Optional) Distance (m) from centerline used as sampling window for dep.
        dz        : (Optional) Vertical offset added to the elevation sampled from dep.
        merge     : (Optional) If True, merge with any existing weir lines instead of overwriting.

    Returns:
        Modified SfincsModel object.
    """

    mod.weirs.create(
        locations=locations,
        elevation=elevation,
        par1=par1,
        dep=dep,
        buffer=buffer,
        dz=dz,
        merge=merge,
    )

    return mod


# ── Grey protect-open: river_levee ───────────────────────────────
def apply_river_levee(
    mod: SfincsModel,
    locations: Path,
    elevation: float,
    par1: float = 0.6,
    dep: Optional[str] = None,
    buffer: Optional[float] = None,
    dz: Optional[float] = None,
    merge: bool = True,
    max_match_distance_m: float = 2.0,
    **kwargs,
) -> SfincsModel:
    """
    Adds a river levee as a SFINCS weir line, built from a user-supplied polyline geojson.
    Called before mod.write() and the SFINCS run (pre-processing method).

    Used by protect_open strategies alongside a sibling coastal_levee measure,
    each independently raising its own portion of the weir to its own
    elevation. merge=False (the default here matches apply_coastal_levee's
    own default expectation for this use case, even though it's spelled
    merge: false explicitly in every strategy that uses it) does the SAME
    replace-matched/preserve-the-rest merge as apply_coastal_levee -- NOT
    hydromt_sfincs's own plain mod.weirs.create(merge=False), which would
    wholesale replace the ENTIRE current weir (including whatever a
    same-strategy coastal_levee measure already built, if it ran first in
    this strategy's own measures: order) with just this river trace alone.

    Args:
        mod       : (Required) Open SfincsModel object (HydroMT)
        elevation : (Required from measures.yml) Levee crest elevation [m above datum], assigned to every line in `locations`.
        locations : (Required from measures.yml) Path, data source name, or GeoDataFrame with the levee polyline(s).
        par1      : (Optional) Weir discharge coefficient, default 0.6.
        dep       : (Optional) Alternative elevation raster to sample crest height from.
        buffer    : (Optional) Distance (m) from centerline used as sampling window for dep.
        dz        : (Optional) Vertical offset added to the elevation sampled from dep.
        merge     : (Optional) If False (protect_open's own default): keep every
            already-existing weir segment whose geometry ISN'T also present in
            `locations`, and replace only the segments that ARE present in
            `locations` with `locations`'s own new elevation -- see
            src.protection_weir.merge_weir_preserve_unmatched. If True: fall
            back to hydromt_sfincs's own plain concatenation (mod.weirs.
            create(..., merge=True)).
        max_match_distance_m : (Optional) Nearest-neighbour distance used to
            match `locations`'s geometry against the existing weir (see
            merge_weir_preserve_unmatched in protection_weir.py src).

    Returns:
        Modified SfincsModel object.
    """
    if merge:
        mod.weirs.create(
            locations=locations,
            elevation=elevation,
            par1=par1,
            dep=dep,
            buffer=buffer,
            dz=dz,
            merge=True,
        )
        return mod

    new_gdf = mod.data_catalog.get_geodataframe(locations, geom=mod.region).to_crs(
        mod.crs
    )
    new_gdf = new_gdf.explode(index_parts=True).reset_index(drop=True)
    if not new_gdf.geometry.type.isin(["LineString"]).all():
        raise ValueError("Weirs must be of type LineString.")
    new_gdf = new_gdf[["geometry"]].copy()
    new_gdf["elevation"] = elevation
    new_gdf["par1"] = par1

    merged = merge_weir_preserve_unmatched(
        mod.weirs.data, new_gdf, max_match_distance_m=max_match_distance_m
    )
    mod.weirs.set(merged, merge=False)
    mod.config.set("weirfile", "sfincs.weir")

    return mod


# # ── Grey protect-open: storm_surge_barrier ───────────────────────────────────
# def apply_storm_surge_barrier(
#     mod: SfincsModel,
#     locations: Path,
#     stype: str = "gate",  # {'pump', 'culvert', 'valve', 'gate'}
#     discharge: float = 0.0,
#     alpha: float = 0.5,
#     width: float = 1.0,
#     sill_elevation: float = 0.0,
#     manning_n: float = 0.024,
#     zmin: float = 0.0,
#     zmax: float = 0.0,
#     closing_time: float = 600.0,
#     merge: bool = True,
#     **kwargs,
# ) -> SfincsModel:
#     """Adds a closed storm surge barrier as a drainage structure."""
#     mod.drainage_structures.create(
#         locations=locations, stype=stype, discharge=discharge, alpha=alpha,
#         width=width, sill_elevation=sill_elevation, manning_n=manning_n,
#         zmin=zmin, zmax=zmax, closing_time=closing_time, merge=merge,
#     )
#     return mod


# ── Protect-closed: coastal_levee ────────────────────────────────────────────
def apply_coastal_levee(
    mod: SfincsModel,
    locations: Path,
    elevation: Optional[float] = None,
    par1: float = 0.6,
    dep: Optional[str] = None,
    buffer: Optional[float] = None,
    dz: Optional[float] = None,
    merge: bool = True,
    max_match_distance_m: float = 2.0,
    **kwargs,
) -> SfincsModel:
    """
    Adds a coastal levee as a SFINCS weir line, built from a user-supplied polyline geojson.
    Called before mod.write() and the SFINCS run (pre-processing method).

    Used by the protect_closed strategy to seal off the coast/river mouth at
    `elevation` while leaving the river's own already-built protection weir
    (further inland, following current protection standards) untouched.
    `locations` is expected to be built by copying that same baseline weir
    file, deleting its river rows, and adding a new segment closing off the
    river mouth -- see src.protection_weir.merge_weir_preserve_unmatched,
    which this dispatches to when merge=False (below).

    Args:
        mod       : (Required) Open SfincsModel object (HydroMT)
        elevation : (Required from measures.yml) Levee crest elevation [m above datum], assigned to every line in `locations`.
        locations : (Required from measures.yml) Path, data source name, or GeoDataFrame with the levee polyline(s).
        par1      : (Optional) Weir discharge coefficient, default 0.6.
        dep       : (Optional) Alternative elevation raster to sample crest height from.
        buffer    : (Optional) Distance (m) from centerline used as sampling window for dep.
        dz        : (Optional) Vertical offset added to the elevation sampled from dep.
        merge     : (Optional) If False (protect_closed's own default): keep every
            already-existing weir segment whose geometry ISN'T also present in
            `locations` (the untouched river), and replace only the segments
            that ARE present in `locations` with `locations`'s own new
            elevation. If True: fall back to hydromt_sfincs's own plain
            concatenation (mod.weirs.create(..., merge=True)), appending
            `locations` behind whatever weirs already exist with no geometry
            matching at all -- dep/dz-based elevation lookup is also only
            available in this branch, since the merge=False path always
            assigns `elevation` directly.
        max_match_distance_m : (Optional) Nearest-neighbour distance used to
            match `locations`'s geometry against the existing baseline weir
            (see merge_weir_preserve_unmatched in protection_weir.py src).
            Default 2.0 m -- comfortably clears both the baseline weir's own
            0.1 m ASCII round-trip truncation and ordinary editing/
            reprojection noise, while staying well under half a grid cell so
            it can't conflate two genuinely different segments. Too tight
            silently leaves old segments in place instead of replacing them.

    Returns:
        Modified SfincsModel object.
    """
    if merge:
        mod.weirs.create(
            locations=locations,
            elevation=elevation,  # adds absolute elevation
            par1=par1,
            dep=dep,
            buffer=buffer,
            dz=dz,  # Adds weir on top of dem
            merge=True,
        )
        return mod

    new_gdf = mod.data_catalog.get_geodataframe(locations, geom=mod.region).to_crs(
        mod.crs
    )
    new_gdf = new_gdf.explode(index_parts=True).reset_index(drop=True)
    if not new_gdf.geometry.type.isin(["LineString"]).all():
        raise ValueError("Weirs must be of type LineString.")
    new_gdf = new_gdf[["geometry"]].copy()
    new_gdf["elevation"] = elevation
    new_gdf["par1"] = par1

    merged = merge_weir_preserve_unmatched(
        mod.weirs.data, new_gdf, max_match_distance_m=max_match_distance_m
    )
    mod.weirs.set(merged, merge=False)
    mod.config.set("weirfile", "sfincs.weir")

    return mod


# ── Advance / Protect-closed: pumps ──────────────────────────────────────────
def apply_pumps(
    mod: SfincsModel,
    locations: Path,
    stype: str = "pump",  # {'pump', 'culvert', 'valve', 'gate'}
    discharge: float = 0.0,
    alpha: float = 0.5,
    width: float = 1.0,
    sill_elevation: float = 0.0,
    manning_n: float = 0.024,
    zmin: float = 0.0,
    zmax: float = 1.0,
    closing_time: float = 600.0,
    merge: bool = True,
    **kwargs,
) -> SfincsModel:
    """
    Adds a pump structure to the SFINCS model, based on user-supplied locations and parameters.

    Args:
        mod            : (Required) Open SfincsModel object (HydroMT)
        locations      : (Required from measures.yml) Path, data source name, or GeoDataFrame with the pump locations
        discharge      : (Required from measures.yml) Pump discharge capacity [m3/s]
        stype          : (Optional) Pump type (default "pump")
        alpha          : (Optional) Pump efficiency coefficient (default 0.5)
        width          : (Optional) Pump width [m] (default 1.0)
        sill_elevation : (Optional) Pump sill elevation [m above datum] (default 0.0)
        manning_n      : (Optional) Manning's n roughness coefficient (default 0.024)
        zmin           : (Optional) Minimum water level for operation [m] (default 0.0)
        zmax           : (Optional) Maximum water level for operation [m] (default 1.0)
        closing_time   : (Optional) Time [s] it takes for the barrier/pump to close (default 600.0)
        merge          : (Optional) If True, merge with any existing drainage structures instead of overwriting.
    """

    mod.drainage_structures.create(
        locations=locations,
        stype=stype,
        discharge=discharge,
        alpha=alpha,
        width=width,
        sill_elevation=sill_elevation,
        manning_n=manning_n,
        zmin=zmin,
        zmax=zmax,
        closing_time=closing_time,
        merge=merge,
    )

    return mod


# ── Accommodate: dike_ring ────────────────────────────────────────────────────
def apply_dike_ring(
    mod: SfincsModel,
    locations: Path,
    elevation: Optional[float] = None,
    par1: float = 0.6,
    dep: Optional[str] = None,
    buffer: Optional[float] = None,
    dz: Optional[float] = None,
    merge: bool = True,
    **kwargs,
) -> SfincsModel:
    """
    Adds a dike ring as a SFINCS weir line, built from a user-supplied polyline geojson.
    Called before mod.write() and the SFINCS run (pre-processing method).

    Args:
        mod       : (Required) Open SfincsModel object (HydroMT)
        elevation : (Required from measures.yml) Dike crest elevation [m above datum], assigned to the whole line.
        locations : (Required from measures.yml) Path, data source name, or GeoDataFrame with the dike ring polyline(s).
        par1      : (Optional) Weir discharge coefficient, default 0.6.
        dep       : (Optional) Alternative elevation raster to sample crest height from.
        buffer    : (Optional) Distance (m) from centerline used as sampling window for dep.
        dz        : (Optional) Vertical offset added to the elevation sampled from dep.
        merge     : (Optional) If True, merge with any existing weir lines instead of overwriting.

    Returns:
        Modified SfincsModel object.
    """

    mod.weirs.create(
        locations=locations,
        elevation=elevation,
        par1=par1,
        dep=dep,
        buffer=buffer,
        dz=dz,
        merge=merge,
    )

    return mod


# ── Retreat ───────────────────────────────────────────────────────────────────
def apply_retreat(
    mod: SfincsModel,
    urban_code: int = 50,
    target_code: int = 60,
    dep_subgrid: Optional[str] = None,
    landuse_path: Optional[str] = None,
    roughness_native_path: Optional[str] = None,
    lu_roughness_lookup_path: Optional[str] = None,
    locations: Optional[Path] = None,
    retreat_fraction: Optional[float] = None,
    flood_map_path: Optional[str] = None,
    flood_threshold: float = 0.05,
    nr_subgrid_pixels: Optional[int] = None,
    out_path: str = "retreat_landuse.tif",  # written under mod.root; read back by generate_risk_metrics()
    **kwargs,
) -> SfincsModel:
    """
    Managed-retreat measure for the apply_adaptation dispatch.

    BUT roughness is baked into the subgrid table, so this rebuilds the subgrid.
    Rebuilds using the merged dep_subgrid.tif as the
    elevation source, so elevation is reproduced exactly and ONLY roughness changes.

    Roughness is patched directly on this basin's own already-built native-
    resolution roughness raster (roughness_native_path, "manning" convention --
    see 13_build_sfincs_skeleton.py's own subgrid.create() call), NOT via
    HydroMT's own on-the-fly lulc+reclass_table machinery -- this repo's data
    catalog doesn't register any lulc/reclass source that mechanism could use.

    Args:
        mod                      : (Required) Open SfincsModel object
        urban_code               : (Required) Code to remove (built-up = 50)
        target_code              : (Required) Code to assign (60 = bare/sparse vegetation)
        dep_subgrid              : (Required) Path to the basin's own built dep_subgrid.tif
                                   (elevation source, reused unchanged)
        landuse_path             : (Required) Path to this basin's own landuse raster,
                                   used only to determine which cells are urban
        roughness_native_path    : (Required) Path to this basin's own already-built
                                   native-resolution roughness raster (Manning's n)
        lu_roughness_lookup_path : (Required) Path to the same landuse->Manning's n
                                   lookup CSV (copernicus_worldcover/manning_n
                                   columns) 05c_get_roughness.py used to build
                                   roughness_native_path -- used here only to look
                                   up target_code's own Manning's n value
        locations         : (Optional) Polygon path/gdf; if None, all flooded urban cells are eligible
        retreat_fraction  : (Optional) Fraction of flooded urban cells to retreat (0-1),
                            deepest flood depth first; if None, all eligible cells are retreated
        flood_map_path    : (Required) Path to the BASELINE max flood depth raster
                            (e.g. downscaled max_flood_depth.tif from the no-adaptation run);
                            defines which urban cells count as flooded/eligible
        flood_threshold   : (Optional) Depth [m] above which a cell counts as flooded
        nr_subgrid_pixels : (Optional) Subgrid refinement factor. If omitted, it is inferred
                            from dep_subgrid and the coarse model grid resolution.
        out_path          : (Optional) Filename/path the reclassed lulc raster is written to under mod.root
    """

    if dep_subgrid is None:
        raise ValueError("apply_retreat needs dep_subgrid (path to dep_subgrid.tif)")
    if flood_map_path is None:
        raise ValueError(
            "apply_retreat needs flood_map_path (baseline max flood depth raster)"
        )
    if landuse_path is None:
        raise ValueError("apply_retreat needs landuse_path")
    if roughness_native_path is None:
        raise ValueError("apply_retreat needs roughness_native_path")
    if lu_roughness_lookup_path is None:
        raise ValueError("apply_retreat needs lu_roughness_lookup_path")
    if retreat_fraction is not None and not 0.0 <= retreat_fraction <= 1.0:
        raise ValueError(f"retreat_fraction must be in [0, 1], got {retreat_fraction}")

    # 1. land use + baseline flood depth, both on the dep_subgrid grid
    dep = mod.data_catalog.get_rasterdataset(dep_subgrid)
    if isinstance(dep, xr.Dataset):
        dep = dep[list(dep.data_vars)[0]]

    lulc = mod.data_catalog.get_rasterdataset(landuse_path)
    if isinstance(lulc, xr.Dataset):
        lulc = lulc[list(lulc.data_vars)[0]]
    lulc = lulc.raster.reproject_like(dep, method="nearest")

    # baseline flood depth on the same grid
    flood = mod.data_catalog.get_rasterdataset(flood_map_path)
    if isinstance(flood, xr.Dataset):
        flood = flood[list(flood.data_vars)[0]]
    flood = flood.raster.reproject_like(dep, method="nearest")

    # eligible cells: urban AND flooded (optionally constrained to the retreat zone)
    eligible = (lulc == urban_code) & (flood > flood_threshold)
    if locations is not None:
        if isinstance(locations, str):
            locations = gpd.read_file(locations)
        locations = locations.to_crs(dep.rio.crs)
        mask = rasterize(
            [(g, 1) for g in locations.geometry],
            out_shape=(dep.rio.height, dep.rio.width),
            transform=dep.rio.transform(),
            fill=0,
            dtype="uint8",
        ).astype(bool)
        mask = xr.DataArray(mask, dims=dep.dims, coords=dep.coords)
        eligible = eligible & mask

    # apply retreat_fraction: deepest-flooded urban cells retreat first
    if retreat_fraction is not None and retreat_fraction < 1.0:
        depth_vals = flood.values[eligible.values]
        threshold = np.nanquantile(depth_vals, 1.0 - retreat_fraction)
        retreat_mask = eligible & (flood >= threshold)
    else:
        retreat_mask = eligible

    lulc_new = lulc.where(~retreat_mask, target_code)
    lulc_new = lulc_new.astype(lulc.dtype)

    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = Path(mod.root.path) / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lulc_new.rio.to_raster(out_path)

    # 2. rebuild subgrid: SAME elevation (dep_subgrid), roughness patched only at
    # retreated cells. target_code's own Manning's n comes from the SAME lookup
    # table the basin's own roughness raster was originally built from.
    lookup = pd.read_csv(lu_roughness_lookup_path)
    target_rows = lookup.loc[
        lookup["copernicus_worldcover"].astype(int) == int(target_code), "manning_n"
    ]
    if target_rows.empty:
        raise ValueError(
            f"target_code={target_code} not found in {lu_roughness_lookup_path}"
        )
    target_manning_n = float(target_rows.iloc[0])

    da_roughness = mod.data_catalog.get_rasterdataset(roughness_native_path)
    if isinstance(da_roughness, xr.Dataset):
        da_roughness = da_roughness[list(da_roughness.data_vars)[0]]
    # reproject_like on a plain bool array fails: hydromt's own reproject()
    # tries to write dep's inherited nodata (-9999.0) into a bool dtype, which
    # can't represent it (OverflowError). uint8 with an explicit, dtype-safe
    # nodata sentinel (2, distinct from the real 0/1 values) reprojects fine;
    # cells outside retreat_mask's own extent land on that sentinel and are
    # correctly treated as "not retreated" by the == 1 check below.
    retreat_mask_u8 = retreat_mask.astype("uint8").rio.write_nodata(2)
    retreat_mask_native = (
        retreat_mask_u8.raster.reproject_like(da_roughness, method="nearest") == 1
    )
    roughness_new = da_roughness.where(~retreat_mask_native, target_manning_n).astype(
        da_roughness.dtype
    )

    roughness_out_path = Path(mod.root.path) / "retreat_roughness.tif"
    roughness_new.rio.to_raster(roughness_out_path)

    mod.subgrid.create(
        elevation_list=[{"elevation": dep_subgrid}],
        roughness_list=[{"manning": str(roughness_out_path)}],
        nr_subgrid_pixels=(
            _infer_nr_subgrid_pixels(mod, dep)
            if nr_subgrid_pixels is None
            else nr_subgrid_pixels
        ),
        write_man_tif=True,
        write_dep_tif=True,
    )
    return mod


# -----------------------------------------------------------------------------
# ----------------------------- Selected measures -----------------------------
# -----------------------------------------------------------------------------

selected_measures = {
    "offshore_barrier": apply_offshore_barrier,
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
    mod: SfincsModel,
    measure_def: dict,
    flood_map_path: Optional[str] = None,
    method: str = "preprocessing",
    **params,
) -> SfincsModel:
    """
    Entry point called by 18_adapt_apply.py. Validates params against
    measure_def (config/measures.yml), then looks up and calls the matching
    rule function with named params.

    Args:
        measure_type   : key in selected_measures (e.g. "coastal_levee")
        mod            : open SfincsModel object, root already redirected to
                         the adaptation run's own output directory
        measure_def    : this measure's own definition dict from measures.yml
        flood_map_path : baseline (non-adapted) scenario's max flood depth
                         raster -- only consumed by measures that need it
                         (currently just apply_retreat)
        method         : "preprocessing" (only method implemented so far)
        **params       : named parameter values from this strategy's own
                         config/adaptation_strategies.yml entry

    Returns:
        Modified SfincsModel object.
    """
    if measure_type not in selected_measures:
        raise ValueError(
            f"Unknown or unsupported preprocessing measure type '{measure_type}'. "
            f"Available: {list(selected_measures.keys())}"
        )
    validate_measure_params(measure_type, params, method, measure_def)
    return selected_measures[measure_type](mod, flood_map_path=flood_map_path, **params)
