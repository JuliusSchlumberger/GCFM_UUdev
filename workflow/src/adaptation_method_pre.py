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

water_retention's own baseline_excess_volume is computed once per basin x
scenario by rule attribution_mask (18c_attribution_mask.py), via
src.postprocessing.compute_excess_volume, and read here (18a_adapt_pre.py)
from its baseline_excess_volume.json output -- apply_water_retention below
never touches attribution_mask.tif itself.
"""

from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from hydromt_sfincs import SfincsModel
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry.base import BaseMultipartGeometry
from shapely.ops import unary_union

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


def _offshore_buffer(
    locations: gpd.GeoDataFrame,
    distance: float,
    water_mask: xr.DataArray,
    transform,
    out_shape: tuple,
):
    """
    Buffer a reference coastline `distance` metres seaward ONLY, not landward too.

    Matches the proven reference implementation (delta_model's own
    apply_nbs_land_reclamation): for each individual line FEATURE/part in
    `locations`, buffer BOTH sides with shapely's single_sided buffer, and
    keep whichever side has the higher total water fraction underneath.
    Resolved ONCE per line part, via a SINGLE buffer call over the whole
    part -- no chunking, no per-chunk direction estimate.

    Earlier chunked/smoothed variants tried to resolve "which side is
    offshore" separately for short pieces of the line, estimating each
    piece's own direction from its start/end chord. That was solving a
    problem that doesn't need solving: GEOS's buffer offsetting already
    computes a proper continuous offset curve for the WHOLE line in one
    call, correctly hugging one consistent side through a smooth bend --
    confirmed on a real ~50 km coastline curving through roughly a right
    angle, with no chunking at all. The chunked version's short, independent
    per-piece tangent estimates were only an approximation of that, and a
    noisy one: small digitising wiggles flipped individual chunks' resolved
    side, producing flat-cap seams and, after various smoothing attempts to
    paper over that noise, occasional wrongly-flipped segments too.

    This single-buffer-per-part approach isn't infallible: a line that
    doubles back on itself sharply (a genuine hairpin, not just a gradual
    bend) could still end up with part of it on the wrong side. If that
    happens, split the offending line into two separate FEATURES in
    `locations` at the hairpin, one per arm -- each feature gets its own
    independent decision, since this resolves per-part rather than per
    merged geometry.
    """

    def water_frac(poly) -> float:
        if poly.is_empty:
            return 0.0
        mask = rasterize(
            [(poly, 1)],
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)
        return float(water_mask.values[mask].mean()) if mask.any() else 0.0

    # resolve per ORIGINAL feature/part, not the merged whole -- so a
    # `locations` split into multiple features (or already multi-part) gets
    # an independent decision per part
    parts = []
    for geom in locations.geometry:
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, BaseMultipartGeometry):
            parts.extend(g for g in geom.geoms if not g.is_empty)
        else:
            parts.append(geom)

    polys = []
    for line in parts:
        if line.length == 0:
            continue
        side_a = line.buffer(distance, single_sided=True)
        side_b = line.buffer(-distance, single_sided=True)
        polys.append(side_a if water_frac(side_a) >= water_frac(side_b) else side_b)

    if not polys:
        raise ValueError("`locations` contains no usable line geometry to buffer.")
    return unary_union(polys)


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


# NBS Protect-open
# ── NbS advance: vegetated_foreshore ──────────────────────────────────────────
def apply_NbS_land_reclamation(
    mod: SfincsModel,
    distance: float,
    locations: Path,
    target_code: int = 90,
    water_code: int = 80,
    unclassified_code: Optional[int] = 200,
    dep_subgrid: Optional[str] = None,
    landuse_path: Optional[str] = None,
    roughness_native_path: Optional[str] = None,
    lu_roughness_lookup_path: Optional[str] = None,
    manning_n: Optional[float] = None,
    max_depth: Optional[float] = None,
    nr_subgrid_pixels: Optional[int] = None,
    elevation: Optional[float] = None,
    min_elevation: Optional[float] = None,
    max_elevation: Optional[float] = None,
    out_path: str = "foreshore_landuse.tif",
    sea_mask_path: Optional[str] = None,
    sea_mask_out_path: str = "sea_mask.tif",
    **kwargs,
) -> SfincsModel:
    """
    Created vegetated foreshore for the apply_adaptation dispatch.

    Extends the coastline seaward: a reference coastline polyline is buffered
    by `distance`, SEAWARD ONLY (see _offshore_buffer), and every currently-
    water cell in that buffer is raised to `elevation` and reclassed to
    `target_code`. Cells that are already land are left alone. Buffering only
    the offshore side (rather than a plain symmetric buffer) matters because
    a back-barrier lagoon/tidal channel immediately behind the coastline is
    `is_water`-true same as the open sea -- a symmetric buffer would reclaim
    land inside the lagoon too, not just seaward of the barrier.

    BOTH elevation and roughness change here (retreat changes only roughness), so
    both rasters are patched and the subgrid is rebuilt from the patched pair.
    Roughness is patched on this basin's own native-resolution roughness raster,
    same convention as apply_retreat -- NOT via HydroMT's lulc+reclass_table
    machinery, which this repo's data catalog has no source for.

    Wave run-up is deliberately NOT modified: the measure's effect enters through
    bed elevation and bottom friction only. This follows World Bank (2024) NBSOS
    sec. B.2.3.4, which disregards vegetation effects on run-up because they are
    carried through bottom friction.

    Assumes the offshore area is ALREADY inside the active model domain (true for
    a coastal SFINCS domain extending seaward of the coastline). Raises if it is
    not, rather than silently growing the mask and invalidating the boundary
    points, which would need the water level forcing to be rebuilt too.

    Args:
        mod                      : (Required) Open SfincsModel object
        distance                 : (Required from measures.yml) Foreshore width [m], seaward from `locations`
        elevation                : (Optional) Bed elevation [m above datum] of the new foreshore.
                                   If omitted, each cell inherits the elevation of its nearest
                                   land cell, so the foreshore follows the local coastline height.
        min_elevation            : (Optional) Floor on the inherited elevation. Without it, a
                                   stretch fronted by low mudflats/tidal flats -- whose "nearest
                                   land" is itself near or below datum -- produces a reclaimed
                                   foreshore that is still mostly underwater rather than emergent
                                   land. Only applies when `elevation` is omitted.
        max_elevation            : (Optional) Cap on the inherited elevation, e.g. MHW. Without
                                   it, a stretch fronted by dunes or a dike produces a platform
                                   at that height -- a barrier rather than an intertidal foreshore.
        locations                : (Required from measures.yml) Path/gdf with the reference coastline
                                   polyline(s) to buffer by `distance`
        target_code              : (Optional) Land use assigned to the new foreshore.
                                   90 = herbaceous wetland, 95 = mangroves.
                                   60 is BARE ground -- use it only for the attribution run below.
        water_code               : (Optional) Land use code treated as "currently water" (80)
        unclassified_code        : (Optional) Land use code ALSO treated as water when the cell's
                                   own dep is below datum (200 by default). Some basins' landuse
                                   rasters fill large stretches of open sea with this sentinel
                                   instead of water_code, wherever their source data doesn't
                                   extend that far offshore -- without this, those stretches get
                                   an empty foreshore footprint despite clearly being open water.
                                   Pass None to disable and only trust water_code.
        dep_subgrid              : (Required) Path to the basin's own built dep_subgrid.tif
        landuse_path             : (Required) Path to this basin's own landuse raster
        roughness_native_path    : (Required) Path to this basin's own native-resolution
                                   roughness raster (Manning's n)
        lu_roughness_lookup_path : (Required) landuse->Manning's n lookup CSV
                                   (copernicus_worldcover/manning_n columns)
        manning_n                : (Optional) Explicit Manning's n for the new foreshore,
                                   overriding the lookup. Use to apply a literature value
                                   (0.05 wetland, NBSOS Map B.2; 0.04-0.08 saltmarsh IQR,
                                   Arefin et al. 2026) rather than the catalog default.
        max_depth                : (Optional) Maximum water depth [m] in which foreshore may be
                                   created, bounding the footprint to plausible depths
                                   (cf. NBSOS bounding reef NBS to the 3 m isobath).
        nr_subgrid_pixels        : (Optional) Subgrid refinement factor; inferred if omitted
        out_path                 : (Optional) Filename the reclassed lulc raster is written to under mod.root
        sea_mask_path            : (Optional) Path to the basin's own corrected open-sea mask
                                   (zsini_sea_cells_on_grid.tif, coarse/computational-grid
                                   resolution, 1.0=sea). When given, an updated copy is written
                                   to `sea_mask_out_path` with reclaimed coarse cells cleared to
                                   land -- without this, 17_flood_metrics.py's
                                   compute_max_inundation keeps masking flood depth to NaN over
                                   the new land, since it has no other way to know it's no
                                   longer open sea. Skipped entirely when None.
        sea_mask_out_path        : (Optional) Filename the updated sea mask is written to under
                                   mod.root, only used when `sea_mask_path` is given

    Attribution:
        Run twice with identical geometry -- once with target_code=60 (bare) and
        once with target_code=90 -- and difference the results to separate the
        land-raising effect from the vegetation effect. van Zelst et al. (2021),
        Tiggeloven et al. (2022) and Moller et al. (2014) all hold the profile
        fixed and vary only the vegetation for exactly this reason.

    Returns:
        Modified SfincsModel object.
    """
    if dep_subgrid is None:
        raise ValueError(
            "apply_NbS_land_reclamation needs dep_subgrid (path to dep_subgrid.tif)"
        )
    if landuse_path is None:
        raise ValueError("apply_NbS_land_reclamation needs landuse_path")
    if roughness_native_path is None:
        raise ValueError("apply_NbS_land_reclamation needs roughness_native_path")
    if lu_roughness_lookup_path is None and manning_n is None:
        raise ValueError(
            "apply_NbS_land_reclamation needs lu_roughness_lookup_path or manning_n"
        )

    # 1. elevation + land use, both on the dep_subgrid grid
    dep = mod.data_catalog.get_rasterdataset(dep_subgrid)
    if isinstance(dep, xr.Dataset):
        dep = dep[list(dep.data_vars)[0]]

    lulc = mod.data_catalog.get_rasterdataset(landuse_path)
    if isinstance(lulc, xr.Dataset):
        lulc = lulc[list(lulc.data_vars)[0]]
    lulc = lulc.raster.reproject_like(dep, method="nearest")

    # "water" = water_code cells, PLUS unclassified_code cells that are also
    # below datum in dep. This basin's own landuse raster fills large stretches
    # of open sea with unclassified_code (its source data simply doesn't cover
    # that far offshore) rather than water_code -- confirmed those cells sit at
    # a median -7 m in dep (even deeper than confirmed water_code cells), i.e.
    # they're real sea, just missing a landuse label. Without this, whole
    # stretches of coastline end up with an empty foreshore footprint even
    # though there's clearly open water right there.
    is_water = lulc == water_code
    if unclassified_code is not None:
        is_water = is_water | ((lulc == unclassified_code) & (dep < 0))

    # 2. footprint: currently water, inside the SEAWARD-only buffered coastline.
    # `locations` is a reference coastline, buffered by `distance` on the
    # offshore side only (see _offshore_buffer) -- so a back-barrier lagoon/
    # tidal channel immediately behind the coastline isn't reclaimed too, even
    # though it's `is_water`-true same as the open sea.
    if isinstance(locations, (str, Path)):
        locations = gpd.read_file(locations)
    locations = locations.to_crs(dep.rio.crs)
    offshore_geom = _offshore_buffer(
        locations,
        distance,
        water_mask=is_water,
        transform=dep.rio.transform(),
        out_shape=(dep.rio.height, dep.rio.width),
    )
    footprint = rasterize(
        [(offshore_geom, 1)],
        out_shape=(dep.rio.height, dep.rio.width),
        transform=dep.rio.transform(),
        fill=0,
        dtype="uint8",
    ).astype(bool)
    footprint = xr.DataArray(footprint, dims=dep.dims, coords=dep.coords)

    foreshore_mask = footprint & is_water
    if max_depth is not None:
        foreshore_mask = foreshore_mask & (dep >= -abs(max_depth))

    n_cells = int(foreshore_mask.sum())
    if n_cells == 0:
        raise ValueError(
            "Foreshore footprint is empty -- check that `locations` follows the "
            f"coastline, that water_code={water_code} (or unclassified_code="
            f"{unclassified_code} with dep<0) matches {landuse_path}, "
            "and that max_depth is not too restrictive."
        )

    # the new foreshore must already be inside the active domain, or the boundary
    # points and water level forcing would need rebuilding too
    active = (mod.grid.mask == 1) | (mod.grid.mask == 2)
    active_hr = (
        active.astype("uint8")
        .rio.write_nodata(2)
        .raster.reproject_like(dep, method="nearest")
        == 1
    )
    if bool((foreshore_mask & ~active_hr).any()):
        raise ValueError(
            "Foreshore footprint extends outside the active model domain. Extend "
            "the domain seaward at build time (NBSOS uses >=4 km) and rebuild, "
            "rather than growing the mask here."
        )

    # 3. patch elevation and land use
    if elevation is None:
        # Nearest-land-cell elevation, so the new foreshore inherits the local
        # coastline height and varies alongshore. Sampling the nearest *land cell*
        # rather than the polyline itself keeps this robust to a reference line
        # that sits slightly seaward or inland of the true coast.
        dep_nodata = dep.rio.nodata
        land = ~is_water
        if dep_nodata is not None and np.isfinite(dep_nodata):
            land = land & (dep != dep_nodata)
        if not bool(land.any()):
            raise ValueError(
                "No land cells found to inherit an elevation from; "
                "pass `elevation` explicitly."
            )
        _, idx = ndimage.distance_transform_edt(~land.values, return_indices=True)
        z_new = xr.DataArray(
            dep.values[idx[0], idx[1]], dims=dep.dims, coords=dep.coords
        )
        if min_elevation is not None:
            z_new = z_new.clip(min=min_elevation)
        if max_elevation is not None:
            z_new = z_new.clip(max=max_elevation)
        z_vals = z_new.values[foreshore_mask.values]
        z_desc = (
            f"{np.median(z_vals):.2f} m median, coastline-derived "
            f"(range {z_vals.min():.2f} to {z_vals.max():.2f})"
        )
    else:
        z_new = elevation
        z_desc = f"{elevation:.2f} m fixed"

    dep_new = dep.where(~foreshore_mask, z_new).astype(dep.dtype)
    lulc_new = lulc.where(~foreshore_mask, target_code).astype(lulc.dtype)

    dep_out_path = Path(mod.root.path) / "foreshore_dep_subgrid.tif"
    dep_new.rio.to_raster(dep_out_path)

    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = Path(mod.root.path) / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lulc_new.rio.to_raster(out_path)

    # 4. patch roughness on the native-resolution raster (same as apply_retreat)
    if manning_n is None:
        lookup = pd.read_csv(lu_roughness_lookup_path)
        rows = lookup.loc[
            lookup["copernicus_worldcover"].astype(int) == int(target_code), "manning_n"
        ]
        if rows.empty:
            raise ValueError(
                f"target_code={target_code} not found in {lu_roughness_lookup_path}"
            )
        manning_n = float(rows.iloc[0])

    print(
        f"  Foreshore: {distance:.0f} m wide, {n_cells} cells, "
        f"z={z_desc}, lulc={target_code}, n={manning_n}"
    )

    da_roughness = mod.data_catalog.get_rasterdataset(roughness_native_path)
    if isinstance(da_roughness, xr.Dataset):
        da_roughness = da_roughness[list(da_roughness.data_vars)[0]]
    # uint8 + explicit nodata sentinel: a plain bool array fails to reproject
    # (hydromt writes dep's inherited -9999.0 nodata, which bool can't hold)
    foreshore_native = (
        foreshore_mask.astype("uint8")
        .rio.write_nodata(2)
        .raster.reproject_like(da_roughness, method="nearest")
        == 1
    )
    roughness_new = da_roughness.where(~foreshore_native, manning_n).astype(
        da_roughness.dtype
    )

    roughness_out_path = Path(mod.root.path) / "foreshore_roughness.tif"
    roughness_new.rio.to_raster(roughness_out_path)

    # 5. rebuild subgrid from the patched elevation + roughness pair
    mod.subgrid.create(
        elevation_list=[{"elevation": str(dep_out_path)}],
        roughness_list=[{"manning": str(roughness_out_path)}],
        nr_subgrid_pixels=(
            _infer_nr_subgrid_pixels(mod, dep)
            if nr_subgrid_pixels is None
            else nr_subgrid_pixels
        ),
        write_man_tif=True,
        write_dep_tif=True,
    )

    # 6. update the sea mask so downstream flood-metrics/animation stops
    # treating reclaimed cells as open sea. compute_max_inundation
    # (17_flood_metrics.py) masks flood depth to NaN wherever this file
    # reads 1.0 -- it's built once at basin-preprocessing time from the
    # ORIGINAL landuse and never touches this measure's own reclaimed
    # cells, so without this update, flood depth over the new land
    # silently disappears from both the risk metrics and the flood
    # animation, even though the SFINCS run itself simulates it correctly.
    if sea_mask_path is not None:
        sea = mod.data_catalog.get_rasterdataset(sea_mask_path)
        if isinstance(sea, xr.Dataset):
            sea = sea[list(sea.data_vars)[0]]
        # foreshore_mask is at dep_subgrid resolution, sea is at the coarse
        # SFINCS grid resolution -- "max" (not "nearest") so ANY reclaimed
        # subgrid pixel inside a coarse cell marks that whole coarse cell as
        # no-longer-sea, not just whichever single pixel "nearest" happens
        # to land on.
        foreshore_u8 = foreshore_mask.astype("uint8").rio.write_nodata(2)
        reclaimed_coarse = foreshore_u8.raster.reproject_like(sea, method="max") == 1
        sea_new = sea.where(~reclaimed_coarse).rio.write_nodata(np.nan)

        sea_mask_out_path = Path(sea_mask_out_path)
        if not sea_mask_out_path.is_absolute():
            sea_mask_out_path = Path(mod.root.path) / sea_mask_out_path
        sea_new.rio.to_raster(sea_mask_out_path)
        print(
            f"  Updated sea mask: cleared {int(reclaimed_coarse.sum())} coarse "
            f"cells to land at {sea_mask_out_path}"
        )

    return mod


def apply_water_retention(
    mod: SfincsModel,
    locations: Path,
    storage_fraction: float,
    baseline_excess_volume: float,
    max_lowering: Optional[float] = None,
    dep_subgrid: Optional[str] = None,
    roughness_native_path: Optional[str] = None,
    nr_subgrid_pixels: Optional[int] = None,
    flat_floor: bool = False,
    weir_par1: float = 0.6,
    weir_buffer: Optional[float] = None,
    **kwargs,
) -> SfincsModel:
    """
    DEM-lowering retention measure (Room-for-the-River style detention basin).

    Lowers the subgrid elevation within a predefined retention polygon to create
    storage, then rebuilds the subgrid tables so the solver sees the depression.
    ALSO adds a weir tracing the zone's own outer boundary -- without it, the
    boundary between "inside the pit" and its surroundings is just ordinary 2D
    subgrid connectivity at the newly (much lower) excavated elevation, so
    water crosses it exactly as freely as any other cell-to-cell flow and
    drains straight back out during recession, no matter how deep the
    excavation goes -- confirmed on a real run where doubling the excavation
    depth had no effect on this.

    The weir's own crest comes from the basin's own EXISTING protection weir
    (segments within 500m of the zone, averaged) -- its "current" adaptation
    protection standard -- NOT raw terrain: raw terrain can dip to
    near-channel-bed level at a genuine natural low point (confirmed on this
    basin: -8 to -10 m at a channel crossing, vs ~7.6 m mean for the existing
    weir's own crest in that same area), which is essentially no barrier at
    all there. Falls back to terrain-sampling (dep_subgrid) only when no
    existing weir segment is found nearby (e.g. a zone with no existing
    protection infrastructure at all) -- in that fallback case, the zone
    fills by genuinely overtopping whatever natural rim IS there, and any
    genuinely low points on that rim (e.g. a channel crossing the polygon
    boundary) stay realistically open rather than either fully sealed or
    wide open.

    Storage target is derived from a fraction of the UNCONTROLLED baseline
    flood volume, i.e. target_volume = fraction * baseline_excess_volume.
    `baseline_excess_volume` is computed once per basin x scenario by rule
    attribution_mask (18c_attribution_mask.py, via
    src.postprocessing.compute_excess_volume, classes=(1,3,4) -- river,
    compound, spin-up baseline, i.e. everything but pure-coastal, since this
    measure stores RIVER water) and read here from its baseline_excess_volume
    .json output (see 18a_adapt_pre.py) -- reused across every
    storage_fraction scenario so each run is scaled against the same
    reference number.

    NOTE: because this is a physically-based measure, the *nominal*
    target_volume is not guaranteed to be fully realized in the resulting
    flood map -- how much volume the pit actually captures depends on
    hydraulic connectivity, event duration, and whether it overflows.
    After running the simulation, re-run compute_excess_volume() on the
    resulting flood map and compare against the baseline to get the
    *realized* retained fraction, and check the printed lowering/`max_lowering`
    values to confirm the pit wasn't depth-capped before it could hold
    the intended volume.

    Roughness is UNCHANGED by this measure (only elevation is lowered), so it is
    never reclassified or looked up -- this basin's own already-built native-
    resolution roughness raster (roughness_native_path, "manning" convention --
    see 13_build_sfincs_skeleton.py's own subgrid.create() call) is passed
    straight through to the rebuilt subgrid, same convention as apply_retreat
    and apply_NbS_land_reclamation, NOT via HydroMT's own on-the-fly
    lulc+reclass_table machinery -- this repo's data catalog doesn't register
    any lulc/reclass source that mechanism could use.

    Args:
        mod                    : (Required) Open SfincsModel object (HydroMT)
        locations              : (Required from measures.yml) Path / data source / GeoDataFrame of the retention polygon
        storage_fraction       : (Required from measures.yml) Fraction (0-1) of baseline_excess_volume to target as storage
        baseline_excess_volume : (Required) Excess flood volume [m3] from the uncontrolled
                                  baseline run (see src.postprocessing.compute_excess_volume;
                                  auto-injected by 18a_adapt_pre.py, never strategy-configured)
        dep_subgrid            : (Required) Path to the basin's own built dep_subgrid.tif
                                  (elevation source, lowered within the retention zone)
        roughness_native_path  : (Required) Path to this basin's own already-built
                                  native-resolution roughness raster (Manning's n),
                                  reused unchanged since this measure never patches roughness
        max_lowering           : (Optional) Cap on excavation depth [m] (plausibility guard).
        nr_subgrid_pixels      : (Optional) Subgrid refinement factor. If omitted, it is inferred
                                 from dep_subgrid and the coarse model grid resolution.
        flat_floor             : (Optional) If True, set the zone to a flat floor at (min_terrain -
                                 lowering); if False (default), subtract `lowering` from
                                 existing terrain, preserving micro-relief.
        weir_par1              : (Optional) Weir discharge coefficient for the containing
                                 ring, default 0.6 (same default as apply_dike_ring).
        weir_buffer             : (Optional) Sampling window [m] used to read the crest
                                 elevation from the original terrain along the boundary.
                                 None uses hydromt_sfincs' own default.

    Returns:
        Modified SfincsModel object.
    """
    if dep_subgrid is None:
        raise ValueError(
            "apply_water_retention needs dep_subgrid (path to dep_subgrid.tif, or opened xr.DataArray)"
        )
    if roughness_native_path is None:
        raise ValueError("apply_water_retention needs roughness_native_path")
    if not (0.0 <= storage_fraction <= 1.0):
        raise ValueError(f"storage_fraction must be in [0, 1], got {storage_fraction}")
    if baseline_excess_volume <= 0:
        raise ValueError(
            f"baseline_excess_volume must be > 0, got {baseline_excess_volume}"
        )

    target_volume = storage_fraction * baseline_excess_volume

    if isinstance(locations, str):
        locations = gpd.read_file(locations)
    locations = locations.to_crs(mod.crs)
    zone_geom = locations.geometry.union_all()

    # high-res elevation (the already-built dep_subgrid). .load() forces this
    # into memory and closes the underlying file handle — dep_subgrid may be
    # the same file this call later overwrites (write_dep_tif=True below),
    # and on Windows a lazily-opened (dask-backed) raster keeps that file
    # locked, so the overwrite fails with a "Permission denied" /
    # CPLE_AppDefinedError from rasterio.
    dep = mod.data_catalog.get_rasterdataset(dep_subgrid)
    if isinstance(dep, xr.Dataset):
        dep = dep[list(dep.data_vars)[0]]
    dep = dep.load()

    # rasterize the retention zone onto the subgrid grid
    zone_arr = rasterize(
        [(zone_geom, 1)],
        out_shape=(dep.rio.height, dep.rio.width),
        transform=dep.rio.transform(),
        fill=0,
        dtype="uint8",
    ).astype(bool)
    zone_mask = xr.DataArray(zone_arr, dims=dep.dims, coords=dep.coords)

    res = dep.rio.resolution()  # (xres, yres) at SUBGRID resolution
    pixel_area = abs(res[0] * res[1])
    zone_area = float(zone_mask.sum()) * pixel_area
    if zone_area == 0:
        raise ValueError(
            "Retention zone covers no subgrid pixels - check the polygon / CRS."
        )

    # resolve excavation depth from the fraction-derived target volume
    lowering = target_volume / zone_area
    capped = False
    if max_lowering is not None and lowering > max_lowering:
        lowering = max_lowering
        capped = True
    created_storage = lowering * zone_area

    # apply the lowering
    if flat_floor:
        floor = float(dep.where(zone_mask).min()) - lowering
        dep_new = dep.where(~zone_mask, floor).astype(dep.dtype)
    else:
        dep_new = dep.where(~zone_mask, dep - lowering).astype(dep.dtype)

    # contain the pit with a weir along its own outer boundary. Crest comes
    # from the basin's own EXISTING protection weir (its "current" adaptation
    # protection standard) near this zone, NOT raw terrain -- raw terrain can
    # dip to near-channel-bed level at a genuine natural low point (confirmed
    # on this basin: -8 to -10 m at a channel crossing, vs ~7.6 m mean for
    # the existing weir's own crest in that same area), which gives
    # essentially no containment there. Without a real barrier here, the pit
    # is only ever separated from its surroundings by ordinary subgrid
    # connectivity at the excavated elevation, so water crosses it as freely
    # as any other cell-to-cell flow and drains straight back out on
    # recession (see this function's own docstring). merge=True keeps
    # whatever weirs already exist in the model (this same existing
    # protection weir, other chained measures).
    #
    # mod.weirs.data is read BEFORE this call adds the new ring, so it's
    # still exactly the basin's own already-built weir. Only segments within
    # weir_search_buffer of the zone are used, so a distant coastal crest
    # doesn't get averaged into a river floodplain zone's own crest (or vice
    # versa). Falls back to terrain-sampling (dep=dep_subgrid) only if no
    # existing weir segment is found nearby -- e.g. a zone with no existing
    # protection infrastructure at all.
    existing_weir = mod.weirs.data
    weir_search_buffer = zone_geom.buffer(500.0)  # m
    nearby_weir = (
        existing_weir[existing_weir.intersects(weir_search_buffer)]
        if existing_weir is not None and not existing_weir.empty
        else existing_weir
    )

    if nearby_weir is not None and not nearby_weir.empty:
        nearby_zs = [pt[2] for geom in nearby_weir.geometry for pt in geom.coords]
        weir_elevation = float(np.mean(nearby_zs))
        weir_source_kwargs = dict(elevation=weir_elevation, dep=None)
        weir_source_note = f"existing protection standard (~{weir_elevation:.2f} m, {len(nearby_weir)} nearby segment(s))"
    else:
        # dep=dep_subgrid (the ORIGINAL file PATH, not the in-memory `dep`
        # DataArray) -- hydromt_sfincs's own determine_weir_elevation does
        # `dep is None or dep == "dep"` internally, which raises "truth
        # value of an array... is ambiguous" when dep is a DataArray
        # (numpy elementwise comparison instead of a scalar check). A path
        # string sidesteps that entirely and points at the exact same,
        # still-untouched source raster (this run's own mod.root is
        # redirected to the adaptation strategy's own folder, never the
        # skeleton dep_subgrid.tif this path points to).
        weir_source_kwargs = dict(dep=dep_subgrid)
        weir_source_note = "original terrain (no existing protection weir found nearby)"

    zone_boundary = gpd.GeoDataFrame(geometry=[zone_geom.boundary], crs=mod.crs)
    mod.weirs.create(
        locations=zone_boundary,
        buffer=weir_buffer,
        par1=weir_par1,
        merge=True,
        **weir_source_kwargs,
    )

    # roughness is unchanged by this measure, but subgrid.create rebuilds tables
    # from scratch -- feed it this basin's own already-built native-resolution
    # roughness raster straight through, unmodified. dep_new already carries
    # forward whatever river bathymetry burn-in dep_subgrid was built with, so
    # no separate river_list re-burn is needed here (same as apply_retreat and
    # apply_NbS_land_reclamation's own subgrid.create() rebuilds).
    mod.subgrid.create(
        elevation_list=[{"elevation": dep_new}],
        roughness_list=[{"manning": str(roughness_native_path)}],
        nr_subgrid_pixels=(
            _infer_nr_subgrid_pixels(mod, dep)
            if nr_subgrid_pixels is None
            else nr_subgrid_pixels
        ),
        write_dep_tif=True,
        write_man_tif=True,
    )

    cap_note = " [CAPPED by max_lowering, target not fully met]" if capped else ""
    print(
        f"  Applied DEM retention: storage_fraction={storage_fraction:.2f} -> target_volume={target_volume:.0f} m3, "
        f"lowered {lowering:.2f} m over {zone_area / 1e6:.2f} km2 -> "
        f"~{created_storage:.0f} m3 nominal storage{cap_note} "
        f"({'flat floor' if flat_floor else 'subtracted'}), "
        f"contained by a weir along the zone boundary (crest: {weir_source_note}). "
        f"Realized retained volume must be checked post-hoc against the simulated flood map."
    )

    return mod


# ── Water retention (storage-volume / FloodAdapt-style green infrastructure) ──
# def apply_water_retention_greening(
#     mod: SfincsModel,
#     locations: Path,
#     storage_fraction: float,
#     baseline_excess_volume: float,
#     **kwargs,
# ) -> SfincsModel:
#     """
#     Storage-volume retention measure -- FloodAdapt's own mechanism for green
#     infrastructure (water square / greening / total storage are all just UIs
#     over this one thing): a total volume is distributed across the cells
#     covered by the retention polygon(s), and SFINCS's own solver removes
#     water entering those cells until each cell's own share of that capacity
#     is used up.

#     This is a deliberate alternative to apply_water_retention's DEM-lowering
#     + containing-weir approach, not a replacement for it -- confirmed on a
#     real run that this repo's basin data supports both. storage_volume is
#     simpler (no DEM excavation, no weir, no "does the polygon cross the
#     channel" question, no dep_subgrid/roughness passthrough or subgrid
#     rebuild at all -- storage_volume lives on the coarse regular grid, not
#     subgrid, and doesn't touch elevation/roughness), but behaves differently:
#     it's a ONE-WAY SINK. SFINCS's own storage_volume output only ever
#     decreases over a run (confirmed empirically) -- water captured here is
#     never released back, unlike a real basin (or apply_water_retention's own
#     DEM pit), which can also drain as levels recede. Use this when that
#     "absorbs and holds" behaviour is the right representation (e.g. a rain
#     garden / infiltration-style measure); use apply_water_retention when the
#     physical basin-with-outflow behaviour matters.

#     Storage target is derived the same way as apply_water_retention:
#     target_volume = storage_fraction * baseline_excess_volume, sized against
#     the SAME fixed reference (see apply_water_retention's own docstring for
#     where baseline_excess_volume comes from) -- so storage_fraction means the
#     same thing across both pre-processing variants and the post-processing
#     method.

#     Realized fill is reported directly by SFINCS itself (config
#     storestoragevolume=1, set below), as a per-cell-per-timestep
#     'storage_volume' output variable (remaining capacity) -- read that back
#     post-hoc to see exactly how much of target_volume actually got used,
#     instead of re-deriving it from a before/after flood-map comparison.

#     hydromt_sfincs's own storage_volume.create() -> workflows.add_storage_volume()
#     only handles geometry.type == "Polygon" (or "Point") -- a MultiPolygon
#     silently adds ZERO volume anywhere, no error or warning at all (confirmed
#     empirically). So `locations` is exploded into individual Polygon rows
#     first, each carrying its own "volume" attribute (target_volume split
#     proportionally by that piece's own area share) before being handed to
#     storage_volume.create() -- this is what lets a single measure span
#     several disconnected zones (e.g. one compartment per riverbank) correctly.

#     Args:
#         mod                     : (Required) Open SfincsModel object (HydroMT)
#         locations               : (Required from measures.yml) Path / data source /
#                                   GeoDataFrame of the retention polygon(s) -- may be
#                                   a MultiPolygon or several disjoint features
#         storage_fraction        : (Required from measures.yml) Fraction (0-1) of
#                                   baseline_excess_volume to target as storage
#         baseline_excess_volume  : (Required) Excess flood volume [m3] from the
#                                   uncontrolled baseline run (see
#                                   src.postprocessing.compute_excess_volume;
#                                   auto-injected by 18a_adapt_pre.py, never
#                                   strategy-configured)

#     Returns:
#         Modified SfincsModel object.
#     """
#     if not (0.0 <= storage_fraction <= 1.0):
#         raise ValueError(f"storage_fraction must be in [0, 1], got {storage_fraction}")
#     if baseline_excess_volume <= 0:
#         raise ValueError(
#             f"baseline_excess_volume must be > 0, got {baseline_excess_volume}"
#         )

#     target_volume = storage_fraction * baseline_excess_volume

#     if isinstance(locations, str):
#         locations = gpd.read_file(locations)
#     locations = locations.to_crs(mod.crs)

#     # explode MultiPolygon/multi-feature input into individual Polygon rows --
#     # see this function's own docstring for why (hydromt_sfincs's own
#     # add_storage_volume() silently drops anything that isn't exactly
#     # geometry.type == "Polygon") -- and split target_volume across them
#     # proportional to each piece's own area share.
#     locations = locations.explode(index_parts=False).reset_index(drop=True)
#     areas = locations.geometry.area
#     total_area = float(areas.sum())
#     if total_area == 0:
#         raise ValueError(
#             "Retention zone covers no area - check the polygon / CRS."
#         )
#     locations["volume"] = target_volume * (areas / total_area)

#     mod.storage_volume.create(storage_locs=locations, merge=True)
#     # direct realized-fill output -- see this function's own docstring
#     mod.config.set("storestoragevolume", 1)

#     print(
#         f"  Applied storage-volume retention: storage_fraction={storage_fraction:.2f} -> "
#         f"target_volume={target_volume:.0f} m3 distributed across {len(locations)} zone "
#         f"piece(s) ({total_area / 1e6:.2f} km2 total). Realized fill is reported directly "
#         f"by SFINCS's own 'storage_volume' output (storestoragevolume=1) -- read that back "
#         f"post-hoc rather than re-deriving it from a flood-map comparison."
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


# # ── Accommodate: dike_ring aroun location ─────────────────────────────────────────────────────
# def apply_dike_ring(
#     mod: SfincsModel,
#     locations: Path,
#     elevation: Optional[float] = None,
#     par1: float = 0.6,
#     dep: Optional[str] = None,
#     buffer: Optional[float] = None,
#     dz: Optional[float] = None,
#     merge: bool = True,
#     **kwargs,
# ) -> SfincsModel:
#     """
#     Adds a dike ring as a SFINCS weir line, built from a user-supplied polyline geojson.
#     Called before mod.write() and the SFINCS run (pre-processing method).

#     Args:
#         mod       : (Required) Open SfincsModel object (HydroMT)
#         elevation : (Required from measures.yml) Dike crest elevation [m above datum], assigned to the whole line.
#         locations : (Required from measures.yml) Path, data source name, or GeoDataFrame with the dike ring polyline(s).
#         par1      : (Optional) Weir discharge coefficient, default 0.6.
#         dep       : (Optional) Alternative elevation raster to sample crest height from.
#         buffer    : (Optional) Distance (m) from centerline used as sampling window for dep.
#         dz        : (Optional) Vertical offset added to the elevation sampled from dep.
#         merge     : (Optional) If True, merge with any existing weir lines instead of overwriting.

#     Returns:
#         Modified SfincsModel object.
#     """

#     mod.weirs.create(
#         locations=locations,
#         elevation=elevation,
#         par1=par1,
#         dep=dep,
#         buffer=buffer,
#         dz=dz,
#         merge=merge,
#     )

#     return mod


# ── Accommodate: urban_raising ────────────────────────────────────────────────
def apply_urban_raising(
    mod: SfincsModel,
    elevation: float,
    urban_code: int = 50,
    dep_subgrid: Optional[str] = None,
    landuse_path: Optional[str] = None,
    roughness_native_path: Optional[str] = None,
    locations: Optional[Path] = None,
    flood_map_path: Optional[str] = None,
    flood_threshold: float = 0.05,
    nr_subgrid_pixels: Optional[int] = None,
    out_path: str = "urban_raising_dep_subgrid.tif",
    **kwargs,
) -> SfincsModel:
    """
    Accommodate measure: raise the ground elevation at EVERY flooded urban
    cell by a fixed amount, added on top of that cell's own current
    elevation. Replaces the earlier dike_ring (weir-ring around a hand-drawn
    polygon, commented out above) approach for the "accommodate" archetype
    -- a delta-wide, per-cell DEM raise is directly comparable to the other
    delta-wide strategies (advance/protect/retreat), whereas dike_ring's
    protection footprint was scoped to one hand-drawn ring and not
    comparable to those.

    Reuses apply_retreat's own urban+flooded eligibility detection (same
    landuse_path/flood_map_path/flood_threshold/locations semantics) but
    raises elevation at eligible cells instead of reclassifying land use.
    `elevation` is an ADDITIVE raise amount, not an absolute target -- every
    eligible cell goes up by exactly `elevation`, regardless of its own
    current ground level, matching adaptation_method_post.py's own
    apply_urban_raising sibling (which subtracts `elevation` from flood
    DEPTH the same way) so the two methods are directly comparable: for the
    same forcing, raising the bed by `elevation` here produces the same
    residual depth as subtracting `elevation` from depth there.

    Land use/roughness are UNCHANGED (the cell is still urban, only its
    elevation changes), so roughness_native_path is passed straight through
    to the rebuilt subgrid unmodified, same convention as
    apply_water_retention.

    Args:
        mod                    : (Required) Open SfincsModel object
        elevation              : (Required from measures.yml) Amount [m] added to eligible
                                 cells' own current ground elevation (additive raise, not an
                                 absolute target -- mirrors the postprocessing sibling's own
                                 depth subtraction).
        urban_code             : (Optional) Land use code marking urban cells (default 50, built-up)
        dep_subgrid            : (Required) Path to the basin's own built dep_subgrid.tif
        landuse_path           : (Required) Path to this basin's own landuse raster,
                                 used only to determine which cells are urban
        roughness_native_path  : (Required) Path to this basin's own already-built
                                 native-resolution roughness raster (Manning's n),
                                 reused unchanged since this measure never patches roughness
        locations               : (Optional) Polygon path/gdf; if None (the default, and the
                                 usual case for this measure), all flooded urban cells
                                 basin-wide are eligible -- delta-wide, not scoped to a
                                 hand-drawn zone.
        flood_map_path          : (Required) Path to the BASELINE max flood depth raster
                                 (no-adaptation run); defines which urban cells count as
                                 flooded/eligible
        flood_threshold          : (Optional) Depth [m] above which a cell counts as flooded (default 0.05)
        nr_subgrid_pixels       : (Optional) Subgrid refinement factor; inferred if omitted
        out_path                : (Optional) Filename the raised dep_subgrid is written to under mod.root

    Returns:
        Modified SfincsModel object.
    """
    if dep_subgrid is None:
        raise ValueError(
            "apply_urban_raising needs dep_subgrid (path to dep_subgrid.tif)"
        )
    if flood_map_path is None:
        raise ValueError(
            "apply_urban_raising needs flood_map_path (baseline max flood depth raster)"
        )
    if landuse_path is None:
        raise ValueError("apply_urban_raising needs landuse_path")
    if roughness_native_path is None:
        raise ValueError("apply_urban_raising needs roughness_native_path")

    # 1. elevation + land use + baseline flood depth, all on the dep_subgrid grid
    dep = mod.data_catalog.get_rasterdataset(dep_subgrid)
    if isinstance(dep, xr.Dataset):
        dep = dep[list(dep.data_vars)[0]]
    dep = dep.load()

    lulc = mod.data_catalog.get_rasterdataset(landuse_path)
    if isinstance(lulc, xr.Dataset):
        lulc = lulc[list(lulc.data_vars)[0]]
    lulc = lulc.raster.reproject_like(dep, method="nearest")

    flood = mod.data_catalog.get_rasterdataset(flood_map_path)
    if isinstance(flood, xr.Dataset):
        flood = flood[list(flood.data_vars)[0]]
    flood = flood.raster.reproject_like(dep, method="nearest")

    # eligible cells: urban AND flooded (optionally constrained to locations)
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

    # every eligible cell is raised by the SAME additive amount -- no
    # comparison against current elevation, unlike a "raise up to a target"
    # rule (see this function's own docstring for why: this must mirror the
    # postprocessing sibling's flat depth subtraction).
    n_eligible = int(eligible.sum())

    dep_new = dep.where(~eligible, dep + elevation).astype(dep.dtype)

    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = Path(mod.root.path) / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dep_new.rio.to_raster(out_path)

    print(
        f"  Applied urban raising: +{elevation:.2f} m to {n_eligible} flooded urban cell(s)."
    )

    # 2. rebuild subgrid: elevation raised at eligible cells, roughness
    # unchanged (this basin's own already-built native-resolution roughness
    # raster passed straight through, same convention as apply_water_retention)
    mod.subgrid.create(
        elevation_list=[{"elevation": dep_new}],
        roughness_list=[{"manning": str(roughness_native_path)}],
        nr_subgrid_pixels=(
            _infer_nr_subgrid_pixels(mod, dep)
            if nr_subgrid_pixels is None
            else nr_subgrid_pixels
        ),
        write_man_tif=True,
        write_dep_tif=True,
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
    "nbs_land_reclamation": apply_NbS_land_reclamation,
    "water_retention": apply_water_retention,
    # "water_retention_greening": apply_water_retention_greening,
    "coastal_levee": apply_coastal_levee,
    "pumps": apply_pumps,
    # "dike_ring": apply_dike_ring,  # apply_dike_ring is commented out above; replaced by urban_raising
    "urban_raising": apply_urban_raising,
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
