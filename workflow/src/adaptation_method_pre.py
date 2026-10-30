"""Pre-processing adaptation measures: each mutates an open SfincsModel in place.

One function per measure type in config/measures.yml. Called by 17b_adapt_pre.py,
once per measure in a strategy's config/adaptation_strategies.yml entry, via
dispatch_rules(). Measures that rebuild the subgrid all start from the
skeleton's dep_subgrid.tif, passed in by 17b_adapt_pre.py.

water_retention's baseline_excess_volume is computed once per basin x scenario by
rule attribution_mask (src.postprocessing.compute_excess_volume) and read from its
baseline_excess_volume.json (17b_adapt_pre.py).
"""

import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from hydromt_sfincs import SfincsModel
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry.base import BaseMultipartGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree
from src.adaptation_measures import _is_current_protection, validate_measure_params
from src.adaptation_scaling import read_polyline
from src.landuse import LANDUSE_SEA_CODE
from src.protection_weir import _flatten_elevation_from_z, merge_weir_preserve_unmatched
from src.sfincs_restart import patch_restart_zs


def _infer_nr_subgrid_pixels(mod: SfincsModel, dep: xr.DataArray) -> int:
    """Infer the model's subgrid refinement factor from dep_subgrid and the coarse grid."""
    da_mask = mod.grid.mask
    coarse_res_x, coarse_res_y = (
        np.abs(da_mask.raster.res[0]),
        np.abs(da_mask.raster.res[1]),
    )
    dep_res_x, dep_res_y = (
        np.abs(dep.raster.res[0]),
        np.abs(dep.raster.res[1]),
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
    """Buffer each line part of `locations` `distance` metres seaward ONLY.

    Per original feature/part, a single single_sided buffer on each side; the side
    with the higher total water fraction underneath wins (as delta_model's
    apply_nbs_land_reclamation). A sharp hairpin can end up partly on the wrong
    side: split it into two features in `locations`, one per arm.
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

    # Per ORIGINAL feature/part, not the merged whole, so each part decides independently.
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


# Advance: offshore_barrier
def apply_offshore_barrier(
    mod: SfincsModel,
    locations: Path,
    elevation: float,
    par1: float = 0.6,
    dep: str | None = None,
    buffer: float | None = None,
    dz: float | None = None,
    merge: bool = True,
    **kwargs,
) -> SfincsModel:
    """Add an offshore barrier as a SFINCS weir line from a polyline geojson.

    elevation: crest [m above datum] for the whole line. dep/buffer/dz: optional
    raster to sample the crest from, sampling window [m] around the centreline,
    vertical offset. par1: weir discharge coefficient. merge: append to existing weirs.
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


def _levee_crests(
    new_gdf,
    elevation,
    baseline_gdf,
    max_match_distance_m: float,
    raise_by: float | None = None,
):
    """Return the crest for every segment of a new levee trace.

    A number without raise_by: that crest on every segment. A number with raise_by
    (src.adaptation_scaling): each segment gets the crest of the baseline segment it
    lies on + raise_by, so the levee keeps today's profile; segments with no baseline
    levee get the number. current_protection: each segment keeps its baseline crest;
    segments with no baseline levee get fill_crest. "Lies on": midpoint within
    max_match_distance_m, as merge_weir_preserve_unmatched.
    """
    current = _is_current_protection(elevation)
    if not current and raise_by is None:
        return elevation

    if current:
        label, add, fill_crest = (
            "current_protection",
            0.0,
            float(elevation["fill_crest"]),
        )
    else:
        label, add, fill_crest = (
            f"raise by {raise_by:.2f} m",
            float(raise_by),
            float(elevation),
        )
    baseline = (
        _flatten_elevation_from_z(baseline_gdf) if baseline_gdf is not None else None
    )
    if baseline is None or baseline.empty:
        print(f"  {label}: no baseline weir -- every segment gets {fill_crest:.2f} m")
        return np.full(len(new_gdf), fill_crest)

    base_crest = np.array([float(np.max(z)) for z in baseline["elevation"]])
    base_geoms = baseline.geometry.values
    mids = shapely.line_interpolate_point(new_gdf.geometry.values, 0.5, normalized=True)
    nearest = STRtree(base_geoms).nearest(mids)
    matched = shapely.distance(mids, base_geoms[nearest]) <= max_match_distance_m
    crests = np.where(matched, base_crest[nearest] + add, fill_crest)
    print(
        f"  {label}: {int(matched.sum())} of {len(new_gdf)} segments on a baseline levee "
        f"-> crest {crests[matched].min() if matched.any() else float('nan'):.2f}-"
        f"{crests[matched].max() if matched.any() else float('nan'):.2f} m, "
        f"{int((~matched).sum())} without a baseline levee get {fill_crest:.2f} m"
    )
    return crests


# Grey protect-open: river_levee
def apply_river_levee(
    mod: SfincsModel,
    locations: Path,
    elevation: float,
    par1: float = 0.6,
    dep: str | None = None,
    buffer: float | None = None,
    dz: float | None = None,
    merge: bool = True,
    max_match_distance_m: float = 2.0,
    raise_by: float | None = None,
    **kwargs,
) -> SfincsModel:
    """Add a river levee as a SFINCS weir line from a polyline geojson.

    Used by protect_open strategies alongside a sibling coastal_levee measure,
    each independently raising its own portion of the weir to its own
    elevation. The signature default is merge=True; every strategy that uses
    this measure sets merge: false, which does the SAME
    replace-matched/preserve-the-rest merge as apply_coastal_levee -- NOT
    hydromt_sfincs's own plain mod.weirs.create(merge=False), which would
    wholesale replace the ENTIRE current weir (including whatever a
    same-strategy coastal_levee measure already built, if it ran first in
    this strategy's own measures: order) with just this river trace alone.

    elevation: crest [m above datum] for every line, or current_protection's
    {"current": True, "fill_crest": x} (merge=False only; see _levee_crests).
    merge=True: hydromt_sfincs's plain concatenation (mod.weirs.create(merge=True)).
    max_match_distance_m: matching distance against the existing weir
    (src.protection_weir.merge_weir_preserve_unmatched).
    raise_by (merge=False only; added by src.adaptation_scaling): raise every segment
    on an existing levee by this much instead of setting it to `elevation`
    (_levee_crests).
    par1, dep, buffer, dz: as in apply_offshore_barrier.
    """
    if merge:
        if _is_current_protection(elevation) or raise_by is not None:
            raise ValueError(
                "river_levee: current_protection / raise_by need merge: false"
            )
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

    new_gdf = read_polyline(
        mod.data_catalog.get_geodataframe(locations, geom=mod.region), mod.crs
    )
    new_gdf["elevation"] = _levee_crests(
        new_gdf, elevation, mod.weirs.data, max_match_distance_m, raise_by
    )
    new_gdf["par1"] = par1

    merged = merge_weir_preserve_unmatched(
        mod.weirs.data, new_gdf, max_match_distance_m=max_match_distance_m
    )
    mod.weirs.set(merged, merge=False)
    mod.config.set("weirfile", "sfincs.weir")

    return mod


# # Grey protect-open: storm_surge_barrier
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


# NbS Protect-open: Land reclamation
def apply_NbS_land_reclamation(
    mod: SfincsModel,
    distance: float,
    locations: Path,
    target_code: int = 90,
    water_code: int = 80,
    unclassified_code: int | None = LANDUSE_SEA_CODE,
    dep_subgrid: str | None = None,
    landuse_path: str | None = None,
    roughness_native_path: str | None = None,
    lu_roughness_lookup_path: str | None = None,
    manning_n: float | None = None,
    max_depth: float | None = None,
    nr_subgrid_pixels: int | None = None,
    elevation: float | None = None,
    min_elevation: float | None = None,
    max_elevation: float | None = None,
    out_path: str = "foreshore_landuse.tif",
    sea_mask_path: str | None = None,
    sea_mask_out_path: str = "sea_mask.tif",
    **kwargs,
) -> SfincsModel:
    """Build a vegetated foreshore seaward of a reference coastline.

    `locations` is buffered by `distance` [m] SEAWARD ONLY (see _offshore_buffer),
    so a back-barrier lagoon behind the coastline is not reclaimed; every water
    cell in the buffer is raised to `elevation` and reclassed to `target_code`,
    land cells are left alone. Elevation and roughness both change: roughness is
    patched on the native-resolution roughness raster (as apply_retreat; the data
    catalog has no lulc/reclass source) and the subgrid is rebuilt.

    Wave run-up is deliberately NOT modified: the effect enters through bed
    elevation and bottom friction only, following World Bank (2024) NBSOS
    sec. B.2.3.4. Raises if the foreshore is not already inside the active model
    domain (growing the mask would invalidate boundary points and forcing).

    elevation: bed elevation [m above datum]; if omitted, each cell inherits its
        nearest land cell's elevation, bounded by min_elevation (else low mudflats
        give a mostly submerged foreshore) and max_elevation, e.g. MHW (else
        dunes or a dike give a barrier, not an intertidal foreshore).
    target_code: 90 = herbaceous wetland, 95 = mangroves; 60 = BARE (attribution only).
    water_code: land use treated as water (80). unclassified_code: also water where
        dep is below datum (200; some landuse rasters fill open sea with it);
        None disables.
    lu_roughness_lookup_path: landuse->Manning's n CSV (copernicus_worldcover/manning_n).
    manning_n: overrides the lookup (0.05 wetland, NBSOS Map B.2; 0.04-0.08
        saltmarsh IQR, Arefin et al. 2026).
    max_depth: max water depth [m] for new foreshore (cf. NBSOS bounding reef NBS
        to the 3 m isobath).
    sea_mask_path: zsini_sea_cells_on_grid.tif (coarse grid, 1.0=sea); when given,
        a copy with reclaimed coarse cells cleared to land is written to
        sea_mask_out_path under mod.root, else compute_max_inundation keeps masking
        flood depth to NaN over the new land.
    out_path: reclassed lulc raster, under mod.root.

    Attribution: run twice with identical geometry, target_code=60 (bare) and 90,
    and difference the results to separate land raising from vegetation; van Zelst
    et al. (2021), Tiggeloven et al. (2022) and Moller et al. (2014) hold the
    profile fixed and vary only the vegetation for this reason.
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

    dep = mod.data_catalog.get_rasterdataset(dep_subgrid)
    if isinstance(dep, xr.Dataset):
        dep = dep[list(dep.data_vars)[0]]

    lulc = mod.data_catalog.get_rasterdataset(landuse_path)
    if isinstance(lulc, xr.Dataset):
        lulc = lulc[list(lulc.data_vars)[0]]
    lulc = lulc.raster.reproject_like(dep, method="nearest")

    # unclassified_code cells below datum are open sea missing a landuse label
    # (median -7 m in dep, deeper than water_code cells).
    is_water = lulc == water_code
    if unclassified_code is not None:
        is_water = is_water | ((lulc == unclassified_code) & (dep < 0))

    if isinstance(locations, (str, Path)):
        locations = gpd.read_file(locations)
    locations = locations.to_crs(dep.rio.crs)
    offshore_geom = _offshore_buffer(
        locations,
        distance,
        water_mask=is_water,
        transform=dep.raster.transform,
        out_shape=(dep.rio.height, dep.rio.width),
    )
    footprint = rasterize(
        [(offshore_geom, 1)],
        out_shape=(dep.rio.height, dep.rio.width),
        transform=dep.raster.transform,
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

    if elevation is None:
        # Nearest land CELL, not the polyline: robust to a reference line slightly
        # seaward or inland of the true coast; the foreshore varies alongshore.
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

    # compute_max_inundation masks flood depth to NaN where the sea mask reads 1.0;
    # the mask is built from the ORIGINAL landuse, so clear the reclaimed cells.
    if sea_mask_path is not None:
        sea = mod.data_catalog.get_rasterdataset(sea_mask_path)
        if isinstance(sea, xr.Dataset):
            sea = sea[list(sea.data_vars)[0]]
        # "max", not "nearest": ANY reclaimed subgrid pixel in a coarse cell marks
        # the whole coarse cell as no longer sea.
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
    max_lowering: float | None = None,
    dep_subgrid: str | None = None,
    roughness_native_path: str | None = None,
    nr_subgrid_pixels: int | None = None,
    flat_floor: bool = False,
    weir_par1: float = 0.6,
    weir_buffer: float | None = None,
    rst_path: str | None = None,
    ind_path: str | None = None,
    out_rst_path: str | None = None,
    inlet: Path | None = None,
    inlet_sill_below_crest: float = 0.2,
    **kwargs,
) -> SfincsModel:
    """Lower the subgrid DEM in a retention polygon and ring it with a weir.

    Room-for-the-River style detention basin. Without the ring, the pit joins its
    surroundings by ordinary subgrid connectivity and drains straight back out on
    recession, whatever the excavation depth. The ring crest is the mean crest of
    existing protection weir segments within 500 m of the zone, NOT raw terrain
    (which can dip to channel-bed level: -8 to -10 m at a channel crossing vs
    ~7.6 m weir crest); terrain sampling (dep_subgrid) only when no weir is near.

    target_volume = storage_fraction * baseline_excess_volume [m3] of the
    UNCONTROLLED baseline (classes=(1,3), river and compound: this measure stores
    river water), auto-injected by 17b_adapt_pre.py. The nominal target is not
    guaranteed to be realized; rerun compute_excess_volume() on the resulting flood
    map and check the printed lowering/`max_lowering`. Roughness is unchanged:
    roughness_native_path ("manning" convention) is passed through.

    max_lowering: cap on excavation depth [m]. flat_floor: True sets a flat floor at
    (min_terrain - lowering); False subtracts `lowering`, keeping micro-relief.
    weir_buffer: crest sampling window [m] (None: hydromt_sfincs default).
    rst_path: spin-up restart, computed on the un-excavated terrain; when given, a
    copy with zs shifted down by the bed lowering per cell (dry stays dry, wet
    keeps its relative depth) is written to out_rst_path (default mod.root /
    rst_path's filename), so the zone is not pre-filled at t=0. Requires ind_path
    (sfincs.ind). None skips it (e.g. spin-up disabled).
    inlet: polyline(s) along the existing levee where the zone takes in river water;
    those levee segments get their own crest - inlet_sill_below_crest [m], and the
    ring weir is left out wherever the zone runs along an existing levee. None: ring
    weir all around, no inlet.
    """
    if dep_subgrid is None:
        raise ValueError(
            "apply_water_retention needs dep_subgrid (path to dep_subgrid.tif, or opened xr.DataArray)"
        )
    if roughness_native_path is None:
        raise ValueError("apply_water_retention needs roughness_native_path")
    if not (0.0 <= storage_fraction <= 1.0):
        raise ValueError(f"storage_fraction must be in [0, 1], got {storage_fraction}")
    if baseline_excess_volume < 0:
        raise ValueError(
            f"baseline_excess_volume must be >= 0, got {baseline_excess_volume}"
        )
    if baseline_excess_volume == 0:
        # No river/compound flooding (e.g. coastal-only): nothing to size the pit
        # against, and the ring alone would keep coastal water out. 17b_adapt_pre.py
        # skips the measure earlier; this keeps a direct call safe, including the
        # restart copy callers expect at out_rst_path.
        print(
            "  WARNING: baseline_excess_volume is 0 (no river/compound flooding in this "
            "scenario) -- water retention has nothing to store, model left unchanged."
        )
        if rst_path is not None:
            dst = (
                Path(out_rst_path)
                if out_rst_path is not None
                else Path(mod.root.path) / Path(rst_path).name
            )
            if Path(rst_path).resolve() != dst.resolve():
                shutil.copy2(rst_path, dst)
        return mod

    target_volume = storage_fraction * baseline_excess_volume

    if isinstance(locations, str):
        locations = gpd.read_file(locations)
    locations = locations.to_crs(mod.crs)
    zone_geom = locations.geometry.union_all()

    # .load() releases the file handle: dep_subgrid may be overwritten below
    # (write_dep_tif=True), and on Windows a lazily opened raster keeps it locked
    # ("Permission denied" / CPLE_AppDefinedError).
    dep = mod.data_catalog.get_rasterdataset(dep_subgrid)
    if isinstance(dep, xr.Dataset):
        dep = dep[list(dep.data_vars)[0]]
    dep = dep.load()

    zone_arr = rasterize(
        [(zone_geom, 1)],
        out_shape=(dep.rio.height, dep.rio.width),
        transform=dep.raster.transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    zone_mask = xr.DataArray(zone_arr, dims=dep.dims, coords=dep.coords)

    res = dep.raster.res  # (xres, yres) at SUBGRID resolution
    pixel_area = abs(res[0] * res[1])
    zone_area = float(zone_mask.sum()) * pixel_area
    if zone_area == 0:
        raise ValueError(
            "Retention zone covers no subgrid pixels - check the polygon / CRS."
        )

    lowering = target_volume / zone_area
    capped = False
    if max_lowering is not None and lowering > max_lowering:
        lowering = max_lowering
        capped = True
    created_storage = lowering * zone_area

    if flat_floor:
        floor = float(dep.where(zone_mask).min()) - lowering
        dep_new = dep.where(~zone_mask, floor).astype(dep.dtype)
    else:
        dep_new = dep.where(~zone_mask, dep - lowering).astype(dep.dtype)

    # Shift zs by the change in each coarse cell's bed reference, so the pit keeps
    # its pre-excavation relative depth instead of being pre-filled at t=0.
    # hydromt_sfincs takes the bed reference as the MINIMUM fine pixel elevation in
    # a cell (subgrid.py subgrid_v_table: `return z, V, elevation.min(), z.max()`),
    # so use zmin(dep_new) - zmin(dep) per cell: "average" under-shifts partly
    # lowered boundary cells, and min-reprojecting the difference misses cells
    # whose original lowest pixel lies outside the zone.
    if rst_path is not None:
        if ind_path is None:
            raise ValueError(
                "apply_water_retention needs ind_path (this basin's own sfincs.ind) "
                "to patch the restart file's initial water level"
            )
        coarse = mod.grid.mask
        zmin_before = dep.raster.reproject_like(coarse, method="min").values
        zmin_after = dep_new.raster.reproject_like(coarse, method="min").values
        dz_coarse = zmin_after - zmin_before
        dz_coarse = np.where(np.isfinite(dz_coarse), dz_coarse, 0.0)

        zone_touch_fine = zone_mask.astype("uint8").rio.write_nodata(2)
        zone_touch_coarse = (
            zone_touch_fine.raster.reproject_like(coarse, method="max").values == 1
        )

        out_rst_path_resolved = (
            Path(out_rst_path)
            if out_rst_path is not None
            else Path(mod.root.path) / Path(rst_path).name
        )
        n_patched = patch_restart_zs(
            src_rst_path=Path(rst_path),
            dst_rst_path=out_rst_path_resolved,
            ind_path=Path(ind_path),
            grid_shape=coarse.shape,
            active_cell_mask=zone_touch_coarse,
            dz=dz_coarse,
        )
        print(
            f"  Patched restart initial water level: {n_patched} active coarse "
            f"cell(s) shifted to match the excavated bed -> {out_rst_path_resolved}"
        )

    # Ring crest from existing protection weir segments within weir_search_buffer
    # (so a distant coastal crest is not averaged into a river zone's crest), not
    # raw terrain (see docstring); terrain sampling only if none is nearby.
    # mod.weirs.data still holds only the pre-existing weir here; merge=True keeps it.
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
        # Pass the PATH, not the DataArray: hydromt_sfincs's determine_weir_elevation
        # does `dep == "dep"`, which raises "truth value ... ambiguous" on an array.
        # The path points at the untouched skeleton dep_subgrid.tif.
        weir_source_kwargs = dict(dep=dep_subgrid)
        weir_source_note = "original terrain (no existing protection weir found nearby)"

    ring = zone_geom.boundary
    if inlet is not None and nearby_weir is not None and not nearby_weir.empty:
        # Where the zone runs along an existing levee, that levee is its wall: a
        # second weir behind it would hold back the water the inlet lets in.
        ring = ring.difference(nearby_weir.buffer(150.0).union_all())
    zone_boundary = gpd.GeoDataFrame(geometry=[ring], crs=mod.crs)
    mod.weirs.create(
        locations=zone_boundary,
        buffer=weir_buffer,
        par1=weir_par1,
        merge=True,
        **weir_source_kwargs,
    )

    if inlet is not None:
        print(
            f"  Retention inlet: levee segments on {Path(str(inlet)).name} lowered by "
            f"{inlet_sill_below_crest:.2f} m"
        )
        apply_river_levee(
            mod,
            locations=inlet,
            elevation=weir_source_kwargs.get("elevation", 0.0) - inlet_sill_below_crest,
            par1=weir_par1,
            merge=False,
            raise_by=-inlet_sill_below_crest,
        )

    # subgrid.create rebuilds the tables from scratch: pass the native roughness
    # raster through unchanged. dep_new already carries the river bathymetry
    # burn-in, so no river_list re-burn is needed.
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


#  Protect-closed: coastal_levee
def apply_coastal_levee(
    mod: SfincsModel,
    locations: Path,
    elevation: float | None = None,
    par1: float = 0.6,
    dep: str | None = None,
    buffer: float | None = None,
    dz: float | None = None,
    merge: bool = True,
    max_match_distance_m: float = 2.0,
    raise_by: float | None = None,
    **kwargs,
) -> SfincsModel:
    """Add a coastal levee as a SFINCS weir line from a polyline geojson.

    Used by protect_closed to seal the coast/river mouth at `elevation`, leaving the
    river's inland protection weir untouched. `locations` is the baseline weir with
    its river rows deleted plus a segment closing the river mouth.
    merge=False: replace only existing segments present in `locations`, keep the
    rest (src.protection_weir.merge_weir_preserve_unmatched). merge=True:
    hydromt_sfincs's plain concatenation; only this branch supports dep/dz crests.
    elevation: crest [m above datum], or current_protection's
    {"current": True, "fill_crest": x} (merge=False only; see _levee_crests).
    max_match_distance_m: 2.0 m clears the baseline weir's 1 cm crest rounding
    (src.sfincs_run.write_weir_file) and editing/reprojection noise while staying well under half a grid
    cell; too tight silently leaves old segments in place.
    raise_by: as in apply_river_levee.
    """
    if merge:
        if _is_current_protection(elevation) or raise_by is not None:
            raise ValueError(
                "coastal_levee: current_protection / raise_by need merge: false"
            )
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

    new_gdf = read_polyline(
        mod.data_catalog.get_geodataframe(locations, geom=mod.region), mod.crs
    )
    new_gdf["elevation"] = _levee_crests(
        new_gdf, elevation, mod.weirs.data, max_match_distance_m, raise_by
    )
    new_gdf["par1"] = par1

    merged = merge_weir_preserve_unmatched(
        mod.weirs.data, new_gdf, max_match_distance_m=max_match_distance_m
    )
    mod.weirs.set(merged, merge=False)
    mod.config.set("weirfile", "sfincs.weir")

    return mod


# Advance / Protect-closed: pumps
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
    """Add pump drainage structures at `locations`.

    discharge: the measure's TOTAL capacity [m3/s], split evenly over the pump lines
    (SFINCS gives every sfincs.drn line its own discharge); the postprocessing pumps
    rule uses the same total. sill_elevation: [m above datum]. zmin/zmax: water
    levels [m] for operation. closing_time: [s].
    """
    n_lines = len(
        read_polyline(
            mod.data_catalog.get_geodataframe(locations, geom=mod.region), mod.crs
        )
    )
    print(
        f"  Pumps: total capacity {discharge:.1f} m3/s over {n_lines} line(s) = "
        f"{discharge / n_lines:.1f} m3/s each"
    )

    mod.drainage_structures.create(
        locations=locations,
        stype=stype,
        discharge=discharge / n_lines,
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


# #  Accommodate: dike_ring aroun location
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


#  Accommodate: urban_raising
def apply_urban_raising(
    mod: SfincsModel,
    raise_fraction: float | None = None,
    freeboard: float = 0.0,
    urban_code: int = 50,
    dep_subgrid: str | None = None,
    landuse_path: str | None = None,
    roughness_native_path: str | None = None,
    locations: Path | None = None,
    flood_map_path: str | None = None,
    flood_threshold: float = 0.05,
    nr_subgrid_pixels: int | None = None,
    out_path: str = "urban_raising_dep_subgrid.tif",
    **kwargs,
) -> SfincsModel:
    """Raise flooded urban cells by their own baseline flood depth (+ freeboard [m]).

    A raised cell sits at its baseline max water level. Eligibility (urban_code
    cells flooded above flood_threshold [m] in the BASELINE, no-adaptation
    flood_map_path, optionally inside `locations`) and raise_fraction selection
    (deepest-flooded first) match apply_retreat; adaptation_method_post.py's
    apply_urban_raising removes the depth at the same cells. Land use and roughness
    are unchanged (roughness_native_path passed through). Delta-wide by default
    (locations=None), so it is comparable to the other delta-wide strategies.
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
    if raise_fraction is not None and not 0.0 <= raise_fraction <= 1.0:
        raise ValueError(f"raise_fraction must be in [0, 1], got {raise_fraction}")

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

    eligible = (lulc == urban_code) & (flood > flood_threshold)
    if locations is not None:
        if isinstance(locations, str):
            locations = gpd.read_file(locations)
        locations = locations.to_crs(dep.rio.crs)
        mask = rasterize(
            [(g, 1) for g in locations.geometry],
            out_shape=(dep.rio.height, dep.rio.width),
            transform=dep.raster.transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)
        mask = xr.DataArray(mask, dims=dep.dims, coords=dep.coords)
        eligible = eligible & mask

    # deepest-flooded first, as apply_retreat's retreat_fraction
    if raise_fraction is not None and raise_fraction < 1.0 and bool(eligible.any()):
        depth_vals = flood.values[eligible.values]
        threshold = np.nanquantile(depth_vals, 1.0 - raise_fraction)
        raise_mask = eligible & (flood >= threshold)
    else:
        raise_mask = eligible

    dep_new = dep.where(~raise_mask, dep + flood + freeboard).astype(dep.dtype)

    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = Path(mod.root.path) / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dep_new.rio.to_raster(out_path)

    raised_by = flood.values[raise_mask.values] + freeboard
    print(
        f"  Applied urban raising: {int(raise_mask.sum())} of {int(eligible.sum())} flooded "
        f"urban cell(s) raised by their own flood depth + {freeboard:.2f} m freeboard "
        f"(mean +{np.nanmean(raised_by) if raised_by.size else 0.0:.2f} m, "
        f"max +{np.nanmax(raised_by) if raised_by.size else 0.0:.2f} m)."
    )

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


#  Retreat
def apply_retreat(
    mod: SfincsModel,
    urban_code: int = 50,
    target_code: int = 60,
    dep_subgrid: str | None = None,
    landuse_path: str | None = None,
    roughness_native_path: str | None = None,
    lu_roughness_lookup_path: str | None = None,
    locations: Path | None = None,
    retreat_fraction: float | None = None,
    flood_map_path: str | None = None,
    flood_threshold: float = 0.05,
    nr_subgrid_pixels: int | None = None,
    out_path: str = "retreat_landuse.tif",  # written under mod.root; read by rule adapt_flood_metrics_pre
    **kwargs,
) -> SfincsModel:
    """Convert flooded urban cells to `target_code` land use (managed retreat).

    Roughness is baked into the subgrid table, so the subgrid is rebuilt from the
    unchanged dep_subgrid.tif: elevation is reproduced exactly and ONLY roughness
    changes. Roughness is patched on the native-resolution roughness raster
    ("manning" convention), not via HydroMT lulc+reclass_table (the data catalog
    registers no such source).
    urban_code: 50 = built-up. target_code: 60 = bare/sparse vegetation; its
    Manning's n comes from lu_roughness_lookup_path (the CSV 09_land_surface.py used).
    Eligible: urban cells flooded above flood_threshold [m] in the BASELINE
    flood_map_path, optionally inside `locations`; retreat_fraction picks the
    deepest-flooded first (None: all). out_path: reclassed lulc raster under mod.root.
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

    dep = mod.data_catalog.get_rasterdataset(dep_subgrid)
    if isinstance(dep, xr.Dataset):
        dep = dep[list(dep.data_vars)[0]]

    lulc = mod.data_catalog.get_rasterdataset(landuse_path)
    if isinstance(lulc, xr.Dataset):
        lulc = lulc[list(lulc.data_vars)[0]]
    lulc = lulc.raster.reproject_like(dep, method="nearest")

    flood = mod.data_catalog.get_rasterdataset(flood_map_path)
    if isinstance(flood, xr.Dataset):
        flood = flood[list(flood.data_vars)[0]]
    flood = flood.raster.reproject_like(dep, method="nearest")

    eligible = (lulc == urban_code) & (flood > flood_threshold)
    if locations is not None:
        if isinstance(locations, str):
            locations = gpd.read_file(locations)
        locations = locations.to_crs(dep.rio.crs)
        mask = rasterize(
            [(g, 1) for g in locations.geometry],
            out_shape=(dep.rio.height, dep.rio.width),
            transform=dep.raster.transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)
        mask = xr.DataArray(mask, dims=dep.dims, coords=dep.coords)
        eligible = eligible & mask

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

    # target_code's Manning's n comes from the lookup the roughness raster was built from.
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
    # A bool array fails to reproject (hydromt writes dep's -9999.0 nodata:
    # OverflowError). uint8 with nodata sentinel 2 works; cells outside the mask's
    # extent read as "not retreated" in the == 1 check below.
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


selected_measures = {
    "offshore_barrier": apply_offshore_barrier,
    "river_levee": apply_river_levee,
    "nbs_land_reclamation": apply_NbS_land_reclamation,
    "water_retention": apply_water_retention,
    "coastal_levee": apply_coastal_levee,
    "pumps": apply_pumps,
    # "dike_ring": apply_dike_ring,
    "urban_raising": apply_urban_raising,
    "retreat": apply_retreat,
}


def dispatch_rules(
    measure_type: str,
    mod: SfincsModel,
    measure_def: dict,
    flood_map_path: str | None = None,
    method: str = "preprocessing",
    **params,
) -> SfincsModel:
    """Validate params and call the measure function for `measure_type`.

    Entry point called by 17b_adapt_pre.py. Validates params against
    measure_def (config/measures.yml), then looks up and calls the matching
    rule function with named params. `mod`'s root is already redirected to the
    adaptation run's output directory.
    flood_map_path : baseline (non-adapted) scenario's max flood depth
                     raster -- only consumed by measures that need it
                     (currently just apply_retreat)
    """
    if measure_type not in selected_measures:
        raise ValueError(
            f"Unknown or unsupported preprocessing measure type '{measure_type}'. "
            f"Available: {list(selected_measures.keys())}"
        )
    validate_measure_params(measure_type, params, method, measure_def)
    return selected_measures[measure_type](mod, flood_map_path=flood_map_path, **params)
