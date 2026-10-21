"""Build the coastal/riverbank protection weir on the SFINCS regular grid.

The weir is the protection standard itself: defences run continuously along
the whole coast and river network, hugging it with no inland offset, so
nothing floods below the crest until overtopped. Built only by rule
modelled_depth_estimation (10); rule 13 imports the final gpkg as-is.
Confinement rounds use a very high crest (e.g. 1000 m) and an all-NaN
water_side_crest_on_grid, confining coast and river alike (the confined
river can exceed a low coastal crest near the mouth); the production weir
uses the confined round's period-max water level per edge, floored at the
datum-corrected coastal crest.

1. Ocean (landuse==200) and river-channel cells form ``water_like``; every
   other valid cell is land. 80 (inland water/tidal flat) / 90 (wetland) are
   land unless in the channel mask or, with unprotected_ocean_wetlands, in a
   patch edge-connected to the open sea.
2. Small connected components on both sides (< min_component_cells) get no
   ring: islands flood normally, ponds/tidal pools are not walled off.
3. The weir is traced on the grid's own cell edges, so it aligns with the
   flux links by construction (no seepage gaps).
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.ndimage import label as ndimage_label
from shapely.geometry import LineString, Polygon
from shapely.ops import linemerge, unary_union
from shapely.strtree import STRtree
from src.grid import cell_size_m
from src.landuse import LANDUSE_SEA_CODE, OCEAN_WETLAND_CLASSES
from src.surge import ceil_crest

log = logging.getLogger(__name__)


class GridArrays:
    """Hold a SFINCS regular grid's loaded arrays (mid-build or read-mode model)."""

    def __init__(
        self,
        crs,
        bed_elevation_m: np.ndarray,
        valid_mask: np.ndarray,
        cell_size_m,
        transform,
    ):
        self.crs = crs
        self.bed_elevation_m = bed_elevation_m
        self.valid_mask = valid_mask
        self.cell_size_m = cell_size_m
        self.transform = transform
        self.shape = bed_elevation_m.shape
        self.n_cells = bed_elevation_m.size
        self._struct4 = generate_binary_structure(2, 1)
        self._struct8 = generate_binary_structure(2, 2)

    @classmethod
    def from_regular(cls, dep_da, mask_da, crs) -> GridArrays:
        """Build from hydromt_sfincs dep and mask DataArrays."""
        bed_elevation_m = dep_da.values.astype(np.float32)
        valid_mask = (mask_da.values > 0) & np.isfinite(bed_elevation_m)
        dx_m = cell_size_m(dep_da.raster.transform)
        return cls(
            crs,
            bed_elevation_m,
            valid_mask,
            dx_m,
            transform=dep_da.raster.transform,
        )

    def close_gaps(self, mask: np.ndarray, iterations: int = 1) -> np.ndarray:
        """Dilate a mask by a fixed 4-connected hop to close sub-cell gaps."""
        return binary_dilation(mask, structure=self._struct4, iterations=iterations)

    def connected_components(self, mask: np.ndarray) -> tuple:
        """Label 8-connected components within mask; return (labels, n_labels).

        8-connectivity: a continuous coastline often narrows to a diagonal-only
        pixel touch; under 4-connectivity the far side would be dropped as a
        small island, leaving a gap in the traced weir.
        """
        labeled, n_labels = ndimage_label(mask, structure=self._struct8)
        return labeled, n_labels

    def seaward_edges(
        self,
        mask_a: np.ndarray,
        water_like: np.ndarray,
        valid_mask: np.ndarray | None = None,
        ocean_exempt: np.ndarray | None = None,
    ) -> list:
        """Trace merged linestrings along the boundary between mask_a and water_like.

        mask_a is the protected/dry side. Cells outside valid_mask (default:
        the grid's active mask) are force-closed to land, except ocean_exempt
        cells (default none), which are open ocean continuing past the domain.
        """
        if valid_mask is None:
            valid_mask = self.valid_mask
        if ocean_exempt is None:
            ocean_exempt = np.zeros_like(valid_mask)
        return _seaward_edges_regular(
            mask_a, water_like, valid_mask, ocean_exempt, self.transform
        )

    def seaward_edges_with_values(
        self,
        mask_a: np.ndarray,
        water_like: np.ndarray,
        value_grid: np.ndarray,
        valid_mask: np.ndarray | None = None,
        ocean_exempt: np.ndarray | None = None,
    ) -> tuple[list, list, np.ndarray]:
        """Trace unmerged edges with the water-side cell's value_grid entry each.

        Returns (segments, values, water_side_mask): one LineString per grid
        edge, so each weir feature keeps a single crest. Edges whose
        water-side value is NaN are dropped. water_side_mask marks the
        water_like cells that supplied a value. valid_mask, ocean_exempt: as
        in seaward_edges.
        """
        if valid_mask is None:
            valid_mask = self.valid_mask
        if ocean_exempt is None:
            ocean_exempt = np.zeros_like(valid_mask)
        return _seaward_edges_regular_with_values(
            mask_a, water_like, value_grid, valid_mask, ocean_exempt, self.transform
        )


def _explode_lines(merged) -> list:
    if merged is None or merged.is_empty:
        return []
    if merged.geom_type == "LineString":
        return [merged]
    return list(merged.geoms)


def _pad_for_edge_tracing(
    mask_a: np.ndarray,
    water_like: np.ndarray,
    valid_mask: np.ndarray,
    ocean_exempt: np.ndarray,
    transform,
):
    # Inactive cells and the array exterior would otherwise leave water_like
    # with no traced edge. Ocean cells are exempt (open water continuing past
    # the domain, not a coastline); the padded ring inherits the nearest real
    # cell's ocean_exempt. The ring decision is applied to the ring only, so
    # interior channel cells are never re-decided.
    closure_valid = valid_mask | ocean_exempt
    mask_a = mask_a | ~closure_valid
    water_like = water_like & closure_valid

    h, w = mask_a.shape
    mask_a_padded = np.zeros((h + 2, w + 2), dtype=bool)
    water_like_padded = np.zeros((h + 2, w + 2), dtype=bool)
    mask_a_padded[1:-1, 1:-1] = mask_a
    water_like_padded[1:-1, 1:-1] = water_like

    ring = np.ones((h + 2, w + 2), dtype=bool)
    ring[1:-1, 1:-1] = False
    ocean_exempt_edge = np.pad(ocean_exempt, 1, mode="edge")
    # Ring is never water_like; it is land unless its nearest real cell is ocean_exempt.
    mask_a_padded[ring] = ~ocean_exempt_edge[ring]

    padded_transform = transform * Affine.translation(-1, -1)
    return mask_a_padded, water_like_padded, padded_transform


def _seaward_edges_regular(
    mask_a: np.ndarray,
    water_like: np.ndarray,
    valid_mask: np.ndarray,
    ocean_exempt: np.ndarray,
    transform,
) -> list:
    mask_a, water_like, transform = _pad_for_edge_tracing(
        mask_a, water_like, valid_mask, ocean_exempt, transform
    )
    segments = []
    h_break = (mask_a[:-1, :] & water_like[1:, :]) | (
        water_like[:-1, :] & mask_a[1:, :]
    )
    for r, c in zip(*np.where(h_break)):
        x0, y0 = transform * (c, r + 1)
        x1, y1 = transform * (c + 1, r + 1)
        segments.append(LineString([(x0, y0), (x1, y1)]))
    v_break = (mask_a[:, :-1] & water_like[:, 1:]) | (
        water_like[:, :-1] & mask_a[:, 1:]
    )
    for r, c in zip(*np.where(v_break)):
        x0, y0 = transform * (c + 1, r)
        x1, y1 = transform * (c + 1, r + 1)
        segments.append(LineString([(x0, y0), (x1, y1)]))
    if not segments:
        return []
    return _explode_lines(linemerge(unary_union(segments)))


def _seaward_edges_regular_with_values(
    mask_a: np.ndarray,
    water_like: np.ndarray,
    value_grid: np.ndarray,
    valid_mask: np.ndarray,
    ocean_exempt: np.ndarray,
    transform,
) -> tuple[list, list, np.ndarray]:
    mask_a, water_like, transform = _pad_for_edge_tracing(
        mask_a, water_like, valid_mask, ocean_exempt, transform
    )
    # Padding only aligns indices: the water side is never in the ring.
    value_grid = np.pad(value_grid, 1, mode="edge")
    water_side = np.zeros(mask_a.shape, dtype=bool)
    segments, values = [], []
    h_break = (mask_a[:-1, :] & water_like[1:, :]) | (
        water_like[:-1, :] & mask_a[1:, :]
    )
    for r, c in zip(*np.where(h_break)):
        water_r = r + 1 if mask_a[r, c] else r
        v = value_grid[water_r, c]
        if np.isnan(v):
            continue
        x0, y0 = transform * (c, r + 1)
        x1, y1 = transform * (c + 1, r + 1)
        segments.append(LineString([(x0, y0), (x1, y1)]))
        values.append(float(v))
        water_side[water_r, c] = True
    v_break = (mask_a[:, :-1] & water_like[:, 1:]) | (
        water_like[:, :-1] & mask_a[:, 1:]
    )
    for r, c in zip(*np.where(v_break)):
        water_c = c + 1 if mask_a[r, c] else c
        v = value_grid[r, water_c]
        if np.isnan(v):
            continue
        x0, y0 = transform * (c + 1, r)
        x1, y1 = transform * (c + 1, r + 1)
        segments.append(LineString([(x0, y0), (x1, y1)]))
        values.append(float(v))
        water_side[r, water_c] = True
    return segments, values, water_side[1:-1, 1:-1]


def discard_small_components(
    mask: np.ndarray, grid: GridArrays, min_cells: int
) -> tuple[np.ndarray, int, int]:
    """Drop components smaller than min_cells; return (mask, n_cells, n_components) dropped."""
    labels, n_labels = grid.connected_components(mask)
    if n_labels == 0:
        return mask, 0, 0
    comp_sizes = np.bincount(labels[mask], minlength=n_labels + 1)
    small_labels = np.where((comp_sizes > 0) & (comp_sizes < min_cells))[0]
    if len(small_labels) == 0:
        return mask, 0, 0
    small_mask = np.isin(labels, small_labels) & mask
    return mask & ~small_mask, int(small_mask.sum()), int(len(small_labels))


def build_weir_geodataframe(
    weir_lines: list, crest_elevation_m: float | list, par1: float, crs
) -> gpd.GeoDataFrame:
    """Build the weir GeoDataFrame; crest_elevation_m is a scalar or one value per line."""
    if not weir_lines:
        return gpd.GeoDataFrame({"elevation": [], "par1": []}, geometry=[], crs=crs)
    elevations = (
        list(crest_elevation_m)
        if isinstance(crest_elevation_m, (list, np.ndarray))
        else [crest_elevation_m] * len(weir_lines)
    )
    return gpd.GeoDataFrame(
        {"elevation": elevations, "par1": [par1] * len(weir_lines)},
        geometry=weir_lines,
        crs=crs,
    )


def _flatten_elevation_from_z(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # hydromt_sfincs SfincsWeirs.read() (utils.linestring2gdf) stores elevation
    # in the geometry's Z when its length matches the vertex count, leaving no
    # "elevation" column; gis/weir.geojson (utils.write_vector) would then show
    # a Null elevation. The sfincs.weir writer itself reconstructs it from Z.
    if gdf.empty:
        return gdf
    gdf = gdf.copy()
    if "elevation" not in gdf.columns:
        gdf["elevation"] = np.nan
    for idx, geom in gdf.geometry.items():
        if geom.has_z and pd.isna(gdf.at[idx, "elevation"]):
            z = [c[2] for c in geom.coords]
            gdf.at[idx, "elevation"] = z[0] if len(set(z)) == 1 else z
            gdf.at[idx, "geometry"] = LineString([(c[0], c[1]) for c in geom.coords])
    return gdf


def merge_weir_preserve_unmatched(
    baseline_gdf: gpd.GeoDataFrame,
    new_gdf: gpd.GeoDataFrame,
    max_match_distance_m: float = 2.0,
) -> gpd.GeoDataFrame:
    """Replace baseline weir segments matched by new_gdf, keep the rest, add all of new_gdf.

    Used by the protect_closed strategy (apply_coastal_levee) to re-crest the
    coast/river mouth while keeping the inland river weir. A baseline segment
    is replaced when its midpoint lies within max_match_distance_m of a
    new_gdf segment. Distance, not coordinate equality: copies drift by cm
    (QGIS re-save, the 0.1 m "%11.1f" sfincs.weir round trip). 2.0 m is well
    under half a ~70 m cell yet clears that drift (observed matches < 1 m,
    distinct segments > 5 m).
    """
    if baseline_gdf is None or baseline_gdf.empty:
        return new_gdf
    if new_gdf.empty:
        return baseline_gdf

    baseline_gdf = _flatten_elevation_from_z(baseline_gdf)

    new_geoms = list(new_gdf.geometry.values)
    tree = STRtree(new_geoms)

    def _is_replaced(geom):
        # Midpoint, not whole-segment distance: segments that merely share an
        # endpoint have distance 0 and would be wrongly dropped, leaving a gap.
        midpoint = geom.interpolate(0.5, normalized=True)
        nearest_idx = tree.nearest(midpoint)
        return midpoint.distance(new_geoms[nearest_idx]) <= max_match_distance_m

    is_replaced = baseline_gdf.geometry.apply(_is_replaced)
    unmatched = baseline_gdf[~is_replaced]

    cols = ["geometry", "elevation", "par1"]
    unmatched = unmatched[[c for c in cols if c in unmatched.columns]]
    new_part = new_gdf[[c for c in cols if c in new_gdf.columns]]

    log.info(
        f"Weir merge: replacing {int(is_replaced.sum())} baseline segment(s) matched by "
        f"the new coastal trace, keeping {len(unmatched)} unmatched baseline segment(s) "
        f"(the river) unchanged, {len(new_part) - int(is_replaced.sum())} new segment(s) "
        "in the new trace have no baseline match and are simply added"
    )

    return gpd.GeoDataFrame(
        pd.concat([unmatched, new_part], ignore_index=True), crs=new_gdf.crs
    )


def _remove_small_dikerings(
    weir_gdf: gpd.GeoDataFrame,
    min_component_cells: int,
    cell_size_m: float,
    coord_ndigits: int = 3,
) -> gpd.GeoDataFrame:
    # Vector counterpart of discard_small_components: a small feature joined to
    # a kept component at a diagonal pinch still traces its own ring, hanging
    # off a degree-4 vertex. Rings closing back to that vertex through
    # degree-2 vertices only and enclosing < min_component_cells cells are
    # dropped; the main boundary's through-segments stay. coord_ndigits is a
    # float-noise margin, not a snapping tolerance.
    if weir_gdf.empty:
        return weir_gdf

    coords = [list(geom.coords) for geom in weir_gdf.geometry]

    def _key(pt):
        return (round(pt[0], coord_ndigits), round(pt[1], coord_ndigits))

    endpoints = [(_key(c[0]), _key(c[-1])) for c in coords]

    vertex_segments: dict[tuple, list[int]] = {}
    for seg_idx, (a, b) in enumerate(endpoints):
        vertex_segments.setdefault(a, []).append(seg_idx)
        vertex_segments.setdefault(b, []).append(seg_idx)

    def _other_endpoint(seg_idx: int, vkey: tuple) -> tuple:
        a, b = endpoints[seg_idx]
        return b if a == vkey else a

    # Caps the walk so a pinch on the main loop does not trace the whole
    # network; generous versus any small loop's perimeter.
    max_trace_segments = max(64, min_component_cells * 8)

    def _trace_loop(start_vkey: tuple, first_seg_idx: int):
        loop_segs = [first_seg_idx]
        loop_coords = [start_vkey]
        prev_seg = first_seg_idx
        cur_vkey = _other_endpoint(first_seg_idx, start_vkey)
        while cur_vkey != start_vkey:
            if len(loop_segs) > max_trace_segments:
                return None
            incident = vertex_segments.get(cur_vkey, [])
            if len(incident) != 2:
                return None
            next_seg = incident[1] if incident[0] == prev_seg else incident[0]
            loop_segs.append(next_seg)
            loop_coords.append(cur_vkey)
            prev_seg = next_seg
            cur_vkey = _other_endpoint(next_seg, cur_vkey)
        return loop_segs, loop_coords

    to_drop: set[int] = set()
    seen_loops: set[frozenset] = set()
    for vkey, seg_idxs in vertex_segments.items():
        if len(seg_idxs) < 4:
            continue
        for seg_idx in seg_idxs:
            result = _trace_loop(vkey, seg_idx)
            if result is None:
                continue
            loop_segs, loop_coords = result
            loop_key = frozenset(loop_segs)
            if loop_key in seen_loops:
                continue
            seen_loops.add(loop_key)
            ring_coords = loop_coords + [loop_coords[0]]
            if len(set(loop_coords)) < 3:
                continue  # degenerate ring
            n_cells = Polygon(ring_coords).area / (cell_size_m**2)
            if n_cells < min_component_cells:
                to_drop.update(loop_segs)

    if not to_drop:
        return weir_gdf

    log.info(
        f"Dropped {len(to_drop)} segment(s) forming small dikering(s) attached to the main "
        f"boundary at a single pinch point (< {min_component_cells} cell(s) enclosed each) -- "
        f"the main boundary's own through-segments at each pinch point are kept"
    )
    return weir_gdf.drop(index=list(to_drop)).reset_index(drop=True)


def ocean_linked_wetland_mask(
    landuse_on_grid: np.ndarray, ocean_mask: np.ndarray, grid: GridArrays
) -> np.ndarray:
    """Return OCEAN_WETLAND_CLASSES cells in patches sharing a cell edge with the open sea.

    4-connectivity on purpose: SFINCS exchanges water only across shared
    cell edges, so a patch touching the sea at a corner could never fill.
    """
    wetland = np.isin(landuse_on_grid, OCEAN_WETLAND_CLASSES) & grid.valid_mask
    labels, n_labels = ndimage_label(wetland, structure=grid._struct4)
    if n_labels == 0:
        return np.zeros_like(wetland)
    touching_sea = binary_dilation(ocean_mask, structure=grid._struct4) & wetland
    linked = np.unique(labels[touching_sea])
    return np.isin(labels, linked[linked > 0])


def build_coastal_protection_weir(
    grid: GridArrays,
    landuse_on_grid: np.ndarray,
    river_channel_mask: np.ndarray,
    crest_elevation_m: float,
    min_component_cells: int,
    weir_par1: float,
    water_side_crest_on_grid: np.ndarray | None = None,
    freeboard_m: float = 0.0,
    channel_mask_gap_free: bool = False,
    unprotected_ocean_wetlands: bool = False,
) -> tuple[gpd.GeoDataFrame, dict]:
    """Classify water_like, drop small components, trace the weir; return (weir_gdf, diagnostics).

    unprotected_ocean_wetlands: sea-linked wetland/lagoon patches join
    water_like, so the dike runs along their landward edge; the cells stay
    land otherwise (not sea in zsini_sea_cells, counted in flood metrics).
    channel_mask_gap_free: True when river_channel_mask comes from the same
    burned DEM fed to hydromt_sfincs (rule 10, 10_depth_estimation_modelled.py);
    skips the 1-cell close_gaps() dilation, which would
    leave an un-excavated ring around the channel.
    water_side_crest_on_grid: per-cell crest looked up on the water side of
    each edge (rule 10 passes the confined round's period-max water level).
    If given, the river channel joins water_like and each crest is
    max(water side, crest_elevation_m); NaN (never wetted) falls back to
    crest_elevation_m, so an all-NaN array gives a uniform weir including the
    riverbanks. If None, no riverbank weir is built (coast only, uniform crest).
    freeboard_m: added after the max(), for under-protection the crest
    calculation does not capture (subgrid roundoff, steady-state assumption).
    Every crest is rounded UP to the next centimetre (src.surge.ceil_crest),
    the precision sfincs.weir is written at, so no edge falls below its crest.
    """
    ocean_mask = landuse_on_grid == LANDUSE_SEA_CODE

    n_ocean = int(ocean_mask.sum())
    if n_ocean == 0:
        log.warning(
            "No ocean cells (landuse==200) found on the model grid -- basin "
            "appears inland/river-only, coastal protection weir is not applicable."
        )
        empty_gdf = build_weir_geodataframe([], crest_elevation_m, weir_par1, grid.crs)
        return empty_gdf, {"applicable": False}

    seaward_mask = ocean_mask
    ocean_wetland = np.zeros_like(ocean_mask)
    if unprotected_ocean_wetlands:
        ocean_wetland = ocean_linked_wetland_mask(landuse_on_grid, ocean_mask, grid)
        seaward_mask = ocean_mask | ocean_wetland
        log.info(
            f"Unprotected ocean-linked wetlands: {int(ocean_wetland.sum()):,} cell(s) (landuse "
            f"{list(OCEAN_WETLAND_CLASSES)}, edge-connected to the open sea) left on the seaward "
            f"side of the coastal dike"
        )

    if water_side_crest_on_grid is None:
        water_like_raw = seaward_mask
    else:
        if not channel_mask_gap_free:
            river_channel_mask = grid.close_gaps(river_channel_mask, iterations=1)
        water_like_raw = seaward_mask | river_channel_mask

    land_mask_raw = ~water_like_raw & grid.valid_mask

    land_mask, n_land_discarded, n_land_components = discard_small_components(
        land_mask_raw, grid, min_component_cells
    )
    water_like, n_water_discarded, n_water_components = discard_small_components(
        water_like_raw, grid, min_component_cells
    )

    # Discarded cells join the opposite class: a cell in neither mask is
    # invisible to the tracer and would break the larger feature's boundary.
    # Both sets come from the pre-reclassification masks, so order is irrelevant.
    _land_discarded = land_mask_raw & ~land_mask
    _water_discarded = water_like_raw & ~water_like
    land_mask = land_mask | _water_discarded
    water_like = water_like | _land_discarded

    if n_land_discarded:
        log.info(
            f"Discarded {n_land_discarded:,} land cell(s) across {n_land_components} small "
            f"island(s) (< {min_component_cells} cells each) -- no weir ring around them"
        )
    if n_water_discarded:
        log.info(
            f"Discarded {n_water_discarded:,} water cell(s) across {n_water_components} small "
            f"patch(es) (< {min_component_cells} cells each) -- likely isolated ponds/artifacts, "
            "no weir ring around them"
        )

    # Protected ocean pockets: cells still landuse==200 although the weir
    # puts them on the protected side.
    #  (a) Grid misalignment along the whole coast: the native landuse.tif
    #      (~30 m) resampled by mode onto the ~70 m SFINCS grid (rule
    #      grid_align_landuse, 09b) generalizes the coastline, so land_mask is
    #      taken as ground truth; callers subtract it from the landuse==200
    #      cells on the same grid, without reprojection.
    #  (b) an ocean component surviving into water_like but not the main open
    #      ocean (the single largest water_like component, as for zsini in
    #      13_build_sfincs_skeleton.py), walled off by its own ring.
    # Returned so callers clear these cells from sea to land in sea_mask.tif,
    # used by zsini and by downstream flood-diagnostic masking.
    water_like_labels, n_water_like_components = grid.connected_components(water_like)
    if n_water_like_components > 0:
        comp_sizes = np.bincount(
            water_like_labels[water_like], minlength=n_water_like_components + 1
        )
        comp_sizes[0] = 0  # label 0 = outside mask, never the "main" component
        main_ocean_label = int(np.argmax(comp_sizes))
        separately_ringed_pocket = (
            ocean_mask & water_like & (water_like_labels != main_ocean_label)
        )
    else:
        separately_ringed_pocket = np.zeros_like(ocean_mask)
    protected_pocket_mask = land_mask | separately_ringed_pocket
    n_separately_ringed = int(separately_ringed_pocket.sum())
    if n_separately_ringed:
        log.info(
            f"{n_separately_ringed:,} ocean-classified (landuse==200) grid cell(s) form their "
            f"own separately-ringed pocket, disconnected from the main open ocean component"
        )
    # protected_pocket_mask is deliberately broad (the whole protected side),
    # so no affected-cell count is logged.

    n_final = int(land_mask.sum())

    # Only ocean cells that survived discarding are exempt; a dropped patch
    # would otherwise be neither land nor water and break continuity.
    ocean_exempt = water_like & ocean_mask

    crest_surface = None
    edge_water_side_mask = None
    if water_side_crest_on_grid is None:
        weir_lines = grid.seaward_edges(
            land_mask, water_like, valid_mask=grid.valid_mask, ocean_exempt=ocean_exempt
        )
        weir_crests: list | float = float(ceil_crest(crest_elevation_m + freeboard_m))
        log.info(
            f"Weir line: {len(weir_lines)} segment(s) extracted directly at the coast "
            f"({n_final:,} land cells), crest={weir_crests:+.2f} m (floor "
            f"{crest_elevation_m:+.3f} m +{freeboard_m:.2f} m freeboard, rounded up to 1 cm) "
            f"-- flat crest, no per-edge water-side crest given"
        )
    else:
        # np.fmax: a NaN (never-wetted) cell falls back to crest_elevation_m,
        # since seaward_edges_with_values drops NaN edges and would leave gaps.
        # float64 so the centimetre-rounded values stay exact in the gpkg.
        crest_surface = ceil_crest(
            np.fmax(water_side_crest_on_grid.astype(float), float(crest_elevation_m))
            + freeboard_m
        )
        weir_lines, weir_crests, edge_water_side_mask = grid.seaward_edges_with_values(
            land_mask,
            water_like,
            crest_surface,
            valid_mask=grid.valid_mask,
            ocean_exempt=ocean_exempt,
        )
        _edge_water_level = water_side_crest_on_grid[edge_water_side_mask]
        n_above_floor = int((_edge_water_level > crest_elevation_m).sum())
        log.info(
            f"Weir line: {len(weir_lines)} segment(s) extracted directly at the coast/riverbank "
            f"({n_final:,} land cells); {n_above_floor:,} of {int(edge_water_side_mask.sum()):,} "
            f"water-side cell(s) set their edges' crest from their own water level, the rest use "
            f"the floor crest={crest_elevation_m:+.3f} m (+{freeboard_m:.2f} m freeboard on all, "
            f"every crest rounded up to 1 cm)"
        )

    weir_gdf = build_weir_geodataframe(weir_lines, weir_crests, weir_par1, grid.crs)

    _dikering_cell_size_m = (
        float(grid.cell_size_m)
        if np.isscalar(grid.cell_size_m)
        else float(np.median(grid.cell_size_m))
    )
    weir_gdf = _remove_small_dikerings(
        weir_gdf, min_component_cells, _dikering_cell_size_m
    )

    diagnostics = {
        "applicable": True,
        "ocean_mask": ocean_mask,
        "river_channel_mask": river_channel_mask,
        "land_mask": land_mask,
        "weir_lines": list(weir_gdf.geometry),
        "n_final": n_final,
        "n_land_discarded": n_land_discarded,
        "n_water_discarded": n_water_discarded,
        "crest_elevation_m": crest_elevation_m,
        "protected_pocket_mask": protected_pocket_mask,
        "ocean_wetland_mask": ocean_wetland,
        # None in the coast-only branch; lets a caller compare a later run's
        # water level against the crest each edge was given.
        "crest_surface": crest_surface,
        "edge_water_side_mask": edge_water_side_mask,
    }
    return weir_gdf, diagnostics
