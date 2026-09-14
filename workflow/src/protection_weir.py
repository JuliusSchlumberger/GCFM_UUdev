"""
protection_weir.py — Builds a coastal/riverbank protection weir directly on
the SFINCS model's regular grid.

The weir represents the protection-level standard itself: flood defenses
are assumed to run continuously along the whole coast and river network (not
just near named reaches), so no flooding should occur below the crest
elevation anywhere in the domain until it's overtopped. It hugs the
coast/riverbank directly -- no inland offset.

Built by rule modelled_depth_estimation (10, 10_depth_estimation_modelled.py)
only -- coast and river channel merged into one water_like boundary, so
calibration and production share the same single weir set (rule 13 imports
the final gpkg as-is). That rule calls this twice: its confinement rounds
with crest_elevation_m set to an artificially high value (e.g. 1000 m) and
an all-NaN water_side_crest_on_grid, so EVERY edge -- coast and riverbank
alike -- sits at that confinement height (the confined river's water level
can exceed a low real coastal crest right at the coast/river transition
near the mouth, so calibration confines everything, not just the river);
and once more for the production weir, with water_side_crest_on_grid = the
confined, excavated round's own period-max water level and the real,
datum-corrected coastal crest as crest_elevation_m -- every edge's crest is
then the water level on its own water side, floored at the coastal
standard.

Design summary
--------------
1. Classify ocean (landuse==200) and river-channel cells/faces on the model
   grid into one combined ``water_like`` set; everything else valid is
   ``land``. landuse==80 (inland water/tidal flat) / 90 (wetland) is just
   land unless it coincides with the channel mask -- or, with
   unprotected_ocean_wetlands=True, its patch is edge-connected to the open
   sea (ocean_linked_wetland_mask), which puts it on the seaward side too.
2. Drop small connected components on *both* sides (``discard_small_
   components``, ``min_component_cells``): small isolated land islands get
   no protective ring around them (they flood normally), and small isolated
   water patches (misclassified ponds, tidal pools) get no ring walling the
   surrounding land off from them either -- neither is a real coastline/
   riverbank worth defending. A component dropped from either side is
   simply excluded from both, so no edge gets traced there at all.
3. Extract the boundary multiline directly between the two resulting sets
   from the grid's own pixel/mesh-face edges -- correctly aligned to real
   flux links by construction, avoiding the seepage risk a line derived
   from a different-resolution source would carry.
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

from src.landuse import OCEAN_WETLAND_CLASSES, SEA_CODE
from src.surge import ceil_water_level

log = logging.getLogger(__name__)

# Re-exported under this module's own long-standing name (scripts import it
# from here). Both are defined in src.landuse, next to the land-use codes
# they belong to: LANDUSE_SEA is the pipeline's own open-sea code (LC100's
# own 200, DERIVED for ESA WorldCover, which has no sea class), and
# OCEAN_WETLAND_CLASSES are the classes that may sit OUTSIDE the coastal
# dike when their patch is linked to the open sea
# (build_coastal_protection_weir's unprotected_ocean_wetlands): 80 =
# permanent water bodies (lagoons, tidal flats), 90 = herbaceous wetland,
# 95 = mangroves (WorldCover only; intertidal by definition).
LANDUSE_SEA = SEA_CODE


class GridArrays:
    """
    Uniform interface over a SFINCS regular grid's already-loaded arrays
    (built from sf.grid.data mid-build, or from a read-mode model's own
    data -- either works, this class only needs the arrays themselves).
    """

    def __init__(
        self,
        crs,
        bed_elevation: np.ndarray,
        valid_mask: np.ndarray,
        cell_size_m,
        transform,
    ):
        self.crs = crs
        self.bed_elevation = bed_elevation
        self.valid_mask = valid_mask
        self.cell_size_m = cell_size_m
        self.transform = transform
        self.shape = bed_elevation.shape
        self.n_cells = bed_elevation.size
        self._struct4 = generate_binary_structure(2, 1)
        self._struct8 = generate_binary_structure(2, 2)

    @classmethod
    def from_regular(cls, dep_da, mask_da, crs) -> "GridArrays":
        bed_elevation = dep_da.values.astype(np.float32)
        valid_mask = (mask_da.values > 0) & np.isfinite(bed_elevation)
        dx = abs(float(dep_da.raster.transform.a))
        return cls(
            crs,
            bed_elevation,
            valid_mask,
            dx,
            transform=dep_da.raster.transform,
        )

    def close_gaps(self, mask: np.ndarray, iterations: int = 1) -> np.ndarray:
        """Small, FIXED hop dilation of a boolean mask -- a safety net for
        any remaining sub-cell gaps in a mask (e.g. a channel narrower than
        one destination cell in only a handful of spots).
        """
        return binary_dilation(mask, structure=self._struct4, iterations=iterations)

    def connected_components(self, mask: np.ndarray) -> tuple:
        """Connected components WITHIN mask (cells outside mask are never
        connected to anything). Returns (labels, n_labels).

        Uses 8-CONNECTIVITY (self._struct8, diagonals count as connected),
        not scipy.ndimage.label's own 4-connectivity default. A real,
        physically continuous coastline/riverbank commonly narrows to a
        single diagonal-only pixel-to-pixel connection at some point (a
        thin, jagged spit -- an ordinary raster-discretization artifact,
        not a genuine break in the landform). Under 4-connectivity that
        diagonal touch does NOT count as connected, so
        discard_small_components (this method's only caller) sees the far
        side as a SEPARATE, small component and drops it as if it were a
        real, isolated small island -- leaving a dangling gap in the traced
        weir exactly at that pinch point, since the dropped cells become
        neither land_mask nor water_like and the edge tracer draws no
        segment against either side of them (see Reference_memory.txt
        section 14, WEIR TOPOLOGY GAPS (DIAGONAL 4-CONNECTIVITY) entry).
        8-connectivity treats a diagonal touch as one continuous component,
        matching the physical reality.
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
        """Linestrings tracing the boundary between mask_a and water_like
        cells -- the seaward edge the weir should follow. Generic: mask_a is
        the protected/dry side (land, or a confined channel).

        valid_mask: which cells get force-closed to "land" wherever
        water_like reaches them (see _pad_for_edge_tracing) -- defaults to
        self.valid_mask (the grid's own true active-cell mask).
        ocean_exempt: which cells are exempt from that force-closure
        even when invalid, because they're genuinely open ocean rather
        than land (see _pad_for_edge_tracing) -- defaults to all-False (no
        exemption). Callers with domain knowledge of which cells are ocean
        (e.g. build_coastal_protection_weir) should pass it explicitly.
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
        """Like seaward_edges, but returns (segments, values,
        water_side_mask) UNMERGED -- one LineString per grid edge, paired
        with the WATER_LIKE-side cell's own value_grid entry (the water the
        edge holds back, e.g. its own simulated water level). Deliberately
        not linemerge'd: adjacent edges can carry different values, so
        segments must stay separable to keep a single crest value per
        output weir feature. Segments whose water-side value is NaN are
        dropped entirely. water_side_mask (grid shape) marks every
        water_like cell that supplied at least one edge's value.

        valid_mask, ocean_exempt: see seaward_edges's own docstring -- same
        override semantics.
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
    """Force-close mask_a/water_like wherever the model doesn't actually
    simulate anything -- outside the array's own edge, or an inactive
    interior cell -- then pad by 1 cell and shift the transform to match.

    Two DISTINCT reasons a cell can be "not really simulated": beyond the
    array's own bounds (no data at all), or an inactive interior cell
    (grid.valid_mask == False, e.g. outside sf.mask.create_active's own
    domain polygon). Both leave a structural gap in the H/V break detection
    below (it only ever compares adjacent cells WITHIN the array, and
    land_mask_raw upstream, ~water_like_raw & grid.valid_mask, already
    excludes invalid cells from ever counting as land) -- so wherever
    water_like reaches either kind of edge, no seaward edge is traced there
    at all.

    NOT every such cell should be force-closed, though: one classified as
    OCEAN (ocean_exempt) is genuinely just open water continuing past
    where the model happens to stop simulating, not a coastline -- forcing
    it closed draws a false "dike" along the domain's own seaward edge. So
    the interior case folds in ~valid_mask & ~ocean_exempt only; the
    array's own exterior ring has no ocean_exempt value of its own (no data
    at all out there) and instead inherits its nearest real neighbour's
    ocean_exempt via edge-padding. A channel
    reach whose own head sits right at the array's edge (not ocean there)
    still gets force-closed exactly as before.

    The exterior-ring decision (via ocean_exempt) is applied ONLY to the
    new border ring added by padding, via explicit slicing -- NOT as a
    blanket elementwise op over the whole padded array, which would
    silently re-decide every INTERIOR cell's closure too, including every
    real river channel cell.
    """
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
    # water_like_padded[ring] is already False (the ring's own initial
    # value) in both cases (exempt or not) -- only mask_a needs deciding:
    # land wherever the nearest real neighbour wasn't exempt, neither
    # land nor water where it was (open ocean continuing past the edge).
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
    # The water side of a traced edge is always a real interior cell (the
    # padding ring's own water_like is always False, see
    # _pad_for_edge_tracing), so value_grid only needs padding to keep
    # indices aligned with the padded masks -- the ring's own values are
    # never read.
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
    """Drop connected components of ``mask`` smaller than min_cells --
    called on both the land side and the water_like side when building a
    protection weir, so neither a small isolated land island nor a small
    isolated water patch (misclassified pond, tidal pool) gets a weir ring
    around it; a farm pond does not need a dike. Returns
    (filtered_mask, n_discarded_cells, n_discarded_components).
    """
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
    """crest_elevation_m: a single scalar (broadcast to every segment,
    today's uniform-crest behavior) or a per-segment list (same length as
    weir_lines, one crest value per feature).
    """
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
    """Give every row a populated, flat "elevation" column and plain 2D
    geometry, regardless of which of the two representations hydromt_sfincs
    happened to hand it in.

    SfincsWeirs.read() (utils.linestring2gdf) embeds a per-vertex elevation
    list INTO the geometry's own Z-coordinate instead of keeping a separate
    "elevation" column, whenever the list's length matches the vertex count
    -- true of every weir segment this repo ever writes (2 vertices, 2
    elevation values) -- so mod.weirs.data read back from a real sfincs.weir
    file has NO "elevation" column at all, only Z-embedded geometry. The
    ASCII sfincs.weir writer (utils.gdf2linestring) still reconstructs
    elevation correctly from Z on write, so the MODEL is unaffected by this
    either way -- but gis/weir.geojson is written straight from this same
    GeoDataFrame (utils.write_vector, no such reconstruction), so any row
    still relying on Z alone shows a blank/Null "elevation" attribute in
    QGIS despite carrying a perfectly real crest -- confirmed via basin
    2433835/protect_closed_1, where every preserved river segment (kept
    as-is from mod.weirs.data by merge_weir_preserve_unmatched, its own only
    caller) showed Null there.
    """
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
    """
    Used by the protect_closed adaptation strategy (apply_coastal_levee,
    src/adaptation_method_pre.py) to seal off the coast/river-mouth at a new
    elevation while leaving the river's own already-built protection weir
    (further inland, following current protection standards) untouched --
    without this, calling mod.weirs.set(new_gdf, merge=False) directly would
    wholesale replace the ENTIRE baseline weir with just the new coastal
    lines, silently deleting the river's protection.

    new_gdf's coastal rows are expected to sit at essentially the same
    location as the baseline rows they're meant to replace, since the
    intended workflow is copying the basin's own baseline weir file (e.g.
    {basin_id}_coastal_protection_weir.gpkg), deleting its river rows, and
    adding one new segment that closes off the river mouth. Matching is
    nearest-neighbour DISTANCE, not exact/rounded-coordinate equality: a
    "copy" in practice still picks up a few cm of drift (re-saved/
    reprojected through QGIS, or round-tripped through the ASCII
    sfincs.weir file's own 1-decimal precision on the baseline side -- see
    below), and that drift lands randomly on either side of any fixed
    rounding boundary, so two coordinates 8 cm apart can round to values
    0.1 m apart and never compare equal no matter how many digits are
    chosen -- confirmed via basin 2433835/protect_closed_1: baseline
    (300087.1, 4492070.0) vs new (300087.085, 4492069.920666667), 8 cm
    apart, whose y rounds to 4492070.0 and 4492069.9 respectively. Distance
    to the nearest new_gdf segment is unambiguous regardless of which side
    of a boundary either coordinate happens to fall on.

    max_match_distance_m defaults to 2.0 m, well under half a grid cell (this
    repo's basins run ~70 m resolution) so it can never conflate two
    genuinely different grid-edge segments, but comfortably clears both the
    baseline's own 0.1 m ASCII round-trip truncation (hydromt_sfincs.utils.
    write_geoms's "%11.1f" format) and the sub-metre editing/reprojection
    drift observed above -- confirmed via the same basin/strategy: every
    baseline segment's distance to its nearest new_gdf segment fell into two
    clean, non-overlapping bands with nothing in between -- under 1 m (the
    same physical location, reformatting noise only) or over 5 m, up to
    8.5 km (a genuinely distinct, inland river segment) -- so any threshold
    in that gap works equally well for this data; 2.0 m leaves margin for
    noisier input on a basin not yet seen.

    Every baseline segment within max_match_distance_m of some new_gdf
    segment is dropped (new_gdf's own row -- carrying its own new elevation
    -- takes its place); every baseline segment with no near match (the
    river) is kept unchanged; every new_gdf row is added regardless (so the
    new river-mouth closure segment, with no baseline counterpart at all,
    is included too).
    """
    if baseline_gdf is None or baseline_gdf.empty:
        return new_gdf
    if new_gdf.empty:
        return baseline_gdf

    baseline_gdf = _flatten_elevation_from_z(baseline_gdf)

    new_geoms = list(new_gdf.geometry.values)
    tree = STRtree(new_geoms)

    def _is_replaced(geom):
        # Midpoint, NOT geom.distance(...) against the whole segment -- two
        # segments that merely CONNECT (share one endpoint, e.g. where a
        # coastal_levee trace hands off to a sibling river_levee trace at
        # their shared junction vertex) register as distance 0 under a
        # whole-geometry distance check despite running in entirely
        # different directions and covering no common ground otherwise,
        # wrongly marking the coastal segment "replaced" with nothing in
        # river_gdf actually covering its own stretch -- confirmed via basin
        # 2433835/grey_protect_open_1: coastal segment (319397.977,
        # 4510261.34)-(319467.944, 4510261.34) shares its second endpoint
        # exactly with a river segment ending at the same point but running
        # north-south, got dropped as "replaced", and nothing else covered
        # that ~70 m stretch -- a real gap in the final weir. A genuine
        # duplicate/near-duplicate segment (the actual replace-in-place case
        # this function exists for) has its WHOLE length close to the new
        # trace, so its midpoint is close too; a segment only touching the
        # new trace at one endpoint has its midpoint far from it, since nothing
        # else about the two segments coincides.
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
    """
    Vector-space counterpart to discard_small_components: that function
    already drops a small isolated land island/water patch that is its OWN
    separate connected component -- but a small feature attached to a
    larger, kept component at a single diagonal-pinch pixel (exactly the
    geometry connected_components' own 8-connectivity treats as "part of
    the larger component," correctly, from a raster-masking standpoint)
    still traces its own tiny closed ring in the final VECTOR output,
    meeting the main boundary at one shared vertex where FOUR segments
    meet -- two continuing the main boundary through that point, two
    entering/leaving the small ring. discard_small_components never sees
    this case at all, since at the raster level the small feature is
    already merged into one large, correctly-kept component; only after
    tracing does it show up as its own separate little loop.

    For every vertex where exactly two of its incident segments trace a
    closed loop back to that SAME vertex through only degree-2 (simple
    chain) intermediate vertices -- i.e. a standalone ring touching the
    rest of the network at exactly that one point -- this computes the
    loop's own enclosed area and, if it's under min_component_cells worth
    of grid cells, drops every segment making up that loop. The other two
    segments at the pinch vertex (the main boundary's own through-path)
    are always left untouched, so the boundary continues with no gap --
    mirroring discard_small_components' own "no ring around a too-small
    feature" outcome, just reached in vector space instead of raster space.

    coord_ndigits rounds endpoint coordinates before matching -- segments
    come from exact grid-edge construction, so this is a floating-point-
    noise safety margin, not a real snapping tolerance. cell_size_m is the
    grid's uniform cell size.
    """
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

    # Bounds how far a trace walks before giving up -- without this, a
    # pinch vertex on the boundary's own big main loop/chain (which also,
    # trivially, eventually closes back on itself if it's a full loop)
    # would walk the ENTIRE remaining network before failing the area
    # check, for every degree-4 vertex, which is needlessly expensive on a
    # coastline with tens of thousands of segments (e.g. the Mississippi).
    # Generous relative to what a genuinely small loop's own perimeter can
    # be, since only small loops are ever actually removed.
    max_trace_segments = max(64, min_component_cells * 8)

    def _trace_loop(start_vkey: tuple, first_seg_idx: int):
        """
        Walk from start_vkey via first_seg_idx through degree-2 chain
        vertices only; returns (loop_segment_indices, loop_vertex_keys) if
        this closes back to start_vkey within max_trace_segments steps,
        else None (hit a dangling end, another junction, or ran too long
        first -- not a standalone SMALL loop through this vertex).
        """
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
                continue  # degenerate -- not enough distinct vertices for a real ring
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
    """
    Wetland/lagoon cells (OCEAN_WETLAND_CLASSES) in a patch that shares a
    cell EDGE with the open sea (landuse==200) -- the patches
    build_coastal_protection_weir(unprotected_ocean_wetlands=True) leaves on
    the seaward side of the coastal dike.

    4-connectivity on purpose, both for the patch itself and for its link to
    the sea -- unlike connected_components' own 8-connectivity (a raster-
    topology safeguard for the weir trace): SFINCS only exchanges water
    across shared cell edges, so a patch touching the sea (or its own
    remainder) only at a corner could never fill from it -- it would end up
    walled in on the "unprotected" side without ever getting sea water.
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
    """
    Top-level orchestrator: classify ocean(+river, if modelled) as
    water_like -> discard small isolated components on both the land and
    water_like side -> weir line extraction directly at the resulting
    boundary. Returns (weir_gdf, diagnostics) where diagnostics carries
    everything needed for logging and the diagnostic plot.

    Args:
        unprotected_ocean_wetlands: True puts every wetland/lagoon patch
            linked to the open sea (ocean_linked_wetland_mask) on the
            seaward side too: it joins water_like, so the coastal dike runs
            along the patch's LANDWARD edge and the patch itself stays
            unprotected. The cells stay land otherwise (not sea in
            zsini_sea_cells, counted in flood metrics). False (default):
            the dike follows the open-sea boundary itself.
        channel_mask_gap_free: True when `river_channel_mask` was derived
            directly from the same burned-DEM raster that was actually fed
            to hydromt_sfincs as elevation (regular-grid production builds,
            13_build_sfincs.py) -- excavated extent and channel_mask are
            then the same set of cells BY CONSTRUCTION, so the 1-cell
            close_gaps() safety-net dilation below is skipped entirely: it
            would otherwise push the traced weir line one cell outside the
            true excavated boundary, leaving a thin un-excavated ring around
            the whole channel perimeter. False (default) preserves the
            dilation for an independently-rasterized mask, which can still
            have genuine sub-cell gaps this safety net exists to catch.
        water_side_crest_on_grid: Optional per-cell crest (same shape as
            landuse_on_grid) looked up on the WATER side of every traced
            edge -- rule modelled_depth_estimation passes its confined,
            excavated round's own period-max water level here, so each edge
            holds back exactly the water level simulated right next to it
            (riverbank, mouth, and the coast near the river plume alike).
            When provided: the river channel participates in water_like,
            and each edge's crest is max(water_side_crest_on_grid,
            crest_elevation_m) -- never below the uniform coastal standard;
            a NaN (never-wetted) water-side cell falls back to
            crest_elevation_m, so no edge is ever dropped for a missing
            value. An all-NaN array therefore gives a uniform
            crest_elevation_m weir that still includes the riverbanks (the
            calibration's own confinement rounds). When None (no modelled
            water level to derive a riverbank dike from): the river channel
            does NOT participate in water_like at all, so no riverbank weir
            is built and rivers interact with their floodplain naturally;
            only the coastal levee is built, with the uniform-crest
            behavior.
        freeboard_m: Added uniformly on top of the final crest, after the
            max() above -- a safety margin absorbing sources of
            under-protection the crest computation itself doesn't capture
            (subgrid/discretization roundoff, the calibration's steady-
            state assumption, etc.). 0.0 (the default) applies no freeboard.

    Every crest is finally rounded UP to the next 0.1 m (src.surge.
    ceil_water_level) -- sfincs.weir stores crests at 0.1 m precision with
    round-to-nearest, which would otherwise put up to half the edges up to
    5 cm below their computed crest.
    """
    ocean_mask = landuse_on_grid == LANDUSE_SEA

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

    # A cell discarded from EITHER side would otherwise become "neither
    # land_mask nor water_like" -- invisible to the edge tracer
    # (h_break/v_break require one side True and the other True; a
    # "neither" cell reads False on both, so no segment is drawn against
    # it from either direction). Where a discarded component sits AT the
    # true edge of a larger, still-kept feature (not fully isolated deep
    # inside the opposite class), that doesn't just skip a ring around the
    # excluded feature itself -- it also breaks the LARGER feature's own
    # boundary trace at exactly that point, leaving a dangling gap in what
    # should still be a continuous coastline/riverbank (confirmed on basin
    # 4267691: several dangling endpoints traced back to exactly this,
    # for genuinely small, isolated islands/patches -- distinct from the
    # diagonal-connectivity artifact discard_small_components' own
    # 8-connectivity already guards against). Reclassifying a discarded
    # cell to the OPPOSITE class instead -- a too-small land island
    # dissolves into the surrounding water_like, a too-small water patch
    # dissolves into the surrounding land -- keeps land_mask/water_like
    # exact complements again, so the tracer sees a genuine, continuous
    # transition at the larger feature's own true edge (or nothing at
    # all, where the discarded feature was fully isolated) instead of a
    # hole either way. This is exactly what "no ring around a small
    # excluded feature" already meant to achieve -- just implemented
    # without leaving an untraceable gap behind. land_mask_raw/
    # water_like_raw are mutually exclusive by construction (land_mask_raw
    # = ~water_like_raw & valid_mask), so the two reclassified sets can
    # never collide with each other or with the surviving mask on the
    # opposite side -- computed from the PRE-reclassification land_mask/
    # water_like explicitly (not chained), so reordering these two lines
    # later can't silently change which cells get reclassified.
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

    # ── protected ocean pockets ────────────────────────────────────────────
    # Two distinct sources of "still landuse==200 despite the weir already
    # deciding otherwise":
    #  (a) GRID-RESOLUTION MISALIGNMENT (the dominant, coastline-wide case):
    #      the weir is traced on landuse_on_grid -- the native landuse.tif
    #      (~30 m) resampled via NEAREST-NEIGHBOR onto the much coarser
    #      SFINCS regular grid (~70 m, rule grid_align_landuse, 09b). That
    #      resampling necessarily generalizes the true coastline to the coarse grid's
    #      own cell edges, so land_mask (the model's own "this is the
    #      protected/dry side" decision, at grid resolution) and the fine
    #      native landuse.tif disagree right along the coast -- overlaying
    #      the weir on the native-resolution raster shows native "sea"
    #      pixels sitting on the land side, continuously, not as isolated
    #      pockets (confirmed: basin 2433835's own landuse.tif/sea_mask.tif
    #      have exactly ONE connected sea component at native resolution --
    #      there is no disconnected "pocket" to find via component analysis
    #      at all). land_mask (final, post small-component reclassification
    #      below) IS the ground truth for "the weir-building algorithm
    #      considers this land" -- trusting it directly, rather than
    #      re-deriving connectivity, is what actually fixes the overlap:
    #      correct_sea_mask_from_grid_mask (src/raster.py, called from
    #      10_depth_estimation_modelled.py) reprojects this mask back onto
    #      NATIVE resolution and only touches pixels that are STILL
    #      landuse==200 there, so this naturally scopes down to exactly the
    #      fringe of mismatch pixels along the coast (an ordinary inland
    #      cell is never landuse==200 in the first place).
    #  (b) a separately-ringed pocket: an ocean_mask component large enough
    #      to survive into water_like but NOT the main open ocean (the
    #      SINGLE LARGEST water_like component, same convention already
    #      used for zsini's own isolated-sea-pocket fix in
    #      13_build_sfincs_skeleton.py) -- the weir walls it off with its
    #      OWN ring, so it's no longer open sea either, despite staying
    #      classified water_like (not land_mask) throughout.
    # Returned so callers can clear these cells from sea to land in
    # sea_mask.tif (see src.raster.correct_sea_mask_from_grid_mask) --
    # both for zsini (13_build_sfincs_skeleton.py) and for flood-diagnostic
    # masking downstream (src.postprocessing reads the same corrected file
    # directly, since sea_mask already encodes exactly the "is this
    # genuinely open sea" boolean those functions need).
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
    # protected_pocket_mask itself (land_mask | separately_ringed_pocket) is
    # grid-resolution and deliberately broad (essentially the whole
    # protected-side footprint) -- the actual number of pixels this affects
    # is only known once it's reprojected onto native landuse.tif resolution
    # and intersected with landuse==200 there (see relabel_landuse_from_grid_
    # mask's own return value / 10_depth_estimation_modelled.py's log line),
    # so no cell count is logged for it here.

    n_final = int(land_mask.sum())

    # Restrict the closure exemption to cells that SURVIVED small-component
    # discarding above, not raw ocean_mask directly -- a small, disconnected
    # ocean-classified patch too small to be a real coastline (exactly what
    # discard_small_components already drops from water_like) would
    # otherwise still count as "exempt" even though it's no longer part of
    # water_like at all, stranding it as neither land nor water and
    # breaking continuity against its forced-land neighbour.
    ocean_exempt = water_like & ocean_mask

    crest_surface = None
    edge_water_side_mask = None
    if water_side_crest_on_grid is None:
        weir_lines = grid.seaward_edges(
            land_mask, water_like, valid_mask=grid.valid_mask, ocean_exempt=ocean_exempt
        )
        weir_crests: list | float = float(
            ceil_water_level(crest_elevation_m + freeboard_m)
        )
        log.info(
            f"Weir line: {len(weir_lines)} segment(s) extracted directly at the coast "
            f"({n_final:,} land cells), crest={weir_crests:+.2f} m (floor "
            f"{crest_elevation_m:+.3f} m +{freeboard_m:.2f} m freeboard, rounded up to 0.1 m) "
            f"-- empirical mode, no riverbank weir"
        )
    else:
        # np.fmax (not np.maximum): a NaN water-side cell (never wetted)
        # falls back to crest_elevation_m, so crest_surface is finite
        # everywhere. That matters beyond just picking a sensible crest:
        # seaward_edges_with_values DROPS a segment wherever its value is
        # NaN, which would otherwise leave a dangling gap in an otherwise
        # continuous coastline/riverbank (confirmed via basin 2433835: 38
        # dangling endpoints traced back to exactly this NaN-drop path).
        # float64 so the rounded-up 0.1 m values stay exact in the gpkg.
        crest_surface = ceil_water_level(
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
            f"every crest rounded up to 0.1 m)"
        )

    weir_gdf = build_weir_geodataframe(weir_lines, weir_crests, weir_par1, grid.crs)

    # Vector-space cleanup pass: a small feature attached to a larger, kept
    # component at a single diagonal-pinch pixel survives discard_small_
    # components (raster level, above) since it's part of one large,
    # correctly-kept component there -- but still traces its own tiny
    # closed ring in this vector output, hanging off the main boundary at
    # one 4-way pinch vertex. See _remove_small_dikerings' own docstring.
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
        # Per-edge crest bookkeeping (None in the ocean-only branch):
        # crest_surface is the crest every edge looked up on its water side,
        # edge_water_side_mask marks which water_like cells were those
        # water sides -- lets a caller compare a later run's water level
        # against exactly the crest each edge was given.
        "crest_surface": crest_surface,
        "edge_water_side_mask": edge_water_side_mask,
    }
    return weir_gdf, diagnostics
