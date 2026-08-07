"""
protection_weir.py — Builds a coastal/riverbank protection weir directly on
the SFINCS model's regular grid.

The weir represents the protection-level standard itself: flood defenses
are assumed to run continuously along the whole coast and river network (not
just near named reaches), so no flooding should occur below the crest
elevation anywhere in the domain until it's overtopped. It hugs the
coast/riverbank directly -- no inland offset.

Also used, unmodified, by rule modelled_depth_estimation (10,
10_depth_estimation_modelled.py) to build its own disposable calibration
model's weir -- coast and river channel merged into the same water_like
boundary there too, so calibration and production build conceptually the
same single weir set. That rule calls it with BOTH crest_elevation_m and
river_crest_on_grid set to the same artificially high confinement value
(e.g. 1000 m) -- not the real coastal crest: even though calibration's own
boundary is flat and guaranteed dry, the confined river's water level can
still exceed a low real coastal crest right at the coast/river transition
near the mouth and leak onto the floodplain through the coastal side of the
same merged boundary, so calibration confines EVERYTHING, not just the
river. Rule 13 passes the real per-reach weir_crest_calibrated
(river_crest_on_grid) once calibration has finished, maxed against the
real, datum-corrected crest_elevation_m -- production always uses real
crests on both sides; the uniform-confinement simplification is
calibration-only.

Design summary
--------------
1. Classify ocean (landuse==200) and river-channel cells/faces on the model
   grid into one combined ``water_like`` set; everything else valid is
   ``land``. landuse==80 (inland water/tidal flat) is not specially
   classified -- it's just land unless it happens to coincide with the
   channel mask.
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
from affine import Affine
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.ndimage import label as ndimage_label
from shapely.geometry import LineString, Polygon
from shapely.ops import linemerge, unary_union

log = logging.getLogger(__name__)

LANDUSE_SEA = 200


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
        one destination cell in only a handful of spots). See dilate_values
        for the equivalent operation on a real-valued array.
        """
        return binary_dilation(mask, structure=self._struct4, iterations=iterations)

    def dilate_values(self, value_grid: np.ndarray, iterations: int = 1) -> np.ndarray:
        """Propagate each finite cell's own value outward to nearby NaN
        cells via nearest-value fill, up to `iterations` cells away (same
        distance semantics as close_gaps, generalized from a boolean mask to
        a real-valued array).

        Needed because a per-reach value rasterized only onto a channel's
        own buffered footprint (e.g. river_crest_on_grid) otherwise has no
        overlap with the LAND cells the weir-crest lookup actually reads
        from once the channel mask itself has been widened by close_gaps --
        the value array must be widened by the same margin (plus one, to
        also cover the land ring just outside the now-wider water_like
        region) or every lookup misses and silently falls back to a
        default.
        """
        valid = np.isfinite(value_grid)
        if not valid.any():
            return value_grid.copy()
        from scipy.ndimage import distance_transform_edt

        dist, (idx_r, idx_c) = distance_transform_edt(~valid, return_indices=True)
        nearest_value = value_grid[idx_r, idx_c]
        return np.where(dist <= iterations, nearest_value, value_grid)

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
    ) -> tuple[list, list]:
        """Like seaward_edges, but returns (segments, values) UNMERGED --
        one LineString per grid edge/mesh edge, paired with the mask_a-side
        cell/face's own value_grid entry. Deliberately not linemerge'd:
        adjacent cells can carry different values (e.g. different
        calibrated river reaches), so segments must stay separable to keep
        a single crest value per output weir feature. Segments whose
        mask_a-side value is NaN are dropped entirely.

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
    ocean_exempt via edge-padding -- exactly the reasoning dilate_values/
    value_grid padding already use elsewhere in this module, now applied to
    "is the edge genuinely open water" instead of a crest value. A channel
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
) -> tuple[list, list]:
    mask_a, water_like, transform = _pad_for_edge_tracing(
        mask_a, water_like, valid_mask, ocean_exempt, transform
    )
    # value_grid padded with mode="edge" -- each padding cell inherits its
    # nearest real neighbour's own value, so a newly-closed edge segment
    # gets a sensible crest with no extra lookup logic (matches how
    # dilate_values already propagates real values outward elsewhere in
    # this module).
    value_grid = np.pad(value_grid, 1, mode="edge")
    segments, values = [], []
    h_break = (mask_a[:-1, :] & water_like[1:, :]) | (
        water_like[:-1, :] & mask_a[1:, :]
    )
    for r, c in zip(*np.where(h_break)):
        land_r = r if mask_a[r, c] else r + 1
        v = value_grid[land_r, c]
        if np.isnan(v):
            continue
        x0, y0 = transform * (c, r + 1)
        x1, y1 = transform * (c + 1, r + 1)
        segments.append(LineString([(x0, y0), (x1, y1)]))
        values.append(float(v))
    v_break = (mask_a[:, :-1] & water_like[:, 1:]) | (
        water_like[:, :-1] & mask_a[:, 1:]
    )
    for r, c in zip(*np.where(v_break)):
        land_c = c if mask_a[r, c] else c + 1
        v = value_grid[r, land_c]
        if np.isnan(v):
            continue
        x0, y0 = transform * (c + 1, r)
        x1, y1 = transform * (c + 1, r + 1)
        segments.append(LineString([(x0, y0), (x1, y1)]))
        values.append(float(v))
    return segments, values


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


def build_coastal_protection_weir(
    grid: GridArrays,
    landuse_on_grid: np.ndarray,
    river_channel_mask: np.ndarray,
    crest_elevation_m: float,
    min_component_cells: int,
    weir_par1: float,
    river_crest_on_grid: np.ndarray | None = None,
    freeboard_m: float = 0.0,
    channel_mask_gap_free: bool = False,
    river_crest_dilation_cells: int = 1,
    coastal_crest_on_grid: np.ndarray | None = None,
) -> tuple[gpd.GeoDataFrame, dict]:
    """
    Top-level orchestrator: classify ocean(+river, if modelled) as
    water_like -> discard small isolated components on both the land and
    water_like side -> weir line extraction directly at the resulting
    boundary. Returns (weir_gdf, diagnostics) where diagnostics carries
    everything needed for logging and the diagnostic plot.

    Args:
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
        river_crest_on_grid: Optional per-cell calibrated riverbank weir
            crest (NaN where not covered by a modelled reach), from
            river_processing.depth_method == "modelled" (rule
            modelled_depth_estimation's weir_crest_calibrated column,
            rasterized -- with smoothing
            across reach junctions so two adjacent reaches' calibrated
            crests don't meet at a hard step -- via
            src.river_burn.build_smoothed_weir_crest_regular).
            When provided: the river channel participates in water_like,
            and each resulting weir segment gets its own crest --
            river-covered land cells use max(crest_elevation_m,
            river_crest_on_grid) (never protect less than the uniform
            coastal standard implies, e.g. at a river-mouth spit adjacent
            to both), other land cells use the uniform crest_elevation_m.
            When None (depth_method == "empirical" -- there is no modelled
            water level to derive a riverbank dike from): the river channel
            does NOT participate in water_like at all, so no riverbank weir
            is built and rivers interact with their floodplain naturally;
            only the coastal levee is built, with the uniform-crest
            behavior.
        freeboard_m: Added uniformly on top of the final crest (both the
            uniform-coastal and the max(coastal, river) case), after any
            combination logic above -- a safety margin absorbing sources of
            under-protection the crest computation itself doesn't capture
            (subgrid/discretization roundoff, the calibration's steady-
            state assumption, etc.), so a design event landing almost
            exactly at the calibrated/coastal crest doesn't overtop it.
            0.0 (the default) applies no freeboard.
        river_crest_dilation_cells: How far (in cells) river_crest_on_grid's
            per-reach calibrated values are propagated onto surrounding land
            before the crest lookup, on top of whatever the mask's own
            dilation already covers -- see the dilate_values call below for
            why any propagation at all is needed. A radius of 1 leaves a
            salt-and-pepper gap right at reach junctions/transitions (e.g.
            near a river mouth): individual land cells at the edge of a
            reach's own 1-cell-wide margin fall inside or outside it
            depending on small geometric noise, so some silently drop to
            the flat crest_elevation_m floor instead of inheriting that
            reach's real, usually higher, calibrated crest. A wider radius
            (e.g. 5) closes most of that gap; it only changes which value
            nearby land cells inherit, not the weir's own traced position
            (channel_mask_gap_free already controls that independently), so
            it cannot reintroduce a weir-to-DEM offset.
        coastal_crest_on_grid: Optional per-cell "elsewhere" crest (same
            shape as landuse_on_grid), replacing the flat crest_elevation_m
            scalar wherever finite -- from rule modelled_depth_estimation's
            own correction rounds,
            where ocean cells near the coast can show a period-max water
            level above baseline_m (backwater from the river reaching the
            coast) that the flat coastal standard alone doesn't cover, even
            just beyond river_crest_on_grid's own dilation radius. Combined
            via the SAME np.maximum as river_crest_on_grid -- never LOWERS
            protection versus the flat scalar, only raises it locally where
            the data says so. None (default) uses the flat crest_elevation_m
            scalar everywhere; wherever this array itself is NaN (a land
            cell beyond its own probe/dilation coverage), also falls back
            to crest_elevation_m rather than leaving a gap -- crest_surface
            is guaranteed finite at every land cell so the traced weir
            never drops a segment for a missing value (see the fallback
            just below).
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

    if river_crest_on_grid is None:
        water_like_raw = ocean_mask
    else:
        _channel_gap_iterations = 0 if channel_mask_gap_free else 1
        if _channel_gap_iterations > 0:
            river_channel_mask = grid.close_gaps(
                river_channel_mask, iterations=_channel_gap_iterations
            )
        water_like_raw = ocean_mask | river_channel_mask
        # river_crest_on_grid was rasterized onto the SAME (undilated) buffered
        # channel footprint river_channel_mask had before the close_gaps() call
        # above -- but the crest lookup in seaward_edges_with_values() reads
        # the LAND-side cell of the land/water_like boundary, which sits
        # _channel_gap_iterations + river_crest_dilation_cells cells away from
        # that original footprint (river_crest_dilation_cells more than the
        # mask's own dilation -- or, when channel_mask_gap_free and
        # _channel_gap_iterations is 0, just river_crest_dilation_cells alone
        # -- since land is one further ring beyond water_like regardless of
        # whether the mask itself was additionally dilated). Without also
        # propagating the value array outward by at least this margin, every
        # land-side lookup misses (always NaN) and crest_surface below
        # silently falls back to the flat coastal crest_elevation_m
        # everywhere, never any calibrated per-reach value.
        # river_crest_dilation_cells=1 (still leaves a salt-and-pepper gap
        # at reach junctions -- see this function's own docstring) is only
        # the FLOOR; callers pass a wider radius.
        river_crest_on_grid = grid.dilate_values(
            river_crest_on_grid,
            iterations=_channel_gap_iterations + river_crest_dilation_cells,
        )

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

    if river_crest_on_grid is None:
        weir_lines = grid.seaward_edges(
            land_mask, water_like, valid_mask=grid.valid_mask, ocean_exempt=ocean_exempt
        )
        weir_crests: list | float = crest_elevation_m + freeboard_m
        log.info(
            f"Weir line: {len(weir_lines)} segment(s) extracted directly at the coast "
            f"({n_final:,} land cells), crest={crest_elevation_m:+.2f} m "
            f"(+{freeboard_m:.2f} m freeboard) -- empirical mode, no riverbank weir"
        )
    else:
        # coastal_crest_on_grid can have NaN gaps beyond its own probe/
        # dilation coverage (e.g. a land cell adjacent to a small isolated
        # water patch that's neither a modelled river reach nor near a
        # coastal probe). Falling back to the flat crest_elevation_m there
        # -- exactly what "no per-cell override available" already means
        # everywhere else this array doesn't cover -- guarantees
        # elsewhere_crest, and therefore crest_surface below, is finite at
        # every land cell. That matters beyond just picking a sensible
        # crest value: seaward_edges_with_values DROPS a weir segment
        # entirely wherever its land-side crest_surface value is NaN, which
        # otherwise breaks an intended requirement of the traced weir --
        # every land/water_like boundary cell gets a segment, so the result
        # is either continuous chains running boundary-to-boundary or fully
        # closed loops, never a dangling gap in the middle of an otherwise
        # continuous coastline/riverbank (confirmed via basin 2433835: 38
        # dangling endpoints traced back to exactly this NaN-drop path).
        if coastal_crest_on_grid is not None:
            elsewhere_crest = np.where(
                np.isfinite(coastal_crest_on_grid),
                coastal_crest_on_grid,
                np.float32(crest_elevation_m),
            )
        else:
            elsewhere_crest = np.float32(crest_elevation_m)
        crest_surface = np.where(
            np.isfinite(river_crest_on_grid),
            np.maximum(elsewhere_crest, river_crest_on_grid),
            elsewhere_crest,
        ) + np.float32(freeboard_m)
        weir_lines, weir_crests = grid.seaward_edges_with_values(
            land_mask,
            water_like,
            crest_surface,
            valid_mask=grid.valid_mask,
            ocean_exempt=ocean_exempt,
        )
        n_river_crest = (
            int(np.isfinite(river_crest_on_grid[land_mask]).sum()) if n_final else 0
        )
        log.info(
            f"Weir line: {len(weir_lines)} segment(s) extracted directly at the coast/riverbank "
            f"({n_final:,} land cells, {n_river_crest:,} with a modelled riverbank crest), "
            f"coastal crest={crest_elevation_m:+.2f} m (+{freeboard_m:.2f} m freeboard)"
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
    }
    return weir_gdf, diagnostics
