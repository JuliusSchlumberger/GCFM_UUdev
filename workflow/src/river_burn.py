"""
river_burn.py — Burns a per-cell river-bed anchor profile (zbed_anchors,
built directly in-memory by rule empirical_depth_estimation or
modelled_depth_estimation -- never a standalone file)
directly into a channel-only DEM at native (fine) resolution, instead of
hydromt_sfincs's own per-tile burn_river_rect, which produces a wavy,
non-monotonic burned bed when re-run on an already-burned raster.

Output covers only the buffered river-channel network (NaN elsewhere),
cropped to the river network's own bounding box rather than the whole basin
domain — meant to be fed to hydromt_sfincs as a higher-priority
elevation_list entry ahead of elevation_merged/elevation_conditioned (which
remains the fallback for everywhere else: ocean, floodplain, gaps).

Processes each reach independently, using ONLY that reach's own
zbed_anchors points projected onto that reach's own full centerline
geometry — unlike burn_river_rect, which clips the centerline per subgrid
tile but matches it against the GLOBAL, un-clipped zbed_anchors (no distance
cutoff), causing spurious cross-tile contamination in the burned bed level.

Also provides build_channel_mask_regular(): a much lighter sibling that
just rasterizes each reach's own buffered channel polygon directly onto an
arbitrary target grid (e.g. the SFINCS model grid, not the DEM working
grid) — no elevation interpolation, zbed_anchors, or natural-DEM reference
needed, since this only ever produces a boolean "is this cell part of the
channel" mask (used by the coastal protection weir, src/protection_weir.py).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_origin
from rasterio.windows import Window, transform as window_transform
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

from src.raster import reproject_nan_aware
from src.river_network import (
    _as_linestring,
    _BOUNDARY_BLEND_EPS_M,
    _junction_value,
    build_downstream_adjacency,
    normalize_reach_id,
)

log = logging.getLogger(__name__)


def _windows_intersect(a: Window, b: Window) -> bool:
    return not (
        a.col_off + a.width <= b.col_off
        or b.col_off + b.width <= a.col_off
        or a.row_off + a.height <= b.row_off
        or b.row_off + b.height <= a.row_off
    )


def burn_river_channel(
    rivers: gpd.GeoDataFrame,
    zbed_anchors: gpd.GeoDataFrame,
    natural_dem_path: str | Path,
    utm_crs,
    resolution_m: float,
    width_column: str = "width",
    margin_m: float = 500.0,
    out_transform=None,
    out_shape: tuple[int, int] | None = None,
    channel_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, object, float, dict]:
    """
    Burn zbed_anchors' rivbed profile into a channel-only DEM.

    Args:
        rivers:         river_network_depth_estimated.gpkg (any CRS) — needs
                        'reach_id', width_column, geometry.
        zbed_anchors:   Per-pixel/per-cell bed anchor points (any CRS) --
                        needs 'reach_id', 'rivbed', geometry. Built directly
                        by rule empirical_depth_estimation
                        (compute_river_bed_points, from the empirical
                        rivdph column) or rule modelled_depth_estimation
                        (per-cell, from its own calibrated depth) -- not a
                        separate rule's own output.
        natural_dem_path: Path to the basin's own conditioned/merged DEM
                        (elevation_conditioned.tif or elevation_merged.tif --
                        FathomDEM merged with GEBCO bathymetry, already
                        GOCO06s-referenced, matching zbed_anchors' own
                        rivbed values). Used both as this function's own
                        along-channel terrain reference and as the floor
                        below (burned bed is never raised ABOVE it). Must be
                        the merged/conditioned DEM, not raw FathomDEM: raw
                        FathomDEM alone doesn't see riverbed/bathymetry, so
                        it can't distinguish a genuinely deep, GEBCO-informed
                        stretch (e.g. near a river mouth straddling the
                        FathomDEM/GEBCO merge boundary) from ordinary land,
                        which can let a shallower, single-anchor-interpolated
                        burn overwrite real bathymetry with a shallower
                        value.
        utm_crs:        target CRS for the output raster (the basin's own
                        working UTM CRS).
        resolution_m:   output pixel size (m) -- ignored (derived from
                        out_transform instead) when out_transform/out_shape
                        are given.
        width_column:   channel width column in `rivers` (always 'width').
        margin_m:       buffer (m) added around the river network's own
                        bounds before fetching FathomDEM tiles, so buffers
                        near the network's own extent edge still get real
                        DEM coverage -- ignored when out_transform/out_shape
                        are given (the caller's grid already has whatever
                        extent it needs).
        out_transform, out_shape: Optional externally-supplied target grid
                        (e.g. the shared SFINCS grid, rule 08c's
                        {basin_id}_sfincs_grid.json) -- when BOTH given,
                        burns directly onto this exact grid instead of
                        self-computing a native-resolution one from the
                        river network's own bounds. The algorithm itself is
                        resolution-agnostic (only evaluates each reach's own
                        along-channel profile at whatever pixel centers it's
                        given), so this is the same burn, just coarser --
                        used to eliminate the reprojection gap that opens up
                        when a separately-computed, native-resolution burn
                        is later resampled onto the (coarser) SFINCS grid by
                        HydroMT's own elevation.create() merge. Must supply
                        both or neither.
        channel_mask:   Optional boolean array, shape == out_shape, ONLY
                        valid together with out_transform/out_shape (raises
                        otherwise). When given, each reach's own buffer
                        polygon is no longer independently rasterized to
                        decide which pixels to burn -- the corresponding
                        window of channel_mask is used directly instead, so
                        the burned/excavated cells are EXACTLY channel_mask's
                        cells, not a second, independently-computed polygon
                        rasterization that merely tends to agree with it.
                        Callers should build channel_mask via
                        build_channel_mask_regular() with the SAME rivers/
                        width_column/out_shape/out_transform, so calibration's
                        own confinement corridor, this excavation, and
                        production's weir corridor are all provably the same
                        set of cells (see 11b_burn_river_dem.py).

    Each reach's own interpolation is extended with one synthetic anchor at
    along=0 and/or along=length, valued from its immediate upstream/
    downstream neighbour's own nearest-anchor rivbed (averaged across
    multiple neighbours at a confluence/bifurcation) -- both reaches sharing
    a junction then interpolate through the same boundary value rather than
    independently clamping to their own nearest anchor, without reaching
    any further than direct neighbours.

    Returns:
        (burned_arr, transform, nodata, stats) — burned_arr is float32, NaN
        outside every reach's own channel buffer; stats is a dict with
        'n_reaches_burned', 'n_reaches_skipped', 'n_pixels_burned'.
    """
    if (out_transform is None) != (out_shape is None):
        raise ValueError(
            "out_transform and out_shape must be given together or not at all"
        )
    if channel_mask is not None and out_transform is None:
        raise ValueError(
            "channel_mask requires out_transform/out_shape (burns onto a known grid)"
        )
    if channel_mask is not None and channel_mask.shape != out_shape:
        raise ValueError(
            f"channel_mask shape {channel_mask.shape} != out_shape {out_shape}"
        )

    rivers_proj = rivers.to_crs(utm_crs) if rivers.crs != utm_crs else rivers.copy()
    zbed_proj = (
        zbed_anchors.to_crs(utm_crs)
        if zbed_anchors.crs != utm_crs
        else zbed_anchors.copy()
    )
    zbed_proj = zbed_proj.assign(
        _reach_id=[normalize_reach_id(x) for x in zbed_proj["reach_id"]]
    )

    # ── output grid: either the externally-supplied one (verbatim), or
    # self-computed from the river network's own bounds + margin, snapped to
    # a resolution_m lattice ──────────────────────────────────────────────────
    if out_transform is not None:
        height_px, width_px = out_shape
        resolution_m = abs(out_transform.a)
        # Corner-based bounds (not rasterio.transform.array_bounds, which
        # assumes a north-up/negative-e transform) -- an externally-supplied
        # transform, e.g. a live SFINCS model's own grid, can have a
        # POSITIVE y-scale, which array_bounds would silently mislabel
        # min/max for.
        corners = [(0, 0), (width_px, 0), (0, height_px), (width_px, height_px)]
        xs, ys = zip(*(out_transform * c for c in corners))
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        log.info(
            f"Burn grid: {width_px}x{height_px} px @ {resolution_m} m "
            f"(externally-supplied SFINCS grid)"
        )
    else:
        xmin, ymin, xmax, ymax = rivers_proj.total_bounds
        xmin, ymin, xmax, ymax = (
            xmin - margin_m,
            ymin - margin_m,
            xmax + margin_m,
            ymax + margin_m,
        )
        width_px = max(1, math.ceil((xmax - xmin) / resolution_m))
        height_px = max(1, math.ceil((ymax - ymin) / resolution_m))
        out_transform = from_origin(xmin, ymax, resolution_m, resolution_m)
        log.info(
            f"Burn grid: {width_px}x{height_px} px @ {resolution_m} m "
            f"(river network bounds + {margin_m:.0f} m margin)"
        )

    # ── natural (conditioned/merged) DEM, cropped to the burn grid -- this
    # function's own along-channel terrain reference AND the floor below.
    # Already GOCO06s-referenced, matching zbed_anchors' own rivbed values,
    # so no separate geoid correction is needed here ───────────────────────
    with rasterio.open(natural_dem_path) as src:
        natural_arr = src.read(1).astype(np.float32)
        natural_nodata = src.nodata
        natural_src_crs = src.crs
        natural_src_transform = src.transform
    if natural_nodata is not None:
        natural_arr[natural_arr == np.float32(natural_nodata)] = np.nan
    terrain = reproject_nan_aware(
        natural_arr,
        natural_src_transform,
        natural_src_crs,
        (height_px, width_px),
        out_transform,
        utm_crs,
        resampling=Resampling.bilinear,
    )
    log.info(
        f"Natural DEM (river-network extent): {(~np.isnan(terrain)).sum():,} valid px"
    )

    # ── precompute each reach's own (line, along, rivbed_vals), and topology ──
    # Done for every reach with valid geometry + zbed points (not just ones
    # that will actually be burned below, e.g. zero-width reaches) so a
    # burnable reach can still borrow a boundary value from a neighbour that
    # itself won't be burned.
    reach_data: dict[str, tuple[object, np.ndarray, np.ndarray]] = {}
    for row in rivers_proj.itertuples(index=False):
        rid = normalize_reach_id(row.reach_id)
        line = _as_linestring(row.geometry)
        if rid is None or line is None or line.length == 0:
            continue
        zbed_reach = zbed_proj[zbed_proj["_reach_id"] == rid]
        if len(zbed_reach) == 0:
            continue
        zbed_points = shapely.points(
            zbed_reach.geometry.x.to_numpy(), zbed_reach.geometry.y.to_numpy()
        )
        along = shapely.line_locate_point(line, zbed_points)
        rivbed_vals = zbed_reach["rivbed"].to_numpy(dtype=float)
        order = np.argsort(along)
        reach_data[rid] = (line, along[order], rivbed_vals[order])

    downstream_adj = build_downstream_adjacency(rivers_proj)
    upstream_adj: dict[str, list[str]] = {rid: [] for rid in downstream_adj}
    for rid, dns in downstream_adj.items():
        for dn in dns:
            upstream_adj.setdefault(dn, []).append(rid)

    # ── burn, one reach at a time ─────────────────────────────────────────────
    output = np.full((height_px, width_px), np.nan, dtype=np.float32)
    full_window = Window(0, 0, width_px, height_px)
    inv_out_transform = ~out_transform
    n_reaches_burned = 0
    n_reaches_skipped = 0
    n_pixels_burned = 0
    n_boundaries_blended = 0

    for row in rivers_proj.itertuples(index=False):
        rid = normalize_reach_id(row.reach_id)
        width = getattr(row, width_column, np.nan)
        if rid is None or pd.isna(width) or width <= 0 or rid not in reach_data:
            n_reaches_skipped += 1
            continue
        line, along, rivbed_vals = reach_data[rid]

        # Borrow a shared junction value from each immediate neighbour (never
        # reaching further than one hop) so this reach's interpolation
        # passes through the SAME value at each junction as whatever's on
        # the other side of it, instead of independently clamping to its
        # own nearest anchor -- see this function's docstring and
        # _junction_value's. At a confluence/bifurcation (multiple
        # neighbours), average across the junction values with each.
        up_vals = [
            v
            for u in upstream_adj.get(rid, [])
            if (v := _junction_value(reach_data, u, rid)) is not None
        ]
        dn_vals = [
            v
            for d in downstream_adj.get(rid, [])
            if (v := _junction_value(reach_data, rid, d)) is not None
        ]
        up_val = float(np.mean(up_vals)) if up_vals else None
        dn_val = float(np.mean(dn_vals)) if dn_vals else None
        ext_along, ext_rivbed = list(along), list(rivbed_vals)
        if up_val is not None and along[0] > _BOUNDARY_BLEND_EPS_M:
            ext_along.insert(0, 0.0)
            ext_rivbed.insert(0, up_val)
            n_boundaries_blended += 1
        if dn_val is not None and (line.length - along[-1]) > _BOUNDARY_BLEND_EPS_M:
            ext_along.append(line.length)
            ext_rivbed.append(dn_val)
            n_boundaries_blended += 1
        along, rivbed_vals = np.asarray(ext_along), np.asarray(ext_rivbed)

        if len(along) == 1:
            _const = rivbed_vals[0]

            def interp(x, _v=_const):
                return np.full(np.shape(x), _v, dtype=float)
        else:
            # Clamp to the first/last anchor value beyond the sampled range
            # (now including any borrowed boundary value above), rather than
            # linearly extrapolating -- pixels near a reach's start/end
            # routinely fall just outside the range its own zbed_anchors
            # points span (anchors are sampled along the centerline, but the
            # buffered channel polygon extends slightly past the line's own
            # endpoints), and extrapolating the local slope out to those
            # pixels can overshoot significantly.
            interp = interp1d(
                along,
                rivbed_vals,
                kind="linear",
                bounds_error=False,
                fill_value=(rivbed_vals[0], rivbed_vals[-1]),
            )

        buf_poly = _flush_capped_buffer(
            line, float(width), clip_start=bool(getattr(row, "is_seed", False))
        )
        # Corner-based window (not rasterio.windows.from_bounds, which
        # assumes a north-up/negative-e transform and raises "Bounds and
        # transform are inconsistent" against a positive-y-scale transform,
        # e.g. a live SFINCS model's own grid) -- same approach as
        # build_smoothed_weir_crest_regular.
        bminx, bminy, bmaxx, bmaxy = buf_poly.bounds
        corners = [(bminx, bminy), (bmaxx, bminy), (bminx, bmaxy), (bmaxx, bmaxy)]
        cols_corners, rows_corners = zip(*(inv_out_transform * c for c in corners))
        col_off = int(np.floor(min(cols_corners)))
        row_off = int(np.floor(min(rows_corners)))
        col_stop = int(np.ceil(max(cols_corners)))
        row_stop = int(np.ceil(max(rows_corners)))
        window = Window(col_off, row_off, col_stop - col_off, row_stop - row_off)
        if not _windows_intersect(window, full_window):
            n_reaches_skipped += 1
            continue
        window = window.intersection(full_window)
        if window.width <= 0 or window.height <= 0:
            n_reaches_skipped += 1
            continue

        win_transform = window_transform(window, out_transform)
        win_shape = (int(window.height), int(window.width))
        if channel_mask is not None:
            # Slice the externally-supplied, shared mask directly -- the
            # burned/excavated cells are then EXACTLY channel_mask's cells
            # within this reach's own window, not a second, independently-
            # rasterized buf_poly that merely tends to agree with it (see
            # this function's channel_mask docstring).
            inside = channel_mask[
                window.row_off : window.row_off + win_shape[0],
                window.col_off : window.col_off + win_shape[1],
            ]
        else:
            # all_touched=True: matches build_channel_mask_regular/
            # build_smoothed_weir_crest_regular's convention for this SAME
            # buf_poly elsewhere in this file -- geometry_mask defaults to
            # False (a pixel only counts if its centre falls inside the
            # polygon), which would narrow the excavated footprint relative
            # to the channel_mask/weir corridor built from the identical
            # buf_poly.
            inside = geometry_mask(
                [buf_poly],
                out_shape=win_shape,
                transform=win_transform,
                invert=True,
                all_touched=True,
            )
        if not inside.any():
            n_reaches_skipped += 1
            continue

        rows_idx, cols_idx = np.where(inside)
        xs, ys = rasterio.transform.xy(win_transform, rows_idx, cols_idx)
        pts = shapely.points(np.asarray(xs), np.asarray(ys))
        pts_along = shapely.line_locate_point(line, pts)
        rivbed_at_pts = np.asarray(interp(pts_along), dtype=float)

        row_off, col_off = int(window.row_off), int(window.col_off)
        terrain_win = terrain[
            row_off : row_off + win_shape[0], col_off : col_off + win_shape[1]
        ]
        output_win = output[
            row_off : row_off + win_shape[0], col_off : col_off + win_shape[1]
        ]

        terrain_at_pts = terrain_win[rows_idx, cols_idx]
        has_terrain = np.isfinite(terrain_at_pts)
        burned_vals = np.where(
            has_terrain, np.minimum(terrain_at_pts, rivbed_at_pts), rivbed_at_pts
        )
        output_win[rows_idx, cols_idx] = burned_vals

        n_reaches_burned += 1
        n_pixels_burned += int(inside.sum())

    stats = {
        "n_reaches_burned": n_reaches_burned,
        "n_reaches_skipped": n_reaches_skipped,
        "n_pixels_burned": n_pixels_burned,
        "n_boundaries_blended": n_boundaries_blended,
    }
    log.info(
        f"Burned {n_reaches_burned} reach(es) ({n_reaches_skipped} skipped: "
        f"no width/zbed points/geometry), {n_pixels_burned:,} channel pixel(s), "
        f"{n_boundaries_blended} reach-boundary value(s) blended with a neighbour"
    )
    return output, out_transform, np.nan, stats


def constrain_to_coarse_channel_mask(
    burned_arr: np.ndarray,
    native_transform,
    utm_crs,
    channel_mask_coarse: np.ndarray,
    coarse_transform,
) -> np.ndarray:
    """Null out any native-resolution burned pixel whose own parent coarse
    cell falls outside channel_mask_coarse.

    burn_river_channel()'s native-resolution call (no channel_mask given)
    independently rasterizes each reach's own buf_poly in its own per-reach
    window -- unlike the coarse-grid call (channel_mask=... passed
    directly), whose excavated footprint is EXACTLY channel_mask's cells by
    construction (see burn_river_channel's own channel_mask docstring).
    Since the native file is layered onto HydroMT's elevation_list at
    higher priority than the coarse background, any native pixel that
    lands in a coarse cell channel_mask doesn't cover becomes a deep,
    unprotected pocket -- the weir's own crest floor there is never raised,
    since crest_surface is only elevated within channel_mask. Guarantees the
    native excavated footprint is always a SUBSET of the coarse
    weir-protection corridor, matching the coarse burn's own guarantee.
    """
    mask_on_native = (
        reproject_nan_aware(
            channel_mask_coarse.astype(np.float32),
            coarse_transform,
            utm_crs,
            burned_arr.shape,
            native_transform,
            utm_crs,
            resampling=Resampling.nearest,
        )
        > 0.5
    )
    return np.where(mask_on_native, burned_arr, np.nan)


def _channel_buffer_polygons(rivers: gpd.GeoDataFrame, width_column: str) -> list:
    """Each reach's own line.buffer(width/2) polygon -- the same channel
    geometry burn_river_channel() rasterizes, without any of the elevation-
    interpolation machinery. Skips reaches with missing/non-positive width
    or degenerate geometry, same as burn_river_channel().
    """
    polys = []
    for row in rivers.itertuples(index=False):
        width = getattr(row, width_column, np.nan)
        if pd.isna(width) or width <= 0:
            continue
        line = _as_linestring(row.geometry)
        if line is None or line.length == 0:
            continue
        polys.append(
            _flush_capped_buffer(
                line, float(width), clip_start=bool(getattr(row, "is_seed", False))
            )
        )
    return polys


def _smoothed_weir_crest_profiles(
    rivers: gpd.GeoDataFrame,
    crest_column: str,
    blend_distance_m: float,
    crest_anchors: gpd.GeoDataFrame | None = None,
) -> dict[str, tuple[object, float, Callable]]:
    """
    Per-reach (line, length, profile) for a calibrated weir crest smoothed
    across reach junctions, so two adjacent reaches with different
    calibrated crests no longer meet at a hard step.

    Only the LOWER-crest reach at a junction is modified: within
    blend_distance_m of that junction -- truncated to the reach's own
    length, so a short reach ramps over its whole length rather than
    reaching past its own extent -- its crest rises linearly from its own
    calibrated value up to the higher neighbour's crest, reaching that
    value exactly at the junction. The higher-crest reach is never
    modified -- it keeps its own crest all the way to the junction (the
    same "never protect less" principle build_coastal_protection_weir
    already applies between coastal and riverine crests, applied here
    across reach junctions instead). At a confluence/bifurcation with
    multiple neighbours at one junction, ramps toward the MAXIMUM crest
    among all of them.

    This blend only ever matters in the flat-per-reach-scalar mode
    (crest_anchors=None -- see build_coastal_protection_weir's own module
    docstring) -- two reaches sharing a junction can genuinely disagree there, since each
    carries a single reach-wide value. With crest_anchors given (every
    caller on the regular grid, including rule modelled_depth_estimation's
    own calibration round loop), two reaches sharing a junction cell are computed from
    the SAME physical location's own simulated data and are therefore
    already identical at that point BY CONSTRUCTION. The blend's own
    trigger condition (neighbour's boundary value > this reach's own) can
    never fire there, so it's skipped entirely in that mode rather than
    computed for nothing.

    Args:
        crest_anchors: Optional per-cell/multi-point crest values (columns
            'reach_id', 'crest', 'along_m' -- one row per calibration point
            along each reach's own centerline, e.g. from
            build_centerline_cells_regular). When given, each reach's OWN
            crest is linearly interpolated between its own anchors
            (clamped beyond the first/last, matching burn_river_channel's
            own along-reach convention) instead of being one flat value
            from `crest_column` -- junction blending then compares each
            reach's own boundary value (its interpolated crest at
            along=0/along=length) against its neighbours' own boundary
            values, rather than a single reach-wide scalar for both ends.
            When None (default), behaves EXACTLY as before: one flat crest
            per reach from `crest_column`, backward compatible with
            production's own per-reach-scalar callers (rule 13).

    Returns {reach_id: (line, length, profile)}, where
    profile(along_m) -> crest elevation(s) at along-channel distance(s)
    from the reach's own start, vectorized over a numpy array.
    """
    reach_line: dict[str, object] = {}
    reach_length: dict[str, float] = {}
    for row in rivers.itertuples(index=False):
        rid = normalize_reach_id(row.reach_id)
        if rid is None:
            continue
        line = _as_linestring(row.geometry)
        if line is None or line.length == 0:
            continue
        reach_line[rid] = line
        reach_length[rid] = line.length

    # own_interp[rid](along_m array) -> crest value(s); boundary_val[rid] =
    # (value at along=0, value at along=length) -- identical for the flat
    # scalar case, genuinely different endpoints for the anchors case.
    own_interp: dict[str, Callable] = {}
    boundary_val: dict[str, tuple[float, float]] = {}

    if crest_anchors is not None and len(crest_anchors) > 0:
        grouped = crest_anchors.assign(
            _reach_id=crest_anchors["reach_id"].apply(normalize_reach_id)
        ).groupby("_reach_id")
        for rid, group in grouped:
            if rid is None or rid not in reach_line:
                continue
            along = group["along_m"].to_numpy(dtype=float)
            crest = group["crest"].to_numpy(dtype=float)
            valid = np.isfinite(along) & np.isfinite(crest)
            if not valid.any():
                continue
            along, crest = along[valid], crest[valid]
            order = np.argsort(along)
            along, crest = along[order], crest[order]
            interp = interp1d(
                along,
                crest,
                kind="linear",
                bounds_error=False,
                fill_value=(crest[0], crest[-1]),
            )
            own_interp[rid] = interp
            length = reach_length[rid]
            boundary_val[rid] = (float(interp(0.0)), float(interp(length)))
    else:
        for row in rivers.itertuples(index=False):
            rid = normalize_reach_id(row.reach_id)
            if rid is None or rid not in reach_line:
                continue
            crest = getattr(row, crest_column, np.nan)
            if crest is None or not np.isfinite(crest):
                continue
            crest = float(crest)
            own_interp[rid] = lambda s, _v=crest: np.full(
                np.asarray(s, dtype=float).shape, _v
            )
            boundary_val[rid] = (crest, crest)

    anchors_mode = crest_anchors is not None and len(crest_anchors) > 0
    if anchors_mode:
        # Per-cell anchors already agree exactly at real junctions (see
        # this function's own docstring) -- no blend needed, each reach's
        # own interpolation is the final profile.
        return {
            rid: (reach_line[rid], reach_length[rid], interp)
            for rid, interp in own_interp.items()
        }

    downstream_adj = build_downstream_adjacency(rivers)
    upstream_adj: dict[str, list[str]] = {rid: [] for rid in downstream_adj}
    for rid, dns in downstream_adj.items():
        for dn in dns:
            upstream_adj.setdefault(dn, []).append(rid)

    def _neighbor_max(neighbor_ids: list[str], end: str) -> float | None:
        # end="end": each neighbour's OWN downstream/end boundary value
        # (they feed INTO this reach's start). end="start": each
        # neighbour's OWN upstream/start boundary value (this reach feeds
        # INTO their start).
        vals = [
            (boundary_val[n][1] if end == "end" else boundary_val[n][0])
            for n in neighbor_ids
            if n in boundary_val
        ]
        return max(vals) if vals else None

    profiles: dict[str, tuple[object, float, Callable]] = {}
    for rid, interp in own_interp.items():
        line = reach_line[rid]
        length = reach_length[rid]
        own_start, own_end = boundary_val[rid]
        start_max = _neighbor_max(upstream_adj.get(rid, []), end="end")
        end_max = _neighbor_max(downstream_adj.get(rid, []), end="start")
        blend = min(blend_distance_m, length) if length > 0 else 0.0

        def profile(
            s,
            _interp=interp,
            _length=length,
            _blend=blend,
            _own_start=own_start,
            _own_end=own_end,
            _start_max=start_max,
            _end_max=end_max,
        ):
            s = np.asarray(s, dtype=float)
            val = _interp(s)
            if _start_max is not None and _start_max > _own_start and _blend > 0:
                frac = np.clip(s / _blend, 0.0, 1.0)
                val = np.maximum(val, _start_max + frac * (_own_start - _start_max))
            if _end_max is not None and _end_max > _own_end and _blend > 0:
                frac = np.clip((_length - s) / _blend, 0.0, 1.0)
                val = np.maximum(val, _end_max + frac * (_own_end - _end_max))
            return val

        profiles[rid] = (line, length, profile)
    return profiles


def build_smoothed_weir_crest_regular(
    rivers: gpd.GeoDataFrame,
    width_column: str,
    crest_column: str,
    out_shape: tuple[int, int],
    out_transform,
    blend_distance_m: float = 1000.0,
    crest_anchors: gpd.GeoDataFrame | None = None,
) -> np.ndarray:
    """
    Per-cell calibrated weir crest, smoothed across reach junctions -- see
    _smoothed_weir_crest_profiles for the blending rule (including what
    `crest_anchors` does). Where two reaches' buffers legitimately overlap
    the same cell (e.g. parallel channels, not a junction), the HIGHER
    value wins rather than a plain last-reach-wins overwrite, consistent
    with never silently lowering an already-painted cell's protection.
    """
    profiles = _smoothed_weir_crest_profiles(
        rivers, crest_column, blend_distance_m, crest_anchors=crest_anchors
    )
    output = np.full(out_shape, np.nan, dtype=np.float32)
    if not profiles:
        return output
    # Inverse-transform the buffer's own bounding-box corners directly,
    # rather than rasterio.windows.from_bounds (used by burn_river_channel,
    # which always builds its own fresh, conventional north-up transform via
    # from_origin) -- an arbitrary caller-supplied transform, e.g. a live
    # SFINCS model's own sf.grid.data["dep"].raster.transform, can have a
    # POSITIVE y-scale (row increases northward, not the north-up/negative-
    # y-scale convention from_bounds assumes), which from_bounds rejects
    # outright. Using all 4 corners (not just the two bounds corners) keeps
    # this correct for a rotated transform too.
    inv_transform = ~out_transform

    for row in rivers.itertuples(index=False):
        rid = normalize_reach_id(row.reach_id)
        width = getattr(row, width_column, np.nan)
        if rid is None or rid not in profiles or pd.isna(width) or width <= 0:
            continue
        line, _length, profile = profiles[rid]

        buf_poly = _flush_capped_buffer(
            line, float(width), clip_start=bool(getattr(row, "is_seed", False))
        )
        bminx, bminy, bmaxx, bmaxy = buf_poly.bounds
        corners = [(bminx, bminy), (bmaxx, bminy), (bminx, bmaxy), (bmaxx, bmaxy)]
        cols_corners, rows_corners = zip(*(inv_transform * c for c in corners))
        col_off = max(0, int(np.floor(min(cols_corners))))
        row_off = max(0, int(np.floor(min(rows_corners))))
        col_stop = min(out_shape[1], int(np.ceil(max(cols_corners))))
        row_stop = min(out_shape[0], int(np.ceil(max(rows_corners))))
        if col_stop <= col_off or row_stop <= row_off:
            continue
        window = Window(col_off, row_off, col_stop - col_off, row_stop - row_off)

        win_transform = window_transform(window, out_transform)
        win_shape = (int(window.height), int(window.width))
        # all_touched=True: geometry_mask defaults to False (a cell only
        # counts if its centre falls inside the polygon), which would
        # narrow channel coverage at cell edges. Every gap here falls back
        # to the flat coastal crest instead of the real calibrated one in
        # build_coastal_protection_weir, so under-coverage would silently
        # weaken part of the dike rather than just leaving a visually
        # thinner line.
        inside = geometry_mask(
            [buf_poly],
            out_shape=win_shape,
            transform=win_transform,
            invert=True,
            all_touched=True,
        )
        if not inside.any():
            continue

        rows_idx, cols_idx = np.where(inside)
        xs, ys = rasterio.transform.xy(win_transform, rows_idx, cols_idx)
        pts = shapely.points(np.asarray(xs), np.asarray(ys))
        pts_along = shapely.line_locate_point(line, pts)
        vals = profile(pts_along)

        row_off, col_off = int(window.row_off), int(window.col_off)
        output_win = output[
            row_off : row_off + win_shape[0], col_off : col_off + win_shape[1]
        ]
        current = output_win[rows_idx, cols_idx]
        output_win[rows_idx, cols_idx] = np.where(
            np.isnan(current), vals, np.maximum(current, vals)
        )

    return output


def build_nearest_weir_crest_regular(
    rivers: gpd.GeoDataFrame,
    width_column: str,
    out_shape: tuple[int, int],
    out_transform,
    cell_gdf: gpd.GeoDataFrame,
    crest_values: np.ndarray,
) -> np.ndarray:
    """
    Per-cell calibrated weir crest, painted with NO along-reach
    interpolation and NO cross-reach junction blending -- every raster
    cell within a reach's own buffer is assigned its NEAREST centerline
    anchor's own crest value directly (nearest (x, y) match against every
    cell_gdf row, any reach), not a value interpolated between anchors
    along a smoothed profile. Kept as a SEPARATE function from
    build_smoothed_weir_crest_regular rather than a mode switch on it.

    Searching the FULL cell_gdf anchor set (not just the current reach's
    own anchors) rather than reusing per-reach along-line profiles also
    means junctions are handled automatically -- a cell near a confluence
    simply takes whichever nearby reach's anchor is physically closest, no
    separate blend_distance_m parameter needed.
    """
    output = np.full(out_shape, np.nan, dtype=np.float32)
    if cell_gdf.empty or len(crest_values) == 0:
        return output
    anchor_xy = cell_gdf[["x", "y"]].to_numpy()
    anchor_tree = cKDTree(anchor_xy)

    inv_transform = ~out_transform
    for row in rivers.itertuples(index=False):
        rid = normalize_reach_id(row.reach_id)
        width = getattr(row, width_column, np.nan)
        line = row.geometry
        if rid is None or line is None or line.is_empty or pd.isna(width) or width <= 0:
            continue

        buf_poly = _flush_capped_buffer(
            line, float(width), clip_start=bool(getattr(row, "is_seed", False))
        )
        bminx, bminy, bmaxx, bmaxy = buf_poly.bounds
        corners = [(bminx, bminy), (bmaxx, bminy), (bminx, bmaxy), (bmaxx, bmaxy)]
        cols_corners, rows_corners = zip(*(inv_transform * c for c in corners))
        col_off = max(0, int(np.floor(min(cols_corners))))
        row_off = max(0, int(np.floor(min(rows_corners))))
        col_stop = min(out_shape[1], int(np.ceil(max(cols_corners))))
        row_stop = min(out_shape[0], int(np.ceil(max(rows_corners))))
        if col_stop <= col_off or row_stop <= row_off:
            continue
        window = Window(col_off, row_off, col_stop - col_off, row_stop - row_off)

        win_transform = window_transform(window, out_transform)
        win_shape = (int(window.height), int(window.width))
        inside = geometry_mask(
            [buf_poly],
            out_shape=win_shape,
            transform=win_transform,
            invert=True,
            all_touched=True,
        )
        if not inside.any():
            continue

        rows_idx, cols_idx = np.where(inside)
        xs, ys = rasterio.transform.xy(win_transform, rows_idx, cols_idx)
        _dist, nearest_idx = anchor_tree.query(np.column_stack([xs, ys]))
        vals = crest_values[nearest_idx]

        row_off, col_off = int(window.row_off), int(window.col_off)
        output_win = output[
            row_off : row_off + win_shape[0], col_off : col_off + win_shape[1]
        ]
        current = output_win[rows_idx, cols_idx]
        output_win[rows_idx, cols_idx] = np.where(
            np.isnan(current), vals, np.maximum(current, vals)
        )

    return output


def build_channel_mask_regular(
    rivers: gpd.GeoDataFrame,
    width_column: str,
    out_shape: tuple[int, int],
    out_transform,
) -> np.ndarray:
    """
    Boolean river-channel mask built DIRECTLY at an arbitrary target
    resolution (the SFINCS model grid, typically far coarser than the DEM
    working grid burn_river_channel() operates at) -- NOT by burning at fine
    resolution and resampling down afterward. `all_touched=True` means a
    channel narrower than one destination cell still registers wherever it
    crosses that cell, avoiding the gaps that plain nearest-neighbour
    resampling of a fine-resolution channel mask onto a coarse grid would
    otherwise introduce (dropping narrow reaches entirely, pinching the
    channel shut).
    """
    polys = _channel_buffer_polygons(rivers, width_column)
    if not polys:
        return np.zeros(out_shape, dtype=bool)
    return rio_rasterize(
        [(p, 1) for p in polys],
        out_shape=out_shape,
        transform=out_transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    ).astype(bool)


def build_centerline_cells_regular(
    rivers: gpd.GeoDataFrame,
    out_shape: tuple[int, int],
    out_transform,
) -> gpd.GeoDataFrame:
    """
    Every grid cell each reach's own RAW centerline (not a buffered
    channel polygon) actually touches, at an arbitrary target resolution
    (typically the SFINCS model grid) -- the per-cell counterpart of
    build_channel_mask_regular's per-reach buffer, used to give every cell
    along the centerline its own calibrated depth/crest instead of one
    value for the whole reach (see 10_depth_estimation_modelled.py's own
    per-cell calibration).

    Reuses the same per-reach window + geometry_mask + rasterio.transform.xy
    + line_locate_point pattern already used identically in
    burn_river_channel and build_smoothed_weir_crest_regular, but
    rasterizes the bare LineString itself (all_touched=True) rather than a
    buffered polygon -- "every cell the line touches", not "every cell
    within half the channel width".

    A cell shared by two reaches (a junction) legitimately gets one row
    per reach -- rows are keyed (reach_id, row, col), not a global cell id,
    matching how zbed_anchors/crest anchors are grouped by reach_id
    everywhere else in this module.

    Returns:
        GeoDataFrame with columns 'reach_id', 'row', 'col', 'along_m'
        (distance from the reach's own centerline start), 'x', 'y',
        geometry (Point at the cell center, in `rivers`' own CRS), sorted
        by (reach_id, along_m). Reaches with missing/degenerate geometry
        are skipped (no row).
    """
    inv_transform = ~out_transform
    records: list[dict] = []

    for row in rivers.itertuples(index=False):
        rid = normalize_reach_id(row.reach_id)
        line = _as_linestring(row.geometry)
        if rid is None or line is None or line.length == 0:
            continue

        bminx, bminy, bmaxx, bmaxy = line.bounds
        corners = [(bminx, bminy), (bmaxx, bminy), (bminx, bmaxy), (bmaxx, bmaxy)]
        cols_corners, rows_corners = zip(*(inv_transform * c for c in corners))
        col_off = max(0, int(np.floor(min(cols_corners))))
        row_off = max(0, int(np.floor(min(rows_corners))))
        col_stop = min(out_shape[1], int(np.ceil(max(cols_corners))) + 1)
        row_stop = min(out_shape[0], int(np.ceil(max(rows_corners))) + 1)
        if col_stop <= col_off or row_stop <= row_off:
            continue
        window = Window(col_off, row_off, col_stop - col_off, row_stop - row_off)

        win_transform = window_transform(window, out_transform)
        win_shape = (int(window.height), int(window.width))
        inside = geometry_mask(
            [line],
            out_shape=win_shape,
            transform=win_transform,
            invert=True,
            all_touched=True,
        )
        if not inside.any():
            continue

        rows_idx, cols_idx = np.where(inside)
        xs, ys = rasterio.transform.xy(win_transform, rows_idx, cols_idx)
        pts = shapely.points(np.asarray(xs), np.asarray(ys))
        along = shapely.line_locate_point(line, pts)
        order = np.argsort(along)

        row_off_i, col_off_i = int(window.row_off), int(window.col_off)
        for i in order:
            records.append(
                {
                    "reach_id": rid,
                    "row": row_off_i + int(rows_idx[i]),
                    "col": col_off_i + int(cols_idx[i]),
                    "along_m": float(along[i]),
                    "x": float(xs[i]),
                    "y": float(ys[i]),
                    "geometry": pts[i],
                }
            )

    if not records:
        return gpd.GeoDataFrame(
            columns=["reach_id", "row", "col", "along_m", "x", "y", "geometry"],
            geometry="geometry",
            crs=rivers.crs,
        )
    return (
        gpd.GeoDataFrame(records, crs=rivers.crs)
        .sort_values(["reach_id", "along_m"])
        .reset_index(drop=True)
    )


def _half_plane_beyond(
    origin: tuple, away_point: tuple, size: float
) -> shapely.Polygon:
    """
    Large rectangle covering the half-plane on `away_point`'s side of the
    line through `origin` perpendicular to (away_point - origin) -- used by
    _flush_capped_buffer() to clip a round buffer end-cap back to a flat
    cut exactly at `origin`.
    """
    d = np.array(away_point) - np.array(origin)
    d = d / np.hypot(*d)
    n = np.array([-d[1], d[0]])
    o = np.array(origin)
    corners = [
        o + n * size,
        o - n * size,
        o - n * size + d * size,
        o + n * size + d * size,
    ]
    return shapely.Polygon(corners)


def _flush_capped_buffer(
    line: shapely.LineString,
    width: float,
    clip_start: bool = False,
    clip_end: bool = False,
) -> shapely.Polygon:
    """
    line.buffer(width/2), with the round end-cap at the start and/or end
    replaced by a flat cut exactly at that endpoint, perpendicular to the
    line's own local tangent there.

    Every reach's buffered channel footprint (channel_mask, the burn
    excavation corridor, the weir smoothing corridor) is built via a plain
    line.buffer(width/2), which defaults to a ROUND end cap at both ends --
    fine at an internal junction (a neighbouring reach's own buffer already
    overlaps and covers the join regardless of angle), but wrong at a
    network SEED (no upstream neighbour): the round cap bulges out in an
    arc beyond the line's own true start vertex, enclosing extra channel
    cells that are never sampled by build_centerline_cells_regular (which
    only follows the raw LINE, not the buffer) and so are invisible to
    per-cell calibration tracking, yet still sit inside the same confined
    pocket as the actual discharge-injection cell.

    Only clip_start/clip_end=True gets this treatment (typically
    clip_start for a reach with is_seed=True) -- every other reach keeps
    its plain round-capped buffer unchanged.

    Degenerate first/last segments (a duplicate leading/trailing
    coordinate) safely fall back to the unclipped round cap for that end,
    rather than raising on a zero-length tangent.
    """
    buf = line.buffer(width / 2.0)
    if not clip_start and not clip_end:
        return buf
    coords = list(line.coords)
    minx, miny, maxx, maxy = buf.bounds
    big = (maxx - minx) + (maxy - miny) + width  # safely larger than buf's own extent
    if clip_start and len(coords) >= 2 and coords[0] != coords[1]:
        buf = buf.intersection(_half_plane_beyond(coords[0], coords[1], big))
    if clip_end and len(coords) >= 2 and coords[-1] != coords[-2]:
        buf = buf.intersection(_half_plane_beyond(coords[-1], coords[-2], big))
    return buf


def snap_points_to_centerline_cells(
    points: gpd.GeoDataFrame,
    centerline_cells: gpd.GeoDataFrame,
    reach_ids: Sequence[str | None] | None = None,
    resolution_m: float | None = None,
) -> gpd.GeoDataFrame:
    """
    Snap each point in `points` onto the nearest cell the river network's own
    RAW centerline actually passes through (centerline_cells, from
    build_centerline_cells_regular) -- not just channel_mask (the buffered
    corridor half a channel-width wide), the exact cell(s) the reach
    LineString itself intersects.

    Motivation: neither hydromt_sfincs's discharge_points.create() nor this
    codebase's own point-wrangling (geometry.snap_points_into_region, which
    only nudges a point back inside the model's own active region) ever
    checks that a discharge/source point's resolved grid cell actually sits
    on the modelled channel. A point can be geometrically exact (e.g. a
    domain-entry point computed directly from the reach's own geometry) and
    still resolve to a neighbouring floodplain cell once rasterized onto a
    coarse grid -- injecting a large constant discharge there has nowhere
    near enough conveyance and can produce an unrealistic local water-level
    pileup (observed: basin 4267691's round-0 calibration, max water level
    369 m at one location).

    Args:
        points:           Point geometry, same CRS as centerline_cells.
        centerline_cells: Output of build_centerline_cells_regular -- must
                           have 'reach_id', 'row', 'col', 'x', 'y', geometry.
        reach_ids:        Optional, one entry per `points` row (e.g.
                           river_forcing.nc's own inside_reach_id, None where
                           unresolved) -- restricts the nearest-cell search
                           to that point's own reach first, which matters at
                           a confluence/bifurcation where a DIFFERENT
                           reach's cell could otherwise be geometrically
                           closer than the correct reach's own cell. Falls
                           back to searching every reach's cells when a
                           point's reach_id is None, or has no rows in
                           centerline_cells at all (e.g. a reach entirely
                           outside the active grid).
        resolution_m:     Grid cell size (m), used only to size the
                           "snapped suspiciously far" warning threshold
                           below. Skipped (no warning possible) if omitted.

    Returns:
        Copy of `points` with:
          - geometry replaced by the resolved cell's own center point (the
            exact coordinate build_centerline_cells_regular already used,
            so it is guaranteed consistent with channel_mask/calibration
            cell tracking elsewhere, not a fresh approximation)
          - new 'row'/'col' columns (the resolved cell's grid indices)
          - new 'snap_distance_m' column (distance from the original point
            to the resolved cell center), for diagnostics/logging
    """
    if centerline_cells.empty:
        raise ValueError("centerline_cells is empty -- cannot snap any points to it")

    all_xy = centerline_cells[["x", "y"]].to_numpy()
    all_rowcol = centerline_cells[["row", "col"]].to_numpy()
    by_reach: dict[str, np.ndarray] = {
        rid: idx.to_numpy()
        for rid, idx in centerline_cells.groupby("reach_id").groups.items()
    }
    if reach_ids is None:
        reach_ids = [None] * len(points)

    snapped_geoms = []
    snapped_rowcol = np.empty((len(points), 2), dtype=int)
    snap_dist_m = np.empty(len(points), dtype=float)

    for i, (pt, rid) in enumerate(zip(points.geometry, reach_ids)):
        norm_rid = normalize_reach_id(rid) if rid is not None else None
        candidate_idx = by_reach.get(norm_rid) if norm_rid is not None else None
        if candidate_idx is None or len(candidate_idx) == 0:
            candidate_idx = np.arange(len(centerline_cells))

        cand_xy = all_xy[candidate_idx]
        d2 = (cand_xy[:, 0] - pt.x) ** 2 + (cand_xy[:, 1] - pt.y) ** 2
        best = candidate_idx[np.argmin(d2)]

        snapped_geoms.append(shapely.Point(all_xy[best]))
        snapped_rowcol[i] = all_rowcol[best]
        snap_dist_m[i] = float(np.sqrt(d2.min()))

    out = points.copy()
    out["geometry"] = snapped_geoms
    out["row"] = snapped_rowcol[:, 0]
    out["col"] = snapped_rowcol[:, 1]
    out["snap_distance_m"] = snap_dist_m

    # A large snap distance usually means the reach_id lookup missed (wrong
    # or unresolved reach) rather than a genuinely distant channel -- worth
    # surfacing rather than silently accepting.
    if resolution_m and (snap_dist_m > 3 * resolution_m).any():
        n_far = int((snap_dist_m > 3 * resolution_m).sum())
        log.warning(
            f"snap_points_to_centerline_cells: {n_far}/{len(points)} point(s) snapped "
            f">3 cells away from their nearest centerline cell (max "
            f"{snap_dist_m.max():.0f} m) -- check reach_id resolution for these points"
        )
    return out
