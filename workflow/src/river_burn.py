"""Burn a per-cell river-bed profile into a channel-only DEM; build channel masks.

zbed_anchors come from rule modelled_depth_estimation. Used instead of
hydromt_sfincs's burn_river_rect, which gives a wavy, non-monotonic bed when
re-run on a burned raster and matches each tile's clipped centreline against
all anchors (cross-tile contamination); here each reach uses only its own
anchors on its own full centreline. Output is NaN outside the channel buffers,
for use as a higher-priority elevation_list entry ahead of
elevation_merged/elevation_conditioned. build_channel_mask_regular feeds the
coastal protection weir (src/protection_weir.py).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from src.grid import cell_size_m
from src.raster import reproject_nan_aware
from src.river_network import (
    _BOUNDARY_BLEND_EPS_M,
    _as_linestring,
    _junction_value,
    build_downstream_adjacency,
    clamped_profile,
    line_window_cells,
    normalize_reach_id,
)

log = logging.getLogger(__name__)


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
    """Burn zbed_anchors' rivbed profile into a channel-only DEM.

    Args:
        rivers: Needs 'reach_id', width_column, geometry (any CRS).
        zbed_anchors: Per-cell bed anchors with 'reach_id', 'rivbed' (any CRS).
        natural_dem_path: Conditioned/merged DEM (FathomDEM + GEBCO, GOCO06s,
            same datum as rivbed). Terrain reference and floor: the burned bed
            is never above it. Not raw FathomDEM, which has no bathymetry and
            would let a shallower interpolated burn overwrite real GEBCO depths
            near a mouth.
        resolution_m: Output pixel size (m); ignored when out_transform is given.
        margin_m: Buffer (m) around the network bounds; ignored when
            out_transform is given.
        out_transform, out_shape: Target grid (rule modelled_depth_estimation
            passes the subgrid pixel grid, so HydroMT needs no resampling).
            Give both or neither.
        channel_mask: Boolean, shape out_shape; requires out_transform. When
            given, the burned cells are exactly its cells, not a per-reach
            polygon rasterization. Build it with build_channel_mask_regular on
            the same inputs so the calibration corridor, the excavation and the
            weir corridor are the same cells (rules modelled_depth_estimation
            and build_sfincs_skeleton).

    Each reach's interpolation gets a synthetic anchor at along=0 and/or
    along=length from its direct neighbours' nearest anchors (averaged at a
    confluence/bifurcation), so both reaches at a junction pass through the
    same value.

    Returns:
        (burned_arr, transform, terrain, stats): burned_arr float32, NaN outside
        every channel buffer; terrain is the natural DEM on the same grid
        (float32, bilinear, NaN where no data), the elevation of every pixel the
        burn leaves untouched; stats is a dict with 'n_reaches_burned',
        'n_reaches_skipped', 'n_pixels_burned', 'n_boundaries_blended'.
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

    if out_transform is not None:
        height_px, width_px = out_shape
        resolution_m = cell_size_m(out_transform)
        # Corner-based bounds: array_bounds assumes north-up and mislabels
        # min/max for a SFINCS grid with a positive y-scale.
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

    # Natural DEM is already GOCO06s, the datum of rivbed: no geoid correction.
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

    # Every reach with geometry and anchors, including unburnable ones, so a
    # burnable reach can borrow a junction value from such a neighbour.
    reach_data: dict[str, tuple[object, np.ndarray, np.ndarray]] = {}
    for row in rivers_proj.itertuples(index=False):
        reach_id = normalize_reach_id(row.reach_id)
        line = _as_linestring(row.geometry)
        if reach_id is None or line is None or line.length == 0:
            continue
        zbed_reach = zbed_proj[zbed_proj["_reach_id"] == reach_id]
        if len(zbed_reach) == 0:
            continue
        zbed_points = shapely.points(
            zbed_reach.geometry.x.to_numpy(), zbed_reach.geometry.y.to_numpy()
        )
        along = shapely.line_locate_point(line, zbed_points)
        rivbed_vals = zbed_reach["rivbed"].to_numpy(dtype=float)
        order = np.argsort(along)
        reach_data[reach_id] = (line, along[order], rivbed_vals[order])

    downstream_adj = build_downstream_adjacency(rivers_proj)
    upstream_adj: dict[str, list[str]] = {reach_id: [] for reach_id in downstream_adj}
    for reach_id, dns in downstream_adj.items():
        for dn in dns:
            upstream_adj.setdefault(dn, []).append(reach_id)

    output = np.full((height_px, width_px), np.nan, dtype=np.float32)
    n_reaches_burned = 0
    n_reaches_skipped = 0
    n_pixels_burned = 0
    n_boundaries_blended = 0

    for row in rivers_proj.itertuples(index=False):
        reach_id = normalize_reach_id(row.reach_id)
        width_m = getattr(row, width_column, np.nan)
        if (
            reach_id is None
            or pd.isna(width_m)
            or width_m <= 0
            or reach_id not in reach_data
        ):
            n_reaches_skipped += 1
            continue
        line, along, rivbed_vals = reach_data[reach_id]

        # Junction values from direct neighbours only, averaged over several.
        up_vals = [
            v
            for u in upstream_adj.get(reach_id, [])
            if (v := _junction_value(reach_data, u, reach_id)) is not None
        ]
        dn_vals = [
            v
            for d in downstream_adj.get(reach_id, [])
            if (v := _junction_value(reach_data, reach_id, d)) is not None
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

        interp = clamped_profile(along, rivbed_vals)
        buf_poly = _flush_capped_buffer(
            line, float(width_m), clip_start=bool(getattr(row, "is_seed", False))
        )
        # all_touched=True as in build_channel_mask_regular; the default
        # (centre inside) would narrow the footprint vs the weir corridor.
        cells = line_window_cells(
            buf_poly, line, out_transform, (height_px, width_px), channel_mask
        )
        if cells is None:
            n_reaches_skipped += 1
            continue
        rows_idx, cols_idx, pts_along = cells
        rivbed_at_pts = np.asarray(interp(pts_along), dtype=float)

        terrain_at_pts = terrain[rows_idx, cols_idx]
        has_terrain = np.isfinite(terrain_at_pts)
        burned_vals = np.where(
            has_terrain, np.minimum(terrain_at_pts, rivbed_at_pts), rivbed_at_pts
        )
        output[rows_idx, cols_idx] = burned_vals

        n_reaches_burned += 1
        n_pixels_burned += len(rows_idx)

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
    return output, out_transform, terrain, stats


def mean_elevation_on_coarse_grid(
    burned_fine: np.ndarray,
    terrain_fine: np.ndarray,
    nr_subgrid_pixels: int,
    chunk_rows: int = 256,
) -> np.ndarray:
    """Return per coarse cell the mean of its subgrid pixels with the burn in place.

    Burned pixels take burned_fine, others terrain_fine, so a channel narrower
    than a cell lowers it only in proportion to its area. Both inputs must be on
    the coarse grid's subgrid pixel grid (same origin, nr_subgrid_pixels per
    cell side). float32; NaN for cells with no burned pixel (they keep their
    background elevation) or no valid pixel.
    """
    n = int(nr_subgrid_pixels)
    if burned_fine.shape != terrain_fine.shape:
        raise ValueError(
            f"burned_fine {burned_fine.shape} and terrain_fine {terrain_fine.shape} differ"
        )
    if burned_fine.shape[0] % n or burned_fine.shape[1] % n:
        raise ValueError(
            f"fine shape {burned_fine.shape} is not a multiple of nr_subgrid_pixels={n}"
        )
    n_rows, n_cols = burned_fine.shape[0] // n, burned_fine.shape[1] // n
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    # In row chunks: the merged array is as large as the inputs themselves.
    for r0 in range(0, n_rows, chunk_rows):
        r1 = min(r0 + chunk_rows, n_rows)
        burned = burned_fine[r0 * n : r1 * n].reshape(r1 - r0, n, n_cols, n)
        has_burn = np.isfinite(burned).any(axis=(1, 3))
        if not has_burn.any():
            continue
        terrain = terrain_fine[r0 * n : r1 * n].reshape(r1 - r0, n, n_cols, n)
        merged = np.where(np.isfinite(burned), burned, terrain)
        valid = np.isfinite(merged)
        count = valid.sum(axis=(1, 3))
        total = np.where(valid, merged, 0.0).sum(axis=(1, 3), dtype=np.float64)
        mean = np.divide(
            total, count, out=np.full(total.shape, np.nan), where=count > 0
        )
        out[r0:r1] = np.where(has_burn, mean, np.nan)
    return out


def constrain_to_coarse_channel_mask(
    burned_arr: np.ndarray,
    native_transform,
    utm_crs,
    channel_mask_coarse: np.ndarray,
    coarse_transform,
) -> np.ndarray:
    """Null fine burned pixels whose parent coarse cell is outside channel_mask_coarse.

    The fine burn ranks above the background in elevation_list, and the weir
    crest is raised only within channel_mask, so a fine pixel outside it would
    be a deep, unprotected pocket. Keeps the excavated footprint a subset of
    the weir corridor.
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


def channel_polygons(rivers: gpd.GeoDataFrame, width_column: str = "width") -> list:
    """Return each reach's channel: its centreline buffered by width / 2, in the rivers' CRS.

    The one definition of the channel: rule 10 burns the bed into it, and the
    channel land use and Manning n (rules 05c, 08b, 09b) are set inside it.
    Reaches without a positive width are skipped. Raises ValueError for a
    geographic CRS, where width / 2 would be read as degrees.
    """
    if rivers.crs is not None and rivers.crs.is_geographic:
        raise ValueError(f"channel_polygons needs a metric CRS, got {rivers.crs}")
    polys = []
    for row in rivers.itertuples(index=False):
        width_m = getattr(row, width_column, np.nan)
        if pd.isna(width_m) or width_m <= 0:
            continue
        line = _as_linestring(row.geometry)
        if line is None or line.length == 0:
            continue
        polys.append(
            _flush_capped_buffer(
                line, float(width_m), clip_start=bool(getattr(row, "is_seed", False))
            )
        )
    return polys


def build_channel_mask_regular(
    rivers: gpd.GeoDataFrame,
    width_column: str,
    out_shape: tuple[int, int],
    out_transform,
) -> np.ndarray:
    """Rasterize every reach's channel buffer directly onto a target (SFINCS) grid.

    all_touched=True: a channel narrower than one cell still registers, unlike
    nearest-neighbour resampling of a fine mask, which drops narrow reaches
    and pinches the channel shut.
    """
    polys = channel_polygons(rivers, width_column)
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
    """Return every grid cell each reach's raw centreline touches (all_touched=True).

    Cells for the per-cell depth/crest calibration in
    10_depth_estimation_modelled.py. A junction cell gets one row per reach
    (keyed reach_id, row, col). Columns 'reach_id', 'row', 'col', 'along_m'
    (from the reach's start), 'x', 'y', geometry (cell centre, CRS of
    `rivers`), sorted by (reach_id, along_m); degenerate reaches are skipped.
    """
    inv_transform = ~out_transform
    records: list[dict] = []

    for row in rivers.itertuples(index=False):
        reach_id = normalize_reach_id(row.reach_id)
        line = _as_linestring(row.geometry)
        if reach_id is None or line is None or line.length == 0:
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
                    "reach_id": reach_id,
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
    width_m: float,
    clip_start: bool = False,
    clip_end: bool = False,
) -> shapely.Polygon:
    # Flat cut at a seed: the round cap would enclose cells that
    # build_centerline_cells_regular never samples, inside the same confined
    # pocket as the injection cell. A degenerate end segment keeps its round cap.
    buf = line.buffer(width_m / 2.0)
    if not clip_start and not clip_end:
        return buf
    coords = list(line.coords)
    minx, miny, maxx, maxy = buf.bounds
    big = (maxx - minx) + (maxy - miny) + width_m  # safely larger than buf's own extent
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
    """Snap points to the nearest cell their reach's raw centreline passes through.

    Neither hydromt_sfincs nor geometry.snap_points_into_region checks that a
    discharge point lands on the channel; one resolving to a floodplain cell
    lacks conveyance and can pile water up (basin 4267691 round 0: 369 m max
    water level).

    Args:
        points: Same CRS as centerline_cells.
        centerline_cells: From build_centerline_cells_regular.
        reach_ids: One per point (e.g. inside_reach_id, None if unresolved);
            searches that reach's cells first, so another reach's cell at a
            confluence cannot win. Falls back to all cells when None or absent.
        resolution_m: Cell size (m); enables the warning for snaps > 3 cells.

    Returns:
        Copy of `points` with geometry at the cell centre used by
        build_centerline_cells_regular, plus 'row', 'col', 'snap_distance_m'.
    """
    if centerline_cells.empty:
        raise ValueError("centerline_cells is empty -- cannot snap any points to it")

    all_xy = centerline_cells[["x", "y"]].to_numpy()
    all_rowcol = centerline_cells[["row", "col"]].to_numpy()
    by_reach: dict[str, np.ndarray] = {
        reach_id: idx.to_numpy()
        for reach_id, idx in centerline_cells.groupby("reach_id").groups.items()
    }
    if reach_ids is None:
        reach_ids = [None] * len(points)

    snapped_geoms = []
    snapped_rowcol = np.empty((len(points), 2), dtype=int)
    snap_dist_m = np.empty(len(points), dtype=float)

    for i, (pt, reach_id) in enumerate(zip(points.geometry, reach_ids)):
        norm_reach_id = normalize_reach_id(reach_id) if reach_id is not None else None
        candidate_idx = (
            by_reach.get(norm_reach_id) if norm_reach_id is not None else None
        )
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

    # A large snap distance usually means a wrong or unresolved reach_id.
    if resolution_m and (snap_dist_m > 3 * resolution_m).any():
        n_far = int((snap_dist_m > 3 * resolution_m).sum())
        log.warning(
            f"snap_points_to_centerline_cells: {n_far}/{len(points)} point(s) snapped "
            f">3 cells away from their nearest centerline cell (max "
            f"{snap_dist_m.max():.0f} m) -- check reach_id resolution for these points"
        )
    return out
