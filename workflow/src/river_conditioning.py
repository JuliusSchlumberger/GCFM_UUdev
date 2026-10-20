"""Condition the DEM along and across river channels (rule enforce_river_monotonicity).

Two passes, writing a new raster:
1. Centreline: walk each reach upstream-to-downstream at pixel spacing with a
   running minimum, lowering any pixel above it; this gives each reach's
   along-channel profile.
2. Full width: overwrite every pixel in the channel buffer (width/2) with that
   profile at its along-channel position, lowering highs AND filling lows.
   Raw off-centreline noise would otherwise create spurious barriers in the
   calibration run (rule modelled_depth_estimation) or give it conveyance the
   flat rectangular burn (rule 11b) lacks. Where buffers overlap at a junction
   the lower value wins.

Needs only topology and width (not rivdph), so it runs before rule
modelled_depth_estimation and before burn_river_rect (build_sfincs).
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from rasterio.features import geometry_mask
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from scipy.interpolate import interp1d
from src.river_network import (
    _as_linestring,
    _sample_line_cells,
    build_downstream_adjacency,
    normalize_reach_id,
)

log = logging.getLogger(__name__)


def enforce_river_monotonicity(
    rivers: gpd.GeoDataFrame,
    elevation_path: str | Path,
    output_path: str | Path,
    width_column: str = "width",
) -> tuple[dict[str, int], int]:
    """Make the DEM non-increasing downstream along and across every river channel.

    A reach with missing/non-positive width is centreline-conditioned only.

    Returns:
        (n_modified_by_reach, n_total_checked, centerline_max_m):
        {reach_id: centreline pixels lowered} and pixels sampled, both from
        pass 1; centerline_max_m is the highest conditioned centreline
        elevation (at a seed reach; pass 2 never exceeds it). Rule 13 and rule
        modelled_depth_estimation use it as the active-cell elevation ceiling.
    """
    with rasterio.open(elevation_path) as src:
        elevation_arr = src.read(1).astype(np.float32)
        transform = src.transform
        nodata = src.nodata
        raster_crs = src.crs
        raster_profile = src.profile.copy()

    step_m = float(np.hypot(transform.a, transform.d))  # pixel size, any orientation
    rivers_proj = (
        rivers.to_crs(raster_crs) if rivers.crs != raster_crs else rivers.copy()
    )
    rivers_proj["reach_id_norm"] = rivers_proj["reach_id"].apply(normalize_reach_id)

    line_by_rid: dict[str, object] = {}
    width_by_rid: dict[str, float] = {}
    for row in rivers_proj.itertuples(index=False):
        rid = row.reach_id_norm
        g = _as_linestring(row.geometry)
        if g is not None and g.length > 0:
            line_by_rid[rid] = g
        w = getattr(row, width_column, np.nan)
        if w is not None and np.isfinite(w) and w > 0:
            width_by_rid[rid] = float(w)

    downstream_adj = build_downstream_adjacency(rivers)
    upstream_adj: dict[str, list[str]] = {rid: [] for rid in downstream_adj}
    for rid, dns in downstream_adj.items():
        for dn in dns:
            upstream_adj.setdefault(dn, []).append(rid)

    in_degree = {rid: len(parents) for rid, parents in upstream_adj.items()}
    queue: deque[str] = deque(rid for rid, d in in_degree.items() if d == 0)
    topo_order: list[str] = []
    while queue:
        rid = queue.popleft()
        topo_order.append(rid)
        for dn in downstream_adj.get(rid, []):
            in_degree[dn] -= 1
            if in_degree[dn] == 0:
                queue.append(dn)
    # Reaches on a cycle go last.
    topo_order.extend(rid for rid, d in in_degree.items() if d > 0)

    # Pass 1: centreline running minimum.
    end_min: dict[str, float] = {}
    n_modified_by_reach: dict[str, int] = {}
    n_total_checked = 0
    centerline_max_m = -np.inf
    reach_profile: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for rid in topo_order:
        line = line_by_rid.get(rid)
        if line is None:
            continue

        # Start from the lowest upstream end so no reach rises above any parent.
        parents = upstream_adj.get(rid, [])
        parent_mins = [end_min[p] for p in parents if p in end_min]
        running_min: float = float(min(parent_mins)) if parent_mins else np.inf

        n_mod = 0
        along_list: list[float] = []
        val_list: list[float] = []
        cells = _sample_line_cells(line, transform, elevation_arr.shape, step_m)
        for c in cells:
            v = float(elevation_arr[c["row"], c["col"]])
            if nodata is not None and v == nodata:
                continue
            if not np.isfinite(v):
                continue
            n_total_checked += 1
            if running_min == np.inf:
                running_min = v
            elif v > running_min:
                elevation_arr[c["row"], c["col"]] = np.float32(running_min)
                n_mod += 1
            else:
                running_min = v
            along_list.append(c["along_m"])
            val_list.append(running_min)

        end_min[rid] = running_min if running_min != np.inf else 0.0
        if n_mod > 0:
            n_modified_by_reach[rid] = n_mod
        if along_list:
            reach_profile[rid] = (np.asarray(along_list), np.asarray(val_list))
            centerline_max_m = max(centerline_max_m, max(val_list))

    n_total = sum(n_modified_by_reach.values())
    n_reaches = len(n_modified_by_reach)
    pct = 100.0 * n_total / n_total_checked if n_total_checked else 0.0
    log.info(
        f"enforce_river_monotonicity (centerline): {n_total}/{n_total_checked} "
        f"pixel(s) lowered ({pct:.1f}%) across {n_reaches}/{len(topo_order)} reach(es)"
    )

    # Pass 2: overwrite the full channel width with the centreline profile.
    orig_is_nodata = (
        np.isclose(elevation_arr, np.float32(nodata))
        if nodata is not None
        else np.zeros(elevation_arr.shape, dtype=bool)
    )
    touched = np.zeros(elevation_arr.shape, dtype=bool)
    full_shape = elevation_arr.shape
    inv_transform = ~transform
    n_width_modified = 0
    n_width_checked = 0

    for rid in topo_order:
        line = line_by_rid.get(rid)
        width = width_by_rid.get(rid)
        if line is None or width is None or rid not in reach_profile:
            continue
        along_arr, val_arr = reach_profile[rid]
        if len(along_arr) == 1:
            _const = float(val_arr[0])

            def interp(s, _v=_const):
                return np.full(np.shape(s), _v, dtype=float)
        else:
            interp = interp1d(
                along_arr,
                val_arr,
                kind="linear",
                bounds_error=False,
                fill_value=(float(val_arr[0]), float(val_arr[-1])),
            )

        buf_poly = line.buffer(width / 2.0)
        bminx, bminy, bmaxx, bmaxy = buf_poly.bounds
        corners = [(bminx, bminy), (bmaxx, bminy), (bminx, bmaxy), (bmaxx, bmaxy)]
        cols_c, rows_c = zip(*(inv_transform * c for c in corners))
        col_off = max(0, int(np.floor(min(cols_c))))
        row_off = max(0, int(np.floor(min(rows_c))))
        col_stop = min(full_shape[1], int(np.ceil(max(cols_c))))
        row_stop = min(full_shape[0], int(np.ceil(max(rows_c))))
        if col_stop <= col_off or row_stop <= row_off:
            continue
        window = Window(col_off, row_off, col_stop - col_off, row_stop - row_off)
        win_transform = window_transform(window, transform)
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
        pts = shapely.points(np.asarray(xs), np.asarray(ys))
        pts_along = shapely.line_locate_point(line, pts)
        target_vals = np.asarray(interp(pts_along), dtype=float)

        abs_rows = rows_idx + row_off
        abs_cols = cols_idx + col_off
        keep = ~orig_is_nodata[abs_rows, abs_cols]
        if not keep.any():
            continue
        rr, cc, tv = abs_rows[keep], abs_cols[keep], target_vals[keep]

        prior_touched = touched[rr, cc]
        prior_vals = elevation_arr[rr, cc]
        new_vals = np.where(prior_touched, np.minimum(prior_vals, tv), tv).astype(
            np.float32
        )
        changed = ~np.isclose(new_vals, prior_vals)
        elevation_arr[rr, cc] = new_vals
        touched[rr, cc] = True
        n_width_modified += int(changed.sum())
        n_width_checked += int(keep.sum())

    log.info(
        f"enforce_river_monotonicity (full width): {n_width_modified}/{n_width_checked} "
        f"pixel(s) flattened across the channel width"
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **raster_profile) as dst:
        dst.write(elevation_arr, 1)
    log.info(f"Written: {output_path}")

    centerline_max_m = float(centerline_max_m) if np.isfinite(centerline_max_m) else 0.0
    return n_modified_by_reach, n_total_checked, centerline_max_m
