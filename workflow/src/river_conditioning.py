"""Condition the DEM along and across river channels (rule enforce_river_monotonicity).

Two passes, writing a new raster:
1. Centreline: walk each reach upstream-to-downstream at pixel spacing with a
   running minimum, lowering any pixel above it; this gives each reach's
   along-channel profile.
2. Full width: overwrite every pixel in the channel buffer (width/2) with that
   profile at its along-channel position, lowering highs AND filling lows.
   Raw off-centreline noise would otherwise create spurious barriers in the
   calibration run (rule modelled_depth_estimation) or give it conveyance the
   rectangular channel burn (src.river_burn.burn_river_channel) lacks. Where buffers overlap at a junction
   the lower value wins.

Needs only topology and width (not river_depth_m), so it runs in rule
enforce_river_monotonicity, before rule modelled_depth_estimation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from src.river_network import (
    _as_linestring,
    _sample_line_cells,
    build_downstream_adjacency,
    clamped_profile,
    kahn_order,
    line_window_cells,
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

    line_by_reach_id: dict[str, object] = {}
    width_by_reach_id: dict[str, float] = {}
    for row in rivers_proj.itertuples(index=False):
        reach_id = row.reach_id_norm
        g = _as_linestring(row.geometry)
        if g is not None and g.length > 0:
            line_by_reach_id[reach_id] = g
        w = getattr(row, width_column, np.nan)
        if w is not None and np.isfinite(w) and w > 0:
            width_by_reach_id[reach_id] = float(w)

    downstream_adj = build_downstream_adjacency(rivers)
    upstream_adj: dict[str, list[str]] = {reach_id: [] for reach_id in downstream_adj}
    for reach_id, dns in downstream_adj.items():
        for dn in dns:
            upstream_adj.setdefault(dn, []).append(reach_id)

    # Reaches on a cycle go last.
    topo_order, _ = kahn_order(
        list(upstream_adj), lambda reach_id: downstream_adj.get(reach_id, [])
    )

    # Pass 1: centreline running minimum.
    end_min: dict[str, float] = {}
    n_modified_by_reach: dict[str, int] = {}
    n_total_checked = 0
    centerline_max_m = -np.inf
    reach_profile: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for reach_id in topo_order:
        line = line_by_reach_id.get(reach_id)
        if line is None:
            continue

        # Start from the lowest upstream end so no reach rises above any parent.
        parents = upstream_adj.get(reach_id, [])
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

        end_min[reach_id] = running_min if running_min != np.inf else 0.0
        if n_mod > 0:
            n_modified_by_reach[reach_id] = n_mod
        if along_list:
            reach_profile[reach_id] = (np.asarray(along_list), np.asarray(val_list))
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
    n_width_modified = 0
    n_width_checked = 0

    for reach_id in topo_order:
        line = line_by_reach_id.get(reach_id)
        width_m = width_by_reach_id.get(reach_id)
        if line is None or width_m is None or reach_id not in reach_profile:
            continue
        along_arr, val_arr = reach_profile[reach_id]
        interp = clamped_profile(along_arr, val_arr)
        buf_poly = line.buffer(width_m / 2.0)
        cells = line_window_cells(buf_poly, line, transform, full_shape)
        if cells is None:
            continue
        abs_rows, abs_cols, pts_along = cells
        target_vals = np.asarray(interp(pts_along), dtype=float)

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
