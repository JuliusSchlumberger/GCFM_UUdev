"""
river_conditioning.py — Hydrological conditioning of the DEM along and
across river channels.

Enforces a monotonically non-increasing elevation profile along every river
reach in the downstream direction, then flattens the ENTIRE channel width
(not just the centerline) to that same along-channel profile. The modified
raster is written to a new file so the original is preserved.

Two passes:
1. Centerline: walk each reach's centerline (in topological, upstream-to-
   downstream order) at DEM pixel spacing, maintaining a running minimum;
   wherever a centerline pixel exceeds it, lower it to that minimum. This
   also builds each reach's own along-channel (distance, elevation) profile,
   used by pass 2.
2. Full width: for every reach with a known channel width, that SAME
   along-channel profile is applied UNCONDITIONALLY across the reach's
   buffered channel width (line.buffer(width/2)) -- every pixel in the
   cross-section, not just the centerline, is overwritten to the profile's
   value at its own along-channel position, flattening the whole
   cross-section rather than just capping it: local highs are lowered AND
   local depressions are filled. Off-centerline terrain is otherwise raw,
   unconditioned DEM noise that can (a) create spurious local flow barriers
   or diversions when the calibration run (rule modelled_depth_estimation)
   simulates directly on this conditioned-but-not-yet-excavated
   terrain, and (b) hide depressions that give that calibration run "free"
   extra conveyance which the later, perfectly flat rectangular channel
   burn (rule 11b) won't actually have -- both cause a mismatch between
   what calibration measures and what actually gets built. Where adjacent
   reaches' buffers legitimately overlap (near a junction), the LOWER of
   the two proposed values wins, keeping the overall downstream-non-
   increasing guarantee intact across the overlap.

This is applied AFTER the river network has been cleaned
(river_network_clean.gpkg, which carries dist_out, is_seed, rch_id_dn,
width and geometry in the local UTM CRS -- only topology/width are needed,
never rivdph, so this runs independently of/before both depth-estimation
alternatives, rule empirical_depth_estimation and rule
modelled_depth_estimation) and BEFORE
burn_river_rect (build_sfincs), so it sees a DEM that already has no
upstream-lower-than-downstream pixels along the channel, and no
unconditioned cross-sectional noise within it either.
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
from rasterio.windows import Window, transform as window_transform
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
    """
    Enforce a monotonically non-increasing DEM elevation profile along and
    across every river channel, in the downstream direction.

    See module docstring for the two-pass algorithm (centerline walk, then
    full-width flattening to the same along-channel profile).

    Args:
        rivers:         River network GeoDataFrame (any CRS) with 'reach_id',
                        'rch_id_dn', 'is_seed', width_column columns —
                        typically river_network_clean.gpkg.
        elevation_path: Path to the source elevation raster
                        (elevation_merged.tif).
        output_path:    Path to write the conditioned raster
                        (elevation_conditioned.tif).
        width_column:   Channel width column used to buffer each reach for
                        the full-width pass (always 'width'). A reach with
                        missing/non-positive width is still centerline-
                        conditioned (pass 1) but skipped by pass 2.

    Returns:
        (n_modified_by_reach, n_total_checked, centerline_max_m) —
        n_modified_by_reach is {reach_id_str: n_centerline_pixels_modified}
        from pass 1 only (kept for the existing diagnostic plot);
        n_total_checked is the number of centerline pixels sampled in pass
        1; centerline_max_m is the maximum conditioned (post-monotonicity)
        elevation across every reach's own centerline -- since the walk
        enforces a downstream-non-increasing profile, this is always
        attained at (one of) the network's own seed reach(es), and pass 2's
        full-width flattening never introduces a value above it (same
        along-channel profile, just repeated across width). Used by rule 13
        (and rule modelled_depth_estimation) to set an elevation ceiling on the active-cell mask:
        terrain far above anything the river itself ever reaches is not
        hydrologically relevant to this basin's own flood system.
    """
    with rasterio.open(elevation_path) as src:
        elevation_arr = src.read(1).astype(np.float32)
        transform = src.transform
        nodata = src.nodata
        raster_crs = src.crs
        raster_profile = src.profile.copy()

    step_m = abs(transform.a)
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

    # ── topological order (Kahn's algorithm on downstream adjacency) ──────────
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
    # Defensive: handle any unvisited (cycle or disconnected)
    topo_order.extend(rid for rid, d in in_degree.items() if d > 0)

    # ── pass 1: centerline walk, applying the running minimum -- also
    # records each reach's own (along, value) profile for pass 2 ────────────
    # end_min[rid] = running minimum at the downstream end of reach 'rid'.
    # Used to initialise the running minimum for each reach's downstream
    # neighbours.
    end_min: dict[str, float] = {}
    n_modified_by_reach: dict[str, int] = {}
    n_total_checked = 0
    centerline_max_m = -np.inf
    reach_profile: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for rid in topo_order:
        line = line_by_rid.get(rid)
        if line is None:
            continue

        # Inherit running minimum from all upstream parents (take the minimum
        # of their end values so we never rise above any upstream channel).
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
                running_min = v  # first valid pixel sets the baseline
            elif v > running_min:
                elevation_arr[c["row"], c["col"]] = np.float32(running_min)
                n_mod += 1
            else:
                running_min = v  # update: found a lower point
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

    # ── pass 2: flatten the full channel width to the same along-channel
    # profile (unconditional overwrite -- see module docstring) ─────────────
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

    # ── write conditioned raster ──────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **raster_profile) as dst:
        dst.write(elevation_arr, 1)
    log.info(f"Written: {output_path}")

    centerline_max_m = float(centerline_max_m) if np.isfinite(centerline_max_m) else 0.0
    return n_modified_by_reach, n_total_checked, centerline_max_m
