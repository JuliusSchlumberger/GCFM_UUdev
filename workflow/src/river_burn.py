"""
river_burn.py — Burn the zbed_anchors.gpkg river-bed profile directly into a
channel-only DEM at native (fine) resolution, bypassing hydromt_sfincs's own
per-tile burn_river_rect (see workflow/rules/11b_burn_river_dem.smk for why).

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
"""

from __future__ import annotations

import logging
import math
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from rasterio.windows import (
    Window,
    from_bounds as window_from_bounds,
    transform as window_transform,
)
from scipy.interpolate import interp1d

from src.raster import (
    compute_geoid_offset_arr,
    find_fathomdem_tiles,
    merge_tiled_raster,
    reproject_nan_aware,
)
from src.river_network import (
    _as_linestring,
    build_downstream_adjacency,
    normalize_reach_id,
)

log = logging.getLogger(__name__)

# Minimum distance (m) an own anchor must already be from a reach's start/end
# before a borrowed neighbour boundary value is skipped as redundant.
_BOUNDARY_BLEND_EPS_M = 1.0


def _windows_intersect(a: Window, b: Window) -> bool:
    return not (
        a.col_off + a.width <= b.col_off
        or b.col_off + b.width <= a.col_off
        or a.row_off + a.height <= b.row_off
        or b.row_off + b.height <= a.row_off
    )


def _junction_value(
    reach_data: dict[str, tuple[object, np.ndarray, np.ndarray]],
    up_rid: str,
    dn_rid: str,
) -> float | None:
    """
    Shared bed value at the junction between two adjacent reaches: the
    average of the upstream reach's own anchor nearest ITS downstream end
    and the downstream reach's own anchor nearest ITS upstream start.

    Using this SAME averaged value as the synthetic boundary anchor on BOTH
    sides of the junction (rather than each side simply adopting the other
    side's raw value) is what actually removes the step there -- both
    reaches' interpolation then passes through an identical value at the
    shared coordinate. Adopting the other side's raw value outright would
    just swap which side has the mismatch instead of removing it.
    """
    up_entry = reach_data.get(up_rid)
    dn_entry = reach_data.get(dn_rid)
    if up_entry is None or dn_entry is None:
        return None
    _, up_along, up_rivbed = up_entry
    _, dn_along, dn_rivbed = dn_entry
    if len(up_along) == 0 or len(dn_along) == 0:
        return None
    up_val = float(up_rivbed[np.argmax(up_along)])
    dn_val = float(dn_rivbed[np.argmin(dn_along)])
    return (up_val + dn_val) / 2.0


def burn_river_channel(
    rivers: gpd.GeoDataFrame,
    zbed_anchors: gpd.GeoDataFrame,
    topo_tiles_dir: str | Path,
    goco_path: str | Path,
    egm_path: str | Path,
    utm_crs,
    resolution_m: float,
    width_column: str = "width",
    margin_m: float = 500.0,
) -> tuple[np.ndarray, object, float, dict]:
    """
    Burn zbed_anchors' rivbed profile into a native-resolution, channel-only DEM.

    Args:
        rivers:         river_network_estuarine.gpkg (any CRS) — needs
                        'reach_id', width_column, geometry.
        zbed_anchors:   rule burn_river_bed's output (any CRS) — needs
                        'reach_id', 'rivbed', geometry.
        topo_tiles_dir: FathomDEM tile directory (data_catalogue 'fathomdem').
        goco_path, egm_path: GOCO06s / EGM2008 .gfc files for the mandatory
                        EGM2008->GOCO06s datum correction — zbed_anchors'
                        rivbed values are already GOCO06s-referenced (via
                        elevation_conditioned), so the terrain we compare
                        against must be too.
        utm_crs:        target CRS for the output raster (the basin's own
                        working UTM CRS).
        resolution_m:   output pixel size (m).
        width_column:   channel width column in `rivers` (always 'width').
        margin_m:       buffer (m) added around the river network's own
                        bounds before fetching FathomDEM tiles, so buffers
                        near the network's own extent edge still get real
                        DEM coverage.

    Each reach's own interpolation is extended with one synthetic anchor at
    along=0 and/or along=length, valued from its immediate upstream/
    downstream neighbour's own nearest-anchor rivbed (averaged across
    multiple neighbours at a confluence/bifurcation) -- both reaches sharing
    a junction then interpolate through the same boundary value, turning
    the old flat-clamp discontinuity there into a gradual ramp, without
    reaching any further than direct neighbours (avoiding burn_river_rect's
    original global-contamination bug this function was written to fix).

    Returns:
        (burned_arr, transform, nodata, stats) — burned_arr is float32, NaN
        outside every reach's own channel buffer; stats is a dict with
        'n_reaches_burned', 'n_reaches_skipped', 'n_pixels_burned'.
    """
    rivers_proj = rivers.to_crs(utm_crs) if rivers.crs != utm_crs else rivers.copy()
    zbed_proj = (
        zbed_anchors.to_crs(utm_crs)
        if zbed_anchors.crs != utm_crs
        else zbed_anchors.copy()
    )
    zbed_proj = zbed_proj.assign(
        _reach_id=[normalize_reach_id(x) for x in zbed_proj["reach_id"]]
    )

    # ── output grid: river network's own bounds + margin, snapped to a
    # resolution_m lattice ────────────────────────────────────────────────────
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

    # ── native-resolution FathomDEM, EGM2008->GOCO06s corrected, cropped to
    # the burn grid (not the whole basin domain) ──────────────────────────────
    wgs84_bounds = transform_bounds(utm_crs, "EPSG:4326", xmin, ymin, xmax, ymax)

    tiles = find_fathomdem_tiles(topo_tiles_dir, wgs84_bounds)
    if not tiles:
        raise FileNotFoundError(
            f"No FathomDEM tiles found for river network bounds {wgs84_bounds}"
        )
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tf:
        tmp_topo = tf.name
    try:
        merge_tiled_raster(tiles, wgs84_bounds, tmp_topo)
        with rasterio.open(tmp_topo) as src:
            topo_arr = src.read(1).astype(np.float32)
            topo_nodata = src.nodata
            topo_src_crs = src.crs
            topo_src_transform = src.transform
    finally:
        Path(tmp_topo).unlink(missing_ok=True)

    if topo_nodata is not None:
        topo_arr[topo_arr == np.float32(topo_nodata)] = np.nan
    topo_arr = topo_arr / 100.0  # FathomDEM tiles: int32 centimeters, not meters

    terrain = reproject_nan_aware(
        topo_arr,
        topo_src_transform,
        topo_src_crs,
        (height_px, width_px),
        out_transform,
        utm_crs,
        resampling=Resampling.bilinear,
    )
    log.info(
        f"Native FathomDEM (river-network extent): "
        f"{(~np.isnan(terrain)).sum():,} valid px"
    )

    # EGM2008 -> GOCO06s (mandatory — matches rule get_elevation's step 2;
    # zbed_anchors' rivbed values are already GOCO06s-referenced).
    offset_arr, offset_transform, offset_crs = compute_geoid_offset_arr(
        goco_path, egm_path
    )
    offset_on_grid = reproject_nan_aware(
        offset_arr.astype(np.float32),
        offset_transform,
        offset_crs,
        (height_px, width_px),
        out_transform,
        utm_crs,
    )
    valid_terrain = ~np.isnan(terrain)
    terrain[valid_terrain] += offset_on_grid[valid_terrain]
    del offset_arr, offset_on_grid

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
            # pixels can overshoot by more than a metre -- confirmed live
            # (basin 2433835, reach 21604100033: extrapolated to -3.9 m at
            # the reach start vs. -2.4 m at the nearest real anchor point).
            interp = interp1d(
                along,
                rivbed_vals,
                kind="linear",
                bounds_error=False,
                fill_value=(rivbed_vals[0], rivbed_vals[-1]),
            )

        buf_poly = line.buffer(float(width) / 2.0)
        window = window_from_bounds(*buf_poly.bounds, transform=out_transform)
        window = window.round_offsets().round_lengths()
        if not _windows_intersect(window, full_window):
            n_reaches_skipped += 1
            continue
        window = window.intersection(full_window)
        if window.width <= 0 or window.height <= 0:
            n_reaches_skipped += 1
            continue

        win_transform = window_transform(window, out_transform)
        win_shape = (int(window.height), int(window.width))
        inside = geometry_mask(
            [buf_poly], out_shape=win_shape, transform=win_transform, invert=True
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
