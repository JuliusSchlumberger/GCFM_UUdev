"""
river_conditioning.py — Hydrological conditioning of the DEM along river
centerlines.

Enforces a monotonically non-increasing elevation profile along every river
reach in the downstream direction: wherever a centerline pixel is higher than
the running minimum from upstream, it is lowered to that minimum.  The
modified raster is written to a new file so the original is preserved.

This is applied AFTER the full river network has been processed
(river_network_processed.gpkg, which carries dist_out, is_seed, rch_id_dn
and geometry in the local UTM CRS) and BEFORE burn_river_rect (build_sfincs),
so it sees a DEM that already has no upstream-lower-than-downstream pixels
along the channel.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

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
) -> dict[str, int]:
    """
    Enforce monotonically non-increasing DEM elevation along river
    centerlines in the downstream direction.

    Algorithm:
    1. Process all reaches in topological (upstream→downstream) order.
    2. For each reach, sample the centerline at DEM pixel spacing using
       _sample_line_cells.
    3. Walk the pixels downstream; whenever a pixel's elevation exceeds the
       running minimum from upstream, set it to that minimum in the output
       array.
    4. At bifurcations both branches independently inherit the running
       minimum from the upstream reach's last pixel.
    5. Reaches with no upstream neighbour (headwaters and seed reaches)
       initialise the running minimum from their own first valid pixel.

    Args:
        rivers:         River network GeoDataFrame (any CRS) with 'reach_id',
                        'rch_id_dn', 'is_seed' columns — typically
                        river_network_processed.gpkg.
        elevation_path: Path to the source elevation raster
                        (elevation_merged.tif).
        output_path:    Path to write the conditioned raster
                        (elevation_conditioned.tif).

    Returns:
        Dict {reach_id_str: n_pixels_modified} with one entry per reach
        where at least one pixel was lowered.
    """
    with rasterio.open(elevation_path) as src:
        elevation_arr = src.read(1).astype(np.float32)
        transform = src.transform
        nodata = src.nodata
        raster_crs = src.crs
        profile = src.profile.copy()

    step_m = abs(transform.a)
    rivers_proj = (
        rivers.to_crs(raster_crs) if rivers.crs != raster_crs else rivers.copy()
    )
    rivers_proj["reach_id_norm"] = rivers_proj["reach_id"].apply(normalize_reach_id)

    line_by_rid: dict[str, object] = {}
    for row in rivers_proj.itertuples(index=False):
        rid = row.reach_id_norm
        g = _as_linestring(row.geometry)
        if g is not None and g.length > 0:
            line_by_rid[rid] = g

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

    # ── walk reaches in topo order, applying running minimum ─────────────────
    # end_min[rid] = running minimum at the downstream end of reach 'rid'.
    # Used to initialise the running minimum for each reach's downstream
    # neighbours.
    end_min: dict[str, float] = {}
    n_modified_by_reach: dict[str, int] = {}
    n_total_checked = 0

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

        end_min[rid] = running_min if running_min != np.inf else 0.0
        if n_mod > 0:
            n_modified_by_reach[rid] = n_mod

    n_total = sum(n_modified_by_reach.values())
    n_reaches = len(n_modified_by_reach)
    pct = 100.0 * n_total / n_total_checked if n_total_checked else 0.0
    log.info(
        f"enforce_river_monotonicity: {n_total}/{n_total_checked} pixel(s) lowered "
        f"({pct:.1f}%) across {n_reaches}/{len(topo_order)} reach(es)"
    )

    # ── write conditioned raster ──────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(elevation_arr, 1)
    log.info(f"Written: {output_path}")

    return n_modified_by_reach, n_total_checked
