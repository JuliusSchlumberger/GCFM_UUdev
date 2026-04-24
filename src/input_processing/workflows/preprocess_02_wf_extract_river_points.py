"""
Extracts river source points for each delta domain polygon.

Reads the pre-built basin domains and for each delta finds the most-downstream
GloFAS discharge cells that can serve as model inflow points.

Performance notes
-----------------
- All heavy files (rivers, coastline, GloFAS) are loaded ONCE via
  ``load_global_data()`` before the loop.
- Per-delta subsets are derived via ``load_data_delta_domain()`` using
  coordinate indexing (.cx) and xarray slicing — no repeated file I/O.
- Basin and delta lookups use pre-built dicts (O(1) per delta).
- GloFAS dataset is periodically re-opened to prevent file-handle / lazy-graph
  accumulation from crashing the process after many iterations.
"""

from __future__ import annotations

import gc
from pathlib import Path

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import MultiLineString, LineString, Polygon
from shapely.geometry.base import BaseGeometry
import xarray as xr

from src.input_processing.config.loader import config
from src.input_processing.utils.util_unify_typing_and_schema import (
    ensure_valid_schema,
    CRS_STANDARD,
    BASIN_COL,
    GEOM_COL,
)
from src.input_processing.utils.preprocess_02_ut_extract_river_points import (
    extract_cells_within_delta,
    clip_basin_boundary_from_coast,
)

from src.input_processing.utils.validation.modify_delta_masks import classify_points
from src.input_processing.utils.loading_files import (
    load_global_data,
    load_data_delta_domain,
    GlobalData,
)
from typing import cast


# ---------------------------------------------------------------------------
# How often to re-open the GloFAS dataset.
# xarray builds up a lazy computation graph on each .sel() slice — after
# many iterations this graph grows large enough to cause slowdowns or crashes.
# Re-opening closes the old graph and starts fresh.
# ---------------------------------------------------------------------------
_FLUSH_INTERVAL: int = 5  # write to disk every N successful results
_GLOFAS_RELOAD_INTERVAL: int = 5  # also reduce this from 10 to 5

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _is_empty(gdf: GeoDataFrame | None) -> bool:
    """Return True when *gdf* is None or contains no rows."""
    return gdf is None or gdf.empty


def _reload_glofas(global_data: GlobalData) -> GlobalData:
    """
    Close the current GloFAS dataset and re-open it.

    Returns a new ``GlobalData`` instance with a fresh xarray Dataset,
    reusing the already-loaded rivers and coastline GeoDataFrames.
    This prevents the xarray lazy-evaluation graph from accumulating
    across many ``.sel()`` calls, which causes slowdowns and crashes.
    """
    print("[MEM] Re-opening GloFAS dataset to clear lazy graph ...", flush=True)
    try:
        global_data.glofas.close()
    except Exception:
        pass  # already closed or not closeable — proceed

    fresh_glofas: xr.Dataset = xr.open_dataset(
        config["filepaths"]["glofas"]
    ).rio.write_crs(CRS_STANDARD)

    return GlobalData(
        rivers=global_data.rivers,  # reuse — already in memory
        coastline=global_data.coastline,  # reuse — already in memory
        glofas=fresh_glofas,
    )


# ---------------------------------------------------------------------------
# Source-point extraction for a single delta
# ---------------------------------------------------------------------------


def find_river_source_points(
    delta_basins_gpd: GeoDataFrame,
    delta_polygon: Polygon,
    rivers_gpd: GeoDataFrame,
    glofas_min: xr.DataArray,
    coast_polygon: BaseGeometry,
    delta_edmonds: GeoDataFrame,
    basin_polygons: GeoDataFrame,
    basin_polygons_domain: GeoDataFrame,
    all_rivers: GeoDataFrame,
    use_basins: bool = True,
) -> tuple[GeoDataFrame | None, GeoDataFrame | None, GeoSeries | None]:
    """
    Find unique and possible river source points for a single delta domain.

    Args:
        delta_basins_gpd:      Basin polygons for this delta, in CRS_STANDARD.
        delta_polygon:         Edmonds delta polygon geometry (used when
                               ``use_basins=False``).
        rivers_gpd:            River network clipped to the delta bounding box.
        glofas_min:            Per-cell minimum GloFAS discharge DataArray for
                               the delta region.
        coast_polygon:         Union of coastline polygons for the delta region.
        delta_edmonds:         Edmonds delta polygon GeoDataFrame (for plotting).
        basin_polygons:        All basin polygons (for plotting context).
        basin_polygons_domain: Basin polygons for this delta only (for plotting).
        all_rivers:            Full river network (for plotting context).
        use_basins:            If True, derive the inland boundary from the
                               watershed clipped to the coast.  If False, use
                               the classify_points fallback.

    Returns:
        ``(unique_sources, possible_sources, buffered_cells)`` — all
        GeoDataFrames/GeoSeries in CRS_STANDARD, or ``(None, None, None)``
        if the inland boundary could not be derived.
    """
    # --- Clip rivers to this basin ---
    relevant_rivers: GeoDataFrame = gpd.sjoin(rivers_gpd, delta_basins_gpd, how="inner")
    relevant_rivers = relevant_rivers[rivers_gpd.columns].copy()
    relevant_rivers[BASIN_COL] = delta_basins_gpd[BASIN_COL].iloc[0]

    # --- Inland boundary ---
    inland_boundary: LineString | MultiLineString
    try:
        if use_basins:
            inland_boundary = clip_basin_boundary_from_coast(
                delta_basins_gpd, coast_polygon
            )
        else:
            bw, s1, s2, _, _ = classify_points(delta_polygon, coast_polygon)
            inland_boundary = MultiLineString(
                [
                    LineString([s1, bw]),
                    LineString([bw, s2]),
                ]
            )
    except ValueError as e:
        delta_id: int | str = delta_basins_gpd[BASIN_COL].iloc[0]
        print(f"[SKIP] Delta {delta_id} — could not derive inland boundary: {e}")
        return None, None, None

    # --- Extract GloFAS source cells ---
    return extract_cells_within_delta(
        glofas_min,
        inland_boundary,
        relevant_rivers,
        delta_edmonds,
        basin_polygons,
        basin_polygons_domain,
        all_rivers,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def extract_all_river_source_points(
    out_unique_sources: str = config["filepaths"]["unique_sources"],
    out_possible_sources: str = config["filepaths"]["possible_sources"],
    out_deltas_no_rivers: str = config["filepaths"]["out_deltas_no_rivers"],
    debug_plots: bool = True,
) -> None:
    """
    Iterate over all delta domains, extract river source points, and save results.

    All heavy files are loaded once before the loop.  Per-delta subsets are
    derived via coordinate-index slicing with no repeated file I/O.
    The GloFAS dataset is re-opened every ``_GLOFAS_RELOAD_INTERVAL`` iterations
    to prevent xarray lazy-graph accumulation.

    Outputs three GeoPackage files:
    - *out_unique_sources*:    Most-downstream unique inflow points per delta.
    - *out_possible_sources*:  All candidate inflow points per delta.
    - *out_deltas_no_rivers*:  Delta polygons for which no sources were found.
    """
    # --- Load heavy datasets once ---
    global_data: GlobalData = load_global_data()

    # --- Load and validate domain files ---
    delta_domains: GeoDataFrame = ensure_valid_schema(
        gpd.read_file(config["filepaths"]["delta_polygons_used"]),
        excluded=[],
    )
    river_basins_gpd: GeoDataFrame = ensure_valid_schema(
        gpd.read_file(config["filepaths"]["new_domains"]),
        excluded=[],
    )

    # all_rivers is already validated inside load_global_data() — no second read.
    all_rivers: GeoDataFrame = global_data.rivers

    # --- Pre-build O(1) lookups by basin ID ---
    basin_lookup: dict[int | str, GeoDataFrame] = {
        cast(int | str, k): cast(GeoDataFrame, v.copy())
        for k, v in river_basins_gpd.groupby(BASIN_COL)
    }
    delta_lookup: dict[int | str, GeoDataFrame] = {
        cast(int | str, k): cast(GeoDataFrame, v.copy())
        for k, v in delta_domains.groupby(BASIN_COL)
    }
    # basin_polygons_lookup reuses river_basins_gpd — same source file,
    # no second read required.
    basin_polygons_lookup: dict[int | str, GeoDataFrame] = basin_lookup

    # --- Accumulators ---
    all_unique_sources: list[GeoDataFrame] = []
    all_possible_sources: list[GeoDataFrame] = []
    deltas_without_sources: list[pd.Series] = []

    success_count: int = 0
    fail_count: int = 0
    flush_count: int = 0  # tracks how many times we've flushed to disk

    # helper — append to a GeoPackage if it exists, create it if not
    def _flush_to_disk(
        frames: list[GeoDataFrame],
        path: str,
        crs: int,
    ) -> None:
        if not frames:
            return
        gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=crs)
        p = Path(path)
        if p.exists():
            # Append to existing file
            gdf.to_file(p, driver="GPKG", mode="a")
        else:
            gdf.to_file(p, driver="GPKG")

    # --- Per-delta loop ---
    idx: int
    row: pd.Series
    for idx, row in delta_domains.iterrows():
        if idx % 10 == 0:
            print(f"{idx} ...")

        # -------------------------------------------------------------------
        # Periodic GloFAS reload — prevents xarray lazy-graph accumulation.
        # Rivers and coastline are already fully in memory so they are reused.
        # -------------------------------------------------------------------
        if idx > 0 and idx % _GLOFAS_RELOAD_INTERVAL == 0:
            global_data = _reload_glofas(global_data)
            gc.collect()

        polygon_id: int | str = row[BASIN_COL]
        delta_polygon: Polygon = row[GEOM_COL]

        # --- basin lookup ---
        if polygon_id not in basin_lookup:
            print(f"[WARNING] Missing basin for delta {polygon_id} — skipping.")
            deltas_without_sources.append(row.copy())
            fail_count += 1
            continue

        delta_basins_gpd: GeoDataFrame = basin_lookup[polygon_id]
        delta_edmonds: GeoDataFrame = delta_lookup[polygon_id]
        basin_polygons_domain: GeoDataFrame = basin_polygons_lookup[polygon_id]

        # --- Per-delta spatial subset (no file I/O) ---
        try:
            rivers_gpd, coast_polygon, coastline_gpd, glofas_min = (
                load_data_delta_domain(delta_basins_gpd, global_data)
            )
        except Exception as e:
            print(f"[ERROR] Failed subsetting data for delta {polygon_id}: {e}")
            deltas_without_sources.append(row.copy())
            fail_count += 1
            continue

        # --- Source point extraction ---
        unique_sources, possible_sources, _ = find_river_source_points(
            delta_basins_gpd=delta_basins_gpd,
            delta_polygon=delta_polygon,
            rivers_gpd=rivers_gpd,
            glofas_min=glofas_min,
            coast_polygon=coast_polygon,
            delta_edmonds=delta_edmonds,
            basin_polygons=river_basins_gpd,
            basin_polygons_domain=basin_polygons_domain,
            all_rivers=all_rivers,
        )

        # --- Release per-delta intermediates explicitly ---
        del rivers_gpd, coast_polygon, coastline_gpd, glofas_min

        # --- Validation ---
        if _is_empty(unique_sources) or _is_empty(possible_sources):
            deltas_without_sources.append(row.copy())
            fail_count += 1
            continue

        assert unique_sources is not None and possible_sources is not None

        unique_sources[BASIN_COL] = polygon_id
        possible_sources[BASIN_COL] = polygon_id

        all_unique_sources.append(unique_sources)
        all_possible_sources.append(possible_sources)
        success_count += 1

        # --- Periodic flush to disk to cap memory usage ---
        if success_count % _FLUSH_INTERVAL == 0:
            print(
                f"[MEM] Flushing {len(all_unique_sources)} results to disk ...",
                flush=True,
            )
            _flush_to_disk(all_unique_sources, out_unique_sources, CRS_STANDARD)
            _flush_to_disk(all_possible_sources, out_possible_sources, CRS_STANDARD)
            all_unique_sources.clear()
            all_possible_sources.clear()
            gc.collect()
            print("[MEM] Flush complete.", flush=True)

    # --- Summary ---
    print(f"Processed: {success_count} domains | {fail_count} without sources.")

    # --- Final flush of any remaining results ---
    _flush_to_disk(all_unique_sources, out_unique_sources, CRS_STANDARD)
    _flush_to_disk(all_possible_sources, out_possible_sources, CRS_STANDARD)

    if deltas_without_sources:
        gpd.GeoDataFrame(
            pd.DataFrame(deltas_without_sources),
            geometry=GEOM_COL,
            crs=CRS_STANDARD,
        ).to_file(Path(out_deltas_no_rivers), driver="GPKG")


def extract_points() -> None:
    extract_all_river_source_points()
