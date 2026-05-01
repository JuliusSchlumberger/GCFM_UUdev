"""Diagnostic wrapper for extract_all_river_source_points.

Drop-in replacement for the main loop that logs every sub-step with timing.
Once you identify which step hangs, you can narrow the fix to that function.

Usage:
    Replace the call to extract_all_river_source_points() with
    extract_all_river_source_points_debug() in your run script.
    Or import and call run_diagnostics() directly.
"""
#
# from __future__ import annotations
#
# import signal
# import time
# import traceback
# from contextlib import contextmanager
# from pathlib import Path
# from typing import Generator, Final
# from collections.abc import Hashable
#
# import geopandas as gpd
# import pandas as pd
# from geopandas import GeoDataFrame
# from shapely.geometry import Polygon
#
# from src.utils.config_loader import load_config
# from src.input_processing.utils.util_unify_typing_and_schema import (
#     ensure_valid_schema,
#     CRS_STANDARD,
#     BASIN_COL,
#     GEOM_COL,
# )
# from src.input_processing.utils.loading_files import (
#     load_global_data,
#     load_data_delta_domain,
# )
# from src.input_processing.utils.preprocess_02_ut_extract_river_points import (
#     extract_cells_within_delta,
#     clip_basin_boundary_from_coast,
# )
# from src.input_processing.workflows.preprocess_02_wf_extract_river_points import (
#     _is_empty,
# )
#
# _CONFIG_PATH = "../src/input_processing/config/decisions.yaml"
# _CONFIG: Final = load_config(_CONFIG_PATH)
#
#
# # ---------------------------------------------------------------------------
# # Timeout context manager (cross-platform safe)
# # ---------------------------------------------------------------------------
#
#
# class StepTimeout(Exception):
#     pass
#
#
# @contextmanager
# def timeout(seconds: int, label: str) -> Generator[None, None, None]:
#     def _handler(signum: int, frame: object) -> None:
#         raise StepTimeout(f"[TIMEOUT] '{label}' exceeded {seconds}s")
#
#     if hasattr(signal, "SIGALRM"):
#         signal.signal(signal.SIGALRM, _handler)
#         signal.alarm(seconds)
#         try:
#             yield
#         finally:
#             signal.alarm(0)
#     else:
#         # Windows fallback: no timeout
#         yield
#
#
# # ---------------------------------------------------------------------------
# # Logging helpers
# # ---------------------------------------------------------------------------
#
#
# def _step(label: str, polygon_id: int | str) -> None:
#     print(f"  [{polygon_id}] >>> {label}", flush=True)
#
#
# def _done(label: str, polygon_id: int | str, t0: float) -> None:
#     elapsed = time.perf_counter() - t0
#     print(f"  [{polygon_id}] ... {label} done ({elapsed:.2f}s)", flush=True)
#
#
# # ---------------------------------------------------------------------------
# # Main diagnostic function
# # ---------------------------------------------------------------------------
#
#
# def extract_all_river_source_points_debug(
#     out_unique_sources: str = _CONFIG["filepaths"]["unique_sources"],
#     out_possible_sources: str = _CONFIG["filepaths"]["possible_sources"],
#     out_deltas_no_rivers: str = _CONFIG["filepaths"]["out_deltas_no_rivers"],
#     step_timeout_seconds: int = 120,
#     max_deltas: int | None = None,
# ) -> None:
#     # -----------------------------------------------------------------------
#     # Step 0 — global load
#     # -----------------------------------------------------------------------
#     print("=== [DIAG] Loading global data ===", flush=True)
#     t = time.perf_counter()
#     global_data = load_global_data()
#     print(f"=== [DIAG] Global data loaded ({time.perf_counter() - t:.1f}s) ===\n")
#
#     delta_domains: GeoDataFrame = ensure_valid_schema(
#         gpd.read_file(_CONFIG["filepaths"]["delta_polygons_used"]), excluded=[]
#     )
#
#     river_basins_gpd: GeoDataFrame = ensure_valid_schema(
#         gpd.read_file(_CONFIG["filepaths"]["new_domains"]), excluded=[]
#     )
#
#     all_rivers: GeoDataFrame = global_data.rivers
#
#     # -----------------------------------------------------------------------
#     # Lookups (correct typing)
#     # -----------------------------------------------------------------------
#     basin_lookup: dict[Hashable, GeoDataFrame] = {
#         k: v.copy() for k, v in river_basins_gpd.groupby(BASIN_COL)
#     }
#
#     delta_lookup: dict[Hashable, GeoDataFrame] = {
#         k: v.copy() for k, v in delta_domains.groupby(BASIN_COL)
#     }
#
#     basin_polygons_lookup = basin_lookup
#
#     all_unique_sources: list[GeoDataFrame] = []
#     all_possible_sources: list[GeoDataFrame] = []
#     deltas_without_sources: list[pd.Series] = []
#
#     success_count = 0
#     fail_count = 0
#
#     # -----------------------------------------------------------------------
#     # Main loop (itertuples = faster + typed)
#     # -----------------------------------------------------------------------
#     for loop_i, row in enumerate(delta_domains.itertuples(index=True)):
#         if max_deltas is not None and loop_i >= max_deltas:
#             print(f"[DIAG] Reached max_deltas={max_deltas}, stopping early.")
#             break
#
#         idx = row.Index
#         polygon_id = getattr(row, BASIN_COL)
#         delta_polygon = getattr(row, GEOM_COL)
#
#         if not isinstance(delta_polygon, Polygon):
#             print(f"  [SKIP] Invalid geometry for {polygon_id}")
#             continue
#
#         print(f"\n--- Delta {loop_i + 1} | id={polygon_id} | row={idx} ---")
#
#         # -------------------------------------------------------------------
#         # Step 1 — basin lookup
#         # -------------------------------------------------------------------
#         if polygon_id not in basin_lookup:
#             print(f"  [SKIP] No basin found for {polygon_id}")
#             deltas_without_sources.append(pd.Series(row._asdict()))
#             fail_count += 1
#             continue
#
#         delta_basins_gpd = basin_lookup[polygon_id]
#         delta_edmonds = delta_lookup[polygon_id]
#         basin_polygons_domain = basin_polygons_lookup[polygon_id]
#
#         # -------------------------------------------------------------------
#         # Step 2 — spatial subset
#         # -------------------------------------------------------------------
#         _step("load_data_delta_domain", polygon_id)
#         t = time.perf_counter()
#
#         try:
#             with timeout(step_timeout_seconds, "load_data_delta_domain"):
#                 rivers_gpd, coast_polygon, coastline_gpd, glofas_min = (
#                     load_data_delta_domain(delta_basins_gpd, global_data)
#                 )
#         except Exception as e:
#             print(f"  [ERROR] load_data_delta_domain: {e}")
#             traceback.print_exc()
#             fail_count += 1
#             deltas_without_sources.append(pd.Series(row._asdict()))
#             continue
#
#         _done("load_data_delta_domain", polygon_id, t)
#
#         print(
#             f"  [DIAG] rivers={len(rivers_gpd)} | "
#             f"coastline={len(coastline_gpd)} | "
#             f"glofas shape={glofas_min.shape}"
#         )
#
#         # -------------------------------------------------------------------
#         # Step 3 — inland boundary
#         # -------------------------------------------------------------------
#         _step("clip_basin_boundary_from_coast", polygon_id)
#         t = time.perf_counter()
#
#         try:
#             with timeout(step_timeout_seconds, "clip_basin_boundary_from_coast"):
#                 inland_boundary = clip_basin_boundary_from_coast(
#                     delta_basins_gpd, coast_polygon
#                 )
#         except Exception as e:
#             print(f"  [ERROR] clip_basin_boundary_from_coast: {e}")
#             traceback.print_exc()
#             fail_count += 1
#             deltas_without_sources.append(pd.Series(row._asdict()))
#             continue
#
#         _done("clip_basin_boundary_from_coast", polygon_id, t)
#
#         # -------------------------------------------------------------------
#         # Step 4 — spatial join
#         # -------------------------------------------------------------------
#         _step("sjoin rivers", polygon_id)
#         t = time.perf_counter()
#
#         try:
#             with timeout(step_timeout_seconds, "sjoin rivers"):
#                 relevant_rivers = gpd.sjoin(rivers_gpd, delta_basins_gpd, how="inner")
#                 relevant_rivers = relevant_rivers[rivers_gpd.columns].copy()
#                 relevant_rivers[BASIN_COL] = polygon_id
#         except Exception as e:
#             print(f"  [ERROR] sjoin: {e}")
#             traceback.print_exc()
#             fail_count += 1
#             deltas_without_sources.append(pd.Series(row._asdict()))
#             continue
#
#         _done("sjoin rivers", polygon_id, t)
#
#         print(f"  [DIAG] relevant_rivers={len(relevant_rivers)}")
#
#         # -------------------------------------------------------------------
#         # Step 5 — extraction
#         # -------------------------------------------------------------------
#         _step("extract_cells_within_delta", polygon_id)
#         t = time.perf_counter()
#
#         try:
#             with timeout(step_timeout_seconds, "extract_cells_within_delta"):
#                 unique_sources, possible_sources, _ = extract_cells_within_delta(
#                     glofas_min,
#                     inland_boundary,
#                     relevant_rivers,
#                     delta_edmonds,
#                     river_basins_gpd,
#                     basin_polygons_domain,
#                     all_rivers,
#                 )
#         except Exception as e:
#             print(f"  [ERROR] extract_cells_within_delta: {e}")
#             traceback.print_exc()
#             fail_count += 1
#             deltas_without_sources.append(pd.Series(row._asdict()))
#             continue
#
#         _done("extract_cells_within_delta", polygon_id, t)
#
#         # -------------------------------------------------------------------
#         # Step 6 — validation
#         # -------------------------------------------------------------------
#         if _is_empty(unique_sources) or _is_empty(possible_sources):
#             print(f"  [SKIP] Empty sources for {polygon_id}")
#             deltas_without_sources.append(pd.Series(row._asdict()))
#             fail_count += 1
#             continue
#
#         unique_sources[BASIN_COL] = polygon_id
#         possible_sources[BASIN_COL] = polygon_id
#
#         all_unique_sources.append(unique_sources)
#         all_possible_sources.append(possible_sources)
#
#         success_count += 1
#
#         print(f"  [OK] unique={len(unique_sources)} | possible={len(possible_sources)}")
#
#     # -----------------------------------------------------------------------
#     # Save results
#     # -----------------------------------------------------------------------
#     print(f"\n=== [DIAG] Done: {success_count} success | {fail_count} failed ===")
#
#     if all_unique_sources:
#         gpd.GeoDataFrame(
#             pd.concat(all_unique_sources, ignore_index=True),
#             crs=CRS_STANDARD,
#         ).to_file(Path(out_unique_sources), driver="GPKG")
#
#     if all_possible_sources:
#         gpd.GeoDataFrame(
#             pd.concat(all_possible_sources, ignore_index=True),
#             crs=CRS_STANDARD,
#         ).to_file(Path(out_possible_sources), driver="GPKG")
#
#     if deltas_without_sources:
#         gpd.GeoDataFrame(
#             pd.DataFrame(deltas_without_sources),
#             geometry=GEOM_COL,
#             crs=CRS_STANDARD,
#         ).to_file(Path(out_deltas_no_rivers), driver="GPKG")
