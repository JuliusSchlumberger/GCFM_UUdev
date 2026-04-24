"""
Diagnostic wrapper for extract_all_river_source_points.

Drop-in replacement for the main loop that logs every sub-step with timing.
Once you identify which step hangs, you can narrow the fix to that function.

Usage:
    Replace the call to extract_all_river_source_points() with
    extract_all_river_source_points_debug() in your run script.
    Or import and call run_diagnostics() directly.
"""

from __future__ import annotations

import signal
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Polygon

from src.input_processing.config.loader import config
from src.input_processing.utils.util_unify_typing_and_schema import (
    ensure_valid_schema,
    CRS_STANDARD,
    BASIN_COL,
    GEOM_COL,
)
from src.input_processing.utils.loading_files import (
    load_global_data,
    load_data_delta_domain,
)
from src.input_processing.utils.preprocess_02_ut_extract_river_points import (
    extract_cells_within_delta,
    clip_basin_boundary_from_coast,
)
from src.input_processing.workflows.preprocess_02_wf_extract_river_points import (
    _is_empty,
)


# ---------------------------------------------------------------------------
# Timeout context manager (Unix only — skip on Windows)
# ---------------------------------------------------------------------------


class StepTimeout(Exception):
    pass


@contextmanager
def timeout(seconds: int, label: str) -> Generator[None, None, None]:
    """
    Raise StepTimeout if the block takes longer than *seconds*.
    Falls back to a no-op on Windows where SIGALRM is unavailable.
    """

    def _handler(signum: int, frame: object) -> None:
        raise StepTimeout(f"[TIMEOUT] '{label}' exceeded {seconds}s")

    try:
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        yield
    except AttributeError:
        # Windows: signal.SIGALRM not available — yield without timeout
        yield
    finally:
        try:
            signal.alarm(0)
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Step timer
# ---------------------------------------------------------------------------


def _step(label: str, polygon_id: int | str) -> None:
    print(f"  [{polygon_id}] >>> {label}", flush=True)


def _done(label: str, polygon_id: int | str, t0: float) -> None:
    elapsed = time.perf_counter() - t0
    print(f"  [{polygon_id}] ... {label} done ({elapsed:.2f}s)", flush=True)


# ---------------------------------------------------------------------------
# Diagnostic main loop
# ---------------------------------------------------------------------------


def extract_all_river_source_points_debug(
    out_unique_sources: str = config["filepaths"]["unique_sources"],
    out_possible_sources: str = config["filepaths"]["possible_sources"],
    out_deltas_no_rivers: str = config["filepaths"]["out_deltas_no_rivers"],
    step_timeout_seconds: int = 120,
    max_deltas: int | None = None,  # set to e.g. 5 to test a short run
) -> None:
    """
    Diagnostic version of extract_all_river_source_points.

    Logs every sub-step with wall-clock timing and wraps each in a timeout
    so a hung operation is caught rather than blocking forever.

    Args:
        step_timeout_seconds: Seconds before a single step is considered hung.
        max_deltas:           If set, stop after this many deltas (for quick tests).
    """

    # -----------------------------------------------------------------------
    # Step 0 — global load
    # -----------------------------------------------------------------------
    print("=== [DIAG] Loading global data ===", flush=True)
    t = time.perf_counter()
    global_data = load_global_data()
    print(
        f"=== [DIAG] Global data loaded ({time.perf_counter() - t:.1f}s) ===\n",
        flush=True,
    )

    delta_domains: GeoDataFrame = ensure_valid_schema(
        gpd.read_file(config["filepaths"]["delta_polygons_used"]), excluded=[]
    )
    river_basins_gpd: GeoDataFrame = ensure_valid_schema(
        gpd.read_file(config["filepaths"]["new_domains"]), excluded=[]
    )
    all_rivers: GeoDataFrame = global_data.rivers

    basin_lookup: dict[int | str, GeoDataFrame] = {
        k: v.copy() for k, v in river_basins_gpd.groupby(BASIN_COL)
    }
    delta_lookup: dict[int | str, GeoDataFrame] = {
        k: v.copy() for k, v in delta_domains.groupby(BASIN_COL)
    }
    basin_polygons_lookup = basin_lookup

    all_unique_sources: list[GeoDataFrame] = []
    all_possible_sources: list[GeoDataFrame] = []
    deltas_without_sources: list[pd.Series] = []
    success_count = fail_count = 0

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    for loop_i, (idx, row) in enumerate(delta_domains.iterrows()):
        if max_deltas is not None and loop_i >= max_deltas:
            print(f"[DIAG] Reached max_deltas={max_deltas}, stopping early.")
            break

        polygon_id: int | str = row[BASIN_COL]
        delta_polygon: Polygon = row[GEOM_COL]

        print(f"\n--- Delta {loop_i + 1} | id={polygon_id} | row={idx} ---", flush=True)

        # -------------------------------------------------------------------
        # Step 1 — basin lookup
        # -------------------------------------------------------------------
        if polygon_id not in basin_lookup:
            print(f"  [SKIP] No basin found for {polygon_id}")
            deltas_without_sources.append(row.copy())
            fail_count += 1
            continue

        delta_basins_gpd = basin_lookup[polygon_id]
        delta_edmonds = delta_lookup[polygon_id]
        basin_polygons_domain = basin_polygons_lookup[polygon_id]

        # -------------------------------------------------------------------
        # Step 2 — spatial subset
        # -------------------------------------------------------------------
        _step("load_data_delta_domain", polygon_id)
        t = time.perf_counter()
        try:
            with timeout(step_timeout_seconds, "load_data_delta_domain"):
                rivers_gpd, coast_polygon, coastline_gpd, glofas_min = (
                    load_data_delta_domain(delta_basins_gpd, global_data)
                )
        except StepTimeout as e:
            print(f"  [HANG] {e}")
            fail_count += 1
            deltas_without_sources.append(row.copy())
            continue
        except Exception as e:
            print(f"  [ERROR] load_data_delta_domain: {e}")
            traceback.print_exc()
            fail_count += 1
            deltas_without_sources.append(row.copy())
            continue
        _done("load_data_delta_domain", polygon_id, t)

        # Log subset sizes — abnormally large subsets are a common hang cause
        print(
            f"  [DIAG] rivers={len(rivers_gpd)} | "
            f"coastline={len(coastline_gpd)} | "
            f"glofas shape={glofas_min.shape}",
            flush=True,
        )

        # -------------------------------------------------------------------
        # Step 3 — inland boundary
        # -------------------------------------------------------------------
        _step("clip_basin_boundary_from_coast", polygon_id)
        t = time.perf_counter()
        try:
            with timeout(step_timeout_seconds, "clip_basin_boundary_from_coast"):
                inland_boundary = clip_basin_boundary_from_coast(
                    delta_basins_gpd, coast_polygon
                )
        except StepTimeout as e:
            print(f"  [HANG] {e}")
            fail_count += 1
            deltas_without_sources.append(row.copy())
            continue
        except ValueError as e:
            print(f"  [SKIP] boundary: {e}")
            fail_count += 1
            deltas_without_sources.append(row.copy())
            continue
        except Exception as e:
            print(f"  [ERROR] clip_basin_boundary_from_coast: {e}")
            traceback.print_exc()
            fail_count += 1
            deltas_without_sources.append(row.copy())
            continue
        _done("clip_basin_boundary_from_coast", polygon_id, t)

        # -------------------------------------------------------------------
        # Step 4 — river clip (sjoin)
        # -------------------------------------------------------------------
        _step("sjoin rivers", polygon_id)
        t = time.perf_counter()
        try:
            with timeout(step_timeout_seconds, "sjoin rivers"):
                relevant_rivers: GeoDataFrame = gpd.sjoin(
                    rivers_gpd, delta_basins_gpd, how="inner"
                )
                relevant_rivers = relevant_rivers[rivers_gpd.columns].copy()
                relevant_rivers[BASIN_COL] = delta_basins_gpd[BASIN_COL].iloc[0]
        except StepTimeout as e:
            print(f"  [HANG] {e}")
            fail_count += 1
            deltas_without_sources.append(row.copy())
            continue
        except Exception as e:
            print(f"  [ERROR] sjoin: {e}")
            traceback.print_exc()
            fail_count += 1
            deltas_without_sources.append(row.copy())
            continue
        _done("sjoin rivers", polygon_id, t)
        print(f"  [DIAG] relevant_rivers={len(relevant_rivers)}", flush=True)

        # -------------------------------------------------------------------
        # Step 5 — extract_cells_within_delta
        # -------------------------------------------------------------------
        _step("extract_cells_within_delta", polygon_id)
        t = time.perf_counter()
        try:
            with timeout(step_timeout_seconds, "extract_cells_within_delta"):
                unique_sources, possible_sources, _ = extract_cells_within_delta(
                    glofas_min,
                    inland_boundary,
                    relevant_rivers,
                    delta_edmonds,
                    river_basins_gpd,
                    basin_polygons_domain,
                    all_rivers,
                )
        except StepTimeout as e:
            print(f"  [HANG] {e}")
            fail_count += 1
            deltas_without_sources.append(row.copy())
            continue
        except Exception as e:
            print(f"  [ERROR] extract_cells_within_delta: {e}")
            traceback.print_exc()
            fail_count += 1
            deltas_without_sources.append(row.copy())
            continue
        _done("extract_cells_within_delta", polygon_id, t)

        # -------------------------------------------------------------------
        # Step 6 — validation and accumulate
        # -------------------------------------------------------------------
        if _is_empty(unique_sources) or _is_empty(possible_sources):
            print(f"  [SKIP] Empty sources for {polygon_id}")
            deltas_without_sources.append(row.copy())
            fail_count += 1
            continue

        assert unique_sources is not None and possible_sources is not None
        unique_sources[BASIN_COL] = polygon_id
        possible_sources[BASIN_COL] = polygon_id
        all_unique_sources.append(unique_sources)
        all_possible_sources.append(possible_sources)
        success_count += 1
        print(
            f"  [OK] unique={len(unique_sources)} | possible={len(possible_sources)}",
            flush=True,
        )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    print(f"\n=== [DIAG] Done: {success_count} success | {fail_count} failed ===")

    if all_unique_sources:
        gpd.GeoDataFrame(
            pd.concat(all_unique_sources, ignore_index=True), crs=CRS_STANDARD
        ).to_file(Path(out_unique_sources), driver="GPKG")

    if all_possible_sources:
        gpd.GeoDataFrame(
            pd.concat(all_possible_sources, ignore_index=True), crs=CRS_STANDARD
        ).to_file(Path(out_possible_sources), driver="GPKG")

    if deltas_without_sources:
        gpd.GeoDataFrame(
            pd.DataFrame(deltas_without_sources), geometry=GEOM_COL, crs=CRS_STANDARD
        ).to_file(Path(out_deltas_no_rivers), driver="GPKG")
