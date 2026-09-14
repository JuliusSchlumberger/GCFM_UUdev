"""
replot_event_figures.py — redraw an event run's figures from its existing
SFINCS output, without rerunning SFINCS.

Rule run_event (16_run_event.py) draws its figures right after the SFINCS
run, and Snakemake only tracks a rule's own script/params -- not the plotting
code in workflow/src/plots.py -- so a plotting change never reaches runs that
already exist unless the whole event run is forced again. This redraws, per
scenario, exactly what rule 16 draws, with the same functions and config
settings, from runs/{scenario}/sfincs/sfincs_map.nc:

    runs/{scenario}/visuals/01_inundation_ratio.png   (plot_inundation_check)
    runs/{scenario}/visuals/02_flood_animation.mp4    (animate_flood_progression)

flood_timeseries.csv and the metrics are left untouched (they don't depend on
plotting). Overwriting the PNG/MP4 in place does not make Snakemake rerun
anything: nothing downstream consumes them.

Usage:
    conda run -n hmt_sfincs_dev python tools/replot_event_figures.py 2433835 coast_500 river_500 compound_500
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
import yaml
from shapely.geometry import Polygon

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "workflow"))
from src.plots import animate_flood_progression, plot_inundation_check  # noqa: E402
from src.postprocessing import compute_flood_progression, compute_max_inundation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
for _name in ("hydromt", "hydromt_sfincs"):
    logging.getLogger(_name).setLevel(logging.WARNING)


def replot(basin_id: str, scenario: str, config: dict) -> None:
    results = Path(config["results_dir"]) / basin_id
    domain = results / "preprocessing_inputs" / "domain"
    sfincs_root = results / "runs" / scenario / "sfincs"
    skeleton_root = results / "sfincs_skeleton"
    visuals = results / "runs" / scenario / "visuals"
    sea_mask_path = domain / f"{basin_id}_zsini_sea_cells_on_grid.tif"
    land_polygons_path = (
        domain / f"{basin_id}_land_mask_on_grid.gpkg"
    )  # grid-aligned land mask (rule 09b)
    river_network_path = domain / f"{basin_id}_river_network_clean.gpkg"
    if not (sfincs_root / "sfincs_map.nc").exists():
        log.warning(
            f"[{scenario}] no sfincs_map.nc under {sfincs_root} -- skipped (run the event first)"
        )
        return

    hmin = float(config["sfincs"]["sanity_checks"]["min_inundation_depth_m"])
    include_subgrid = bool(config["sfincs"]["subgrid"]["enabled"])
    fps = int(config["sfincs"]["sanity_checks"]["animation_fps"])

    # Same domain polygon handling as 16_run_event.py.
    domain_gdf = gpd.read_file(domain / f"{basin_id}_domain.gpkg")
    if domain_gdf.crs is not None and domain_gdf.crs.to_epsg() != 4326:
        domain_gdf = domain_gdf.to_crs("EPSG:4326")
    union = domain_gdf.geometry.union_all()
    domain_poly = cast(
        Polygon, union if isinstance(union, Polygon) else union.convex_hull
    )

    # 01_inundation_ratio.png -- mirrors 16_run_event.py's Check 1.
    da_hmax, da_dep = compute_max_inundation(
        sfincs_root,
        skeleton_root,
        sea_mask_path,
        hmin=hmin,
        include_subgrid=include_subgrid,
    )
    if da_hmax is None or da_dep is None:
        log.warning(
            f"[{scenario}] could not compute max inundation -- 01_inundation_ratio.png not redrawn"
        )
    else:
        n_land = int(da_dep.notnull().sum().item())
        n_flooded = int(da_hmax.notnull().sum().item())
        plot_inundation_check(
            da_hmax,
            hmin,
            n_flooded,
            n_land,
            str(land_polygons_path),
            str(river_network_path),
            str(visuals / "01_inundation_ratio.png"),
            basin_id=basin_id,
            run_label="event",
        )
        log.info(
            f"[{scenario}] 01_inundation_ratio.png redrawn ({n_flooded:,}/{n_land:,} land pixels flooded)"
        )

    # 02_flood_animation.mp4
    da_h = compute_flood_progression(sfincs_root, sea_mask_path)
    if da_h is None:
        log.warning(
            f"[{scenario}] no 'zs' in sfincs_map.nc -- 02_flood_animation.mp4 not redrawn"
        )
        return
    animate_flood_progression(
        da_h,
        domain_poly,
        str(land_polygons_path),
        str(river_network_path),
        str(visuals / "02_flood_animation.mp4"),
        basin_id=basin_id,
        run_label="event",
        fps=fps,
    )
    log.info(
        f"[{scenario}] 02_flood_animation.mp4 redrawn ({int(np.prod(da_h.shape[:1]))} frame(s))"
    )


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(
            "usage: python tools/replot_event_figures.py <basin_id> <scenario> [<scenario> ...]"
        )
    with open(REPO_ROOT / "config" / "config.yml") as fh:
        config = yaml.safe_load(fh)
    basin_id, scenarios = sys.argv[1], sys.argv[2:]
    for scenario in scenarios:
        replot(basin_id, scenario, config)


if __name__ == "__main__":
    main()
