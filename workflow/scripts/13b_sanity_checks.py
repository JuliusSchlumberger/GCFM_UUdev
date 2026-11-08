"""Draw the sanity-check figures of a finished SFINCS run from its outputs.

Run by rules sanity_checks (spin-up) and analysis_run_event (each scenario).
Inundation ratio: fraction of land-domain pixels (non-null in the sea-masked,
subgrid-downscaled bed level) whose max inundation depth exceeds
min_inundation_depth_m. When declared, also the flood-progression animation
and the water level at the observation points (sfincs_his.nc).
"""

from pathlib import Path
from typing import cast

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from src.log import setup_logging
from src.plots import (
    animate_flood_progression,
    plot_inundation_check,
    plot_observation_levels,
)
from src.postprocessing import (
    compute_flood_progression,
    compute_max_inundation,
    inundation_stats,
    read_land_bed_level,
)
from src.sfincs_run import read_observation_levels

log = setup_logging(snakemake.log[0])
inp, out, params = snakemake.input, snakemake.output, snakemake.params
run_root = Path(params.run_root)
run = str(params.run_label)
basin_id = snakemake.wildcards.basin_id
land_polygons_path = str(inp.land_mask_on_grid)
river_network_path = str(inp.clean_river_network)
threshold_m = float(params.min_inundation_depth_m)

_union = gpd.read_file(inp.domain_gpkg).to_crs("EPSG:4326").geometry.union_all()
domain_poly = cast(
    Polygon, _union if isinstance(_union, Polygon) else _union.convex_hull
)

da_hmax, da_dep = compute_max_inundation(
    run_root, read_land_bed_level(inp.land_bed_level), hmin=threshold_m
)
if da_hmax is None or da_dep is None:
    raise RuntimeError(f"No 'zsmax' or bed level in the SFINCS output of {run_root}")
stats = inundation_stats(da_hmax, da_dep)
log.info(
    f"[{run}] Inundation ratio  hmin={threshold_m} m  "
    f"flooded={stats['n_flooded']:,}/{stats['n_land']:,} pixels ({stats['frac']:.2%})  "
    f"area={stats['flooded_km2']:.1f}/{stats['land_km2']:.1f} km²"
)
plot_inundation_check(
    da_hmax,
    threshold_m,
    stats["n_flooded"],
    stats["n_land"],
    land_polygons_path,
    river_network_path,
    str(out.plot_inundation_ratio),
    basin_id=basin_id,
    run_label=run,
)

if hasattr(out, "animation_flood_progress"):
    # Needs zs in sfincs_map.nc (finite dtmapout).
    da_h = compute_flood_progression(run_root, inp.sea_mask)
    if da_h is None:
        raise RuntimeError(
            f"No 'zs' or 'zb' in the SFINCS output of {run_root}; dtmapout must be set"
        )
    animate_flood_progression(
        da_h,
        domain_poly,
        land_polygons_path,
        river_network_path,
        str(out.animation_flood_progress),
        basin_id=basin_id,
        run_label=run,
        fps=int(params.animation_fps),
    )

if hasattr(out, "plot_water_level"):
    t_hr, zs = read_observation_levels(Path(inp.sfincs_his).parent)
    spinup_hr = float(pd.read_csv(inp.stability)["spinup_hr"].iloc[0])
    plot_observation_levels(
        t_hr,
        zs,
        gpd.read_file(inp.obs_points),
        str(out.plot_water_level),
        markers_hr=[(spinup_hr, "end of the spin-up, restart written; lines should be flat before it")],
    )
log.info("Done")
