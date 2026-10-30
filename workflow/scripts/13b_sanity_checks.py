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
import numpy as np
import xarray as xr
from shapely.geometry import Polygon
from src.log import setup_logging
from src.plots import (
    animate_flood_progression,
    plot_inundation_check,
    plot_water_level_timeseries,
)
from src.postprocessing import (
    compute_flood_progression,
    compute_max_inundation,
    inundation_stats,
    read_land_bed_level,
)

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
    try:
        ds = xr.open_dataset(inp.sfincs_his)
    except ValueError:
        # Time units SFINCS wrote that xarray cannot decode: use raw seconds.
        ds = xr.open_dataset(inp.sfincs_his, decode_times=False)
    t = ds["time"].values
    if np.issubdtype(t.dtype, np.datetime64):
        t_days = (t - t[0]) / np.timedelta64(1, "D")
    else:
        t_days = np.asarray(t, dtype=float) / 86400.0
    zs_var = next((v for v in ("point_zs", "zs") if v in ds), None)
    if zs_var is None:
        raise KeyError(
            f"No water-level variable (point_zs, zs) in {inp.sfincs_his}; available: {list(ds.data_vars)}"
        )
    zs = ds[zs_var].values
    zs = zs[:, np.newaxis] if zs.ndim == 1 else zs
    zs = zs.T if zs.shape[0] != len(t_days) else zs
    spinup_days = params.spinup_days
    plot_water_level_timeseries(
        t_days,
        zs,
        str(out.plot_water_level),
        day_markers=[(spinup_days, f"Day {spinup_days} (restart written)")],
        basin_id=f"Basin {basin_id}",
        run_label=f"Spin-up validation | lines should be near-flat at day {spinup_days} if spin-up is sufficient",
        ylabel=f"Water level — {zs_var} (m)",
    )
log.info("Done")
