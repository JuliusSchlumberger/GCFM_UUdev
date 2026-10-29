"""Run the SFINCS event simulation and write its diagnostics.

Run by rule run_event (and adapt_run_event_pre). Executes the sfincs.inp from
build_sfincs in sfincs_root: tstop = end of forcing; with spin-up, tstart =
spin-up end and rstfile = the basin-level restart; without, tstart = tref from
a dry bed.

Writes the inundation-ratio map (zsmax envelope, as rule sanity_checks), the
flood animation and flood_timeseries.csv. flood_volume_m3 is the total volume
at each output step (instantaneous depth × pixel area), so its max is the true
peak; summing each pixel's zsmax would overstate it.
"""

import gc
from pathlib import Path
from typing import cast

import geopandas as gpd
from shapely.geometry import Polygon

from src.log import setup_logging
from src.plots import animate_flood_progression, plot_inundation_check
from src.postprocessing import (
    compute_flood_progression,
    compute_flood_timeseries_stats,
    compute_max_inundation,
    inundation_stats,
    read_land_bed_level,
)
from src.sfincs_run import run_sfincs_subprocess

log = setup_logging(snakemake.log[0])

# ── params ──
sfincs_root                = Path(snakemake.params.sfincs_root)
sfincs_exe                 = Path(snakemake.params.sfincs_exe)
timeout_s                  = int(snakemake.params.timeout_s)
min_inundation_depth_m     = float(snakemake.params.min_inundation_depth_m)
animation_fps               = int(snakemake.params.animation_fps)
land_polygons_path         = Path(snakemake.input.land_mask_on_grid)  # plot background
sea_mask_path               = Path(snakemake.input.sea_mask)
da_land_dep                 = read_land_bed_level(snakemake.input.land_bed_level)
river_network_path          = Path(snakemake.input.clean_river_network)
domain_gpkg_path            = Path(snakemake.input.domain_gpkg)

event_dir = Path(snakemake.output.plot_inundation_ratio).parent
event_dir.mkdir(parents=True, exist_ok=True)

_domain_gdf = gpd.read_file(domain_gpkg_path)
if _domain_gdf.crs is not None and _domain_gdf.crs.to_epsg() != 4326:
    _domain_gdf = _domain_gdf.to_crs("EPSG:4326")
_union = _domain_gdf.geometry.union_all()
domain_poly = cast(Polygon, _union if isinstance(_union, Polygon) else _union.convex_hull)
basin_id = sfincs_root.parent.name

# ── run SFINCS ──
# cwd = sfincs_root so the relative file references in sfincs.inp resolve.
run_sfincs_subprocess(sfincs_exe, sfincs_root, timeout_s, log, label="SFINCS event run", n_threads=snakemake.threads)

sfincs_map_path = Path(snakemake.output.sfincs_map_nc)
if not sfincs_map_path.exists():
    raise FileNotFoundError(f"SFINCS ran but expected map output not found: {sfincs_map_path}")
log.info(f"Event map output written: {sfincs_map_path} ({sfincs_map_path.stat().st_size / 1e6:.1f} MB)")

# ── Check 1: inundation ratio (envelope, zsmax), as in 13b_sanity_checks.py ──
plot_ratio_path = Path(snakemake.output.plot_inundation_ratio)

da_hmax, da_dep = compute_max_inundation(
    sfincs_root, da_land_dep, hmin=min_inundation_depth_m,
)
if da_hmax is None or da_dep is None:
    raise RuntimeError(f"No 'zsmax' or bed level in the SFINCS event output of {sfincs_root}")
_stats    = inundation_stats(da_hmax, da_dep)
n_land    = _stats["n_land"]
n_flooded = _stats["n_flooded"]
frac      = _stats["frac"]
flooded_km2 = _stats["flooded_km2"]
land_km2    = _stats["land_km2"]

log.info(
    f"[Check 1] Inundation ratio  hmin={min_inundation_depth_m} m  "
    f"flooded={n_flooded:,}/{n_land:,} pixels ({frac:.2%})  "
    f"area={flooded_km2:.1f}/{land_km2:.1f} km²"
)

plot_inundation_check(
    da_hmax, min_inundation_depth_m, n_flooded, n_land,
    str(land_polygons_path), str(river_network_path),
    str(plot_ratio_path), basin_id=basin_id,
    run_label="event",
)
log.info(f"Inundation ratio plot written: {plot_ratio_path}")

# ── flood progression: animation + per-timestep area/volume CSV ──
animation_out_path = Path(snakemake.output.animation_flood_progress)
csv_out_path        = Path(snakemake.output.flood_timeseries_csv)
animation_out_path.parent.mkdir(parents=True, exist_ok=True)

da_h = compute_flood_progression(sfincs_root, sea_mask_path)
if da_h is None:
    raise RuntimeError(f"No 'zs' or 'zb' in the SFINCS event output of {sfincs_root}")
animate_flood_progression(
    da_h, domain_poly,
    str(land_polygons_path), str(river_network_path),
    str(animation_out_path),
    basin_id=basin_id,
    run_label="event",
    fps=animation_fps,
)
log.info(f"Flood animation written: {animation_out_path}")

# Free da_h: the celltree-based rasterize_like below needs the memory.
del da_h
gc.collect()

# At the subgrid-downscaled resolution of compute_max_inundation, one
# frame at a time; da_h is mesh resolution and unmasked.
df = compute_flood_timeseries_stats(
    sfincs_root, da_land_dep, min_inundation_depth_m,
)
if df is None:
    raise RuntimeError(f"No 'zs' in the SFINCS event output of {sfincs_root}")
csv_out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(csv_out_path, index=False)

if len(df):
    i_peak = int(df["flood_volume_m3"].idxmax())
    log.info(
        f"Flood timeseries written: {csv_out_path} ({len(df)} step(s)). "
        f"Peak total flood volume {df['flood_volume_m3'].iloc[i_peak]:.3e} m³ "
        f"(area {df['flooded_area_km2'].iloc[i_peak]:.2f} km²) at "
        f"t={df['time'].iloc[i_peak]}"
    )
else:
    log.warning("Flood timeseries is empty — no output timesteps found")

log.info("Done")
