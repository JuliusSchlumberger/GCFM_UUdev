"""Run sanity checks on the spin-up SFINCS output.

Run by rule sanity_checks. Check 1, inundation ratio: fraction of land-domain
pixels (non-null in the sea-masked, subgrid-downscaled bed level) whose max
inundation depth exceeds min_inundation_depth_m. Also renders the spin-up
flood-progression animation.
"""

from pathlib import Path

from typing import cast

import geopandas as gpd
from shapely.geometry import Polygon

from src.log import setup_logging
from src.plots import animate_flood_progression, plot_inundation_check
from src.postprocessing import compute_flood_progression, compute_max_inundation, inundation_stats, read_land_bed_level

log = setup_logging(snakemake.log[0])

# ── inputs / params ──
sea_mask_path      = Path(snakemake.input.sea_mask)
land_polygons_path = Path(snakemake.input.land_mask_on_grid)  # plot background
river_network_path = Path(snakemake.input.clean_river_network)
domain_gpkg_path   = Path(snakemake.input.domain_gpkg)
plot_out_path      = Path(snakemake.output.plot_inundation_ratio)
animation_out_path      = Path(snakemake.output.animation_flood_progress)
spin_up_root            = Path(snakemake.params.spin_up_root)
threshold_m             = float(snakemake.params.min_inundation_depth_m)
animation_fps           = int(snakemake.params.animation_fps)

_domain_gdf = gpd.read_file(domain_gpkg_path)
if _domain_gdf.crs is not None and _domain_gdf.crs.to_epsg() != 4326:
    _domain_gdf = _domain_gdf.to_crs("EPSG:4326")
_union = _domain_gdf.geometry.union_all()
domain_poly = cast(Polygon, _union if isinstance(_union, Polygon) else _union.convex_hull)

plot_out_path.parent.mkdir(parents=True, exist_ok=True)
basin_id   = spin_up_root.parent.name
spinup_dir = spin_up_root

da_hmax, da_dep = compute_max_inundation(
    spinup_dir, read_land_bed_level(snakemake.input.land_bed_level),
    hmin=threshold_m,
)
if da_hmax is None or da_dep is None:
    raise RuntimeError(f"No 'zsmax' or bed level in the SFINCS spin-up output of {spinup_dir}")
log.info(f"da_hmax shape: {da_hmax.shape}")

_stats    = inundation_stats(da_hmax, da_dep)
n_land    = _stats["n_land"]
n_flooded = _stats["n_flooded"]
frac      = _stats["frac"]
flooded_km2 = _stats["flooded_km2"]
land_km2    = _stats["land_km2"]

log.info(
    f"[Check 1] Inundation ratio  hmin={threshold_m} m  "
    f"flooded={n_flooded:,}/{n_land:,} pixels ({frac:.2%})  "
    f"area={flooded_km2:.1f}/{land_km2:.1f} km²"
)

plot_inundation_check(
    da_hmax, threshold_m, n_flooded, n_land,
    str(land_polygons_path), str(river_network_path),
    str(plot_out_path), basin_id=basin_id,
)
log.info(f"Inundation ratio plot written: {plot_out_path}")

# ── animation: flood progression ──
# Needs zs in sfincs_map.nc (finite dtmapout in the spin-up inp).
animation_out_path.parent.mkdir(parents=True, exist_ok=True)
da_h = compute_flood_progression(spinup_dir, sea_mask_path)
if da_h is None:
    raise RuntimeError(
        f"No 'zs' or 'zb' in the SFINCS spin-up output of {spinup_dir}; "
        f"rule 14 must run with dtmapout enabled"
    )
animate_flood_progression(
    da_h, domain_poly,
    str(land_polygons_path), str(river_network_path),
    str(animation_out_path),
    basin_id=basin_id,
    run_label="spinup",
    fps=animation_fps,
)
log.info(f"Flood animation written: {animation_out_path}")
