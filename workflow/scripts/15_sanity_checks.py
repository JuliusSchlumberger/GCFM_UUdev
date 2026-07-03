"""
10_sanity_checks.py — Sanity checks for the baseline (spinup) SFINCS condition.

Check 1 — Inundation ratio
--------------------------
Fraction of land-domain pixels with max inundation depth > `min_inundation_depth_m`.

Method:
  1. Compute the downscaled max inundation depth (`da_hmax`) and the bed-level
     reference grid (`da_dep`), both masked to exclude sea via the land-use
     raster, via `src.postprocessing.compute_max_inundation` — see that
     function's docstring for the full method (SfincsModel loading,
     subgrid-aware bed level, `downscale_floodmap`, land-use sea mask).
  2. Denominator: non-null pixels in `da_dep` = total land-domain pixels.
  3. Numerator:   non-null pixels in `da_hmax` = flooded land pixels above hmin.

Additional checks can be appended below and wired to new outputs in the rule.
"""

from pathlib import Path

from typing import cast

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

from src.log import setup_logging
from src.plots import animate_flood_progression, plot_inundation_check
from src.postprocessing import compute_flood_progression, compute_max_inundation

log = setup_logging(snakemake.log[0])

# ── inputs / params ───────────────────────────────────────────────────────────
sfincs_map_nc_path = Path(snakemake.input.sfincs_map_nc)
landuse_path       = Path(snakemake.input.landuse)
land_polygons_path = Path(snakemake.input.land_polygons)
river_network_path = Path(snakemake.input.clean_river_network)
domain_gpkg_path   = Path(snakemake.input.domain_gpkg)
plot_out_path      = Path(snakemake.output.plot_inundation_ratio)
animation_out_path = Path(snakemake.output.animation_flood_progress)
sfincs_root        = Path(snakemake.params.sfincs_root)
threshold_m        = float(snakemake.params.min_inundation_depth_m)
include_subgrid    = bool(snakemake.params.include_subgrid)
animation_fps      = int(snakemake.params.animation_fps)

# Load domain polygon in WGS84 for overlay plots.
_domain_gdf = gpd.read_file(domain_gpkg_path)
if _domain_gdf.crs is not None and _domain_gdf.crs.to_epsg() != 4326:
    _domain_gdf = _domain_gdf.to_crs("EPSG:4326")
_union = _domain_gdf.geometry.union_all()
domain_poly = cast(Polygon, _union if isinstance(_union, Polygon) else _union.convex_hull)

plot_out_path.parent.mkdir(parents=True, exist_ok=True)
basin_id   = sfincs_root.parent.name
spinup_dir = sfincs_root / "spinup"

# ── guard: empty sentinel from rule 09 ───────────────────────────────────────
if sfincs_map_nc_path.stat().st_size == 0:
    log.warning(f"{sfincs_map_nc_path} is empty — SFINCS produced no map output; skipping")
    plot_out_path.touch()
else:
    # ── 1. compute max inundation depth & land-domain reference grid ─────────
    da_hmax, da_dep = compute_max_inundation(
        spinup_dir, sfincs_root, landuse_path,
        hmin=threshold_m, include_subgrid=include_subgrid,
    )

    if da_hmax is None or da_dep is None:
        log.warning("Could not compute max inundation depth (missing 'zsmax' or bed level) — skipping")
        plot_out_path.touch()
    else:
        log.info(f"da_hmax shape: {da_hmax.shape}")

        # ── 2. statistics ─────────────────────────────────────────────────────
        n_land    = int(da_dep.notnull().sum().item())
        n_flooded = int(da_hmax.notnull().sum().item())
        frac      = n_flooded / n_land if n_land > 0 else 0.0

        try:
            res = abs(da_dep.rio.resolution()[0] * da_dep.rio.resolution()[1])
        except Exception:
            res = np.nan
        flooded_km2 = n_flooded * res / 1e6
        land_km2    = n_land    * res / 1e6

        log.info(
            f"[Check 1] Inundation ratio  hmin={threshold_m} m  "
            f"flooded={n_flooded:,}/{n_land:,} pixels ({frac:.2%})  "
            f"area={flooded_km2:.1f}/{land_km2:.1f} km²"
        )

        # ── 3. plot ───────────────────────────────────────────────────────────
        # Pass the DataArray directly — plot_inundation_check uses
        # da_hmax.plot() which reads coordinate metadata and handles
        # north-up orientation automatically.
        plot_inundation_check(
            da_hmax, threshold_m, n_flooded, n_land,
            str(land_polygons_path), str(river_network_path),
            str(plot_out_path), basin_id=basin_id,
        )
        log.info(f"Inundation ratio plot written: {plot_out_path}")

# ── animation: flood progression ─────────────────────────────────────────────
# compute_flood_progression reads zs (instantaneous water level) from the
# spinup sfincs_map.nc — available because dtmapout is now set to a finite
# interval in the spinup inp.  Returns None when zs is absent (e.g. old runs).
animation_out_path.parent.mkdir(parents=True, exist_ok=True)
if sfincs_map_nc_path.stat().st_size == 0:
    log.warning("sfincs_map.nc is empty — skipping flood animation")
    animation_out_path.touch()
else:
    da_h = compute_flood_progression(spinup_dir, landuse_path)
    if da_h is None:
        log.warning(
            "compute_flood_progression returned None (no 'zs' in sfincs_map.nc) — "
            "re-run rule 09 to regenerate with dtmapout enabled"
        )
        animation_out_path.touch()
    else:
        animate_flood_progression(
            da_h, domain_poly,
            str(land_polygons_path), str(river_network_path),
            str(animation_out_path),
            basin_id=basin_id,
            run_label="spinup",
            fps=animation_fps,
        )
        log.info(f"Flood animation written: {animation_out_path}")
