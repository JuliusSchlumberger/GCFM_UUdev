"""
08b_optimize_grid_resolution.py -- Compute a per-basin SFINCS main-grid
resolution instead of using a single fixed value for every basin, taking
into account the river network's channel widths and the delta domain's
size (see src.grid_resolution module docstring/functions for the formula).

Runs after rule clean_river_network (08, needs its bankfull_discharge_acc
column to filter reaches the same way rule empirical_depth_estimation does) and before every rule
that consumes the SFINCS grid resolution -- rule get_boundary_forcings (07,
diagnostic-only use) and rule build_sfincs (13, the real one) -- both read
this rule's output JSON instead of a static config value.

Inputs:
    domain_gpkg:         Delta domain polygon.
    spec_basins_meta:    domain_bbox.json (gives the basin's UTM CRS).
    clean_river_network: Cleaned river network (rule 08), with
                          bankfull_discharge_acc and a canonicalized 'width'
                          column already present.

Outputs:
    grid_resolution: JSON with the final resolution (m) plus diagnostics
                      (preferred target, cell-budget cap, which one won,
                      estimated active cells, source).
"""

import json
from pathlib import Path

import geopandas as gpd

from src.domain import load_domain
from src.grid_resolution import compute_optimal_resolution, compute_width_p20_target
from src.log import setup_logging
from src.profiling import ScriptProfiler

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
compute_optimal_resolution = profiler.wrap(compute_optimal_resolution)

enabled = bool(snakemake.params.enabled)
default_resolution_m = float(snakemake.params.default_resolution_m)
target_cells_per_width = float(snakemake.params.target_cells_per_width)
max_active_cells = float(snakemake.params.max_active_cells)

# ── preferred target: width-based (enabled) or flat default (disabled) ──────

source = None
width_p20 = None
if enabled:
    rivers = gpd.read_file(snakemake.input.clean_river_network)
    if rivers.empty:
        log.warning(
            f"Empty river network -- falling back to "
            f"default_resolution_m={default_resolution_m} m"
        )
        preferred_target_m = default_resolution_m
        source = "fallback_no_reaches"
    else:
        preferred_target_m = compute_width_p20_target(
            rivers["width"].to_numpy(dtype=float), target_cells_per_width
        )
        width_p20 = preferred_target_m * target_cells_per_width
        source = "width_optimized"
        log.info(
            f"{len(rivers)} reach(es); "
            f"width p20={width_p20:.1f} m, target_cells_per_width={target_cells_per_width} "
            f"-> preferred_target={preferred_target_m:.1f} m"
        )
else:
    preferred_target_m = default_resolution_m
    source = "fixed_default"
    log.info(f"Width-based optimization disabled -- preferred_target={preferred_target_m:.1f} m")

# ── domain area (own UTM CRS) ────────────────────────────────────────────────

_, domain_crs, domain_poly = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)
domain_poly_utm = gpd.GeoSeries([domain_poly], crs="EPSG:4326").to_crs(domain_crs).iloc[0]
domain_area_m2 = float(domain_poly_utm.area)

# ── combine with the cell-budget cap (applies unconditionally) ──────────────

result = compute_optimal_resolution(preferred_target_m, domain_area_m2, max_active_cells)
result["source"] = source
if width_p20 is not None:
    result["width_p20_m"] = width_p20

log.info(
    f"resolution={result['resolution']:.1f} m "
    f"(preferred_target={result['preferred_target_m']:.1f} m, "
    f"cell_budget_cap={result['cell_budget_cap']:.1f} m, "
    f"capped={result['capped']}, "
    f"estimated_active_cells={result['estimated_active_cells']:.0f}, "
    f"source={source})"
)

Path(snakemake.output.grid_resolution).parent.mkdir(parents=True, exist_ok=True)
with open(snakemake.output.grid_resolution, "w") as f:
    json.dump(result, f, indent=2)
log.info(f"Written: {snakemake.output.grid_resolution}")

profiler.stop()
log.info("Done")
