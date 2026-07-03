from pathlib import Path

import geopandas as gpd

from src.estuarine_depth import (
    compute_estuarine_depths,
    load_nienhuis,
    match_basin_to_delta,
)
from src.log import setup_logging
from src.plots import plot_hydraulic_relations_with_estuarine
from src.profiling import ScriptProfiler

log = setup_logging(snakemake.log[0])
profiler = ScriptProfiler(snakemake)
load_nienhuis            = profiler.wrap(load_nienhuis)
match_basin_to_delta     = profiler.wrap(match_basin_to_delta)
compute_estuarine_depths = profiler.wrap(compute_estuarine_depths)

enabled = bool(snakemake.params.enabled)
log.info(f"Estuarine depth model: {'ENABLED' if enabled else 'DISABLED'}")

# ── load river network from add_river_depth ───────────────────────────────────

rivers = gpd.read_file(snakemake.input.processed_river_network)
log.info(f"Loaded {len(rivers)} reaches")

# ── apply estuarine depth model (only when enabled) ───────────────────────────

L_e_m = None
if enabled:
    nienhuis_df = load_nienhuis(snakemake.input.nienhuis)
    delta_params = match_basin_to_delta(
        delta_polygon_path=snakemake.input.delta_polygon,
        nienhuis_df=nienhuis_df,
        max_dist_km=snakemake.params.max_match_dist_km,
    )

    if delta_params is None:
        log.warning(
            f"No Nienhuis delta found within {snakemake.params.max_match_dist_km:.0f} km "
            "of basin bbox — retaining power-law depths for all reaches"
        )
        rivers_out = rivers.copy()
        rivers_out["rivdph_estuarine"] = False
        rivers_out["rivdph_powerlaw"] = rivers_out["rivdph"]
        rivers_out["rivdph_blend_alpha"] = float("nan")
    else:
        log.info(
            f"Matched Nienhuis delta: {delta_params.get('name', delta_params['id'])} "
            f"(L_e={float(delta_params['L_e'])/1000:.1f} km, P={float(delta_params['P']):.3g} m³)"
        )
        L_e_m = float(delta_params["L_e"])
        rivers_out = compute_estuarine_depths(
            rivers=rivers,
            delta_params=delta_params,
            obrien_C=snakemake.params.obrien_C,
            obrien_alpha=snakemake.params.obrien_alpha,
            convergence_ratio_k=snakemake.params.convergence_ratio_k,
            blend_fraction=snakemake.params.blend_fraction,
            min_depth_m=snakemake.params.min_depth_m,
            width_column=snakemake.params.width_column,
        )
else:
    # Disabled: pass network through unchanged, no estuarine columns added.
    rivers_out = rivers.copy()

# ── write river_network_estuarine.gpkg ────────────────────────────────────────

Path(snakemake.output.estuarine_river_network).parent.mkdir(parents=True, exist_ok=True)
rivers_out.to_file(snakemake.output.estuarine_river_network, driver="GPKG")
log.info(f"Written: {snakemake.output.estuarine_river_network}")

# ── hydraulics plot ───────────────────────────────────────────────────────────
# Always produced here (not in add_river_depth) so the plot reflects
# estuarine-depth information when available.

rivers_wgs = (
    rivers_out.to_crs("EPSG:4326")
    if rivers_out.crs is not None and rivers_out.crs.to_epsg() != 4326
    else rivers_out
)
Path(snakemake.output.plot_hydraulic_relations).parent.mkdir(parents=True, exist_ok=True)
plot_hydraulic_relations_with_estuarine(
    rivers_wgs=rivers_wgs,
    output_path=snakemake.output.plot_hydraulic_relations,
    L_e_m=L_e_m,
)
log.info(f"Plot written: {snakemake.output.plot_hydraulic_relations}")

profiler.stop()
log.info("Done")
