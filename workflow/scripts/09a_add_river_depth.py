from pathlib import Path

import geopandas as gpd
import numpy as np

from src.domain import load_domain
from src.log import setup_logging
from src.plots import (
    plot_hydraulic_relations,
    plot_longitudinal_profile,
    plot_river_depth,
    plot_river_network_width_discharge,
)
from src.profiling import ScriptProfiler
from src.river_network import compute_hydraulic_depth

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
compute_hydraulic_depth = profiler.wrap(compute_hydraulic_depth)

# ── domain ────────────────────────────────────────────────────────────────────

wgs84_bounds, domain_crs, domain_poly = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)
log.info(f"Domain WGS84 bounds: {wgs84_bounds}, CRS: {domain_crs}")

# hydraulic geometry: D = (a·c) · Q^(b+f) / W
hg_alpha = snakemake.params.hg_a * snakemake.params.hg_c
hg_beta  = snakemake.params.hg_b + snakemake.params.hg_f
log.info(f"Hydraulic geometry: alpha={hg_alpha:.4f}, beta={hg_beta:.4f}")

# ── load clean network (discharge already propagated by rule 05) ──────────────

rivers = gpd.read_file(snakemake.input.clean_river_network)
log.info(f"Loaded {len(rivers)} reaches (bankfull_discharge_acc pre-computed)")

# Seed reach IDs stored by rule 05 as the 'is_seed' flag
seed_reach_ids: set[str] = set(
    rivers.loc[rivers["is_seed"].astype(bool), "reach_id"].astype(str)
) if "is_seed" in rivers.columns else set()
log.info(f"Seed reaches: {len(seed_reach_ids)}")

# ── filter by discharge threshold ────────────────────────────────────────────

threshold = snakemake.params.discharge_threshold
rivers_out = rivers[rivers["bankfull_discharge_acc"] >= threshold].copy()
log.info(
    f"After discharge threshold ({threshold} m³ s⁻¹): "
    f"{len(rivers_out)}/{len(rivers)} reaches retained"
)

# ── compute hydraulic depth ───────────────────────────────────────────────────

min_w = snakemake.params.min_width_m
widths = np.maximum(
    rivers_out["width"].fillna(min_w).to_numpy(dtype=float), min_w
)
rivers_out["rivdph"] = compute_hydraulic_depth(
    rivers_out["bankfull_discharge_acc"].values,
    widths,
    alpha=hg_alpha,
    beta=hg_beta,
)
if len(rivers_out) > 0:
    log.info(
        f"Hydraulic depth: min={rivers_out['rivdph'].min():.3f} m, "
        f"max={rivers_out['rivdph'].max():.3f} m, "
        f"median={rivers_out['rivdph'].median():.3f} m"
    )

# ── write output ──────────────────────────────────────────────────────────────

Path(snakemake.output.processed_river_network).parent.mkdir(parents=True, exist_ok=True)
rivers_out.to_file(snakemake.output.processed_river_network, driver="GPKG")
log.info(f"Written: {snakemake.output.processed_river_network}")

# ── plots ─────────────────────────────────────────────────────────────────────

rivers_wgs = (
    rivers_out.to_crs("EPSG:4326")
    if rivers_out.crs is not None and rivers_out.crs.to_epsg() != 4326
    else rivers_out
)
plot_river_depth(
    rivers_wgs=rivers_wgs,
    bbox_poly=domain_poly,
    osm_land_path=snakemake.input.land_polygons,
    output_path=snakemake.output.plot_river_depth,
)
plot_hydraulic_relations(
    rivers_wgs=rivers_wgs,
    output_path=snakemake.output.plot_hydraulic_relations,
)
plot_river_network_width_discharge(
    rivers_wgs=rivers_wgs,
    bbox_poly=domain_poly,
    osm_land_path=snakemake.input.land_polygons,
    seed_reach_ids=seed_reach_ids,
    output_path=snakemake.output.plot_river_network_width_discharge,
)
plot_longitudinal_profile(
    rivers_wgs=rivers_wgs,
    seed_reach_ids=seed_reach_ids,
    output_path=snakemake.output.plot_longitudinal_profile,
)

profiler.stop()
log.info("Done")
