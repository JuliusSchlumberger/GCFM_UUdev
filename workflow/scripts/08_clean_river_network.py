"""Prune, fix and route discharge through the river network; run by rule clean_river_network."""

from pathlib import Path

import geopandas as gpd

from src.domain import load_domain
from src.log import setup_logging
from src.profiling import ScriptProfiler
from src.river_forcing import load_forcing_crossings, snap_crossings_to_reaches
from src.plots import plot_clean_network_discharge, plot_cleaned_network
from src.river_network import (
    _as_linestring,
    accumulate_discharge,
    build_downstream_adjacency,
    canal_excluded_arms,
    collect_downstream_main_paths,
    enforce_mouth_width_monotonic,
    fix_tjunction_tails,
    identify_delta_outflow_points,
    normalize_channel_widths,
    outward_seed_reaches,
    remove_reaches_with_missing_width,
)


def _norm_id(x):
    try:
        return str(int(float(x)))
    except (ValueError, TypeError):
        s = str(x).strip()
        return s if s else None

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
load_forcing_crossings       = profiler.wrap(load_forcing_crossings)
snap_crossings_to_reaches    = profiler.wrap(snap_crossings_to_reaches)
collect_downstream_main_paths = profiler.wrap(collect_downstream_main_paths)
build_downstream_adjacency   = profiler.wrap(build_downstream_adjacency)
accumulate_discharge         = profiler.wrap(accumulate_discharge)

# Domain

wgs84_bounds, domain_crs, domain_poly = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)
log.info(f"Domain WGS84 bounds: {wgs84_bounds}, CRS: {domain_crs}")

# Inputs

rivers = gpd.read_file(snakemake.input.spec_river_network)
log.info(f"Loaded {len(rivers)} reaches")

# Reaches starting on another reach's interior leave a ~18-111 m dangling
# tail that fragments the merged-line connectivity used by burn_river_rect.
rivers = fix_tjunction_tails(rivers)

crossings = load_forcing_crossings(
    snakemake.input.river_forcing,
    discharge_variable=snakemake.params.discharge_variable,
)
log.info(f"Active GloFAS crossings: {len(crossings)}")

# Main-path downstream connectivity

seed_q = snap_crossings_to_reaches(crossings)
reachable = collect_downstream_main_paths(rivers, set(seed_q.keys()))
log.info(f"Main-path traversal: {len(reachable)} reachable reaches out of {len(rivers)}")

# Filter

rivers_clean = rivers[rivers["reach_id"].map(_norm_id).isin(reachable)].copy()
rivers_clean["linked_to_source"] = True
rivers_clean["is_seed"] = rivers_clean["reach_id"].map(_norm_id).isin(set(seed_q.keys()))
log.info(f"Retained {len(rivers_clean)}/{len(rivers)} reaches")

# A seed reach fed from upstream reaches INSIDE the network is a branch SWORD
# routes out of the delta through the crossing (basin 1248635: Drecht): its
# crossing is an outflow, not an inflow. It loses its seed status and inflow
# here, which also lets identify_delta_outflow_points below register the
# crossing as a free-outflow boundary; rules 10/13/14 skip its discharge via
# 'seed_flows_out' (src.river_network.outflow_crossing_mask).
_outward = outward_seed_reaches(
    rivers_clean, seed_q.keys(), build_downstream_adjacency(rivers_clean)
)
rivers_clean["seed_flows_out"] = rivers_clean["reach_id"].map(_norm_id).isin(_outward)
if _outward:
    rivers_clean.loc[rivers_clean["seed_flows_out"], "is_seed"] = False
    seed_q = {k: v for k, v in seed_q.items() if _norm_id(k) not in _outward}
    log.warning(
        f"Seed reach(es) fed from upstream reaches in the network, treated as outflow "
        f"(no inflow, free-outflow boundary): {sorted(_outward)}"
    )

# Trim seed reaches to the delta polygon rule 07 uses for the discharge entry
# point: rule 06's bbox clip (envelope + domain.delta_buffer_m) leaves them
# extending outside, so a point at their midpoint can fall at or beyond the
# domain edge. Only the reach geometry changes; the entry point in
# river_forcing.nc is unaffected.
seed_mask = rivers_clean["is_seed"]
if seed_mask.any():
    domain_poly_native = gpd.GeoSeries([domain_poly], crs="EPSG:4326").to_crs(rivers_clean.crs).iloc[0]
    n_trimmed = 0
    for idx in rivers_clean.index[seed_mask]:
        geom = rivers_clean.at[idx, "geometry"]
        clipped = geom.intersection(domain_poly_native)
        if clipped.is_empty:
            log.warning(f"Seed reach {rivers_clean.at[idx, 'reach_id']}: intersection with domain polygon is empty, keeping original geometry")
            continue
        clipped_line = _as_linestring(clipped)
        if clipped_line is None or clipped_line.length == 0:
            log.warning(f"Seed reach {rivers_clean.at[idx, 'reach_id']}: clipped geometry has no length, keeping original geometry")
            continue
        if clipped_line.length < geom.length:
            n_trimmed += 1
            rivers_clean.at[idx, "geometry"] = clipped_line
    log.info(f"Seed reach(es) trimmed to domain polygon: {n_trimmed}/{int(seed_mask.sum())}")

# Delta-outline outflow points: a reach crossing the delta outline that is
# neither seed nor mouth is where flow leaves the network; it is flagged
# 'is_delta_outflow' and its crossing becomes an SFINCS outflow boundary
# (mask=3) in build_sfincs.
delta_polygon = gpd.read_file(snakemake.input.delta_polygon)
rivers_clean, outflow_points = identify_delta_outflow_points(rivers_clean, delta_polygon)
log.info(f"Delta-outline outflow points identified: {len(outflow_points)}")

# Width attributes. Missing SWORD width/max_width (-9999) must go before
# normalize_channel_widths, whose swap step would otherwise copy the sentinel
# into 'width'.
rivers_clean = remove_reaches_with_missing_width(rivers_clean)

# Some SWORD reaches have max_width < width. The resulting 'width' feeds
# discharge propagation, hydraulic depth and SFINCS rivwth.
rivers_clean = normalize_channel_widths(
    rivers_clean, max_width_fill_m=float(snakemake.params.max_width_fill_m)
)

# Channels do not narrow just before the outlet; where SWORD says otherwise
# (e.g. a mouth obscured by tidal flats/vegetation), the mouth takes its
# widest upstream neighbour's width.
rivers_clean = enforce_mouth_width_monotonic(rivers_clean)

# Discharge propagation: split proportional to width at bifurcations, summed
# at confluences. Canals (SWORD lakeflag 2 / locks) get no share at a
# bifurcation; stored as 'zero_discharge_share' so rule 10 excludes the same arms.
adjacency = build_downstream_adjacency(rivers_clean)
_canal_cfg = dict(snakemake.params.canal_routing)
_excluded = canal_excluded_arms(
    rivers_clean, adjacency,
    force_exclude=_canal_cfg.get("force_exclude") or [],
    force_include=_canal_cfg.get("force_include") or [],
    auto=bool(_canal_cfg.get("auto", True)),
)
rivers_clean["zero_discharge_share"] = rivers_clean["reach_id"].map(_norm_id).isin(_excluded)
q_acc = accumulate_discharge(rivers_clean, seed_q, adjacency, excluded_arms=_excluded)
rivers_clean["bankfull_discharge_acc"] = q_acc
log.info(
    f"Discharge propagation: {(q_acc > 0).sum()} reaches with Q > 0, "
    f"max Q = {q_acc.max():.2f} m³ s⁻¹" if len(q_acc) > 0 else
    "Discharge propagation: 0 reaches (empty network)"
)

# Write

Path(snakemake.output.clean_river_network).parent.mkdir(parents=True, exist_ok=True)
rivers_clean.to_file(snakemake.output.clean_river_network, driver="GPKG")
log.info(f"Written: {snakemake.output.clean_river_network}")

Path(snakemake.output.delta_outflow_points).parent.mkdir(parents=True, exist_ok=True)
outflow_points.to_file(snakemake.output.delta_outflow_points, driver="GPKG")
log.info(f"Written: {snakemake.output.delta_outflow_points} ({len(outflow_points)} point(s))")

# Summary plots

rivers_wgs = (
    rivers_clean.to_crs("EPSG:4326")
    if rivers_clean.crs is not None and rivers_clean.crs.to_epsg() != 4326
    else rivers_clean
)
rivers_orig_wgs = (
    rivers.to_crs("EPSG:4326")
    if rivers.crs is not None and rivers.crs.to_epsg() != 4326
    else rivers
)

plot_cleaned_network(
    rivers_orig=rivers_orig_wgs,
    rivers_clean=rivers_wgs,
    bbox_poly=domain_poly,
    land_polygons_path=snakemake.input.land_polygons,
    output_path=snakemake.output.plot_clean_network,
)
plot_clean_network_discharge(
    rivers_wgs=rivers_wgs,
    bbox_poly=domain_poly,
    land_polygons_path=snakemake.input.land_polygons,
    output_path=snakemake.output.plot_discharge_network,
)

profiler.stop()
log.info("Done")
