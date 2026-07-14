from pathlib import Path

import geopandas as gpd

from src.domain import load_domain
from src.log import setup_logging
from src.profiling import ScriptProfiler
from src.river_forcing import load_forcing_crossings, snap_crossings_to_reaches
from src.plots import plot_clean_network_discharge, plot_cleaned_network
from src.river_network import (
    accumulate_discharge,
    build_downstream_adjacency,
    collect_downstream_main_paths,
    enforce_mouth_width_monotonic,
    fix_tjunction_tails,
    identify_delta_outflow_points,
    normalize_channel_widths,
    remove_reaches_with_missing_width,
)


def _norm_id(x):
    """Normalise a nullable reach_id to a plain integer string (matches BFS output)."""
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

# ── domain ────────────────────────────────────────────────────────────────────

wgs84_bounds, domain_crs, domain_poly = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)
log.info(f"Domain WGS84 bounds: {wgs84_bounds}, CRS: {domain_crs}")

# ── load inputs ───────────────────────────────────────────────────────────────

rivers = gpd.read_file(snakemake.input.spec_river_network)
log.info(f"Loaded {len(rivers)} reaches")

# Fix T-junction topology: reaches whose start lies on another reach's interior
# rather than at its true endpoint, leaving a short (~18-111 m) dangling tail
# that fragments the merged-line connectivity used by burn_river_rect.
rivers = fix_tjunction_tails(rivers)

crossings = load_forcing_crossings(
    snakemake.input.river_forcing,
    discharge_variable=snakemake.params.discharge_variable,
)
log.info(f"Active GloFAS crossings: {len(crossings)}")

# ── main-path downstream connectivity ─────────────────────────────────────────

seed_q = snap_crossings_to_reaches(crossings)
reachable = collect_downstream_main_paths(rivers, set(seed_q.keys()))
log.info(f"Main-path traversal: {len(reachable)} reachable reaches out of {len(rivers)}")

# ── filter ────────────────────────────────────────────────────────────────────

rivers_clean = rivers[rivers["reach_id"].map(_norm_id).isin(reachable)].copy()
rivers_clean["linked_to_source"] = True
rivers_clean["is_seed"] = rivers_clean["reach_id"].map(_norm_id).isin(set(seed_q.keys()))
log.info(f"Retained {len(rivers_clean)}/{len(rivers)} reaches")

# ── delta-outline outflow points (neither seed nor mouth) ────────────────────
# A reach that crosses the delta polygon's OUTLINE but is neither the seed
# (the real inflow entry point) nor a mouth is a genuine place where flow
# exits the modelled network into the delta plain -- it is kept in the 1D
# network unchanged, flagged 'is_delta_outflow', and its outline-crossing
# point is registered as an SFINCS outflow boundary (mask=3) in build_sfincs.
delta_polygon = gpd.read_file(snakemake.input.delta_polygon)
rivers_clean, outflow_points = identify_delta_outflow_points(rivers_clean, delta_polygon)
log.info(f"Delta-outline outflow points identified: {len(outflow_points)}")

# ── fix width/max_width attributes, then choose the canonical 'width' ─────────
# Some SWORD reaches have width/max_width missing (-9999) entirely -- must be
# handled BEFORE normalize_channel_widths, since its swap step would otherwise
# treat a missing max_width as "smaller than width" and corrupt 'width' with
# the sentinel instead (see remove_reaches_with_missing_width's docstring).
rivers_clean = remove_reaches_with_missing_width(rivers_clean)

# Some SWORD reaches have max_width < width (swapped back) -- see
# normalize_channel_widths. The resulting 'width' is what every downstream
# width-dependent step uses (discharge propagation here, hydraulic depth,
# quadtree refinement buffer, SFINCS rivwth).

rivers_clean = normalize_channel_widths(rivers_clean)

# River mouths shouldn't be narrower than the reach feeding into them --
# channels don't narrow immediately before the outlet; where SWORD's own
# attributes show otherwise (e.g. a mouth obscured by tidal flats/vegetation
# in the source imagery), raise the mouth's width to match its widest
# upstream neighbour. Always operates on 'width' (not width_column) --
# normalize_channel_widths above has already made 'width' the canonical,
# live column regardless of which original SWORD attribute it was chosen from.
rivers_clean = enforce_mouth_width_monotonic(rivers_clean)

# ── discharge propagation ─────────────────────────────────────────────────────
# Propagate seed bankfull discharges downstream through the clean network.
# At bifurcations discharge is split proportional to channel width; at
# confluences contributions from all upstream reaches are summed.

adjacency = build_downstream_adjacency(rivers_clean)
q_acc = accumulate_discharge(
    rivers_clean, seed_q, adjacency,
    n_iterations=snakemake.params.flow_accumulation_iterations,
    min_width_m=snakemake.params.min_width_m,
)
rivers_clean["bankfull_discharge_acc"] = q_acc
log.info(
    f"Discharge propagation: {(q_acc > 0).sum()} reaches with Q > 0, "
    f"max Q = {q_acc.max():.2f} m³ s⁻¹" if len(q_acc) > 0 else
    "Discharge propagation: 0 reaches (empty network)"
)

# ── write ─────────────────────────────────────────────────────────────────────

Path(snakemake.output.clean_river_network).parent.mkdir(parents=True, exist_ok=True)
rivers_clean.to_file(snakemake.output.clean_river_network, driver="GPKG")
log.info(f"Written: {snakemake.output.clean_river_network}")

Path(snakemake.output.delta_outflow_points).parent.mkdir(parents=True, exist_ok=True)
outflow_points.to_file(snakemake.output.delta_outflow_points, driver="GPKG")
log.info(f"Written: {snakemake.output.delta_outflow_points} ({len(outflow_points)} point(s))")

# ── summary plots ─────────────────────────────────────────────────────────────

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
    osm_land_path=snakemake.input.land_polygons,
    river_basins=snakemake.input.specific_basins,
    output_path=snakemake.output.plot_clean_network,
)
plot_clean_network_discharge(
    rivers_wgs=rivers_wgs,
    bbox_poly=domain_poly,
    osm_land_path=snakemake.input.land_polygons,
    output_path=snakemake.output.plot_discharge_network,
)

profiler.stop()
log.info("Done")
