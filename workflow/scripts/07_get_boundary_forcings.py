import json
from pathlib import Path

import geopandas as gpd
import numpy as np

from src.domain import load_domain
from src.extreme_values import (
    STANDARD_RETURN_PERIODS_YR,
    EVAResult,
    analyse_cell,
    bias_correct_discharge,
    gpd_return_value_table,
    plot_bias_correction,
    plot_cell_diagnostics,
)
from src.log import setup_logging
from src.river_forcing import (
    build_design_discharge_matrix,
    build_river_dataset,
    find_best_glofas_cell,
    find_boundary_crossings,
    find_nearest_grdc_station,
    has_downstream_in_domain,
    load_glofas_clip,
    load_grdc_series,
    load_grdc_stations,
    resolve_inside_domain_reaches,
)
from src.plots import plot_domain_map, plot_forcing_timeseries, plot_surge_corrections
from src.river_network import normalize_channel_widths
from src.surge import (
    apply_mdt_correction,
    apply_slr_fingerprint,
    build_surge_dataset,
    build_time_axis,
    compute_distances_to_bbox,
    compute_global_mean_slr,
    interpolate_protection_level,
    load_coastrp_stations,
    load_mdt,
    load_slr_fingerprint,
    select_nearest_stations,
)
from src.profiling import ScriptProfiler

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
load_coastrp_stations         = profiler.wrap(load_coastrp_stations)
compute_distances_to_bbox     = profiler.wrap(compute_distances_to_bbox)
select_nearest_stations       = profiler.wrap(select_nearest_stations)
load_mdt                       = profiler.wrap(load_mdt)
apply_mdt_correction           = profiler.wrap(apply_mdt_correction)
load_slr_fingerprint           = profiler.wrap(load_slr_fingerprint)
compute_global_mean_slr        = profiler.wrap(compute_global_mean_slr)
apply_slr_fingerprint          = profiler.wrap(apply_slr_fingerprint)
build_surge_dataset           = profiler.wrap(build_surge_dataset)
find_boundary_crossings       = profiler.wrap(find_boundary_crossings)
resolve_inside_domain_reaches = profiler.wrap(resolve_inside_domain_reaches)
load_glofas_clip              = profiler.wrap(load_glofas_clip)
find_best_glofas_cell         = profiler.wrap(find_best_glofas_cell)
analyse_cell                  = profiler.wrap(analyse_cell)
build_river_dataset           = profiler.wrap(build_river_dataset)
load_grdc_stations             = profiler.wrap(load_grdc_stations)
find_nearest_grdc_station      = profiler.wrap(find_nearest_grdc_station)
load_grdc_series                = profiler.wrap(load_grdc_series)
bias_correct_discharge          = profiler.wrap(bias_correct_discharge)

# ── domain ────────────────────────────────────────────────────────────────────

wgs84_bounds, domain_crs, domain_poly = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)
lon_min, lat_min, lon_max, lat_max = wgs84_bounds
domain_gdf = gpd.GeoDataFrame(geometry=[domain_poly], crs="EPSG:4326")
domain_utm = domain_gdf.to_crs(domain_crs)
log.info(f"Domain WGS84 bounds: {wgs84_bounds}, CRS: {domain_crs}")

# ── existing flood-protection level (optional correction) ────────────────────
# Identified independently in rule get_protection_levels (always runs); only
# read/applied here when protection_levels.enabled, in
# which case the corresponding discharge/water-level is subtracted from the
# river/surge forcing timeseries below (see the "river forcing" and "surge
# forcing" sections). riverine_rp_yr/coastal_rp_yr stay None when disabled,
# so no other code path in this script is affected.
protection_levels_enabled = bool(snakemake.params.protection_levels_enabled)
riverine_rp_yr = None
coastal_rp_yr = None
if protection_levels_enabled:
    with open(snakemake.input.protection_levels) as f:
        protection_summary = json.load(f)
    riverine_rp_yr = float(protection_summary["riverine_rp_yr"])
    coastal_rp_yr = float(protection_summary["coastal_rp_yr"])
    log.info(
        f"Protection-level correction enabled: riverine RP={riverine_rp_yr:.1f} yr "
        f"({protection_summary['riverine_source']}), "
        f"coastal RP={coastal_rp_yr:.1f} yr ({protection_summary['coastal_source']}), "
        f"dominant unit={protection_summary['dominant_iso']} "
        f"(id={protection_summary['dominant_geounit_id']})"
    )

# ── surge forcing ─────────────────────────────────────────────────────────────

log.info("--- Surge forcing ---")

surge_lead    = snakemake.params.lead_days
surge_period  = snakemake.params.surge_period_hr
surge_dt      = snakemake.params.dt_hr
return_period = snakemake.params.surge_return_period

stations = load_coastrp_stations(snakemake.input.surge_data, return_period)
log.info(f"CoastRP stations loaded: {len(stations)}")
stations = compute_distances_to_bbox(stations, domain_utm, domain_crs)
stations = select_nearest_stations(
    stations,
    snakemake.params.min_surge_stations,
    snakemake.params.max_surge_stations,
    snakemake.params.search_radii_km,
    snakemake.params.surge_dedupe_radius_km,
    domain_crs,
)
log.info(f"Most distant selected station: {stations['dist_m'].max() / 1000:.1f} km")

# ── vertical-reference correction (MDT: local MSL -> GOCO06s geoid) ──────────
# Mandatory, not optional: the coastal DEM (FathomDEM) is always referenced
# to GOCO06s via the mandatory datum correction in 05a_get_elevation.py, so
# leaving the surge boundary in local MSL would put the DEM and the water-
# level forcing in inconsistent vertical references. Mirrors the sign
# convention used in 05a to re-reference GEBCO to GOCO06s (gebco -= mdt).
mdt_fallback_search_deg = float(snakemake.params.mdt_fallback_search_deg)
mdt_da = load_mdt(snakemake.input.mdt_data)
stations = apply_mdt_correction(stations, mdt_da, mdt_fallback_search_deg)
n_nan_mdt = int(stations["mdt"].isna().sum())
if n_nan_mdt:
    log.warning(
        f"{n_nan_mdt}/{len(stations)} surge station(s) have no valid MDT within "
        f"+/-{mdt_fallback_search_deg} deg; mdt set to 0 for these"
    )
    stations["mdt"] = stations["mdt"].fillna(0.0)
stations["rp_level"] = stations["rp_level_raw"] - stations["mdt"]
baseline_m = float(stations["mdt"].mean())
log.info(
    f"MDT vertical correction applied (local MSL -> GOCO06s): "
    f"delta = [{(-stations['mdt']).min():.3f}, {(-stations['mdt']).max():.3f}] m"
)
# ── SLR fingerprint scenario ───────────────────────────────────────────────
slr_cfg = snakemake.params.surge_slr
if slr_cfg["enabled"] and slr_cfg["slr_m"] != 0:
    slr_ds = load_slr_fingerprint(
        snakemake.input.slr_data, slr_cfg["ssp_scenario"],
        slr_cfg["confidence_level"], slr_cfg["year"], slr_cfg["quantile"],
    )
    global_mean_slr = compute_global_mean_slr(slr_ds)
    stations = apply_slr_fingerprint(
        stations, slr_ds, global_mean_slr, slr_cfg["slr_m"], slr_cfg["fallback_search_deg"]
    )
    log.info(
        f"Applied SLR fingerprint ({slr_cfg['ssp_scenario']}, {slr_cfg['year']}, "
        f"target {slr_cfg['slr_m']} m global mean): "
        f"delta = [{stations['slr_m'].min():.3f}, {stations['slr_m'].max():.3f}] m"
    )
else:
    stations["slr_fingerprint"] = np.nan
    stations["slr_m"] = 0.0

# River shares lead_days/dt_hr with surge (build_time_axis's single, shared
# time axis); only period_hr genuinely differs (tidal-like vs. river event
# wave duration). Read early so we can compute the shared total duration
# before building the surge time axis.
river_lead   = surge_lead
river_period = snakemake.params.river_period_hr
river_dt     = surge_dt

forcing_total_hr = max(surge_lead * 24 + surge_period, river_lead * 24 + river_period)
log.info(
    f"Forcing total duration: {forcing_total_hr:.0f} h "
    f"(surge: {surge_lead * 24 + surge_period:.0f} h, "
    f"river: {river_lead * 24 + river_period:.0f} h)"
)
t_surge = build_time_axis(surge_lead, surge_period, surge_dt, total_hr=forcing_total_hr)

# Baseline = mean of total correction actually applied to rp_level (MDT + SLR,
# but only when the respective flag is enabled).  Always 0.0 when both are off.
# Stored in surge_forcing.nc and read by rule 13 to initialise sea cells at the
# same vertical reference as the boundary forcing lead period.
baseline_m = float((stations["rp_level"] - stations["rp_level_raw"]).mean())
log.info(
    f"Surge boundary baseline: {baseline_m:+.4f} m "
    f"(= mean(−MDT + SLR) across {len(stations)} stations — local MSL in model coords)"
)

# Per-station lead-period baselines: each station's own local MSL in model
# coordinates (= rp_level_i - rp_level_raw_i = -mdt_i + slr_m_i).
# Using these (not the scalar mean) ensures each station's sinusoidal wave
# amplitude = rp_level_raw_i exactly, regardless of how MDT varies across
# the selected stations.  The scalar baseline_m is still stored for zsini.
station_baselines = (stations["rp_level"] - stations["rp_level_raw"]).to_numpy()
surge_ds = build_surge_dataset(
    stations, t_surge, surge_lead, surge_period, return_period,
    baseline_m=baseline_m, station_baselines=station_baselines,
)

_plot_protection_level_raw = None   # passed to plot_surge_corrections below
if protection_levels_enabled:
    # The FLOPROS RP is a regional constant for the whole delta; we therefore
    # use the mean COAST-RP storm-tide value across all selected stations at
    # that RP as a single representative protection height.  Per-station
    # interpolation would introduce 10–50 cm spatial variation in COAST-RP
    # values (exposed vs sheltered locations) that does not reflect actual
    # defense heights, causing the corrected water-level timeseries to spread
    # artifically across stations.  The scalar mean keeps the post-correction
    # spread the same as the original spread (MDT variation, a few cm).
    protection_level_raw = interpolate_protection_level(stations, coastal_rp_yr)
    mean_prot_raw = float(protection_level_raw.mean())
    _plot_protection_level_raw = np.full(len(stations), mean_prot_raw)
    protection_level = np.full(len(stations), mean_prot_raw)   # uniform across stations

    surge_ds["water_level_uncorrected"] = surge_ds["water_level"]
    surge_ds["protection_level"] = (
        ["station"],
        protection_level,
        {
            "units": "m",
            "long_name": (
                f"existing flood-protection level (RP{coastal_rp_yr:g} yr, FLOPROS coastal, "
                "local MSL — subtracted as-is from GOCO6s water_level timeseries)"
            ),
        },
    )
    surge_ds["protection_rp_yr"] = (
        [],
        float(coastal_rp_yr),
        {"units": "yr", "long_name": "FLOPROS coastal protection return period used"},
    )
    surge_ds["water_level"] = surge_ds["water_level"] - surge_ds["protection_level"]

    # Update baseline_m = mean(MWL − MDT + SLR − prot) so rule 13 initialises
    # zsini at the correct flat ocean level during spinup.  Without this update,
    # zsini would be set to the MDT-only baseline (≈ −MDT) while the forcing
    # lead period sits at −MDT + SLR − prot_raw, causing coastal cells to
    # flood/drain during spinup.
    sb = (
        surge_ds["station_baseline"].values
        if "station_baseline" in surge_ds
        else np.full(len(protection_level), float(surge_ds["baseline_m"].values))
    )
    effective_baseline_m = float(np.mean(sb - protection_level))
    surge_ds["baseline_m"] = (
        [],
        effective_baseline_m,
        surge_ds["baseline_m"].attrs,
    )
    log.info(
        f"Protection-level correction applied to surge: water_level -= "
        f"{mean_prot_raw:.3f} m (mean across {len(stations)} stations, "
        f"RP{coastal_rp_yr:g} yr; per-station range was "
        f"[{protection_level_raw.min():.3f}, {protection_level_raw.max():.3f}] m); "
        f"effective baseline_m updated to {effective_baseline_m:+.4f} m"
    )

Path(snakemake.output.surge_forcing).parent.mkdir(parents=True, exist_ok=True)
surge_ds.to_netcdf(snakemake.output.surge_forcing)
log.info(f"Written surge forcing ({len(stations)} stations): {snakemake.output.surge_forcing}")

plot_surge_corrections(
    stations,
    output_path=snakemake.output.plot_surge_correction,
    protection_level_raw=_plot_protection_level_raw,
)
log.info(f"Wrote surge correction diagnostic plot: {snakemake.output.plot_surge_correction}")

# ── river forcing ─────────────────────────────────────────────────────────────

log.info("--- River forcing ---")

eva_cfg         = dict(snakemake.params.eva)
# boundary_setup.design_rp_river_yr (not boundary_forcings.river.eva -- lives
# with the other build-time SFINCS settings) drives the diagnostic q_rp100/
# CI/plot-vertical-line under the same "rp_fl" key analyse_cell already reads
# -- doesn't change analyse_cell itself, just which RP those diagnostics
# reflect (the actual production discharge_rp_table below is computed at
# every standard RP regardless, independent of this value).
eva_cfg["rp_fl"] = float(snakemake.params.design_rp_river_yr)
# Main grid resolution only -- subgrid does not change what cell size SFINCS
# actually solves the hydrodynamics on, so it has no bearing on whether a
# crossing is "visible" at the model's own resolution (see Step 4 below,
# which is diagnostic-only in any case and no longer excludes anything).
sfincs_res      = float(snakemake.params.sfincs_resolution)
glofas_radius_m = float(snakemake.params.glofas_search_radius_km) * 1000.0
glofas_min_q    = float(snakemake.params.glofas_min_mean_discharge)
glofas_variable = "dis24"  # sole variable in data_catalogue's river_discharge (GloFAS v4) source
bias_cfg        = dict(snakemake.params.bias_correction)
grdc_radius_m   = float(bias_cfg["grdc_search_radius_km"]) * 1000.0

grdc_stations = load_grdc_stations(snakemake.input.grdc_data)
log.info(f"  Loaded GRDC station table: {len(grdc_stations)} station(s)")

river_gdf = gpd.read_file(snakemake.input.spec_river_network)
if river_gdf.crs is not None and river_gdf.crs != domain_gdf.crs:
    river_gdf = river_gdf.to_crs("EPSG:4326")

# Same width/max_width fix + canonical-column choice as rule clean_river_network
# (this rule reads river_network.gpkg independently and earlier in the DAG, so
# it can't just inherit the cleaned network's already-fixed 'width').
river_gdf = normalize_channel_widths(river_gdf)

domain_reach_ids = set(str(rid) for rid in river_gdf["reach_id"].dropna())

# ── Step 1: boundary crossings ────────────────────────────────────────────────
crossings = find_boundary_crossings(river_gdf, domain_poly)
log.info(
    f"Step 1 — boundary crossings: {len(crossings)} reach(es) intersect domain boundary"
)

if crossings.empty:
    log.warning("No river boundary crossings found; river forcing will be empty")
    has_glofas = np.zeros(0, dtype=bool)
    cell_lon = np.array([])
    cell_lat = np.array([])
    bankfull_q = np.array([])
    flood_q = np.array([])
    protection_q = np.array([])
    results: list = []
    bias_corrected_arr    = np.zeros(0, dtype=np.int8)
    grdc_station_id_arr   = np.full(0, -1, dtype=np.int64)
    grdc_correlation_arr  = np.full(0, np.nan)
    grdc_overlap_days_arr = np.full(0, np.nan)
    # No crossings ⇒ has_glofas is empty ⇒ rule 08's clean network has no
    # is_seed reaches — write an empty sentinel so the declared Snakemake
    # output still exists regardless.
    Path(snakemake.output.glofas_clip).parent.mkdir(parents=True, exist_ok=True)
    Path(snakemake.output.glofas_clip).touch()
else:
    # ── Step 2: clip crossing reach to its domain-entry point ─────────────────
    # For each crossing reach, clip its geometry against the domain polygon and
    # use the point where it first enters the domain (almost always the same
    # point Step 1 already found, on the crossing reach itself).  Only walks
    # to the next downstream reach via rch_id_dn if the crossing reach merely
    # touches the domain boundary without any of its length lying inside it.
    # Updates crossings.geometry to that entry point and overwrites
    # width/max_width with the resolved reach's attributes.
    crossings = resolve_inside_domain_reaches(crossings, river_gdf, domain_poly)
    log.info(
        f"Step 2 — domain entry point: geometry updated to the domain-entry "
        f"point for {len(crossings)} crossing(s)"
    )

    # ── Step 3: Filter — enters_domain ───────────────────────────────────────
    # A crossing 'enters' the domain when at least one downstream reach ID
    # listed in rch_id_dn of the crossing reach is present in the domain network.
    crossings["enters_domain"] = crossings["rch_id_dn"].apply(
        lambda dn: has_downstream_in_domain(dn, domain_reach_ids)
    )
    n_enters = int(crossings["enters_domain"].sum())
    log.info(
        f"Step 3 — enters_domain: {n_enters}/{len(crossings)} crossing(s) have "
        f"a downstream reach inside the domain"
    )
    if n_enters < len(crossings):
        excluded_ids = crossings.loc[~crossings["enters_domain"], "reach_id"].tolist()
        log.info(
            f"  Excluded (rch_id_dn not in domain): reach_id(s) = {excluded_ids}"
        )

    # ── Step 4: visible_on_grid (diagnostic only -- does NOT gate anything) ──
    # Uses the inside-domain reach's 'width' (set by resolve_inside_domain_reaches;
    # already the canonical value, via normalize_channel_widths above) compared
    # against the SFINCS grid resolution. This USED TO also gate
    # Step 5 (a crossing narrower than one grid cell was skipped entirely), but
    # that coupled river-network cleaning (rule 08's BFS seed set, and therefore
    # which reaches survive at all) to the SFINCS grid/subgrid/quadtree
    # configuration -- a narrow crossing failing this check at a fine
    # resolution could strand and drop an entire otherwise-valid downstream
    # branch, purely because of this width heuristic rather than any genuine
    # network issue. Kept only as an informational column (used by the
    # diagnostic plot) -- Step 5 now runs for every enters_domain crossing
    # regardless of width.
    w_col = "width"
    if w_col in crossings.columns:
        crossings["visible_on_grid"] = (
            crossings[w_col].fillna(0.0).astype(float) >= sfincs_res
        )
    else:
        log.warning(
            f"Width column '{w_col}' not found in river network; "
            f"visible_on_grid = True for all crossings"
        )
        crossings["visible_on_grid"] = True

    n_visible = int((crossings["enters_domain"] & crossings["visible_on_grid"]).sum())
    log.info(
        f"Step 4 — visible_on_grid ({w_col} >= {sfincs_res:.0f} m), informational only: "
        f"{n_visible}/{n_enters} entering crossing(s) wide enough for the grid"
    )
    if n_visible < n_enters:
        too_narrow = crossings["enters_domain"] & ~crossings["visible_on_grid"]
        narrow_widths = crossings.loc[too_narrow, w_col].tolist()
        log.info(
            f"  Narrower than grid resolution (NOT excluded): {w_col} = "
            f"{[f'{w:.0f}' for w in narrow_widths]} m"
        )

    # ── Step 5: GloFAS matching ───────────────────────────────────────────────
    # For each qualifying crossing (enters_domain -- visible_on_grid no longer
    # gates this, see Step 4), search within glofas_radius_m for GloFAS cells
    # whose mean discharge exceeds glofas_min_q.  Among qualifying cells, pick
    # the one with the highest mean discharge.  Run EVA on that cell; mark
    # has_glofas=True only when RP2 is finite (EVA converged).
    n_qualifying = int(crossings["enters_domain"].sum())
    log.info(
        f"Step 5 — GloFAS matching: "
        f"radius = {glofas_radius_m / 1000:.1f} km, "
        f"min mean discharge = {glofas_min_q:.1f} m³/s"
    )

    glofas_clip = load_glofas_clip(
        Path(snakemake.input.river_discharge),
        glofas_variable,
        wgs84_bounds,
        snakemake.params.glofas_buffer_deg,
    )
    log.info(f"  Loaded GloFAS clip: {dict(glofas_clip.sizes)}")

    # Persist the clipped subset so rule test_discharge_comparison (07) can reuse
    # it directly instead of re-running this same ~120-150 s clip for the same
    # basin/domain bounds.
    Path(snakemake.output.glofas_clip).parent.mkdir(parents=True, exist_ok=True)
    glofas_clip.to_netcdf(snakemake.output.glofas_clip)
    log.info(f"  Written GloFAS clip cache: {snakemake.output.glofas_clip}")

    lat_dim  = "latitude"   if "latitude"   in glofas_clip.dims else "lat"
    lon_dim  = "longitude"  if "longitude"  in glofas_clip.dims else "lon"
    time_dim = "valid_time" if "valid_time" in glofas_clip.dims else "time"
    lat_arr  = glofas_clip[lat_dim].values
    lon_arr  = glofas_clip[lon_dim].values
    times_arr = glofas_clip[time_dim].values

    n = len(crossings)
    has_glofas = np.zeros(n, dtype=bool)
    cell_lon   = np.full(n, np.nan)
    cell_lat   = np.full(n, np.nan)
    bankfull_q   = np.full(n, np.nan)
    discharge_rp_table = np.full((n, len(STANDARD_RETURN_PERIODS_YR)), np.nan)
    protection_q = np.full(n, np.nan)
    results      = [None] * n

    bias_corrected_arr    = np.zeros(n, dtype=np.int8)
    grdc_station_id_arr   = np.full(n, -1, dtype=np.int64)
    grdc_correlation_arr  = np.full(n, np.nan)
    grdc_overlap_days_arr = np.full(n, np.nan)

    cell_i_lat = np.full(n, -1, dtype=int)
    cell_i_lon = np.full(n, -1, dtype=int)

    eva_cache: dict[tuple[int, int], EVAResult] = {}
    bias_cache: dict[tuple[int, int], dict | None] = {}
    # Final (bias-corrected where applicable) series actually fed to
    # analyse_cell, keyed the same way as eva_cache/bias_cache -- reused for
    # the EVA diagnostic plot below so it re-fits on the SAME data the
    # reported bankfull/flood/protection discharges came from, rather than
    # silently re-fetching raw (pre-bias-correction) GloFAS values.
    ts_cache: dict[tuple[int, int], np.ndarray] = {}

    for i, row in enumerate(crossings.itertuples()):
        if not row.enters_domain:
            continue

        pt_lon = row.geometry.x
        pt_lat = row.geometry.y
        inside_id = getattr(row, "inside_reach_id", None) or getattr(row, "reach_id", i)
        label = f"crossing{i}_reach{inside_id}"

        match = find_best_glofas_cell(
            glofas_clip, glofas_variable,
            pt_lon, pt_lat, glofas_radius_m, glofas_min_q,
            utm_crs=domain_crs,
        )
        if match is None:
            log.info(
                f"  {label}: no GloFAS cell within {glofas_radius_m / 1000:.1f} km "
                f"with mean Q > {glofas_min_q:.1f} m³/s"
            )
            continue

        i_lat, i_lon = match

        if (i_lat, i_lon) in eva_cache:
            eva = eva_cache[(i_lat, i_lon)]
        else:
            ts = glofas_clip[glofas_variable].isel(
                {lat_dim: i_lat, lon_dim: i_lon}
            ).values.astype(float)

            # ── GRDC bias correction (first crossing reaching this cell) ──
            # TODO: the current GRDC-Daily.nc extract (see grdc_discharge in
            # data_catalogue.yml) covers the 8 basins in the 8_polygons.json
            # delta-polygon test set; it is not a global extract, so basins
            # outside that test set will typically find no station and fall
            # back to GloFAS-as-truth.
            bc_diag = None
            grdc_id = find_nearest_grdc_station(
                grdc_stations, pt_lon, pt_lat, grdc_radius_m, utm_crs=domain_crs,
            )
            if grdc_id is not None:
                grdc_times, grdc_values = load_grdc_series(snakemake.input.grdc_data, grdc_id)
                ts_corrected, bc_diag = bias_correct_discharge(
                    times_arr, ts, grdc_times, grdc_values, bias_cfg, label=label,
                )
                if bc_diag is not None:
                    ts = ts_corrected
                    bc_diag["grdc_station_id"] = grdc_id
            else:
                log.info(
                    f"  {label}: no GRDC station within "
                    f"{grdc_radius_m / 1000:.1f} km — GloFAS-as-truth retained"
                )
            bias_cache[(i_lat, i_lon)] = bc_diag
            ts_cache[(i_lat, i_lon)] = ts

            eva = analyse_cell(
                times_arr, ts, eva_cfg, label=label,
                protection_rp=riverine_rp_yr if protection_levels_enabled else None,
            )
            eva_cache[(i_lat, i_lon)] = eva

        results[i] = eva
        cell_i_lat[i] = i_lat
        cell_i_lon[i] = i_lon

        if not np.isfinite(eva.q_rp2):
            log.warning(f"  {label}: EVA did not produce a finite RP2 — crossing inactive")
            continue

        has_glofas[i] = True
        cell_lon[i]   = float(lon_arr[i_lon])
        cell_lat[i]   = float(lat_arr[i_lat])
        bankfull_q[i] = eva.q_rp2
        # Full return-period discharge table from the already-fitted POT/GPD
        # curve -- no re-fitting. Replaces the old single flood_discharge
        # (RP=eva.rp_fl) scalar; the actual design discharge used to build
        # the model is now looked up from this table at SFINCS-build time
        # (boundary_setup.design_rp_river_yr), see
        # src.river_forcing.build_design_discharge_matrix.
        discharge_rp_table[i] = gpd_return_value_table(
            eva.pot_threshold, eva.pot_scale, eva.pot_shape, eva.pot_peaks_per_year,
        )
        if protection_levels_enabled:
            protection_q[i] = eva.q_protection if np.isfinite(eva.q_protection) else 0.0
        log.info(
            f"  {label}: GloFAS ({lat_arr[i_lat]:.3f}°N, {lon_arr[i_lon]:.3f}°E)  "
            f"Q_bankfull = {bankfull_q[i]:.1f} m³/s  "
            f"Q_rp{eva_cfg['rp_fl']:g} (diagnostic) = {eva.q_rp100:.1f} m³/s"
        )

        bc_diag = bias_cache.get((i_lat, i_lon))
        if bc_diag is not None:
            bias_corrected_arr[i]    = 1
            grdc_station_id_arr[i]   = bc_diag["grdc_station_id"]
            grdc_correlation_arr[i]  = bc_diag["correlation_raw"]
            grdc_overlap_days_arr[i] = bc_diag["grdc_overlap_days"]

    n_has_glofas = int(has_glofas.sum())
    log.info(
        f"Step 5 summary — has_glofas: {n_has_glofas}/{n_qualifying} qualifying "
        f"crossing(s) matched to GloFAS and EVA successful"
    )

t_river  = build_time_axis(river_lead, river_period, river_dt, total_hr=forcing_total_hr)
river_ds = build_river_dataset(
    crossings, has_glofas, bankfull_q, discharge_rp_table, STANDARD_RETURN_PERIODS_YR,
    cell_lon, cell_lat, t_river, river_lead, river_period, results,
    rp_bankfull=eva_cfg.get("rp_bf", 2),
    bias_corrected=bias_corrected_arr,
    grdc_station_id=grdc_station_id_arr,
    grdc_correlation=grdc_correlation_arr,
    grdc_overlap_days=grdc_overlap_days_arr,
)

# protection_discharge/protection_rp_yr stay simple per-crossing scalars,
# written here as before -- the actual protection-floor CORRECTION (applying
# them against the design discharge) now happens at SFINCS-build time (rule
# 13), on the scalar design discharge looked up from discharge_rp_table, not
# on a full timeseries here -- see src.river_forcing.build_design_discharge_matrix.
if protection_levels_enabled:
    river_ds["protection_discharge"] = (
        ["crossing"],
        protection_q,
        {
            "units": "m3 s-1",
            "long_name": f"existing flood-protection discharge (RP{riverine_rp_yr:g} yr, FLOPROS riverine, POT/GPD fit)",
        },
    )
    river_ds["protection_rp_yr"] = (
        [],
        float(riverine_rp_yr),
        {"units": "yr", "long_name": "FLOPROS riverine protection return period used"},
    )
    if protection_q.size > 0 and np.isfinite(protection_q).any():
        log.info(
            f"Protection discharge (RP{riverine_rp_yr:g} yr) stored: range "
            f"[{np.nanmin(protection_q):.1f}, {np.nanmax(protection_q):.1f}] m³/s "
            f"-- correction itself applied at SFINCS-build time"
        )
    else:
        log.info(
            f"Protection-level correction enabled (RP{riverine_rp_yr:g} yr) but no "
            f"active crossings to apply it to"
        )

Path(snakemake.output.river_forcing).parent.mkdir(parents=True, exist_ok=True)
river_ds.to_netcdf(snakemake.output.river_forcing)
log.info(
    f"Written river forcing ({len(crossings)} crossings, "
    f"{int(has_glofas.sum())} with GloFAS): {snakemake.output.river_forcing}"
)

# Preview discharge at the currently configured design_rp_river_yr, for the
# summary timeseries plot only -- NOT written to river_forcing.nc (the file
# above is already saved). Mirrors exactly what rule 13 will build at this
# design RP, including the protection-discharge floor.
if int(has_glofas.sum()) > 0:
    preview_matrix = np.full((len(crossings), len(t_river)), np.nan)
    preview_matrix[has_glofas] = build_design_discharge_matrix(
        river_ds, has_glofas, eva_cfg["rp_fl"]
    )
    river_ds["discharge"] = (["crossing", "time"], preview_matrix)

# ── summary plots ─────────────────────────────────────────────────────────────

log.info("--- Summary plots ---")

plot_domain_map(
    bbox_poly=domain_poly,
    river_gdf=river_gdf,
    stations=stations,
    crossings=crossings,
    has_glofas=has_glofas,
    osm_land_path=snakemake.input.land_polygons,
    output_path=snakemake.output.plot_map,
)
plot_forcing_timeseries(
    surge_ds=surge_ds,
    river_ds=river_ds,
    return_period=return_period,
    output_path=snakemake.output.plot_timeseries,
)

# ── EVA diagnostic plot for the most hydrologically significant active
#    crossing (highest design-flood discharge) ────────────────────────────────
# Previously picked by raw SWORD reach "width" -- width does not reliably
# track actual GloFAS-matched discharge magnitude (a crossing's width and the
# accumulated flow at its matched GloFAS cell can diverge, e.g. distributary
# vs. mainstem reaches), so this could diagnose a minor crossing while a
# much larger one (with a much larger protection-level discharge threshold)
# went unplotted -- exactly the mismatch that made 07_forcing_eva.png look
# implausible next to 07_forcing_timeseries.png's protection-discharge lines.
# Selecting by flood_q directly ties this diagnostic to the same quantity
# that drives the actual forcing and protection-level correction.
active_idx = [i for i, g in enumerate(has_glofas) if g]
if active_idx:
    # eva.q_rp100 reflects design_rp_river_yr (injected into eva_cfg["rp_fl"]
    # above), so this still ties the diagnostic to the same quantity driving
    # the actual forcing/protection-level correction.
    diag_i = active_idx[int(np.argmax([results[i].q_rp100 for i in active_idx]))]

    i_lat, i_lon = int(cell_i_lat[diag_i]), int(cell_i_lon[diag_i])
    diag_label = (
        f"crossing{diag_i}_cell({cell_lat[diag_i]:.3f},{cell_lon[diag_i]:.3f})"
    )
    # Reuse the SAME (bias-corrected, if applicable) series analyse_cell was
    # actually fit on for this cell -- re-fetching raw glofas_clip here would
    # silently re-fit on different data than what bankfull_discharge/
    # discharge_rp_table/protection_discharge were computed from whenever GRDC
    # bias correction was applied to this cell.
    plot_cell_diagnostics(
        times=glofas_clip[time_dim].values,
        values=ts_cache[(i_lat, i_lon)],
        eva_cfg=eva_cfg,
        output_path=snakemake.output.plot_eva_diagnostics,
        label=diag_label,
    )
    log.info(f"Written EVA diagnostic plot: {snakemake.output.plot_eva_diagnostics}")
else:
    log.warning("No active GloFAS crossings — writing placeholder EVA diagnostic plot")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, "No active GloFAS crossings\n(EVA could not be fitted)",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=13, color="grey")
    ax.set_axis_off()
    Path(snakemake.output.plot_eva_diagnostics).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(snakemake.output.plot_eva_diagnostics, dpi=100, bbox_inches="tight")
    plt.close(fig)

# ── Bias-correction diagnostic plots (one per crossing/cell with a GRDC match) ──

bc_out_dir = Path(snakemake.output.plot_bias_correction)
bc_out_dir.mkdir(parents=True, exist_ok=True)

n_bc_plots = 0
plotted_cells: set[tuple[int, int]] = set()
for i in range(len(crossings)):
    if results[i] is None:
        continue
    cell_key = (int(cell_i_lat[i]), int(cell_i_lon[i]))
    bc_diag = bias_cache.get(cell_key)
    if bc_diag is None or cell_key in plotted_cells:
        continue
    plotted_cells.add(cell_key)

    i_lat, i_lon = cell_key
    label_i = f"crossing{i}_cell({lat_arr[i_lat]:.3f},{lon_arr[i_lon]:.3f})"
    plot_bias_correction(
        glofas_times=glofas_clip[time_dim].values,
        glofas_values=glofas_clip[glofas_variable]
            .isel({lat_dim: i_lat, lon_dim: i_lon})
            .values.astype(float),
        diagnostics=bc_diag,
        output_path=bc_out_dir / f"07_bias_correction_{label_i}.png",
        label=label_i,
    )
    n_bc_plots += 1

if n_bc_plots:
    log.info(f"Written {n_bc_plots} bias-correction diagnostic plot(s) to {bc_out_dir}")
else:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(
        0.5, 0.5,
        "No GRDC bias correction applied for any crossing\n"
        "(no GRDC station within search radius / insufficient overlap)",
        ha="center", va="center", transform=ax.transAxes,
        fontsize=13, color="grey",
    )
    ax.set_axis_off()
    fig.savefig(bc_out_dir / "07_bias_correction_none.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

profiler.stop()
log.info("Done")
