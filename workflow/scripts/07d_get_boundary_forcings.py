"""Build surge and river boundary forcing for a basin; run by rule get_boundary_forcings."""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr

from src.domain import load_domain
from src.extreme_values import (
    STANDARD_RETURN_PERIODS_YR,
    EVAResult,
    analyse_cell,
    bias_correct_discharge,
    eva_active_flags,
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
from src.plots import plot_domain_map, plot_forcing_timeseries, plot_surge_corrections, save_figure
from src.raster import latlon_dims
from src.river_network import normalize_channel_widths
from src.surge import (
    apply_mdt_correction,
    apply_slr_fingerprint,
    build_design_surge_matrix,
    build_surge_dataset,
    build_time_axis,
    compute_global_mean_slr,
    extract_hydrograph_components,
    interpolate_protection_level,
    load_mdt,
    load_selected_stations,
    load_slr_fingerprint,
    preview_station_rp_yr,
)
from src.profiling import ScriptProfiler

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
load_selected_stations        = profiler.wrap(load_selected_stations)
load_mdt                       = profiler.wrap(load_mdt)
apply_mdt_correction           = profiler.wrap(apply_mdt_correction)
load_slr_fingerprint           = profiler.wrap(load_slr_fingerprint)
compute_global_mean_slr        = profiler.wrap(compute_global_mean_slr)
apply_slr_fingerprint          = profiler.wrap(apply_slr_fingerprint)
build_surge_dataset           = profiler.wrap(build_surge_dataset)
extract_hydrograph_components = profiler.wrap(extract_hydrograph_components)
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

# domain
wgs84_bounds, domain_crs, domain_poly = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)
lon_min, lat_min, lon_max, lat_max = wgs84_bounds
domain_gdf = gpd.GeoDataFrame(geometry=[domain_poly], crs="EPSG:4326")
domain_utm = domain_gdf.to_crs(domain_crs)
log.info(f"Domain WGS84 bounds: {wgs84_bounds}, CRS: {domain_crs}")

# existing flood-protection level
# surge_protection_rp_yr sets the GOCO06s coastal protection crest
# (coastal_protection_crest_m) for the coastal weir in rule 12a. river_protection_rp_yr
# is unused: riverine protection is the calibrated riverbank weirs (rule 11).
with open(snakemake.input.protection_levels) as f:
    protection_summary = json.load(f)
surge_protection_rp_yr = float(protection_summary["surge_protection_rp_yr"])
log.info(
    f"Coastal protection RP={surge_protection_rp_yr:.1f} yr ({protection_summary['surge_protection_source']}), "
    f"dominant unit={protection_summary['dominant_iso']} (id={protection_summary['dominant_geounit_id']}) "
    f"-- feeds the coastal protection weir crest (always computed below)"
)

# surge forcing
log.info("--- Surge forcing ---")

surge_lead    = snakemake.params.lead_days
surge_period  = snakemake.params.surge_period_hr
surge_dt      = snakemake.params.dt_hr
# Diagnostic-preview RP only (storm_tide_at_rp_m columns, 07_surge_correction.png): the
# "default" scenario's surge_rp_yr. Rule 13 uses the requested scenario's surge_rp_yr.
surge_rp_yr = snakemake.params.surge_rp_yr
station_rp_yr = preview_station_rp_yr(surge_rp_yr)
surge_rp_is_numeric = station_rp_yr == surge_rp_yr
if not surge_rp_is_numeric:
    log.info(
        f"Default scenario surge_rp_yr={surge_rp_yr!r} has no storm-tide RP: "
        f"storm_tide_at_rp_m holds RP{station_rp_yr}; the water_level preview is the "
        f"{surge_rp_yr!r} boundary"
    )

# The stations rule select_surge_stations (07a) chose.
stations = load_selected_stations(
    snakemake.input.surge_data,
    domain_utm,
    domain_crs,
    snakemake.input.surge_stations,
    rp_yr=station_rp_yr,
)
log.info(f"Most distant selected station: {stations['dist_m'].max() / 1000:.1f} km")

# vertical-reference correction (MDT: local MSL -> GOCO06s geoid)
# Mandatory: the DEM is GOCO06s-referenced (rule 06). MDT is the mean sea
# surface ABOVE the geoid, so MSL-referenced COAST-RP levels get +MDT
# (H_GOCO06s = H_MSL + MDT), as 05a does for GEBCO.
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
stations["storm_tide_at_rp_m"] = stations["storm_tide_at_rp_raw_m"] + stations["mdt"]
log.info(
    f"MDT vertical correction applied (local MSL -> GOCO06s, +MDT): "
    f"delta = [{stations['mdt'].min():.3f}, {stations['mdt'].max():.3f}] m"
)
# SLR fingerprint (dimensionless, target-independent)
# Not scaled by slr_m and storm_tide_at_rp_m untouched: surge_forcing.nc is independent
# of any scenario's slr_m (config/scenarios.yml), which only rules
# build_sfincs/adapt_build_forcing_pre multiply by slr_fingerprint. Changing
# slr_m reruns only that scenario's build and event run.
slr_cfg = snakemake.params.surge_slr
if slr_cfg["enabled"]:
    slr_ds = load_slr_fingerprint(
        snakemake.input.slr_data, slr_cfg["ssp_scenario"],
        slr_cfg["confidence_level"], slr_cfg["year"], slr_cfg["quantile"],
    )
    global_mean_slr_m = compute_global_mean_slr(slr_ds)
    stations = apply_slr_fingerprint(
        stations, slr_ds, global_mean_slr_m, slr_cfg["fallback_search_deg"]
    )
    log.info(
        f"SLR fingerprint computed ({slr_cfg['ssp_scenario']}, {slr_cfg['year']}, "
        f"confidence={slr_cfg['confidence_level']}, quantile={slr_cfg['quantile']}): "
        f"ratio range [{stations['slr_fingerprint'].min():.3f}, "
        f"{stations['slr_fingerprint'].max():.3f}] (dimensionless; scaled by the "
        f"actual slr_m target downstream, at scenario build time)"
    )
else:
    stations["slr_fingerprint"] = np.nan

# River shares lead_days/dt_hr with surge on one time axis; only period_hr differs.
river_lead   = surge_lead
river_period = snakemake.params.river_period_hr
river_dt     = surge_dt

# Storm-tide hydrograph mode: the peaks sit after the lead-in plus the longer
# of the storm-tide half-window and, for real GloFAS floods, the room for the
# river's rising limb; the river wave's own lead is set so its peak lands on
# the surge peak. Before its window the surge holds the calm level.
hg_cfg = snakemake.params.surge_hydrograph
if hg_cfg["enabled"]:
    hg_window_hr = float(snakemake.params.hydrograph_window_hr)
    if river_period / 2.0 > hg_window_hr:
        raise ValueError(
            f"river period_hr/2 ({river_period / 2:g} h) exceeds the hydrograph window "
            f"({hg_window_hr:g} h) -- the river wave would not fit around the surge peak"
        )
    river_event_cfg = snakemake.params.river_event
    rising_hr = (
        24.0 * float(river_event_cfg["max_rising_days"])
        if river_event_cfg["source"] == "glofas_event"
        else 0.0
    )
    hg_peak_hr = surge_lead * 24 + max(hg_window_hr, rising_hr)
    river_lead = (hg_peak_hr - river_period / 2.0) / 24.0
    forcing_total_hr = hg_peak_hr + hg_window_hr
    log.info(
        f"Forcing total duration: {forcing_total_hr:.0f} h (lead-in {surge_lead * 24:.0f} h + "
        f"{max(hg_window_hr, rising_hr):g} h before and {hg_window_hr:g} h after the peak); "
        f"surge and river peaks at {hg_peak_hr:.0f} h"
    )
else:
    forcing_total_hr = max(surge_lead * 24 + surge_period, river_lead * 24 + river_period)
    log.info(
        f"Forcing total duration: {forcing_total_hr:.0f} h "
        f"(surge: {surge_lead * 24 + surge_period:.0f} h, "
        f"river: {river_lead * 24 + river_period:.0f} h)"
    )
t_surge = build_time_axis(surge_lead, surge_period, surge_dt, total_hr=forcing_total_hr)

# Calm sea level = mean MDT applied to storm_tide_at_rp_m, MDT-only (no SLR). Rules 10 and 13
# initialise sea cells at it.
calm_sea_level_m = float((stations["storm_tide_at_rp_m"] - stations["storm_tide_at_rp_raw_m"]).mean())
log.info(
    f"Surge boundary baseline (MDT-only, SLR applied downstream): {calm_sea_level_m:+.4f} m "
    f"(= mean(+MDT) across {len(stations)} stations — local MSL in model coords)"
)

# Per-station calm sea levels (+mdt_i, local MSL in model coordinates) keep each
# station's wave amplitude = storm_tide_at_rp_raw_m_i; scalar calm_sea_level_m is for zsini.
station_calm_sea_level_m = (stations["storm_tide_at_rp_m"] - stations["storm_tide_at_rp_raw_m"]).to_numpy()
surge_ds = build_surge_dataset(
    stations, t_surge, surge_lead, surge_period, station_rp_yr,
    calm_sea_level_m=calm_sea_level_m, station_calm_sea_level_m=station_calm_sea_level_m,
)

# Protection is a weir crest (rule 12a), never subtracted from water_level:
# a scalar offset would not act as a barrier for land below the corrected level.
# FLOPROS RP is regional, so the crest is the mean COAST-RP level at that RP
# over all stations; per-station values vary 10-50 cm (exposed vs sheltered)
# without reflecting actual defense heights.
protection_level_raw = interpolate_protection_level(stations, surge_protection_rp_yr)
mean_prot_raw = float(protection_level_raw.mean())
protection_level = np.full(len(stations), mean_prot_raw)   # uniform across stations

# COAST-RP is local MSL; +MDT (as for storm_tide_at_rp_m) puts the crest in the DEM's GOCO06s frame.
coastal_protection_crest_m = mean_prot_raw + float(stations["mdt"].mean())

surge_ds["protection_level"] = (
    ["station"],
    protection_level,
    {
        "units": "m",
        "long_name": f"existing flood-protection level (RP{surge_protection_rp_yr:g} yr, FLOPROS coastal, local MSL)",
    },
)
surge_ds["protection_rp_yr"] = (
    [],
    float(surge_protection_rp_yr),
    {"units": "yr", "long_name": "FLOPROS coastal protection return period used"},
)
surge_ds["coastal_protection_crest_m"] = (
    [],
    coastal_protection_crest_m,
    {
        "units": "m",
        "long_name": (
            f"GOCO06s-referenced coastal protection crest elevation (RP{surge_protection_rp_yr:g} yr, "
            "FLOPROS coastal, MDT-corrected) -- the weir crest baked into the model in rule 12a"
        ),
    },
)
log.info(
    f"Coastal protection crest: raw(local MSL)={mean_prot_raw:+.4f} m -> "
    f"crest(GOCO06s)={coastal_protection_crest_m:+.4f} m (mean across {len(stations)} stations, "
    f"RP{surge_protection_rp_yr:g} yr; per-station range was "
    f"[{protection_level_raw.min():.3f}, {protection_level_raw.max():.3f}] m)"
)

# storm-tide hydrographs (HGRAPHER): tide + surge shape per station
# Used by build_design_surge_matrix for every surge scenario (an RP and "Tide").
if hg_cfg["enabled"]:
    hg_time_hr, hg_tide, hg_shape, hg_rps_yr, hg_match_km = extract_hydrograph_components(
        snakemake.input.storm_tide_hydrographs,
        stations.geometry.x.to_numpy(), stations.geometry.y.to_numpy(), hg_window_hr,
    )
    max_match_km = float(hg_cfg["max_match_km"])
    if np.nanmax(hg_match_km) > max_match_km:
        raise ValueError(
            f"selected surge station(s) lie up to {np.nanmax(hg_match_km):.1f} km from the nearest "
            f"storm-tide hydrograph location (max_match_km={max_match_km:g}) -- storm_tide_hydrographs.nc "
            f"was built for other stations; rerun rules select_surge_stations/extract_gtsm_series/"
            f"storm_tide_hydrographs for this basin"
        )
    surge_ds = surge_ds.assign_coords(hg_time_hr=("hg_time_hr", hg_time_hr), hg_rp_yr=("hg_rp_yr", hg_rps_yr))
    surge_ds["hg_rp_yr"].attrs = {"units": "yr", "long_name": "return period each surge shape was derived for"}
    surge_ds["hg_time_hr"].attrs = {"units": "hours", "long_name": "hours relative to the surge peak / tidal high water"}
    surge_ds["hg_tide_m"] = (
        ["station", "hg_time_hr"], hg_tide,
        {"units": "m", "long_name": "average tide signal (HGRAPHER), local MSL, high water at hg_time_hr=0"},
    )
    surge_ds["hg_surge_shape"] = (
        ["station", "hg_rp_yr", "hg_time_hr"], hg_shape,
        {"units": "1", "long_name": "normalised surge hydrograph per return period (HGRAPHER), peak=1 at hg_time_hr=0"},
    )
    surge_ds["hg_match_km"] = (
        ["station"], hg_match_km,
        {"units": "km", "long_name": "distance to the matched storm-tide hydrograph location"},
    )
    surge_ds.attrs.update({
        "hg_peak_hr": float(hg_peak_hr),
        "hg_window_hr": float(hg_window_hr),
        "hg_ramp_hr": float(hg_cfg["ramp_hr"]),
    })
    # Diagnostic preview (07 timeseries plot) at the default RP, built the
    # same way the production boundary is (rule 14) instead of the cosine.
    surge_ds["water_level"] = (
        ["station", "time"], build_design_surge_matrix(surge_ds, surge_rp_yr),
        surge_ds["water_level"].attrs,
    )
    log.info(
        f"Storm-tide hydrographs: {len(hg_time_hr)} steps (+-{hg_window_hr:g} h), peak at "
        f"{hg_peak_hr:.0f} h, tide high water [{hg_tide.max(axis=1).min():.2f}, "
        f"{hg_tide.max(axis=1).max():.2f}] m (local MSL), match distance <= {np.nanmax(hg_match_km):.2f} km"
    )
elif not surge_rp_is_numeric:
    # The cosine above rises to RP{station_rp_yr}; show the default scenario's own
    # boundary instead (None: calm sea; 'Tide' raises, as rule 14 would).
    surge_ds["water_level"] = (
        ["station", "time"], build_design_surge_matrix(surge_ds, surge_rp_yr),
        surge_ds["water_level"].attrs,
    )

Path(snakemake.output.surge_forcing).parent.mkdir(parents=True, exist_ok=True)
surge_ds.to_netcdf(snakemake.output.surge_forcing)
log.info(f"Written surge forcing ({len(stations)} stations): {snakemake.output.surge_forcing}")

plot_surge_corrections(
    stations,
    output_path=snakemake.output.plot_surge_correction,
)
log.info(f"Wrote surge correction diagnostic plot: {snakemake.output.plot_surge_correction}")

# river forcing
log.info("--- River forcing ---")

eva_cfg = dict(snakemake.params.eva)
# eva_cfg["preview_rp_yr"] is a fixed basin-level diagnostic RP, not a scenario's
# river_rp_yr: this rule has no {scenario} wildcard, so a scenario-derived value
# would rerun everything downstream on every scenario switch.
# discharge_rp_table covers every standard RP regardless.
# Main grid resolution only: the subgrid does not change the cell size SFINCS
# solves on.
sfincs_resolution_m = float(snakemake.params.sfincs_resolution_m)
glofas_radius_m = float(snakemake.params.glofas_search_radius_km) * 1000.0
glofas_min_q    = float(snakemake.params.glofas_min_mean_discharge_m3s)
glofas_variable = "dis24"  # sole variable in data_catalogue's river_discharge (GloFAS v4) source
bias_cfg        = dict(snakemake.params.bias_correction)
grdc_radius_m   = float(bias_cfg["grdc_search_radius_km"]) * 1000.0

grdc_stations = load_grdc_stations(snakemake.input.grdc_data)
log.info(f"  Loaded GRDC station table: {len(grdc_stations)} station(s)")

river_gdf = gpd.read_file(snakemake.input.spec_river_network)
if river_gdf.crs is not None and river_gdf.crs != domain_gdf.crs:
    river_gdf = river_gdf.to_crs("EPSG:4326")

# Same width fix as rule clean_river_network, which runs later in the DAG.
river_gdf = normalize_channel_widths(
    river_gdf, max_width_fill_m=float(snakemake.params.max_width_fill_m)
)

domain_reach_ids = set(str(reach_id) for reach_id in river_gdf["reach_id"].dropna())

# Step 1: boundary crossings
crossings = find_boundary_crossings(river_gdf, domain_poly)
log.info(
    f"Step 1 — boundary crossings: {len(crossings)} reach(es) intersect domain boundary"
)

if crossings.empty:
    log.warning("No river boundary crossings found; river forcing will be empty")
    has_glofas = np.zeros(0, dtype=bool)
    cell_lon = np.array([])
    cell_lat = np.array([])
    bankfull_discharge_m3s = np.array([])
    mean_discharge_m3s = np.array([])
    flood_q = np.array([])
    results: list = []
    bias_corrected_arr    = np.zeros(0, dtype=np.int8)
    grdc_station_id_arr   = np.full(0, -1, dtype=np.int64)
    grdc_correlation_arr  = np.full(0, np.nan)
    grdc_overlap_days_arr = np.full(0, np.nan)
    # Empty dataset so the declared output exists and is valid netCDF.
    Path(snakemake.output.glofas_clip).parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset().to_netcdf(snakemake.output.glofas_clip)
else:
    # Step 2: clip crossing reach to its domain-entry point
    # Walks to the next downstream reach (rch_id_dn) only when the crossing reach
    # merely touches the boundary; width/max_width come from the resolved reach.
    crossings = resolve_inside_domain_reaches(crossings, river_gdf, domain_poly)
    log.info(
        f"Step 2 — domain entry point: geometry updated to the domain-entry "
        f"point for {len(crossings)} crossing(s)"
    )

    # Step 3: Filter — enters_domain
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

    # Step 4: visible_on_grid (diagnostic only -- does NOT gate anything)
    # Not a gate: it would tie rule 08's seed set (which reaches survive) to the
    # grid resolution and could drop a whole valid branch behind a narrow crossing.
    w_col = "width"
    if w_col in crossings.columns:
        crossings["visible_on_grid"] = (
            crossings[w_col].fillna(0.0).astype(float) >= sfincs_resolution_m
        )
    else:
        log.warning(
            f"Width column '{w_col}' not found in river network; "
            f"visible_on_grid = True for all crossings"
        )
        crossings["visible_on_grid"] = True

    n_visible = int((crossings["enters_domain"] & crossings["visible_on_grid"]).sum())
    log.info(
        f"Step 4 — visible_on_grid ({w_col} >= {sfincs_resolution_m:.0f} m), informational only: "
        f"{n_visible}/{n_enters} entering crossing(s) wide enough for the grid"
    )
    if n_visible < n_enters:
        too_narrow = crossings["enters_domain"] & ~crossings["visible_on_grid"]
        narrow_widths = crossings.loc[too_narrow, w_col].tolist()
        log.info(
            f"  Narrower than grid resolution (NOT excluded): {w_col} = "
            f"{[f'{w:.0f}' for w in narrow_widths]} m"
        )

    # Step 5: GloFAS matching
    # Highest-mean-discharge GloFAS cell within glofas_radius_m above glofas_min_q;
    # has_glofas only when EVA gives a finite RP2.
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

    # Persist the clipped subset as a rule output, reusable without re-running
    # this ~120-150 s clip for the same basin/domain bounds.
    Path(snakemake.output.glofas_clip).parent.mkdir(parents=True, exist_ok=True)
    glofas_clip.to_netcdf(snakemake.output.glofas_clip)
    log.info(f"  Written GloFAS clip cache: {snakemake.output.glofas_clip}")

    lat_dim, lon_dim = latlon_dims(glofas_clip)
    time_dim = "valid_time" if "valid_time" in glofas_clip.dims else "time"
    lat_arr  = glofas_clip[lat_dim].values
    lon_arr  = glofas_clip[lon_dim].values
    times_arr = glofas_clip[time_dim].values

    n = len(crossings)
    has_glofas = np.zeros(n, dtype=bool)
    cell_lon   = np.full(n, np.nan)
    cell_lat   = np.full(n, np.nan)
    bankfull_discharge_m3s   = np.full(n, np.nan)
    mean_discharge_m3s       = np.full(n, np.nan)
    discharge_rp_table = np.full((n, len(STANDARD_RETURN_PERIODS_YR)), np.nan)
    results      = [None] * n

    bias_corrected_arr    = np.zeros(n, dtype=np.int8)
    grdc_station_id_arr   = np.full(n, -1, dtype=np.int64)
    grdc_correlation_arr  = np.full(n, np.nan)
    grdc_overlap_days_arr = np.full(n, np.nan)

    cell_i_lat = np.full(n, -1, dtype=int)
    cell_i_lon = np.full(n, -1, dtype=int)

    eva_cache: dict[tuple[int, int], EVAResult] = {}
    bias_cache: dict[tuple[int, int], dict | None] = {}
    # Series fed to analyse_cell (bias-corrected where applicable), reused so
    # plots and daily series use the same data as the fitted discharges.
    ts_cache: dict[tuple[int, int], np.ndarray] = {}
    analysed: list[tuple[int, str]] = []

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

            # GRDC bias correction (first crossing reaching this cell)
            # TODO: GRDC-Daily.nc covers only the 8 basins of 8_polygons.json, not
            # the globe; other basins typically fall back to GloFAS-as-truth.
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

            eva = analyse_cell(times_arr, ts, eva_cfg, label=label)
            eva_cache[(i_lat, i_lon)] = eva

        results[i] = eva
        cell_i_lat[i] = i_lat
        cell_i_lon[i] = i_lon
        analysed.append((i, label))

    active_flags = eva_active_flags([(label, results[i]) for i, label in analysed])
    for (i, label), is_active in zip(analysed, active_flags):
        if not is_active:
            continue
        eva = results[i]
        i_lat, i_lon = int(cell_i_lat[i]), int(cell_i_lon[i])
        has_glofas[i] = True
        cell_lon[i]   = float(lon_arr[i_lon])
        cell_lat[i]   = float(lat_arr[i_lat])
        bankfull_discharge_m3s[i] = eva.q_rp2
        # Same series as the bankfull fit, so "mean" and "bankfull" river_rp_yr agree.
        mean_discharge_m3s[i] = float(np.nanmean(ts_cache[(i_lat, i_lon)]))
        # From the fitted POT/GPD curve; each scenario's river_rp_yr is looked up
        # here at SFINCS build time.
        discharge_rp_table[i] = gpd_return_value_table(
            eva.pot_threshold, eva.pot_scale, eva.pot_shape, eva.pot_peaks_per_year,
        )
        log.info(
            f"  {label}: GloFAS ({lat_arr[i_lat]:.3f}°N, {lon_arr[i_lon]:.3f}°E)  "
            f"bankfull_discharge_m3s = {bankfull_discharge_m3s[i]:.1f} m³/s  "
            f"Q_rp{eva_cfg['preview_rp_yr']:g} (diagnostic) = {eva.q_rp100:.1f} m³/s"
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
    crossings, has_glofas, bankfull_discharge_m3s, discharge_rp_table, STANDARD_RETURN_PERIODS_YR,
    cell_lon, cell_lat, t_river, river_lead, river_period, results,
    bankfull_rp_yr=eva_cfg["bankfull_rp_yr"],
    mean_discharge_m3s=mean_discharge_m3s,
    bias_corrected=bias_corrected_arr,
    grdc_station_id=grdc_station_id_arr,
    grdc_correlation=grdc_correlation_arr,
    grdc_overlap_days=grdc_overlap_days_arr,
)

# Daily series the return periods were fitted on, for rule 14's real-flood
# events. 'day' counts days since 1970-01-01 (read with decode_times=False).
if int(has_glofas.sum()) > 0:
    daily = np.full((len(crossings), len(times_arr)), np.nan, dtype=np.float32)
    for i in np.flatnonzero(has_glofas):
        daily[i] = ts_cache[(cell_i_lat[i], cell_i_lon[i])]
    river_ds = river_ds.assign_coords(
        day=("day", (times_arr.astype("datetime64[D]") - np.datetime64("1970-01-01")).astype(np.int64),
             {"long_name": "day of the daily discharge series, days since 1970-01-01"})
    )
    river_ds["discharge_daily"] = (
        ["crossing", "day"], daily,
        {"units": "m3 s-1", "long_name": "daily mean discharge (GloFAS, bias-corrected where bias_corrected)"},
    )
# End of the lead-in: where a real flood event's discharge blends in from the mean.
river_ds.attrs["event_start_hr"] = float(surge_lead) * 24.0

Path(snakemake.output.river_forcing).parent.mkdir(parents=True, exist_ok=True)
river_ds.to_netcdf(snakemake.output.river_forcing)
log.info(
    f"Written river forcing ({len(crossings)} crossings, "
    f"{int(has_glofas.sum())} with GloFAS): {snakemake.output.river_forcing}"
)

# Preview at the diagnostic RP for the timeseries plot only; not written to
# river_forcing.nc.
if int(has_glofas.sum()) > 0:
    preview_matrix = np.full((len(crossings), len(t_river)), np.nan)
    preview_matrix[has_glofas] = build_design_discharge_matrix(
        river_ds, has_glofas, eva_cfg["preview_rp_yr"], event=snakemake.params.river_event,
    )
    river_ds["discharge"] = (["crossing", "time"], preview_matrix)

# summary plots
log.info("--- Summary plots ---")

plot_domain_map(
    bbox_poly=domain_poly,
    river_gdf=river_gdf,
    stations=stations,
    crossings=crossings,
    has_glofas=has_glofas,
    land_polygons_path=snakemake.input.land_polygons,
    output_path=snakemake.output.plot_map,
)
plot_forcing_timeseries(
    surge_ds=surge_ds,
    river_ds=river_ds,
    rp_yr=surge_rp_yr,
    output_path=snakemake.output.plot_timeseries,
)

# EVA diagnostic plot for the active crossing with the highest q_rp100
# Selected by discharge rather than raw SWORD reach "width": a crossing's
# width and the accumulated flow at its matched GloFAS cell can diverge
# (e.g. distributary vs. mainstem reaches). Selecting by q_rp100 ties this
# diagnostic to discharge, the quantity that drives the forcing.
active_idx = [i for i, g in enumerate(has_glofas) if g]
if active_idx:
    # eva.q_rp100 is at the fixed diagnostic RP (eva_cfg["preview_rp_yr"]): a stable,
    # scenario-independent size ranking.
    diag_i = active_idx[int(np.argmax([results[i].q_rp100 for i in active_idx]))]

    i_lat, i_lon = int(cell_i_lat[diag_i]), int(cell_i_lon[diag_i])
    diag_label = (
        f"crossing{diag_i}_cell({cell_lat[diag_i]:.3f},{cell_lon[diag_i]:.3f})"
    )
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
    save_figure(fig, snakemake.output.plot_eva_diagnostics, dpi=100, bbox_inches="tight")

# Bias-correction diagnostic plots (one per crossing/cell with a GRDC match)
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
    save_figure(fig, bc_out_dir / "07_bias_correction_none.png", dpi=100, bbox_inches="tight")

profiler.stop()
log.info("Done")
