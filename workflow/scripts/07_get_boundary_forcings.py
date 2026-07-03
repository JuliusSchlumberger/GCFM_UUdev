from pathlib import Path

import geopandas as gpd
import numpy as np

from src.domain import load_domain
from src.extreme_values import EVAResult, analyse_cell, plot_cell_diagnostics
from src.log import setup_logging
from src.river_forcing import (
    build_river_dataset,
    find_best_glofas_cell,
    find_boundary_crossings,
    has_downstream_in_domain,
    load_glofas_clip,
    resolve_dem_elevation_reach,
    resolve_inside_domain_reaches,
)
from src.plots import plot_domain_map, plot_forcing_timeseries
from src.surge import (
    build_surge_dataset,
    build_time_axis,
    compute_distances_to_bbox,
    load_coastrp_stations,
    select_nearest_stations,
)
from src.profiling import ScriptProfiler

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
load_coastrp_stations         = profiler.wrap(load_coastrp_stations)
compute_distances_to_bbox     = profiler.wrap(compute_distances_to_bbox)
select_nearest_stations       = profiler.wrap(select_nearest_stations)
build_surge_dataset           = profiler.wrap(build_surge_dataset)
find_boundary_crossings       = profiler.wrap(find_boundary_crossings)
resolve_inside_domain_reaches = profiler.wrap(resolve_inside_domain_reaches)
load_glofas_clip              = profiler.wrap(load_glofas_clip)
find_best_glofas_cell         = profiler.wrap(find_best_glofas_cell)
analyse_cell                  = profiler.wrap(analyse_cell)
build_river_dataset           = profiler.wrap(build_river_dataset)

# ── domain ────────────────────────────────────────────────────────────────────

wgs84_bounds, domain_crs, domain_poly = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)
lon_min, lat_min, lon_max, lat_max = wgs84_bounds
domain_gdf = gpd.GeoDataFrame(geometry=[domain_poly], crs="EPSG:4326")
domain_utm = domain_gdf.to_crs(domain_crs)
log.info(f"Domain WGS84 bounds: {wgs84_bounds}, CRS: {domain_crs}")

# ── surge forcing ─────────────────────────────────────────────────────────────

log.info("--- Surge forcing ---")

surge_lead    = snakemake.params.surge_lead_days
surge_period  = snakemake.params.surge_period_hr
surge_dt      = snakemake.params.surge_dt_hr
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

# Read river timing params early so we can compute the shared total duration
# before building the surge time axis.  They are re-used (same values) when the
# river section starts below.
river_lead   = snakemake.params.river_lead_days
river_period = snakemake.params.river_period_hr
river_dt     = snakemake.params.river_dt_hr

forcing_total_hr = max(surge_lead * 24 + surge_period, river_lead * 24 + river_period)
log.info(
    f"Forcing total duration: {forcing_total_hr:.0f} h "
    f"(surge: {surge_lead * 24 + surge_period:.0f} h, "
    f"river: {river_lead * 24 + river_period:.0f} h)"
)
t_surge  = build_time_axis(surge_lead, surge_period, surge_dt, total_hr=forcing_total_hr)
surge_ds = build_surge_dataset(stations, t_surge, surge_lead, surge_period, return_period)

Path(snakemake.output.surge_forcing).parent.mkdir(parents=True, exist_ok=True)
surge_ds.to_netcdf(snakemake.output.surge_forcing)
log.info(f"Written surge forcing ({len(stations)} stations): {snakemake.output.surge_forcing}")

# ── river forcing ─────────────────────────────────────────────────────────────

log.info("--- River forcing ---")

eva_cfg         = dict(snakemake.params.eva)
sfincs_grid_res = float(snakemake.params.sfincs_resolution)
subgrid_nr_cels = int(snakemake.params.sfincs_nr_subgridcells)
sfincs_res      = float(sfincs_grid_res/(2* subgrid_nr_cels)) if subgrid_nr_cels > 0 else sfincs_grid_res
width_column    = snakemake.params.width_column
glofas_radius_m = float(snakemake.params.glofas_search_radius_km) * 1000.0
glofas_min_q    = float(snakemake.params.glofas_min_mean_discharge)
glofas_variable = snakemake.params.glofas_variable

river_gdf = gpd.read_file(snakemake.input.spec_river_network)
if river_gdf.crs is not None and river_gdf.crs != domain_gdf.crs:
    river_gdf = river_gdf.to_crs("EPSG:4326")

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
    results: list = []
    # No crossings ⇒ has_glofas is empty ⇒ rule 05's clean network has no
    # is_seed reaches ⇒ rule 07 exits before reading glofas_clip — write an
    # empty sentinel so the declared Snakemake output still exists.
    Path(snakemake.output.glofas_clip).parent.mkdir(parents=True, exist_ok=True)
    Path(snakemake.output.glofas_clip).touch()
else:
    # ── Step 2: walk downstream to inside-domain reach ────────────────────────
    # For each crossing reach, follow rch_id_dn until a reach centroid lies
    # inside the domain polygon.  Updates crossings.geometry to that centroid
    # and overwrites width/max_width with the inside-domain reach's attributes.
    crossings = resolve_inside_domain_reaches(crossings, river_gdf, domain_poly)
    log.info(
        f"Step 2 — inside-domain reach: geometry updated to inside-domain centroid "
        f"for {len(crossings)} crossing(s)"
    )

    # ── Step 2b: DEM elevation filter ────────────────────────────────────────
    # Walk further downstream until the inside-domain reach's max_elevation
    # falls within the valid DEM range (clip_elevation_m + buffer).  Crossings
    # that never reach a qualifying reach are marked within_dem_range=False and
    # excluded from GloFAS matching in Step 5.
    if snakemake.params.dem_elev_filter_enabled:
        dem_cap      = float(snakemake.params.dem_elev_clip_m)
        elev_buf     = float(snakemake.params.dem_elev_filter_buffer_m)
        elev_thresh  = dem_cap + elev_buf
        crossings = resolve_dem_elevation_reach(
            crossings, river_gdf, elev_thresh, domain_poly
        )
        n_in_range = int(crossings["within_dem_range"].sum())
        log.info(
            f"Step 2b — DEM elevation filter (max_elevation ≤ {elev_thresh:.1f} m): "
            f"{n_in_range}/{len(crossings)} crossing(s) within DEM range"
        )
    else:
        crossings["within_dem_range"] = True

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

    # ── Step 4: Filter — visible_on_grid ─────────────────────────────────────
    # Uses the inside-domain reach's width (set by resolve_inside_domain_reaches).
    # The configured width_column ('width' or 'max_width') is compared against
    # the SFINCS grid resolution; reaches narrower than one grid cell are
    # invisible to the model and receive no forcing.
    w_col = width_column if width_column in crossings.columns else "width"
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
        f"Step 4 — visible_on_grid ({w_col} >= {sfincs_res:.0f} m): "
        f"{n_visible}/{n_enters} entering crossing(s) wide enough for the grid"
    )
    if n_visible < n_enters:
        too_narrow = crossings["enters_domain"] & ~crossings["visible_on_grid"]
        narrow_widths = crossings.loc[too_narrow, w_col].tolist()
        log.info(
            f"  Excluded (too narrow): {w_col} = {[f'{w:.0f}' for w in narrow_widths]} m"
        )

    # ── Step 5: GloFAS matching ───────────────────────────────────────────────
    # For each qualifying crossing (enters_domain AND visible_on_grid AND
    # within_dem_range), search within glofas_radius_m for GloFAS cells whose
    # mean discharge exceeds glofas_min_q.  Among qualifying cells, pick the
    # one with the highest mean discharge.  Run EVA on that cell; mark
    # has_glofas=True only when RP2 is finite (EVA converged).
    n_qualifying = int(
        (crossings["enters_domain"] & crossings["visible_on_grid"] & crossings["within_dem_range"]).sum()
    )
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
    bankfull_q = np.full(n, np.nan)
    flood_q    = np.full(n, np.nan)
    results    = [None] * n

    eva_cache: dict[tuple[int, int], EVAResult] = {}

    for i, row in enumerate(crossings.itertuples()):
        if not (row.enters_domain and row.visible_on_grid and row.within_dem_range):
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
            eva = analyse_cell(times_arr, ts, eva_cfg, label=label)
            eva_cache[(i_lat, i_lon)] = eva

        results[i] = eva

        if not np.isfinite(eva.q_rp2):
            log.warning(f"  {label}: EVA did not produce a finite RP2 — crossing inactive")
            continue

        has_glofas[i] = True
        cell_lon[i]   = float(lon_arr[i_lon])
        cell_lat[i]   = float(lat_arr[i_lat])
        bankfull_q[i] = eva.q_rp2
        flood_q[i]    = eva.q_rp100 if np.isfinite(eva.q_rp100) else eva.q_rp2
        log.info(
            f"  {label}: GloFAS ({lat_arr[i_lat]:.3f}°N, {lon_arr[i_lon]:.3f}°E)  "
            f"Q_bankfull = {bankfull_q[i]:.1f} m³/s  Q_flood = {flood_q[i]:.1f} m³/s"
        )

    n_has_glofas = int(has_glofas.sum())
    log.info(
        f"Step 5 summary — has_glofas: {n_has_glofas}/{n_qualifying} qualifying "
        f"crossing(s) matched to GloFAS and EVA successful"
    )

t_river  = build_time_axis(river_lead, river_period, river_dt, total_hr=forcing_total_hr)
river_ds = build_river_dataset(
    crossings, has_glofas, bankfull_q, flood_q, cell_lon, cell_lat,
    t_river, river_lead, river_period, results,
    rp_bankfull=eva_cfg.get("rp_bf", 2),
    rp_flood=eva_cfg.get("rp_fl", 100),
)

Path(snakemake.output.river_forcing).parent.mkdir(parents=True, exist_ok=True)
river_ds.to_netcdf(snakemake.output.river_forcing)
log.info(
    f"Written river forcing ({len(crossings)} crossings, "
    f"{int(has_glofas.sum())} with GloFAS): {snakemake.output.river_forcing}"
)

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

# ── EVA diagnostic plot for the widest active crossing ────────────────────────

active_idx = [i for i, g in enumerate(has_glofas) if g]
if active_idx:
    w_col_diag = width_column if width_column in crossings.columns else "width"
    if w_col_diag in crossings.columns:
        widths = [float(crossings.iloc[i].get(w_col_diag) or 0) for i in active_idx]
        diag_i = active_idx[int(np.argmax(widths))]
    else:
        diag_i = active_idx[0]

    lat_dim  = "latitude"   if "latitude"   in glofas_clip.dims else "lat"
    lon_dim  = "longitude"  if "longitude"  in glofas_clip.dims else "lon"
    time_dim = "valid_time" if "valid_time" in glofas_clip.dims else "time"
    lat_vals = glofas_clip[lat_dim].values
    lon_vals = glofas_clip[lon_dim].values
    i_lat = int(np.argmin(np.abs(lat_vals - cell_lat[diag_i])))
    i_lon = int(np.argmin(np.abs(lon_vals - cell_lon[diag_i])))
    diag_label = (
        f"crossing{diag_i}_cell({cell_lat[diag_i]:.3f},{cell_lon[diag_i]:.3f})"
    )
    plot_cell_diagnostics(
        times=glofas_clip[time_dim].values,
        values=glofas_clip[glofas_variable]
               .isel({lat_dim: i_lat, lon_dim: i_lon})
               .values.astype(float),
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

profiler.stop()
log.info("Done")
