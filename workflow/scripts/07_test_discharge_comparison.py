"""Validate bankfull discharges from three sources against Lin et al. rivers_ge30m Q2.

Dataset naming
--------------
SWORD ∩ GloFAS  –  GloFAS-seeded bankfull Q propagated through the SWORD network
                    (= bankfull_discharge_acc from the clean river network)
Lin             –  Lin et al. modelled Q2 (2-yr return period)
Lin ∩ GloFAS    –  GloFAS Q2 derived by AMAX/GEV at the position of each Lin reach

Figure layout (2 rows × 3 columns; bottom-right cell is empty)
--------------------------------------------------------------
Row 1 — SWORD ∩ GloFAS vs Lin
  (1,1)  Q: SWORD∩GloFAS vs Lin                — main discharge comparison
  (1,2)  ΔQ SWORD∩GloFAS-Lin vs reach offset   — spatial mismatch effect
  (1,3)  Channel width — SWORD vs Lin

Row 2 — Lin ∩ GloFAS vs Lin
  (2,1)  Q: Lin∩GloFAS vs Lin                  — raw GloFAS against reference
  (2,2)  Q: SWORD∩GloFAS vs Lin∩GloFAS         — SWORD∩GloFAS relative to the
                                                   GloFAS signal at the same locations
  (2,3)  [empty]
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from src.domain import load_domain
from src.extreme_values import analyse_cell_gev_only
from src.log import setup_logging
from src.profiling import ScriptProfiler

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
analyse_cell_gev_only = profiler.wrap(analyse_cell_gev_only)


# ── helpers ───────────────────────────────────────────────────────────────────

def _placeholder(output_path: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center",
            transform=ax.transAxes, fontsize=12, color="grey")
    ax.set_axis_off()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _scatter(ax, x, y, xlabel, ylabel, title,
             log_scale=False, unity_line=False):
    """Scatter with Pearson r, regression line, and optional 1:1 reference."""
    mask = np.isfinite(x) & np.isfinite(y)
    if log_scale:
        mask &= (x > 0) & (y > 0)
    xm, ym = np.asarray(x)[mask], np.asarray(y)[mask]

    ax.scatter(xm, ym, alpha=0.65, edgecolors="none", s=40, color="steelblue")

    if len(xm) >= 3:
        r, p   = stats.pearsonr(xm, ym)
        p_str  = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
        m, b   = np.polyfit(xm, ym, 1)
        xl     = np.array([xm.min(), xm.max()])
        ax.plot(xl, m * xl + b, "r--", linewidth=1.2,
                label=f"fit  r={r:.3f}  p={p_str}")
        ax.legend(fontsize=8, framealpha=0.9)

    if unity_line:
        lim = [min(xm.min(), ym.min()), max(xm.max(), ym.max())]
        ax.plot(lim, lim, "k:", linewidth=0.8, alpha=0.6)

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  (n={len(xm)})")
    ax.grid(True, alpha=0.3, linewidth=0.5)


# ── domain ────────────────────────────────────────────────────────────────────

wgs84_bounds, domain_crs, domain_poly = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)
lon_min, lat_min, lon_max, lat_max  = wgs84_bounds
buffer     = float(snakemake.params.buffer_deg)
eva_cfg    = dict(snakemake.params.eva)
glofas_var = str(snakemake.params.glofas_variable)
log.info(f"Domain: {wgs84_bounds}  CRS: {domain_crs}")

# ── SWORD ∩ GloFAS — seed reaches from clean network ────────────────────────

rivers = gpd.read_file(snakemake.input.clean_river_network)
if "is_seed" not in rivers.columns:
    _placeholder(snakemake.output.plot_discharge_comparison,
                 "No 'is_seed' column — re-run rule 05 to regenerate clean network")
    raise SystemExit(0)

seed_reaches = rivers[rivers["is_seed"].astype(bool)].copy()
# is_seed=True marks the SWORD reaches snapped from has_glofas=1 crossings in
# river_forcing.nc.  Those crossings were already deduplicated in rule 04:
#   (1) width-based — widest reach per GloFAS cell kept
#   (2) main-path — most-downstream crossing per river arm with downstream reaches kept
# So the testing uses the same filtered seed set as the rest of the pipeline.
log.info(
    f"SWORD ∩ GloFAS seed reaches: {len(seed_reaches)} / {len(rivers)} "
    f"(= SWORD reaches from deduplicated has_glofas=1 crossings in river_forcing.nc)"
)
if seed_reaches.empty:
    _placeholder(snakemake.output.plot_discharge_comparison,
                 "No active seed reaches for this basin")
    raise SystemExit(0)

# ── Lin — clipped to domain + buffer ─────────────────────────────────────────

bbox_buf = (lon_min - buffer, lat_min - buffer, lon_max + buffer, lat_max + buffer)
lin = gpd.read_file(snakemake.input.rivers_lin, bbox=bbox_buf,
                    layer=snakemake.params.lin_layer)
log.info(f"Lin reaches loaded: {len(lin)} within domain + {buffer}° buffer (unfiltered)")

min_order = int(snakemake.params.min_stream_order)
if "order" in lin.columns:
    lin = lin[pd.to_numeric(lin["order"], errors="coerce") > min_order].copy()
    log.info(f"Lin reaches after stream order filter (order > {min_order}): {len(lin)}")
else:
    log.warning("Lin dataset has no 'order' column; stream order filter skipped")

if lin.empty:
    _placeholder(snakemake.output.plot_discharge_comparison,
                 f"No Lin reaches in domain + {buffer}° buffer")
    raise SystemExit(0)

# Keep WGS84 centroids for GloFAS cell lookup before projecting.
# Centroids are computed in the projected (metric) CRS and reprojected back to
# WGS84 — computing them directly in a geographic CRS distorts the result and
# triggers a geopandas UserWarning.
lin["centroid_wgs"] = lin.geometry.to_crs(domain_crs).centroid.to_crs("EPSG:4326")
lin_centroid_wgs = lin.set_index("COMID")["centroid_wgs"].to_dict()

# ── nearest-reach matching: SWORD seed → Lin ─────────────────────────────────

seed_utm = seed_reaches[
    [c for c in ["reach_id", "bankfull_discharge_acc", "width", "geometry"]
     if c in seed_reaches.columns]
].copy().to_crs(domain_crs)
seed_utm["geometry"] = seed_utm.geometry.centroid

lin_utm = lin[
    [c for c in ["COMID", "Q2", "width_m", "geometry"] if c in lin.columns]
].copy().to_crs(domain_crs)
lin_utm["geometry"] = lin_utm.geometry.centroid

matched = gpd.sjoin_nearest(
    seed_utm.reset_index(drop=True),
    lin_utm.reset_index(drop=True),
    how="left", distance_col="dist_m",
)
log.info(
    f"Matched {len(matched)} seed reaches; "
    f"median offset = {matched['dist_m'].median() / 1000:.1f} km"
)

df = pd.DataFrame({
    "q_sword_glofas": pd.to_numeric(
        matched.get("bankfull_discharge_acc", np.nan), errors="coerce"),
    "q_lin":          pd.to_numeric(matched.get("Q2",      np.nan), errors="coerce"),
    "width_sword":    pd.to_numeric(matched.get("width",   np.nan), errors="coerce"),
    "width_lin":      pd.to_numeric(matched.get("width_m", np.nan), errors="coerce"),
    "dist_sword_lin_km": pd.to_numeric(matched["dist_m"],  errors="coerce") / 1000,
    "comid":          matched.get("COMID", pd.Series(dtype=object)),
})

# ── distance filter ───────────────────────────────────────────────────────────
# Matches beyond max_match_dist_km are unreliable (≈ the borderline case of two
# neighbouring GloFAS cells at opposing boundaries).  These pairs are excluded
# from all scatter plots and from the Lin ∩ GloFAS analysis below.

max_match_km = float(snakemake.params.max_match_dist_km)
within_range = df["dist_sword_lin_km"] <= max_match_km
n_excluded   = int((~within_range).sum())
df_valid     = df[within_range].copy()
log.info(
    f"Distance filter (≤ {max_match_km} km): "
    f"{len(df_valid)} pairs kept, {n_excluded} excluded"
)

# ── Lin ∩ GloFAS — GloFAS Q2 at Lin-reach positions (valid pairs only) ────────

# Reuse the GloFAS subset already clipped for this basin/domain by rule
# get_boundary_forcings (04) — recomputing it here duplicated a ~120 s load for
# the exact same bounds (river_discharge, wgs84_bounds, glofas_buffer_deg all
# match), so we read the cached netCDF instead.
log.info("Loading cached GloFAS clip …")
glofas_clip = xr.load_dataset(snakemake.input.glofas_clip)
lat_dim  = "latitude"   if "latitude"   in glofas_clip.dims else "lat"
lon_dim  = "longitude"  if "longitude"  in glofas_clip.dims else "lon"
time_dim = "valid_time" if "valid_time" in glofas_clip.dims else "time"
lat_vals = glofas_clip[lat_dim].values
lon_vals = glofas_clip[lon_dim].values
times    = glofas_clip[time_dim].values
log.info(f"GloFAS clip: {dict(glofas_clip.sizes)}")

glofas_search_km = float(snakemake.params.glofas_search_radius_km)

cell_cache: dict[tuple, float] = {}
q_lin_glofas: list[float] = []

for _, row in df_valid.iterrows():
    comid  = row["comid"]
    pt_wgs = lin_centroid_wgs.get(comid)
    if pt_wgs is None or not np.isfinite(row["q_lin"]):
        q_lin_glofas.append(np.nan)
        continue

    pt_lon, pt_lat = pt_wgs.x, pt_wgs.y
    if not (lat_vals.min() <= pt_lat <= lat_vals.max()
            and lon_vals.min() <= pt_lon <= lon_vals.max()):
        q_lin_glofas.append(np.nan)
        continue

    # Candidate cells within the search radius — pick the one with the highest
    # Q50 (median daily discharge), which identifies the most flow-active cell.
    lat_radius_deg = glofas_search_km / 111.0
    lon_radius_deg = glofas_search_km / (111.0 * max(np.cos(np.radians(abs(pt_lat))), 0.01))

    lat_cands = np.where(np.abs(lat_vals - pt_lat) <= lat_radius_deg)[0]
    lon_cands = np.where(np.abs(lon_vals - pt_lon) <= lon_radius_deg)[0]

    best_q50   = -np.inf
    best_i_lat = int(np.argmin(np.abs(lat_vals - pt_lat)))  # nearest fallback
    best_i_lon = int(np.argmin(np.abs(lon_vals - pt_lon)))

    for il in lat_cands:
        for il2 in lon_cands:
            # Exact metric distance to stay within the circle (not just the bbox)
            dlat_km = float(np.abs(lat_vals[il]  - pt_lat)) * 111.0
            dlon_km = float(np.abs(lon_vals[il2] - pt_lon)) * 111.0 * np.cos(np.radians(abs(pt_lat)))
            if np.sqrt(dlat_km**2 + dlon_km**2) > glofas_search_km:
                continue
            ts = glofas_clip[glofas_var].isel(
                {lat_dim: int(il), lon_dim: int(il2)}
            ).values.astype(float)
            ts_valid = ts[np.isfinite(ts)]
            if len(ts_valid) == 0:
                continue
            q50 = float(np.percentile(ts_valid, 50))
            if q50 > best_q50:
                best_q50   = q50
                best_i_lat = int(il)
                best_i_lon = int(il2)

    cell_key = (best_i_lat, best_i_lon)
    if cell_key not in cell_cache:
        series = glofas_clip[glofas_var].isel(
            {lat_dim: best_i_lat, lon_dim: best_i_lon}
        ).values.astype(float)
        cell_cache[cell_key] = analyse_cell_gev_only(
            times, series, eva_cfg,
            label=f"cell({lat_vals[best_i_lat]:.3f},{lon_vals[best_i_lon]:.3f})",
        )
    q_lin_glofas.append(cell_cache[cell_key])

df_valid["q_lin_glofas"] = q_lin_glofas
log.info(
    f"Lin ∩ GloFAS Q2 computed for {np.isfinite(q_lin_glofas).sum()}/{len(q_lin_glofas)} "
    f"valid positions ({len(cell_cache)} unique GloFAS cells)"
)

# ── figure (2 × 3; bottom-right cell empty) ──────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
excl_note = (
    f"  —  {n_excluded} seed point(s) excluded (SWORD↔Lin offset > {max_match_km} km)"
    if n_excluded > 0 else ""
)
fig.suptitle(
    f"Bankfull discharge validation — SWORD ∩ GloFAS and Lin ∩ GloFAS vs Lin et al."
    f"{excl_note}",
    fontsize=11, y=1.01,
)

# Row 1 — SWORD ∩ GloFAS vs Lin (only pairs within distance threshold) ─────────

_scatter(axes[0, 0],
         df_valid["q_lin"], df_valid["q_sword_glofas"],
         "Q₂ Lin (m³ s⁻¹)", "Q₂ SWORD ∩ GloFAS (m³ s⁻¹)",
         "SWORD ∩ GloFAS vs Lin",
         log_scale=True, unity_line=True)

df_valid["dq_sword_lin"] = df_valid["q_sword_glofas"] - df_valid["q_lin"]
_scatter(axes[0, 1],
         df_valid["dist_sword_lin_km"], df_valid["dq_sword_lin"],
         "Offset SWORD ∩ GloFAS ↔ Lin reach (km)",
         "ΔQ = Q_SWORD∩GloFAS − Q_Lin (m³ s⁻¹)",
         "SWORD ∩ GloFAS discharge residual vs spatial offset")

_scatter(axes[0, 2],
         df_valid["width_lin"], df_valid["width_sword"],
         "Width Lin (m)", "Width SWORD (m)",
         "Channel width — SWORD vs Lin",
         log_scale=True, unity_line=True)

# Row 2 — Lin ∩ GloFAS vs Lin ─────────────────────────────────────────────────

_scatter(axes[1, 0],
         df_valid["q_lin"], df_valid["q_lin_glofas"],
         "Q₂ Lin (m³ s⁻¹)", "Q₂ Lin ∩ GloFAS (m³ s⁻¹)",
         "Lin ∩ GloFAS vs Lin",
         log_scale=True, unity_line=True)

_scatter(axes[1, 1],
         df_valid["q_lin_glofas"], df_valid["q_sword_glofas"],
         "Q₂ Lin ∩ GloFAS (m³ s⁻¹)", "Q₂ SWORD ∩ GloFAS (m³ s⁻¹)",
         "SWORD ∩ GloFAS vs Lin ∩ GloFAS\n(same GloFAS input, different reach matching)",
         log_scale=True, unity_line=True)

# (2,3): network map — SWORD ∩ GloFAS, Lin (filtered), seed reaches ────────────
ax_map = axes[1, 2]

rivers_wgs84 = (
    rivers.to_crs("EPSG:4326")
    if rivers.crs is not None and rivers.crs.to_epsg() != 4326
    else rivers
)
seed_wgs84 = (
    seed_reaches.to_crs("EPSG:4326")
    if seed_reaches.crs is not None and seed_reaches.crs.to_epsg() != 4326
    else seed_reaches
)

# Identify excluded seed reach IDs (offset > threshold)
excluded_reach_ids: set[str] = set()
if n_excluded > 0 and "reach_id" in matched.columns:
    excl_mask = matched.index[~within_range]
    excluded_reach_ids = set(
        matched.loc[excl_mask, "reach_id"].astype(str).dropna()
    ) if "reach_id" in matched.columns else set()

if not rivers_wgs84.empty:
    rivers_wgs84.plot(ax=ax_map, color="steelblue", linewidth=0.4, alpha=0.6, zorder=2,
                      label=f"SWORD ∩ GloFAS ({len(rivers_wgs84)} reaches)")

if not lin.empty:
    lin.plot(ax=ax_map, color="darkorange", linewidth=0.5, alpha=0.7, zorder=3,
             label=f"Lin  order > {min_order}  ({len(lin)} reaches)")

bx, by = domain_poly.exterior.xy
ax_map.plot(bx, by, color="black", linewidth=1.5, zorder=1, label="Domain bbox")

if not seed_wgs84.empty:
    seed_centroids = seed_wgs84.copy()
    # Compute centroids in the projected CRS, then reproject back to WGS84 for plotting
    seed_centroids["geometry"] = seed_centroids.geometry.to_crs(domain_crs).centroid.to_crs("EPSG:4326")
    n_valid_seeds = len(seed_centroids) - n_excluded
    seed_centroids.plot(ax=ax_map, color="green", markersize=10, marker="^",
                        zorder=5, label=f"Seed ≤ {max_match_km} km  ({n_valid_seeds})")
    if n_excluded > 0:
        excl_seeds = seed_centroids[
            seed_centroids.get("reach_id", pd.Series(dtype=str)).astype(str).isin(excluded_reach_ids)
        ] if "reach_id" in seed_centroids.columns else gpd.GeoDataFrame()
        if not excl_seeds.empty:
            excl_seeds.plot(ax=ax_map, color="grey", markersize=10, marker="^",
                            zorder=5, label=f"Seed > {max_match_km} km  ({n_excluded}, excluded)")

lon_min_p, lat_min_p, lon_max_p, lat_max_p = domain_poly.bounds
margin = max(lon_max_p - lon_min_p, lat_max_p - lat_min_p) * 0.15
ax_map.set_xlim(lon_min_p - margin, lon_max_p + margin)
ax_map.set_ylim(lat_min_p - margin, lat_max_p + margin)
ax_map.set_aspect("equal")
ax_map.set_xlabel("Longitude (°)")
ax_map.set_ylabel("Latitude (°)")
ax_map.set_title(f"Network comparison — SWORD ∩ GloFAS vs Lin\n"
                 f"(▲ green = used, ▲ grey = excluded offset > {max_match_km} km)")
ax_map.legend(loc="best", fontsize=7, framealpha=0.9)
ax_map.grid(True, alpha=0.3, linewidth=0.5)

# Row labels ──────────────────────────────────────────────────────────────────

axes[0, 0].set_ylabel("SWORD ∩ GloFAS\n" + axes[0, 0].get_ylabel())
axes[1, 0].set_ylabel("Lin ∩ GloFAS\n"   + axes[1, 0].get_ylabel())

fig.tight_layout()
Path(snakemake.output.plot_discharge_comparison).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(snakemake.output.plot_discharge_comparison, dpi=120, bbox_inches="tight")
plt.close(fig)
profiler.stop()
log.info(f"Written: {snakemake.output.plot_discharge_comparison}")
