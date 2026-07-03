"""
07_test_upstream_boundary.py — Check that river boundary forcing locations are
far enough upstream from river mouths to avoid surge-wave interaction.

Wave propagation distance (in metres):
    d = T_eff × (c − v)
where:
    T_eff = effective_period_fraction × surge_period_hr × 3600  [s]
    c     = sqrt(g × (depth_mouth + A_surge))                    [m s⁻¹]
    v     = max(v_min, Q_bf / (W × D))                           [m s⁻¹]
    g     = 9.81 m s⁻²
    D     = max(rivdph, min_depth_at_mouth_m)                     [m]
    A_surge = max RP storm-tide level across all CoastRP stations  [m]

When c − v ≤ 0 (river faster than wave), the mouth is flagged as
"undetermined/blocked" — a zero-radius marker is drawn but no circle.

River mouths: reaches in river_network_processed.gpkg whose rch_id_dn
targets are absent from the network (outlets / domain-exits).

Active forcing points: has_glofas=1 crossings in river_forcing.nc.

Check: straight-line geodesic distance. A forcing point inside any circle
is "too close" (red); outside all circles is "OK" (green).
"""

import math
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pyproj import Geod
from shapely.geometry import box as shapely_box, Polygon as ShapelyPolygon

from src.domain import load_domain
from src.log import setup_logging
from src.plots import map_background
from src.profiling import ScriptProfiler

log = setup_logging(snakemake.log[0])
profiler = ScriptProfiler(snakemake)

# ── constants ─────────────────────────────────────────────────────────────────
G    = 9.81  # m s⁻²
GEOD = Geod(ellps="WGS84")
_CIRCLE_COLOR  = "#D94040"   # red for wave-propagation zone
_BLOCKED_COLOR = "#E07A10"   # orange for blocked/undetermined mouth
_OK_COLOR      = "#2ECC40"   # green for forcing point outside all circles
_BAD_COLOR     = "#D94040"   # red for forcing point inside a circle

# ── params ────────────────────────────────────────────────────────────────────
eff_frac    = float(snakemake.params.effective_period_fraction)
min_depth_m = float(snakemake.params.min_depth_at_mouth_m)
min_v_ms    = float(snakemake.params.min_river_velocity_ms)
period_hr   = float(snakemake.params.surge_period_hr)
T_eff_s     = eff_frac * period_hr * 3600.0
log.info(
    f"Parameters:"
    f"\n  effective_period_fraction = {eff_frac}"
    f"\n  surge_period_hr           = {period_hr} h"
    f"\n  T_eff = {eff_frac} × {period_hr} h × 3600 s/h = {T_eff_s:.0f} s  ({T_eff_s/3600:.2f} h)"
    f"\n  min_depth_at_mouth_m      = {min_depth_m} m"
    f"\n  min_river_velocity_ms     = {min_v_ms} m/s"
    f"\n  g                         = {G} m/s²"
)

# ── domain ────────────────────────────────────────────────────────────────────
wgs84_bounds, domain_crs, domain_poly = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)
log.info(f"Domain: {wgs84_bounds}, CRS: {domain_crs}")

# ── surge amplitude ───────────────────────────────────────────────────────────
with xr.open_dataset(snakemake.input.surge_forcing, decode_times=False) as surge_ds:
    rp_levels = surge_ds["rp_level"].values.astype(float)
    surge_amplitude = float(rp_levels.max())
log.info(
    f"Surge amplitude (rp_level across {len(rp_levels)} selected station(s)):"
    f"\n  min={rp_levels.min():.3f} m, mean={rp_levels.mean():.3f} m, "
    f"max={rp_levels.max():.3f} m"
    f"\n  Using max = {surge_amplitude:.3f} m for celerity calculation"
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _norm_id(x) -> str | None:
    s = str(x).strip()
    if s.lower() in ("nan", "none", "<na>", ""):
        return None
    try:
        return str(int(float(s)))
    except (ValueError, OverflowError):
        return s or None


def _parse_dn(raw) -> list[str]:
    s = str(raw).strip().strip("[]")
    if not s or s.lower() in ("nan", "none", "<na>"):
        return []
    return [nid for t in s.split(",") if (nid := _norm_id(t.strip()))]


def _geodesic_circle(lon: float, lat: float, radius_m: float, n_pts: int = 72) -> ShapelyPolygon:
    """Return a Shapely Polygon approximating a geodesic circle."""
    azimuths = np.linspace(0.0, 360.0, n_pts, endpoint=False)
    lons_c, lats_c, _ = GEOD.fwd(
        np.full(n_pts, lon), np.full(n_pts, lat),
        azimuths, np.full(n_pts, radius_m),
    )
    return ShapelyPolygon(zip(lons_c, lats_c))


def _placeholder(output_path: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, message, ha="center", va="center",
            transform=ax.transAxes, fontsize=12, color="grey")
    ax.set_axis_off()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── river network ─────────────────────────────────────────────────────────────
rivers = gpd.read_file(snakemake.input.processed_river_network)
if rivers.crs is not None and rivers.crs.to_epsg() != 4326:
    rivers = rivers.to_crs("EPSG:4326")

if rivers.empty:
    _placeholder(
        snakemake.output.plot_upstream_check,
        "Empty processed river network — upstream boundary check skipped",
    )
    profiler.stop()
    raise SystemExit(0)

all_ids: set[str] = {
    nid for rid in rivers["reach_id"].dropna()
    if (nid := _norm_id(rid))
}
log.info(f"River network: {len(rivers)} reaches, {len(all_ids)} unique IDs")

# ── identify river mouths ─────────────────────────────────────────────────────
mouth_rows = [
    row for _, row in rivers.iterrows()
    if not any(_norm_id(d) in all_ids for d in _parse_dn(row.get("rch_id_dn", "")))
]

if not mouth_rows:
    _placeholder(
        snakemake.output.plot_upstream_check,
        "No river mouths found — all reaches have downstream neighbours in network",
    )
    profiler.stop()
    raise SystemExit(0)

log.info(f"River mouths identified: {len(mouth_rows)}")

# ── per-mouth wave propagation ────────────────────────────────────────────────
MouthData = dict  # typed alias for readability
mouths_data: list[MouthData] = []

for i, row in enumerate(mouth_rows):
    reach_id_str = _norm_id(row.get("reach_id")) or "?"
    centroid = row.geometry.interpolate(0.5, normalized=True)

    # ── depth ──
    depth_raw = float(row.get("rivdph", np.nan) or np.nan)
    if np.isfinite(depth_raw):
        depth = max(depth_raw, min_depth_m)
        depth_note = (
            f"{depth_raw:.2f} m (clamped up to min_depth = {depth:.2f} m)"
            if depth_raw < min_depth_m
            else f"{depth_raw:.2f} m"
        )
    else:
        depth = min_depth_m
        depth_note = f"NaN → using min_depth = {depth:.2f} m"

    # ── flow velocity ──
    width = float(row.get("width", np.nan) or np.nan)
    q_bf  = float(row.get("bankfull_discharge_acc", np.nan) or np.nan)

    if np.isfinite(width) and width > 0 and np.isfinite(q_bf) and q_bf >= 0:
        v_real = q_bf / (width * depth)
        v = max(min_v_ms, v_real)
        v_note = (
            f"Q/(W×D) = {q_bf:.1f}/({width:.0f}×{depth:.2f}) = {v_real:.3f} m/s"
            f" → clamped to min = {v:.2f} m/s"
            if v_real < min_v_ms
            else f"Q/(W×D) = {q_bf:.1f}/({width:.0f}×{depth:.2f}) = {v:.3f} m/s"
        )
    else:
        v = min_v_ms
        v_note = (
            f"width={'NaN' if not np.isfinite(width) else f'{width:.0f} m'}, "
            f"Q_bf={'NaN' if not np.isfinite(q_bf) else f'{q_bf:.1f} m³/s'}"
            f" → using min_velocity = {v:.2f} m/s"
        )

    # ── celerity & wave propagation ──
    celerity  = math.sqrt(G * (depth + surge_amplitude))
    net_speed = celerity - v
    blocked   = net_speed <= 0.0
    radius_m  = 0.0 if blocked else T_eff_s * net_speed

    log.info(
        f"Mouth {i+1}/{len(mouth_rows)} — reach_id={reach_id_str}"
        f" @ ({centroid.y:.4f}°N, {centroid.x:.4f}°E)"
        f"\n  depth:     {depth_note}"
        f"\n  velocity:  {v_note}"
        f"\n  celerity:  c = √(g × (d + A)) = √({G} × ({depth:.2f} + {surge_amplitude:.3f}))"
        f" = √{G * (depth + surge_amplitude):.3f} = {celerity:.3f} m/s"
        f"\n  net speed: c − v = {celerity:.3f} − {v:.3f} = {net_speed:.3f} m/s"
        + (
            f"\n  radius:    T_eff × net = {T_eff_s:.0f} s × {net_speed:.3f} m/s"
            f" = {radius_m:.0f} m  ({radius_m/1000:.2f} km)"
            if not blocked
            else f"\n  [WAVE BLOCKED — river velocity ({v:.3f} m/s) ≥ celerity ({celerity:.3f} m/s)]"
        )
    )

    mouths_data.append({
        "lon": centroid.x, "lat": centroid.y,
        "radius_m": radius_m, "blocked": blocked,
        "celerity": celerity, "v": v, "depth": depth,
        "reach_id": reach_id_str,
    })

# ── active forcing points ─────────────────────────────────────────────────────
with xr.open_dataset(snakemake.input.river_forcing, decode_times=False) as rf_ds:
    has_glofas   = rf_ds["has_glofas"].values.astype(bool)
    forcing_lons = rf_ds["longitude"].values[has_glofas]
    forcing_lats = rf_ds["latitude"].values[has_glofas]

n_forcing = int(has_glofas.sum())
log.info(f"Active forcing points: {n_forcing}")

# ── inside/outside check ──────────────────────────────────────────────────────
forcing_inside = np.zeros(n_forcing, dtype=bool)
for k in range(n_forcing):
    flon, flat = forcing_lons[k], forcing_lats[k]
    # Compute distance to every non-blocked mouth; find the closest
    dists = []
    for mp in mouths_data:
        if mp["radius_m"] <= 0:
            continue
        _, _, dist_m = GEOD.inv(flon, flat, mp["lon"], mp["lat"])
        dists.append((dist_m, mp))

    if dists:
        nearest_dist, nearest_mp = min(dists, key=lambda x: x[0])
        inside = nearest_dist <= nearest_mp["radius_m"]
        forcing_inside[k] = inside
        log.info(
            f"  Forcing point {k+1}/{n_forcing} @ ({flat:.4f}°N, {flon:.4f}°E):"
            f"\n    nearest mouth: reach_id={nearest_mp['reach_id']}"
            f" @ ({nearest_mp['lat']:.4f}°N, {nearest_mp['lon']:.4f}°E)"
            f"\n    distance to mouth: {nearest_dist/1000:.2f} km"
            f"  |  wave propagation radius: {nearest_mp['radius_m']/1000:.2f} km"
            f"\n    → {'INSIDE — too close [WARNING]' if inside else 'OUTSIDE — OK'}"
        )
    else:
        log.info(
            f"  Forcing point {k+1}/{n_forcing} @ ({flat:.4f}°N, {flon:.4f}°E):"
            f" no non-blocked mouths to check against → OK"
        )

n_inside  = int(forcing_inside.sum())
n_outside = n_forcing - n_inside
log.info(
    f"Summary: {n_forcing} active forcing point(s) — "
    f"{n_outside} OK (outside all wave zones), "
    f"{n_inside} WARNING (inside at least one wave zone)"
)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
map_background(ax, shapely_box(*wgs84_bounds), snakemake.input.land_polygons)

# River network
rivers.plot(ax=ax, color="steelblue", linewidth=0.5, alpha=0.55, zorder=2,
            label=f"River network ({len(rivers)} reaches)")

# Wave-propagation circles and mouth markers
n_blocked = sum(1 for mp in mouths_data if mp["blocked"])
n_circles = len(mouths_data) - n_blocked
for mp in mouths_data:
    lon, lat = mp["lon"], mp["lat"]
    if mp["blocked"]:
        ax.plot(lon, lat, "x", color=_BLOCKED_COLOR, markersize=11,
                markeredgewidth=2.5, zorder=7)
    else:
        circle = _geodesic_circle(lon, lat, mp["radius_m"])
        x, y = circle.exterior.xy
        ax.fill(x, y, color=_CIRCLE_COLOR, alpha=0.12, zorder=3)
        ax.plot(x, y, color=_CIRCLE_COLOR, linewidth=0.9, alpha=0.6, zorder=4)
        ax.plot(lon, lat, "^", color=_CIRCLE_COLOR, markersize=7,
                markeredgecolor="white", markeredgewidth=0.6, zorder=7)

# Forcing points (outside first so red ones render on top)
if n_forcing > 0:
    for k in range(n_forcing):
        color = _BAD_COLOR if forcing_inside[k] else _OK_COLOR
        ax.plot(forcing_lons[k], forcing_lats[k], "o", color=color,
                markersize=9, markeredgecolor="white", markeredgewidth=0.8,
                zorder=8)

# Legend
max_radius_km = max(
    (mp["radius_m"] / 1000 for mp in mouths_data if not mp["blocked"]),
    default=0.0,
)
legend_handles = [
    plt.Line2D([0], [0], color="steelblue", linewidth=1.5, alpha=0.7,
               label=f"River network ({len(rivers)} reaches)"),
]
if n_circles:
    legend_handles.append(
        mpatches.Patch(
            color=_CIRCLE_COLOR, alpha=0.35,
            label=(
                f"Wave propagation zone ({n_circles} mouth(s))\n"
                f"T_eff = {T_eff_s/3600:.1f} h, "
                f"A_surge = {surge_amplitude:.2f} m, "
                f"max radius = {max_radius_km:.0f} km"
            ),
        )
    )
if n_blocked:
    legend_handles.append(
        plt.Line2D([0], [0], marker="x", color=_BLOCKED_COLOR, linestyle="none",
                   markersize=10, markeredgewidth=2.5,
                   label=f"Mouth — wave blocked (v > c)  ({n_blocked})")
    )
if n_outside:
    legend_handles.append(
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=_OK_COLOR,
                   markersize=9, markeredgecolor="grey", markeredgewidth=0.5,
                   label=f"Forcing point — OK  ({n_outside} outside)")
    )
if n_inside:
    legend_handles.append(
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=_BAD_COLOR,
                   markersize=9, markeredgecolor="grey", markeredgewidth=0.5,
                   label=f"Forcing point — too close  ({n_inside} inside)")
    )

ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.92)
ax.set_title(
    f"Upstream boundary check — wave propagation distance\n"
    f"d = T_eff × (c − v),  "
    f"T_eff = {eff_frac} × {period_hr} h = {T_eff_s/3600:.1f} h,  "
    f"min depth = {min_depth_m} m,  min v = {min_v_ms} m/s",
    fontsize=9,
)
ax.set_xlabel("Longitude (°)")
ax.set_ylabel("Latitude (°)")
ax.grid(True, alpha=0.25, linewidth=0.5)

Path(snakemake.output.plot_upstream_check).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(snakemake.output.plot_upstream_check, dpi=130, bbox_inches="tight")
plt.close(fig)
profiler.stop()
log.info(f"Written: {snakemake.output.plot_upstream_check}")
