"""
12_test_upstream_boundary.py — Check that river boundary forcing locations are
far enough upstream from river mouths to avoid surge-wave interaction.

Final wave-propagation distance = min(kinematic distance, attenuation distance):

  kinematic distance (time-bounded front position):
    d_kin = T_eff × (c − v)
    T_eff = effective_period_fraction × surge_period_hr × 3600  [s]
    c     = sqrt(g × (depth_mouth + A0))                         [m s⁻¹]
    v     = Q_bf / (W × D)                                       [m s⁻¹]
    depth_mouth = rivdph, used as calculated -- no config floor. A mouth
    with no usable calculated depth or velocity is skipped entirely.

  attenuation distance (friction-damped amplitude decay, marching upstream
  reach-by-reach along the mainstem -- the largest bankfull_discharge_acc
  branch at each confluence):
    A(x) = A0 · exp(−μx),   μ = r / (2c),   r = (8/3π) · g · n² · U0 / h^(4/3)
    U0 = c0 · A0 / depth_mouth, computed ONCE at the mouth and held fixed
    for the whole march (the standard linearised-friction assumption -- r
    approximately constant per reach, based on a representative tidal/surge
    velocity). Re-deriving U from the locally-decaying amplitude at every
    step instead turns the equation into dA/dx = -k·A², a hyperbolic decay
    that (confirmed by direct comparison) fails to converge at all for
    low-amplitude mouths -- this is why U is frozen at the mouth, not
    recomputed per reach.
    Stops when A decays below amplitude_threshold_fraction of A0, or the
    upstream network runs out.

Both distances need A0, the surge amplitude driving that specific mouth --
each mouth uses its OWN NEAREST CoastRP station's rp_level, not the
domain-wide max (stations can vary 5-10x across one basin's coastline).

When c ≤ v (river faster than wave), the mouth is flagged as
"undetermined/blocked" -- a zero-radius marker is drawn but no circle.

River mouths: reaches in river_network_estuarine.gpkg whose rch_id_dn
targets are absent from the network (outlets / domain-exits).

Active forcing points: has_glofas=1 crossings in river_forcing.nc.

Check: straight-line geodesic distance from each forcing point to the
nearest mouth, compared against that mouth's final distance (circle). A
forcing point inside any circle is "too close" (red); outside all circles
is "OK" (green). Reaches actually reached by a mouth's attenuation march are
additionally colored by remaining amplitude (max across mouths, where more
than one march reaches the same reach) -- a more literal picture of how far
the wave itself gets, distinct from the circle's straight-line abstraction.
The plot title reports the tightest margin (distance-from-mouth minus
final-distance) found across all forcing-point/mouth pairs.
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
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pyproj import Geod
from shapely.geometry import box as shapely_box, Polygon as ShapelyPolygon

from src.domain import load_domain
from src.log import setup_logging
from src.plots import map_background
from src.profiling import ScriptProfiler
from src.river_network import _as_linestring, build_downstream_adjacency, normalize_reach_id

log = setup_logging(snakemake.log[0])
profiler = ScriptProfiler(snakemake)

# ── constants ─────────────────────────────────────────────────────────────────
G    = 9.81  # m s⁻²
GEOD = Geod(ellps="WGS84")
_CIRCLE_COLOR  = "#D94040"   # red for wave-propagation zone
_BLOCKED_COLOR = "#E07A10"   # orange for blocked/undetermined mouth
_OK_COLOR      = "#2ECC40"   # green for forcing point outside all circles
_BAD_COLOR     = "#D94040"   # red for forcing point inside a circle
_AMPLITUDE_CMAP = "YlOrRd"

# ── params ────────────────────────────────────────────────────────────────────
eff_frac      = float(snakemake.params.effective_period_fraction)
period_hr     = float(snakemake.params.surge_period_hr)
channel_n     = float(snakemake.params.channel_manning_n)
amp_thresh_frac = float(snakemake.params.amplitude_threshold_fraction)
T_eff_s       = eff_frac * period_hr * 3600.0
log.info(
    f"Parameters:"
    f"\n  effective_period_fraction   = {eff_frac}"
    f"\n  surge_period_hr             = {period_hr} h"
    f"\n  T_eff = {eff_frac} × {period_hr} h × 3600 s/h = {T_eff_s:.0f} s  ({T_eff_s/3600:.2f} h)"
    f"\n  channel_manning_n           = {channel_n}"
    f"\n  amplitude_threshold_fraction = {amp_thresh_frac}"
    f"\n  g                           = {G} m/s²"
    f"\n  depth/velocity: calculated per mouth (rivdph; Q_bf/(W×D)), no config floor"
)

# ── domain ────────────────────────────────────────────────────────────────────
wgs84_bounds, domain_crs, domain_poly = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)
log.info(f"Domain: {wgs84_bounds}, CRS: {domain_crs}")

# ── surge stations (per-station amplitude, NOT a domain-wide max) ────────────
with xr.open_dataset(snakemake.input.surge_forcing, decode_times=False) as surge_ds:
    station_lons = surge_ds["longitude"].values.astype(float)
    station_lats = surge_ds["latitude"].values.astype(float)
    station_rp   = surge_ds["rp_level"].values.astype(float)
log.info(
    f"Surge stations: {len(station_rp)}, rp_level range "
    f"[{station_rp.min():.3f}, {station_rp.max():.3f}] m "
    f"(each mouth uses its own nearest station, not the domain max)"
)


def _nearest_station_rp(lon: float, lat: float) -> tuple[float, float]:
    """Return (rp_level, distance_m) of the CoastRP station nearest (lon, lat)."""
    _, _, dists = GEOD.inv(
        np.full(len(station_lons), lon), np.full(len(station_lats), lat),
        station_lons, station_lats,
    )
    i = int(np.argmin(dists))
    return float(station_rp[i]), float(dists[i])


# ── helpers ───────────────────────────────────────────────────────────────────

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
rivers = gpd.read_file(snakemake.input.river_network_estuarine)
if rivers.crs is not None and rivers.crs.to_epsg() != 4326:
    rivers = rivers.to_crs("EPSG:4326")

if rivers.empty:
    _placeholder(
        snakemake.output.plot_upstream_check,
        "Empty processed river network — upstream boundary check skipped",
    )
    profiler.stop()
    raise SystemExit(0)

rivers = rivers.copy()
rivers["_rid"] = rivers["reach_id"].apply(normalize_reach_id)
by_rid = {r["_rid"]: r for _, r in rivers.iterrows() if r["_rid"] is not None}
log.info(f"River network: {len(rivers)} reaches, {len(by_rid)} unique IDs")

# Downstream adjacency (shared helper -- same parsing as elsewhere in the
# pipeline, e.g. 10_condition_elevation.py) inverted to get upstream
# adjacency for the attenuation march.
downstream_adj = build_downstream_adjacency(rivers)
upstream_adj: dict[str, list[str]] = {rid: [] for rid in downstream_adj}
for rid, dns in downstream_adj.items():
    for dn in dns:
        upstream_adj.setdefault(dn, []).append(rid)

# Projected copy for accurate reach lengths (domain's own working UTM CRS --
# pixel-perfect with the rest of the pipeline, not just an estimate).
rivers_proj = rivers.to_crs(domain_crs)
by_rid_proj = {r["_rid"]: r for _, r in rivers_proj.iterrows() if r["_rid"] is not None}


def _reach_length_m(rid: str) -> float:
    g = _as_linestring(by_rid_proj[rid].geometry)
    return g.length if g is not None else 0.0


# ── identify river mouths ─────────────────────────────────────────────────────
mouth_ids = [rid for rid, dns in downstream_adj.items() if not dns]

if not mouth_ids:
    _placeholder(
        snakemake.output.plot_upstream_check,
        "No river mouths found — all reaches have downstream neighbours in network",
    )
    profiler.stop()
    raise SystemExit(0)

log.info(f"River mouths identified: {len(mouth_ids)}")


def _mainstem_upstream(rid: str) -> str | None:
    """Largest bankfull_discharge_acc among rid's upstream neighbours, or None."""
    ups = upstream_adj.get(rid, [])
    if not ups:
        return None
    def _q(u):
        qv = by_rid[u].get("bankfull_discharge_acc", np.nan)
        return float(qv) if qv is not None and np.isfinite(float(qv)) else -1.0
    return max(ups, key=_q)


# ── per-mouth wave propagation ────────────────────────────────────────────────
MouthData = dict  # typed alias for readability
mouths_data: list[MouthData] = []
reach_amplitude: dict[str, float] = {}  # reach_id -> max remaining amplitude (m) across all mouths' marches

n_skipped_no_depth = 0
n_skipped_no_velocity = 0

for i, mrid in enumerate(mouth_ids):
    row = by_rid[mrid]
    reach_id_str = mrid
    centroid = row.geometry.interpolate(0.5, normalized=True)

    # ── depth (calculated only -- no config floor) ──
    depth = float(row.get("rivdph", np.nan) or np.nan)
    if not np.isfinite(depth):
        log.warning(
            f"Mouth {i+1}/{len(mouth_ids)} — reach_id={reach_id_str}: "
            f"no calculated river depth (rivdph is NaN) — skipping"
        )
        n_skipped_no_depth += 1
        continue

    # ── flow velocity (calculated only -- no config floor) ──
    width = float(row.get("width", np.nan) or np.nan)
    q_bf  = float(row.get("bankfull_discharge_acc", np.nan) or np.nan)
    if not (np.isfinite(width) and width > 0 and np.isfinite(q_bf) and q_bf >= 0):
        log.warning(
            f"Mouth {i+1}/{len(mouth_ids)} — reach_id={reach_id_str}: "
            f"no calculated flow velocity "
            f"(width={'NaN' if not np.isfinite(width) else f'{width:.0f} m'}, "
            f"Q_bf={'NaN' if not np.isfinite(q_bf) else f'{q_bf:.1f} m³/s'}) — skipping"
        )
        n_skipped_no_velocity += 1
        continue
    v = q_bf / (width * depth)

    # ── local surge amplitude (nearest station, not the domain max) ──
    A0, station_dist_m = _nearest_station_rp(centroid.x, centroid.y)

    # ── kinematic distance ──
    c0  = math.sqrt(G * (depth + A0))
    net = c0 - v
    if net <= 0:
        log.info(
            f"Mouth {i+1}/{len(mouth_ids)} — reach_id={reach_id_str}: "
            f"blocked (v={v:.3f} m/s ≥ c={c0:.3f} m/s) — undetermined"
        )
        mouths_data.append({
            "lon": centroid.x, "lat": centroid.y, "radius_m": 0.0, "blocked": True,
            "celerity": c0, "v": v, "depth": depth, "reach_id": reach_id_str, "A0": A0,
        })
        continue
    kinematic_dist = T_eff_s * net

    # ── attenuation distance (friction-damped amplitude decay) ──
    # U0: reference tidal/surge current-velocity scale, fixed ONCE at the
    # mouth and held constant for the whole march -- see module docstring
    # for why this must NOT be re-derived from the locally-decaying A.
    U0 = c0 * A0 / depth
    A = A0
    cum_dist = 0.0
    cur = mrid
    visited = {cur}
    reach_amplitude[cur] = max(reach_amplitude.get(cur, 0.0), A0)
    steps = 0
    stop_reason = "amplitude threshold reached"
    while A > amp_thresh_frac * A0:
        nxt = _mainstem_upstream(cur)
        if nxt is None or nxt in visited:
            stop_reason = "ran out of upstream network"
            break
        h = float(by_rid[nxt].get("rivdph", np.nan) or np.nan)
        if not np.isfinite(h) or h <= 0:
            stop_reason = "invalid depth on next reach"
            break
        L = _reach_length_m(nxt)
        c_i = math.sqrt(G * (h + A))  # local celerity -- genuinely varies with local A
        r_i = (8.0 / (3 * math.pi)) * G * channel_n**2 * U0 / h**(4.0 / 3.0)  # friction uses FIXED U0
        mu_i = r_i / (2 * c_i)
        A = A * math.exp(-mu_i * L)
        cum_dist += L
        cur = nxt
        visited.add(cur)
        reach_amplitude[cur] = max(reach_amplitude.get(cur, 0.0), A)
        steps += 1
        if steps > 2000:
            stop_reason = "step cap (2000) hit"
            break
    attenuation_dist = cum_dist
    if stop_reason != "amplitude threshold reached":
        log.warning(
            f"Mouth {i+1}/{len(mouth_ids)} — reach_id={reach_id_str}: "
            f"attenuation march stopped early ({stop_reason}) after {steps} reach(es) "
            f"— {attenuation_dist/1000:.1f} km is a lower bound, not a converged distance"
        )

    radius_m = min(kinematic_dist, attenuation_dist)

    log.info(
        f"Mouth {i+1}/{len(mouth_ids)} — reach_id={reach_id_str}"
        f" @ ({centroid.y:.4f}°N, {centroid.x:.4f}°E)"
        f"\n  nearest station: {station_dist_m/1000:.1f} km away, rp_level={A0:.2f} m"
        f"\n  depth: {depth:.2f} m,  velocity: Q/(W×D) = {q_bf:.1f}/({width:.0f}×{depth:.2f}) = {v:.3f} m/s"
        f"\n  celerity: c = √(g×(d+A0)) = {c0:.3f} m/s,  net speed: c−v = {net:.3f} m/s"
        f"\n  kinematic distance   = {kinematic_dist/1000:.2f} km"
        f"\n  attenuation distance = {attenuation_dist/1000:.2f} km  ({steps} reach(es), {stop_reason})"
        f"\n  final distance = min(...) = {radius_m/1000:.2f} km"
    )

    mouths_data.append({
        "lon": centroid.x, "lat": centroid.y,
        "radius_m": radius_m, "blocked": False,
        "celerity": c0, "v": v, "depth": depth,
        "reach_id": reach_id_str, "A0": A0,
    })

log.info(
    f"Mouths assessed: {len(mouths_data)}/{len(mouth_ids)} "
    f"({n_skipped_no_depth} skipped: no rivdph, "
    f"{n_skipped_no_velocity} skipped: no width/Q_bf)"
)

if not mouths_data:
    _placeholder(
        snakemake.output.plot_upstream_check,
        "No mouth has both a calculated depth and flow velocity — "
        "upstream boundary check skipped",
    )
    profiler.stop()
    raise SystemExit(0)

# ── active forcing points ─────────────────────────────────────────────────────
with xr.open_dataset(snakemake.input.river_forcing, decode_times=False) as rf_ds:
    has_glofas   = rf_ds["has_glofas"].values.astype(bool)
    forcing_lons = rf_ds["longitude"].values[has_glofas]
    forcing_lats = rf_ds["latitude"].values[has_glofas]

n_forcing = int(has_glofas.sum())
log.info(f"Active forcing points: {n_forcing}")

# ── inside/outside check ──────────────────────────────────────────────────────
# Also tracks the tightest margin (distance-from-mouth minus final distance)
# across every forcing-point/mouth pair -- the single most critical
# comparison, reported in the plot's main title.
forcing_inside = np.zeros(n_forcing, dtype=bool)
tightest: dict | None = None
for k in range(n_forcing):
    flon, flat = forcing_lons[k], forcing_lats[k]
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
        margin = nearest_dist - nearest_mp["radius_m"]
        if tightest is None or margin < tightest["margin"]:
            tightest = {
                "margin": margin, "dist_m": nearest_dist,
                "radius_m": nearest_mp["radius_m"], "reach_id": nearest_mp["reach_id"],
            }
        log.info(
            f"  Forcing point {k+1}/{n_forcing} @ ({flat:.4f}°N, {flon:.4f}°E):"
            f"\n    nearest mouth: reach_id={nearest_mp['reach_id']}"
            f" @ ({nearest_mp['lat']:.4f}°N, {nearest_mp['lon']:.4f}°E)"
            f"\n    distance to mouth: {nearest_dist/1000:.2f} km"
            f"  |  final distance: {nearest_mp['radius_m']/1000:.2f} km"
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
map_background(ax, shapely_box(*wgs84_bounds), snakemake.input.land_polygons,
               water_bodies_path=snakemake.input.spec_landuse)

# River network background
rivers.plot(ax=ax, color="steelblue", linewidth=0.5, alpha=0.55, zorder=2,
            label=f"River network ({len(rivers)} reaches)")

# Reaches actually reached by a mouth's attenuation march, colored by
# remaining amplitude (max across mouths where more than one march reaches
# the same reach).
if reach_amplitude:
    amp_norm = Normalize(vmin=0.0, vmax=max(reach_amplitude.values()))
    cmap = matplotlib.colormaps[_AMPLITUDE_CMAP]
    segments, colors = [], []
    for rid, amp in reach_amplitude.items():
        g = _as_linestring(by_rid[rid].geometry)
        if g is None:
            continue
        segments.append(np.asarray(g.coords))
        colors.append(cmap(amp_norm(amp)))
    lc = LineCollection(segments, colors=colors, linewidths=2.2, zorder=5)
    ax.add_collection(lc)
    sm = ScalarMappable(norm=amp_norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Remaining surge amplitude along river (m)")

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
legend_handles = [
    plt.Line2D([0], [0], color="steelblue", linewidth=1.5, alpha=0.7,
               label=f"River network ({len(rivers)} reaches)"),
]
if n_circles:
    legend_handles.append(
        mpatches.Patch(
            color=_CIRCLE_COLOR, alpha=0.35,
            label=f"Wave propagation zone ({n_circles} mouth(s))",
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

# Main title reports the tightest margin found across all forcing-point/mouth
# pairs (the single most critical comparison); subtitle reports the
# parameters used to compute it -- min depth/min v are the smallest
# CALCULATED values found across the assessed mouths (not config floors),
# and A_surge is the amplitude at the tightest-margin mouth's own nearest
# station (not a domain-wide value, since amplitude is now per-mouth).
if tightest is not None:
    title_main = (
        f"Upstream boundary check: wave propagation distance "
        f"({tightest['radius_m']/1000:.1f} km) vs. forcing point "
        f"({tightest['dist_m']/1000:.1f} km)"
    )
    tightest_mp = next(mp for mp in mouths_data if mp["reach_id"] == tightest["reach_id"])
    A_report = tightest_mp["A0"]
else:
    title_main = "Upstream boundary check: no forcing point/mouth pair to compare"
    A_report = float(station_rp.max())
min_depth_calc = min(mp["depth"] for mp in mouths_data)
min_v_calc = min(mp["v"] for mp in mouths_data)
ax.set_title(
    f"{title_main}\n"
    f"d = min(T_eff × (c − v), attenuation),  T_eff = {T_eff_s/3600:.1f} h,  "
    f"min depth = {min_depth_calc:.2f} m,  min v = {min_v_calc:.2f} m/s,  "
    f"A_surge = {A_report:.2f} m",
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
