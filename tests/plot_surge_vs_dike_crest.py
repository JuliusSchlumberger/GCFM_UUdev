"""
plot_surge_vs_dike_crest.py — Debugging aid: "surprisingly little flooding at
a given surge RP" almost always traces back to one question first — does the
design surge level at that RP actually clear the coastal dike/weir crest
built into the model? This compares the two directly, basin-level, with NO
dependency on any particular scenario run having been executed (surge levels
come from surge_forcing.nc's own COAST-RP table; the weir crest comes from
whatever was actually built into the model, both basin-level and
scenario-independent).

Reads:
  - {basin_id}/preprocessing_inputs/forcing/surge_forcing.nc (falls back to
    the pre-reorg {basin_id}/inputs/forcing/surge_forcing.nc path) — the
    same storm_tide_rp_table + station_baseline COAST-RP lookup production
    uses (src.surge.lookup_storm_tide_at_rp), so this is exactly what a
    scenario's own surge_rp would resolve to, per station.
  - The basin's built coastal protection weir gpkg (elevation column, one
    row per weir segment) — tries the current sfincs_skeleton/ location
    first, then older pre-reorg locations, since local results directories
    can lag behind the current pipeline layout.
  - coastal_protection_crest_m from surge_forcing.nc — the FLAT, per-basin
    FLOPROS design crest BEFORE any freeboard/river-crest combination (see
    src.protection_weir.build_coastal_protection_weir's own docstring):
    pure-coastal weir segments equal this value exactly; segments also
    covered by a calibrated river reach can be pushed much higher via
    max(coastal, river_crest) -- so the built weir's own min matches this
    floor while its max can be dominated entirely by inland river reaches
    that have nothing to do with the coastal surge at all.

Usage:
    conda run -n hmt_sfincs_dev python tests/plot_surge_vs_dike_crest.py [basin_id] [scenario_surge_rp]
    conda run -n hmt_sfincs_dev python tests/plot_surge_vs_dike_crest.py 2433835 500
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))
from src.surge import lookup_storm_tide_at_rp

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
BASIN_ID = sys.argv[1] if len(sys.argv) > 1 else "2433835"
SCENARIO_SURGE_RP = float(sys.argv[2]) if len(sys.argv) > 2 else 500.0

with open(REPO_ROOT / "config" / "config.yml") as fh:
    config = yaml.safe_load(fh)
RESULTS_DIR = Path(config["results_dir"])
FIGS_DIR = REPO_ROOT / "figs" / "surge_vs_dike_crest"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

basin_dir = RESULTS_DIR / BASIN_ID


def _first_existing(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "None of the candidate paths exist:\n" + "\n".join(f"  {p}" for p in candidates)
    )


surge_forcing_path = _first_existing(
    [
        basin_dir / "preprocessing_inputs" / "forcing" / "surge_forcing.nc",
        basin_dir / "inputs" / "forcing" / "surge_forcing.nc",
    ]
)
log.info(f"surge_forcing.nc: {surge_forcing_path}")

weir_candidates = (
    [
        basin_dir / "sfincs_skeleton" / f"{BASIN_ID}_coastal_protection_weir.gpkg",
        basin_dir
        / "preprocessing_inputs"
        / "domain"
        / f"{BASIN_ID}_coastal_protection_weir.gpkg",
        basin_dir / "inputs" / "domain" / f"{BASIN_ID}_coastal_protection_weir.gpkg",
    ]
    + sorted(basin_dir.glob("runs/*/sfincs/*_coastal_protection_weir.gpkg"))
    + sorted(basin_dir.glob("scenarios/*/sfincs/*_coastal_protection_weir.gpkg"))
)
weir_path = _first_existing(weir_candidates)
log.info(f"coastal_protection_weir.gpkg: {weir_path}")

# ── surge: per-station storm tide at every COAST-RP tabulated RP ────────────
# surge_forcing.nc's own rp_level/baseline_m are MDT-only by design (SLR is
# stored as a dimensionless slr_fingerprint, not baked in -- see
# src.surge.apply_slr_fingerprint) so that this basin-level file never
# changes when only the slr_m target changes. This script isn't
# snakemake-driven, so it reads the CURRENT config.yml target itself and
# applies it the same way 13_build_sfincs.py does at real build time, to
# show the actual, currently-configured SLR-inclusive levels.
surge_ds = xr.open_dataset(surge_forcing_path, decode_times=False)
table_rps = surge_ds["table_rp"].values.astype(float)
n_stations = surge_ds.sizes["station"]
coastal_protection_crest_m = float(surge_ds["coastal_protection_crest_m"].values)
baseline_m = float(surge_ds["baseline_m"].values)

slr_cfg = config["boundary_forcings"]["surge"]["slr"]
effective_slr_m = float(slr_cfg["slr_m"]) if slr_cfg["enabled"] else 0.0

levels_by_rp = np.stack(
    [lookup_storm_tide_at_rp(surge_ds, rp, slr_m=effective_slr_m) for rp in table_rps]
)  # (n_rp, n_station)
rp_min = levels_by_rp.min(axis=1)
rp_mean = levels_by_rp.mean(axis=1)
rp_max = levels_by_rp.max(axis=1)

log.info(f"Stations: {n_stations}")
log.info(
    f"coastal_protection_crest_m (flat FLOPROS design crest, MDT-only): {coastal_protection_crest_m:+.4f} m"
)
log.info(f"baseline_m (mean sea level correction, MDT-only): {baseline_m:+.4f} m")
log.info(
    f"SLR target applied (config boundary_forcings.surge.slr): "
    f"{effective_slr_m:+.3f} m x per-station fingerprint "
    f"(enabled={slr_cfg['enabled']}) -- levels below already include this"
)
log.info("RP (yr)   min      mean     max      (storm tide level, m)")
for rp, lo, mu, hi in zip(table_rps, rp_min, rp_mean, rp_max):
    flag = (
        "  <-- exceeds coastal crest floor" if mu > coastal_protection_crest_m else ""
    )
    log.info(f"{rp:>7.0f}   {lo:+.3f}   {mu:+.3f}   {hi:+.3f}{flag}")

# ── built weir crest ──────────────────────────────────────────────────────
weir_gdf = gpd.read_file(weir_path)
weir_elev = weir_gdf["elevation"].to_numpy(dtype=float)
n_at_floor = int(np.isclose(weir_elev, coastal_protection_crest_m, atol=0.01).sum())
log.info(
    f"Built weir: {len(weir_elev)} segment(s), elevation "
    f"[{weir_elev.min():.3f}, {weir_elev.max():.3f}] m "
    f"({n_at_floor} segment(s) sit AT the flat coastal floor -- pure-coastal, "
    f"no river-crest influence; the rest are pushed higher by a calibrated "
    f"river reach via max(coastal, river_crest))"
)

# ── scenario RP under investigation ──────────────────────────────────────────
if not np.any(np.isclose(table_rps, SCENARIO_SURGE_RP)):
    raise ValueError(
        f"surge RP {SCENARIO_SURGE_RP} not tabulated in COAST-RP ({table_rps.tolist()})"
    )
scenario_idx = int(np.nonzero(np.isclose(table_rps, SCENARIO_SURGE_RP))[0][0])
scenario_levels = levels_by_rp[scenario_idx]
log.info(
    f"\nAt the scenario's own surge_rp={SCENARIO_SURGE_RP:.0f}: storm tide "
    f"[{scenario_levels.min():+.3f}, {scenario_levels.max():+.3f}] m "
    f"(mean {scenario_levels.mean():+.3f} m) vs. coastal crest floor "
    f"{coastal_protection_crest_m:+.4f} m -> margin "
    f"{scenario_levels.mean() - coastal_protection_crest_m:+.4f} m "
    f"({'OVERTOPS' if scenario_levels.mean() > coastal_protection_crest_m else 'below crest -- no coastal overtopping expected'})"
)

# ── plot ──────────────────────────────────────────────────────────────────────
# Two panels, DIFFERENT y-scales on purpose: the built weir's full elevation
# range is dominated by inland, river-crest-driven segments (up to
# {weir_elev.max()} m here) that have nothing to do with coastal surge at
# all -- sharing one axis with the surge levels (all under ~1 m) would
# squash the actual question ("does surge clear the COASTAL crest?") down
# to an invisible sliver at the bottom of the chart.
fig, (ax_zoom, ax_ctx) = plt.subplots(
    1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [2.2, 1]}
)

# -- left panel: zoomed on the actual coastal question --------------------
ax_zoom.fill_between(
    table_rps,
    rp_min,
    rp_max,
    color="steelblue",
    alpha=0.25,
    label="Surge level across stations (min-max)",
)
ax_zoom.plot(
    table_rps,
    rp_mean,
    color="steelblue",
    marker="o",
    linewidth=1.8,
    label="Surge level across stations (mean)",
)

ax_zoom.axhline(
    coastal_protection_crest_m,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label=f"Coastal design crest (flat, no freeboard) = {coastal_protection_crest_m:+.3f} m",
)

ax_zoom.axvline(
    SCENARIO_SURGE_RP,
    color="darkorange",
    linestyle=":",
    linewidth=1.8,
    label=f"Scenario surge_rp = {SCENARIO_SURGE_RP:.0f}",
)
ax_zoom.plot(
    [SCENARIO_SURGE_RP],
    [scenario_levels.mean()],
    marker="D",
    color="darkorange",
    markersize=9,
    zorder=5,
)
ax_zoom.annotate(
    f"{scenario_levels.mean():+.3f} m\n(margin over coastal crest: "
    f"{scenario_levels.mean() - coastal_protection_crest_m:+.3f} m)",
    xy=(SCENARIO_SURGE_RP, scenario_levels.mean()),
    xytext=(10, 12),
    textcoords="offset points",
    fontsize=9,
    color="darkorange",
    fontweight="bold",
)

ax_zoom.set_xscale("log")
ax_zoom.set_xticks(table_rps)
ax_zoom.set_xticklabels([f"{int(rp)}" for rp in table_rps])
_pad = max(0.05, 0.15 * (rp_max.max() - rp_min.min()))
ax_zoom.set_ylim(
    min(rp_min.min(), coastal_protection_crest_m) - _pad,
    max(rp_max.max(), coastal_protection_crest_m) + _pad,
)
ax_zoom.set_xlabel("Surge return period (yr)")
ax_zoom.set_ylabel("Water level (m, model vertical datum)")
ax_zoom.set_title("Zoomed: surge vs. coastal design crest")
ax_zoom.legend(loc="best", fontsize=8, framealpha=0.9)
ax_zoom.grid(True, alpha=0.3, linewidth=0.5, which="both")

# -- right panel: full built-weir crest distribution, for context ---------
ax_ctx.hist(weir_elev, bins=40, orientation="horizontal", color="firebrick", alpha=0.6)
ax_ctx.axhline(coastal_protection_crest_m, color="black", linestyle="--", linewidth=1.5)
ax_ctx.set_xlabel("Weir segment count")
ax_ctx.set_ylabel("Built weir crest elevation (m)")
ax_ctx.set_title("Full built weir\n(all segments, incl. river reaches)")
ax_ctx.text(
    0.97,
    0.97,
    f"{n_at_floor}/{len(weir_elev)} segment(s) sit AT the\n"
    f"coastal floor ({coastal_protection_crest_m:.2f} m) --\n"
    f"the rest are pushed up by a\n"
    f"calibrated river reach via\n"
    f"max(coastal, river_crest)",
    transform=ax_ctx.transAxes,
    ha="right",
    va="top",
    fontsize=8,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
)
ax_ctx.grid(True, alpha=0.3, linewidth=0.5)

fig.suptitle(f"Surge level vs. coastal dike crest — basin {BASIN_ID}", fontsize=13)
fig.tight_layout()

out_path = FIGS_DIR / f"{BASIN_ID}_surge_vs_dike_crest.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
log.info(f"\nPlot written: {out_path}")
