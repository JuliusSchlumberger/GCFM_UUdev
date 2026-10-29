"""Compare per-station design storm tide per RP with the built coastal weir crest.

Basin-level, needs no scenario run. Storm tide comes from surge_forcing.nc's
COAST-RP table via src.surge.lookup_storm_tide_at_rp, as a scenario's surge_rp_yr
resolves. coastal_protection_crest_m is the flat per-basin FLOPROS design crest;
plus freeboard_m, rounded up to 0.1 m, it is the floor of every built weir
segment. Pure-coastal segments sit at that floor; segments holding back higher
river water (banks, the mouth) are higher, so the weir's max can be set by
inland reaches unrelated to surge. Older results layouts are tried as fallbacks.
Writes <results_dir>/<basin_id>/figures/analysis/
90_plot_surge_vs_dike_crest_comparison.png.

Usage:
    conda run -n <env> python analysis/plot_surge_vs_dike_crest.py [basin_id] [scenario_surge_rp]
    conda run -n <env> python analysis/plot_surge_vs_dike_crest.py 2433835 500
"""

from __future__ import annotations

import argparse
import logging
import sys

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from _common import (  # noqa: E402
    REPO_ROOT,
    add_basin_id_argument,
    analysis_path,
    first_existing,
    script_parser,
    site_path,
)

sys.path.insert(0, str(REPO_ROOT / "workflow"))
from src.surge import ceil_crest, lookup_storm_tide_at_rp


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = script_parser(__doc__)
    add_basin_id_argument(parser)
    parser.add_argument(
        "scenario_surge_rp",
        nargs="?",
        type=float,
        default=500.0,
        help="scenario surge return period in years (default: 500)",
    )
    return parser


def main() -> None:
    """Run the surge-vs-crest comparison for one basin and write its figure."""
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger(__name__)

    BASIN_ID = args.basin_id
    SCENARIO_SURGE_RP = args.scenario_surge_rp

    with open(REPO_ROOT / "config" / "config.yml") as fh:
        config = yaml.safe_load(fh)
    with open(REPO_ROOT / "config" / "scenarios.yml") as fh:
        SCENARIO_DEFS = yaml.safe_load(fh) or {}
    # results_dir is a site setting; see config/site.example.yml.
    RESULTS_DIR = site_path("results_dir")

    basin_dir = RESULTS_DIR / BASIN_ID

    surge_forcing_path = first_existing(
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
            basin_dir
            / "inputs"
            / "domain"
            / f"{BASIN_ID}_coastal_protection_weir.gpkg",
        ]
        + sorted(basin_dir.glob("runs/*/sfincs/*_coastal_protection_weir.gpkg"))
        + sorted(basin_dir.glob("scenarios/*/sfincs/*_coastal_protection_weir.gpkg"))
    )
    weir_path = first_existing(weir_candidates)
    log.info(f"coastal_protection_weir.gpkg: {weir_path}")

    # surge: per-station storm tide at every COAST-RP tabulated RP
    # surge_forcing.nc's storm_tide_at_rp_m/calm_sea_level_m are MDT-only (SLR is stored as a
    # dimensionless slr_fingerprint, see src.surge.apply_slr_fingerprint). The
    # "default" scenario's slr_m from scenarios.yml is applied as
    # 14_build_sfincs.py does at build time.
    surge_ds = xr.open_dataset(surge_forcing_path, decode_times=False)
    rps_yr = surge_ds["rp_yr"].values.astype(float)
    n_stations = surge_ds.sizes["station"]
    coastal_protection_crest_raw_m = float(
        surge_ds["coastal_protection_crest_m"].values
    )
    # The floor as built (src.protection_weir): + freeboard_m, rounded up to the
    # next 0.1 m, as the storm-tide levels below.
    _freeboard_m = float(
        config["river_processing"]["river_depth_modelling"]["freeboard_m"]
    )
    coastal_protection_crest_m = float(
        ceil_crest(coastal_protection_crest_raw_m + _freeboard_m)
    )
    calm_sea_level_m = float(surge_ds["calm_sea_level_m"].values)

    slr_cfg = config["boundary_forcings"]["surge"]["slr"]
    default_slr_m = float(SCENARIO_DEFS.get("default", {}).get("slr_m", 0.0))
    effective_slr_m = default_slr_m if slr_cfg["enabled"] else 0.0

    levels_by_rp = np.stack(
        [
            lookup_storm_tide_at_rp(surge_ds, rp_yr, slr_m=effective_slr_m)
            for rp_yr in rps_yr
        ]
    )  # (n_rp, n_station)
    rp_min = levels_by_rp.min(axis=1)
    rp_mean = levels_by_rp.mean(axis=1)
    rp_max = levels_by_rp.max(axis=1)

    log.info(f"Stations: {n_stations}")
    log.info(
        f"coastal crest floor as built: {coastal_protection_crest_m:+.2f} m (FLOPROS design crest "
        f"{coastal_protection_crest_raw_m:+.4f} m, MDT-only, +{_freeboard_m:.2f} m freeboard, rounded up to 0.1 m)"
    )
    log.info(
        f"calm_sea_level_m (mean sea level correction, MDT-only): {calm_sea_level_m:+.4f} m"
    )
    log.info(
        f"SLR target applied (scenarios.yml 'default' slr_m): "
        f"{effective_slr_m:+.3f} m x per-station fingerprint "
        f"(enabled={slr_cfg['enabled']}) -- levels below already include this"
    )
    log.info("RP (yr)   min      mean     max      (storm tide level, m)")
    for rp_yr, lo, mu, hi in zip(rps_yr, rp_min, rp_mean, rp_max):
        flag = (
            "  <-- exceeds coastal crest floor"
            if mu > coastal_protection_crest_m
            else ""
        )
        log.info(f"{rp_yr:>7.0f}   {lo:+.3f}   {mu:+.3f}   {hi:+.3f}{flag}")

    # built weir crest
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

    # scenario RP under investigation
    if not np.any(np.isclose(rps_yr, SCENARIO_SURGE_RP)):
        raise ValueError(
            f"surge RP {SCENARIO_SURGE_RP} not tabulated in COAST-RP ({rps_yr.tolist()})"
        )
    scenario_idx = int(np.nonzero(np.isclose(rps_yr, SCENARIO_SURGE_RP))[0][0])
    scenario_levels = levels_by_rp[scenario_idx]
    log.info(
        f"\nAt the scenario's own surge_rp_yr={SCENARIO_SURGE_RP:.0f}: storm tide "
        f"[{scenario_levels.min():+.3f}, {scenario_levels.max():+.3f}] m "
        f"(mean {scenario_levels.mean():+.3f} m) vs. coastal crest floor "
        f"{coastal_protection_crest_m:+.4f} m -> margin "
        f"{scenario_levels.mean() - coastal_protection_crest_m:+.4f} m "
        f"({'OVERTOPS' if scenario_levels.mean() > coastal_protection_crest_m else 'below crest -- no coastal overtopping expected'})"
    )

    # plot
    # Separate y-scales: the weir's full range is set by inland river-crest
    # segments, which would squash surge levels (under ~1 m) to a sliver.
    fig, (ax_zoom, ax_ctx) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [2.2, 1]}
    )

    # left panel: zoom on the coastal crest
    ax_zoom.fill_between(
        rps_yr,
        rp_min,
        rp_max,
        color="steelblue",
        alpha=0.25,
        label="Surge level across stations (min-max)",
    )
    ax_zoom.plot(
        rps_yr,
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
        label=f"Coastal crest floor as built = {coastal_protection_crest_m:+.2f} m",
    )

    ax_zoom.axvline(
        SCENARIO_SURGE_RP,
        color="darkorange",
        linestyle=":",
        linewidth=1.8,
        label=f"Scenario surge_rp_yr = {SCENARIO_SURGE_RP:.0f}",
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
    ax_zoom.set_xticks(rps_yr)
    ax_zoom.set_xticklabels([f"{int(rp_yr)}" for rp_yr in rps_yr])
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

    # right panel: full built-weir crest distribution
    ax_ctx.hist(
        weir_elev, bins=40, orientation="horizontal", color="firebrick", alpha=0.6
    )
    ax_ctx.axhline(
        coastal_protection_crest_m, color="black", linestyle="--", linewidth=1.5
    )
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

    out_path = analysis_path(BASIN_ID, __file__, "comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"\nPlot written: {out_path}")


if __name__ == "__main__":
    main()
