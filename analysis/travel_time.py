"""River peak travel from the inflow to the main mouth: SFINCS reference points against the parametric estimates.

From the river-only event run (inflow: sum of sfincs.dis; main mouth: the
mouth cross-section with the largest tide-filtered discharge, Godin filter as
in peak_lag.py), times after the inflow peak of:

  first maximum: the filtered mouth discharge's maximum;
  half rise: the filtered mouth discharge first reaching half way from its
    value at the run start to its maximum, minus the same for the inflow;
  plateau start / midpoint / end: the span within --plateau-frac of the
    maximum (lag_plateau.py);
    open-ended when the run ends on the plateau;
  configured: the river travel implied by the configured compound lag,
    surge travel - lag.

Parametric (src.river_profile.walled_travel_time from each crossing to its
network end, the slowest crossing as rule 11 takes it, on rule 11's bed and
calibration sea level): at the calibration discharge (rule 11's walled_travel.csv)
and at the routed bankfull discharge.

Writes summary.csv and travel_time.png (with travel_time.txt, its description) to
<results_dir>/figures/analysis/travel_time/.

Usage: python analysis/travel_time.py 2433835 3279946 [--coast coast_100_03SLR] [--river river_100_03SLR]
"""

import logging
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _basin_inputs import basin_paths, load_config
from _common import REPO_ROOT, script_parser, site_path
from _figure_style import panel_letters, save_figure, write_caption
from _upstream_profile import routed_discharge
from lag_plateau import plateau_basin

sys.path.insert(0, str(REPO_ROOT / "workflow"))

log = logging.getLogger(__name__)

POINTS = {
    "first_max": ("first maximum", "tab:red", "-"),
    "half_rise": ("half rise", "tab:orange", "-"),
    "plateau_start": ("plateau start", "tab:green", ":"),
    "plateau_mid": ("plateau midpoint", "tab:green", "-"),
    "plateau_end": ("plateau end", "tab:green", ":"),
    "configured": ("configured lag", "tab:blue", "--"),
    "walled_calibration": ("walled, calibration Q", "0.4", "--"),
    "walled_bankfull": ("walled, bankfull Q", "k", "-."),
}


def half_rise_hr(t: np.ndarray, q: np.ndarray) -> float:
    ok = np.isfinite(q)
    t, q = t[ok], q[ok]
    k = int(np.argmax(q))
    half = q[0] + 0.5 * (q[k] - q[0])
    return float(t[: k + 1][np.argmax(q[: k + 1] >= half)])


def walled_times(basin_id: str) -> dict[str, float]:
    from src.river_network import normalize_reach_id
    from src.river_profile import mainstem_profile, walled_travel_time

    cfg = load_config()
    rdm = cfg["river_processing"]["river_depth_modelling"]
    p = basin_paths(basin_id)
    bed_dir = p["node_table"].parent
    walled = pd.read_csv(
        bed_dir / f"{basin_id}_walled_travel.csv", dtype={"crossing_reach_id": str}
    )
    nodes = pd.read_csv(p["node_table"])
    nodes["reach_id"] = nodes["reach_id"].map(normalize_reach_id)
    import geopandas as gpd
    from src.river_network import build_downstream_adjacency

    rivers = gpd.read_file(p["rivers_depth"])
    downstream = build_downstream_adjacency(rivers)
    q_bf = routed_discharge(rivers, p["river_forcing"], "bankfull_discharge_m3s")
    t_cal, t_bf = [], []
    for reach, w in walled.groupby("crossing_reach_id", sort=True):
        reach = normalize_reach_id(reach)
        prof = mainstem_profile(nodes, downstream, reach, reach_q=q_bf)
        q = prof["reach_id"].map(q_bf).to_numpy(float)
        t = walled_travel_time(
            prof,
            q,
            float(w["level_m"].iloc[-1]),
            float(cfg["river_processing"]["channel_manning_n"]),
            forcing_hr=float(rdm["ramp_hr"]),
            regime_rT=tuple(rdm["wave_regime_rT"]),
        )
        t_cal.append(float(w["t_wave_hr"].iloc[-1]))
        t_bf.append(float(t["t_wave_hr"].iloc[-1]))
    return {"walled_calibration": max(t_cal), "walled_bankfull": max(t_bf)}


def basin_times(basin_id: str, coast: str, river: str, frac: float) -> dict:
    r = plateau_basin(site_path("results_dir"), basin_id, coast, river, frac)
    t_dis, q_in, tr, _q, q_f, _tc, _zs, _ts = r["_series"]
    t0 = r["t_river_inflow_hr"]
    times = {
        "first_max": r["t_mouth_q_max_hr"] - t0,
        "half_rise": half_rise_hr(tr, q_f) - half_rise_hr(t_dis, q_in),
        "plateau_start": r["plateau_start_hr"] - t0,
        "plateau_mid": r["plateau_mid_hr"] - t0,
        "plateau_end": r["plateau_end_hr"] - t0,
        "configured": r["surge_travel_hr"] - r["lag_config_hr"],
        **walled_times(basin_id),
    }
    row = {
        "basin_id": basin_id,
        "mouth": r["mouth"],
        "q_inflow_peak_m3s": r["q_inflow_peak_m3s"],
        "q_mouth_max_m3s": r["q_mouth_max_m3s"],
        "plateau_open_ended": r["plateau_open_ended"],
        **{f"{k}_hr": v for k, v in times.items()},
    }
    return {"row": row, "series": (t_dis - t0, q_in, tr - t0, q_f), "times": times}


def figure(results: list[dict], path, river: str, frac: float) -> None:
    n = len(results)
    fig, axes = plt.subplots(
        2, n, figsize=(6.5 * n, 8), squeeze=False, gridspec_kw={"height_ratios": [3, 2]}
    )
    for c, res in enumerate(results):
        (t_in, q_in, t_m, q_f), times = res["series"], res["times"]
        a = axes[0, c]
        a.plot(t_in, q_in / np.nanmax(q_in), color="0.3", lw=1.4)
        a.plot(t_m, q_f / np.nanmax(q_f), color="tab:red", lw=1.6)
        a.axvspan(
            times["plateau_start"], times["plateau_end"], color="tab:green", alpha=0.12
        )
        for key, (_label, col, ls) in POINTS.items():
            a.axvline(times[key], color=col, ls=ls, lw=1.1)
        a.axvline(0, color="0.3", lw=0.8)
        a.set_xlim(-72, max(times["plateau_end"], times["walled_bankfull"]) + 48)
        a.set_ylim(0, 1.08)
        a.set_ylabel("discharge / its maximum")
        a.grid(alpha=0.3)
        b = axes[1, c]
        keys = list(POINTS)
        for k, key in enumerate(keys):
            b.plot(times[key], k, "o", color=POINTS[key][1], ms=7)
        b.set_yticks(range(len(keys)))
        b.set_yticklabels([POINTS[k][0] for k in keys], fontsize=8)
        b.invert_yaxis()
        b.set_xlim(a.get_xlim())
        b.set_xlabel("hours after the inflow peak")
        b.grid(alpha=0.3, axis="x")
    panel_letters(np.r_[axes[0], axes[1]])
    fig.tight_layout()
    save_figure(fig, path, dpi=130)
    plt.close(fig)
    letters = [f"({chr(97 + k)})" for k in range(2 * n)]
    basins = [r["row"] for r in results]
    per_basin = "\n".join(
        f"- {letters[k]} and {letters[n + k]}: basin {r['basin_id']}, inflow peak "
        f"{r['q_inflow_peak_m3s']:.0f} m3/s, main-mouth maximum {r['q_mouth_max_m3s']:.0f} m3/s"
        + (
            "; the plateau is still open when the run ends, so its end and midpoint are lower bounds"
            if r["plateau_open_ended"]
            else ""
        )
        for k, r in enumerate(basins)
    )
    write_caption(
        path,
        f"""
River peak travel from the model inflow to the main mouth in the river-only event {river}, against the parametric estimates. Time 0 is the peak of the summed inflow; the values are in summary.csv.
Top row: dark grey line, summed inflow; red line, tide-filtered (Godin) discharge through the main mouth, each divided by its maximum; green band, the mouth plateau, where the filtered discharge stays within {1 - frac:.0%} of its maximum. Vertical lines mark the reference points listed in the bottom row, in the same colours: red, first maximum of the mouth discharge; orange, half rise (time the mouth discharge reaches half way from its start value to its maximum, minus the same for the inflow); green dotted, plateau start and end; green solid, plateau midpoint; blue dashed, travel implied by the configured compound lag (surge travel minus lag); grey dashed, walled estimate at the calibration discharge (rule 11); black dash-dotted, walled estimate at bankfull discharge.
Bottom row: the same reference points as dots, one row each.
{per_basin}
""",
    )


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("basin_ids", nargs="+")
    parser.add_argument("--coast", default="coast_100_03SLR")
    parser.add_argument("--river", default="river_100_03SLR")
    parser.add_argument("--plateau-frac", type=float, default=0.98)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = site_path("results_dir") / "figures" / "analysis" / "travel_time"
    out.mkdir(parents=True, exist_ok=True)
    results = [
        basin_times(str(b), args.coast, args.river, args.plateau_frac)
        for b in args.basin_ids
    ]
    table = pd.DataFrame([r["row"] for r in results])
    table.to_csv(out / "summary.csv", index=False, float_format="%.2f")
    figure(results, out / "travel_time.png", args.river, args.plateau_frac)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(table.round(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
