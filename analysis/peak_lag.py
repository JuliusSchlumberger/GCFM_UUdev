"""Peak arrival of the surge and the river at each river mouth, and the compound lag that aligns them.

From a coastal-only and a river-only event run of the same return period and
SLR (default coast_100_03SLR and river_100_03SLR), at every mouth of the basin
(the skeleton's <basin>_mouths.gpkg: an obs point and a cross-section
mouth_upstream_m up each mouth reach):

  coastal run: t_surge_boundary, the peak of the mean water-level boundary
  (sfincs.bzs); t_surge_mouth, the peak water level at the mouth (sfincs_his.nc);
  river run: t_river_inflow, the peak of the summed inflow (sfincs.dis);
  t_river_mouth, the peak of the tide-filtered discharge through the mouth
  cross-section (Godin 24-24-25 h running means), and of the tide-filtered
  level as a check.

lag_hr = surge_travel_hr - river_travel_hr is the river peak minus the surge
peak at the boundary (sfincs.boundary_setup.compound.lag_hr, > 0 delays the
river; auto takes rule event_timing's estimate) that makes both peaks reach that mouth together in a
compound run; travel times do not depend on where each run places its peak.
The basin's value is the main mouth's: the one with the largest filtered
river-peak discharge. Times are hours from the forcing's tref.

Writes one figure per mouth, 90_peak_lag_<coast>_<river>_<mouth>.png, to
<results_dir>/<basin>/figures/analysis/ and peak_lag_summary.csv (one row per
basin and mouth) to <results_dir>/figures/analysis/.

Usage: python analysis/peak_lag.py [basin_id ...] [--coast coast_100_03SLR] [--river river_100_03SLR]
"""

import logging

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from _common import analysis_path, script_parser, site_path
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from src.figure_style import (
    EVENT_STYLE,
    figure_size,
    panel_letters,
    save_figure,
    style_context,
    write_caption,
)

log = logging.getLogger(__name__)

DEFAULT_BASINS = ["2433835", "3279946"]


def godin(series: np.ndarray, dt_hr: float) -> np.ndarray:
    """Return the Godin tide filter (running means of 24, 24 and 25 h) of an evenly sampled series; NaN at the ends."""
    out = np.asarray(series, dtype=float)
    for hours in (24.0, 24.0, 25.0):
        n = max(int(round(hours / dt_hr)), 1)
        out = pd.Series(out).rolling(n, center=True, min_periods=n).mean().to_numpy()
    return out


def peak_hr(t_hr: np.ndarray, series: np.ndarray) -> float:
    """Return the time of the series' maximum (NaN if all NaN)."""
    v = np.asarray(series, dtype=float)
    return float(t_hr[np.nanargmax(v)]) if np.isfinite(v).any() else float("nan")


def _his(run_dir):
    ds = xr.open_dataset(run_dir / "sfincs_his.nc", decode_times=False)
    t_hr = ds["time"].values.astype(float) / 3600.0
    return ds, t_hr


def _table(run_dir, name: str) -> tuple[np.ndarray, np.ndarray]:
    a = np.loadtxt(run_dir / name, ndmin=2)
    return a[:, 0] / 3600.0, a[:, 1:]


def lag_basin(results_dir, basin_id: str, coast: str, river: str) -> list[dict]:
    root = results_dir / basin_id
    mouths = gpd.read_file(root / "sfincs_skeleton" / f"{basin_id}_mouths.gpkg")
    coast_dir = root / "runs" / coast / "sfincs"
    river_dir = root / "runs" / river / "sfincs"

    t_bzs, bzs = _table(coast_dir, "sfincs.bzs")
    t_surge_boundary = peak_hr(t_bzs, bzs.mean(axis=1))
    t_dis, dis = _table(river_dir, "sfincs.dis")
    t_river_inflow = peak_hr(t_dis, dis.sum(axis=1))

    cds, tc = _his(coast_dir)
    rds, tr = _his(river_dir)
    dt_r = float(np.median(np.diff(tr)))
    crs_names = [
        str(n).strip() for n in np.asarray(rds["crosssection_name"].values).astype(str)
    ]
    rows, panels = [], []
    for _, m in mouths.iterrows():
        k = int(m["obs_index"])
        zs_coast = cds["point_zs"].values[:, k]
        zs_river = rds["point_zs"].values[:, k]
        q_river = np.abs(
            rds["crosssection_discharge"].values[:, crs_names.index(m["name"])]
        )
        q_f, zs_f = godin(q_river, dt_r), godin(zs_river, dt_r)
        t_surge_mouth = peak_hr(tc, zs_coast)
        t_river_mouth = peak_hr(tr, q_f)
        rows.append(
            {
                "basin_id": basin_id,
                "mouth": m["name"],
                "coast_run": coast,
                "river_run": river,
                "t_surge_boundary_hr": t_surge_boundary,
                "t_surge_mouth_hr": t_surge_mouth,
                "surge_travel_hr": t_surge_mouth - t_surge_boundary,
                "zs_surge_mouth_max_m": float(np.nanmax(zs_coast)),
                "t_river_inflow_hr": t_river_inflow,
                "t_river_mouth_hr": t_river_mouth,
                "t_river_mouth_level_hr": peak_hr(tr, zs_f),
                "river_travel_hr": t_river_mouth - t_river_inflow,
                "q_river_mouth_peak_m3s": float(np.nanmax(q_f)),
                "lag_hr": (t_surge_mouth - t_surge_boundary)
                - (t_river_mouth - t_river_inflow),
            }
        )
        panels.append((rows[-1], zs_coast, q_river, q_f))
    cds.close()
    rds.close()
    main = max(rows, key=lambda r: r["q_river_mouth_peak_m3s"])
    for r in rows:
        r["main_mouth"] = r is main

    for row, zs_c, q, qf in panels:
        plot_mouth(row, tc, zs_c, tr, q, qf)
    return rows


def plot_mouth(row: dict, tc, zs_coast, tr, q_raw, q_filtered) -> None:
    """Plot one mouth's surge and river peaks against the time since each run's boundary peak.

    (a) water level at the mouth in the coastal run, (b) discharge through the
    mouth in the river run, raw and tide-filtered. Both mouth-peak times are
    drawn in both panels, so their gap is the compound lag.
    """
    surge, river = EVENT_STYLE["coast"]["color"], EVENT_STYLE["river"]["color"]
    path = analysis_path(
        row["basin_id"],
        __file__,
        f"{row['coast_run']}_{row['river_run']}_{row['mouth']}.png",
    )
    with style_context():
        fig, (ax_l, ax_q) = plt.subplots(
            2, 1, figsize=figure_size(110), sharex=True, layout="constrained"
        )
        ax_l.plot(tc - row["t_surge_boundary_hr"], zs_coast, color=surge)
        ax_q.plot(
            tr - row["t_river_inflow_hr"], q_raw, color=river, linewidth=0.5, alpha=0.5
        )
        ax_q.plot(tr - row["t_river_inflow_hr"], q_filtered, color=river, linewidth=1.4)
        for ax in (ax_l, ax_q):
            ax.axvline(0.0, color="0.6", linewidth=0.6)
            ax.axvline(
                row["surge_travel_hr"], color=surge, linestyle="--", linewidth=0.9
            )
            ax.axvline(
                row["river_travel_hr"], color=river, linestyle="--", linewidth=0.9
            )
        # style-override: annotations kept to a minimum | user: the lag as an arrow between the peak lines
        at_bottom = blended_transform_factory(ax_l.transData, ax_l.transAxes)
        ax_l.annotate(
            "",
            xy=(row["surge_travel_hr"], 0.08),
            xytext=(row["river_travel_hr"], 0.08),
            xycoords=at_bottom,
            textcoords=at_bottom,
            arrowprops=dict(
                arrowstyle="<->", color="0.3", linewidth=0.8, shrinkA=0, shrinkB=0
            ),
        )
        ax_l.text(
            max(row["surge_travel_hr"], row["river_travel_hr"]),
            0.08,
            f"  lag {row['lag_hr']:+.1f} h",
            transform=at_bottom,
            ha="left",
            va="center",
            color="0.3",
        )
        ax_l.set_ylabel("Water level (m +GOCO06s)")
        ax_q.set_ylabel(r"Discharge (m$^3$ s$^{-1}$)")
        ax_q.set_xlabel("Time since the peak at the model boundary (h)")
        panel_letters([ax_l, ax_q])
        fig.legend(
            handles=[
                Line2D([0], [0], color=surge, label="Water level, coastal run"),
                Line2D(
                    [0],
                    [0],
                    color=river,
                    linewidth=0.5,
                    alpha=0.5,
                    label="Discharge, river run",
                ),
                Line2D(
                    [0],
                    [0],
                    color=river,
                    linewidth=1.4,
                    label="Discharge, tide-filtered",
                ),
                Line2D(
                    [0], [0], color="0.4", linestyle="--", label="Peak at the mouth"
                ),
            ],
            loc="outside lower center",
            ncols=4,
        )
        save_figure(fig, path)
        plt.close(fig)
    lag = row["lag_hr"]
    order = (
        "the river peak leaves the inflow first"
        if lag < 0
        else "the surge peak leaves the sea boundary first"
    )
    write_caption(
        path,
        f"At mouth {row['mouth']} of basin {row['basin_id']} the surge peak takes "
        f"{row['surge_travel_hr']:.1f} h from the sea boundary and the river peak "
        f"{row['river_travel_hr']:.1f} h from the inflow, so the peaks meet at the mouth with a "
        f"compound lag of {lag:+.1f} h ({order}).\n"
        f"(a) Water level at the mouth in the coastal run {row['coast_run']} (blue), against the "
        "time since the peak of the mean water-level boundary. "
        f"(b) Discharge through the mouth cross-section in the river run {row['river_run']}: raw "
        "(thin, light orange) and tide-filtered (Godin 24-24-25 h running means, solid orange), "
        "against the time since the peak of the summed inflow. "
        "Grey line: the boundary peak (0 h); dashed blue and orange lines, in both panels: the "
        "peak at the mouth of the water level and of the filtered discharge; their gap, marked by the arrow in (a), is the lag "
        f"(river-peak delay at the boundary, sfincs.boundary_setup.compound.lag_hr). Peak level "
        f"{row['zs_surge_mouth_max_m']:.2f} m above the GOCO06s geoid, peak filtered discharge "
        f"{row['q_river_mouth_peak_m3s']:,.0f} m3/s.",
    )


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("basin_ids", nargs="*", default=DEFAULT_BASINS)
    parser.add_argument("--coast", default="coast_100_03SLR")
    parser.add_argument("--river", default="river_100_03SLR")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results_dir = site_path("results_dir")
    rows = [
        r
        for b in args.basin_ids
        for r in lag_basin(results_dir, str(b), args.coast, args.river)
    ]
    df = pd.DataFrame(rows)
    df.to_csv(
        analysis_path(None, __file__, "summary.csv"), index=False, float_format="%.3f"
    )
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(df.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
