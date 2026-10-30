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
peak at the boundary (sfincs.boundary_setup.compound.lag_overrides_hr, > 0
delays the river) that makes both peaks reach that mouth together in a
compound run; travel times do not depend on where each run places its peak.
The basin's value is the main mouth's: the one with the largest filtered
river-peak discharge. Times are hours from the forcing's tref.

Writes per basin 90_peak_lag_<coast>_<river>.png to
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
from check_river_bed import _git_hash

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


def lag_basin(
    results_dir, basin_id: str, coast: str, river: str, tag: str
) -> list[dict]:
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
        panels.append(
            (
                m["name"],
                zs_coast,
                q_river,
                q_f,
                t_surge_mouth,
                t_river_mouth,
                rows[-1]["lag_hr"],
            )
        )
    cds.close()
    rds.close()
    main = max(rows, key=lambda r: r["q_river_mouth_peak_m3s"])
    for r in rows:
        r["main_mouth"] = r is main

    fig, axes = plt.subplots(
        len(panels), 1, figsize=(11, 3.2 * len(panels)), squeeze=False
    )
    for ax, (name, zs_c, q, qf, ts, trv, lag) in zip(axes[:, 0], panels):
        ax.plot(tc, zs_c, color="tab:blue", lw=1.0, label=f"{coast}: level at mouth")
        ax.axvline(ts, color="tab:blue", ls="--", lw=1)
        ax2 = ax.twinx()
        ax2.plot(
            tr,
            q,
            color="tab:orange",
            lw=0.6,
            alpha=0.5,
            label=f"{river}: discharge at mouth",
        )
        ax2.plot(tr, qf, color="tab:red", lw=1.4, label="tide-filtered (Godin)")
        ax2.axvline(trv, color="tab:red", ls="--", lw=1)
        ax.set_ylabel("water level (m)")
        ax2.set_ylabel("discharge (m3/s)")
        ax.set_title(
            f"{name}: lag = {lag:+.1f} h (surge travel - river travel to the mouth)",
            fontsize=9,
        )
        ax.grid(alpha=0.3)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    axes[-1, 0].set_xlabel("hours from tref")
    fig.suptitle(
        f"peak lag at the river mouths | {basin_id} | {coast} vs {river} | {tag}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(analysis_path(basin_id, __file__, f"{coast}_{river}.png"), dpi=120)
    plt.close(fig)
    return rows


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("basin_ids", nargs="*", default=DEFAULT_BASINS)
    parser.add_argument("--coast", default="coast_100_03SLR")
    parser.add_argument("--river", default="river_100_03SLR")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results_dir, tag = site_path("results_dir"), _git_hash()
    rows = [
        r
        for b in args.basin_ids
        for r in lag_basin(results_dir, str(b), args.coast, args.river, tag)
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
