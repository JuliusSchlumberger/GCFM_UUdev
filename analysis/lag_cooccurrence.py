"""Observed timing of surge and river peaks from reanalysis, against the configured compound lag.

Per basin, over the overlap of GTSM hourly surge (gtsm_hourly.nc, the station
nearest a mouth) and GloFAS daily discharge (river_forcing.nc discharge_daily,
the crossing with the largest mean flow):

  peaks: independent maxima above the median annual maximum of each series,
  declustered over SURGE_SEPARATION_DAYS (surge) and RIVER_SEPARATION_DAYS
  (discharge, as river_forcing.real_flood_event_matrix);
  pairing: each surge peak with the nearest discharge peak within
  PAIR_WINDOW_DAYS, and the reverse; lag = t_Q - t_surge (h; discharge is
  stamped at noon, so lags are whole days);
  at the coast: lag - T_river, T_river the inflow-to-mouth travel time of the
  river run at the main mouth (peak_lag_summary.csv, from analysis/peak_lag.py);
  materiality: tau95, the half-width of the river run's inflow hydrograph
  above 95 % of its peak; a coastal lag within tau95 still meets at least
  95 % of the peak discharge.

The configured lag (sfincs.boundary_setup.compound) uses the same sign
convention at the boundary: river peak minus surge peak.

Writes lag_cooccurrence_summary.csv, _pairs.csv and _hist.png to
<results_dir>/figures/analysis/.

Usage: python analysis/lag_cooccurrence.py [basin_id ...] [--river river_100_03SLR]
"""

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from _common import analysis_path, script_parser, site_path
from check_river_bed import _git_hash
from lag_plateau import configured_lag_hr
from peak_lag import DEFAULT_BASINS, _table

SURGE_SEPARATION_DAYS = 3
RIVER_SEPARATION_DAYS = 14
PAIR_WINDOW_DAYS = 7


def declustered_peaks(s: pd.Series, separation_days: float) -> pd.Series:
    """Return the maxima above the median annual maximum, at least separation_days apart."""
    s = s.dropna()
    threshold = float(s.groupby(s.index.year).max().median())
    sep = pd.Timedelta(days=separation_days)
    picked: list[pd.Timestamp] = []
    for t in s[s > threshold].sort_values(ascending=False).index:
        if all(abs(t - p) >= sep for p in picked):
            picked.append(t)
    return s.loc[sorted(picked)]


def nearest_lag_hr(a: pd.Series, b: pd.Series, window_days: float) -> np.ndarray:
    """Return, per peak in a, the time to the nearest peak in b in hours (NaN beyond window_days)."""
    tb = b.index.to_numpy()
    out = np.full(len(a), np.nan)
    for i, t in enumerate(a.index.to_numpy()):
        d = (tb - t) / np.timedelta64(1, "h")
        if len(d):
            k = int(np.argmin(np.abs(d)))
            if abs(d[k]) <= 24.0 * window_days:
                out[i] = d[k]
    return out


def series(basin_id: str) -> tuple[pd.Series, pd.Series, str]:
    root = site_path("results_dir") / basin_id
    forcing = root / "preprocessing_inputs" / "forcing"
    with xr.open_dataset(forcing / "river_forcing.nc", decode_times=False) as rf:
        q = rf["discharge_daily"].values.astype(float)
        days = rf["day"].values
    main = int(np.nanargmax(np.nanmean(q, axis=1)))
    t_q = (
        pd.to_datetime("1970-01-01") + pd.to_timedelta(days, "D") + pd.Timedelta("12h")
    )
    discharge = pd.Series(q[main], index=t_q)

    mouths = gpd.read_file(root / "sfincs_skeleton" / f"{basin_id}_mouths.gpkg")
    centres = mouths.geometry.centroid.to_crs(4326)
    with xr.open_dataset(forcing / "gtsm_hourly.nc") as g:
        lon, lat = g["longitude"].values, g["latitude"].values
        mx, my = centres.x.to_numpy(), centres.y.to_numpy()
        dist = np.min(
            np.hypot(lon[:, None] - mx[None, :], lat[:, None] - my[None, :]), axis=1
        )
        st = int(np.argmin(dist))
        surge = pd.Series(
            g["surge"].values[:, st].astype(float),
            index=pd.DatetimeIndex(g["time"].values),
        )
        station = str(g["gtsm_station"].values[st])
    start = max(surge.index[0], discharge.index[0])
    end = min(surge.index[-1], discharge.index[-1])
    return surge[start:end], discharge[start:end], station


def tau95_hr(basin_id: str, river: str) -> float:
    t, dis = _table(
        site_path("results_dir") / basin_id / "runs" / river / "sfincs", "sfincs.dis"
    )
    q = dis.sum(axis=1)
    above = t[q >= 0.95 * q.max()]
    return 0.5 * float(above.max() - above.min())


def river_travel_hr(basin_id: str) -> float:
    s = pd.read_csv(
        site_path("results_dir") / "figures" / "analysis" / "peak_lag_summary.csv"
    )
    s = s[(s["basin_id"].astype(str) == basin_id) & s["main_mouth"].astype(bool)]
    return float(s["river_travel_hr"].iloc[0])


def lag_basin(basin_id: str, river: str) -> tuple[dict, pd.DataFrame]:
    surge, discharge, station = series(basin_id)
    sp = declustered_peaks(surge, SURGE_SEPARATION_DAYS)
    qp = declustered_peaks(discharge, RIVER_SEPARATION_DAYS)
    lag_from_surge = nearest_lag_hr(sp, qp, PAIR_WINDOW_DAYS)  # t_Q - t_surge
    lag_from_river = -nearest_lag_hr(qp, sp, PAIR_WINDOW_DAYS)  # t_Q - t_surge
    t_river = river_travel_hr(basin_id)
    tau = tau95_hr(basin_id, river)
    lag_cfg = configured_lag_hr(basin_id, river)
    row = {
        "basin_id": basin_id,
        "gtsm_station": station,
        "period": f"{surge.index[0]:%Y-%m-%d} to {surge.index[-1]:%Y-%m-%d}",
        "n_surge_peaks": len(sp),
        "n_river_peaks": len(qp),
        "t_river_hr": t_river,
        "tau95_hr": tau,
        "lag_config_hr": lag_cfg,
    }
    for name, lag in (("surge_first", lag_from_surge), ("river_first", lag_from_river)):
        paired = lag[np.isfinite(lag)]
        med = float(np.median(paired)) if paired.size else np.nan
        row.update(
            {
                f"{name}_n_paired": int(paired.size),
                f"{name}_share_unpaired": float(1 - paired.size / max(len(lag), 1)),
                f"{name}_lag_median_hr": med,
                f"{name}_lag_p25_hr": float(np.percentile(paired, 25))
                if paired.size
                else np.nan,
                f"{name}_lag_p75_hr": float(np.percentile(paired, 75))
                if paired.size
                else np.nan,
                f"{name}_share_within_1day": float(np.mean(np.abs(paired) < 24.0))
                if paired.size
                else np.nan,
                f"{name}_coast_lag_median_hr": med - t_river,
                f"{name}_material": bool(abs(med - t_river) > tau)
                if paired.size
                else False,
            }
        )
    pairs = pd.concat(
        [
            pd.DataFrame(
                {
                    "basin_id": basin_id,
                    "pairing": "surge_first",
                    "t_peak": sp.index,
                    "lag_hr": lag_from_surge,
                }
            ),
            pd.DataFrame(
                {
                    "basin_id": basin_id,
                    "pairing": "river_first",
                    "t_peak": qp.index,
                    "lag_hr": lag_from_river,
                }
            ),
        ]
    )
    return row, pairs


def plot(rows: list[dict], pairs: pd.DataFrame, tag: str, path) -> None:
    fig, axes = plt.subplots(
        1, len(rows), figsize=(6.5 * len(rows), 4.2), squeeze=False
    )
    bins = np.arange(-PAIR_WINDOW_DAYS * 24 - 12, PAIR_WINDOW_DAYS * 24 + 13, 24)
    for ax, r in zip(axes[0], rows):
        p = pairs[pairs["basin_id"] == r["basin_id"]]
        for name, color in (("surge_first", "tab:blue"), ("river_first", "tab:orange")):
            lag = p.loc[p["pairing"] == name, "lag_hr"].dropna()
            ax.hist(
                lag,
                bins=bins,
                histtype="step",
                lw=1.5,
                color=color,
                label=f"{name}: n={len(lag)}, median {r[f'{name}_lag_median_hr']:+.0f} h, "
                f"unpaired {r[f'{name}_share_unpaired']:.0%}",
            )
        ax.axvline(
            r["lag_config_hr"],
            color="tab:red",
            ls="--",
            label=f"configured lag {r['lag_config_hr']:+.0f} h",
        )
        ax.axvspan(
            r["t_river_hr"] - r["tau95_hr"],
            r["t_river_hr"] + r["tau95_hr"],
            color="tab:green",
            alpha=0.12,
            label=f"T_river {r['t_river_hr']:.0f} h +/- tau95 {r['tau95_hr']:.0f} h",
        )
        ax.set_xlabel("lag at the boundary, t_Q - t_surge (h)")
        ax.set_ylabel("peak pairs")
        ax.set_title(
            f"{r['basin_id']} | GTSM {r['gtsm_station']} | {r['period']}", fontsize=9
        )
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"lag_cooccurrence: surge-river peak co-occurrence (GTSM, GloFAS) | {tag}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("basin_ids", nargs="*", default=DEFAULT_BASINS)
    parser.add_argument("--river", default="river_100_03SLR")
    args = parser.parse_args()
    out = [lag_basin(str(b), args.river) for b in args.basin_ids]
    rows = [r for r, _ in out]
    pairs = pd.concat([p for _, p in out], ignore_index=True)
    plot(rows, pairs, _git_hash(), analysis_path(None, __file__, "hist.png"))
    pairs.to_csv(
        analysis_path(None, __file__, "pairs.csv"), index=False, float_format="%.1f"
    )
    df = pd.DataFrame(rows)
    df.to_csv(
        analysis_path(None, __file__, "summary.csv"), index=False, float_format="%.3f"
    )
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(df.round(2).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
