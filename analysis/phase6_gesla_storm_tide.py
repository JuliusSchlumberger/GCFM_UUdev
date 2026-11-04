"""Phase 6.2: storm-tide bias of the GTSM forcing against GESLA v4.1 gauges.

Gauges: the open-coast gauges inspect_gesla_gauges.screen_basin finds usable
(>= 10 complete years, outside the domain, within 50 km of a GTSM station),
paired with that station; gauges excluded there only for a timing lag under
an hour are kept. Per complete year, the annual
maximum of the level above that year's mean, over the hours both series
have; removing each year's mean takes out the gauge datum and its steps.
Bias = median over years of gauge - GTSM, and the difference of the 10-year
Gumbel return levels fitted to each.

Writes per basin 90_phase6_gesla_storm_tide_annual_maxima.png to
<results_dir>/<basin>/figures/analysis/ and phase6_gesla_storm_tide_summary.csv
to <results_dir>/figures/analysis/.

Usage: python analysis/phase6_gesla_storm_tide.py [basin_id ...]
"""

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from _common import analysis_path, script_parser, site_path
from _phase1 import PHASE1_BASINS, write_summary
from inspect_gesla_gauges import (
    basin_gauges,
    complete_years,
    load_on_hour,
    read_metadata,
    screen_basin,
)
from scipy import stats

log = logging.getLogger(__name__)
RP_YEARS = 10.0


def annual_maxima(gauge: pd.Series, model: pd.Series, years: list[int]) -> pd.DataFrame:
    both = pd.concat([gauge.rename("g"), model.rename("m")], axis=1).dropna()
    both = both[both.index.year.isin(years)]
    anom = both - both.groupby(both.index.year).transform("mean")
    return anom.groupby(anom.index.year).max()


def gumbel_rl(am: np.ndarray, rp_years: float = RP_YEARS) -> float:
    loc, scale = stats.gumbel_r.fit(am)
    return float(stats.gumbel_r.ppf(1.0 - 1.0 / rp_years, loc, scale))


def basin(basin_id: str, meta, gesla_dir) -> list[dict]:
    d = site_path("results_dir") / basin_id
    gauges = basin_gauges(d, meta)
    with xr.open_dataset(
        d / "preprocessing_inputs" / "forcing" / "gtsm_hourly.nc"
    ) as gtsm:
        gtsm = gtsm.load()
    screen, _ = (
        screen_basin(basin_id, gauges, gtsm, gesla_dir)
        if len(gauges)
        else (pd.DataFrame(), {})
    )
    # A sub-hour timing lag barely moves an annual maximum: such gauges count here.
    usable = (
        screen[
            (screen["status"] == "usable")
            | (
                screen["reason"].fillna("").str.fullmatch(r"timing lag -?\d+ min")
                & (screen["lag_min"].abs() < 60)
            )
        ]
        if len(screen)
        else screen
    )
    rows, am_all = [], {}
    for _, g in usable.iterrows():
        s = load_on_hour(gesla_dir / g["file_name"])
        wl = (
            gtsm["waterlevel"]
            .isel(station=int(g["gtsm_station"]))
            .to_series()
            .astype(float)
        )
        am = annual_maxima(s, wl, complete_years(s))
        if len(am) < 5:
            continue
        am_all[g["site_name"]] = am
        rows.append(
            {
                "basin_id": basin_id,
                "site_name": g["site_name"],
                "gtsm_station": int(g["gtsm_station"]),
                "gtsm_station_km": g["gtsm_station_km"],
                "n_years": int(len(am)),
                "am_gauge_median_m": float(am["g"].median()),
                "am_gtsm_median_m": float(am["m"].median()),
                "am_bias_median_m": float((am["g"] - am["m"]).median()),
                f"rl{RP_YEARS:g}_gauge_m": gumbel_rl(am["g"].to_numpy()),
                f"rl{RP_YEARS:g}_gtsm_m": gumbel_rl(am["m"].to_numpy()),
            }
        )
        rows[-1][f"rl{RP_YEARS:g}_bias_m"] = (
            rows[-1][f"rl{RP_YEARS:g}_gauge_m"] - rows[-1][f"rl{RP_YEARS:g}_gtsm_m"]
        )
    if not rows:
        rows.append({"basin_id": basin_id, "site_name": "no usable gauge"})
    _figure(basin_id, am_all)
    return rows


def _figure(basin_id: str, am_all: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    lim = [0.0, 0.5]
    for name, am in am_all.items():
        ax.plot(am["m"], am["g"], "o", ms=4, label=f"{name} ({len(am)} yr)")
        lim = [min(lim[0], am.min().min()), max(lim[1], am.max().max())]
    ax.plot(lim, lim, "k-", lw=0.8)
    for off in (-0.3, 0.3):
        ax.plot(lim, [v + off for v in lim], "k:", lw=0.8)
    ax.set_xlabel("GTSM annual maximum above the annual mean (m)")
    ax.set_ylabel("GESLA annual maximum above the annual mean (m)")
    ax.set_title(
        f"phase6 6.2 | {basin_id} | storm tide, GESLA vs GTSM (dotted: +-0.3 m)",
        fontsize=9,
    )
    if am_all:
        ax.legend(fontsize=7)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(analysis_path(basin_id, __file__, "annual_maxima.png"), dpi=120)
    plt.close(fig)


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("basin_ids", nargs="*", default=PHASE1_BASINS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    gesla_dir = site_path("raw_data_root") / "GESLA_v41"
    meta = read_metadata(gesla_dir / "GESLA4-1_ALL.csv")
    rows = [r for b in args.basin_ids for r in basin(str(b), meta, gesla_dir)]
    print(
        write_summary(rows, analysis_path(None, __file__, "summary.csv")).to_string(
            index=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
