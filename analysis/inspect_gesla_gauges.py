"""Screen GESLA v4.1 tide gauges against the GTSM surge forcing of each delta basin.

Backs the "No tide-gauge bias correction" note in docs/methods.md (Forcing):
for every basin with a gtsm_hourly.nc it counts the GESLA gauges within 50 km
of the domain (inside it or not, by type), keeps open-coast gauges outside the
domain with >= 10 complete years in 1950-2024, and screens each against its
nearest GTSM station: duplicate sites, marsh/river networks (CRMS, USGS),
datum steps in the monthly gauge - GTSM difference, hourly timing lag and
correlation of the mean-removed levels. Heights stay on each gauge's own
datum, so only variability is compared. Writes one figure per basin to
<results_dir>/<basin>/figures/analysis/90_inspect_gesla_gauges_screen.png (with its caption) and
the cross-basin inspect_gesla_gauges_screen.csv and _coverage.csv to
<results_dir>/figures/analysis/.

GESLA files are read from <raw_data_root>/GESLA_v41/ (station files and
GESLA4-1_ALL.csv).

Usage:
    python analysis/inspect_gesla_gauges.py              # all basins with gtsm_hourly.nc
    python analysis/inspect_gesla_gauges.py 1248635 ...  # only these basins
"""

import logging
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from _common import REPO_ROOT, analysis_path, script_parser, site_path
from scipy import stats

sys.path.insert(0, str(REPO_ROOT / "workflow"))
from matplotlib.lines import Line2D  # noqa: E402
from src.domain import load_domain  # noqa: E402
from src.figure_style import (  # noqa: E402
    OKABE_ITO,
    figure_size,
    panel_letters,
    save_figure,
    style_context,
    write_caption,
)

log = logging.getLogger(__name__)

GTSM_PERIOD = ("1950-01-01", "2024-12-31")
SEARCH_KM = 50.0
MIN_YEARS = 10
MAX_STATION_KM = 50.0
INLAND_NETWORKS = (
    "CRMS",
    "USGS",
)  # marsh and river monitoring networks, not open coast
MIN_STEP_M = 0.05
MAX_LAG_HR = 6
DIFF_LIM_M = 0.3  # fixed y range of the gauge-minus-GTSM panels

# The metadata CSV header merges the last two names: 24 names for 25 fields.
METADATA_COLUMNS = [
    "file_name",
    "site_name",
    "site_code",
    "country",
    "contributor",
    "contributor_full",
    "contributor_website",
    "contributor_contact",
    "originator",
    "originator_website",
    "originator_contact",
    "latitude",
    "longitude",
    "coordinate_system",
    "start",
    "end",
    "n_years",
    "time_zone_hr",
    "datum_information",
    "instrument",
    "precision",
    "null_value",
    "gauge_type",
    "record_quality",
    "download_link",
]


def read_metadata(path: Path) -> gpd.GeoDataFrame:
    """Return one row per GESLA station file (EPSG:4326)."""
    meta = pd.read_csv(path, header=None, skiprows=1, names=METADATA_COLUMNS, dtype=str)
    meta = meta.apply(lambda s: s.str.strip())
    if not meta["gauge_type"].isin(("Coastal", "River", "Lake")).all():
        raise ValueError(f"{path}: column layout differs from METADATA_COLUMNS")
    for col in ("latitude", "longitude"):
        meta[col] = pd.to_numeric(meta[col], errors="coerce")
    return gpd.GeoDataFrame(
        meta,
        geometry=gpd.points_from_xy(meta["longitude"], meta["latitude"]),
        crs="EPSG:4326",
    )


def load_on_hour(path: Path) -> pd.Series:
    """Return the on-the-hour sea level (m, UTC) of one station file, use-flag 1 only.

    On-the-hour samples, not hourly means: GTSM output is instantaneous, and an
    hourly mean labelled at the hour start is centred 30 min later.
    """
    header = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            for key in ("TIME ZONE HOURS", "NULL VALUE", "COLUMN 3"):
                if line[1:].strip().startswith(key):
                    header[key] = line[1:].strip()[len(key) :].strip()
    if "(m)" not in header.get("COLUMN 3", ""):
        raise ValueError(f"{path}: column 3 is not in metres")
    data = pd.read_csv(
        path,
        comment="#",
        sep=r"\s+",
        header=None,
        names=["date", "time", "level", "qc", "use"],
        dtype={"date": str, "time": str, "level": float},
    )
    data = data[
        ~np.isclose(data["level"], float(header["NULL VALUE"])) & (data["use"] == 1)
    ]
    stamp = pd.to_datetime(
        data["date"] + " " + data["time"], format="%Y/%m/%d %H:%M:%S"
    )
    stamp = stamp - pd.Timedelta(hours=float(header.get("TIME ZONE HOURS") or 0))
    s = pd.Series(data["level"].to_numpy(), index=stamp.to_numpy())
    s = s[s.index == s.index.floor("h")]
    return s[~s.index.duplicated()].sort_index()


def complete_years(s: pd.Series, period=GTSM_PERIOD, frac: float = 0.9) -> list[int]:
    """Return the calendar years in period with >= frac of their hours present."""
    s = s.loc[period[0] : period[1]]
    n = s.groupby(s.index.year).size()
    hours = {
        y: 8784 if pd.Timestamp(year=y, month=1, day=1).is_leap_year else 8760
        for y in n.index
    }
    return [int(y) for y, c in n.items() if c >= frac * hours[y]]


def great_circle_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    a, b, c, d = map(np.radians, (lat1, lon1, lat2, lon2))
    return 6371.0 * np.arccos(
        np.clip(np.sin(a) * np.sin(c) + np.cos(a) * np.cos(c) * np.cos(b - d), -1, 1)
    )


def monthly_difference(gauge: pd.Series, model: pd.Series) -> pd.Series:
    """Return monthly means of gauge - model over common hours (months >= 90 % complete)."""
    diff = (gauge - model).dropna()
    month = diff.resample("MS")
    n = month.count()
    return month.mean()[n >= 0.9 * n.index.days_in_month * 24]


def detect_steps(
    monthly: pd.Series, min_segment: int = 24, alpha: float = 0.01
) -> tuple[list[dict], float]:
    """Return datum steps (OLS trend + greedy Heaviside steps) and the trend (m/yr).

    The monthly climatology is removed first; standard errors are inflated
    for lag-1 autocorrelation of the trend-only residuals. A step must be
    >= MIN_STEP_M, p < alpha and >= min_segment months from ends and other steps.
    """
    n = len(monthly)
    if n < 2 * min_segment:
        return [], np.nan
    y = (monthly - monthly.groupby(monthly.index.month).transform("mean")).to_numpy()
    t = (monthly.index - monthly.index[0]).days.to_numpy() / 365.25
    base = np.column_stack([np.ones(n), t])

    def ols(x):
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        r = y - x @ beta
        cov = np.linalg.inv(x.T @ x) * (r @ r / max(n - x.shape[1], 1))
        return beta, np.sqrt(np.diag(cov)), r

    _, _, r0 = ols(base)
    r1 = float(np.clip(np.corrcoef(r0[:-1], r0[1:])[0, 1], 0.0, 0.95))
    inflate = np.sqrt((1 + r1) / (1 - r1))

    def design(ks):
        return np.column_stack([base] + [(np.arange(n) >= k).astype(float) for k in ks])

    found: list[int] = []
    while len(found) < 3:
        best = None
        for k in range(min_segment, n - min_segment + 1):
            if any(abs(k - f) < min_segment for f in found):
                continue
            beta, se, _ = ols(design(found + [k]))
            tval = beta[-1] / (se[-1] * inflate)
            if abs(beta[-1]) >= MIN_STEP_M and (
                best is None or abs(tval) > abs(best[1])
            ):
                best = (k, tval)
        if best is None or 2 * stats.t.sf(abs(best[1]), n - 3 - len(found)) >= alpha:
            break
        found.append(best[0])
    found.sort()
    beta, se, _ = ols(design(found))
    steps = [
        {
            "date": monthly.index[k],
            "step_m": float(beta[2 + i]),
            "step_se_m": float(se[2 + i] * inflate),
        }
        for i, k in enumerate(found)
    ]
    return steps, float(beta[1])


def lag_minutes(gauge: pd.Series, model: pd.Series) -> float:
    """Return the lag (min, gauge after model > 0) maximising the hourly cross-correlation."""
    both = pd.concat([gauge.rename("g"), model.rename("m")], axis=1).asfreq("1h")
    corr = np.array(
        [both["g"].corr(both["m"].shift(k)) for k in range(-MAX_LAG_HR, MAX_LAG_HR + 1)]
    )
    if not np.isfinite(corr).any():
        return np.nan
    i = int(np.nanargmax(corr))
    off = 0.0
    if 0 < i < len(corr) - 1 and np.isfinite(corr[[i - 1, i + 1]]).all():
        den = corr[i - 1] - 2 * corr[i] + corr[i + 1]
        off = 0.5 * (corr[i - 1] - corr[i + 1]) / den if den else 0.0
    return float((i - MAX_LAG_HR + off) * 60.0)


def basin_gauges(basin_dir: Path, meta: gpd.GeoDataFrame) -> pd.DataFrame:
    """Return the gauges within SEARCH_KM of the domain or inside it, with distance and inside flag."""
    dom = basin_dir / "preprocessing_inputs" / "domain"
    _, utm, polygon = load_domain(
        dom / "domain_bbox.json", dom / f"{basin_dir.name}_domain.gpkg"
    )
    poly = gpd.GeoSeries([polygon], crs="EPSG:4326").to_crs(utm).iloc[0]
    near = meta.cx[
        polygon.bounds[0] - 1 : polygon.bounds[2] + 1,
        polygon.bounds[1] - 1 : polygon.bounds[3] + 1,
    ]
    if near.empty:
        return pd.DataFrame()
    g = near.to_crs(utm).geometry
    inside = g.within(poly).to_numpy()
    dist = np.where(inside, 0.0, g.distance(poly.exterior).to_numpy() / 1000.0)
    out = pd.DataFrame(near.drop(columns="geometry")).assign(
        inside_domain=inside, dist_to_domain_km=dist
    )
    return out[out["dist_to_domain_km"] <= SEARCH_KM]


def screen_basin(
    basin: str, gauges: pd.DataFrame, gtsm: xr.Dataset, gesla_dir: Path
) -> tuple[pd.DataFrame, dict]:
    """Return one row per open-coast candidate with its screening result, and monthly differences."""
    cand = gauges[~gauges["inside_domain"] & (gauges["gauge_type"] == "Coastal")].copy()
    rows, monthly = [], {}
    series = {}
    for _, g in cand.iterrows():
        s = load_on_hour(gesla_dir / g["file_name"])
        years = complete_years(s)
        if len(years) >= MIN_YEARS:
            series[g["file_name"]] = (s, years)
    cand = cand[cand["file_name"].isin(series)].copy()
    cand["n_years"] = cand["file_name"].map(lambda f: len(series[f][1]))
    kept: list[tuple[float, float, str]] = []
    for _, g in cand.sort_values("n_years", ascending=False, kind="stable").iterrows():
        s, years = series[g["file_name"]]
        row = {
            "basin": basin,
            "file_name": g["file_name"],
            "site_name": g["site_name"],
            "contributor": g["contributor"],
            "dist_to_domain_km": round(g["dist_to_domain_km"], 1),
            "n_complete_years": len(years),
            "record_quality": g["record_quality"],
            "datum_information": g["datum_information"],
        }
        dup = next(
            (
                f
                for la, lo, f in kept
                if great_circle_km(g["latitude"], g["longitude"], la, lo) <= 1.0
            ),
            "",
        )
        d = great_circle_km(
            g["latitude"],
            g["longitude"],
            gtsm["latitude"].values,
            gtsm["longitude"].values,
        )
        st = int(np.argmin(d))
        row.update(gtsm_station=st, gtsm_station_km=round(float(d[st]), 1))
        if dup:
            row.update(status="excluded", reason=f"duplicate site of {dup}")
        elif g["contributor"] in INLAND_NETWORKS:
            row.update(
                status="excluded",
                reason=f"{g['contributor']}: marsh/river network, not open coast",
            )
        elif d[st] > MAX_STATION_KM:
            row.update(status="excluded", reason=f"nearest GTSM station {d[st]:.0f} km")
        else:
            kept.append((g["latitude"], g["longitude"], g["file_name"]))
            wl = gtsm["waterlevel"].isel(station=st).to_series().astype(float)
            gs = s[s.index.year.isin(years)]
            md = monthly_difference(gs, wl)
            steps, trend = detect_steps(md)
            monthly[g["file_name"]] = (md, steps)
            if steps:  # longest step-free run
                edges = (
                    [pd.Timestamp.min] + [x["date"] for x in steps] + [pd.Timestamp.max]
                )
                runs = [
                    [
                        y
                        for y in years
                        if lo <= pd.Timestamp(year=y, month=1, day=1)
                        and pd.Timestamp(year=y, month=12, day=31) < hi
                    ]
                    for lo, hi in zip(edges[:-1], edges[1:])
                ]
                years = max(runs, key=len)
                gs = s[s.index.year.isin(years)]
            ga = gs - gs.groupby(gs.index.year).transform("mean")
            ma = wl[wl.index.year.isin(years)]
            ma = ma - ma.groupby(ma.index.year).transform("mean")
            both = pd.concat([ga.rename("g"), ma.rename("m")], axis=1).dropna()
            lag = lag_minutes(both["g"], both["m"])
            row.update(
                n_years_used=len(years),
                lag_min=round(lag, 1),
                corr=round(float(both["g"].corr(both["m"])), 3),
                rmse_m=round(float(np.sqrt(((both["g"] - both["m"]) ** 2).mean())), 3),
                trend_mm_per_yr=round(trend * 1000, 1),
                steps="; ".join(
                    f"{x['date']:%Y-%m} {x['step_m'] * 100:+.0f}+-{x['step_se_m'] * 100:.0f} cm"
                    for x in steps
                ),
            )
            fails = []
            if round(lag / 60.0) != 0:
                fails.append(f"timing lag {lag:.0f} min")
            if len(years) < MIN_YEARS:
                fails.append(f"{len(years)} step-free complete years")
            row.update(
                status="excluded" if fails else "usable", reason="; ".join(fails)
            )
        rows.append(row)
    return pd.DataFrame(rows), monthly


def coverage_row(basin: str, gauges: pd.DataFrame, screen: pd.DataFrame) -> dict:
    """Return the gauge counts and the best open-coast record of one basin."""
    open_coast = (
        screen
        if not screen.empty
        else pd.DataFrame(columns=["n_complete_years", "status"])
    )
    return {
        "basin": basin,
        "gauges_within_50km": len(gauges),
        "inside_domain": int(gauges["inside_domain"].sum()) if len(gauges) else 0,
        "river_or_lake": int((gauges["gauge_type"] != "Coastal").sum())
        if len(gauges)
        else 0,
        "open_coast_ge_10yr": len(open_coast),
        "usable_after_screen": int((open_coast["status"] == "usable").sum()),
        "best_open_coast_years": int(open_coast["n_complete_years"].max())
        if len(open_coast)
        else 0,
    }


def plot_basin(basin: str, screen: pd.DataFrame, monthly: dict, out_png: Path) -> None:
    """One panel per screened gauge: monthly gauge - GTSM (mean removed) with the detected steps.

    Panels share both axes; the y axis is fixed at +-DIFF_LIM_M and values
    beyond it are clipped at the edge (counted in the caption).
    """
    if screen.empty or not monthly:
        return
    rows = screen[screen["file_name"].isin(monthly)]
    if rows.empty:
        return
    step_colour = OKABE_ITO[6]
    n_clipped = 0
    with style_context():
        fig, axes = plt.subplots(
            len(rows),
            1,
            figsize=figure_size(35 * len(rows) + 20),
            sharex=True,
            sharey=True,
            squeeze=False,
            layout="constrained",
        )
        for ax, (_, r) in zip(axes[:, 0], rows.iterrows()):
            md, steps = monthly[r["file_name"]]
            anomaly = md - md.mean()
            n_clipped += int((anomaly.abs() > DIFF_LIM_M).sum())
            ax.axhline(0.0, color="0.7", linewidth=0.5)
            ax.plot(
                anomaly.index,
                anomaly.clip(-DIFF_LIM_M, DIFF_LIM_M),
                color="black",
                linewidth=0.8,
            )
            for x in steps:
                ax.axvline(x["date"], color=step_colour, linestyle="--", linewidth=1.0)
            ax.set_ylim(-DIFF_LIM_M, DIFF_LIM_M)
            ax.set_ylabel("Gauge minus GTSM (m)")
            # small-multiple row header: site and screening status
            ax.text(
                0.01,
                0.97,
                f"{r['site_name']}, {r['status']}",
                transform=ax.transAxes,
                ha="left",
                va="top",
            )
        if len(rows) > 1:
            panel_letters(axes[:, 0])
        handles = [
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=0.8,
                label="Monthly gauge minus GTSM, mean removed",
            ),
        ]
        if any(monthly[f][1] for f in rows["file_name"]):
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=step_colour,
                    linestyle="--",
                    label="Detected datum step",
                )
            )
        fig.legend(handles=handles, loc="outside lower center", ncols=len(handles))
        save_figure(fig, out_png)
        plt.close(fig)

    letters = "abcdefghijklmnopqrstuvwxyz"

    def _line(k: int, r: pd.Series) -> str:
        head = f"({letters[k]}) " if len(rows) > 1 else ""
        reason = f" ({r['reason']})" if r["reason"] else ""
        steps = f", steps {r['steps']}" if r.get("steps") else ""
        return (
            f"{head}{r['site_name']} ({r['contributor']}): {r['status']}{reason}; "
            f"{r['n_years_used']} complete years used, timing lag {r['lag_min']:.0f} min, "
            f"r {r['corr']:.2f}, RMSE {r['rmse_m']:.3f} m, trend {r['trend_mm_per_yr']:+.1f} mm/yr{steps}."
        )

    n_usable = int((rows["status"] == "usable").sum())
    write_caption(
        out_png,
        f"{n_usable} of {len(rows)} open-coast GESLA gauge(s) near basin {basin} pass the screen "
        "against the nearest GTSM station.\n"
        "Black line: monthly mean of the hourly gauge level minus the GTSM water level, each on "
        "its own datum, with the record mean removed, so only variability is compared; thin grey "
        "line: zero; dashed vermillion lines: detected datum steps. Axis fixed at "
        f"+-{DIFF_LIM_M:g} m"
        + (f"; {n_clipped} month(s) beyond it drawn at the edge" if n_clipped else "")
        + ".\n"
        + "\n".join(_line(k, r) for k, (_, r) in enumerate(rows.iterrows())),
    )


def main() -> None:
    """Screen the requested basins and print the coverage table."""
    parser = script_parser(__doc__)
    parser.add_argument(
        "basin_ids",
        nargs="*",
        help="only these basin ids (default: all with gtsm_hourly.nc)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    results_dir = site_path("results_dir")
    gesla_dir = site_path("raw_data_root") / "GESLA_v41"
    meta = read_metadata(gesla_dir / "GESLA4-1_ALL.csv")
    basins = [
        d
        for d in sorted(results_dir.iterdir())
        if (d / "preprocessing_inputs" / "forcing" / "gtsm_hourly.nc").exists()
        and (not args.basin_ids or d.name in args.basin_ids)
    ]

    screens, coverage = [], []
    for d in basins:
        gauges = basin_gauges(d, meta)
        with xr.open_dataset(
            d / "preprocessing_inputs" / "forcing" / "gtsm_hourly.nc"
        ) as gtsm:
            screen, monthly = (
                screen_basin(d.name, gauges, gtsm.load(), gesla_dir)
                if len(gauges)
                else (pd.DataFrame(), {})
            )
        coverage.append(coverage_row(d.name, gauges, screen))
        screens.append(screen)
        plot_basin(
            d.name, screen, monthly, analysis_path(d.name, __file__, "screen.png")
        )
        log.info(f"{d.name}: {coverage[-1]}")
    pd.concat(screens, ignore_index=True).to_csv(
        analysis_path(None, __file__, "screen.csv"), index=False
    )
    cov = pd.DataFrame(coverage)
    cov.to_csv(analysis_path(None, __file__, "coverage.csv"), index=False)
    print(cov.to_string(index=False))


if __name__ == "__main__":
    main()
