"""Why is a delta's stylised surge hydrograph so wide?

Diagnostic for tests/KL_gtsm_storm_tide.py's surge shape (HGRAPHER): the shape
is the average, over the largest surge peaks of the GTSM reanalysis, of the
time each event spends above every fraction of its own peak within +-36 h. A
wide shape therefore means those events stay near their peak for long -- this
script finds out why, for the GTSM stations within RADIUS_KM of one delta:

  1  peak vs duration      every 72 h-separated surge peak: its height against
                           the time it stays above 50 % of that height
  2  water level           the same for the peak TOTAL water level of the
                           events the shape is built from
  3  counting              HGRAPHER counts all time above a level in the
                           window; compared with the unbroken spell around
                           the peak (secondary bumps)
  4  composite             median/IQR of the selected events' surge (m), 10
                           days either side -- does it return to zero?
  5  background            surge level 4-10 days away from each event, as a
                           fraction of its peak
  6  seasonal cycle        monthly mean surge residual
  7  timing                month the selected events fall in
  8  shape variants        the stored shape against the same method with the
                           slow background removed, the seasonal cycle
                           removed, and a +-120 h window (no cut at 36 h)
  9  selection             shape for stricter / looser peak selection
  10 events                the largest events themselves, with dates

Usage:  python tests/check_surge_shape_duration.py [BASIN_ID]     (default 620947, Calvert)
        python tests/check_surge_shape_duration.py --extract-only  (only build the station cache)
Reads the hourly reanalysis from GCFM_RAW_DATA_ROOT/GTSM_storm_tide_hourly
(unzipped {storm_surge_residual,total_water_level}/ folders, else the yearly
zips). Like KL_gtsm_storm_tide.py's stage 1, the slow pass over the global
monthly files happens once, for EVERY station of the stored hydrograph file
(all deltas), into one cache file per variable and year under
GCFM_RESULTS_DIR/diagnostics/surge_shape/_cache/ -- an interrupted run resumes
at the first missing year, and every delta afterwards takes seconds. Figure,
events.csv and summary.csv go to .../surge_shape/{BASIN_ID}/.
"""

import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as ss
import xarray as xr
import yaml
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import KL_gtsm_storm_tide as hg  # the method under test -- its own functions are reused as-is

EXTRACT_ONLY = "--extract-only" in sys.argv
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
BASIN_ID = int(_args[0]) if _args else 620947
RADIUS_KM = 100.0  # as KL_gtsm_storm_tide.RADIUS_KM

# Machine-specific paths, read from environment variables (see
# CONTRIBUTING.md "Local machine paths").
RAW = Path(os.environ["GCFM_RAW_DATA_ROOT"])
GTSM_DIR = RAW / "GTSM_storm_tide_hourly"
DIAG_DIR = Path(os.environ["GCFM_RESULTS_DIR"]) / "diagnostics" / "surge_shape"
OUT_DIR = DIAG_DIR / str(BASIN_ID)
CACHE_DIR = DIAG_DIR / "_cache"
CATALOGUE = Path(__file__).parents[1] / "config" / "data_catalogue.yml"

STEPS = hg.STEPS_PER_HR  # 10-min steps per hour
LEVELS = hg.LEVELS
FAR_DAYS = (4, 10)  # "background" = surge this many days before/after an event
BACKGROUND_WINDOW_HR = (
    30 * 24
)  # running-median window of the "slow background removed" variant
WIDE_WINDOW_HR = 120
VARS = {
    "waterlevel": ("total_water_level", "reanalysis_waterlevel_hourly"),
    "surge": ("storm_surge_residual", "reanalysis_surge_hourly"),
}

# Chart colours: reference categorical palette, fixed order.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e6e5e1"


# ── data ──────────────────────────────────────────────────────────────────────
def delta_polygon():
    with open(CATALOGUE, encoding="utf-8") as f:
        cat = yaml.safe_load(f)
    entry = next(d for d in cat["datasets"] if d["name"] == "delta_polygons")
    path = RAW / entry["file_path"]
    if not path.exists():
        path = path.with_suffix(".geojson")
    deltas = gpd.read_file(path).to_crs(4326)
    sel = deltas[deltas["BasinID2"].astype(int) == BASIN_ID]
    if sel.empty:
        raise SystemExit(f"basin {BASIN_ID} not in {path}")
    return sel.geometry.iloc[0]


def delta_stations(stored):
    """Stations of the stored hydrograph file within RADIUS_KM of the delta, nearest first."""
    geom = gpd.GeoSeries([delta_polygon()], crs=4326)
    utm = geom.estimate_utm_crs()
    pts = gpd.GeoSeries(
        gpd.points_from_xy(
            stored["station_x_coordinate"].values, stored["station_y_coordinate"].values
        ),
        crs=4326,
    )
    dist = pts.to_crs(utm).distance(geom.to_crs(utm).iloc[0]).values
    order = np.argsort(dist)
    order = order[dist[order] <= RADIUS_KM * 1000]
    return stored["stations"].values[order].astype(int), dist[order] / 1000


def monthly_files(folder, prefix):
    """(label, opener) per monthly file: unzipped folder if present, else the yearly zips."""
    unzipped = sorted((GTSM_DIR / folder).glob(f"{prefix}_*.nc"))
    if unzipped:
        return [(p.name, p) for p in unzipped]
    out = []
    for z in sorted(GTSM_DIR.glob(f"{folder}_*.zip")):
        with zipfile.ZipFile(z) as zf:
            out += [(n, (z, n)) for n in sorted(zf.namelist()) if n.endswith(".nc")]
    return out


def read_stations(src, var, ids):
    def _read(path):
        with xr.open_dataset(path) as ds:
            idx = np.flatnonzero(np.isin(ds["stations"].values, ids))
            return ds[var].isel(stations=idx).load()

    if isinstance(src, Path):
        return _read(src)
    z, name = src
    with zipfile.ZipFile(z) as zf, tempfile.TemporaryDirectory() as tmp:
        return _read(zf.extract(name, tmp))


def build_cache(all_ids):
    """One cache file per variable and year holding every stored station (all deltas).
    Years already cached are skipped, so an interrupted run resumes where it stopped.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for var, (folder, prefix) in VARS.items():
        files = monthly_files(folder, prefix)
        if not files:
            raise SystemExit(f"no {folder} files under {GTSM_DIR}")
        by_year = {}
        for label, src in files:
            by_year.setdefault(
                int(re.search(r"_(\d{4})_\d{2}_", label).group(1)), []
            ).append(src)
        todo = [
            y for y in sorted(by_year) if not (CACHE_DIR / f"{var}_{y}.nc").exists()
        ]
        if todo:
            print(
                f"{var}: extracting {len(todo)} of {len(by_year)} year(s), {len(all_ids)} stations"
            )
        for year in todo:
            if len(by_year[year]) != 12:
                print(
                    f"  {var} {year}: {len(by_year[year])} monthly file(s), not 12 -- skipped (incomplete download?)"
                )
                continue
            data = xr.concat(
                [read_stations(src, var, all_ids) for src in by_year[year]], "time"
            ).astype("float32")
            tmp = CACHE_DIR / f"{var}_{year}.nc.part"
            data.to_netcdf(tmp)
            tmp.replace(CACHE_DIR / f"{var}_{year}.nc")
            print(f"  cached {var}_{year}.nc")


def load_series(ids, all_ids):
    """Hourly waterlevel + surge of the given stations, from the all-stations year cache."""
    build_cache(all_ids)
    years = sorted(
        int(p.stem.rsplit("_", 1)[1])
        for p in CACHE_DIR.glob("surge_*.nc")
        if (CACHE_DIR / p.name.replace("surge_", "waterlevel_")).exists()
    )
    if not years:
        raise SystemExit(f"no complete years cached under {CACHE_DIR}")
    print(f"{len(years)} cached year(s): {years[0]}-{years[-1]}")
    parts = []
    for var in VARS:
        with xr.open_mfdataset(
            [CACHE_DIR / f"{var}_{y}.nc" for y in years], combine="by_coords"
        ) as ds:
            parts.append(ds[var].sel(stations=ids).load())
    return xr.merge(parts, compat="override")


# ── the method, generalised (window, selection) ───────────────────────────────
def duration_curves(surge10, half_hr=36, quantile=hg.SURGE_PERCENTILE):
    """hg.surge_hydrograph with the window half-width and the peak quantile as arguments."""
    half = int(half_hr * STEPS)
    peaks, props = ss.find_peaks(surge10, distance=72 * STEPS, height=-10)
    peaks = peaks[props["peak_heights"] >= np.quantile(props["peak_heights"], quantile)]
    before, after = [], []
    for p in peaks:
        if p - half < 0 or p + half >= len(surge10) or surge10[p] <= 0:
            continue
        b = surge10[p - half : p + 1] / surge10[p]
        a = surge10[p : p + half + 1] / surge10[p]
        neg_b, neg_a = np.where(b < 0)[0], np.where(a < 0)[0]
        b = b[neg_b[-1] :] if neg_b.size else b[1:]
        a = a[: neg_a[0]] if neg_a.size else a[:half]
        before.append((b[:, None] > LEVELS).sum(axis=0))
        after.append((a[:, None] > LEVELS).sum(axis=0))
    return np.mean(before, axis=0), np.mean(after, axis=0), len(before)


def shape_on_grid(dur_before, dur_after, half_hr=150):
    """hg.surge_shape on a wider grid: (hours, normalised surge)."""
    steps = np.arange(-half_hr * STEPS, half_hr * STEPS + 1)
    rise = np.interp(np.abs(steps), dur_before[::-1], LEVELS[::-1], right=0.0)
    fall = np.interp(np.abs(steps), dur_after[::-1], LEVELS[::-1], right=0.0)
    return steps / STEPS, np.where(steps <= 0, rise, fall)


def width_hr(hours, shape, level):
    return float((shape > level).sum()) / STEPS


# ── per station ───────────────────────────────────────────────────────────────
def prepare(twl, surge):
    """Gap filling and mean-sea-level removal exactly as hg.stage2."""
    twl = twl.interpolate(limit=hg.MAX_GAP_HR, limit_area="inside")
    surge = surge.interpolate(limit=hg.MAX_GAP_HR, limit_area="inside")
    if twl.isna().any() or surge.isna().any():
        return None, None
    msl = (
        (twl - surge)
        .rolling(hg.MSL_WINDOW_HR, center=True, min_periods=hg.MSL_WINDOW_HR // 2)
        .mean()
    )
    return twl - msl, surge


def event_table(surge, twl):
    """One row per 72 h-separated surge peak that has 10 days of data either side."""
    s10 = hg.to_10min(surge)
    t10 = pd.date_range(surge.index[0], periods=len(s10), freq="10min")
    half, far0, far1 = 36 * STEPS, FAR_DAYS[0] * 24 * STEPS, FAR_DAYS[1] * 24 * STEPS
    peaks, props = ss.find_peaks(s10, distance=72 * STEPS, height=-10)
    threshold = np.quantile(props["peak_heights"], hg.SURGE_PERCENTILE)
    rows = []
    for p in peaks:
        pk = s10[p]
        if pk <= 0 or p - far1 < 0 or p + far1 >= len(s10):
            continue
        b, a = s10[p - half : p + 1] / pk, s10[p : p + half + 1] / pk
        neg_b, neg_a = np.where(b < 0)[0], np.where(a < 0)[0]
        b = b[neg_b[-1] :] if neg_b.size else b[1:]
        a = a[: neg_a[0]] if neg_a.size else a[:half]
        # unbroken spell above half the peak, not limited to the +-36 h window
        below_b = np.where(s10[p - far1 : p + 1][::-1] < 0.5 * pk)[0]
        below_a = np.where(s10[p : p + far1 + 1] < 0.5 * pk)[0]
        spell = (below_b[0] if below_b.size else far1) + (
            below_a[0] if below_a.size else far1
        )
        far = np.concatenate([s10[p - far1 : p - far0], s10[p + far0 : p + far1]])
        tp = t10[p]
        rows.append(
            {
                "time": tp,
                "month": tp.month,
                "surge_peak_m": pk,
                "selected": bool(pk >= threshold),
                "dur50_hr": ((b > 0.5).sum() + (a > 0.5).sum()) / STEPS,
                "dur25_hr": ((b > 0.25).sum() + (a > 0.25).sum()) / STEPS,
                "spell50_hr": spell / STEPS,
                "background_m": float(np.median(far)),
                "edge_frac": float(np.mean([s10[p - half], s10[p + half]]) / pk),
                "twl_peak_m": float(
                    twl.loc[tp - pd.Timedelta("6h") : tp + pd.Timedelta("6h")].max()
                ),
                "i10": p,
            }
        )
    return pd.DataFrame(rows), s10, threshold


def analyse(twl, surge, stored_shape):
    events, s10, threshold = event_table(surge, twl)
    variants = {}

    dur_b, dur_a, n = hg.surge_hydrograph(s10)
    own_b, own_a, _ = duration_curves(s10)
    assert np.allclose(dur_b, own_b) and np.allclose(dur_a, own_a), (
        "duration_curves != hg.surge_hydrograph"
    )
    reproduced = hg.surge_shape(dur_b, dur_a)
    repro_err = float(np.abs(reproduced - stored_shape).max())
    variants["as stored"] = shape_on_grid(dur_b, dur_a)

    background = surge.rolling(
        BACKGROUND_WINDOW_HR, center=True, min_periods=BACKGROUND_WINDOW_HR // 2
    ).median()
    variants["30-day background removed"] = shape_on_grid(
        *duration_curves(hg.to_10min(surge - background))[:2]
    )
    climatology = surge.groupby(surge.index.month).transform("mean")
    variants["seasonal cycle removed"] = shape_on_grid(
        *duration_curves(hg.to_10min(surge - climatology))[:2]
    )
    variants[f"window ±{WIDE_WINDOW_HR} h"] = shape_on_grid(
        *duration_curves(s10, half_hr=WIDE_WINDOW_HR)[:2]
    )

    selection = {}
    for q, label in ((0.95, "top 5 %"), (0.99, "top 1 % (used)"), (0.998, "top 0.2 %")):
        b, a, k = duration_curves(s10, quantile=q)
        selection[f"{label}, {k} events"] = shape_on_grid(b, a)

    sel = events[events["selected"]]
    big = events[events["surge_peak_m"] >= events["surge_peak_m"].median()]
    summary = (
        {
            "n_selected": len(sel),
            "selection_threshold_m": threshold,
            "reproduction_max_abs_diff": repro_err,
            "spearman_peak_vs_dur50_all": spearmanr(
                big["surge_peak_m"], big["dur50_hr"]
            ).statistic,
            "spearman_peak_vs_dur50_selected": spearmanr(
                sel["surge_peak_m"], sel["dur50_hr"]
            ).statistic,
            "dur50_selected_median_hr": sel["dur50_hr"].median(),
            "spell50_selected_median_hr": sel["spell50_hr"].median(),
            "background_over_peak_median": (
                sel["background_m"] / sel["surge_peak_m"]
            ).median(),
            "selected_with_edge_above_25pct": float((sel["edge_frac"] > 0.25).mean()),
            "surge_mean_m": float(surge.mean()),
            "seasonal_range_m": float(
                surge.groupby(surge.index.month)
                .mean()
                .pipe(lambda m: m.max() - m.min())
            ),
            "selected_in_top3_months": float(
                sel["month"].isin(sel["month"].value_counts().index[:3]).mean()
            ),
        }
        | {f"w50_hr [{k}]": width_hr(*v, 0.5) for k, v in variants.items()}
        | {
            f"w50_hr [{k.split(',')[0]}]": width_hr(*v, 0.5)
            for k, v in selection.items()
        }
    )
    return events, s10, variants, selection, summary


# ── figure ────────────────────────────────────────────────────────────────────
def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=9.5, loc="left", color=INK)
    ax.set_xlabel(xlabel, fontsize=8.5, color=INK2)
    ax.set_ylabel(ylabel, fontsize=8.5, color=INK2)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(labelsize=8, colors=INK2)
    ax.set_facecolor(SURFACE)


def figure(sid, km, surge, events, s10, variants, selection, summary):
    sel, rest = events[events["selected"]], events[~events["selected"]]
    fig, axes = plt.subplots(4, 3, figsize=(16, 17), facecolor=SURFACE)
    ax = axes.ravel()
    months = np.arange(1, 13)
    month_lbl = list("JFMAMJJASOND")

    ax[0].scatter(
        rest["surge_peak_m"],
        rest["dur50_hr"],
        s=7,
        color=MUTED,
        alpha=0.35,
        lw=0,
        label="other peaks",
    )
    ax[0].scatter(
        sel["surge_peak_m"],
        sel["dur50_hr"],
        s=22,
        color=BLUE,
        edgecolor=SURFACE,
        lw=0.6,
        label=f"selected for the shape ({len(sel)})",
    )
    ax[0].axhline(72, color=MUTED, lw=0.8, ls=":")
    ax[0].annotate(
        "window limit 72 h",
        (ax[0].get_xlim()[1], 72),
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=INK2,
    )
    style(
        ax[0],
        "1  Peak surge vs time above half the peak",
        "peak surge residual (m)",
        "hours above 50 % of own peak (±36 h)",
    )
    ax[0].legend(frameon=False, fontsize=8, loc="center right")

    ax[1].scatter(
        sel["twl_peak_m"], sel["dur50_hr"], s=22, color=BLUE, edgecolor=SURFACE, lw=0.6
    )
    style(
        ax[1],
        "2  Peak total water level vs duration (selected events)",
        "peak total water level, mean sea level removed (m)",
        "hours above 50 % of own surge peak",
    )

    lim = max(events["dur50_hr"].max(), 72) * 1.02
    ax[2].plot([0, lim], [0, lim], color=MUTED, lw=0.8)
    ax[2].scatter(
        sel["spell50_hr"].clip(upper=lim),
        sel["dur50_hr"],
        s=22,
        color=BLUE,
        edgecolor=SURFACE,
        lw=0.6,
    )
    style(
        ax[2],
        "3  Counted time vs unbroken spell above half the peak (selected)",
        "unbroken spell around the peak, no window (h)",
        "time counted by the method (h)",
    )

    span = FAR_DAYS[1] * 24 * STEPS
    hours = np.arange(-span, span + 1) / STEPS
    stack = np.array([s10[i - span : i + span + 1] for i in sel["i10"]])
    q25, q50, q75 = np.quantile(stack, [0.25, 0.5, 0.75], axis=0)
    ax[3].fill_between(
        hours / 24, q25, q75, color=BLUE, alpha=0.22, lw=0, label="interquartile range"
    )
    ax[3].plot(hours / 24, q50, color=BLUE, lw=2, label="median of selected events")
    ax[3].axhline(0, color=INK2, lw=0.8)
    for d in (-1.5, 1.5):
        ax[3].axvline(d, color=MUTED, lw=0.8, ls=":")
    style(
        ax[3],
        "4  Surge around the selected events (dotted: ±36 h window)",
        "days from the surge peak",
        "surge residual (m)",
    )
    ax[3].legend(frameon=False, fontsize=8)

    ratio = (sel["background_m"] / sel["surge_peak_m"]).clip(-0.5, 1.0)
    ax[4].hist(
        ratio, bins=np.arange(-0.5, 1.0001, 0.05), color=BLUE, edgecolor=SURFACE, lw=1
    )
    ax[4].axvline(ratio.median(), color=INK, lw=1)
    ax[4].annotate(
        f"median {ratio.median():.2f}",
        (ratio.median(), ax[4].get_ylim()[1]),
        xytext=(4, -10),
        textcoords="offset points",
        fontsize=8,
        color=INK,
    )
    style(
        ax[4],
        f"5  Surge {FAR_DAYS[0]}–{FAR_DAYS[1]} days from the event, as a share of its peak",
        "background surge / peak surge",
        "selected events",
    )

    clim = surge.groupby(surge.index.month).mean().reindex(months)
    ax[5].bar(months, clim.values, width=0.7, color=BLUE)
    ax[5].axhline(0, color=INK2, lw=0.8)
    ax[5].set_xticks(months, month_lbl)
    style(
        ax[5],
        "6  Monthly mean surge residual, 1950–2024",
        "month",
        "mean surge residual (m)",
    )

    counts = sel["month"].value_counts().reindex(months, fill_value=0)
    ax[6].bar(months, counts.values, width=0.7, color=BLUE)
    ax[6].set_xticks(months, month_lbl)
    style(ax[6], "7  Month of the selected events", "month", "selected events")

    for (label, (h, shp)), color in zip(variants.items(), (BLUE, ORANGE, AQUA, YELLOW)):
        ax[7].plot(
            h,
            shp,
            color=color,
            lw=2,
            label=f"{label} — {width_hr(h, shp, 0.5):.0f} h above 50 %",
        )
    ax[7].set_xlim(-130, 130)
    style(
        ax[7],
        "8  Surge shape: stored method vs variants",
        "hours from the peak",
        "normalised surge",
    )
    ax[7].legend(frameon=False, fontsize=7.5, loc="upper left")

    for (label, (h, shp)), color in zip(selection.items(), (BLUE, ORANGE, AQUA)):
        ax[8].plot(
            h, shp, color=color, lw=2, label=f"{label} — {width_hr(h, shp, 0.5):.0f} h"
        )
    ax[8].set_xlim(-45, 45)
    style(
        ax[8],
        "9  Surge shape by peak selection",
        "hours from the peak",
        "normalised surge",
    )
    ax[8].legend(frameon=False, fontsize=7.5, loc="lower center")

    top = sel.nlargest(8, "surge_peak_m")
    day = 6 * 24 * STEPS
    for k, (axis, part) in enumerate(((ax[9], top.iloc[:4]), (ax[10], top.iloc[4:]))):
        for (_, ev), color in zip(part.iterrows(), (BLUE, ORANGE, AQUA, YELLOW)):
            i = int(ev["i10"])
            axis.plot(
                np.arange(-day, day + 1) / STEPS / 24,
                s10[i - day : i + day + 1],
                color=color,
                lw=1.6,
                label=f"{ev['time']:%Y-%m-%d}  {ev['surge_peak_m']:.2f} m",
            )
        axis.axhline(0, color=INK2, lw=0.8)
        style(
            axis,
            f"10  Largest events, {1 + 4 * k}–{4 + 4 * k}",
            "days from the surge peak",
            "surge residual (m)",
        )
        axis.legend(frameon=False, fontsize=8)

    ax[11].axis("off")
    lines = [
        f"selected events: {summary['n_selected']} (peak ≥ {summary['selection_threshold_m']:.2f} m)",
        f"median hours above 50 %: {summary['dur50_selected_median_hr']:.0f} counted, "
        f"{summary['spell50_selected_median_hr']:.0f} unbroken",
        f"peak vs duration, rank correlation: {summary['spearman_peak_vs_dur50_all']:+.2f} (upper half of all peaks), "
        f"{summary['spearman_peak_vs_dur50_selected']:+.2f} (selected)",
        f"background / peak, median: {summary['background_over_peak_median']:.2f}",
        f"events still above 25 % of the peak at ±36 h: {100 * summary['selected_with_edge_above_25pct']:.0f} %",
        f"seasonal range of the monthly mean surge: {summary['seasonal_range_m']:.2f} m",
        f"selected events in their 3 busiest months: {100 * summary['selected_in_top3_months']:.0f} %",
        f"stored shape reproduced to {summary['reproduction_max_abs_diff']:.1e}",
    ]
    ax[11].text(
        0,
        0.98,
        "Summary\n\n" + "\n".join(lines),
        va="top",
        fontsize=9,
        color=INK,
        linespacing=1.6,
    )

    fig.suptitle(
        f"Surge shape diagnostics — basin {BASIN_ID}, GTSM station {sid} ({km:.0f} km from the delta polygon)",
        fontsize=12,
        x=0.01,
        ha="left",
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    png = OUT_DIR / f"surge_shape_diagnostics_{BASIN_ID}_station{sid}.png"
    fig.savefig(png, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", png)


def main():
    stored = xr.open_dataset(hg.OUT_NC)
    all_ids = stored["stations"].values.astype(int)
    if EXTRACT_ONLY:
        build_cache(all_ids)
        print("cache complete:", CACHE_DIR)
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids, km = delta_stations(stored)
    if not len(ids):
        raise SystemExit(
            f"no stored hydrograph station within {RADIUS_KM:g} km of basin {BASIN_ID}"
        )
    print(f"basin {BASIN_ID}: {len(ids)} station(s) within {RADIUS_KM:g} km")
    ds = load_series(ids, all_ids)

    summaries, all_events, plotted = [], [], False
    for sid, dist in zip(ids, km):
        one = ds.sel(stations=sid)
        twl, surge = prepare(
            one["waterlevel"].to_series().astype(float),
            one["surge"].to_series().astype(float),
        )
        if twl is None:
            print(f"  station {sid}: gaps longer than {hg.MAX_GAP_HR} h, skipped")
            continue
        stored_shape = stored["surge_shape"].sel(stations=sid).values
        events, s10, variants, selection, summary = analyse(twl, surge, stored_shape)
        summaries.append(
            {"station": sid, "km_from_delta": round(float(dist), 1)} | summary
        )
        all_events.append(events.drop(columns="i10").assign(station=sid))
        if not plotted:  # the station nearest the delta
            figure(sid, dist, surge, events, s10, variants, selection, summary)
            plotted = True

    pd.concat(all_events).to_csv(OUT_DIR / "events.csv", index=False)
    table = pd.DataFrame(summaries).set_index("station")
    table.to_csv(OUT_DIR / "summary.csv")
    with pd.option_context(
        "display.width",
        250,
        "display.max_columns",
        None,
        "display.float_format",
        "{:.2f}".format,
    ):
        print(table.T.to_string())
    print("wrote", OUT_DIR / "events.csv", "and summary.csv")


if __name__ == "__main__":
    main()
