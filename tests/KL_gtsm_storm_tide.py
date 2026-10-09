"""Storm-tide return levels + storm-tide hydrographs from the GTSM reanalysis

Input:  hourly CDS GTSM v3 reanalysis zips (sis-water-level-change-timeseries-cmip6),
        {storm_surge_residual,total_water_level}_{year}.zip, as downloaded by download_cds_waterlevel.py
Output: OUT_NC, laid out like COAST-RP + COAST-HG so rule 07 can read it in place of those files:
          storm_tide_rp_XXXX              storm-tide level (m, MSL) per RP, from COAST-RP (RP_SOURCE)
          pot_storm_tide_rp_XXXX          own POT/GPD fit on the GTSM reanalysis, for comparison
          hydrograph_average_tide_signal  one lunar day of the average tide (for surge_rp: Tide)
          hydrograph_spring_tide_signal   one lunar day centred on the average spring-tide maximum
          average_tide_event / spring_tide_event   +-74.5 h signals centred on high water
          surge_shape                     normalised (0-1) average surge hydrograph, peak at t=0
          storm_tide_hydrograph_rpXXXX    average tide + scaled surge, surge peak on high water
        plus a diagnostic figure for the station nearest PLOT_DELTA_ID.

Method: HGRAPHER, Dullaart et al. (2023), NHESS 23:1847, https://doi.org/10.5194/nhess-23-1847-2023
        and its code (github.com/jobdullaart/HGRAPHER, HGRAPHER_functions.py), ported to hourly data:
  mean sea level: the reanalysis total water level carries a slowly varying mean level (annual mean of
         twl - surge rises over 1950-2024); with DETREND_MSL its 1-yr running mean is removed first, so
         tide and return levels refer to a fixed MSL (as COAST-RP / HGRAPHER's SLR-removed tide)
  tide = total water level - surge residual (GTSM surge = storm tide minus tide-only run), both
         cubic-interpolated hourly -> 10 min so the step counts below match HGRAPHER's 10-min data
  average tide: anchor on LW or HW (whichever extreme is larger), next anchor searched +-4 h around
         24 h 50 min later, mean of all 149-step (24 h 50 min) cycles, tiled, centred on HW
  spring tide: tidal maxima >= 12 days apart, mean of +-74.5 h windows around them
  surge shape: surge peaks >= 72 h apart, keep the top (1 - SURGE_PERCENTILE) of those peaks,
         +-36 h windows normalised by the peak and cut at the first negative value, mean time above
         each level 0..1 (step 0.005) per limb
  hydrograph: surge peak on HW of the average tide, scaled so HW + surge = storm-tide RP level
  return levels: COAST-RP (as HGRAPHER), taken at the COAST-RP location coinciding with each GTSM
         station (offshore grid points have none and are dropped); a POT/GPD fit on the hourly total
         water level is stored alongside for comparison (RP_SOURCE = "pot" uses it instead)

Stage 1 is cached per variable-year, so the script can be rerun while the download is running.
"""

import os
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as ss
import xarray as xr
from pyextremes import EVA

# ── settings ──────────────────────────────────────────────────────────────────
# Machine-specific path, read from the GCFM_RAW_DATA_ROOT environment variable.
# Set it once in PowerShell, then restart your terminal (see
# CONTRIBUTING.md "Local machine paths"):
#   [Environment]::SetEnvironmentVariable("GCFM_RAW_DATA_ROOT", "D:\your\raw_data\path", "User")
ZIP_DIR = Path(os.environ["GCFM_RAW_DATA_ROOT"]) / "GTSM_storm_tide_hourly"
CACHE_DIR = Path(os.environ["GCFM_RAW_DATA_ROOT"]) / "GTSM_storm_tide_hourly/cache"
OUT_NC = (
    Path(os.environ["GCFM_RAW_DATA_ROOT"])
    / "GTSM_storm_tide_hourly/GTSM_storm_tide_rp_hg.nc"
)
DELTAS = Path(os.environ["GCFM_RAW_DATA_ROOT"]) / "DeltaWebs/modified/9_polygons.gpkg"
COAST_RP = (
    Path(os.environ["GCFM_RAW_DATA_ROOT"]) / "CoastRP/COAST-RP.nc"
)  # data_catalogue: storm_tide_return_periods

RP_SOURCE = "coast_rp"  # "coast_rp" (as HGRAPHER; includes STORM tropical cyclones) or "pot" (own fit)
COAST_RP_MATCH_KM = 0.1  # COAST-RP location must coincide with the GTSM station (coastal points do exactly)

DELTA_IDS = None  # list of BasinID2 to keep, None = every polygon in DELTAS
RADIUS_KM = 100.0  # keep GTSM stations within this distance of a delta polygon
YEARS = range(1950, 2025)
RPS = (
    1,
    2,
    5,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
)  # = COAST-RP's tabulated RPs (src/surge.py)
HYDROGRAPH_RPS = (10, 100)  # storm_tide_hydrograph_rpXXXX written for these
POT_QUANTILE = (
    0.99  # return levels: threshold = this quantile of hourly total water level
)
POT_SEPARATION = "72h"  # return levels: declustering run length
SURGE_PERCENTILE = (
    0.99  # surge shape: keep peaks above this quantile of the 72h-separated peaks
)
PLOT_DELTA_ID = 2433835  # Ebro -- diagnostic figure for the station nearest this delta
MAX_GAP_HR = (
    3  # fill missing hours in gaps up to this long (the reanalysis has isolated
)
# 1-2 h dropouts at ~0.05% of hours); stations with longer gaps are skipped
DETREND_MSL = (
    True  # remove the slowly varying mean sea level (1-yr running mean of twl - surge)
)
# from total water level, so tide/return levels sit on a fixed MSL like COAST-RP
MSL_WINDOW_HR = 8766  # running-mean window (1 year of hours)
EXPORT_NC = OUT_NC.with_name(
    "GTSM_boundary_timeseries.nc"
)  # model-ready tide-only + storm-tide series
EXPORT_RPS = (
    10,
    100,
)  # storm-tide series written for these return periods (any of RPS)
EXPORT_WINDOW_HR = (
    60  # series run from -EXPORT_WINDOW_HR to +EXPORT_WINDOW_HR around the peak
)

VARS = {"total_water_level": "waterlevel", "storm_surge_residual": "surge"}
STEPS_PER_HR = 6  # 10-min, as HGRAPHER
LUNAR = 149  # 24 h 50 min
HALF_EVENT = 447  # +-74.5 h event window, peak at index 447
EVENT_HR = np.arange(-HALF_EVENT, HALF_EVENT) / STEPS_PER_HR
LEVELS = np.round(np.arange(0, 1.0001, 0.005), 3)


# ── stage 1: extract the stations near the deltas, one cache file per variable-year ─────
def select_stations(ds):
    x = ds["station_x_coordinate"].values
    y = ds["station_y_coordinate"].values
    pts = gpd.GeoSeries(gpd.points_from_xy(x, y), crs=4326)
    deltas = gpd.read_file(DELTAS).to_crs(4326)
    if DELTA_IDS is not None:
        deltas = deltas[deltas["BasinID2"].isin(DELTA_IDS)]
    keep = np.zeros(len(x), bool)
    for geom in deltas.geometry:
        g = gpd.GeoSeries([geom], crs=4326)
        utm = g.estimate_utm_crs()
        c = geom.centroid
        near = np.where((np.abs(x - c.x) < 10) & (np.abs(y - c.y) < 10))[0]
        dist = pts.iloc[near].to_crs(utm).distance(g.to_crs(utm).iloc[0]).values
        keep[near[dist <= RADIUS_KM * 1000]] = True
    idx = np.where(keep & np.isfinite(x) & np.isfinite(y))[0]
    print(
        f"{len(idx)} stations within {RADIUS_KM:g} km of {len(deltas)} delta polygon(s)"
    )
    return idx


def extract_year(zip_path, var, idx_file):
    out = CACHE_DIR / f"{zip_path.stem}.nc"
    if out.exists():
        return
    parts = []
    with zipfile.ZipFile(zip_path) as zf, tempfile.TemporaryDirectory() as tmp:
        for name in sorted(n for n in zf.namelist() if n.endswith(".nc")):
            path = zf.extract(name, tmp)
            with xr.open_dataset(path) as ds:
                if not idx_file.exists():
                    np.save(idx_file, select_stations(ds))
                parts.append(ds[[var]].isel(stations=np.load(idx_file)).load())
            Path(path).unlink()
    xr.concat(parts, "time").astype({var: "float32"}).to_netcdf(out)
    print("cached", out.name)


def stage1():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    idx_file = CACHE_DIR / "station_index.npy"
    for folder, var in VARS.items():
        for year in YEARS:
            z = ZIP_DIR / f"{folder}_{year}.zip"
            if not z.exists():
                continue
            try:
                extract_year(z, var, idx_file)
            except (zipfile.BadZipFile, OSError) as err:  # zip still being downloaded
                print("skip", z.name, "-", err)


def load_cache():
    years = [
        y for y in YEARS if all((CACHE_DIR / f"{f}_{y}.nc").exists() for f in VARS)
    ]
    if not years:
        raise SystemExit("no complete years cached yet")
    print(f"{len(years)} complete year(s): {years[0]}-{years[-1]}")
    return xr.merge(
        [
            xr.open_mfdataset(
                [CACHE_DIR / f"{f}_{y}.nc" for y in years], combine="by_coords"
            )[v]
            for f, v in VARS.items()
        ],
        compat="override",
    ).load()


# ── stage 2: HGRAPHER per station ─────────────────────────────────────────────
def to_10min(s):
    return s.resample("10min").interpolate("cubic").values


def tide_signals(tide):
    """HGRAPHER generate_tide_signals: average and spring tide, each +-447 steps around HW."""
    use_min = abs(np.quantile(tide, 0.99)) < abs(np.quantile(tide, 0.01))
    arg = np.argmin if use_min else np.argmax
    anchors = [int(arg(tide[:150]))]
    while True:
        c = anchors[-1] + LUNAR
        if c + 25 > len(tide):
            break
        anchors.append(c - 24 + int(arg(tide[c - 24 : c + 25])))
    cycles = np.array([tide[a : a + LUNAR] for a in anchors if a + LUNAR <= len(tide)])
    mean_cycle = cycles.mean(axis=0)
    i_max = int(mean_cycle.argmax()) + 3 * LUNAR
    average = np.tile(mean_cycle, 7)[i_max - HALF_EVENT : i_max + HALF_EVENT]

    peaks, _ = ss.find_peaks(tide, distance=12 * 24 * STEPS_PER_HR, height=-15)
    springs = [
        tide[p - HALF_EVENT : p + HALF_EVENT]
        for p in peaks[1:-1]
        if p - HALF_EVENT >= 0 and p + HALF_EVENT <= len(tide)
    ]
    spring = np.mean(springs, axis=0)
    return average, spring, len(cycles), len(springs)


def surge_hydrograph(surge):
    """HGRAPHER generate_surge_hydrograph: mean steps above each normalised level, per limb."""
    half = 36 * STEPS_PER_HR
    peaks, props = ss.find_peaks(surge, distance=72 * STEPS_PER_HR, height=-10)
    peaks = peaks[
        props["peak_heights"] >= np.quantile(props["peak_heights"], SURGE_PERCENTILE)
    ]
    before, after = [], []
    for p in peaks:
        if p - half < 0 or p + half >= len(surge) or surge[p] <= 0:
            continue
        b = surge[p - half : p + 1] / surge[p]
        a = surge[p : p + half + 1] / surge[p]
        neg_b, neg_a = np.where(b < 0)[0], np.where(a < 0)[0]
        b = b[neg_b[-1] :] if neg_b.size else b[1:]
        a = a[: neg_a[0]] if neg_a.size else a[:half]
        before.append((b[:, None] > LEVELS).sum(axis=0))
        after.append((a[:, None] > LEVELS).sum(axis=0))
    return np.mean(before, axis=0), np.mean(after, axis=0), len(before)


def surge_shape(dur_before, dur_after):
    """Normalised surge on the event grid (peak = 1 at index HALF_EVENT)."""
    steps = np.abs(np.arange(-HALF_EVENT, HALF_EVENT))
    rise = np.interp(steps, dur_before[::-1], LEVELS[::-1], right=0.0)
    fall = np.interp(steps, dur_after[::-1], LEVELS[::-1], right=0.0)
    return np.where(np.arange(-HALF_EVENT, HALF_EVENT) <= 0, rise, fall)


def return_levels(twl):
    eva = EVA(twl)
    eva.get_extremes(
        method="POT",
        extremes_type="high",
        threshold=twl.quantile(POT_QUANTILE),
        r=POT_SEPARATION,
    )
    eva.fit_model(model="MLE", distribution="genpareto")
    rl = np.asarray(
        eva.get_return_value(
            return_period=list(RPS), return_period_size="365.2425D", alpha=None
        )[0]
    )
    n_years = (twl.index[-1] - twl.index[0]) / pd.Timedelta(days=365.2425)
    return rl, float(eva.extremes.min()), len(eva.extremes) / n_years, eva


def coast_rp_levels(x, y):
    """COAST-RP storm-tide levels (n_station, len(RPS)) at the COAST-RP location coinciding with
    each GTSM station; NaN where none lies within COAST_RP_MATCH_KM (e.g. offshore grid points).
    """
    with xr.open_dataset(COAST_RP) as c:
        cx, cy = c["station_x_coordinate"].values, c["station_y_coordinate"].values
        table = np.stack(
            [c[f"storm_tide_rp_{rp:04d}"].values for rp in RPS], axis=1
        ).astype(float)
        cid = c["station_id"].values
    levels = np.full((len(x), len(RPS)), np.nan)
    ids = np.full(len(x), "", dtype=object)
    for i, (a, b) in enumerate(zip(x, y)):
        d = np.hypot((cx - a) * np.cos(np.radians(b)), cy - b) * 111.0
        j = int(np.nanargmin(d))
        if d[j] <= COAST_RP_MATCH_KM:
            levels[i], ids[i] = table[j], cid[j]
    return levels, ids


def analyse_station(twl, surge, coast_rl):
    tide10 = to_10min((twl - surge).dropna())
    surge10 = to_10min(surge)
    average, spring, n_cycles, n_springs = tide_signals(tide10)
    dur_b, dur_a, n_surge = surge_hydrograph(surge10)
    shape = surge_shape(dur_b, dur_a)
    pot_rl, thr, rate, eva = return_levels(twl)
    rl = coast_rl if RP_SOURCE == "coast_rp" else pot_rl
    i25 = int(np.where(LEVELS == 0.25)[0][0])
    c = HALF_EVENT - LUNAR // 2
    r = {
        "rl": rl,
        "pot_rl": pot_rl,
        "thr": thr,
        "rate": rate,
        "average": average,
        "spring": spring,
        "average_day": average[c : c + LUNAR + 1],
        "spring_day": spring[c : c + LUNAR + 1],
        "shape": shape,
        "n_cycles": n_cycles,
        "n_springs": n_springs,
        "n_surge": n_surge,
        "dur25": (dur_b[i25] + dur_a[i25]) / STEPS_PER_HR,
    }
    for rp in HYDROGRAPH_RPS:
        amp = max(rl[RPS.index(rp)] - average.max(), 0.0)
        r[f"hg{rp}"] = average + amp * shape
    return r, eva


def stage2():
    ds = load_cache()
    coast_rl, coast_ids = coast_rp_levels(
        ds["station_x_coordinate"].values, ds["station_y_coordinate"].values
    )
    print(
        f"{int(np.isfinite(coast_rl[:, 0]).sum())}/{ds.sizes['stations']} stations coincide with a COAST-RP location"
    )
    results, evas, keep, filled, msl_stats = [], {}, [], [], []
    n = ds.sizes["stations"]
    for i in range(n):
        if RP_SOURCE == "coast_rp" and not np.isfinite(coast_rl[i]).all():
            continue  # offshore grid point, no COAST-RP level
        twl = ds["waterlevel"].isel(stations=i).to_series().astype(float)
        surge = ds["surge"].isel(stations=i).to_series().astype(float)
        n_gap = int(twl.isna().sum() + surge.isna().sum())
        twl = twl.interpolate(limit=MAX_GAP_HR, limit_area="inside")
        surge = surge.interpolate(limit=MAX_GAP_HR, limit_area="inside")
        if twl.isna().any() or surge.isna().any() or twl.std() == 0:
            print(
                f"  station {int(ds.stations[i])}: gaps longer than {MAX_GAP_HR} h or constant, skipped"
            )
            continue
        filled.append(n_gap)
        if DETREND_MSL:
            msl = (
                (twl - surge)
                .rolling(MSL_WINDOW_HR, center=True, min_periods=MSL_WINDOW_HR // 2)
                .mean()
            )
            twl = twl - msl
            ann = msl.resample("YS").mean()
            msl_stats.append(
                (
                    np.polyfit(ann.index.year, ann.values, 1)[0] * 1000,
                    float(ann.max() - ann.min()),
                )
            )
        else:
            msl_stats.append((np.nan, np.nan))
        r, eva = analyse_station(twl, surge, coast_rl[i])
        results.append(r)
        evas[int(ds.stations[i])] = eva
        keep.append(i)
        if (i + 1) % 25 == 0 or i == n - 1:
            print(f"  {i + 1}/{n} stations")

    sub = ds.isel(stations=keep)

    def st(k):
        return np.stack([r[k] for r in results])

    t = pd.DatetimeIndex(ds["time"].values)
    lunar_hr = np.arange(LUNAR + 1) / STEPS_PER_HR
    src = "COAST-RP" if RP_SOURCE == "coast_rp" else "POT/GPD on GTSM reanalysis"
    out = xr.Dataset(
        {
            f"storm_tide_rp_{rp:04d}": (
                "stations",
                st("rl")[:, j],
                {
                    "units": "m",
                    "long_name": f"storm tide level, {rp}-yr return period ({src})",
                },
            )
            for j, rp in enumerate(RPS)
        }
        | {
            f"pot_storm_tide_rp_{rp:04d}": (
                "stations",
                st("pot_rl")[:, j],
                {
                    "units": "m",
                    "long_name": f"storm tide level, {rp}-yr return period (POT/GPD on GTSM reanalysis, comparison)",
                },
            )
            for j, rp in enumerate(RPS)
        },
        coords={
            "stations": sub["stations"].values,
            "station_x_coordinate": ("stations", sub["station_x_coordinate"].values),
            "station_y_coordinate": ("stations", sub["station_y_coordinate"].values),
            "time": pd.Timestamp("2000-01-01") + pd.to_timedelta(lunar_hr, "h"),
            "event_time_hr": (
                "event_time_hr",
                EVENT_HR,
                {"long_name": "hours relative to high water / surge peak"},
            ),
        },
    )

    def add(name, dims, k, **attrs):
        out[name] = (dims, st(k), attrs)

    out["coast_rp_station_id"] = ("stations", coast_ids[keep].astype(str))
    out["n_filled_hours"] = (
        "stations",
        np.array(filled),
        {
            "long_name": f"missing hourly values (twl + surge) filled by linear interpolation, gaps <= {MAX_GAP_HR} h"
        },
    )
    msl_stats = np.array(msl_stats)
    out["msl_trend_mm_per_yr"] = (
        "stations",
        msl_stats[:, 0],
        {
            "units": "mm/yr",
            "long_name": "linear trend of the removed mean sea level (annual means)",
        },
    )
    out["msl_range_m"] = (
        "stations",
        msl_stats[:, 1],
        {
            "units": "m",
            "long_name": "range (max - min) of annual-mean sea level removed from twl",
        },
    )
    add(
        "pot_threshold_m",
        "stations",
        "thr",
        units="m",
        long_name=f"POT threshold (q{POT_QUANTILE} of hourly total water level)",
    )
    add("pot_events_per_year", "stations", "rate")
    add(
        "hydrograph_average_tide_signal",
        ("stations", "time"),
        "average_day",
        units="m",
        long_name="average tide, one lunar day centred on HW (HGRAPHER)",
    )
    add(
        "hydrograph_spring_tide_signal",
        ("stations", "time"),
        "spring_day",
        units="m",
        long_name="average spring tide, one lunar day centred on the spring maximum (HGRAPHER)",
    )
    add(
        "average_tide_event",
        ("stations", "event_time_hr"),
        "average",
        units="m",
        long_name="average tide signal, HW at t=0",
    )
    add(
        "spring_tide_event",
        ("stations", "event_time_hr"),
        "spring",
        units="m",
        long_name="average spring tide signal, maximum at t=0",
    )
    add(
        "surge_shape",
        ("stations", "event_time_hr"),
        "shape",
        units="1",
        long_name="normalised average surge hydrograph, peak=1 at t=0",
    )
    add(
        "surge_duration_025_hr",
        "stations",
        "dur25",
        units="h",
        long_name="surge hydrograph duration above 0.25 x peak",
    )
    add("n_tidal_cycles", "stations", "n_cycles")
    add("n_spring_tides", "stations", "n_springs")
    add("n_surge_events", "stations", "n_surge")
    for rp in HYDROGRAPH_RPS:
        add(
            f"storm_tide_hydrograph_rp{rp:04d}",
            ("stations", "event_time_hr"),
            f"hg{rp}",
            units="m",
            long_name=f"RP{rp} storm tide hydrograph: average tide + scaled surge, surge peak on HW",
        )
    out.attrs = {
        "source": "Copernicus CDS sis-water-level-change-timeseries-cmip6, GTSM v3 reanalysis, hourly",
        "period": f"{t[0]:%Y-%m-%d} to {t[-1]:%Y-%m-%d}",
        "method": "HGRAPHER (Dullaart et al. 2023, NHESS 23:1847) on 10-min-interpolated hourly data",
        "return_levels": f"storm_tide_rp_*: {src}; pot_storm_tide_rp_*: POT-GPD "
        f"(threshold q{POT_QUANTILE}, declustering {POT_SEPARATION}, MLE)",
        "surge_percentile": SURGE_PERCENTILE,
        "msl_detrending": (
            f"1-yr ({MSL_WINDOW_HR} h) centred running mean of (twl - surge) subtracted from twl"
            if DETREND_MSL
            else "none"
        ),
        "datum": "model MSL (same as COAST-RP); add MDT/SLR as for COAST-RP",
        "script": "tests/KL_gtsm_storm_tide.py",
    }
    OUT_NC.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(OUT_NC)
    print("wrote", OUT_NC)
    plot(out, evas)
    return out


def export_boundary(out=None):
    """Model-ready coastal boundary series per station, 10-min, -EXPORT_WINDOW_HR..+EXPORT_WINDOW_HR h:
    tide only (average tide, HW at t=0) and storm tide per RP in EXPORT_RPS (tide + surge scaled so
    HW + surge = storm_tide_rp level). Levels are relative to (fixed) MSL -- MDT and SLR are
    added by the pipeline, as for COAST-RP. Rebuilt from OUT_NC, so it can run without stage 2.
    """
    if out is None:
        out = xr.open_dataset(OUT_NC)
    win = np.abs(out["event_time_hr"].values) <= EXPORT_WINDOW_HR + 1e-9
    t_hr = out["event_time_hr"].values[win]
    tide = out["average_tide_event"].isel(event_time_hr=np.where(win)[0])
    shape = out["surge_shape"].isel(event_time_hr=np.where(win)[0])
    hw = out["average_tide_event"].max(
        "event_time_hr"
    )  # same HW as the stored hydrographs

    ex = xr.Dataset(
        coords={
            "stations": out["stations"].values,
            "station_x_coordinate": ("stations", out["station_x_coordinate"].values),
            "station_y_coordinate": ("stations", out["station_y_coordinate"].values),
            "time": (
                "time",
                pd.Timestamp("2000-01-01") + pd.to_timedelta(t_hr - t_hr[0], "h"),
            ),
            "time_hr": (
                "time",
                t_hr,
                {"long_name": "hours relative to the peak / tidal high water"},
            ),
        },
    )
    ex["coast_rp_station_id"] = out["coast_rp_station_id"]
    ex["tide_only"] = (
        ("stations", "time"),
        tide.values,
        {
            "units": "m",
            "long_name": "average tide (HGRAPHER), high water at time_hr = 0",
        },
    )
    for rp in EXPORT_RPS:
        level = out[f"storm_tide_rp_{rp:04d}"]
        amp = np.maximum(level - hw, 0.0)
        ex[f"storm_tide_rp{rp:04d}"] = (
            ("stations", "time"),
            (tide + amp * shape).values,
            {
                "units": "m",
                "long_name": f"RP{rp} storm tide: average tide + surge peaking on HW at time_hr = 0",
            },
        )
        ex[f"surge_rp{rp:04d}"] = (
            ("stations", "time"),
            (amp * shape).values,
            {
                "units": "m",
                "long_name": f"RP{rp} storm surge component of storm_tide_rp{rp:04d}",
            },
        )
        ex[f"storm_tide_level_rp{rp:04d}"] = (
            "stations",
            level.values,
            {
                "units": "m",
                "long_name": f"RP{rp} storm tide level (peak of storm_tide_rp{rp:04d})",
            },
        )
    ex.attrs = {
        "source": OUT_NC.name + " (" + out.attrs.get("period", "") + ")",
        "datum": "local MSL (fixed, mean-sea-level trend removed); add MDT and SLR as for COAST-RP",
        "window": f"-{EXPORT_WINDOW_HR} to +{EXPORT_WINDOW_HR} h, 10-min steps",
        "return_levels": out.attrs.get("return_levels", ""),
        "script": "tests/KL_gtsm_storm_tide.py (export_boundary)",
    }
    ex.to_netcdf(EXPORT_NC)
    print("wrote", EXPORT_NC)
    return ex


def plot(out, evas):
    d = gpd.read_file(DELTAS)
    d = d[d["BasinID2"] == PLOT_DELTA_ID]
    if d.empty:
        return
    c = d.to_crs(4326).geometry.iloc[0].centroid
    x, y = out["station_x_coordinate"].values, out["station_y_coordinate"].values
    i = int(np.argmin(np.hypot((x - c.x) * np.cos(np.radians(c.y)), y - c.y)))
    s = out.isel(stations=i)
    sid = int(s["stations"])
    blue, orange, aqua, muted, grid = (
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#52514e",
        "#e4e3df",
    )
    t = out["event_time_hr"].values

    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    ext = evas[sid].extremes.sort_values(ascending=False).values
    rate = float(s["pot_events_per_year"])
    emp_rp = (len(ext) + 1) / (rate * np.arange(1, len(ext) + 1))
    ax[0].scatter(emp_rp, ext, s=10, color=muted, label="POT peaks (empirical)")
    ax[0].plot(
        RPS,
        [float(s[f"pot_storm_tide_rp_{rp:04d}"]) for rp in RPS],
        color=orange,
        lw=2,
        marker="o",
        ms=4,
        label="GPD fit (GTSM reanalysis)",
    )
    if RP_SOURCE == "coast_rp":
        ax[0].plot(
            RPS,
            [float(s[f"storm_tide_rp_{rp:04d}"]) for rp in RPS],
            color=blue,
            lw=2,
            marker="o",
            ms=4,
            label="COAST-RP (used)",
        )
    ax[0].set_xscale("log")
    ax[0].set(
        xlabel="Return period (yr)", ylabel="Storm tide (m)", title="Return levels"
    )

    ax[1].plot(t, s["average_tide_event"], color=blue, lw=2, label="Average tide")
    ax[1].plot(t, s["spring_tide_event"], color=orange, lw=2, label="Spring tide")
    ax[1].set(
        xlim=(-36, 36), xlabel="Hours from high water", ylabel="m", title="Tide signals"
    )

    ax[2].plot(
        t,
        s["surge_shape"],
        color=aqua,
        lw=2,
        label=f"{int(s['n_surge_events'])} events",
    )
    ax[2].set(
        xlim=(-36, 36),
        xlabel="Hours from peak",
        ylabel="Normalised surge",
        title="Surge hydrograph",
    )

    rl100 = float(s["storm_tide_rp_0100"])
    ax[3].plot(
        t,
        s["storm_tide_hydrograph_rp0100"],
        color=blue,
        lw=2,
        label=f"Storm tide, peak {rl100:.2f} m",
    )
    ax[3].plot(t, s["average_tide_event"], color=muted, lw=1.5, label="Average tide")
    ax[3].plot(
        t,
        s["storm_tide_hydrograph_rp0100"] - s["average_tide_event"],
        color=aqua,
        lw=2,
        label="Scaled surge",
    )
    ax[3].set(
        xlim=(-60, 60),
        xticks=np.arange(-60, 61, 12),
        xlabel="Hours from peak",
        ylabel="m",
        title="RP100 storm tide hydrograph",
    )

    for a in ax:
        a.grid(color=grid, lw=0.8)
        a.set_axisbelow(True)
        a.spines[["top", "right"]].set_visible(False)
        a.legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"GTSM station {sid} ({float(s.station_x_coordinate):.3f}E, {float(s.station_y_coordinate):.3f}N), "
        f"nearest to delta {PLOT_DELTA_ID}, {out.attrs['period']}",
        fontsize=10,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    png = OUT_NC.with_name(f"diagnostic_{PLOT_DELTA_ID}.png")
    fig.savefig(png, dpi=150, facecolor="#fcfcfb")
    print("wrote", png)


if __name__ == "__main__":
    stage1()
    export_boundary(stage2())
