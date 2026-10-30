"""Derive average tide and per-RP surge shapes per station; run by rule storm_tide_hydrographs.

HGRAPHER (Dullaart et al. 2023) with return-period-specific shapes
(src.storm_tide_hydrograph), from hourly GTSM series (rule extract_gtsm_series)
and COAST-RP storm-tide levels per return period.

Output
------
storm_tide_hydrographs.nc, per station:
    average_tide (station, time_hr)            m, local MSL, high water at t=0
    surge_shape  (station, rp_yr, time_hr)     0..1, peak = 1 at t=0
    shape_n_events / shape_window_m / shape_mode / shape_event_surge_m /
    shape_dur50_hr (station, rp_yr)            which events each shape is
                                               built from, and how wide it is
(rule analysis_preprocess draws the station nearest the domain).

TODO: inspect the relative timing (lag) of the storm-tide peaks across a
basin's stations; the boundary puts every station's peak at the same instant.
"""

from pathlib import Path

import numpy as np
import xarray as xr

from src.log import setup_logging
from src.storm_tide_hydrograph import station_hydrograph_components
from src.surge import _COASTRP_RPS

log = setup_logging(snakemake.log[0])

half_window_hr = float(snakemake.params.window_hr)
min_events = int(snakemake.params.min_events)
match_window_m = float(snakemake.params.match_window_m)
match_on = str(snakemake.params.match_on)
max_match_km = float(snakemake.params.max_match_km)
rps_yr = np.array(_COASTRP_RPS, dtype=float)

gtsm = xr.open_dataset(snakemake.input.gtsm_hourly).load()
lons, lats = gtsm["longitude"].values, gtsm["latitude"].values

# COAST-RP return levels at the same stations (local MSL).
with xr.open_dataset(snakemake.input.surge_data) as c:
    cx, cy = c["station_x_coordinate"].values, c["station_y_coordinate"].values
    table = np.stack(
        [c[f"storm_tide_rp_{int(rp_yr):04d}"].values for rp_yr in rps_yr], axis=1
    ).astype(float)
levels = np.full((len(lons), len(rps_yr)), np.nan)
for i, (lon, lat) in enumerate(zip(lons, lats)):
    d = np.hypot((cx - lon) * np.cos(np.radians(lat)), cy - lat) * 111.0
    j = int(np.nanargmin(d))
    if d[j] > max_match_km:
        raise ValueError(
            f"no COAST-RP station within {max_match_km:g} km of ({lon:.4f}, {lat:.4f})"
        )
    levels[i] = table[j]

results = []
for i in range(gtsm.sizes["station"]):
    one = gtsm.isel(station=i)
    r = station_hydrograph_components(
        one["waterlevel"].to_series().astype(float),
        one["surge"].to_series().astype(float),
        levels[i],
        half_window_hr,
        min_events,
        match_window_m,
        match_on=match_on,
    )
    if r is None:
        raise ValueError(
            f"GTSM series at station {int(one['gtsm_station'])} ({lons[i]:.4f}, {lats[i]:.4f}) has gaps "
            f"longer than 3 h -- no hydrograph can be derived for it"
        )
    results.append(r)
    log.info(
        f"station {int(one['gtsm_station'])}: high water {r['high_water_m']:+.2f} m, "
        f"{r['n_candidate_events']} events (highest {r['max_peak_m']:.2f} m); hours above half the peak per RP "
        f"{[f'{int(rp_yr)}: {d:.0f} ({m})' for rp_yr, d, m in zip(rps_yr, r['dur50_hr'], r['mode'])]}"
    )


def stack(key):
    return np.stack([r[key] for r in results])


time_hr = results[0]["time_hr"]
out = xr.Dataset(
    {
        "average_tide": (
            ("station", "time_hr"),
            stack("tide"),
            {
                "units": "m",
                "long_name": "average tide signal, local MSL, high water at time_hr=0",
            },
        ),
        "surge_shape": (
            ("station", "rp_yr", "time_hr"),
            stack("shape"),
            {
                "units": "1",
                "long_name": "normalised surge hydrograph per return period, peak=1 at time_hr=0",
            },
        ),
        "storm_tide_level_m": (
            ("station", "rp_yr"),
            levels,
            {"units": "m", "long_name": "COAST-RP storm-tide level (local MSL)"},
        ),
        "shape_n_events": (
            ("station", "rp_yr"),
            stack("n_events"),
            {"long_name": "events the shape is averaged over"},
        ),
        "shape_window_m": (
            ("station", "rp_yr"),
            stack("window_m"),
            {
                "units": "m",
                "long_name": "half-width of the peak window those events lie in",
            },
        ),
        "shape_mode": (
            ("station", "rp_yr"),
            stack("mode").astype(str),
            {
                "long_name": "window | widened (nearest events) | above_record (largest events)"
            },
        ),
        "shape_event_surge_m": (
            ("station", "rp_yr"),
            stack("surge_peak_m"),
            {"units": "m", "long_name": "median surge peak of those events"},
        ),
        "shape_dur50_hr": (
            ("station", "rp_yr"),
            stack("dur50_hr"),
            {"units": "h", "long_name": "time the shape stays above 0.5"},
        ),
        "high_water_m": (
            ("station",),
            np.array([r["high_water_m"] for r in results]),
            {"units": "m", "long_name": "high water of the average tide"},
        ),
    },
    coords={
        "time_hr": (
            "time_hr",
            time_hr,
            {
                "units": "hours",
                "long_name": "hours relative to the surge peak / tidal high water",
            },
        ),
        "rp_yr": ("rp_yr", rps_yr, {"units": "yr"}),
        "longitude": ("station", lons),
        "latitude": ("station", lats),
        "gtsm_station": ("station", gtsm["gtsm_station"].values),
    },
    attrs={
        "method": "HGRAPHER (Dullaart et al. 2023, NHESS 23:1847) on 10-min-interpolated hourly GTSM data, "
        "surge shape per return period (src.storm_tide_hydrograph)",
        "period": gtsm.attrs.get("period", ""),
        "match_on": match_on,
        "min_events": min_events,
        "match_window_m": match_window_m,
        "datum": "local mean sea level (as COAST-RP); MDT/SLR are added when the boundary is built",
    },
)
Path(snakemake.output.hydrographs).parent.mkdir(parents=True, exist_ok=True)
out.to_netcdf(snakemake.output.hydrographs)
log.info(
    f"Written: {snakemake.output.hydrographs} ({len(results)} station(s), {len(rps_yr)} return periods)"
)
