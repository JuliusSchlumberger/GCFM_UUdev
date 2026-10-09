"""
07c_storm_tide_hydrographs.py -- Storm-tide hydrograph components at a
basin's own surge stations: the average tide and one normalised surge shape
per COAST-RP return period (src.storm_tide_hydrograph; HGRAPHER, Dullaart
et al. 2023, with return-period-specific shapes).

Inputs
------
gtsm_hourly     Hourly GTSM total water level + surge residual at the surge
                stations (rule extract_gtsm_series).
surge_data      COAST-RP -- the storm-tide level per return period each
                shape is derived for.

Output
------
storm_tide_hydrographs.nc, per station:
    average_tide (station, time_hr)            m, local MSL, high water at t=0
    surge_shape  (station, return_period, time_hr)   0..1, peak = 1 at t=0
    shape_n_events / shape_window_m / shape_mode / shape_event_surge_m /
    shape_dur50_hr (station, return_period)    which events each shape is
                                               built from, and how wide it is
plus a diagnostic figure.

TODO (follow-up, not done here): inspect the relative timing (lag) of the
storm-tide peaks across a basin's stations in the GTSM series -- the
boundary currently puts every station's peak at the same instant.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
rps = np.array(_COASTRP_RPS, dtype=float)

gtsm = xr.open_dataset(snakemake.input.gtsm_hourly).load()
lons, lats = gtsm["longitude"].values, gtsm["latitude"].values

# COAST-RP return levels at the same stations (local MSL).
with xr.open_dataset(snakemake.input.surge_data) as c:
    cx, cy = c["station_x_coordinate"].values, c["station_y_coordinate"].values
    table = np.stack([c[f"storm_tide_rp_{int(rp):04d}"].values for rp in rps], axis=1).astype(float)
levels = np.full((len(lons), len(rps)), np.nan)
for i, (lon, lat) in enumerate(zip(lons, lats)):
    d = np.hypot((cx - lon) * np.cos(np.radians(lat)), cy - lat) * 111.0
    j = int(np.nanargmin(d))
    if d[j] > max_match_km:
        raise ValueError(f"no COAST-RP station within {max_match_km:g} km of ({lon:.4f}, {lat:.4f})")
    levels[i] = table[j]

results = []
for i in range(gtsm.sizes["station"]):
    one = gtsm.isel(station=i)
    r = station_hydrograph_components(
        one["waterlevel"].to_series().astype(float),
        one["surge"].to_series().astype(float),
        levels[i], half_window_hr, min_events, match_window_m, match_on=match_on,
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
        f"{[f'{int(rp)}: {d:.0f} ({m})' for rp, d, m in zip(rps, r['dur50_hr'], r['mode'])]}"
    )


def stack(key):
    return np.stack([r[key] for r in results])


time_hr = results[0]["time_hr"]
out = xr.Dataset(
    {
        "average_tide": (("station", "time_hr"), stack("tide"),
                         {"units": "m", "long_name": "average tide signal, local MSL, high water at time_hr=0"}),
        "surge_shape": (("station", "return_period", "time_hr"), stack("shape"),
                        {"units": "1", "long_name": "normalised surge hydrograph per return period, peak=1 at time_hr=0"}),
        "storm_tide_level": (("station", "return_period"), levels,
                             {"units": "m", "long_name": "COAST-RP storm-tide level (local MSL)"}),
        "shape_n_events": (("station", "return_period"), stack("n_events"),
                           {"long_name": "events the shape is averaged over"}),
        "shape_window_m": (("station", "return_period"), stack("window_m"),
                           {"units": "m", "long_name": "half-width of the peak window those events lie in"}),
        "shape_mode": (("station", "return_period"), stack("mode").astype(str),
                       {"long_name": "window | widened (nearest events) | above_record (largest events)"}),
        "shape_event_surge_m": (("station", "return_period"), stack("surge_peak_m"),
                                {"units": "m", "long_name": "median surge peak of those events"}),
        "shape_dur50_hr": (("station", "return_period"), stack("dur50_hr"),
                           {"units": "h", "long_name": "time the shape stays above 0.5"}),
        "high_water_m": (("station",), np.array([r["high_water_m"] for r in results]),
                         {"units": "m", "long_name": "high water of the average tide"}),
    },
    coords={
        "time_hr": ("time_hr", time_hr, {"units": "hours", "long_name": "hours relative to the surge peak / tidal high water"}),
        "return_period": ("return_period", rps, {"units": "yr"}),
        "longitude": ("station", lons),
        "latitude": ("station", lats),
        "gtsm_station": ("station", gtsm["gtsm_station"].values),
    },
    attrs={
        "method": "HGRAPHER (Dullaart et al. 2023, NHESS 23:1847) on 10-min-interpolated hourly GTSM data, "
                  "surge shape per return period (src.storm_tide_hydrograph)",
        "period": gtsm.attrs.get("period", ""),
        "match_on": match_on, "min_events": min_events, "match_window_m": match_window_m,
        "datum": "local mean sea level (as COAST-RP); MDT/SLR are added when the boundary is built",
    },
)
Path(snakemake.output.hydrographs).parent.mkdir(parents=True, exist_ok=True)
out.to_netcdf(snakemake.output.hydrographs)
log.info(f"Written: {snakemake.output.hydrographs} ({len(results)} station(s), {len(rps)} return periods)")

# ── diagnostic figure: the station nearest the domain (first) ────────────────
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e6e5e1"
RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # one hue, light -> dark = rarer
shown = [1, 10, 50, 100, 250, 1000]
r0 = results[0]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), facecolor=SURFACE)
for rp, color in zip(shown, RAMP):
    j = int(np.flatnonzero(rps == rp)[0])
    ax[0].plot(time_hr, r0["shape"][j], color=color, lw=1.8,
               label=f"RP {rp}: {r0['dur50_hr'][j]:.0f} h, {r0['n_events'][j]} events ({r0['mode'][j]})")
    amp = max(levels[0, j] - r0["high_water_m"], 0.0)
    ax[2].plot(time_hr, r0["tide"] + amp * r0["shape"][j], color=color, lw=1.6, label=f"RP {rp}: {levels[0, j]:.2f} m")
ax[1].plot(time_hr, r0["tide"], color="#2a78d6", lw=1.8)
titles = ("Surge shape per return period (hours above half the peak)", "Average tide", "Storm tide = tide + scaled surge")
ylabels = ("normalised surge", "m above local mean sea level", "m above local mean sea level")
for a, title, ylabel in zip(ax, titles, ylabels):
    a.set_title(title, fontsize=10, loc="left", color=INK)
    a.set_xlabel("hours from the peak", fontsize=9, color=INK2)
    a.set_ylabel(ylabel, fontsize=9, color=INK2)
    a.grid(color=GRID, lw=0.8)
    a.set_axisbelow(True)
    a.spines[["top", "right"]].set_visible(False)
    a.spines[["left", "bottom"]].set_color(MUTED)
    a.tick_params(labelsize=8, colors=INK2)
    a.set_facecolor(SURFACE)
ax[0].legend(frameon=False, fontsize=7.5, loc="upper left")
ax[2].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.suptitle(
    f"Storm-tide hydrograph components -- basin {snakemake.wildcards.basin_id}, GTSM station "
    f"{int(gtsm['gtsm_station'].values[0])} (nearest of {len(results)}), events matched on {match_on.replace('_', ' ')}",
    fontsize=11, x=0.01, ha="left", color=INK,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
Path(snakemake.output.plot).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(snakemake.output.plot, dpi=140, facecolor=SURFACE)
log.info(f"Written: {snakemake.output.plot}")
