"""Figure: RP100 storm tide event per delta, split into tide and storm surge

Reads the output of KL_gtsm_storm_tide.py and plots, for the GTSM station nearest to each delta
polygon, the RP100 storm tide hydrograph (average tide + scaled surge, surge peak on high water).
"""

import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Machine-specific path, read from the GCFM_RAW_DATA_ROOT environment variable.
# Set it once in PowerShell, then restart your terminal (see
# CONTRIBUTING.md "Local machine paths"):
#   [Environment]::SetEnvironmentVariable("GCFM_RAW_DATA_ROOT", "D:\your\raw_data\path", "User")
HG_NC = (
    Path(os.environ["GCFM_RAW_DATA_ROOT"])
    / "GTSM_storm_tide_hourly/GTSM_storm_tide_rp_hg.nc"
)
DELTAS = Path(os.environ["GCFM_RAW_DATA_ROOT"]) / "DeltaWebs/modified/9_polygons.gpkg"
RP = 100
XLIM_HR = 60
OUT_PNG = HG_NC.with_name(f"rp{RP:04d}_event_per_delta.png")

blue, aqua, muted, grid = "#2a78d6", "#1baf7a", "#52514e", "#e4e3df"

ds = xr.open_dataset(HG_NC)
deltas = gpd.read_file(DELTAS).to_crs(4326)
stations = gpd.GeoSeries(
    gpd.points_from_xy(ds.station_x_coordinate, ds.station_y_coordinate), crs=4326
)
t = ds["event_time_hr"].values

n = len(deltas)
ncol = 3
fig, axes = plt.subplots(
    int(np.ceil(n / ncol)), ncol, figsize=(13, 3.2 * np.ceil(n / ncol)), sharex=True
)
for ax, (_, d) in zip(axes.flat, deltas.iterrows()):
    g = gpd.GeoSeries([d.geometry], crs=4326)
    utm = g.estimate_utm_crs()
    dist = stations.to_crs(utm).distance(g.to_crs(utm).iloc[0]).values / 1000
    i = int(np.argmin(dist))
    s = ds.isel(stations=i)

    level = float(s[f"storm_tide_rp_{RP:04d}"])
    tide = s["average_tide_event"].values
    surge = (level - tide.max()) * s["surge_shape"].values
    total = tide + surge

    ax.plot(t, total, color=blue, lw=2, label=f"Storm tide (RP{RP})")
    ax.plot(t, tide, color=muted, lw=1.5, label="Tide")
    ax.plot(t, surge, color=aqua, lw=2, label="Storm surge")
    ax.annotate(
        f"{level:.2f} m",
        (0, level),
        xytext=(6, 2),
        textcoords="offset points",
        fontsize=8,
    )
    ax.set_title(
        f"{d.delta_name}  (GTSM {int(s.stations)}, {dist[i]:.0f} km)",
        loc="left",
        fontsize=10,
    )
    ax.set_xlim(-XLIM_HR, XLIM_HR)
    ax.set_xticks(np.arange(-XLIM_HR, XLIM_HR + 1, 12))
    ax.grid(color=grid, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
for ax in axes.flat[n:]:
    ax.set_visible(False)
for ax in axes[:, 0]:
    ax.set_ylabel("Water level (m, MSL)")
for ax in axes[-1]:
    ax.set_xlabel("Hours from peak")

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 1.0),
)
fig.suptitle(
    f"RP{RP} storm tide event per delta: average tide + storm surge peaking at high water "
    f"(COAST-RP level, HGRAPHER shape, GTSM {ds.attrs['period']})",
    fontsize=10,
    y=1.03,
)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150, facecolor="#fcfcfb", bbox_inches="tight")
print("wrote", OUT_PNG)
