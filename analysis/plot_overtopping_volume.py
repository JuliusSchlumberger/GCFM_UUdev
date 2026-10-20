"""Estimate coastal-crest overtopping volume from a run's boundary forcing.

Uses the forcing SFINCS was given (sfincs.bnd/sfincs.bzs: per-station water
level, seconds since tref, model CRS) and SFINCS's free-flow weir formula
Q = par1 * 1.7049 * L * head^1.5, with 1.7049 = (2/3)^1.5 * sqrt(g) (see
config/config.yml's weir_par1 comment). Each weir segment takes the nearest
station's level; head = station level - crest, clipped at 0.

Crude: no interpolation between stations, no local set-up or 2-D dynamics.
SFINCS's submerged-weir switch (landward side above 2/3 of the seaward head) is
ignored, so discharge is overestimated once the land behind a segment floods.
Inland segments raised above every station level never overtop.

Usage:
    conda run -n <env> python analysis/plot_overtopping_volume.py [basin_id] [scenario]
    conda run -n <env> python analysis/plot_overtopping_volume.py 2433835 coast_250
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from _common import (  # noqa: E402
    REPO_ROOT,
    basin_id_from_argv,
    first_existing,
    site_path,
)
from scipy.integrate import cumulative_trapezoid

sys.path.insert(0, str(REPO_ROOT / "workflow"))
from src.sfincs_run import parse_sfincs_inp

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

BASIN_ID = basin_id_from_argv()
SCENARIO = sys.argv[2] if len(sys.argv) > 2 else "coast_250"

with open(REPO_ROOT / "config" / "config.yml") as fh:
    config = yaml.safe_load(fh)
# results_dir is a site setting; see config/site.example.yml.
RESULTS_DIR = site_path("results_dir")
FIGS_DIR = REPO_ROOT / "figs" / "surge_vs_dike_crest"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

basin_dir = RESULTS_DIR / BASIN_ID
sfincs_dir = basin_dir / "runs" / SCENARIO / "sfincs"


weir_candidates = (
    [
        basin_dir / "sfincs_skeleton" / f"{BASIN_ID}_coastal_protection_weir.gpkg",
    ]
    + sorted(basin_dir.glob("runs/*/sfincs/*_coastal_protection_weir.gpkg"))
    + sorted(basin_dir.glob("scenarios/*/sfincs/*_coastal_protection_weir.gpkg"))
)
weir_path = first_existing(weir_candidates)

bnd_path = sfincs_dir / "sfincs.bnd"
bzs_path = sfincs_dir / "sfincs.bzs"
inp_path = sfincs_dir / "sfincs.inp"
for _p in (bnd_path, bzs_path, inp_path):
    if not _p.exists():
        raise FileNotFoundError(
            f"Required run output missing: {_p} — scenario {SCENARIO!r} for basin "
            f"{BASIN_ID!r} must have been built AND run (rule build_sfincs + run_event) "
            f"before this script can read its actual boundary forcing."
        )

log.info(f"weir: {weir_path}")
log.info(f"sfincs run dir: {sfincs_dir}")

# forcing fed to SFINCS for this run
inp_cfg = parse_sfincs_inp(inp_path)
tref = datetime.strptime(inp_cfg["tref"], "%Y%m%d %H%M%S")

stations_xy = pd.read_csv(
    bnd_path, sep=r"\s+", header=None
).to_numpy()  # (n_station, 2)
bzs = pd.read_csv(
    bzs_path, sep=r"\s+", header=None
).to_numpy()  # (n_time, 1 + n_station)
time_s = bzs[:, 0]
levels = bzs[:, 1:]  # (n_time, n_station)
n_station = stations_xy.shape[0]
assert levels.shape[1] == n_station, (
    f"sfincs.bzs has {levels.shape[1]} station columns, sfincs.bnd has {n_station} stations"
)
times = [tref + timedelta(seconds=float(s)) for s in time_s]
log.info(
    f"tref={tref}, {n_station} station(s), {len(time_s)} time step(s), "
    f"{time_s[-1] / 3600:.1f} h total"
)

# nearest boundary station per weir segment
weir_gdf = gpd.read_file(weir_path)
centroids = np.column_stack(
    [weir_gdf.geometry.centroid.x, weir_gdf.geometry.centroid.y]
)
dist2 = ((centroids[:, None, :] - stations_xy[None, :, :]) ** 2).sum(axis=2)
nearest_station = dist2.argmin(axis=1)  # (n_segment,)

elevation = weir_gdf["elevation"].to_numpy(dtype=float)
par1 = weir_gdf["par1"].to_numpy(dtype=float)
length_m = weir_gdf.geometry.length.to_numpy(dtype=float)
n_segment = len(weir_gdf)
log.info(
    f"Weir: {n_segment} segment(s), crest [{elevation.min():.3f}, {elevation.max():.3f}] m"
)

# per-segment head and discharge: Q = par1 * 1.7049 * L * max(0, head)^1.5
SFINCS_CWEIR = 1.7049  # (2/3)^1.5 * sqrt(g), SFINCS free-flow weir constant
level_at_segment = levels[:, nearest_station]  # (n_time, n_segment)
head = np.maximum(0.0, level_at_segment - elevation[None, :])
q_segment = (
    par1[None, :] * SFINCS_CWEIR * length_m[None, :] * head**1.5
)  # (n_time, n_segment), m^3/s
q_total = q_segment.sum(axis=1)  # (n_time,), m^3/s

overtopped_mask = np.any(head > 0, axis=0)
n_overtopped = int(overtopped_mask.sum())
log.info(
    f"{n_overtopped}/{n_segment} segment(s) overtop at some point during the run "
    f"(peak combined discharge {q_total.max():.1f} m3/s)"
)

volume_m3 = cumulative_trapezoid(q_total, time_s, initial=0.0)
volume_Mm3 = volume_m3 / 1e6
log.info(f"Cumulative overtopping volume at end of run: {volume_Mm3[-1]:.4f} Mm3")

peak_idx = int(np.argmax(q_total))
log.info(
    f"Peak overtopping discharge at {times[peak_idx]} "
    f"({time_s[peak_idx] / 3600:.1f} h since tref): {q_total[peak_idx]:.1f} m3/s"
)

# plot: discharge (top) + cumulative volume (bottom)
fig, (ax_q, ax_vol) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

ax_q.plot(times, q_total, color="firebrick", linewidth=1.5)
ax_q.set_ylabel("Overtopping discharge\n(m³/s, summed over all segments)")
ax_q.set_title(
    f"Coastal-crest overtopping — basin {BASIN_ID}, scenario {SCENARIO}\n"
    f"crude approx.: Q = par1·1.7049·L·head^1.5 per weir segment, driven by the actual "
    f"boundary timeseries SFINCS used (nearest station per segment)"
)
ax_q.grid(True, alpha=0.3, linewidth=0.5)

ax_vol.plot(times, volume_Mm3, color="steelblue", linewidth=2)
ax_vol.set_ylabel("Cumulative overtopping volume\n(Mm³)")
ax_vol.set_xlabel("Time")
ax_vol.grid(True, alpha=0.3, linewidth=0.5)
ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax_vol.annotate(
    f"{volume_Mm3[-1]:.3f} Mm³ total\n{n_overtopped}/{n_segment} segment(s) overtop",
    xy=(times[-1], volume_Mm3[-1]),
    xytext=(-10, -25),
    textcoords="offset points",
    ha="right",
    fontsize=9,
    fontweight="bold",
    color="steelblue",
)

fig.autofmt_xdate()
fig.tight_layout()

out_path = FIGS_DIR / f"{BASIN_ID}_{SCENARIO}_overtopping_volume.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
log.info(f"Plot written: {out_path}")
