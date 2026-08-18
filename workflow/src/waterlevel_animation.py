"""
waterlevel_animation.py -- Ad-hoc diagnostic, NOT wired into Snakemake.

Animates a `pre`-method adaptation strategy's own instantaneous water level
(zs, the full water surface elevation -- not depth) alongside its own
boundary water-level forcing (bzs), with a moving marker showing the current
frame -- lets you visually confirm a measure (levee, barrier, ...) is
actually registering and holding back water as forcing rises. Direct port of
the old delta_model project's own animation script onto this repo's basin/
scenario/strategy result layout and installed hydromt_sfincs API
(SfincsModel.get_component(...) instead of the old .water_level attribute
shortcut).

Only works for `pre`-method runs: they're the only ones with a real SFINCS
time series (`zs`) and their own rebuilt `sfincs.bzs` forcing. `post` never
reruns SFINCS -- it only ever produces one static max_flood_depth.tif, so
there's nothing to animate there.

Edit the CONFIG block below and run directly:
    python workflow/src/waterlevel_animation.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.animation import FFMpegWriter, FuncAnimation

# workflow/ on sys.path so `from src...` resolves the same way it does when
# Snakemake invokes a script/ -- this file isn't run through Snakemake, so
# that wiring doesn't happen automatically here.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.postprocessing import load_sfincs_output  # noqa: E402

# ── CONFIG -- edit these before each run ─────────────────────────────────────
RESULTS_DIR = "D:/GCFM_UU/results"
BASIN_ID = "2433835"
SCENARIO = "coast_500"
STRATEGY = "protect_closed_04"  # any already-run `pre` strategy folder
BZS_IDX = 4  # boundary point index to track (column order in sfincs.bzs)
T_START = None  # e.g. "2000-01-02 06:00:00", or None for full range
T_END = None
HMIN = 0.01  # water level [m] below which a cell is treated as dry
VMIN, VMAX = 0.0, 5.0  # colour scale for the water level map
FPS = 4

# ── paths ─────────────────────────────────────────────────────────────────────
basin_root = Path(RESULTS_DIR) / BASIN_ID
sfincs_root = (
    basin_root / "runs" / SCENARIO / "adaptation" / "pre" / STRATEGY / "sfincs"
)
spinup_root = basin_root / "spin_up"  # basin-level, shared by every scenario/strategy
out_path = (
    sfincs_root.parent / "03_waterlevel_animation.mp4"
)  # same flat strategy folder as the other pre outputs


def _load_zs(run_dir):
    # load_sfincs_output (same helper compute_max_inundation/compute_flood_progression
    # use elsewhere in this repo) reads sfincs_map.nc only, safely for this
    # installed hydromt_sfincs version -- avoids a bare SfincsModel(...).read()
    # also pulling in sfincs_his.nc, which isn't needed here.
    m = load_sfincs_output(run_dir)
    if "zs" not in m.output.data:
        raise RuntimeError(f"No 'zs' in {run_dir}/sfincs_map.nc")
    da = m.output.data["zs"]
    da = da.where(da > HMIN).drop_vars("spatial_ref", errors="ignore")
    da.attrs.update(long_name="water level", unit="m")
    return m, da


# ── model + water level (zs) time series ──────────────────────────────────────
mod, da_event = _load_zs(sfincs_root)

# This run's own sfincs_map.nc only covers [tstart, tstop] -- tstart is the
# SPIN-UP's own end time, not tref (see this run's own sfincs.inp), so on its
# own da_event starts partway through the full forcing timeseries. The
# missing lead-in is the spin-up's own separate run (basin-level, same for
# every strategy) -- prepend its own zs so the animation covers the whole
# simulated period from the very beginning, not just this event's own window.
_, da_spinup = _load_zs(spinup_root)
# Drop the spin-up's own last frame -- it's the same restart instant as the
# event's own first frame (continuity point), so keeping both would repeat
# one frame.
da_h = xr.concat([da_spinup.isel(time=slice(0, -1)), da_event], dim="time")

if T_START or T_END:
    da_h = da_h.sel(time=slice(T_START, T_END))
print(
    f"Animation time range: {da_h.time.min().values} to {da_h.time.max().values} ({da_h.time.size} frames)"
)

# Boundary water-level forcing (bzs), this strategy's own rebuilt forcing
# (same event, adapted model) -- tracked in a panel below the map.
wl = mod.get_component("water_level")
wl.read()
bzs = wl.data["bzs"]
ts_h = bzs.isel(index=BZS_IDX)
if T_START or T_END:
    ts_h = ts_h.sel(time=slice(T_START, T_END))
ts_times = ts_h.time.values


def update_plot(i, da_h, cax_h, ts_line):
    da_hi = da_h.isel(time=i)
    t = da_hi.time.dt.strftime("%d-%B-%Y %H:%M:%S").item()
    ax.set_title(f"SFINCS water level {t} — {STRATEGY} (pre) | {BASIN_ID}/{SCENARIO}")
    cax_h.set_array(da_hi.values.ravel())
    current_time = da_hi.time.values  # use the map's own timestamp, not the bzs index
    ts_line.set_xdata([current_time, current_time])


fig, ax = mod.plot_basemap(
    fn_out=None,
    variable="",
    bmap="sat",
    plot_bounds=False,
    figsize=(11, 12),
    zoomlevel=12,
)

# Shrink the map to make room for the timeseries panel below.
ax.set_position((0.1, 0.35, 0.8, 0.6))

cax_h = da_h.isel(time=0).plot(
    ax=ax,
    vmin=VMIN,
    vmax=VMAX,
    cmap=plt.cm.Blues,
    cbar_kwargs={"shrink": 0.6, "anchor": (0, 0)},
    add_labels=False,  # prevents xc/yc from overwriting axis labels
)

# Timeseries panel: boundary water level with a moving line marking the current frame.
ax2 = fig.add_axes((0.1, 0.01, 0.65, 0.14))
ax2.plot(ts_times, ts_h.values, color="k", lw=1)
ts_line = ax2.axvline(ts_times[0], color="red", lw=1.5)
ax2.set_ylabel("water level [m+ref]")
ax2.set_xlim(ts_times[0], ts_times[-1])
fig.autofmt_xdate()

anim = FuncAnimation(
    fig,
    update_plot,
    frames=np.arange(0, da_h.time.size, 1),
    interval=250,  # ms between frames, only used for on-screen playback
    fargs=(da_h, cax_h, ts_line),
)

out_path.parent.mkdir(parents=True, exist_ok=True)
anim.save(
    str(out_path),
    writer=FFMpegWriter(fps=FPS, extra_args=["-crf", "15", "-pix_fmt", "yuv420p"]),
)
plt.close(fig)
print(f"Written: {out_path}")
