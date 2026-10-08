"""
waterlevel_animation.py -- Ad-hoc diagnostic, NOT wired into Snakemake.

Animates a `pre`-method adaptation strategy's own instantaneous flood depth
(h = max(zs - zb, 0), the same quantity 02_flood_animation.mp4 is built
from -- see compute_flood_progression in src/postprocessing.py) alongside
its own boundary water-level forcing (bzs), with a moving marker showing the
current frame -- lets you visually confirm a measure (levee, barrier, ...)
is actually registering and holding back water as forcing rises.

Plots depth rather than the raw water level `zs` on purpose: `zs` is the
absolute water-surface elevation, which is meaningless as a flood
mask/threshold wherever the bed itself sits below the vertical reference --
e.g. this basin's protect_closed strategies wall off a below-sea-level
polder (bed elevation down to roughly -8 m) with a coastal levee + pumps, so
when the pumps are overwhelmed the water flooding the polder can sit at an
absolute `zs` still under HMIN even though the depth over the polder floor
is metres. Thresholding/colouring by depth instead, and excluding the open
sea the same way compute_flood_progression does, is what actually shows the
flood signal.

Direct port of the old delta_model project's own animation script onto this
repo's basin/scenario/strategy result layout and installed hydromt_sfincs
API (SfincsModel.get_component(...) instead of the old .water_level
attribute shortcut).

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))

from src.postprocessing import load_sfincs_output

# ── CONFIG -- edit these before each run ─────────────────────────────────────
RESULTS_DIR = "D:/GCFM_UU/results"
BASIN_ID = "2433835"
SCENARIO = "coast_500"
STRATEGY = "protect_closed_1"  # any already-run `pre` strategy folder
BZS_IDX = 4  # boundary point index to track (column order in sfincs.bzs)
T_START = None  # "2000-01-02 00:00:00" #, or None for full range
T_END = None
HMIN = 0.05  # flood depth [m] below which a cell is treated as dry
VMIN, VMAX = 0.0, 5.0  # colour scale for the flood depth map
FPS = 4

# ── paths ─────────────────────────────────────────────────────────────────────
basin_root = Path(RESULTS_DIR) / BASIN_ID
sfincs_root = (
    basin_root / "runs" / SCENARIO / "adaptation" / "pre" / STRATEGY / "sfincs"
)
spinup_root = basin_root / "spin_up"  # basin-level, shared by every scenario/strategy
# Basin-level sea/land classification -- same file 16_run_event.py's own
# compute_flood_progression call is given (see 16_run_event.smk's sea_mask
# input), so open sea is excluded here exactly like it is in
# 02_flood_animation.mp4.
sea_mask_path = (
    basin_root
    / "preprocessing_inputs"
    / "domain"
    / f"{BASIN_ID}_zsini_sea_cells_on_grid.tif"
)
out_path = (
    sfincs_root.parent / "03_waterlevel_animation.mp4"
)  # same flat strategy folder as the other pre outputs


def _load_h(run_dir):
    # load_sfincs_output (same helper compute_max_inundation/compute_flood_progression
    # use elsewhere in this repo) reads sfincs_map.nc only, safely for this
    # installed hydromt_sfincs version -- avoids a bare SfincsModel(...).read()
    # also pulling in sfincs_his.nc, which isn't needed here.
    m = load_sfincs_output(run_dir)
    if "zs" not in m.output.data:
        raise RuntimeError(f"No 'zs' in {run_dir}/sfincs_map.nc")
    if "zb" not in m.output.data:
        raise RuntimeError(f"No 'zb' in {run_dir}/sfincs_map.nc")

    # Depth above bed, not raw water level -- see module docstring for why
    # zs alone is the wrong quantity to threshold/colour by here.
    da_zb = m.output.data["zb"].squeeze()
    da_h = (m.output.data["zs"] - da_zb).clip(min=0.0)

    da_sea = m.data_catalog.get_rasterdataset(str(sea_mask_path))
    da_sea_grid = da_sea.raster.reproject_like(da_h, method="nearest")
    da_h = da_h.where(da_sea_grid != 1.0)  # exclude open sea

    da_h = da_h.where(da_h > HMIN).drop_vars("spatial_ref", errors="ignore")
    da_h.attrs.update(long_name="flood depth", unit="m")
    return m, da_h


# ── model + flood depth (h = zs - zb) time series ─────────────────────────────
mod, da_event = _load_h(sfincs_root)

# This run's own sfincs_map.nc only covers [tstart, tstop] -- tstart is the
# SPIN-UP's own end time, not tref (see this run's own sfincs.inp), so on its
# own da_event starts partway through the full forcing timeseries. The
# missing lead-in is the spin-up's own separate run (basin-level, same for
# every strategy) -- prepend its own depth so the animation covers the whole
# simulated period from the very beginning, not just this event's own window.
_, da_spinup = _load_h(spinup_root)
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
    ax.set_title(f"SFINCS flood depth {t} — {STRATEGY} (pre) | {BASIN_ID}/{SCENARIO}")
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
