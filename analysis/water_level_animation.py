"""Animate a `pre` strategy's flood depth with its boundary water level (bzs).

Ad-hoc diagnostic, not part of Snakemake. Depth is h = max(zs - zb, 0) with open
sea excluded, as for 02_flood_animation.mp4 (compute_flood_progression): raw `zs`
fails as a flood threshold where the bed is below the datum, e.g. a polder at
about -8 m behind a coastal levee + pumps. Only `pre` runs have a `zs` time
series and their own sfincs.bzs; `post` writes a single max_flood_depth.tif.

Edit the CONFIG block below and run directly:
    python analysis/water_level_animation.py
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from _common import REPO_ROOT  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation

sys.path.insert(0, str(REPO_ROOT / "workflow"))

from src.postprocessing import load_sfincs_output

# CONFIG
# GCFM_RESULTS_DIR is machine-specific; see CONTRIBUTING.md "Local machine paths".
RESULTS_DIR = os.environ["GCFM_RESULTS_DIR"]
BASIN_ID = "2433835"
SCENARIO = "coast_500"
STRATEGY = "protect_closed_1"  # an already-run `pre` strategy folder
BZS_IDX = 4  # boundary point index to track (column order in sfincs.bzs)
T_START = None  # "2000-01-02 00:00:00" #, or None for full range
T_END = None
HMIN = 0.05  # flood depth [m] below which a cell is treated as dry
VMIN, VMAX = 0.0, 5.0  # colour scale for the flood depth map
FPS = 4

# paths
basin_root = Path(RESULTS_DIR) / BASIN_ID
sfincs_root = (
    basin_root / "runs" / SCENARIO / "adaptation" / "pre" / STRATEGY / "sfincs"
)
spinup_root = basin_root / "spin_up"  # shared by every scenario/strategy
# Same sea mask as rule 16's compute_flood_progression (sea_mask input).
sea_mask_path = (
    basin_root
    / "preprocessing_inputs"
    / "domain"
    / f"{BASIN_ID}_zsini_sea_cells_on_grid.tif"
)
out_path = sfincs_root.parent / "03_water_level_animation.mp4"


def _load_h(run_dir):
    # Reads sfincs_map.nc only; a bare SfincsModel.read() also loads sfincs_his.nc.
    m = load_sfincs_output(run_dir)
    if "zs" not in m.output.data:
        raise RuntimeError(f"No 'zs' in {run_dir}/sfincs_map.nc")
    if "zb" not in m.output.data:
        raise RuntimeError(f"No 'zb' in {run_dir}/sfincs_map.nc")

    # Depth, not zs: see module docstring.
    da_zb = m.output.data["zb"].squeeze()
    da_h = (m.output.data["zs"] - da_zb).clip(min=0.0)

    da_sea = m.data_catalog.get_rasterdataset(str(sea_mask_path))
    da_sea_grid = da_sea.raster.reproject_like(da_h, method="nearest")
    da_h = da_h.where(da_sea_grid != 1.0)

    da_h = da_h.where(da_h > HMIN).drop_vars("spatial_ref", errors="ignore")
    da_h.attrs.update(long_name="flood depth", unit="m")
    return m, da_h


# flood depth time series
mod, da_event = _load_h(sfincs_root)

# The event run starts at the spin-up's end (tstart != tref), so prepend the
# basin-level spin-up run to cover the whole simulated period.
_, da_spinup = _load_h(spinup_root)
# The spin-up's last frame is the event's first (restart instant).
da_h = xr.concat([da_spinup.isel(time=slice(0, -1)), da_event], dim="time")

if T_START or T_END:
    da_h = da_h.sel(time=slice(T_START, T_END))
print(
    f"Animation time range: {da_h.time.min().values} to {da_h.time.max().values} ({da_h.time.size} frames)"
)

# This strategy's own rebuilt boundary forcing.
wl = mod.get_component("water_level")
wl.read()
bzs = wl.data["bzs"]
ts_h = bzs.isel(index=BZS_IDX)
if T_START or T_END:
    ts_h = ts_h.sel(time=slice(T_START, T_END))
ts_times = ts_h.time.values


def update_plot(i, da_h, cax_h, ts_line):
    """Draw animation frame i (reads the global ax)."""
    da_hi = da_h.isel(time=i)
    t = da_hi.time.dt.strftime("%d-%B-%Y %H:%M:%S").item()
    ax.set_title(f"SFINCS flood depth {t} — {STRATEGY} (pre) | {BASIN_ID}/{SCENARIO}")
    cax_h.set_array(da_hi.values.ravel())
    current_time = da_hi.time.values  # the map's timestamp, not the bzs index
    ts_line.set_xdata([current_time, current_time])


fig, ax = mod.plot_basemap(
    fn_out=None,
    variable="",
    bmap="sat",
    plot_bounds=False,
    figsize=(11, 12),
    zoomlevel=12,
)

ax.set_position((0.1, 0.35, 0.8, 0.6))

cax_h = da_h.isel(time=0).plot(
    ax=ax,
    vmin=VMIN,
    vmax=VMAX,
    cmap=plt.cm.Blues,
    cbar_kwargs={"shrink": 0.6, "anchor": (0, 0)},
    add_labels=False,  # prevents xc/yc from overwriting axis labels
)

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
