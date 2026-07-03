"""
09_create_rstart.py — Run a 10-day SFINCS spin-up to produce a restart file.

The main model's forcing files (sfincs.bzs, sfincs.dis) begin with a constant
lead period at bankfull discharge and 0 m water level before ramping up to the
event peak.  The spin-up runs SFINCS for the first `spinup_days` of that lead
period so the river network fills to a steady state.  The resulting restart file
(sfincs.rst) is written to spinup/ inside the main model directory; the main
model's sfincs.inp already references 'rstfile = spinup/sfincs.rst' so SFINCS
will automatically read it when the user starts the actual event run.

Strategy
--------
* All geometry and forcing files (dep, msk, ind, manning, ini, bnd, bzs, src,
  dis, sbg) are referenced from the spinup directory as relative `../` paths —
  no data is duplicated.
* Only sfincs.inp is written from scratch with shorter timing and restart output.
* SFINCS is executed via subprocess; the working directory is spinup/ so all
  relative paths in the spinup inp are resolved correctly.

Outputs
-------
spinup/sfincs.rst   SFINCS binary restart file at t = spinup_days.
"""

import os
import subprocess
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path

from src.log import setup_logging

log = setup_logging(snakemake.log[0])

# ── params ────────────────────────────────────────────────────────────────────
sfincs_root       = Path(snakemake.params.sfincs_root)
spinup_days       = snakemake.params.spinup_days
sfincs_exe        = Path(snakemake.params.sfincs_exe)
rst_fname         = snakemake.params.rst_fname  # e.g. sfincs.20000111.000000.rst
dtmapout_s        = int(snakemake.params.dtmapout_s)
land_polygons_path = Path(snakemake.input.land_polygons)
spinup_dir        = sfincs_root / "spinup"
spinup_inp        = spinup_dir / "sfincs.inp"

spinup_dir.mkdir(parents=True, exist_ok=True)

# ── parse main model sfincs.inp ───────────────────────────────────────────────
# Read as key=value pairs; SFINCS format uses ' = ' separator.
main_inp = sfincs_root / "sfincs.inp"
cfg: dict[str, str] = {}
with open(main_inp) as fh:
    for line in fh:
        line = line.strip()
        if "=" in line and not line.startswith("!"):
            key, _, val = line.partition("=")
            cfg[key.strip().lower()] = val.strip()

log.info(f"Parsed main sfincs.inp: {len(cfg)} parameters")

# ── compute spin-up timing ────────────────────────────────────────────────────
tref_raw = cfg.get("tref", "20000101 000000")
tref = datetime.strptime(tref_raw.replace("  ", " "), "%Y%m%d %H%M%S")
tstop_spinup = tref + timedelta(days=spinup_days)
trstout_sec = int(spinup_days * 86400)

log.info(
    f"Spin-up: {tref} → {tstop_spinup} ({spinup_days} days), "
    f"restart written at t={trstout_sec} s"
)

# ── list of data files to reference from parent directory ─────────────────────
# Map sfincs.inp key → filename in the main model directory.
PARENT_FILES = {
    "depfile":     "sfincs.dep",
    "mskfile":     "sfincs.msk",
    "indexfile":   "sfincs.ind",
    "manningfile": "sfincs.manning",
    "inifile":     "sfincs.ini",
    "sbgfile":     "sfincs.sbg",
    "srcfile":     "sfincs.src",
    "disfile":     "sfincs.dis",
    "bndfile":     "sfincs.bnd",
    "bzsfile":     "sfincs.bzs",
    "obsfile":     "sfincs.obs",  # needed for sfincs_his.nc → validation plot
}

# ── write spinup sfincs.inp ───────────────────────────────────────────────────
def fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d %H%M%S")

lines: list[str] = []

# Grid parameters — copied verbatim from main model
for key in ("mmax", "nmax", "dx", "dy", "x0", "y0", "rotation", "epsg", "crsgeo"):
    if key in cfg:
        lines.append(f"{key:<20} = {cfg[key]}")

# Timing
lines += [
    f"{'tref':<20} = {fmt_dt(tref)}",
    f"{'tstart':<20} = {fmt_dt(tref)}",
    f"{'tstop':<20} = {fmt_dt(tstop_spinup)}",
    f"{'trstout':<20} = {trstout_sec}",           # write restart at spinup end
    f"{'dthisout':<20} = 3600",                   # hourly obs-point output → sfincs_his.nc
    # dtmapout from config: controls how often zs snapshots are written → animation frame rate
    f"{'dtmapout':<20} = {dtmapout_s}",
    # dtmaxout = simulation duration: write max-envelope (zsmax) at spinup end for inundation map.
    f"{'dtmaxout':<20} = {trstout_sec}",
    f"{'dtrstout':<20} = 0",                      # no interval rst output, only trstout
]

# Physics — carry over from main model (keep same numerics)
for key in ("alpha", "huthresh", "advection", "viscosity", "nuvisc", "coriolis",
            "baro", "rhoa", "rhow", "pavbnd", "btfilter", "latitude"):
    if key in cfg:
        lines.append(f"{key:<20} = {cfg[key]}")

# Output storage — disable everything for speed
lines += [
    f"{'storevel':<20} = 0",
    f"{'storevelmax':<20} = 0",
    f"{'storecumprcp':<20} = 0",
    f"{'storemeteo':<20} = 0",
    f"{'storetwet':<20} = 0",
]

# rstfile is intentionally NOT set here.
# Setting it would make SFINCS try to READ it at startup — but it doesn't exist
# yet for a cold start and SFINCS would error ("not found!").
# Without rstfile, SFINCS starts from inifile (../sfincs.ini = spatially varying
# zsini) and writes sfincs.rst to the spinup directory when trstout fires.

# File format
lines += [
    f"{'inputformat':<20} = {cfg.get('inputformat', 'bin')}",
    f"{'outputformat':<20} = net",
]

# Data files — reference parent directory with ../
# Skip empty files (e.g. placeholder sfincs.sbg written when include_subgrid=false)
for key, fname in PARENT_FILES.items():
    fpath = sfincs_root / fname
    if fpath.exists() and fpath.stat().st_size > 0:
        lines.append(f"{key:<20} = ../{fname}")
        log.info(f"  referencing: ../{fname}")
    elif key in cfg:
        # Main model references it but via a different name — use verbatim
        lines.append(f"{key:<20} = ../{cfg[key]}")

with open(spinup_inp, "w") as fh:
    fh.write("\n".join(lines) + "\n")

log.info(f"Spinup sfincs.inp written: {spinup_inp}")

# ── run SFINCS ────────────────────────────────────────────────────────────────
# Resolve and validate the executable path before attempting to launch.
sfincs_exe = sfincs_exe.resolve()

n_threads = snakemake.threads
log.info(f"Running SFINCS: {sfincs_exe} (OMP_NUM_THREADS={n_threads})")

# Pass OMP_NUM_THREADS so SFINCS uses the number of threads Snakemake reserved
# for this job (set via threads: in the rule and config["sfincs"]["rstart"]["threads"]).
env = os.environ.copy()
env["OMP_NUM_THREADS"] = str(n_threads)

# Stream stdout and stderr line-by-line so output appears in the terminal
# immediately (via sys.stderr) AND is written to the Snakemake log file.
proc = subprocess.Popen(
    [str(sfincs_exe)],
    cwd=str(spinup_dir),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,   # line-buffered
    env=env,
)

def _forward(pipe, log_fn):
    for line in pipe:
        line = line.rstrip()
        if line:
            log_fn(f"[sfincs] {line}")
            print(f"[sfincs] {line}", file=sys.stderr, flush=True)

t_out = threading.Thread(target=_forward, args=(proc.stdout, log.info))
t_err = threading.Thread(target=_forward, args=(proc.stderr, log.warning))
t_out.start()
t_err.start()
t_out.join()
t_err.join()

try:
    proc.wait(timeout=7200)   # 2-hour total timeout
except subprocess.TimeoutExpired:
    proc.kill()
    raise RuntimeError("SFINCS spin-up exceeded 2-hour timeout")

if proc.returncode != 0:
    raise RuntimeError(
        f"SFINCS spin-up failed with exit code {proc.returncode}. "
        f"Check log: {snakemake.log[0]}"
    )

# SFINCS restart files sfincs.YYYYMMDD.HHMMSS.rst (timestamp at trstout).
# rst_fname is computed in the rule from tref + spinup_days and passed as a param.
rst_path = spinup_dir / rst_fname
if not rst_path.exists():
    written = [f.name for f in spinup_dir.iterdir() if f.is_file()]
    raise FileNotFoundError(
        f"SFINCS ran but expected restart file not found: {rst_path}\n"
        f"Files in spinup dir: {written}"
    )

log.info(f"Restart file written: {rst_path} ({rst_path.stat().st_size / 1e6:.1f} MB)")

# ── validation plot ───────────────────────────────────────────────────────────
# Plot water level timeseries at all observation points.  Convergence by day
# spinup_days is visible as flat lines at the right edge of the plot.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

plt.ioff()
plot_path = Path(snakemake.output.plot_spinup)
his_path  = spinup_dir / "sfincs_his.nc"

if not his_path.exists():
    log.warning(f"No history file found at {his_path} — skipping validation plot")
    plot_path.touch()   # empty sentinel so Snakemake output check passes
else:
    # Time coordinate: SFINCS writes seconds since tref as a CF time variable.
    # Try standard CF decoding first; fall back to raw seconds.
    try:
        ds = xr.open_dataset(his_path)
        t = ds["time"].values
        if np.issubdtype(t.dtype, np.datetime64):
            t_days = (t - t[0]) / np.timedelta64(1, "D")
        else:
            t_days = np.asarray(t, dtype=float) / 86400.0
    except Exception:
        ds = xr.open_dataset(his_path, decode_times=False)
        t_days = np.asarray(ds["time"].values, dtype=float) / 86400.0

    # SFINCS v2.3 uses 'point_zs'; older versions used 'zs'. Try both.
    zs_var = next((v for v in ("point_zs", "zs") if v in ds), None)
    if zs_var is None:
        log.warning(
            f"No water-level variable found in {his_path}. "
            f"Available: {list(ds.data_vars)}. Skipping plot."
        )
        plot_path.touch()
    else:
        zs = ds[zs_var].values   # expected shape: (time, stations) or (stations, time)
        if zs.ndim == 1:
            zs = zs[:, np.newaxis]
        if zs.shape[0] != len(t_days):
            zs = zs.T   # transpose to (time, stations)
        n_stations = zs.shape[1]

        fig, ax = plt.subplots(figsize=(11, 5))
        for i in range(n_stations):
            ax.plot(t_days, zs[:, i], lw=1.0, alpha=0.75, label=f"obs {i + 1}")

        ax.axvline(
            spinup_days, color="red", linestyle="--", linewidth=1.2,
            label=f"Day {spinup_days} (restart written)",
        )
        ax.set_xlabel("Time since spin-up start (days)")
        ax.set_ylabel(f"Water level — {zs_var} (m)")
        ax.set_title(
            f"Spin-up validation — water level at {n_stations} observation points\n"
            f"Basin {Path(sfincs_root).parent.name} | "
            f"Lines should be near-flat at day {spinup_days} if spin-up is sufficient"
        )
        ax.legend(fontsize=7, ncol=max(1, n_stations // 5), loc="upper left")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Validation plot written: {plot_path}")

# ── max inundation depth map ──────────────────────────────────────────────────
# sfincs_map.nc is created by SFINCS because dtmaxout = trstout_sec.
# We read zsmax (max water surface) and zb (bed level) from it to produce a
# grid-resolution inundation depth plot.  Rule 10 reads sfincs_map.nc directly
# via hydromt_sfincs for the high-resolution downscaled sanity-check map.

map_path             = spinup_dir / "sfincs_map.nc"
plot_inundation_path = Path(snakemake.output.plot_max_inundation)

if not map_path.exists():
    log.warning(f"No map file at {map_path} — creating empty inundation plot sentinel")
    plot_inundation_path.touch()
else:
    try:
        ds_map = xr.open_dataset(map_path, decode_times=False)
        log.info(f"sfincs_map.nc variables: {list(ds_map.data_vars)}")

        hmax = None

        # Try hmax directly
        for _var in ("hmax", "h_max"):
            if _var in ds_map:
                _h = ds_map[_var].values
                # Take maximum over the timemax dimension (not just the last step)
                hmax = _h.max(axis=0) if _h.ndim == 3 else _h
                log.info(f"Using '{_var}' for max inundation depth")
                break

        # Fall back to zsmax − zb
        if hmax is None:
            zsmax_arr = None
            zb_arr = None
            for _var in ("zsmax", "zs_max"):
                if _var in ds_map:
                    _zs = ds_map[_var].values
                    # max over the timemax dimension gives the true envelope maximum
                    zsmax_arr = _zs.max(axis=0) if _zs.ndim == 3 else _zs
                    log.info(f"Using '{_var}' for max water surface elevation")
                    break
            for _var in ("zb", "dep"):
                if _var in ds_map:
                    _zb = ds_map[_var].values
                    zb_arr = _zb.squeeze() if _zb.ndim > 2 else _zb
                    log.info(f"Using '{_var}' for bed level")
                    break

            if zsmax_arr is not None and zb_arr is not None:
                nodata = (zsmax_arr < -9990) | (zb_arr < -9990)
                hmax = np.where(nodata, np.nan, np.maximum(zsmax_arr - zb_arr, 0.0))

        if hmax is None:
            log.warning(
                f"Cannot extract max inundation depth from {map_path}. "
                f"Available vars: {list(ds_map.data_vars)}"
            )
            plot_inundation_path.touch()
        else:
            x0_val   = float(cfg.get("x0", 0))
            y0_val   = float(cfg.get("y0", 0))
            dx_val   = float(cfg.get("dx", 1))
            dy_val   = float(cfg.get("dy", 1))
            mmax_val = int(cfg["mmax"]) if "mmax" in cfg else hmax.shape[-1]
            nmax_val = int(cfg["nmax"]) if "nmax" in cfg else hmax.shape[-2]

            from src.plots import plot_max_inundation_depth
            extent = (x0_val, x0_val + mmax_val * dx_val,
                      y0_val, y0_val + nmax_val * dy_val)
            plot_max_inundation_depth(
                hmax, extent, cfg["epsg"], str(plot_inundation_path),
                basin_id=Path(sfincs_root).parent.name,
                osm_land_path=str(land_polygons_path),
            )
            log.info(f"Max inundation plot written: {plot_inundation_path}")

        ds_map.close()

    except Exception as exc:
        log.warning(f"Max inundation extraction failed: {exc}")
        plot_inundation_path.touch()
