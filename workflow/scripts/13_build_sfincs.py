"""Build a scenario's forcing and sfincs.inp on top of the basin's SFINCS skeleton.

Run by rule build_sfincs (and adapt_build_forcing_pre). Loads the skeleton,
redirects writes to the scenario directory (sf.root.set), writes only the
water-level and discharge forcing, and hand-writes sfincs.inp. Skeleton grid
scalars and non-forcing "*file" entries are referenced by os.path.relpath back
to the skeleton (src.sfincs_run.forward_geometry_files), never copied: HydroMT's
config writer absolutizes any file reference outside the model root.

forcing_mode (from scenario river_rp/surge_rp, scenario_params in 00_common.smk):
  "compound"     — surge/tide boundary + river discharge. compound.lag_hr
                   shifts discharge relative to surge (positive = river peak
                   later).
  "coastal_only" — surge/tide boundary; discharge points kept with a zeroed
                   timeseries so the model structure matches other modes.
  "river_only"   — river discharge; flat boundary at river_only_flat_level_m,
                   below terrain.gebco_max_depth_m so the coast stays dry; no
                   inifile forwarded (uniform dry start).

discharge_multiplier (default 1.0) and slr_m (default 0.0, global-mean SLR in
m scaled by each station's slr_fingerprint) are applied here at build time,
never baked into river_forcing.nc / surge_forcing.nc, so changing them reruns
only this scenario's build and event run.
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
from hydromt_sfincs import SfincsModel

from src.geometry import snap_points_into_region
from src.river_forcing import build_design_discharge_matrix
from src.river_network import outflow_crossing_mask
from src.sfincs_run import forward_geometry_files, parse_sfincs_inp
from src.surge import build_design_surge_matrix

# ── logging ──
logging.basicConfig(
    filename=snakemake.log[0],
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)
for _name in ("hydromt", "hydromt_sfincs"):
    _l = logging.getLogger(_name)
    _l.setLevel(logging.INFO)
    _l.handlers = []
    _l.addHandler(logging.FileHandler(snakemake.log[0]))

# ── paths & params ──
surge_forcing_path = Path(snakemake.input.surge_forcing)
river_forcing_path = Path(snakemake.input.river_forcing)
skeleton_root       = Path(snakemake.params.skeleton_root)
sfincs_root         = Path(snakemake.params.sfincs_root)
spin_up_root        = Path(snakemake.params.spin_up_root)
resolution        = snakemake.params.resolution

tref_str          = snakemake.params.tref
dtmapout          = snakemake.params.dtmapout
dtmaxout          = snakemake.params.dtmaxout
dthisout          = snakemake.params.dthisout
storevelmax       = snakemake.params.storevelmax
storetwet         = snakemake.params.storetwet
forcing_mode      = snakemake.params.forcing_mode
river_only_flat_level_m = float(snakemake.params.river_only_flat_level_m)
# Design RPs may be None (no driver on that side: zero discharge), "mean"
# (river: constant mean-discharge hydrograph) or "tide" (surge: each station's
# representative tidal cycle); only numeric values are coerced to float.
# design_rp_surge_yr is only used outside river_only, where it is non-None.
design_rp_river_yr = snakemake.params.design_rp_river_yr
if isinstance(design_rp_river_yr, str) and design_rp_river_yr.strip().lower() == "mean":
    pass
elif design_rp_river_yr is not None:
    design_rp_river_yr = float(design_rp_river_yr)
design_rp_surge_yr = snakemake.params.design_rp_surge_yr
if isinstance(design_rp_surge_yr, str) and design_rp_surge_yr.strip().lower() == "tide":
    pass
elif design_rp_surge_yr is not None:
    design_rp_surge_yr = float(design_rp_surge_yr)
compound_lag_hr   = float(snakemake.params.compound_lag_hr)
river_event       = snakemake.params.river_event
discharge_multiplier = float(snakemake.params.discharge_multiplier)
# Target global-mean SLR (m); 0.0 when SLR is disabled, leaving the MDT-only fields.
effective_slr_m = float(snakemake.params.slr_m) if snakemake.params.slr_enabled else 0.0
# The spin-up restart has no SLR: the SLR term rises in over the event's first
# hours instead of stepping in at the event start (src.surge._ramp_in_slr).
slr_ramp_hours = float(snakemake.params.slr_ramp_hours)
flat_boundary_point_spacing_m = snakemake.params.flat_boundary_point_spacing_m
waterlevel_buffer_m = snakemake.params.waterlevel_buffer_m
include_rstart    = snakemake.params.include_rstart
spinup_days       = snakemake.params.spinup_days
rst_fname         = snakemake.params.rst_fname

sfincs_root.mkdir(parents=True, exist_ok=True)
log.info(f"Forcing mode: {forcing_mode!r}")

# ── load skeleton, redirect writes to the scenario directory ──
sf = SfincsModel(root=str(skeleton_root), mode="r")
sf.read()
sf.root.set(sfincs_root, mode="w+")
log.info(f"Skeleton loaded from {skeleton_root}, writes redirected to {sfincs_root}")

skeleton_cfg = parse_sfincs_inp(skeleton_root / "sfincs.inp")

# ── simulation config ──
# Forcing time is "hours since simulation start", so t=0 maps to tref and any
# fixed reference date works. tstop follows the end of the forcing files.
tref = datetime.strptime(tref_str, "%Y-%m-%d %H:%M:%S")

with xr.open_dataset(surge_forcing_path, decode_times=False) as surge_ds:
    surge_end_hr = float(surge_ds.time.max())
with xr.open_dataset(river_forcing_path, decode_times=False) as river_ds:
    river_end_hr = float(river_ds.time.max())

sim_hours = max(surge_end_hr, river_end_hr)
tstop = tref + timedelta(hours=sim_hours)

log.info(
    f"Simulation period: {tref} → {tstop} "
    f"({sim_hours:.0f} h from forcing files)"
)

# The .create() calls slice timeseries by the in-memory model time. The
# skeleton's sfincs.inp sets no tref/tstart/tstop, so the defaults (today's
# date) would not overlap the forcing and raise NoDataException ("DataFrame has
# no data after time slicing"). Only affects in-memory config; sfincs.inp is
# hand-written below.
sf.config.set("tref", tref)
sf.config.set("tstart", tref)
sf.config.set("tstop", tstop)

# ── initial conditions ──
# river_only: every cell, ocean included, starts dry at its bed level
# (zsini=-9999) and the below-GEBCO-clamp boundary keeps the coast dry, so only
# the river contributes water. Other modes forward the skeleton's
# spatially-varying inifile.
if forcing_mode == "river_only":
    ini_exclude_keys = frozenset({"inifile", "ncinifile"})
    log.info(
        "Initial conditions: uniform zsini=-9999 (dry at own bed level, "
        "incl. ocean) -- forcing_mode='river_only', boundary stays dry "
        "(below the GEBCO depth clamp) for the entire run"
    )
else:
    ini_exclude_keys = frozenset()
    log.info("Initial conditions: forwarding skeleton's own spatially-varying inifile")

# ── water-level boundary forcing (surge) ──
# Each station is matched to the nearest boundary cell (mask=2) within the buffer.
if forcing_mode == "river_only":
    sf.water_level.create_boundary_points_from_mask(bnd_dist=flat_boundary_point_spacing_m)
    sf.water_level.create_timeseries(shape="constant", offset=river_only_flat_level_m)
    log.info(
        f"Water-level forcing: flat constant boundary at {river_only_flat_level_m:+.4f} m "
        f"(forcing_mode='river_only')"
    )
else:
    surge_ds = xr.open_dataset(surge_forcing_path, decode_times=False)
    surge_times = pd.DatetimeIndex(
        [tref + timedelta(hours=float(h)) for h in surge_ds.time.values]
    )

    n_stations = surge_ds.sizes["station"]
    stations_gdf = gpd.GeoDataFrame(
        {"index": range(n_stations)},
        geometry=gpd.points_from_xy(
            surge_ds.longitude.values,
            surge_ds.latitude.values,
        ),
        crs="EPSG:4326",
    )

    wl_df = pd.DataFrame(
        data=build_design_surge_matrix(
            surge_ds, design_rp_surge_yr, slr_m=effective_slr_m, slr_ramp_hours=slr_ramp_hours
        ).T,
        index=surge_times,
        columns=range(n_stations),
    )

    sf.water_level.create(
        timeseries=wl_df,
        locations=stations_gdf,
        buffer=waterlevel_buffer_m,
    )
    log.info(
        f"Water-level forcing: {n_stations} stations, {len(surge_times)} time steps "
        f"(SLR: {effective_slr_m:+.3f} m target × per-station fingerprint, "
        f"ramped in over {slr_ramp_hours:g} h after the event start)"
    )
sf.water_level.write()

# ── river discharge forcing ──
# Only has_glofas=1 crossings have an EVA fit; the others carry zero discharge.
river_ds = xr.open_dataset(river_forcing_path, decode_times=False)
# Crossings on a seed that flows out of the network (rule 08's
# 'seed_flows_out') inject nothing -- they are free-outflow boundaries.
active = river_ds.has_glofas.values.astype(bool) & ~outflow_crossing_mask(
    river_ds, gpd.read_file(snakemake.input.river_network)
)
n_active = int(active.sum())
n_total  = len(active)
log.info(f"River forcing: {n_active}/{n_total} crossings with valid GloFAS data")

if n_active == 0:
    log.warning("No active river crossings — discharge forcing skipped")
else:
    river_times = pd.DatetimeIndex(
        [tref + timedelta(hours=float(h)) for h in river_ds.time.values]
    )

    crossings_gdf = gpd.GeoDataFrame(
        {"index": range(n_active)},
        geometry=gpd.points_from_xy(
            river_ds.longitude.values[active],
            river_ds.latitude.values[active],
        ),
        crs="EPSG:4326",
    )
    inside_reach_ids = (
        river_ds.inside_reach_id.values[active] if "inside_reach_id" in river_ds else [None] * n_active
    )

    dis_df = pd.DataFrame(
        data=build_design_discharge_matrix(
            river_ds, active, design_rp_river_yr,
            discharge_multiplier=discharge_multiplier, event=river_event,
        ).T,
        index=river_times,
        columns=range(n_active),
    )
    log.info(f"Discharge multiplier applied: x{discharge_multiplier:.3f}")

    if forcing_mode == "compound" and compound_lag_hr != 0 and len(river_times) > 1:
        dt_hr = float((river_times[1] - river_times[0]).total_seconds() / 3600.0)
        shift_steps = int(round(compound_lag_hr / dt_hr))
        if shift_steps != 0:
            # Pad the vacated side with the series' first/last value (the base
            # flow, mean discharge already multiplied) so the shift adds no step.
            arr = dis_df.to_numpy()
            shifted = np.empty_like(arr)
            n = min(abs(shift_steps), arr.shape[0])
            if shift_steps > 0:
                shifted[:n, :] = arr[0][np.newaxis, :]
                shifted[n:, :] = arr[: arr.shape[0] - n, :]
            else:
                shifted[: arr.shape[0] - n, :] = arr[n:, :]
                shifted[arr.shape[0] - n :, :] = arr[-1][np.newaxis, :]
            dis_df = pd.DataFrame(shifted, index=dis_df.index, columns=dis_df.columns)
            log.info(
                f"Compound lag applied: river discharge shifted {compound_lag_hr:+.1f} h "
                f"relative to surge ({shift_steps:+d} step(s) at dt={dt_hr:.2f} h); "
                f"{'start' if shift_steps > 0 else 'end'} padded with each "
                f"crossing's base flow"
            )

    if forcing_mode == "coastal_only":
        # Zeroed rather than skipped so the model structure matches other modes.
        dis_df.loc[:, :] = 0.0
        log.info("River discharge forced to 0.0 m3/s at all crossings (forcing_mode='coastal_only')")

    # Snap each crossing onto a cell its own reach's centerline passes through.
    from src.river_burn import build_centerline_cells_regular, snap_points_to_centerline_cells

    rivers_utm = gpd.read_file(snakemake.input.river_network).to_crs(sf.crs)
    centerline_cells = build_centerline_cells_regular(
        rivers_utm, sf.grid.data["dep"].shape, sf.grid.data["dep"].raster.transform
    )
    crossings_gdf = snap_points_to_centerline_cells(
        crossings_gdf.to_crs(sf.crs), centerline_cells,
        reach_ids=inside_reach_ids, resolution_m=resolution,
    ).to_crs("EPSG:4326")

    # Keep crossings in the active region; snap near-misses back inside.
    buf_deg = float(resolution) / 111_000.0
    region_wgs84 = sf.region.to_crs("EPSG:4326").geometry.union_all()
    crossings_filt, in_region = snap_points_into_region(crossings_gdf, region_wgs84, buf_deg)
    n_outside = int((~in_region).sum())
    if n_outside:
        log.warning(
            f"  {n_outside}/{n_active} discharge crossing(s) outside active SFINCS "
            f"region — skipped"
        )

    if crossings_filt.empty:
        log.warning("All discharge crossings outside active region — discharge forcing skipped")
    else:
        sf.discharge_points.create(
            timeseries=dis_df,
            locations=crossings_filt,
        )
        log.info(
            f"Discharge forcing: {len(crossings_filt)}/{n_active} source point(s), "
            f"{len(river_times)} time steps"
        )
sf.discharge_points.write()

# ── sfincs.inp (hand-written; see module docstring) ──
exclude_keys = frozenset({"rstfile", "bzsfile", "bndfile", "disfile", "srcfile", "netsrcdisfile"}) | ini_exclude_keys
geometry_lines = forward_geometry_files(skeleton_cfg, skeleton_root, sfincs_root, exclude=exclude_keys)

# Skeleton scalars forwarded verbatim, except the keys this script sets below.
OWNED_SCALAR_KEYS = frozenset({"tref", "tstart", "tstop", "dtmapout", "dtmaxout", "dthisout",
                                "storevelmax", "storetwet", "baro", "zsini"})
scalar_lines = [
    f"{k:<20} = {v}" for k, v in skeleton_cfg.items()
    if not k.endswith("file") and k not in OWNED_SCALAR_KEYS
]

lines = list(scalar_lines) + list(geometry_lines)

# rstfile: the basin-level run_spinup restart (RP=1 river + calm sea), a
# sibling tree of sfincs_root, referenced by a computed relative path.
if include_rstart:
    tstart_event = tref + timedelta(days=spinup_days)
    rst_rel = os.path.relpath(spin_up_root / rst_fname, start=sfincs_root)
    lines.append(f"{'rstfile':<20} = {rst_rel}")
    log.info(f"tstart updated to {tstart_event} ({spinup_days} days after tref); rstfile = {rst_rel}")
else:
    tstart_event = tref

lines += [
    f"{'tref':<20} = {tref.strftime('%Y%m%d %H%M%S')}",
    f"{'tstart':<20} = {tstart_event.strftime('%Y%m%d %H%M%S')}",
    f"{'tstop':<20} = {tstop.strftime('%Y%m%d %H%M%S')}",
    f"{'dtmapout':<20} = {dtmapout}",
    f"{'dtmaxout':<20} = {dtmaxout}",
    f"{'dthisout':<20} = {dthisout}",
    f"{'storevelmax':<20} = {storevelmax}",
    f"{'storetwet':<20} = {storetwet}",
    f"{'baro':<20} = 0",           # no wind/atmosphere data
    # HydroMT sets zsini=0.0; -9999 makes cells not in sfincs.ini start dry at
    # bed level (for river_only, every cell).
    f"{'zsini':<20} = -9999.0",
]

# Forcing files written by this script live in sfincs_root: bare filenames.
for _key in ("bndfile", "bzsfile", "srcfile", "disfile", "netsrcdisfile"):
    _val = sf.config.get(_key)
    if _val:
        lines.append(f"{_key:<20} = {_val}")

with open(sfincs_root / "sfincs.inp", "w") as fh:
    fh.write("\n".join(lines) + "\n")
log.info(f"sfincs.inp written: {sfincs_root / 'sfincs.inp'}")

# ── diagnostic plot: forcing timeseries at this scenario's design RPs ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as _mdates

plt.ioff()
figs_dir = Path(snakemake.output.sfincs_inp).parent.parent / "visuals" / "sfincs_build"
figs_dir.mkdir(parents=True, exist_ok=True)


def _hours_to_dt(hours_arr):
    return [tref + timedelta(hours=float(h)) for h in hours_arr]


with xr.open_dataset(surge_forcing_path, decode_times=False) as _sds:
    _surge_times = _hours_to_dt(_sds.time.values)
    # Rebuilt rather than read from surge_forcing.nc's 'water_level', which is
    # a basin-level preview at a fixed diagnostic RP. Mirrors the water-level
    # boundary section above, RP-for-RP and mode-for-mode, so this plot always
    # shows what was really built.
    if forcing_mode == "river_only":
        _wl = np.full((_sds.sizes["station"], len(_surge_times)), river_only_flat_level_m)
    else:
        _wl = build_design_surge_matrix(
            _sds, design_rp_surge_yr, slr_m=effective_slr_m, slr_ramp_hours=slr_ramp_hours
        )
    _n_stn = _wl.shape[0]

with xr.open_dataset(river_forcing_path, decode_times=False) as _rds:
    _river_times  = _hours_to_dt(_rds.time.values)
    _active_mask  = _rds.has_glofas.values.astype(bool)
    _n_cross      = int(_active_mask.sum())
    _dis_active   = (
        build_design_discharge_matrix(
            _rds, _active_mask, design_rp_river_yr,
            discharge_multiplier=discharge_multiplier, event=river_event,
        )
        if _n_cross > 0 else np.zeros((0, len(_river_times)))
    )

_tstart_event = tstart_event if include_rstart else None

fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

for _i in range(_n_stn):
    ax5a.plot(_surge_times, _wl[_i, :], linewidth=0.8, alpha=0.7)
ax5a.set_ylabel("Water level (m+ref)")
if forcing_mode == "river_only":
    _surge_rp_label = "flat (river_only)"
elif isinstance(design_rp_surge_yr, str):
    _surge_rp_label = design_rp_surge_yr
elif design_rp_surge_yr:
    _surge_rp_label = f"RP{design_rp_surge_yr:g}"
else:
    _surge_rp_label = "flat (no surge)"
ax5a.set_title(f"Surge boundary — water level, {_surge_rp_label} (all stations)")
ax5a.grid(True, alpha=0.3)

for _i in range(_n_cross):
    ax5b.plot(_river_times, _dis_active[_i, :], linewidth=0.8, alpha=0.7)
ax5b.set_ylabel("Discharge (m³/s)")
ax5b.set_title(f"River discharge ({_n_cross} active GloFAS crossing(s))")
ax5b.grid(True, alpha=0.3)

# Peak boxes: max over all stations/crossings and the full timeseries; the
# river boundary is a discharge boundary, so its box reports max Q.
_max_box = dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.85)
if _wl.size:
    ax5a.text(0.01, 0.97, f"Max WL = {np.nanmax(_wl):.2f} m", transform=ax5a.transAxes,
              fontsize=9, va="top", ha="left", bbox=_max_box)
if _dis_active.size:
    ax5b.text(0.01, 0.97, f"Max Q river = {np.nanmax(_dis_active):,.0f} m³/s",
              transform=ax5b.transAxes, fontsize=9, va="top", ha="left", bbox=_max_box)

if _tstart_event is not None:
    for _ax in (ax5a, ax5b):
        _ax.axvline(_tstart_event, color="black", linewidth=1.2,
                    linestyle="--", label=f"event start (after {spinup_days}d spinup)")
        _ax.legend(fontsize=8, loc="upper right")

_locator   = _mdates.AutoDateLocator()
_formatter = _mdates.ConciseDateFormatter(_locator)
ax5b.xaxis.set_major_locator(_locator)
ax5b.xaxis.set_major_formatter(_formatter)

fig5.suptitle(
    f"SFINCS forcing — basin {sfincs_root.parent.parent.name}, scenario {sfincs_root.parent.name} "
    f"(full timeseries; dashed = event model tstart)",
    fontsize=10,
)
fig5.tight_layout()
fig5.savefig(figs_dir / "05_forcing.png", dpi=150, bbox_inches="tight")
plt.close(fig5)
log.info("Plot written: 05_forcing.png")
