"""Build a scenario's forcing and sfincs.inp on top of the basin's SFINCS skeleton.

Run by rule build_sfincs (and adapt_build_forcing_pre). Loads the skeleton,
redirects writes to the scenario directory (sf.root.set), writes only the
water-level and discharge forcing, and hand-writes sfincs.inp. Skeleton grid
scalars and non-forcing "*file" entries are referenced by os.path.relpath back
to the skeleton (src.sfincs_run.forward_geometry_files), never copied: HydroMT's
config writer absolutizes any file reference outside the model root.

forcing_mode (from scenario river_rp_yr/surge_rp_yr, scenario_params in 00_common.smk):
  "compound"     — surge/tide boundary + river discharge. With a surge and
                   a river RP, compound.lag_hr is the river peak minus the
                   surge peak (positive = river peak later); in storm-tide
                   mode the surge moves and the river keeps its place.
  "coastal_only" — surge/tide boundary; discharge points kept with a zeroed
                   timeseries so the model structure matches other modes.
  "river_only"   — river discharge; flat boundary at river_only_flat_level_m,
                   below terrain.gebco_max_depth_m. With the spin-up enabled
                   the run starts from the spin-up restart (sea at its calm
                   level); without it every cell starts dry (no inifile).

slr_m (default 0.0, global-mean SLR in m scaled by each station's
slr_fingerprint) is applied here at build time, never baked into
surge_forcing.nc, so changing it reruns only this scenario's build and event
run.
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

from src.log import setup_logging
from src.plots import plot_run_forcing
from src.river_burn import build_centerline_cells_regular
from src.river_forcing import build_design_discharge_matrix
from src.river_network import outflow_crossing_mask
from src.sfincs_run import (
    FORCING_FILE_KEYS,
    event_timeline,
    parse_sfincs_inp,
    place_discharge_points,
    write_sfincs_inp,
)
from src.surge import build_design_surge_matrix

# ── logging ──
setup_logging(snakemake.log[0], verbose=True)
log = logging.getLogger(__name__)
for _name in ("hydromt", "hydromt_sfincs"):
    _l = logging.getLogger(_name)
    _l.setLevel(logging.INFO)
    _l.handlers = []
    _l.addHandler(logging.FileHandler(snakemake.log[0]))

# ── paths & params ──
surge_forcing_path = Path(snakemake.input.surge_forcing)
river_forcing_path = Path(snakemake.input.river_forcing)
skeleton_root = Path(snakemake.params.skeleton_root)
sfincs_root = Path(snakemake.params.sfincs_root)
spin_up_root = Path(snakemake.params.spin_up_root)
resolution_m = snakemake.params.resolution_m

tref_str = snakemake.params.tref
dtmapout = snakemake.params.dtmapout
dtmaxout = snakemake.params.dtmaxout
dthisout = snakemake.params.dthisout
storevelmax = snakemake.params.storevelmax
storetwet = snakemake.params.storetwet
forcing_mode = snakemake.params.forcing_mode
river_only_flat_level_m = float(snakemake.params.river_only_flat_level_m)
# Design RPs may be None (no driver on that side: zero discharge), "mean"
# (river: constant mean-discharge hydrograph) or "tide" (surge: each station's
# representative tidal cycle); only numeric values are coerced to float.
# surge_rp_yr is only used outside river_only, where it is non-None.
river_rp_yr = snakemake.params.river_rp_yr
if isinstance(river_rp_yr, str) and river_rp_yr.strip().lower() == "mean":
    pass
elif river_rp_yr is not None:
    river_rp_yr = float(river_rp_yr)
surge_rp_yr = snakemake.params.surge_rp_yr
if isinstance(surge_rp_yr, str) and surge_rp_yr.strip().lower() == "tide":
    pass
elif surge_rp_yr is not None:
    surge_rp_yr = float(surge_rp_yr)
compound_lag_hr = float(snakemake.params.compound_lag_hr)
event_after_peak_hr = float(snakemake.params.event_after_peak_hr)
river_event = snakemake.params.river_event
# Target global-mean SLR (m); 0.0 when SLR is disabled, leaving the MDT-only fields.
effective_slr_m = float(snakemake.params.slr_m) if snakemake.params.slr_enabled else 0.0
# The spin-up restart has the spin-up scenario's SLR: the SLR term goes from
# it to the scenario's over the event's first hours instead of stepping at the
# event start (src.surge._ramp_in_slr).
slr_ramp_hr = float(snakemake.params.slr_ramp_hr)
spinup_slr_m = (
    float(snakemake.params.spinup_slr_m)
    if bool(snakemake.params.include_rstart)
    else 0.0
)
flat_boundary_point_spacing_m = snakemake.params.flat_boundary_point_spacing_m
water_level_buffer_m = snakemake.params.water_level_buffer_m
include_rstart = snakemake.params.include_rstart
spinup_days = snakemake.params.spinup_days
rst_fname = snakemake.params.rst_fname

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
# fixed reference date works.
tref = datetime.strptime(tref_str, "%Y-%m-%d %H:%M:%S")

surge_ds = xr.open_dataset(surge_forcing_path, decode_times=False)
river_ds = xr.open_dataset(river_forcing_path, decode_times=False)


def _is_event(rp_yr) -> bool:
    return rp_yr is not None and not isinstance(rp_yr, str)


# Storm-tide hydrograph mode: the event timeline is set here, not in rule 07d,
# so a new compound lag rebuilds the event runs only (rule 11's crests stay).
# The river keeps rule 07d's place after the restart; the surge peak, the run
# length and so the time axis follow from the lag (src.sfincs_run.event_timeline).
hydrograph_mode = "hg_peak_hr" in surge_ds.attrs
if hydrograph_mode:
    dt_hr = float(surge_ds.attrs.get("dt_hr", np.median(np.diff(surge_ds.time.values))))
    start_hr = float(surge_ds.attrs["lead_days"]) * 24.0
    river_period_hr = float(river_ds.attrs["period_hr"])
    surge_peak_hr, river_peak_hr, sim_duration_hr = event_timeline(
        start_hr,
        float(river_ds.attrs["lead_days"]) * 24.0 + river_period_hr / 2.0,
        float(surge_ds.attrs["hg_window_hr"]),
        float(surge_ds.attrs.get("hg_ramp_hr", 0.0)),
        event_after_peak_hr,
        compound_lag_hr if forcing_mode == "compound" else 0.0,
        surge_event=_is_event(surge_rp_yr),
        river_event=_is_event(river_rp_yr),
    )
    sim_duration_hr = float(np.ceil(sim_duration_hr / dt_hr) * dt_hr)
    event_times_hr = np.arange(0.0, sim_duration_hr + dt_hr / 2.0, dt_hr)
    # The builders read only the time axis, the attrs and non-time variables.
    surge_ds = surge_ds.drop_dims("time").assign_coords(time=event_times_hr)
    surge_ds.attrs["hg_peak_hr"] = surge_peak_hr
    river_ds = river_ds.drop_dims("time").assign_coords(time=event_times_hr)
    river_ds.attrs["lead_days"] = (river_peak_hr - river_period_hr / 2.0) / 24.0
    river_ds.attrs["event_start_hr"] = start_hr
    log.info(
        f"Event timeline: restart at {start_hr:.0f} h, surge peak at {surge_peak_hr:.0f} h, "
        f"river peak at {river_peak_hr:.0f} h (lag {river_peak_hr - surge_peak_hr:+.0f} h), "
        f"end at {sim_duration_hr:.0f} h"
    )
else:
    sim_duration_hr = max(float(surge_ds.time.max()), float(river_ds.time.max()))
tstop = tref + timedelta(hours=sim_duration_hr)

log.info(f"Simulation period: {tref} → {tstop} ({sim_duration_hr:.0f} h)")

# The .create() calls slice timeseries by the in-memory model time. The
# skeleton's sfincs.inp sets no tref/tstart/tstop, so the defaults (today's
# date) would not overlap the forcing and raise NoDataException ("DataFrame has
# no data after time slicing"). Only affects in-memory config; sfincs.inp is
# hand-written below.
sf.config.set("tref", tref)
sf.config.set("tstart", tref)
sf.config.set("tstop", tstop)

# ── initial conditions ──
# river_only: no inifile, so without a restart every cell, ocean included,
# starts dry at its bed level (zsini=-9999). With the spin-up enabled the
# rstfile takes precedence and the run starts from the spin-up state (sea at
# its calm level). Other modes forward the skeleton's spatially-varying inifile.
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
    sf.water_level.create_boundary_points_from_mask(
        bnd_dist=flat_boundary_point_spacing_m
    )
    sf.water_level.create_timeseries(shape="constant", offset=river_only_flat_level_m)
    log.info(
        f"Water-level forcing: flat constant boundary at {river_only_flat_level_m:+.4f} m "
        f"(forcing_mode='river_only')"
    )
else:
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
            surge_ds,
            surge_rp_yr,
            slr_m=effective_slr_m,
            slr_ramp_hr=slr_ramp_hr,
            spinup_slr_m=spinup_slr_m,
        ).T,
        index=surge_times,
        columns=range(n_stations),
    )

    sf.water_level.create(
        timeseries=wl_df,
        locations=stations_gdf,
        buffer=water_level_buffer_m,
    )
    log.info(
        f"Water-level forcing: {n_stations} stations, {len(surge_times)} time steps "
        f"(SLR: {effective_slr_m:+.3f} m target × per-station fingerprint, "
        f"ramped from the spin-up's {spinup_slr_m:+.3f} m over {slr_ramp_hr:g} h after the event start)"
    )
sf.water_level.write()

# ── river discharge forcing ──
# Only has_glofas=1 crossings have an EVA fit; the others carry zero discharge.
# Crossings on a seed that flows out of the network (rule 08's
# 'seed_flows_out') inject nothing -- they are free-outflow boundaries.
active = river_ds.has_glofas.values.astype(bool) & ~outflow_crossing_mask(
    river_ds, gpd.read_file(snakemake.input.river_network)
)
n_active = int(active.sum())
n_total = len(active)
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
        river_ds.inside_reach_id.values[active]
        if "inside_reach_id" in river_ds
        else [None] * n_active
    )

    dis_df = pd.DataFrame(
        data=build_design_discharge_matrix(
            river_ds,
            active,
            river_rp_yr,
            event=river_event,
        ).T,
        index=river_times,
        columns=range(n_active),
    )

    if (
        not hydrograph_mode
        and forcing_mode == "compound"
        and compound_lag_hr != 0
        and len(river_times) > 1
    ):
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
        log.info(
            "River discharge forced to 0.0 m3/s at all crossings (forcing_mode='coastal_only')"
        )

    rivers_utm = gpd.read_file(snakemake.input.river_network).to_crs(sf.crs)
    centerline_cells = build_centerline_cells_regular(
        rivers_utm, sf.grid.data["dep"].shape, sf.grid.data["dep"].raster.transform
    )
    crossings_filt = place_discharge_points(
        sf, crossings_gdf, dis_df, centerline_cells, inside_reach_ids, resolution_m
    )
    log.info(
        f"Discharge forcing: {len(crossings_filt)}/{n_active} source point(s), "
        f"{len(river_times)} time steps"
    )
sf.discharge_points.write()

# ── sfincs.inp (hand-written; see module docstring) ──
inp_settings: dict[str, object] = {}
# rstfile: the basin-level run_spinup restart (mean river discharge + calm
# sea), used by every forcing mode including river_only; a sibling tree of
# sfincs_root, referenced by a computed relative path.
if include_rstart:
    tstart_event = tref + timedelta(days=spinup_days)
    rst_rel = os.path.relpath(spin_up_root / rst_fname, start=sfincs_root)
    inp_settings["rstfile"] = rst_rel
    log.info(
        f"tstart updated to {tstart_event} ({spinup_days} days after tref); rstfile = {rst_rel}"
    )
else:
    tstart_event = tref

inp_settings |= {
    "tref": tref.strftime("%Y%m%d %H%M%S"),
    "tstart": tstart_event.strftime("%Y%m%d %H%M%S"),
    "tstop": tstop.strftime("%Y%m%d %H%M%S"),
    "dtmapout": dtmapout,
    "dtmaxout": dtmaxout,
    "dthisout": dthisout,
    "storevelmax": storevelmax,
    "storetwet": storetwet,
    "baro": 0,  # no wind/atmosphere data
    # HydroMT sets zsini=0.0; -9999 makes cells not in sfincs.ini start dry at
    # bed level (for river_only without a restart, every cell).
    "zsini": -9999.0,
}

# Forcing files written by this script live in sfincs_root: bare filenames.
write_sfincs_inp(
    sfincs_root / "sfincs.inp",
    skeleton_cfg,
    skeleton_root,
    inp_settings,
    {k: sf.config.get(k) for k in FORCING_FILE_KEYS},
    exclude_files=ini_exclude_keys,
)
log.info(f"sfincs.inp written: {sfincs_root / 'sfincs.inp'}")

# ── forcing figure: the boundary timeseries as written for SFINCS ──
plot_run_forcing(
    sfincs_root,
    snakemake.output.plot_forcing,
    snakemake.params.basin_id,
    snakemake.params.run_label,
)
log.info(f"Plot written: {snakemake.output.plot_forcing}")
