"""Run the basin-level SFINCS spin-up that writes the restart file for every event.

Run by rule run_spinup, once per basin; independent of any scenario's design
RP. Geometry and the default initial condition come from the skeleton by
relative path (src.sfincs_run.forward_geometry_files), as in 13_build_sfincs.py.

Forcing, held constant over spinup_days: each crossing's mean_discharge_m3s (the
record's grand mean, the base flow every event hydrograph starts from) and
each station's calm-sea level (calm_sea_levels, where every event's boundary
lead-in starts). Matching both avoids a jump at the restart; an RP=1 spin-up
sea dropped the boundary 0.30 m at the event start (basin 2433835) and
overtopped the coastal dike during spin-up.

Writes, under results/{basin_id}/spin_up/: sfincs.inp, the restart
sfincs.YYYYMMDD.HHMMSS.rst at t = spinup_days, sfincs_map.nc/sfincs_his.nc,
and the validation plots.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Polygon
from hydromt_sfincs import SfincsModel

from src.log import setup_logging
from src.plots import plot_max_inundation_map, plot_water_level_timeseries
from src.river_network import outflow_crossing_mask
from src.postprocessing import compute_max_inundation
from src.river_burn import build_centerline_cells_regular
from src.sfincs_run import (
    FORCING_FILE_KEYS,
    parse_sfincs_inp,
    place_discharge_points,
    run_sfincs_subprocess,
    write_sfincs_inp,
)
from src.surge import calm_sea_levels, read_calm_sea_level_m

log = setup_logging(snakemake.log[0])

# ── params ──
skeleton_root       = Path(snakemake.params.skeleton_root)
spin_up_root        = Path(snakemake.params.spin_up_root)
resolution_m          = snakemake.params.resolution_m
tref_str            = snakemake.params.tref
spinup_days         = snakemake.params.spinup_days
sfincs_exe          = Path(snakemake.params.sfincs_exe)
rst_fname           = snakemake.params.rst_fname  # e.g. sfincs.20000102.000000.rst
dtmapout_s          = int(snakemake.params.dtmapout_s)
dthisout_s          = int(snakemake.params.dthisout_s)
timeout_s           = int(snakemake.params.timeout_s)
water_level_buffer_m = snakemake.params.water_level_buffer_m
boundary_ramp_hr = float(snakemake.params.boundary_ramp_hr)
land_polygons_path  = Path(snakemake.input.land_mask_on_grid)  # plot background
sea_mask_path       = Path(snakemake.input.sea_mask)
river_network_path  = Path(snakemake.input.clean_river_network)
domain_gpkg_path    = Path(snakemake.input.domain_gpkg)
surge_forcing_path  = Path(snakemake.input.surge_forcing)
river_forcing_path  = Path(snakemake.input.river_forcing)

spin_up_root.mkdir(parents=True, exist_ok=True)

_domain_gdf = gpd.read_file(domain_gpkg_path)
if _domain_gdf.crs is not None and _domain_gdf.crs.to_epsg() != 4326:
    _domain_gdf = _domain_gdf.to_crs("EPSG:4326")
_union = _domain_gdf.geometry.union_all()
domain_poly = cast(Polygon, _union if isinstance(_union, Polygon) else _union.convex_hull)

# ── load skeleton, redirect writes to spin_up_root ──
sf = SfincsModel(root=str(skeleton_root), mode="r")
sf.read()
sf.root.set(spin_up_root, mode="w+")
log.info(f"Skeleton loaded from {skeleton_root}, writes redirected to {spin_up_root}")

skeleton_cfg = parse_sfincs_inp(skeleton_root / "sfincs.inp")

# ── timing ──
tref = datetime.strptime(tref_str, "%Y-%m-%d %H:%M:%S")
tstop = tref + timedelta(days=spinup_days)
trstout_s = int(spinup_days * 86400)
spinup_times = pd.DatetimeIndex([tref, tstop])

log.info(f"Spin-up: {tref} → {tstop} ({spinup_days} days), restart written at t={trstout_s} s")

# The .create() calls slice timeseries by the in-memory model time. The
# skeleton's sfincs.inp sets no tref/tstart/tstop, so the defaults (today's
# date) would not overlap the forcing and raise NoDataException ("DataFrame has
# no data after time slicing"). Only affects in-memory config; sfincs.inp is
# hand-written below.
sf.config.set("tref", tref)
sf.config.set("tstart", tref)
sf.config.set("tstop", tstop)

# ── water-level boundary forcing: calm sea ──
# The sea starts at calm_sea_level_m (skeleton zsini, the basin mean), not each
# station's calm level, so the boundary ramps between them over
# boundary_ramp_hr. A step sends in a front that shoals and reflects at the
# coast (a 0.30 m step built up to +1.1 m at the 0.40 m coastal dike of basin
# 2433835).
surge_ds = xr.open_dataset(surge_forcing_path, decode_times=False)
n_stations = surge_ds.sizes["station"]
stations_gdf = gpd.GeoDataFrame(
    {"index": range(n_stations)},
    geometry=gpd.points_from_xy(surge_ds.longitude.values, surge_ds.latitude.values),
    crs="EPSG:4326",
)
start_level = read_calm_sea_level_m(surge_ds)
calm_levels = calm_sea_levels(surge_ds)  # (n_station,)
ramp_end = tref + timedelta(hours=min(boundary_ramp_hr, spinup_days * 24.0))
wl_times = pd.DatetimeIndex(sorted({tref, ramp_end, tstop}))
wl_values = np.array([np.full(n_stations, start_level) if t == tref else calm_levels for t in wl_times])
wl_df = pd.DataFrame(data=wl_values, index=wl_times, columns=range(n_stations))
sf.water_level.create(timeseries=wl_df, locations=stations_gdf, buffer=water_level_buffer_m)
log.info(
    f"Water-level forcing: {n_stations} station(s) at the calm-sea level "
    f"({calm_levels.min():+.2f}..{calm_levels.max():+.2f} m; ramped from the starting sea "
    f"{start_level:+.2f} m over {boundary_ramp_hr:.1f} h), held until {tstop}"
)
sf.water_level.write()

# ── river discharge forcing: mean discharge, constant over time ──
river_ds = xr.open_dataset(river_forcing_path, decode_times=False)
# Crossings on a seed that flows out of the network (rule 08's
# 'seed_flows_out') inject nothing -- they are free-outflow boundaries.
active = river_ds.has_glofas.values.astype(bool) & ~outflow_crossing_mask(
    river_ds, gpd.read_file(river_network_path)
)
n_active = int(active.sum())
log.info(f"River forcing: {n_active}/{len(active)} crossings with valid GloFAS data")

if n_active == 0:
    log.warning("No active river crossings — discharge forcing skipped")
else:
    base_discharge = river_ds["mean_discharge_m3s"].values[active]  # (n_active,)

    crossings_gdf = gpd.GeoDataFrame(
        {"index": range(n_active)},
        geometry=gpd.points_from_xy(
            river_ds.longitude.values[active], river_ds.latitude.values[active],
        ),
        crs="EPSG:4326",
    )
    inside_reach_ids = (
        river_ds.inside_reach_id.values[active] if "inside_reach_id" in river_ds else [None] * n_active
    )

    dis_df = pd.DataFrame(
        data=np.tile(base_discharge, (len(spinup_times), 1)),
        index=spinup_times,
        columns=range(n_active),
    )

    rivers_utm = gpd.read_file(river_network_path).to_crs(sf.crs)
    centerline_cells = build_centerline_cells_regular(
        rivers_utm, sf.grid.data["dep"].shape, sf.grid.data["dep"].raster.transform
    )
    crossings_filt = place_discharge_points(
        sf, crossings_gdf, dis_df, centerline_cells, inside_reach_ids, resolution_m
    )
    log.info(f"Discharge forcing: {len(crossings_filt)}/{n_active} source point(s) at mean discharge, held constant")
sf.discharge_points.write()

# ── sfincs.inp (hand-written) ──
# inifile is always forwarded: spin-up always starts from the calm_sea_level_m sea.
# No rstfile: SFINCS would try to read it at startup; this is a cold start
# from inifile.
spinup_inp = write_sfincs_inp(
    spin_up_root / "sfincs.inp",
    skeleton_cfg,
    skeleton_root,
    {
        "tref": tref.strftime("%Y%m%d %H%M%S"),
        "tstart": tref.strftime("%Y%m%d %H%M%S"),
        "tstop": tstop.strftime("%Y%m%d %H%M%S"),
        "trstout": trstout_s,
        "dtrstout": 0,  # restart only at trstout
        "dthisout": dthisout_s,
        "dtmapout": dtmapout_s,
        "dtmaxout": trstout_s,  # zsmax over the whole spin-up
        "baro": 0,
        "zsini": -9999.0,
        # Spin-up needs none of these instantaneous/diagnostic fields.
        "storevel": 0,
        "storevelmax": 0,
        "storecumprcp": 0,
        "storemeteo": 0,
        "storetwet": 0,
    },
    {k: sf.config.get(k) for k in FORCING_FILE_KEYS},
)
log.info(f"Spinup sfincs.inp written: {spinup_inp}")

# ── run SFINCS ──
run_sfincs_subprocess(sfincs_exe, spin_up_root, timeout_s, log, label="SFINCS spin-up", n_threads=snakemake.threads)

rst_path = spin_up_root / rst_fname
if not rst_path.exists():
    written = [f.name for f in spin_up_root.iterdir() if f.is_file()]
    raise FileNotFoundError(
        f"SFINCS ran but expected restart file not found: {rst_path}\n"
        f"Files in spin_up dir: {written}"
    )
log.info(f"Restart file written: {rst_path} ({rst_path.stat().st_size / 1e6:.1f} MB)")

# ── validation plot ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.ioff()
plot_path = Path(snakemake.output.plot_spinup)
his_path  = spin_up_root / "sfincs_his.nc"

if not his_path.exists():
    raise FileNotFoundError(f"SFINCS ran but expected history file not found: {his_path}")
try:
    ds = xr.open_dataset(his_path)
except ValueError:
    # Time units SFINCS wrote that xarray cannot decode: use raw seconds.
    ds = xr.open_dataset(his_path, decode_times=False)
t = ds["time"].values
if np.issubdtype(t.dtype, np.datetime64):
    t_days = (t - t[0]) / np.timedelta64(1, "D")
else:
    t_days = np.asarray(t, dtype=float) / 86400.0

zs_var = next((v for v in ("point_zs", "zs") if v in ds), None)
if zs_var is None:
    raise KeyError(f"No water-level variable (point_zs, zs) in {his_path}; available: {list(ds.data_vars)}")
zs = ds[zs_var].values
if zs.ndim == 1:
    zs = zs[:, np.newaxis]
if zs.shape[0] != len(t_days):
    zs = zs.T

plot_water_level_timeseries(
    t_days, zs, plot_path,
    day_markers=[(spinup_days, f"Day {spinup_days} (restart written)")],
    basin_id=f"Basin {spin_up_root.parent.name}",
    run_label=f"Spin-up validation | lines should be near-flat at day {spinup_days} if spin-up is sufficient",
    ylabel=f"Water level — {zs_var} (m)",
)
log.info(f"Validation plot written: {plot_path}")

# ── max inundation depth map ──
plot_inundation_path = Path(snakemake.output.plot_max_inundation)

da_hmax, _da_dep = compute_max_inundation(
    spin_up_root, skeleton_root, sea_mask_path, hmin=0.0,
)
if da_hmax is None:
    raise RuntimeError(f"No 'zsmax' in the SFINCS spin-up output of {spin_up_root}")
plot_max_inundation_map(
    da_hmax, domain_poly, str(land_polygons_path), str(river_network_path),
    str(plot_inundation_path), basin_id=spin_up_root.parent.name, run_label="spinup",
)
log.info(f"Max inundation plot written: {plot_inundation_path}")

