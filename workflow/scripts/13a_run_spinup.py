"""Run the SFINCS spin-up of one slr_m that writes the restart file for its events.

Run by rule run_spinup, once per basin and slr_m the requested scenarios use;
independent of any design RP. Geometry and the default initial condition come
from the skeleton by relative path (src.sfincs_run.forward_geometry_files), as
in 14_build_sfincs.py.

Forcing, held constant: each crossing's mean_discharge_m3s (the record's grand
mean, the base flow every event hydrograph starts from) and each station's
calm-sea level plus slr_m x fingerprint (calm_sea_levels, where the events of
that slr_m start). Matching both avoids a jump at the restart; an RP=1 spin-up
sea dropped the boundary 0.30 m at the event start (basin 2433835) and
overtopped the coastal dike during spin-up.

Runs in chunks of chunk_hr, each restarting from the last, until rule 11's
steady test (steady_check window_hr / tol_m, both directions) passes on the
river-network cells, at most max_days; a spin-up still not steady is kept and
marked (stability.csv steady, "Check stability" banner). Writes, under
spin_up/slr_<slr_m>/: sfincs.inp (last chunk), the restart at the end
(sfincs.spinup.rst), the last chunk's sfincs_map.nc, the observation series
of all chunks (sfincs_his.nc), stability.csv (spinup_hr, the events' restart
time; steady test; river-network swing check over all chunks) and the
forcing and max flood depth figures.
"""

import gc
import os
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
from src.plots import (
    plot_max_inundation_map,
    plot_run_forcing,
    stamp_unstable,
)
from src.river_network import outflow_crossing_mask
from src.postprocessing import compute_max_inundation, read_land_bed_level
from src.river_burn import build_centerline_cells_regular
from src.sfincs_run import (
    FORCING_FILE_KEYS,
    parse_sfincs_inp,
    place_discharge_points,
    river_network_stability,
    river_network_steady,
    run_sfincs_subprocess,
    stability_note,
    write_sfincs_inp,
)
from src.surge import calm_sea_levels, read_calm_sea_level_m

log = setup_logging(snakemake.log[0])

# ── params ──
skeleton_root = Path(snakemake.params.skeleton_root)
spin_up_root = Path(snakemake.params.spin_up_root)
resolution_m = snakemake.params.resolution_m
tref_str = snakemake.params.tref
slr_m = float(snakemake.params.slr_m)
chunk_hr = float(snakemake.params.chunk_hr)
max_hr = float(snakemake.params.max_days) * 24.0
steady_cfg = dict(snakemake.params.steady_check)
sfincs_exe = Path(snakemake.params.sfincs_exe)
rst_fname = snakemake.params.rst_fname  # fixed name (00_common RST_FNAME)
dtmapout_s = int(snakemake.params.dtmapout_s)
dthisout_s = int(snakemake.params.dthisout_s)
timeout_s = int(snakemake.params.timeout_s)
water_level_buffer_m = snakemake.params.water_level_buffer_m
boundary_ramp_hr = float(snakemake.params.boundary_ramp_hr)
land_polygons_path = Path(snakemake.input.land_mask_on_grid)  # plot background
land_bed_level_path = Path(snakemake.input.land_bed_level)
river_network_path = Path(snakemake.input.clean_river_network)
domain_gpkg_path = Path(snakemake.input.domain_gpkg)
surge_forcing_path = Path(snakemake.input.surge_forcing)
river_forcing_path = Path(snakemake.input.river_forcing)

spin_up_root.mkdir(parents=True, exist_ok=True)

_domain_gdf = gpd.read_file(domain_gpkg_path)
if _domain_gdf.crs is not None and _domain_gdf.crs.to_epsg() != 4326:
    _domain_gdf = _domain_gdf.to_crs("EPSG:4326")
_union = _domain_gdf.geometry.union_all()
domain_poly = cast(
    Polygon, _union if isinstance(_union, Polygon) else _union.convex_hull
)

# ── load skeleton, redirect writes to spin_up_root ──
sf = SfincsModel(root=str(skeleton_root), mode="r")
sf.read()
sf.root.set(spin_up_root, mode="w+")
log.info(f"Skeleton loaded from {skeleton_root}, writes redirected to {spin_up_root}")

skeleton_cfg = parse_sfincs_inp(skeleton_root / "sfincs.inp")

# ── timing: the forcing spans every chunk; each run covers one ──
tref = datetime.strptime(tref_str, "%Y-%m-%d %H:%M:%S")
n_chunks_max = int(np.ceil(max_hr / chunk_hr))
tstop = tref + timedelta(hours=n_chunks_max * chunk_hr)
spinup_times = pd.DatetimeIndex([tref, tstop])

log.info(
    f"Spin-up at slr_m {slr_m:g}: chunks of {chunk_hr:g} h until steady, at most "
    f"{n_chunks_max} ({n_chunks_max * chunk_hr:g} h)"
)

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
# Each station's calm level plus slr_m x fingerprint.
calm_levels = calm_sea_levels(surge_ds, slr_m)  # (n_station,)
ramp_end = tref + timedelta(hours=min(boundary_ramp_hr, chunk_hr))
wl_times = pd.DatetimeIndex(sorted({tref, ramp_end, tstop}))
wl_values = np.array(
    [np.full(n_stations, start_level) if t == tref else calm_levels for t in wl_times]
)
wl_df = pd.DataFrame(data=wl_values, index=wl_times, columns=range(n_stations))
sf.water_level.create(
    timeseries=wl_df, locations=stations_gdf, buffer=water_level_buffer_m
)
log.info(
    f"Water-level forcing: {n_stations} station(s) at the calm-sea level + {slr_m:g} m SLR "
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
    log.info(
        f"Discharge forcing: {len(crossings_filt)}/{n_active} source point(s) at mean discharge, held constant"
    )
sf.discharge_points.write()

# ── chunks: hand-written sfincs.inp, run, steady test ──
# Chunk 1 is a cold start from the skeleton's inifile; every later chunk
# restarts from the previous chunk's restart (rstfile, no inifile).
rivers = gpd.read_file(river_network_path)
land_bed_level = read_land_bed_level(land_bed_level_path)
CHUNK_RST = "sfincs.chunk.rst"
his_parts, swings, hmax_parts = [], [], []
for k in range(1, n_chunks_max + 1):
    t0 = tref + timedelta(hours=(k - 1) * chunk_hr)
    t1 = tref + timedelta(hours=k * chunk_hr)
    settings = {
        "tref": tref.strftime("%Y%m%d %H%M%S"),
        "tstart": t0.strftime("%Y%m%d %H%M%S"),
        "tstop": t1.strftime("%Y%m%d %H%M%S"),
        "trstout": int(k * chunk_hr * 3600),  # seconds since tref, like the forcing
        "dtrstout": 0,  # restart only at trstout
        "dthisout": dthisout_s,
        "dtmapout": dtmapout_s,
        "dtmaxout": int(chunk_hr * 3600),  # zsmax over the chunk
        "baro": 0,
        "zsini": -9999.0,
        # Spin-up needs none of these instantaneous/diagnostic fields.
        "storevel": 0,
        "storevelmax": 0,
        "storecumprcp": 0,
        "storemeteo": 0,
        "storetwet": 0,
    }
    if k > 1:
        settings["rstfile"] = CHUNK_RST
    write_sfincs_inp(
        spin_up_root / "sfincs.inp",
        skeleton_cfg,
        skeleton_root,
        settings,
        {key: sf.config.get(key) for key in FORCING_FILE_KEYS},
        exclude_files=frozenset({"inifile", "ncinifile"}) if k > 1 else frozenset(),
    )
    run_sfincs_subprocess(
        sfincs_exe,
        spin_up_root,
        timeout_s,
        log,
        label=f"SFINCS spin-up chunk {k}",
        n_threads=snakemake.threads,
    )
    # SFINCS names the restart by its time; the workflow uses fixed names.
    sfincs_rst = spin_up_root / f"sfincs.{t1.strftime('%Y%m%d.%H%M%S')}.rst"
    if not sfincs_rst.exists():
        written = [f.name for f in spin_up_root.iterdir() if f.is_file()]
        raise FileNotFoundError(
            f"SFINCS ran but expected restart file not found: {sfincs_rst}\n"
            f"Files in spin_up dir: {written}"
        )
    os.replace(sfincs_rst, spin_up_root / CHUNK_RST)

    swings.append(
        river_network_stability(
            spin_up_root, rivers, float(snakemake.params.alpha_reversal_max_m)
        )
    )
    steady = river_network_steady(
        spin_up_root, rivers, float(steady_cfg["window_hr"]), float(steady_cfg["tol_m"])
    )
    hmax_k, _ = compute_max_inundation(spin_up_root, land_bed_level, hmin=0.0)
    if hmax_k is None:
        raise RuntimeError(f"No 'zsmax' in the SFINCS spin-up output of {spin_up_root}")
    hmax_parts.append(hmax_k)
    part = spin_up_root / f"sfincs_his_chunk{k:02d}.nc"
    os.replace(spin_up_root / "sfincs_his.nc", part)
    his_parts.append(part)
    log.info(
        f"Spin-up chunk {k} ({(k - 1) * chunk_hr:g}-{k * chunk_hr:g} h): largest "
        f"{steady_cfg['window_hr']:g} h mean level change on the river network "
        f"{steady['max_level_change_m']:.3f} m ({steady['cells_changing']} cell(s) above "
        f"{steady_cfg['tol_m']:g} m) -- {'steady' if steady['steady'] else 'not steady'}"
    )
    if steady["steady"] or k == n_chunks_max:
        break
    # A rerun in the same directory cannot recreate netCDF files SFINCS left
    # behind on Windows; the map's readers above may still hold it open.
    gc.collect()
    xr.backends.file_manager.FILE_CACHE.clear()
    (spin_up_root / "sfincs_map.nc").unlink()

spinup_hr = k * chunk_hr
rst_path = spin_up_root / rst_fname
os.replace(spin_up_root / CHUNK_RST, rst_path)
log.info(
    f"Restart file written: {rst_path} at {spinup_hr:g} h "
    f"({rst_path.stat().st_size / 1e6:.1f} MB)"
)

# Observation series of all chunks, for the spin-up validation plot (rule 13b).
his = xr.concat(
    [xr.load_dataset(p_) for p_ in his_parts],
    dim="time",
    data_vars="minimal",
    coords="minimal",
    compat="override",
)
his.to_netcdf(spin_up_root / "sfincs_his.nc")
for part in his_parts:
    part.unlink()

basin_id = spin_up_root.parents[1].name

# ── forcing figure ──
plot_run_forcing(spin_up_root, snakemake.output.plot_forcing, basin_id, "spinup")

# ── max inundation depth map, over all chunks ──
plot_max_inundation_map(
    xr.concat(hmax_parts, dim="chunk").max("chunk", skipna=True),
    domain_poly,
    str(land_polygons_path),
    str(river_network_path),
    str(snakemake.output.plot_max_depth),
    basin_id=basin_id,
    run_label=f"spinup slr {slr_m:g}",
)

# ── stability.csv: length, steady test, river-network swing check over all chunks ──
stability = max(swings, key=lambda r: r["max_swing_m"]) | {
    "unstable": any(r["unstable"] for r in swings),
    "spinup_hr": spinup_hr,
    "chunks": k,
    **steady,
}
pd.DataFrame([stability]).to_csv(snakemake.output.stability, index=False)
note = stability_note(stability)
if note:
    log.warning(f"Spin-up: {note}")
    for path in (snakemake.output.plot_forcing, snakemake.output.plot_max_depth):
        stamp_unstable(path, note)
else:
    log.info(
        f"Spin-up steady after {spinup_hr:g} h ({k} chunk(s)); largest hourly level "
        f"swing {stability['max_swing_m']:.2f} m"
    )
