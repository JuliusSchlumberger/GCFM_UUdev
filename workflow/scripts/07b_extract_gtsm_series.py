"""Extract hourly GTSM water level and surge at the stations; run by rule extract_gtsm_series.

Reads the station subset built once from the monthly reanalysis files by
tools/subset_gtsm_stations.py; fails if a station is missing from it. COAST-RP
coastal stations are GTSM output stations, matched by coordinates.

Output: gtsm_hourly.nc -- waterlevel, surge (time, station) in m, float32,
        with each station's lon/lat and GTSM station id.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from src.gtsm import GTSM_VARIABLES, SOURCE, subset_positions
from src.log import setup_logging

log = setup_logging(snakemake.log[0])

with open(snakemake.input.surge_stations) as f:
    stations = json.load(f)["stations"]
lons = np.array([s["lon"] for s in stations])
lats = np.array([s["lat"] for s in stations])

with xr.open_dataset(snakemake.params.gtsm_station_subset) as subset:
    gtsm_ids, positions = subset_positions(subset, lons, lats, float(snakemake.params.max_match_km))
    log.info(f"{len(stations)} surge station(s) matched to GTSM stations {gtsm_ids.tolist()}")
    picked = subset[list(GTSM_VARIABLES)].isel(station=positions).load()

time = pd.DatetimeIndex(picked["time"].values)
ds = xr.Dataset(
    {name: (("time", "station"), picked[name].values.astype("float32"), {"units": "m"}) for name in GTSM_VARIABLES},
    coords={
        "time": time,
        "longitude": ("station", lons),
        "latitude": ("station", lats),
        "gtsm_station": ("station", gtsm_ids),
    },
    attrs={"source": SOURCE, "period": f"{time[0]:%Y-%m-%d} to {time[-1]:%Y-%m-%d}"},
)
Path(snakemake.output.gtsm_hourly).parent.mkdir(parents=True, exist_ok=True)
ds.to_netcdf(snakemake.output.gtsm_hourly, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
log.info(
    f"Written: {snakemake.output.gtsm_hourly} ({len(time):,} hourly steps, {ds.attrs['period']}, "
    f"{int(np.isnan(ds['surge'].values).sum()) + int(np.isnan(ds['waterlevel'].values).sum())} missing value(s))"
)
