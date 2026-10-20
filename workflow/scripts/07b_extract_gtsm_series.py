"""Extract hourly GTSM water level and surge at the stations; run by rule extract_gtsm_series.

The reanalysis is one file per month for all stations, so 1950-2024 means 900
files per variable (~60 GB; disk-bound, tens of minutes on a hard disk). The only
input is surge_stations.json; the GTSM folders are params, not inputs, so the
rule reruns only when the selected stations change. COAST-RP coastal stations
are GTSM output stations, matched by coordinates.

Output: gtsm_hourly.nc -- waterlevel, surge (time, station) in m, float32,
        with each station's lon/lat and GTSM station id.
"""

import json
from pathlib import Path

import numpy as np
import xarray as xr

from src.gtsm import GTSM_VARIABLES, extract_station_series, match_stations, station_coordinates
from src.log import setup_logging

if __name__ == "__main__":  # the parallel readers re-import this file
    log = setup_logging(snakemake.log[0])

    with open(snakemake.input.surge_stations) as f:
        stations = json.load(f)["stations"]
    lons = np.array([s["lon"] for s in stations])
    lats = np.array([s["lat"] for s in stations])

    folders = {"waterlevel": snakemake.params.gtsm_waterlevel_dir, "surge": snakemake.params.gtsm_surge_dir}
    gtsm_ids, gtsm_lon, gtsm_lat = station_coordinates(folders["surge"])
    idx, match_km = match_stations(lons, lats, gtsm_lon, gtsm_lat, float(snakemake.params.max_match_km))
    log.info(f"{len(stations)} surge station(s) matched to GTSM stations {gtsm_ids[idx].tolist()}")

    series = {
        name: extract_station_series(folders[name], var, idx, n_workers=int(snakemake.threads))
        for name, var in GTSM_VARIABLES.items()
    }
    time = series["surge"].index.intersection(series["waterlevel"].index)
    ds = xr.Dataset(
        {
            name: (("time", "station"), df.loc[time].to_numpy(dtype="float32"), {"units": "m"})
            for name, df in series.items()
        },
        coords={
            "time": time,
            "longitude": ("station", lons),
            "latitude": ("station", lats),
            "gtsm_station": ("station", gtsm_ids[idx]),
        },
        attrs={
            "source": "Copernicus CDS sis-water-level-change-timeseries-cmip6, GTSM v3 reanalysis, hourly",
            "period": f"{time[0]:%Y-%m-%d} to {time[-1]:%Y-%m-%d}",
        },
    )
    Path(snakemake.output.gtsm_hourly).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(snakemake.output.gtsm_hourly, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    log.info(
        f"Written: {snakemake.output.gtsm_hourly} ({len(time):,} hourly steps, {ds.attrs['period']}, "
        f"{int(np.isnan(ds['surge'].values).sum()) + int(np.isnan(ds['waterlevel'].values).sum())} missing value(s))"
    )
