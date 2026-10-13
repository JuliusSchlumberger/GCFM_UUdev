"""
gtsm.py -- hourly station series from the GTSM reanalysis (Copernicus CDS
sis-water-level-change-timeseries-cmip6, GTSMv3 reanalysis, hourly: total
water level and storm surge residual).

The dataset is laid out by TIME: one file per month, each holding every one
of the ~43,000 output stations. One station's record is therefore spread
over every monthly file (900 per variable for 1950-2024), and extracting
any set of stations means opening all of them -- extract_station_series
does that once, in parallel, reading only the slab of stations it needs.

Files: {folder}/reanalysis_{waterlevel|surge}_hourly_YYYY_MM_v3.nc, as
unzipped from the yearly CDS archives (tools/download_cds_waterlevel.py).
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# variable name in the monthly files, per catalogue dataset
GTSM_VARIABLES = {"waterlevel": "waterlevel", "surge": "surge"}
_SLAB_GAP = 2000  # station indices closer than this are read as one contiguous slab


def monthly_files(folder: str | Path) -> list[Path]:
    """The monthly NetCDF files under `folder`, in time order."""
    files = sorted(Path(folder).glob("reanalysis_*_hourly_????_??_v3.nc"))
    if not files:
        raise FileNotFoundError(
            f"no reanalysis_*_hourly_YYYY_MM_v3.nc files under {folder} -- download the hourly "
            f"GTSM reanalysis (tools/download_cds_waterlevel.py) and unzip the yearly archives there"
        )
    return files


def station_coordinates(
    folder: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(station ids, lon, lat) of the GTSM output stations (same in every file)."""
    import netCDF4

    with netCDF4.Dataset(monthly_files(folder)[0]) as ds:
        return (
            np.asarray(ds["stations"][:]).astype(int),
            np.asarray(ds["station_x_coordinate"][:], dtype=float),
            np.asarray(ds["station_y_coordinate"][:], dtype=float),
        )


def match_stations(
    lons: np.ndarray,
    lats: np.ndarray,
    gtsm_lon: np.ndarray,
    gtsm_lat: np.ndarray,
    max_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Position of the GTSM station coinciding with each (lon, lat) and its
    distance (km). COAST-RP's coastal stations ARE GTSM output stations, so
    max_km is a coincidence tolerance, not a search radius; raises if a
    point has no GTSM station within it.
    """
    idx = np.empty(len(lons), dtype=int)
    dist = np.empty(len(lons))
    for i, (lon, lat) in enumerate(zip(lons, lats)):
        d = np.hypot((gtsm_lon - lon) * np.cos(np.radians(lat)), gtsm_lat - lat) * 111.0
        idx[i] = int(np.nanargmin(d))
        dist[i] = d[idx[i]]
    if (dist > max_km).any():
        bad = [
            f"({lons[i]:.4f}, {lats[i]:.4f}): {dist[i]:.2f} km"
            for i in np.flatnonzero(dist > max_km)
        ]
        raise ValueError(
            f"no GTSM station within {max_km:g} km of {len(bad)} surge station(s): {bad}"
        )
    return idx, dist


def _read_month(args) -> tuple[np.ndarray, np.ndarray]:
    """(time as datetime64[s], values (n_time, n_station) float32) of one monthly file."""
    import netCDF4

    path, var, station_idx = args
    order = np.argsort(station_idx)
    sorted_idx = station_idx[order]
    out = None
    with netCDF4.Dataset(path) as ds:
        v = ds[var]
        time = netCDF4.num2date(
            ds["time"][:],
            ds["time"].units,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
        start = 0
        while start < len(sorted_idx):  # one hyperslab per run of nearby stations
            stop = start + 1
            while (
                stop < len(sorted_idx)
                and sorted_idx[stop] - sorted_idx[stop - 1] < _SLAB_GAP
            ):
                stop += 1
            lo = int(sorted_idx[start])
            slab = np.ma.filled(v[:, lo : int(sorted_idx[stop - 1]) + 1], np.nan)
            if out is None:
                out = np.empty((slab.shape[0], len(station_idx)), dtype=np.float32)
            out[:, order[start:stop]] = slab[:, sorted_idx[start:stop] - lo]
            start = stop
    return np.asarray(time, dtype="datetime64[s]"), out


def extract_station_series(
    folder: str | Path, var: str, station_idx: np.ndarray, n_workers: int = 1
) -> pd.DataFrame:
    """Hourly series of the given station positions over every monthly file.

    Args:
        folder:      directory of the monthly files of one variable.
        var:         variable name in the files ("waterlevel" or "surge").
        station_idx: positions along the files' `stations` dimension.
        n_workers:   processes reading files in parallel.

    Returns:
        DataFrame (time index, one float32 column per entry of station_idx,
        in that order).
    """
    files = monthly_files(folder)
    tasks = [(str(p), var, np.asarray(station_idx, dtype=int)) for p in files]
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            parts = list(pool.map(_read_month, tasks, chunksize=8))
    else:
        parts = [_read_month(t) for t in tasks]
    log.info(f"{var}: read {len(files)} monthly file(s), {len(station_idx)} station(s)")
    df = pd.DataFrame(
        np.concatenate([p[1] for p in parts]),
        index=pd.DatetimeIndex(np.concatenate([p[0] for p in parts])),
    )
    return df[~df.index.duplicated()].sort_index()
