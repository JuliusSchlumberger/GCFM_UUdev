"""Extract hourly station series from the GTSMv3 reanalysis (total water level, surge).

Source: Copernicus CDS sis-water-level-change-timeseries-cmip6. Files are laid
out by time (one per month, all ~43,000 stations), so any station subset needs
every monthly file. Files: {folder}/reanalysis_{waterlevel|surge}_hourly_YYYY_MM_v3.nc,
unzipped from the yearly CDS archives (tools/download_cds_waterlevel.py).
tools/subset_gtsm_stations.py reads them once into a station subset that rule
extract_gtsm_series reads.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)

# variable name in the monthly files, per catalogue dataset
GTSM_VARIABLES = {"waterlevel": "waterlevel", "surge": "surge"}
SOURCE = (
    "Copernicus CDS sis-water-level-change-timeseries-cmip6, GTSM v3 reanalysis, hourly"
)
_SLAB_GAP = 2000  # station indices closer than this are read as one contiguous slab


def monthly_files(folder: str | Path) -> list[Path]:
    """Return the monthly NetCDF files under `folder`, in time order."""
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
    """Return (station ids, lon, lat) of the GTSM output stations (same in every file)."""
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
    """Return the GTSM station index coinciding with each (lon, lat) and its distance (km).

    COAST-RP's coastal stations are GTSM output stations, so max_km is a
    coincidence tolerance, not a search radius; raises ValueError if a point
    has no GTSM station within it.
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
    """Return hourly series of the given station positions over every monthly file.

    station_idx: positions along the files' `stations` dimension. Returns one
    float32 column per entry of station_idx, in that order.
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


def build_station_subset(
    folders: dict[str, str | Path], station_idx: np.ndarray, n_workers: int = 1
) -> xr.Dataset:
    """Return hourly series of every GTSM_VARIABLES entry at station_idx, on common hours.

    folders: monthly-file folder per GTSM_VARIABLES key. The dataset also carries
    every GTSM station's id and position (dimension `gtsm_station`), so a reader
    can tell a station left out of the subset from a point with no GTSM station.
    """
    ids, lon, lat = station_coordinates(folders["surge"])
    station_idx = np.asarray(station_idx, dtype=int)
    series = {
        name: extract_station_series(folders[name], var, station_idx, n_workers)
        for name, var in GTSM_VARIABLES.items()
    }
    time = series["surge"].index.intersection(series["waterlevel"].index)
    return xr.Dataset(
        {
            name: (
                ("time", "station"),
                df.loc[time].to_numpy(dtype="float32"),
                {"units": "m"},
            )
            for name, df in series.items()
        },
        coords={
            "time": time,
            "station_id": ("station", ids[station_idx]),
            "all_station_id": ("gtsm_station", ids),
            "all_lon": ("gtsm_station", lon),
            "all_lat": ("gtsm_station", lat),
        },
        attrs={"source": SOURCE},
    )


def subset_positions(
    subset: xr.Dataset, lons: np.ndarray, lats: np.ndarray, max_km: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (GTSM station ids, positions along `station`) of each (lon, lat) in a subset.

    Raises ValueError if a point has no GTSM station within max_km, or if its
    station is missing from the subset (rebuild it with tools/subset_gtsm_stations.py).
    """
    idx, _ = match_stations(
        lons, lats, subset["all_lon"].values, subset["all_lat"].values, max_km
    )
    ids = subset["all_station_id"].values[idx]
    position = {int(s): i for i, s in enumerate(subset["station_id"].values)}
    missing = sorted({int(s) for s in ids} - position.keys())
    if missing:
        raise ValueError(
            f"GTSM station(s) {missing} are not in the station subset; rebuild it with "
            f"tools/subset_gtsm_stations.py after rule select_surge_stations has run for "
            f"every basin"
        )
    return ids, np.array([position[int(s)] for s in ids])
