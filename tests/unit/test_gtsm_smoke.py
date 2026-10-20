"""Smoke tests for GTSM station matching and monthly-file extraction in src.gtsm."""

import netCDF4
import numpy as np
import pandas as pd
import pytest
from src.gtsm import extract_station_series, match_stations, station_coordinates

_LON = np.array([4.0, 4.5, 5.0, 30.0])
_LAT = np.array([52.0, 52.5, 53.0, -10.0])


def _write_month(folder, year, month, n_hours, offset):
    path = folder / f"reanalysis_waterlevel_hourly_{year}_{month:02d}_v3.nc"
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", n_hours)
        ds.createDimension("stations", _LON.size)
        t = ds.createVariable("time", "f8", ("time",))
        t.units = f"hours since {year}-{month:02d}-01 00:00:00"
        t[:] = np.arange(n_hours)
        ds.createVariable("stations", "i4", ("stations",))[:] = (
            np.arange(_LON.size) + 100
        )
        ds.createVariable("station_x_coordinate", "f8", ("stations",))[:] = _LON
        ds.createVariable("station_y_coordinate", "f8", ("stations",))[:] = _LAT
        v = ds.createVariable(
            "waterlevel", "f4", ("time", "stations"), fill_value=-999.0
        )
        data = offset + np.arange(n_hours)[:, None] + 1000.0 * np.arange(_LON.size)
        data = np.ma.masked_array(data, mask=np.zeros_like(data, dtype=bool))
        data.mask[0, 1] = True
        v[:] = data


@pytest.fixture
def gtsm_folder(tmp_path):
    _write_month(tmp_path, 2000, 2, 5, 0.0)
    _write_month(tmp_path, 2000, 1, 4, 0.0)
    return tmp_path


def test_match_stations_picks_nearest_within_tolerance():
    idx, dist_km = match_stations(
        np.array([4.501, 30.0]), np.array([52.5, -10.001]), _LON, _LAT, max_km=1.0
    )
    np.testing.assert_array_equal(idx, [1, 3])
    assert np.all(dist_km >= 0.0) and np.all(dist_km <= 1.0)


def test_match_stations_raises_beyond_tolerance():
    with pytest.raises(ValueError, match="no GTSM station"):
        match_stations(np.array([10.0]), np.array([52.0]), _LON, _LAT, max_km=1.0)


def test_station_coordinates_reads_ids_and_positions(gtsm_folder):
    ids, lon, lat = station_coordinates(gtsm_folder)
    np.testing.assert_array_equal(ids, np.arange(_LON.size) + 100)
    np.testing.assert_allclose(lon, _LON)
    np.testing.assert_allclose(lat, _LAT)


def test_extract_station_series_spans_all_months_in_requested_order(gtsm_folder):
    df = extract_station_series(gtsm_folder, "waterlevel", np.array([3, 1]))
    assert df.shape == (9, 2)
    assert df.index.is_monotonic_increasing
    assert df.index[0] == pd.Timestamp("2000-01-01 00:00")
    assert df.index[-1] == pd.Timestamp("2000-02-01 04:00")
    assert all(dt == np.float32 for dt in df.dtypes)
    assert (df[0].dropna() >= 3000.0).all()
    assert (df[1].dropna() < 2000.0).all()
    assert df[1].isna().sum() == 2


def test_extract_station_series_raises_without_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_station_series(tmp_path, "waterlevel", np.array([0]))
