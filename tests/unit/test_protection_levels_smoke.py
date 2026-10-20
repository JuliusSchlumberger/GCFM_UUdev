"""Smoke tests for FLOPROS protection-level lookup in src.protection_levels."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box
from src.protection_levels import (
    identify_dominant_protection,
    load_flopros_table,
    load_geogunit_iso_lookup,
)

FILL = -9999.0


def _geogunit_nc(tmp_path):
    lon = np.arange(0.05, 2.0, 0.1)
    lat = np.arange(0.05, 2.0, 0.1)
    ids = np.full((lat.size, lon.size), 1.0)
    ids[:, 14:] = 2.0  # unit 2 only in the eastern strip
    ids[:3, :3] = np.nan
    ds = xr.Dataset(
        {"Geogunits": (("lat", "lon"), ids)}, coords={"lat": lat, "lon": lon}
    )
    path = tmp_path / "geogunit.nc"
    ds.to_netcdf(path, encoding={"Geogunits": {"_FillValue": FILL}})
    return path


def _flopros(riverine_1=np.nan, coastal_1=50.0):
    return pd.DataFrame(
        {
            "ISO": ["AAA", "BBB"],
            "Riverine": [riverine_1, 100.0],
            "Coastal": [coastal_1, 10.0],
        },
        index=pd.Index([1, 2], name="FID_Aque"),
    )


ISO = pd.Series(["AAA", "BBB"], index=[1, 2])


def test_dominant_unit_missing_riverine_falls_back_to_coastal(tmp_path):
    out = identify_dominant_protection(
        box(0, 0, 2, 2), str(_geogunit_nc(tmp_path)), _flopros(), ISO
    )
    assert out["dominant_geounit_id"] == 1
    assert out["dominant_iso"] == "AAA"
    assert out["river_protection_rp_raw_yr"] is None
    assert out["river_protection_rp_yr"] == out["surge_protection_rp_yr"] == 50.0
    assert out["river_protection_source"] == "flopros_other_hazard_fallback"
    assert out["surge_protection_source"] == "flopros"
    assert 0.5 < out["pixel_fraction"] <= 1.0
    assert out["n_geounits_in_delta"] == 2


def test_resolved_rp_is_capped_at_max(tmp_path):
    out = identify_dominant_protection(
        box(0, 0, 2, 2),
        str(_geogunit_nc(tmp_path)),
        _flopros(riverine_1=5000.0, coastal_1=np.nan),
        ISO,
        max_rp_yr=1000.0,
    )
    assert out["river_protection_rp_yr"] == 1000.0
    assert out["surge_protection_rp_yr"] == 1000.0


def test_polygon_without_valid_pixels_uses_default_rp(tmp_path):
    out = identify_dominant_protection(
        box(0.0, 0.0, 0.25, 0.25),
        str(_geogunit_nc(tmp_path)),
        _flopros(),
        ISO,
        default_rp_yr=2.0,
    )
    assert out["dominant_geounit_id"] is None
    assert out["river_protection_rp_yr"] == out["surge_protection_rp_yr"] == 2.0
    assert out["river_protection_source"] == out["surge_protection_source"] == "default"


def test_flopros_and_iso_tables_round_trip_from_excel(tmp_path):
    flopros_path = tmp_path / "flopros.xlsx"
    pd.DataFrame(
        {
            "FID_Aque": [1, 2],
            "ISO": ["AAA", "BBB"],
            "Riverine": [np.nan, 100.0],
            "Coastal": [50.0, 10.0],
            "extra": [0, 0],
        }
    ).to_excel(flopros_path, index=False)
    iso_path = tmp_path / "iso.xlsx"
    pd.DataFrame([[1, "AAA"], [2, "BBB"]]).to_excel(iso_path, index=False, header=False)

    flopros = load_flopros_table(str(flopros_path))
    iso = load_geogunit_iso_lookup(str(iso_path))
    assert list(flopros.columns) == ["ISO", "Riverine", "Coastal"]
    assert flopros.loc[2, "Riverine"] == pytest.approx(100.0)
    assert np.isnan(flopros.loc[1, "Riverine"])
    assert iso.to_dict() == {1: "AAA", 2: "BBB"}
