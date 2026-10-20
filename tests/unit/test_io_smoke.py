"""Smoke tests for the data-catalogue helpers in src.io."""

from pathlib import Path

import geopandas as gpd
import pytest
import yaml
from shapely.geometry import box
from src.io import (
    catalogue_entry,
    general_path,
    load_catalogue,
    local_path,
    raw_input_path,
    read_geometry,
)


def _catalogue(tmp_path):
    path = tmp_path / "data_catalogue.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "meta": {"root": "placeholder"},
                "datasets": [
                    {"name": "dem", "file_path": "dem/tiles"},
                    {"name": "landuse", "file_path": "lc/lc100.tif"},
                ],
            }
        )
    )
    return path


def test_local_path_raises_when_unset(monkeypatch):
    monkeypatch.delenv("GCFM_RESULTS_DIR", raising=False)
    with pytest.raises(EnvironmentError, match="GCFM_RESULTS_DIR"):
        local_path("GCFM_RESULTS_DIR")


def test_load_catalogue_roots_paths_at_raw_data_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GCFM_RAW_DATA_ROOT", str(tmp_path / "raw"))
    cat = load_catalogue(_catalogue(tmp_path))
    assert cat["meta"]["root"] == str(tmp_path / "raw")
    assert Path(raw_input_path(cat, "landuse")) == tmp_path / "raw" / "lc" / "lc100.tif"
    assert Path(general_path(cat, "dem")).parts[-2:] == ("dem", "tiles")


def test_catalogue_entry_unknown_name_raises_keyerror(tmp_path, monkeypatch):
    monkeypatch.setenv("GCFM_RAW_DATA_ROOT", str(tmp_path))
    cat = load_catalogue(_catalogue(tmp_path))
    assert catalogue_entry(cat, "dem")["file_path"] == "dem/tiles"
    with pytest.raises(KeyError, match="nope"):
        catalogue_entry(cat, "nope")


def test_read_geometry_round_trips_gpkg(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2]}, geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)], crs="EPSG:4326"
    )
    path = tmp_path / "polys.gpkg"
    gdf.to_file(path)
    back = read_geometry(str(path))
    assert back.crs == gdf.crs
    assert back["id"].tolist() == [1, 2]
    assert back.geometry.geom_equals(gdf.geometry).all()
