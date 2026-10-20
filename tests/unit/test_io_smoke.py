"""Smoke tests for the data-catalogue helpers in src.io."""

from pathlib import Path

import geopandas as gpd
import pytest
import src.io
import yaml
from shapely.geometry import box
from src.io import (
    catalogue_entry,
    general_path,
    load_catalogue,
    raw_input_path,
    read_geometry,
    require_keys,
    site_setting,
)

SITE_VALUES = {
    "results_dir": "/site/results",
    "raw_data_root": "/site/raw_data",
    "sfincs_exe": "/site/bin/sfincs",
}


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


@pytest.fixture
def site_file(tmp_path, monkeypatch):
    """Point site_setting at a tmp site.yml and clear the override variables."""
    path = tmp_path / "site.yml"
    monkeypatch.setattr(src.io, "SITE_FILE", path)
    for env_var in src.io.SITE_ENV_VARS.values():
        monkeypatch.delenv(env_var, raising=False)
    return path


def test_site_setting_reads_site_file(site_file):
    site_file.write_text(yaml.safe_dump(SITE_VALUES))
    for key, value in SITE_VALUES.items():
        assert site_setting(key) == value


def test_site_setting_env_var_overrides_site_file(site_file, monkeypatch):
    site_file.write_text(yaml.safe_dump(SITE_VALUES))
    for key, env_var in src.io.SITE_ENV_VARS.items():
        monkeypatch.setenv(env_var, f"/env/{key}")
        assert site_setting(key) == f"/env/{key}"


def test_site_setting_raises_naming_key_env_var_and_copy_command(site_file):
    for key, env_var in src.io.SITE_ENV_VARS.items():
        with pytest.raises(OSError, match=key) as err:
            site_setting(key)
        assert env_var in str(err.value)
        assert "cp config/site.example.yml config/site.yml" in str(err.value)


@pytest.mark.parametrize("value", ["", "  ", "<results_dir>"])
def test_site_setting_treats_empty_and_placeholder_as_unset(site_file, value):
    site_file.write_text(yaml.safe_dump({"results_dir": value}))
    with pytest.raises(OSError, match="GCFM_RESULTS_DIR"):
        site_setting("results_dir")


def test_site_setting_rejects_unknown_keys(site_file):
    with pytest.raises(ValueError, match="results"):
        site_setting("results")
    site_file.write_text(yaml.safe_dump({**SITE_VALUES, "result_dir": "/x"}))
    with pytest.raises(ValueError, match="result_dir"):
        site_setting("results_dir")


def test_load_catalogue_roots_paths_at_raw_data_root(tmp_path, monkeypatch):
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


def test_require_keys_names_every_missing_key_path():
    require_keys({"a": 1, "b": None}, ("a", "b"), "block")
    with pytest.raises(KeyError, match=r"block\.b, block\.c"):
        require_keys({"a": 1}, ("a", "b", "c"), "block")
