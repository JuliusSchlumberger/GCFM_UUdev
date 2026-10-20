"""Smoke tests for geometry forwarding and the weir writer in src.sfincs_run."""

import os
from types import SimpleNamespace

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString
from src.sfincs_run import forward_geometry_files, write_weir_file


def test_forward_geometry_files_skips_missing_empty_and_excluded(tmp_path):
    source = tmp_path / "skeleton"
    dest = tmp_path / "events" / "run1"
    source.mkdir()
    dest.mkdir(parents=True)
    (source / "sfincs.dep").write_bytes(b"\x00" * 8)
    (source / "sfincs.weir").write_bytes(b"")
    (source / "sfincs.rst").write_bytes(b"\x01")
    cfg = {
        "depfile": "sfincs.dep",
        "weirfile": "sfincs.weir",
        "rstfile": "sfincs.rst",
        "mskfile": "sfincs.msk",
        "tref": "20200101 000000",
    }
    lines = forward_geometry_files(cfg, source, dest, exclude=frozenset({"rstfile"}))
    assert len(lines) == 1
    key, rel = (part.strip() for part in lines[0].split("="))
    assert key == "depfile"
    assert not os.path.isabs(rel)
    assert (dest / rel).resolve() == (source / "sfincs.dep").resolve()


def _fake_model(tmp_path, gdf):
    path = tmp_path / "sfincs.weir"
    return SimpleNamespace(
        weirs=SimpleNamespace(data=gdf),
        crs=CRS.from_epsg(32631),
        config=SimpleNamespace(get_set_file_variable=lambda **_: str(path)),
    ), path


def _weir_values(path):
    rows = [line.split() for line in path.read_text().splitlines()]
    return [r for r in rows if len(r) == 4]


def test_weir_file_keeps_centimetre_crest(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"elevation": [2.37, 1.04], "par1": [0.6, 0.6]},
        geometry=[LineString([(0, 0), (100, 0)]), LineString([(100, 0), (100, 100)])],
        crs="EPSG:32631",
    )
    sf, path = _fake_model(tmp_path, gdf)
    written = write_weir_file(sf)
    assert written == path and path.exists()
    vertices = _weir_values(path)
    assert len(vertices) == 4
    crests = sorted({r[2] for r in vertices})
    assert crests == ["1.04", "2.37"]


def test_weir_file_skipped_without_weirs(tmp_path):
    empty = gpd.GeoDataFrame(
        {"elevation": [], "par1": []}, geometry=[], crs="EPSG:32631"
    )
    sf, path = _fake_model(tmp_path, empty)
    assert write_weir_file(sf) is None
    assert not path.exists()
