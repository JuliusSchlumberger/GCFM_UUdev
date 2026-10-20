"""Smoke tests for geometry forwarding and the weir writer in src.sfincs_run."""

import os
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pyproj import CRS
from shapely.geometry import LineString, box
from src.sfincs_run import (
    forward_geometry_files,
    parse_sfincs_inp,
    place_discharge_points,
    write_sfincs_inp,
    write_weir_file,
)


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


def test_write_sfincs_inp_settings_override_skeleton_and_restart_is_never_forwarded(
    tmp_path,
):
    skeleton = tmp_path / "skeleton"
    skeleton.mkdir()
    for name in ("sfincs.dep", "sfincs.ini", "old.rst", "sfincs.bzs"):
        (skeleton / name).write_bytes(b"x")
    cfg = {
        "mmax": "4",
        "zsini": "0.0",
        "depfile": "sfincs.dep",
        "inifile": "sfincs.ini",
        "rstfile": "old.rst",
        "bzsfile": "sfincs.bzs",
    }
    run = tmp_path / "runs" / "sfincs"
    run.mkdir(parents=True)

    path = write_sfincs_inp(
        run / "sfincs.inp",
        cfg,
        skeleton,
        {"zsini": -9999.0, "tref": "20000101 000000"},
        {"bzsfile": "event.bzs", "disfile": None},
        exclude_files=frozenset({"inifile"}),
    )
    out = parse_sfincs_inp(path)

    assert out["mmax"] == "4"
    assert out["zsini"] == "-9999.0"
    assert Path(out["depfile"]) == Path("../../skeleton/sfincs.dep")
    assert out["bzsfile"] == "event.bzs"
    assert "rstfile" not in out and "inifile" not in out and "disfile" not in out
    assert (
        list(out).index("depfile")
        < list(out).index("tref")
        < list(out).index("bzsfile")
    )


def _discharge_model(created):
    region = gpd.GeoDataFrame(geometry=[box(0, 0, 1000, 1000)], crs="EPSG:32631")
    return SimpleNamespace(
        crs=CRS.from_epsg(32631),
        region=region,
        discharge_points=SimpleNamespace(create=lambda **kw: created.update(kw)),
    )


def _centerline_cells():
    xy = [(150.0, 500.0), (250.0, 500.0), (5000.0, 500.0)]
    return gpd.GeoDataFrame(
        {
            "reach_id": ["1", "1", "2"],
            "row": [5, 5, 5],
            "col": [1, 2, 50],
            "x": [x for x, _ in xy],
            "y": [y for _, y in xy],
        },
        geometry=gpd.points_from_xy(*zip(*xy)),
        crs="EPSG:32631",
    )


def _crossings(xy):
    return gpd.GeoDataFrame(
        {"index": range(len(xy))},
        geometry=gpd.points_from_xy(*zip(*xy)),
        crs="EPSG:32631",
    ).to_crs("EPSG:4326")


def test_place_discharge_points_drops_outside_crossing_and_its_column():
    created = {}
    times = pd.date_range("2000-01-01", periods=3, freq="h")
    discharge = pd.DataFrame({0: [10.0] * 3, 1: [20.0] * 3, 2: [30.0] * 3}, index=times)
    kept = place_discharge_points(
        _discharge_model(created),
        _crossings([(5010.0, 510.0), (160.0, 520.0), (240.0, 480.0)]),
        discharge,
        _centerline_cells(),
        ["2", "1", "1"],
        100.0,
    )
    assert kept["index"].tolist() == [1, 2]
    assert created["timeseries"].columns.tolist() == [1, 2]
    np.testing.assert_array_equal(
        created["timeseries"].to_numpy(), discharge[[1, 2]].to_numpy()
    )
    snapped = kept.to_crs("EPSG:32631").geometry
    np.testing.assert_allclose(snapped.x, [150.0, 250.0], atol=1e-3)


def test_place_discharge_points_raises_when_no_crossing_is_inside():
    created = {}
    discharge = pd.DataFrame({0: [1.0]}, index=pd.date_range("2000-01-01", periods=1))
    with pytest.raises(RuntimeError, match="outside the active SFINCS region"):
        place_discharge_points(
            _discharge_model(created),
            _crossings([(5010.0, 510.0)]),
            discharge,
            _centerline_cells(),
            ["2"],
            100.0,
        )
    assert not created
