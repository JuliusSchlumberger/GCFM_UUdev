"""Tests for src.sword: published SWORD v17c regions in the pipeline schema."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString
from src.river_network import build_downstream_adjacency
from src.sword import (
    apply_width_corrections,
    read_sword_reaches,
    read_width_corrections,
)


def _region(path, rows):
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    gdf.to_file(path, layer="reaches", driver="GPKG")
    return str(path)


def _reach(rid, x0, dn=(), up=(), width=100.0, obstr=0):
    row = {
        "reach_id": rid,
        "obstr_type": obstr,
        "width": width,
        "max_width": 500.0,
        "n_rch_down": len(dn),
        "is_mainstem": 1,
        "geometry": LineString([(x0, 0.0), (x0 + 0.5, 0.0)]),
    }
    for k in range(1, 5):
        row[f"rch_id_dn_{k}"] = dn[k - 1] if k <= len(dn) else 0
        row[f"rch_id_up_{k}"] = up[k - 1] if k <= len(up) else 0
    return row


@pytest.fixture
def regions(tmp_path):
    eu = _region(
        tmp_path / "sword_EU_v17c.gpkg",
        [
            _reach(21, 0.0, dn=(22, 23), obstr=2),
            _reach(22, 0.5, up=(21,), obstr=5),
            _reach(23, 0.5, up=(21,)),
        ],
    )
    na = _region(tmp_path / "sword_NA_v17c.gpkg", [_reach(71, 50.0)])
    return [eu, na]


def test_published_topology_becomes_bracketed_lists_and_builds_adjacency(regions):
    rivers = read_sword_reaches(regions, (-1.0, -1.0, 2.0, 1.0))
    assert sorted(rivers["reach_id"]) == [21, 22, 23]  # NA reach is outside the box
    first = rivers.set_index("reach_id").loc[21]
    assert first["rch_id_dn"] == "[22, 23]" and first["rch_id_up"] == "[]"
    assert first["n_rch_dn"] == 2 and first["is_mainstem_edge"] == 1
    assert build_downstream_adjacency(rivers)["21"] == ["22", "23"]


def test_undocumented_obstruction_type_5_reads_as_none(regions):
    rivers = read_sword_reaches(regions, (-1.0, -1.0, 2.0, 1.0)).set_index("reach_id")
    assert rivers.loc[22, "obstr_type"] == 0
    assert rivers.loc[22, "obstr_type_published"] == 5
    assert rivers.loc[21, "obstr_type"] == 2  # documented types are kept


def test_reaches_are_clipped_to_the_box(regions):
    rivers = read_sword_reaches(regions, (-1.0, -1.0, 0.25, 1.0))
    assert list(rivers["reach_id"]) == [21]
    assert rivers.geometry.iloc[0].bounds[2] == pytest.approx(0.25)


def test_reach_drawn_upstream_is_reversed(tmp_path):
    up = _reach(31, 0.0, dn=(32,))
    down = _reach(32, 0.5, up=(31,))
    down["geometry"] = LineString([(1.0, 0.0), (0.75, 0.0), (0.5, 0.0)])
    path = _region(tmp_path / "sword_EU_v17c.gpkg", [up, down])
    rivers = read_sword_reaches([path], (-1.0, -1.0, 2.0, 1.0)).set_index("reach_id")
    assert list(rivers.loc[31].geometry.coords) == [(0.0, 0.0), (0.5, 0.0)]
    assert list(rivers.loc[32].geometry.coords) == [(0.5, 0.0), (0.75, 0.0), (1.0, 0.0)]


def test_no_reach_in_box_fails_loudly(regions):
    with pytest.raises(ValueError, match="No SWORD reach"):
        read_sword_reaches(regions, (100.0, 10.0, 101.0, 11.0))


def _table(tmp_path, rows):
    path = tmp_path / "widths.csv"
    pd.DataFrame(
        rows, columns=["reach_id", "expected_width_m", "width_m", "delta", "note"]
    ).to_csv(path, index=False)
    return read_width_corrections(path)


def test_width_correction_applied_and_marked_manual(regions, tmp_path):
    rivers = read_sword_reaches(regions, (-1.0, -1.0, 2.0, 1.0))
    table = _table(
        tmp_path, [(22, 100.4, 40.0, "x", ""), (999, 10.0, 5.0, "elsewhere", "")]
    )
    out = apply_width_corrections(rivers, table).set_index("reach_id")
    assert out.loc[22, "width"] == 40.0 and out.loc[22, "width_source"] == "manual"
    assert (
        out.loc[21, "width"] == 100.0 and out.loc[21, "width_source"] == "sword_width"
    )


def test_width_correction_with_wrong_expected_width_stops(regions, tmp_path):
    rivers = read_sword_reaches(regions, (-1.0, -1.0, 2.0, 1.0))
    with pytest.raises(ValueError, match="does not match"):
        apply_width_corrections(rivers, _table(tmp_path, [(22, 150.0, 40.0, "x", "")]))


def test_repo_width_table_is_valid():
    from pathlib import Path

    table = read_width_corrections(
        Path(__file__).resolve().parents[2] / "config" / "sword_width_corrections.csv"
    )
    assert len(table) == 9 and (table["width_m"] > 0).all()
