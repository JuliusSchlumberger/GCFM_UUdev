"""Smoke tests for src.adaptation_lines on a synthetic delta with one river mouth."""

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import Affine
from shapely.geometry import LineString
from src.adaptation_lines import (
    KIND_COAST,
    KIND_RIVER,
    build_adaptation_lines,
    build_info,
    centrelines_off_sea,
    measure_line_layers,
    plot_adaptation_lines,
    require_sealed,
)
from src.adaptation_strategies import ADAPTATION_LINE_NAMES
from src.landuse import LANDUSE_SEA_CODE

N, CELL_M = 30, 100.0
COAST_ROW, CHANNEL_COLS = 20, (14, 15)  # sea from row 20 down; channel in cols 14-15
CRS = "EPSG:32631"
CLOSURE = {
    "direction_length_m": 300.0,
    "max_tilt_deg": 60.0,
    "max_half_length_m": 5000.0,
    "ext_cells": 2,
    "max_width_factor": 1.5,
    "max_walk_m": 10000.0,
    "refine_m": 1000.0,
    "pump_half_length_m": 200.0,
    "pumps_per_mouth": 1,
}
TRANSFORM = Affine(CELL_M, 0.0, 0.0, 0.0, -CELL_M, N * CELL_M)


def _delta():
    """Return (weir, landuse, valid, rivers): coast weir plus both river banks."""
    landuse = np.full((N, N), 40, dtype=np.int16)
    landuse[COAST_ROW:, :] = LANDUSE_SEA_CODE
    t = TRANSFORM
    coast = [
        LineString([t * (c, COAST_ROW), t * (c + 1, COAST_ROW)])
        for c in range(N)
        if c not in CHANNEL_COLS
    ]
    banks = [
        LineString([t * (col, r), t * (col, r + 1)])
        for col in (CHANNEL_COLS[0], CHANNEL_COLS[-1] + 1)
        for r in range(COAST_ROW)
    ]
    weir = gpd.GeoDataFrame(geometry=coast + banks, crs=CRS)
    x = (CHANNEL_COLS[0] + 1) * CELL_M  # channel centreline
    rivers = gpd.GeoDataFrame(
        {
            "reach_id": [1, 2],
            "n_rch_dn": [1, 0],
            "rch_id_up": ["[0]", "[1]"],
            "width": [2 * CELL_M, 2 * CELL_M],
        },
        geometry=[
            LineString([(x, N * CELL_M - 50), (x, 1500.0)]),
            LineString([(x, 1500.0), (x, 500.0)]),
        ],
        crs=CRS,
    )
    return weir, landuse, np.ones((N, N), dtype=bool), rivers


def _build():
    weir, landuse, valid, rivers = _delta()
    return build_adaptation_lines(
        weir,
        landuse,
        valid,
        TRANSFORM,
        rivers,
        min_component_cells=30,
        unprotected_ocean_wetlands=False,
        closure=CLOSURE,
    )


def test_seaward_tail_of_the_outlet_reach_is_cut_at_the_coast():
    _weir, landuse, _valid, rivers = _delta()
    out = centrelines_off_sea(rivers, landuse == LANDUSE_SEA_CODE, TRANSFORM)
    coast_y = (N - COAST_ROW) * CELL_M
    outlet = out.set_index("reach_id").geometry[2]
    assert min(y for _, y in outlet.coords) >= coast_y - CELL_M / 4
    assert out.set_index("reach_id").geometry[1].equals(rivers.geometry[0])


def test_mouth_is_closed_sealed_and_pump_runs_inland_to_sea():
    lines = _build()
    (mouth,) = lines.report
    assert mouth["closed"] and mouth["sealed"]
    require_sealed(lines.report)
    assert lines.weir["kind"].value_counts().to_dict() == {
        KIND_RIVER: 2 * COAST_ROW,
        KIND_COAST: N - len(CHANNEL_COLS),
    }
    pump = lines.pump.geometry.iloc[0]
    assert pump.coords[0][1] > pump.coords[-1][1]  # intake upstream (north)


def test_measure_layers_split_coast_closure_and_river_banks():
    lines = _build()
    layers = measure_line_layers(lines)
    assert tuple(layers) == ADAPTATION_LINE_NAMES
    n_coast = len(layers["coastal_protection_weir_PO"])
    assert len(layers["coastal_protection_weir_PC"]) == n_coast + len(lines.closure)
    assert len(lines.closure) > 0
    assert len(layers["river_protection_weir_PO"]) == 2 * COAST_ROW
    assert all(gdf.crs == CRS for gdf in layers.values())


def test_build_info_drops_check_point_and_plot_writes_png(tmp_path):
    lines = _build()
    info = build_info(lines, "1", "now", "sfincs.weir", {"dx": CELL_M, "extra": 0})
    assert info["grid"] == {"dx": CELL_M}
    assert "river_check_point" not in info["mouths"][0]
    assert "river_check_point" in lines.report[0]
    png = tmp_path / "adaptation_lines.png"
    plot_adaptation_lines(png, lines, TRANSFORM, "1")
    assert png.stat().st_size > 0


def test_unsealed_closure_raises_with_the_reach_id():
    report = [
        {"reach_id": 7, "closed": True, "sealed": True},
        {"reach_id": 8, "closed": True, "sealed": False},
        {"reach_id": 9, "closed": False},
    ]
    with pytest.raises(RuntimeError, match=r"does not seal outlet reach\(es\) \[8\]"):
        require_sealed(report)


def test_missing_closure_setting_names_the_config_key():
    weir, landuse, valid, rivers = _delta()
    partial = {k: v for k, v in CLOSURE.items() if k != "max_tilt_deg"}
    with pytest.raises(KeyError, match="adaptation.lines.max_tilt_deg"):
        build_adaptation_lines(
            weir,
            landuse,
            valid,
            TRANSFORM,
            rivers,
            min_component_cells=30,
            unprotected_ocean_wetlands=False,
            closure=partial,
        )
