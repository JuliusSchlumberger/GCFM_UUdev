"""Smoke tests for src.adaptation_scaling on a synthetic baseline run directory."""

import logging

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import LineString, MultiLineString, Polygon
from src.adaptation_scaling import read_polyline, resolve_forcing_params
from src.sfincs_run import read_weir_file

CRS = "EPSG:32631"
LOG = logging.getLogger("test")
BZS_MAX, DIS_MAX = 2.5, 1200.0


def _weir_text(segments):
    out = []
    for k, pts in enumerate(segments):
        out.append(f"weir{k}")
        out.append(f"{len(pts)} 4")
        out.extend(f"{x} {y} {z} 0.6" for x, y, z in pts)
    return "\n".join(out) + "\n"


@pytest.fixture
def baseline(tmp_path):
    root = tmp_path / "baseline"
    root.mkdir()
    (root / "sfincs.inp").write_text("epsg = 32631\ndx = 100\n")
    np.savetxt(
        root / "sfincs.bzs", [[0, 1.0, 0.5], [3600, 2.0, BZS_MAX], [7200, 0.1, 0.2]]
    )
    np.savetxt(root / "sfincs.dis", [[0, 100.0], [3600, DIS_MAX]])

    weir = tmp_path / "skeleton.weir"
    weir.write_text(
        _weir_text(
            [
                [(0, 0, 3.0), (500, 0, 3.0)],
                [(500, 0, 4.0), (1000, 0, 4.0)],
                [(0, 5000, 9.0), (500, 5000, 9.0)],  # far from the polyline
            ]
        )
    )

    adapt = tmp_path / "adapt"
    adapt.mkdir()
    gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1500, 0)])], crs=CRS).to_file(
        adapt / "levee.geojson"
    )

    yy, xx = np.meshgrid(
        np.arange(-250, 300, 100.0), np.arange(50, 1500, 100.0), indexing="ij"
    )
    zsmax = np.stack([np.full(xx.shape, 50.0), np.full(xx.shape, 1.0)])
    zsmax[1, 2, 3] = 1.7  # cell on the polyline, after the start-up step
    zsmax[1, 0, 0] = 8.0  # cell far from the polyline
    map_nc = tmp_path / "sfincs_map.nc"
    xr.Dataset(
        {"zsmax": (("timemax", "n", "m"), zsmax)},
        coords={"x": (("n", "m"), xx), "y": (("n", "m"), yy)},
    ).to_netcdf(map_nc)
    return dict(
        baseline_sfincs_root=root,
        baseline_map_nc=map_nc,
        skeleton_weir=weir,
        adaptation_root=adapt,
        log=LOG,
    )


def _strategy(kind, scale, **extra):
    return {
        "measures": {"m": {"height": {"from_forcing": kind, "scale": scale}, **extra}}
    }


def _resolve(kind, scale, paths, **extra):
    out = resolve_forcing_params(_strategy(kind, scale, **extra), **paths)
    return out["measures"]["m"]["height"]


def test_read_weir_file_one_row_per_segment_with_max_crest(baseline):
    gdf = read_weir_file(baseline["skeleton_weir"])
    assert len(gdf) == 3
    assert list(gdf["elevation"]) == [3.0, 4.0, 9.0]
    assert all(g.geom_type == "LineString" for g in gdf.geometry)


@pytest.mark.parametrize(
    ("kind", "base"),
    [("max_coastal_water_level", BZS_MAX), ("max_discharge", DIS_MAX)],
)
def test_scale_one_reproduces_baseline_peak_and_scaling_is_monotonic(
    baseline, kind, base
):
    v1 = _resolve(kind, 1.0, baseline)
    v04 = _resolve(kind, 0.4, baseline)
    assert v1 == pytest.approx(base)
    assert v1 >= v04 >= 0


def test_river_waterlevel_uses_cells_along_polyline_and_drops_first_step(baseline):
    v = _resolve("max_river_water_level", 1.0, baseline, locations="levee.geojson")
    assert v == pytest.approx(1.7)
    assert (
        _resolve("max_river_water_level", 0.4, baseline, locations="levee.geojson") <= v
    )


def test_current_protection_is_unscaled_max_of_matched_crests(baseline):
    v = _resolve("current_protection", 0.4, baseline, locations="levee.geojson")
    assert v == {"current": True, "fill_crest": 4.0}


def test_input_strategy_is_not_mutated(baseline):
    strategy = _strategy("max_discharge", 1.0)
    resolve_forcing_params(strategy, **baseline)
    assert strategy["measures"]["m"]["height"] == {
        "from_forcing": "max_discharge",
        "scale": 1.0,
    }


def test_unknown_from_forcing_kind_is_rejected(baseline):
    with pytest.raises(ValueError, match="unknown from_forcing kind"):
        _resolve("max_rainfall", 1.0, baseline)


def test_missing_baseline_forcing_file_is_reported(baseline):
    (baseline["baseline_sfincs_root"] / "sfincs.dis").unlink()
    with pytest.raises(FileNotFoundError, match="sfincs.dis"):
        _resolve("max_discharge", 1.0, baseline)


def test_read_polyline_drops_empty_explodes_multi_and_reprojects():
    gdf = gpd.GeoDataFrame(
        geometry=[
            LineString([(0, 0), (1, 0)]),
            None,
            LineString(),
            MultiLineString([[(0, 1), (1, 1)], [(0, 2), (1, 2)]]),
        ],
        crs="EPSG:32631",
    )
    out = read_polyline(gdf, "EPSG:4326")
    assert len(out) == 3
    assert out.geometry.type.eq("LineString").all()
    assert out.crs == "EPSG:4326"
    assert list(out.columns) == ["geometry"]


def test_read_polyline_requires_crs():
    with pytest.raises(ValueError, match="has no CRS"):
        read_polyline(
            gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 0)])]), "EPSG:32631"
        )


def test_read_polyline_rejects_polygons(tmp_path):
    path = tmp_path / "area.geojson"
    gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (1, 0), (1, 1)])], crs="EPSG:32631"
    ).to_file(path)
    with pytest.raises(ValueError, match="LineString"):
        read_polyline(path, "EPSG:32631")
