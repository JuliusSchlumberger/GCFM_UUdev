"""Shared fixtures for the unit tests (synthetic data only, no raw data).

The pipeline's own modules live under ``workflow/src`` and are imported as
``src.<module>``, as the Snakemake scripts do.
"""

import sys
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workflow"))


def _network(reaches: dict[str, dict]) -> gpd.GeoDataFrame:
    """Build a SWORD-like reach table from {reach_id: {"dn": [...], "width": w}}.

    ``rch_id_dn`` is written as a bracketed string, the way the cleaned SWORD
    network stores it. Geometries are dummy straight lines; only topology and
    width matter to the code under test.
    """
    rows = []
    for k, (reach_id, spec) in enumerate(reaches.items()):
        dn = spec.get("dn", [])
        rows.append(
            {
                "reach_id": reach_id,
                "rch_id_dn": "[" + ", ".join(dn) + "]",
                "width": float(spec.get("width", 100.0)),
                "geometry": LineString([(k * 1000.0, 0.0), (k * 1000.0 + 900.0, 0.0)]),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:32631")


@pytest.fixture
def network():
    """Return the factory that builds a synthetic reach table from a dict."""
    return _network


@pytest.fixture(scope="session")
def gvf_cfg() -> dict:
    """Return a complete ``river_depth_modelling.gvf`` block on a coarse test grid."""
    return {
        "node_spacing_m": 500,
        "target_smoothing_m": 3000,
        "step_threshold_m": 0.5,
        "structure_min_drop_m": 0.2,
        "use_sword_obstructions": True,
        "cut_rising_edges": True,
        "structures": [],
        "slope_floor": 1.0e-5,
        "mouth_level_tol_m": 0.5,
        "q_min_m3s": 1.0,
        "h_min_m": 0.5,
        "h_max_m": 30.0,
        "froude_cap": 0.95,
        "water_body": {"min_width_m": 1000.0, "sea_class_fraction": 0.5},
        "nudging": {"alpha": 1.0, "passes": 3},
        "least_squares": {
            "apply": "all",
            "rmse_tol_m": 0.10,
            "rho_m": 0.05,
            "prior_weight_m": 0.05,
        },
    }
