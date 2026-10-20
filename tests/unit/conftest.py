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
    for k, (rid, spec) in enumerate(reaches.items()):
        dn = spec.get("dn", [])
        rows.append(
            {
                "reach_id": rid,
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
