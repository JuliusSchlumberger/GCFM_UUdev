from __future__ import annotations

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from shapely.strtree import STRtree
from typing import Final
from src.input_processing.config.loader import config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CRS_STANDARD: Final[int] = config["CRS"]["standard"]
CRS_PROJECTED: Final[int] = config["CRS"]["for_distances"]

GEOM_COL: Final[str] = config["DomainSchema"]["geometry_lbl"]
BASIN_COL: Final[str] = config["DomainSchema"]["delta_id_lbl"]
AREA_COL: Final[str] = config["DomainSchema"]["area_lbl"]
MIN_DELTA_AREA_KM2: Final[float] = config["Delta_masks"]["min_delta_area"]


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------
def _validate_schema(gdf: GeoDataFrame, excluded: list) -> None:
    """Raise ValueError if any required column is absent from *gdf*."""
    required: list[str] = [
        config["DomainSchema"][k]
        for k in config["DomainSchema"]
        if config["DomainSchema"][k] not in excluded
    ]
    missing: list[str] = [col for col in required if col not in gdf.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def ensure_valid_schema(
    gdf: GeoDataFrame, excluded: list[str] | None = None
) -> GeoDataFrame:
    """
    Validate schema and return a normalised copy of *gdf*.

    Performs:
    - Schema presence check (raises on missing columns).
    - CRS enforcement to CRS_STANDARD.
    - Removal of rows whose geometry is null.
    """
    excluded = excluded or []
    _validate_schema(gdf, excluded)

    gdf = gdf.copy()

    # Enforce CRS — re-project if the file was saved with a different CRS.
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=CRS_STANDARD)
    elif gdf.crs.to_epsg() != CRS_STANDARD:
        gdf = gdf.to_crs(epsg=CRS_STANDARD)

    # Drop null geometries that would cause silent errors downstream.
    null_geom_count: int = gdf.geometry.isna().sum()
    if null_geom_count > 0:
        print(
            f"[ensure_valid_schema] Dropping {null_geom_count} row(s) with null geometry."
        )
        gdf = gdf[gdf.geometry.notna()].copy()

    return gdf


def compute_area_km2(gdf: GeoDataFrame) -> pd.Series:
    """
    Return a Series of integer area values in km² for each row of *gdf*.

    *gdf* is assumed to be in CRS_STANDARD (geographic, degrees).
    It is projected to CRS_PROJECTED (metric) before area calculation.
    """
    projected: GeoDataFrame = gdf.to_crs(epsg=CRS_PROJECTED)
    areas: pd.Series = projected.geometry.area / 1e6
    return areas.astype(int)


def build_spatial_index(gdf: GeoDataFrame) -> STRtree:
    """Return an STRtree spatial index built from the geometries of *gdf*."""
    return STRtree(gdf.geometry.values)


def find_intersections(polygons: GeoDataFrame, river_tree: STRtree) -> np.ndarray:
    """
    Return a boolean mask (shape ``(len(polygons),)``) where True means the
    polygon at that position intersects at least one geometry in *river_tree*.

    Uses a vectorised bulk query instead of a Python-level loop.
    ``STRtree.query`` with an array of geometries returns a (2, N) array of
    ``[geometry_index, tree_index]`` pairs for every hit.
    """
    # query() with an array input returns shape (2, n_hits):
    #   row 0 → index into `polygons.geometry`
    #   row 1 → index into the STRtree
    hits: np.ndarray = river_tree.query(polygons.geometry, predicate="intersects")

    mask: np.ndarray = np.zeros(len(polygons), dtype=bool)
    if hits.size > 0:
        mask[np.unique(hits[0])] = True
    return mask
