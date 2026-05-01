"""Global and per-delta data loading utilities for the river source pipeline.

Heavy datasets (rivers, coastline, GloFAS) are loaded once via
``load_global_data()`` and stored in a frozen ``GlobalData`` dataclass.
Per-delta subsets are then derived cheaply via ``load_data_delta_domain()``
using coordinate-index slicing and xarray selection — no repeated file I/O.

Example:
    >>> global_data = load_global_data()
    >>> rivers, coast, coastline_gdf, glofas_min = load_data_delta_domain(
    ...     delta_basins_gdf, global_data
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import xarray as xr
from geopandas import GeoDataFrame
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from src.input_processing.utils.validation.modify_delta_masks import build_local_bbox
from src.utils.config_loader import load_config
from src.utils.setup_logger import setup_logging

_LOG = setup_logging("data_loader")

_CONFIG_PATH = "../src/input_processing/config/decisions.yaml"
_CONFIG: dict = load_config(_CONFIG_PATH)  # type: ignore[type-arg]

# ---------------------------------------------------------------------------
# GlobalData container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GlobalData:
    """Immutable container for the heavy global datasets loaded once at startup.

    Storing all three datasets in a single frozen dataclass makes it easy to
    pass the full context through the pipeline without repeated file I/O, and
    ``frozen=True`` prevents accidental mutation during the per-delta loop.

    Attributes:
        rivers: Full global river network (SWORD) in the project standard CRS.
        coastline: Global coastline polygons in the project standard CRS.
        glofas: GloFAS discharge Dataset covering all time steps and the full
            global extent. Kept as a lazily-evaluated xarray Dataset so that
            per-delta ``.sel()`` slices remain cheap.

    Example:
        >>> data = load_global_data()
        >>> print(type(data.glofas))
        <class 'xarray.core.dataset.Dataset'>
    """

    rivers: GeoDataFrame
    coastline: GeoDataFrame
    glofas: xr.Dataset


# ---------------------------------------------------------------------------
# Global loader — call once at pipeline startup
# ---------------------------------------------------------------------------


def load_global_data() -> GlobalData:
    """Load and validate the three heavy global datasets.

    Reads the river network, coastline, and GloFAS discharge file from the
    paths defined in the project config. Both vector datasets are reprojected
    to the project standard CRS on load. The GloFAS dataset is opened lazily
    so the full file is not read into memory until a ``.sel()`` slice is
    evaluated.

    Returns:
        A frozen ``GlobalData`` instance ready to be passed into
        ``load_data_delta_domain()`` for per-delta subsetting.

    Raises:
        FileNotFoundError: If any of the configured file paths do not exist.
        ValueError: If the CRS reprojection fails for the river or coastline
            datasets.

    Example:
        >>> global_data = load_global_data()
        >>> print(len(global_data.rivers))  # number of river segments loaded
        142301
    """
    _LOG.info("Loading global datasets ...")

    _LOG.info(
        "  Reading river network (SWORD): %s", _CONFIG["filepaths"]["river_sword"]
    )
    rivers: GeoDataFrame = gpd.read_file(_CONFIG["filepaths"]["river_sword"]).to_crs(
        epsg=_CONFIG["CRS"]["standard"]
    )
    _LOG.info("  Rivers loaded: %d segments", len(rivers))

    _LOG.info("  Reading coastline: %s", _CONFIG["filepaths"]["coastline"])
    coastline: GeoDataFrame = gpd.read_file(_CONFIG["filepaths"]["coastline"]).to_crs(
        epsg=_CONFIG["CRS"]["standard"]
    )
    _LOG.info("  Coastline loaded: %d features", len(coastline))

    _LOG.info("  Opening GloFAS dataset (lazy): %s", _CONFIG["filepaths"]["glofas"])
    glofas: xr.Dataset = xr.open_dataset(_CONFIG["filepaths"]["glofas"])
    glofas = glofas.rio.write_crs(_CONFIG["CRS"]["standard"])
    _LOG.info(
        "  GloFAS opened: dims=%s, variables=%s",
        dict(glofas.dims),
        list(glofas.data_vars),
    )

    _LOG.info("All global datasets loaded.")
    return GlobalData(rivers=rivers, coastline=coastline, glofas=glofas)


# ---------------------------------------------------------------------------
# Per-delta subset — cheap slice, no I/O
# ---------------------------------------------------------------------------


def load_data_delta_domain(
    delta_basin_mask: GeoDataFrame | Polygon | MultiPolygon,
    global_data: GlobalData,
) -> tuple[GeoDataFrame, BaseGeometry, GeoDataFrame, xr.DataArray]:
    """Clip global datasets to the bounding box of a single delta basin.

    Accepts either a GeoDataFrame of basin polygons or a pre-dissolved shapely
    geometry. All three global datasets are subset using coordinate-index
    slicing (``.cx`` for vectors, ``.sel`` for GloFAS) so no file I/O occurs.
    The coastline union is simplified before returning to prevent
    ``clip_basin_boundary_from_coast`` from hanging on geometrically complex
    coastlines.

    Args:
        delta_basin_mask: Either a GeoDataFrame of basin polygons for the
            target delta, or a pre-dissolved ``Polygon`` or ``MultiPolygon``.
            If a GeoDataFrame is passed, all geometries are dissolved into one
            before the bounding box is computed.
        global_data: The ``GlobalData`` instance returned by
            ``load_global_data()``. Must not be None.

    Returns:
        A tuple of ``(rivers_gpd, coast_polygon, coastline_gpd, glofas_min)``
        where:

        - *rivers_gpd*: River segments clipped to the delta bounding box, in
          the project standard CRS.
        - *coast_polygon*: Simplified union of coastline polygons within the
          bounding box. Typed as ``BaseGeometry`` since ``union_all()`` may
          return ``Polygon`` or ``MultiPolygon`` depending on coastline
          complexity in the region.
        - *coastline_gpd*: Raw clipped coastline GeoDataFrame before unioning,
          retained for plotting and diagnostics.
        - *glofas_min*: Per-cell minimum discharge ``DataArray`` over all
          time steps in the clipped region, derived from the ``dis24``
          variable.

    Raises:
        KeyError: If the GloFAS dataset does not contain a ``dis24`` variable
            or the expected ``longitude`` / ``latitude`` dimension names.

    Example:
        >>> global_data = load_global_data()
        >>> rivers, coast, coastline_gdf, glofas_min = load_data_delta_domain(
        ...     delta_basins_gdf, global_data
        ... )
        >>> print(glofas_min.dims)
        ('latitude', 'longitude')
    """
    # --- Normalise input to a single geometry ---
    delta_geom: BaseGeometry
    if isinstance(delta_basin_mask, (Polygon, MultiPolygon)):
        delta_geom = delta_basin_mask
    else:
        _LOG.debug("Dissolving GeoDataFrame basin mask into a single geometry.")
        delta_geom = delta_basin_mask.geometry.union_all()

    # --- Bounding box in WGS-84 for coordinate-index slicing ---
    bbox_4326: BaseGeometry = build_local_bbox(delta_geom)
    minx: float
    miny: float
    maxx: float
    maxy: float
    minx, miny, maxx, maxy = bbox_4326.bounds
    _LOG.debug(
        "Delta bounding box (WGS-84): %.4f, %.4f → %.4f, %.4f", minx, miny, maxx, maxy
    )

    # --- Spatial subsets (no file I/O) ---
    rivers_gpd: GeoDataFrame = global_data.rivers.cx[minx:maxx, miny:maxy].copy()
    _LOG.debug("Rivers clipped to bbox: %d segments", len(rivers_gpd))

    coastline_gpd: GeoDataFrame = global_data.coastline.cx[minx:maxx, miny:maxy].copy()
    _LOG.debug("Coastline clipped to bbox: %d features", len(coastline_gpd))

    # Simplify the union to prevent .difference() from hanging on highly
    # fragmented coastlines (e.g. regions with 200+ small polygons).
    tolerance: float = _CONFIG["Delta_masks"]["tolerance_deg"]
    coast_polygon: BaseGeometry = coastline_gpd.geometry.union_all().simplify(
        tolerance, preserve_topology=True
    )
    _LOG.debug(
        "Coastline union simplified (tolerance=%.6f deg): geometry type=%s",
        tolerance,
        coast_polygon.geom_type,
    )

    # --- GloFAS temporal minimum over the clipped region ---
    # latitude is stored north→south so the slice order is (maxy, miny).
    glofas_clip: xr.Dataset = global_data.glofas.sel(
        longitude=slice(minx, maxx),
        latitude=slice(maxy, miny),
    )
    glofas_min: xr.DataArray = glofas_clip.dis24.min(dim="valid_time")
    _LOG.debug("GloFAS clipped: dims=%s", dict(glofas_clip.dims))

    return rivers_gpd, coast_polygon, coastline_gpd, glofas_min
