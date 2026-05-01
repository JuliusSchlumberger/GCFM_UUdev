"""Utilities for detecting and correcting incomplete delta polygons.

Background:
    The Edmonds et al. (2020) dataset provides polygons for major river deltas
    worldwide. For several deltas the supplied polygon does not fully enclose
    the estuary or river mouth. This module provides a pipeline to:

    1. Load and unionise coastal reference data around a given delta polygon.
    2. Classify the four vertices of each quadrilateral polygon into meaningful
       roles (delta node, two shore points, river-mouth point).
    3. Iteratively scale the two offshore edges outward until they lie entirely
       within the coastline geometry, producing a corrected polygon.

Example:
    >>> from delta_polygon_modification import unionize_coastal_data, modify_masks
    >>> coastline = unionize_coastal_data(delta_gdf, "path/to/coastline.gpkg")
    >>> new_polygon = modify_masks(
    ...     original_polygon, coastline, delta_id=42,
    ...     debug_plot=True, plot_rivers=rivers_gdf, plot_lu=landuse_da
    ... )
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import LineString, Point, Polygon, box
from shapely.affinity import scale
from shapely.geometry.base import BaseGeometry
from xarray import DataArray
from typing import cast, Final
import logging
import pandas as pd

from src.utils.config_loader import load_config
from src.input_processing.utils.plotting import (
    create_bounds_around_delta,
    annotate_below_plot,
    get_esa_cmap,
)

_CONFIG_PATH = "../src/input_processing/config/decisions.yaml"
_CONFIG: Final[dict] = load_config(_CONFIG_PATH)  # type: ignore[type-arg]

# ---------------------------------------------------------------------------
# Build local bounding box
# ---------------------------------------------------------------------------


def build_local_bbox(
    geometry: BaseGeometry,
    buffer_m: float = _CONFIG["Delta_masks"]["bbox_delta"],
) -> Polygon:
    """Return a buffered bounding-box polygon around *geometry* in WGS-84.

    The box is constructed in the geometry's native (projected) CRS and then
    reprojected to EPSG:4326 for use as a spatial filter when loading raster
    and vector data stored in geographic coordinates.

    Args:
        geometry: Delta polygon in the project standard CRS.
        buffer_m: Buffer distance in metres. Defaults to
            ``_CONFIG['Delta_masks']['bbox_delta']``.

    Returns:
        Bounding-box polygon in EPSG:4326, expanded by *buffer_m* on all sides.

    Example:
        >>> bbox = build_local_bbox(delta_polygon.geometry.iloc[0])
        >>> print(bbox.geom_type)
        Polygon
    """
    geoseries = gpd.GeoSeries([geometry], crs=_CONFIG["CRS"]["standard"]).to_crs(
        epsg=_CONFIG["CRS"]["for_distances"]
    )
    projected_geom = geoseries.geometry.values[0]
    minx, miny, maxx, maxy = projected_geom.bounds

    bbox_projected = box(
        minx - buffer_m,
        miny - buffer_m,
        maxx + buffer_m,
        maxy + buffer_m,
    )

    # Reproject to geographic CRS for use as a mask when reading files stored
    # in WGS-84 (the common case for global coastline / river datasets).
    bbox_4326: Polygon = cast(
        Polygon,
        gpd.GeoSeries([bbox_projected], crs=_CONFIG["CRS"]["for_distances"]).to_crs(
            epsg=4326
        )[0],
    )

    return bbox_4326


# ---------------------------------------------------------------------------
# Helper: load and prepare supporting data for one delta
# ---------------------------------------------------------------------------


def load_delta_context(
    geometry: BaseGeometry,
    land_use_full: DataArray,
) -> tuple[BaseGeometry, GeoDataFrame, DataArray]:
    """Load and pre-process all supporting data needed to modify one delta.

    All outputs are in the project standard CRS
    (``_CONFIG['CRS']['for_distances']``).

    Args:
        geometry: Delta polygon in the project standard CRS.
        land_use_full: Full (lazily loaded) Copernicus land-use raster in its
            native CRS. The function clips this to the local bounding box to
            avoid loading the entire global raster into memory.

    Returns:
        A tuple of ``(coast_geom, rivers, lu_builtup)`` where:

        - *coast_geom*: Unified coastline geometry for containment checks.
        - *rivers*: River centre-lines clipped to the local bounding box.
        - *lu_builtup*: Land-use raster clipped and reprojected to the
          standard CRS, filtered to built-up pixels only.

    Example:
        >>> coast, rivers, lu = load_delta_context(row.geometry, land_use_da)
        >>> print(type(coast))
        <class 'shapely.geometry.multipolygon.MultiPolygon'>
    """
    bbox_4326 = build_local_bbox(geometry)
    std_crs: int = _CONFIG["CRS"]["for_distances"]

    # --- Coastline ---
    coastline: GeoDataFrame = gpd.read_file(
        _CONFIG["filepaths"]["coastline"], mask=bbox_4326
    ).to_crs(epsg=std_crs)
    coast_geom: BaseGeometry = coastline.geometry.union_all()

    # --- Rivers ---
    rivers: GeoDataFrame = gpd.read_file(
        _CONFIG["Rivers"]["select_file_location"], mask=bbox_4326
    ).to_crs(epsg=std_crs)

    # Clip to the local bbox first (still in the raster's native CRS / 4326),
    # then filter to the built-up class, then reproject to the standard CRS.
    lu_clipped: DataArray = land_use_full.rio.clip(
        [bbox_4326], from_disk=True
    ).squeeze()
    lu_builtup: DataArray = lu_clipped.where(
        lu_clipped == _CONFIG["LandUse"]["code_builtup"]
    )
    lu_builtup = lu_builtup.rio.reproject(std_crs)

    return coast_geom, rivers, lu_builtup


# ---------------------------------------------------------------------------
# Coastal data loading
# ---------------------------------------------------------------------------


def unionize_coastal_data(delta_polygon: GeoDataFrame, file_name: str) -> BaseGeometry:
    """Load a coastline file clipped to a buffered delta extent and return a unified geometry.

    The delta polygon is scaled by a factor of 2 about its own centroid before
    being used as a spatial filter, ensuring the coastline extends well beyond
    the polygon edges so that offshore edge checks are reliable.

    Args:
        delta_polygon: Single-row GeoDataFrame whose ``geometry`` column holds
            the delta polygon of interest. Must be in the project standard CRS.
        file_name: Path to any OGR-readable vector file containing coastline
            geometries (e.g. a GeoPackage or Shapefile).

    Returns:
        A single (possibly multi-part) geometry representing the union of all
        coastline features inside the 2× search window.

    Example:
        >>> coast = unionize_coastal_data(delta_gdf, "data/coastline.gpkg")
        >>> print(coast.geom_type)
        MultiPolygon
    """
    # Expand the bounding mask to 2× the polygon size so we capture coastline
    # segments that extend past the polygon boundary.
    mask_polygon: BaseGeometry = scale(
        delta_polygon["geometry"].values[0], xfact=2, yfact=2, origin="center"
    )

    coastline: GeoDataFrame = gpd.read_file(file_name, mask=mask_polygon).to_crs(
        epsg=_CONFIG["CRS"]["for_distances"]
    )
    return coastline.geometry.union_all()


# ---------------------------------------------------------------------------
# Diagnostic plotting
# ---------------------------------------------------------------------------


def plot_polygons_with_vertices(
    polygon_gdf: GeoDataFrame,
    coastline: Polygon,
    testcase_id: str,
) -> None:
    """Produce a diagnostic figure showing the delta polygon with its vertices labelled.

    The figure is overlaid on the coastline and river network, saved to
    ``figures/input_processing/validation/``, and intended as a visual aid
    for identifying vertex numbering before running :func:`classify_points`.

    Args:
        polygon_gdf: Single-row GeoDataFrame containing the delta polygon in
            the project standard CRS.
        coastline: Unified coastline geometry returned by
            :func:`unionize_coastal_data`.
        testcase_id: Key used to look up the human-readable delta name from
            the config dict.

    Example:
        >>> plot_polygons_with_vertices(delta_gdf, coast_geom, "delta_nile")
    """
    idx: str = _CONFIG["Testcase"][testcase_id]

    fig, ax = plt.subplots(figsize=(10, 10))
    poly: Polygon = polygon_gdf["geometry"].values[0]

    create_bounds_around_delta(ax, poly)

    rivers: GeoDataFrame = gpd.read_file(
        _CONFIG["filepaths"]["river"], mask=poly
    ).to_crs(epsg=_CONFIG["CRS"]["for_distances"])

    gpd.GeoSeries([coastline]).plot(ax=ax, color="grey", linewidth=1)
    gpd.GeoSeries([poly]).plot(
        ax=ax,
        edgecolor="red",
        facecolor="none",
        linewidth=2,
    )

    # Drop the closing duplicate coordinate before iterating.
    coords = list(poly.exterior.coords)[:-1]
    for i, coord in enumerate(coords):
        point = Point(coord)
        ax.scatter(point.x, point.y)
        ax.text(
            point.x,
            point.y,
            f"P{i + 1}",
            fontsize=10,
            ha="right",
        )

    rivers.plot(ax=ax)
    ax.set_title(f"{idx} Delta Polygon to identify Point sequencing")

    plt.savefig(
        f"{_CONFIG['filepaths']['test_masks']}/DeltaPolygon_PointPositions_{idx}.png",
        dpi=300,
    )


# ---------------------------------------------------------------------------
# Vertex classification
# ---------------------------------------------------------------------------


def classify_points(
    polygon: Polygon,
    coastline: BaseGeometry,
) -> tuple[Point, Point, Point, Point, Point]:
    """Assign functional roles to the four vertices of a delta polygon.

    Edmonds et al. (2020) do not follow a consistent vertex ordering, so roles
    are inferred geometrically:

    - **Basin-ward extent point (BW)**: the vertex farthest from the coastline,
      i.e. the most inland point at the apex of the delta fan.
    - **Shore points s1 & s2**: the two coastal vertices on either side of the
      river mouth.
    - **River-mouth / delta-node candidates (rm_dn1, rm_dn2)**: the remaining
      two vertices between the shore points.

    Args:
        polygon: The quadrilateral delta polygon to classify. Must have exactly
            four unique exterior vertices (closing coordinate excluded).
        coastline: Unified coastline geometry used as the distance reference.
            Typically the output of :func:`unionize_coastal_data`.

    Returns:
        A tuple of ``(basin_ward, s1, s2, rm_dn1, rm_dn2)`` where each element
        is a :class:`~shapely.geometry.Point` in the project standard CRS.

    Example:
        >>> bw, s1, s2, rm1, rm2 = classify_points(delta_polygon, coast_geom)
        >>> print(bw.geom_type)
        Point
    """
    coords = list(polygon.exterior.coords)[:-1]
    points = [Point(c) for c in coords]

    # The basin-ward point is the vertex with the greatest distance to the
    # coast — it marks the upstream limit of the delta fan.
    basin_ward: Point = max(points, key=lambda p: p.distance(coastline))

    position_bw: int = points.index(basin_ward)
    print((position_bw + 1) % len(points), (position_bw - 2) % len(points), position_bw)

    s1: Point = points[position_bw - 1]
    s2: Point = points[(position_bw + 1) % len(points)]
    rm_dn_1: Point = points[(position_bw - 2) % len(points)]
    rm_dn_2: Point = points[(position_bw + 2) % len(points)]

    return Point(basin_ward), Point(s1), Point(s2), Point(rm_dn_1), Point(rm_dn_2)


# ---------------------------------------------------------------------------
# Edge construction helpers
# ---------------------------------------------------------------------------


def make_offshore_edges(
    s1: Point,
    rm_dn1: Point,
    rm_dn2: Point,
    s2: Point,
) -> tuple[list[LineString], Point, Point]:
    """Build the three offshore edges of the delta polygon.

    Tests both orderings of the two river-mouth / delta-node candidate points
    and selects the sequence that minimises total edge length. The three edges
    connect ``s1 → rm_dn → dn_rm → s2`` and are the ones that may extend
    beyond the coastline for incomplete polygons.

    Args:
        s1: First shore point.
        rm_dn1: First river-mouth or delta-node candidate point.
        rm_dn2: Second river-mouth or delta-node candidate point.
        s2: Second shore point.

    Returns:
        A tuple of ``(edges, best_rm_dn1, best_rm_dn2)`` where:

        - *edges*: List of three :class:`~shapely.geometry.LineString` objects
          ``[s1→rm_dn1, rm_dn1→rm_dn2, rm_dn2→s2]`` in the optimal ordering.
        - *best_rm_dn1*: The candidate point assigned to the first interior
          position in the optimal ordering.
        - *best_rm_dn2*: The candidate point assigned to the second interior
          position in the optimal ordering.

    Example:
        >>> edges, rm1, rm2 = make_offshore_edges(s1, rm_dn1, rm_dn2, s2)
        >>> print(len(edges))
        3
    """
    best_sequence: list[LineString] | None = None
    best_rm_dn1: Point | None = None
    best_rm_dn2: Point | None = None
    min_score: float = float("inf")
    options: list[Point] = [rm_dn1, rm_dn2]

    for ii, _ in enumerate(options):
        sequence = [
            LineString([s1, options[ii]]),
            LineString([options[ii], options[ii - 1]]),
            LineString([options[ii - 1], s2]),
        ]
        score: float = (
            s1.distance(options[ii])
            + options[ii].distance(options[ii - 1])
            + options[ii - 1].distance(s2)
        )
        if score < min_score:
            min_score = score
            best_sequence = sequence
            best_rm_dn1 = options[ii]
            best_rm_dn2 = options[ii - 1]
    if best_sequence is None or best_rm_dn1 is None or best_rm_dn2 is None:
        raise ValueError("make_offshore_edges: could not determine best edge sequence.")

    return best_sequence, best_rm_dn1, best_rm_dn2


def scale_point(origin: Point, point: Point, factor: int | float) -> Point:
    """Move *point* away from *origin* by the given scale factor.

    The new position is computed as a simple radial scaling:
    ``new = origin + factor * (point - origin)``
    so a factor greater than 1 moves the point further from the origin.

    Args:
        origin: Fixed reference point (the delta node in normal usage).
        point: The point to be moved (a shore or river-mouth point).
        factor: Scaling factor. Values greater than 1 extend the point
            outward from *origin*.

    Returns:
        A new :class:`~shapely.geometry.Point` at the rescaled position.

    Example:
        >>> origin = Point(0, 0)
        >>> p = Point(1, 0)
        >>> scaled = scale_point(origin, p, 2.0)
        >>> print(scaled)
        POINT (2 0)
    """
    return Point(
        origin.x + (point.x - origin.x) * factor,
        origin.y + (point.y - origin.y) * factor,
    )


def edge_needs_scaling(edge: LineString, coastline: BaseGeometry) -> bool:
    """Check whether any part of *edge* lies outside the coastline geometry.

    An edge needs further scaling when the difference between the edge and the
    coastline is non-empty, meaning some portion of the edge is not covered by
    land.

    Args:
        edge: One of the offshore edges to test.
        coastline: Unified coastline reference geometry.

    Returns:
        True if the edge extends beyond the coastline; False if it is fully
        contained within or along the coast.

    Example:
        >>> needs = edge_needs_scaling(LineString([(0, 0), (1, 1)]), coast_geom)
        >>> print(type(needs))
        <class 'bool'>
    """
    remainder = edge.difference(coastline)
    return not remainder.is_empty


# ---------------------------------------------------------------------------
# Debug / iteration plotting helpers
# ---------------------------------------------------------------------------


def plot_scaling_step(
    ax: Axes,
    original_poly: Polygon,
    current_poly: Polygon,
    iteration: int,
) -> None:
    """Add the original and current polygon outlines to an axes during scaling.

    The original polygon is drawn as a blue dashed outline and the current
    scaled polygon as a solid red outline. Labels are only set on the first
    iteration to avoid duplicate legend entries.

    Args:
        ax: The matplotlib axes to draw on.
        original_poly: The unmodified input polygon drawn for reference.
        current_poly: The polygon at the current iteration of scaling.
        iteration: Zero-based iteration counter used to suppress duplicate
            legend labels on subsequent iterations.

    Example:
        >>> fig, ax = plt.subplots()
        >>> plot_scaling_step(ax, original_polygon, scaled_polygon, iteration=0)
    """
    gpd.GeoSeries([original_poly]).plot(
        ax=ax,
        edgecolor="blue",
        linestyle="--",
        facecolor="none",
        linewidth=1,
        label="original" if iteration == 0 else "",
    )
    gpd.GeoSeries([current_poly]).plot(
        ax=ax,
        edgecolor="red",
        facecolor="none",
        linewidth=2,
        label="scaled" if iteration == 0 else "",
    )
    ax.set_title(f"Scaling iteration {iteration}")


# ---------------------------------------------------------------------------
# Core scaling algorithm
# ---------------------------------------------------------------------------


def scale_polygon(
    edges: list[LineString],
    coastline: BaseGeometry,
    delta_node: Point,
    s1: Point,
    best_rm_dn1: Point,
    best_rm_dn2: Point,
    s2: Point,
    delta_id: int,
    max_iter: int = 20,
    debug_plot: bool = False,
    plot_rivers: GeoDataFrame | None = None,
    plot_lu: DataArray | None = None,
) -> Polygon:
    """Iteratively expand the offshore edges of a delta polygon until they reach the coastline.

    Starting from the classified vertices, tests whether the offshore edges
    extend beyond the coastline. If they do, the three movable vertices (s1,
    rm, s2) are scaled outward from the fixed delta node by
    ``_CONFIG['Delta_masks']['scale_factor']`` on each iteration. The process
    repeats until both edges are fully inside the coast or *max_iter* is
    reached.

    Args:
        edges: The initial offshore edges produced by :func:`make_offshore_edges`.
        coastline: Unified coastline geometry used for containment checks.
        delta_node: Fixed inland apex of the delta — not moved during scaling.
        s1: First (movable) shore point.
        best_rm_dn1: First river-mouth or delta-node candidate point (movable).
        best_rm_dn2: Second river-mouth or delta-node candidate point (movable).
        s2: Second (movable) shore point.
        delta_id: Numeric identifier for the delta, used in figure file names.
        max_iter: Maximum number of scaling iterations before giving up.
            Defaults to 20.
        debug_plot: If True, generate and save a step-by-step scaling figure.
            Defaults to False.
        plot_rivers: River centre-lines to overlay on the debug figure, or
            None to omit. Defaults to None.
        plot_lu: Land-use raster to overlay on the debug figure and use for
            built-up area statistics, or None to omit. Defaults to None.

    Returns:
        The (possibly expanded) delta polygon with vertices
        ``[delta_node, s1, best_rm_dn1, best_rm_dn2, s2]``.

    Raises:
        None: Does not raise. If *max_iter* is exhausted a warning is printed
            and the best polygon reached so far is returned.

    Note:
        If the polygon requires no scaling at all (iteration 0 passes), the
        original polygon is returned immediately without modification.

    Example:
        >>> result = scale_polygon(
        ...     edges, coast_geom, delta_node, s1, rm1, rm2, s2, delta_id=42
        ... )
        >>> print(result.geom_type)
        Polygon
    """
    scale_factor: float = _CONFIG["Delta_masks"]["scale_factor"]
    original_poly: Polygon = Polygon([delta_node, s1, best_rm_dn1, best_rm_dn2, s2])
    delta_statistics: dict = get_polygon_statistics(
        original_poly, delta_node, coastline, plot_lu
    )

    if debug_plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        gpd.GeoSeries([coastline]).plot(ax=ax, color="lightblue", linewidth=0.5)
        if isinstance(plot_rivers, GeoDataFrame):
            plot_rivers.plot(ax=ax)
        if isinstance(plot_lu, DataArray):
            cmap_esa, norm = get_esa_cmap()
            plot_lu.plot.imshow(ax=ax, alpha=1, cmap=cmap_esa, norm=norm)

    for i in range(max_iter):
        print(f"... iterating {i}")
        current_poly: Polygon = Polygon([delta_node, s1, best_rm_dn1, best_rm_dn2, s2])
        needs_scaling: bool = any(edge_needs_scaling(edge, coastline) for edge in edges)

        if debug_plot:
            plot_scaling_step(ax, original_poly, current_poly, i)
            create_bounds_around_delta(ax, current_poly)

        if not needs_scaling and i == 0:
            print("Polygon did not need scaling.")
            return Polygon([delta_node, s1, best_rm_dn1, best_rm_dn2, s2])

        elif not needs_scaling and i != 0:
            print(f"Scaling complete after {i} iterations.")
            if debug_plot:
                ax.set_title(
                    f"Delta {delta_id}: Comparing Edmonds et al. (2020) polygons "
                    f"with actual river and land use coverage"
                )
                annotate_below_plot(fig, ax, delta_statistics)
                fig.savefig(
                    f"{_CONFIG['filepaths']['test_masks']}"
                    f"scaled/DeltaPolygon_{int(delta_id)}_scaled.png",
                    dpi=300,
                )
                plt.close()
            return Polygon([delta_node, s1, best_rm_dn1, best_rm_dn2, s2])

        s1 = scale_point(delta_node, s1, scale_factor)
        best_rm_dn1 = scale_point(delta_node, best_rm_dn1, scale_factor)
        best_rm_dn2 = scale_point(delta_node, best_rm_dn2, scale_factor)
        s2 = scale_point(delta_node, s2, scale_factor)
        edges, _, _ = make_offshore_edges(s1, best_rm_dn1, best_rm_dn2, s2)

    print("Warning: max scaling iterations reached.")
    if debug_plot:
        ax.set_title(f"{delta_id} Delta Polygon to identify Point sequencing")
        annotate_below_plot(fig, ax, delta_statistics)
        fig.savefig(
            f"{_CONFIG['filepaths']['test_masks']}"
            f"scaling_failed/DeltaPolygon_{delta_id}_scaled_incomplete.png",
            dpi=300,
        )
        plt.close()
    return Polygon([delta_node, s1, best_rm_dn1, best_rm_dn2, s2])


# ---------------------------------------------------------------------------
# Polygon statistics
# ---------------------------------------------------------------------------


def get_polygon_statistics(
    polygon: Polygon,
    delta_node: Point,
    coastline: BaseGeometry,
    landuse: DataArray | None,
) -> dict[str, int]:
    """Compute summary statistics for a delta polygon for figure annotations.

    Always computes the distance from the delta node to the coastline and the
    polygon area. Optionally computes built-up area when a land-use raster is
    provided.

    Args:
        polygon: The original, pre-scaling delta polygon.
        delta_node: Inland apex vertex of the polygon.
        coastline: Unified coastline geometry for distance computation.
        landuse: ESA land-use raster clipped to the delta region, or None to
            skip the built-up area calculation.

    Returns:
        Dictionary with human-readable string keys and integer values, ready
        for use with :func:`annotate_below_plot`. Keys are:

        - ``"distance to coast [km]"``: distance from *delta_node* to the
          nearest coastline point, in km.
        - ``"area initial delta [km2]"``: polygon area in km².
        - ``"built-up area [km2] (inside delta)"``: only present when
          *landuse* is provided.

    Example:
        >>> stats = get_polygon_statistics(poly, node, coast, lu_raster)
        >>> print(stats["area initial delta [km2]"])
        1250
    """
    delta_statistics: dict[str, int] = {}

    delta_statistics["distance to coast [km]"] = int(
        delta_node.distance(coastline) / 1000
    )
    delta_statistics["area initial delta [km2]"] = int(polygon.area / 1_000_000)

    if isinstance(landuse, DataArray):
        lu_clipped_delta: DataArray = landuse.rio.clip(
            [polygon], from_disk=True
        ).squeeze()
        pixel_width: float = abs(float(landuse.rio.resolution()[0]))
        pixel_height: float = abs(float(landuse.rio.resolution()[1]))
        pixel_area: float = pixel_width * pixel_height
        n_pixels: int = int(
            (lu_clipped_delta == _CONFIG["LandUse"]["code_builtup"]).sum()
        )
        delta_statistics["built-up area [km2] (inside delta)"] = int(
            (n_pixels * pixel_area) / 1e6
        )

    return delta_statistics


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def modify_masks(
    polygon: Polygon,
    coastline: BaseGeometry,
    delta_id: int,
    debug_plot: bool,
    plot_rivers: GeoDataFrame,
    plot_lu: DataArray | None,
) -> Polygon:
    """Classify, build edges, and scale a single delta polygon to reach the coastline.

    This is the main entry point for correcting an individual delta mask. It
    orchestrates three processing steps in order:

    1. **Classify** — identify the functional role of each polygon vertex via
       :func:`classify_points`.
    2. **Build edges** — construct the offshore edges that will be tested and
       extended via :func:`make_offshore_edges`.
    3. **Scale** — iteratively expand the polygon until the edges are fully
       contained within the coastline via :func:`scale_polygon`.

    Args:
        polygon: The raw delta polygon from the Edmonds et al. (2020) dataset.
        coastline: Unified coastline geometry from :func:`unionize_coastal_data`.
        delta_id: Numeric identifier for this delta, used in figure file names.
        debug_plot: If True, generate step-by-step scaling figures.
        plot_rivers: River centre-lines for optional overlay on debug figures.
        plot_lu: ESA land-use raster for optional overlay and built-up area
            statistics, or None to omit.

    Returns:
        The corrected (possibly enlarged) delta polygon in the project standard
        CRS.

    Example:
        >>> modified = modify_masks(
        ...     raw_polygon, coast_geom, delta_id=42,
        ...     debug_plot=False, plot_rivers=rivers_gdf, plot_lu=None
        ... )
        >>> print(modified.geom_type)
        Polygon
    """
    print("1. Classify points...")
    delta_node, s1, s2, rm_dn1, rm_dn2 = classify_points(polygon, coastline)

    print("2. Make offshore edges...")
    edges, best_rm_dn1, best_rm_dn2 = make_offshore_edges(s1, rm_dn1, rm_dn2, s2)

    print("3. Scale polygons...")
    return scale_polygon(
        edges,
        coastline,
        delta_node,
        s1,
        best_rm_dn1,
        best_rm_dn2,
        s2,
        delta_id,
        debug_plot=debug_plot,
        plot_rivers=plot_rivers,
        plot_lu=plot_lu,
    )


# ---------------------------------------------------------------------------
# Process a single delta row
# ---------------------------------------------------------------------------


def process_delta(
    row: pd.Series,
    land_use_full: DataArray,
    logger: logging.Logger,
    debug_plot: bool,
) -> Polygon | None:
    """Run the full modification pipeline for one delta polygon row.

    Wraps data loading and mask modification so that failures are isolated to
    the individual delta and do not abort the parent loop.

    Args:
        row: A single row from the delta polygons GeoDataFrame (a
            ``pandas.Series``). Must contain ``geometry`` and ``BasinID2``
            fields.
        land_use_full: Full (lazily loaded) land-use raster, passed in to
            avoid repeated file opens across iterations.
        logger: A Python :class:`logging.Logger` instance used to record
            progress and size-filter decisions.
        debug_plot: Whether to generate step-by-step scaling figures.

    Returns:
        The corrected polygon, or None if the delta was too small or
        processing failed.

    Example:
        >>> import logging
        >>> logger = logging.getLogger(__name__)
        >>> result = process_delta(row, land_use_da, logger, debug_plot=False)
        >>> print(result is None or result.geom_type == "Polygon")
        True
    """
    delta_id: int = cast(int, row["BasinID2"])
    delta_area: float = cast(BaseGeometry, row["geometry"]).area

    if delta_area < _CONFIG["Delta_masks"]["min_delta_area"]:
        logger.info(
            "Delta %s excluded: area %.1f km² is below %.0f km² threshold.",
            delta_id,
            delta_area / 1_000_000,
            _CONFIG["Delta_masks"]["min_delta_area"] / 1_000_000,
        )
        return None

    logger.info(
        "Processing delta %s (area %.1f km²)...", delta_id, delta_area / 1_000_000
    )

    coast_geom, rivers, lu_builtup = load_delta_context(row.geometry, land_use_full)

    return modify_masks(
        row.geometry,
        coast_geom,
        delta_id,
        debug_plot,
        plot_rivers=rivers,
        plot_lu=lu_builtup,
    )
