"""
delta_polygon_modification.py
==============================
Utilities for detecting and correcting incomplete delta polygons from the
Edmonds et al. (2020) dataset.

Background
----------
The Edmonds et al. (2020) dataset provides polygons for major river deltas
worldwide. For several deltas the supplied polygon does not fully enclose the
estuary / river mouth. This module provides a pipeline to:

1. Load and unionise coastal reference data around a given delta polygon.
2. Classify the four vertices of each quadrilateral polygon into meaningful
   roles (delta node, two shore points, river-mouth point).
3. Iteratively scale the two offshore edges outward until they lie entirely
   within the coastline geometry, producing a corrected polygon.

Typical usage
-------------
    from delta_polygon_modification import unionize_coastal_data, modify_masks

    coastline  = unionize_coastal_data(delta_gdf, "path/to/coastline.gpkg")
    new_polygon = modify_masks(original_polygon, coastline, delta_id=42,
                               debug_plot=True, plot_rivers=rivers_gdf,
                               plot_lu=landuse_da)

Dependencies
------------
- geopandas, shapely, matplotlib, xarray
- Internal: src.input_processing.config.loader (config dict)
            src.input_processing.utils.plotting  (helper functions)
"""

from geopandas import GeoDataFrame
from shapely.geometry import LineString, Point, Polygon, box
from shapely.affinity import scale
from shapely.geometry.base import BaseGeometry

from src.input_processing.config.loader import config
from src.input_processing.utils.plotting import (
    create_bounds_around_delta,
    annotate_below_plot,
    get_esa_cmap,
)
import matplotlib.pyplot as plt
import geopandas as gpd
from typing import Tuple
from xarray import DataArray


# ---------------------------------------------------------------------------
# Build local bounding box
# ---------------------------------------------------------------------------


def build_local_bbox(
    geometry: BaseGeometry, buffer_m: float = config["Delta_masks"]["bbox_delta"]
):
    """
    Return a buffered bounding-box polygon around *geometry* in WGS-84.

    The box is constructed in the geometry's native (projected) CRS and then
    reprojected to EPSG:4326 for use as a spatial filter when loading raster
    and vector data that are stored in geographic coordinates.

    Parameters
    ----------
    geometry : shapely geometry
        Delta polygon in the project standard CRS.
    buffer_m : float
        Buffer distance in metres. Defaults to :data:`SPATIAL_BUFFER_M`.

    Returns
    -------
    shapely.geometry.Polygon
        Bounding-box polygon in EPSG:4326.
    """
    geometry = gpd.GeoSeries([geometry], crs=config["CRS"]["standard"]).to_crs(
        epsg=config["CRS"]["for_distances"]
    )
    geometry = geometry.geometry.values[0]
    minx, miny, maxx, maxy = geometry.bounds
    bbox_projected = box(
        minx - buffer_m,
        miny - buffer_m,
        maxx + buffer_m,
        maxy + buffer_m,
    )
    # Reproject to geographic CRS for use as a mask when reading files stored
    # in WGS-84 (the common case for global coastline / river datasets).
    bbox_4326 = gpd.GeoSeries(
        [bbox_projected], crs=config["CRS"]["for_distances"]
    ).to_crs(epsg=4326)[0]

    return bbox_4326


# ---------------------------------------------------------------------------
# Helper: load and prepare supporting data for one delta
# ---------------------------------------------------------------------------


def load_delta_context(geometry, land_use_full):
    """
    Load and pre-process all supporting data needed to modify one delta.

    All outputs are in the project standard CRS
    (``config['CRS']['for_distances']``).

    Parameters
    ----------
    geometry : shapely geometry
        Delta polygon in the project standard CRS.
    land_use_full : xarray.DataArray
        Full (lazily loaded) Copernicus land-use raster in its native CRS.

    Returns
    -------
    coast_geom : shapely geometry
        Unified coastline geometry for containment checks.
    rivers : GeoDataFrame
        River centre-lines clipped to the local bounding box.
    lu_builtup : xarray.DataArray
        Land-use raster clipped to the bounding box, reprojected to the
        standard CRS, filtered to built-up pixels only.
    """
    bbox_4326 = build_local_bbox(geometry)
    std_crs = config["CRS"]["for_distances"]

    # --- Coastline ---
    coastline = gpd.read_file(config["filepaths"]["coastline"], mask=bbox_4326).to_crs(
        epsg=std_crs
    )
    coast_geom = coastline.geometry.union_all()

    # --- Rivers ---
    rivers = gpd.read_file(
        config["Rivers"]["select_file_location"], mask=bbox_4326
    ).to_crs(epsg=std_crs)

    # --- Land use (built-up pixels only) ---
    # Clip to the local bbox first (still in the raster's native CRS / 4326),
    # then filter to the built-up class, then reproject to the standard CRS.
    lu_clipped = land_use_full.rio.clip([bbox_4326], from_disk=True).squeeze()
    lu_builtup = lu_clipped.where(lu_clipped == config["LandUse"]["code_builtup"])
    lu_builtup = lu_builtup.rio.reproject(std_crs)

    return coast_geom, rivers, lu_builtup


# ---------------------------------------------------------------------------
# Coastal data loading
# ---------------------------------------------------------------------------


def unionize_coastal_data(delta_polygon: GeoDataFrame, file_name: str):
    """
    Load a coastline vector file clipped to a buffered delta extent and return
    its geometry as a single unified shape.

    The delta polygon is scaled by a factor of 2 (about its own centroid)
    before being used as a spatial filter, ensuring the coastline extends
    well beyond the polygon edges so that offshore edge checks are reliable.

    Parameters
    ----------
    delta_polygon : GeoDataFrame
        Single-row GeoDataFrame whose ``geometry`` column holds the delta
        polygon of interest.
    file_name : str
        Path to any OGR-readable vector file containing coastline geometries
        (e.g. a GeoPackage or Shapefile).

    Returns
    -------
    shapely.geometry.base.BaseGeometry
        A single (possibly multi-part) geometry representing the union of all
        coastline features inside the search window.
    """
    # Expand the bounding mask to 2× the polygon size so we capture coastline
    # segments that extend past the polygon boundary.
    mask_polygon = scale(
        delta_polygon["geometry"].values[0], xfact=2, yfact=2, origin="center"
    )

    # Read only the features that intersect our enlarged mask, then reproject
    # to the project-wide standard CRS before unioning everything into one
    # geometry.
    coastline = gpd.read_file(file_name, mask=mask_polygon).to_crs(
        epsg=config["CRS"]["for_distances"]
    )
    return coastline.geometry.union_all()


# ---------------------------------------------------------------------------
# Diagnostic plotting
# ---------------------------------------------------------------------------


def plot_polygons_with_vertices(
    polygon_gdf: GeoDataFrame,
    coastline: Polygon,
    testcase_id: str,
):
    """
    Produce a diagnostic figure showing the delta polygon with its vertices
    labelled, overlaid on the coastline and river network.

    The figure is saved to ``figures/input_processing/validation/`` and is
    intended as a visual aid for identifying vertex numbering before running
    :func:`classify_points`.

    Parameters
    ----------
    polygon_gdf : GeoDataFrame
        Single-row GeoDataFrame containing the delta polygon.
    coastline : Polygon
        Unified coastline geometry returned by :func:`unionize_coastal_data`.
    testcase_id : str
        Key used to look up the human-readable delta name from the config.
    """
    # Look up the display name for axis title / file name.
    idx = config["Testcase"][testcase_id]

    fig, ax = plt.subplots(figsize=(10, 10))
    poly = polygon_gdf["geometry"].values[0]

    # Zoom the axes to a padded bounding box around the polygon.
    create_bounds_around_delta(ax, poly)

    # Load river centre-lines clipped to the polygon extent.
    rivers = gpd.read_file(config["filepaths"]["river"], mask=poly).to_crs(
        epsg=config["CRS"]["for_distances"]
    )

    # --- Coastline (grey reference line) ---
    gpd.GeoSeries([coastline]).plot(ax=ax, color="grey", linewidth=1)

    # --- Delta polygon outline (red, no fill) ---
    gpd.GeoSeries([poly]).plot(
        ax=ax,
        edgecolor="red",
        facecolor="none",
        linewidth=2,
    )

    # --- Vertex labels ---
    # Drop the closing duplicate coordinate before iterating.
    coords = list(poly.exterior.coords)[:-1]
    for i, coord in enumerate(coords):
        point = Point(coord)
        ax.scatter(point.x, point.y)
        ax.text(
            point.x,
            point.y,
            f"P{i + 1}",  # 1-based label for human readability
            fontsize=10,
            ha="right",
        )

    rivers.plot(ax=ax, legend="Lin et al.")
    ax.set_title(f"{idx} Delta Polygon to identify Point sequencing")

    plt.savefig(
        f"{config['filepaths']['test_masks']}/DeltaPolygon_PointPositions_{idx}.png",
        dpi=300,
    )


# ---------------------------------------------------------------------------
# Vertex classification
# ---------------------------------------------------------------------------


def classify_points(
    polygon: Polygon,
    coastline,
) -> Tuple[Point, Point, Point, Point, Point]:
    """
    Assign functional roles to the four vertices of a delta polygon.

    Edmonds et al. (2020) do not follow a consistent vertex ordering, so the
    roles must be inferred geometrically:

    * **Basin-ward extent point (BW)** – the vertex farthest from the coastline (i.e. most
      inland, at the apex of the delta).
    * **River-mouth point** – of the remaining three vertices, the one
      whose summed distance to the other two is smallest (i.e. it lies
      between the two shore points).
    * **Shore points s1 & s2** – the two remaining coastal vertices on
      either side of the river mouth.
    * **Delta node (DN)**: The DN is defined as either (1) the upstream-most bifurcation of the parent channel,
    or if no bifurcation is present, as (2) the intersection of the main channel with the delta shoreline vector
    (LS) which is defined as the line connecting S1 and S2.

    Parameters
    ----------
    polygon : Polygon
        The quadrilateral delta polygon to classify.
    coastline : shapely geometry
        Unified coastline geometry used as the distance reference.

    Returns
    -------
    tuple of Point
        ``(delta_node, s1, river_mouth, s2)``
    """
    coords = list(polygon.exterior.coords)[:-1]
    points = [Point(c) for c in coords]

    # The delta node is the vertex with the greatest distance to the coast —
    # it marks the upstream limit of the delta fan.
    basin_ward = max(points, key=lambda p: p.distance(coastline))

    position_bw = points.index(basin_ward)
    print((position_bw + 1) % len(points), (position_bw - 2) % len(points), position_bw)
    s1, s2 = points[position_bw - 1], points[(position_bw + 1) % len(points)]
    rm_dn_1, rm_dn_2 = (
        points[(position_bw - 2) % len(points)],
        points[(position_bw + 2) % len(points)],
    )

    # # Among the three remaining points we identify the river-mouth point as
    # # the one that minimises the sum of its distances to the other two.
    # # This works because the river mouth sits *between* the two shore points
    # # while those two are separated from each other on opposite banks.
    # best_rm = None
    # min_score = float("inf")
    #
    # for rm in remaining:
    #     shores = [p for p in remaining if p != rm]
    #     s1, s2 = shores
    #     score = rm.distance(s1) + rm.distance(s2)
    #
    #     if score < min_score:
    #         min_score = score
    #         best_rm = rm
    #         best_s1, best_s2 = s1, s2

    return Point(basin_ward), Point(s1), Point(s2), Point(rm_dn_1), Point(rm_dn_2)


# ---------------------------------------------------------------------------
# Edge construction helpers
# ---------------------------------------------------------------------------


def make_offshore_edges(s1: Point, rm_dn1: Point, rm_dn2: Point, s2: Point) -> list:
    """
    Build the three offshore edges of the delta polygon.

    These edges connect each shore point to the river-mouth and delta-node point and are the
    ones that may extend beyond the coastline for incomplete polygons.

    Parameters
    ----------
    s1 : Point
        First shore point.
    rm_dn1 : Point
        River-mouth point or delta-node
    rm_dn2 : Point
        River-mouth point or delta-node
    s2 : Point
        Second shore point.

    Returns
    -------
    list of LineString
        ``[edge_s1_rm, edge_rm_dn, edge_dn_s2]``
    """
    best_sequence = None
    min_score = float("inf")
    options = [rm_dn1, rm_dn2]
    for ii, _ in enumerate(options):
        sequence = [
            LineString([s1, options[ii]]),
            LineString([options[ii], options[ii - 1]]),
            LineString([options[ii - 1], s2]),
        ]

        score = (
            s1.distance(options[ii])
            + options[ii].distance(options[ii - 1])
            + options[ii - 1].distance(s2)
        )

        if score < min_score:
            min_score = score
            best_sequence = sequence
            best_rm_dn1 = options[ii]
            best_rm_dn2 = options[ii - 1]

    return best_sequence, best_rm_dn1, best_rm_dn2


def scale_point(origin: Point, point: Point, factor: int | float) -> Point:
    """
    Move *point* away from *origin* by the given scale factor.

    The new position is computed as a simple radial scaling:
    ``new = origin + factor * (point - origin)``
    so ``factor > 1`` moves the point further from the origin.

    Parameters
    ----------
    origin : Point
        Fixed reference point (the delta node in normal usage).
    point : Point
        The point to be moved (a shore or river-mouth point).
    factor : int or float
        Scaling factor. Values > 1 extend the point outward.

    Returns
    -------
    Point
        The rescaled point.
    """
    return Point(
        origin.x + (point.x - origin.x) * factor,
        origin.y + (point.y - origin.y) * factor,
    )


def edge_needs_scaling(edge: LineString, coastline: Polygon) -> bool:
    """
    Check whether any part of *edge* lies outside the coastline geometry.

    An edge needs further scaling (i.e. the polygon still doesn't reach the
    coast) when the difference between the edge and the coastline is
    non-empty — meaning some portion of the edge is not covered by land.

    Parameters
    ----------
    edge : LineString
        One of the two offshore edges to test.
    coastline : Polygon
        Unified coastline reference geometry.

    Returns
    -------
    bool
        ``True`` if the edge extends beyond the coastline; ``False`` if it
        is fully contained within or along the coast.
    """
    remainder = edge.difference(coastline)
    return not remainder.is_empty


# ---------------------------------------------------------------------------
# Debug / iteration plotting helpers
# ---------------------------------------------------------------------------


def plot_scaling_step(
    ax,
    original_poly: Polygon,
    current_poly: Polygon,
    iteration: int,
):
    """
    Add the original (blue dashed) and current (red solid) polygon outlines
    to an existing axes object during the iterative scaling loop.

    Labels are only set on the first iteration to avoid duplicate legend
    entries.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    original_poly : Polygon
        The unmodified input polygon (drawn for reference).
    current_poly : Polygon
        The polygon at the current iteration of scaling.
    iteration : int
        Zero-based iteration counter; used to suppress duplicate legend labels.
    """
    # Original polygon — blue dashed outline, label only on first iteration.
    gpd.GeoSeries([original_poly]).plot(
        ax=ax,
        edgecolor="blue",
        linestyle="--",
        facecolor="none",
        linewidth=1,
        label="original" if iteration == 0 else "",
    )

    # Current (scaled) polygon — solid red outline.
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
    edges: list,
    coastline: Polygon,
    delta_node: Point,
    s1: Point,
    best_rm_dn1: Point,
    best_rm_dn2: Point,
    s2: Point,
    delta_id: int,
    max_iter: int = 20,
    debug_plot: bool = False,
    plot_rivers=None,
    plot_lu=None,
):
    """
    Iteratively expand the offshore edges of a delta polygon until they are
    fully contained within the coastline geometry.

    Starting from the classified vertices, the function tests whether the two
    offshore edges (shore-to-river-mouth) extend beyond the coastline. If they
    do, the three movable vertices (s1, rm, s2) are scaled outward from the
    fixed delta node by ``config['Delta_masks']['scale_factor']`` on each
    iteration. The process repeats until both edges are fully inside the coast,
    or ``max_iter`` is reached.

    Parameters
    ----------
    edges : list of LineString
        The initial offshore edges produced by :func:`make_offshore_edges`.
    coastline : Polygon
        Unified coastline geometry used for containment checks.
    delta_node : Point
        Fixed inland apex of the delta (not moved during scaling).
    s1 : Point
        First (movable) shore point.
    best_rm_dn1, best_rm_dn2 : Point
        River-mouth or Delta-node point (movable).
    s2 : Point
        Second (movable) shore point.
    delta_id : int
        Numeric identifier for the delta, used in figure file names.
    max_iter : int, optional
        Maximum number of scaling iterations before giving up (default 20).
    debug_plot : bool, optional
        If ``True``, generate and save a step-by-step scaling figure.
    plot_rivers : GeoDataFrame or None, optional
        River centre-lines to overlay on the debug figure.
    plot_lu : DataArray or None, optional
        Land-use raster to overlay on the debug figure.

    Returns
    -------
    Polygon
        The (possibly expanded) delta polygon ``[delta_node, s1, rm, s2]``.

    Notes
    -----
    * If the polygon requires no scaling at all (iteration 0 passes), the
      original polygon is returned immediately.
    * A warning is printed if ``max_iter`` is exhausted; the best polygon
      reached so far is returned regardless.
    """
    scale_factor = config["Delta_masks"]["scale_factor"]

    # Keep a reference to the pre-scaling polygon for diagnostics.
    original_poly = Polygon([delta_node, s1, best_rm_dn1, best_rm_dn2, s2])

    # Gather statistics on the original polygon for figure annotations.
    delta_statistics = get_polygon_statistics(
        original_poly, delta_node, coastline, plot_lu
    )

    # --- Set up debug figure (optional) ---
    if debug_plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        gpd.GeoSeries([coastline]).plot(ax=ax, color="lightblue", linewidth=0.5)

        if isinstance(plot_rivers, GeoDataFrame):
            plot_rivers.plot(ax=ax)
        if isinstance(plot_lu, DataArray):
            cmap_esa, norm = get_esa_cmap()
            plot_lu.plot.imshow(ax=ax, alpha=1, cmap=cmap_esa, norm=norm)

    # --- Main iteration loop ---
    for i in range(max_iter):
        print(f"... iterating {i}")
        current_poly = Polygon([delta_node, s1, best_rm_dn1, best_rm_dn2, s2])

        # Check whether either offshore edge still extends beyond the coast.
        needs_scaling = any(edge_needs_scaling(edge, coastline) for edge in edges)

        if debug_plot:
            plot_scaling_step(ax, original_poly, current_poly, i)
            create_bounds_around_delta(ax, current_poly)

        # --- Termination conditions ---
        if not needs_scaling and i == 0:
            # Polygon was already correct; return without modification.
            print("Polygon did not need scaling.")
            return Polygon([delta_node, s1, best_rm_dn1, best_rm_dn2, s2])

        elif not needs_scaling and i != 0:
            # Scaling converged; save figure if requested and return.
            print(f"Scaling complete after {i} iterations.")
            if debug_plot:
                print("Plotting...")
                ax.set_title(
                    f"Delta {delta_id}: Comparing Edmonds et al. (2020) polygons with actual river and land use coverage"
                )
                annotate_below_plot(fig, ax, delta_statistics)
                fig.savefig(
                    f"{config['filepaths']['test_masks']}"
                    f"scaled/DeltaPolygon_{int(delta_id)}_scaled.png",
                    dpi=300,
                )
                plt.close()
            return Polygon([delta_node, s1, best_rm_dn1, best_rm_dn2, s2])

        # --- Scale the three movable vertices outward from the delta node ---
        s1 = scale_point(delta_node, s1, scale_factor)
        best_rm_dn1 = scale_point(delta_node, best_rm_dn1, scale_factor)
        best_rm_dn2 = scale_point(delta_node, best_rm_dn2, scale_factor)
        s2 = scale_point(delta_node, s2, scale_factor)

        # Recompute edges with the updated vertex positions for the next check.
        edges, rm_dn1, rm_dn2 = make_offshore_edges(s1, best_rm_dn1, best_rm_dn2, s2)

    # --- Fallback: max iterations reached without convergence ---
    print("Warning: max scaling iterations reached.")
    if debug_plot:
        print("Plotting...")
        ax.set_title(f"{delta_id} Delta Polygon to identify Point sequencing")
        annotate_below_plot(fig, ax, delta_statistics)
        fig.savefig(
            f"{config['filepaths']['test_masks']}"
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
    coastline: Polygon,
    landuse: None | GeoDataFrame,
) -> dict:
    """
    Compute summary statistics for a delta polygon for use in figure
    annotations and logging.

    Always computed
    ~~~~~~~~~~~~~~~
    * Distance from the delta node to the coastline (km).
    * Area of the initial polygon (km²).

    Computed when *landuse* is provided
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * Built-up area inside the polygon according to the ESA land-use
      classification (km²), derived by counting pixels whose value equals
      ``config['LandUse']['code_builtup']``.

    Parameters
    ----------
    polygon : Polygon
        The (original, pre-scaling) delta polygon.
    delta_node : Point
        Inland apex vertex of the polygon.
    coastline : Polygon
        Unified coastline geometry for distance computation.
    landuse : DataArray or None
        ESA land-use raster. Pass ``None`` to skip built-up area estimation.

    Returns
    -------
    dict
        Dictionary with human-readable string keys and integer values, ready
        for use with :func:`annotate_below_plot`.
    """
    delta_statistics = {}

    # Distance from the delta node to the nearest coastline point [km].
    delta_statistics["distance to coast [km]"] = int(
        delta_node.distance(coastline) / 1000
    )

    # Polygon area converted from m² to km².
    delta_statistics["area initial delta [km2]"] = int(polygon.area / 1_000_000)

    # --- Optional: built-up area from ESA land-use raster ---
    if isinstance(landuse, DataArray):
        # Clip the raster to the polygon extent.
        lu_clipped_delta = landuse.rio.clip([polygon], from_disk=True).squeeze()

        # Pixel dimensions in the raster's native CRS units (metres assumed).
        pixel_width = abs(float(landuse.rio.resolution()[0]))
        pixel_height = abs(float(landuse.rio.resolution()[1]))
        pixel_area = pixel_width * pixel_height  # m² per pixel

        # Count pixels classified as built-up and convert to km².
        n_pixels = int((lu_clipped_delta == config["LandUse"]["code_builtup"]).sum())
        area_km2 = (n_pixels * pixel_area) / 1e6

        delta_statistics["built-up area [km2] (inside delta)"] = int(area_km2)

    return delta_statistics


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def modify_masks(
    polygon: Polygon,
    coastline: Polygon,
    delta_id: int,
    debug_plot: bool,
    plot_rivers: GeoDataFrame,
    plot_lu: None | DataArray,
) -> Polygon:
    """
    High-level pipeline function: classify, build edges, and scale a single
    delta polygon so that its offshore edges reach the coastline.

    This is the main entry point for correcting an individual delta mask. It
    orchestrates the three processing steps in order:

    1. **Classify** — Identify the functional role of each polygon vertex
       using :func:`classify_points`.
    2. **Build edges** — Construct the two offshore edges that will be tested
       and extended via :func:`make_offshore_edges`.
    3. **Scale** — Iteratively expand the polygon until the edges are fully
       contained within the coastline via :func:`scale_polygon`.

    Parameters
    ----------
    polygon : Polygon
        The raw delta polygon from the Edmonds et al. (2020) dataset.
    coastline : Polygon
        Unified coastline geometry from :func:`unionize_coastal_data`.
    delta_id : int
        Numeric identifier for this delta (used in figure file names).
    debug_plot : bool
        If ``True``, generate step-by-step scaling figures.
    plot_rivers : GeoDataFrame
        River centre-lines for optional overlay on debug figures.
    plot_lu : DataArray or None
        ESA land-use raster for optional overlay and built-up area statistics.

    Returns
    -------
    Polygon
        The corrected (possibly enlarged) delta polygon.
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


def process_delta(row, land_use_full, logger, debug_plot: bool):
    """
    Run the full modification pipeline for one delta polygon.

    Wraps data loading and mask modification so that failures are isolated to
    the individual delta and do not abort the parent loop.

    Parameters
    ----------
    row : pandas.Series
        A single row from the delta polygons GeoDataFrame. Must contain
        ``geometry`` and ``BasinID2`` fields.
    land_use_full : xarray.DataArray
        Full (lazily loaded) land-use raster, passed in to avoid repeated
        file opens.
    debug_plot : bool
        Whether to generate step-by-step scaling figures.

    Returns
    -------
    shapely.geometry.Polygon or None
        The corrected polygon, or ``None`` if processing failed.
    """
    delta_id = row.BasinID2
    delta_area = row.geometry.area

    # --- Size filter ---
    if delta_area < config["Delta_masks"]["min_delta_area"]:
        logger.info(
            "Delta %s excluded: area %.1f km² is below %.0f km² threshold.",
            delta_id,
            delta_area / 1_000_000,
            config["Delta_masks"]["min_delta_area"] / 1_000_000,
        )
        return None

    logger.info(
        "Processing delta %s (area %.1f km²)...", delta_id, delta_area / 1_000_000
    )

    # Load all supporting data for this delta.
    # Heavy objects (clipped raster, GeoDataFrames) are scoped to this
    # function and released automatically when it returns.
    coast_geom, rivers, lu_builtup = load_delta_context(row.geometry, land_use_full)

    # Run the iterative polygon scaling.
    modified_polygon = modify_masks(
        row.geometry,
        coast_geom,
        delta_id,
        debug_plot,
        plot_rivers=rivers,
        plot_lu=lu_builtup,
    )

    return modified_polygon
