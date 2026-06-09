"""Summary plot functions for all workflow steps."""

from __future__ import annotations

import logging
from typing import Any
import math
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rioxarray  # noqa: F401  — registers the .rio accessor used for reprojection
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import Polygon

from src.geometry import pick_utm_crs

matplotlib.use("Agg")

log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

_PLOT_DPI = 100
_PLOT_MAX_PX = 2_000_000  # downsample rasters larger than this before rendering

_VMIN_ELEV = -30.0
_VMAX_ELEV = 80.0
_THRESH_COLOR = "#FF4500"  # used for pixels above _VMAX_ELEV in topography plots

_LC_NAMES: dict[int, str] = {
    0: "Unknown",
    20: "Shrubland",
    30: "Herbaceous vegetation",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow / ice",
    80: "Inland water",
    90: "Herbaceous wetland",
    100: "Moss / lichen",
    111: "Closed ENF",
    112: "Closed EBF",
    113: "Closed DNF",
    114: "Closed DBF",
    115: "Closed mixed forest",
    116: "Closed unknown forest",
    121: "Open ENF",
    122: "Open EBF",
    123: "Open DNF",
    124: "Open DBF",
    125: "Open mixed forest",
    126: "Open unknown forest",
    200: "Sea",
}

_LC_COLORS: dict[int, str] = {
    0: "#808080",
    20: "#B27800",
    30: "#A0C050",
    40: "#E8E858",
    50: "#C03830",
    60: "#D8CC80",
    70: "#F0F0F0",
    80: "#3860D0",
    90: "#70C0A0",
    100: "#788850",
    111: "#005000",
    112: "#006800",
    113: "#003830",
    114: "#004820",
    115: "#004010",
    116: "#003818",
    121: "#288028",
    122: "#309030",
    123: "#186818",
    124: "#208820",
    125: "#188818",
    126: "#188870",
    200: "#0030A0",
}


# ── shared helpers ────────────────────────────────────────────────────────────


def read_raster_for_plot(
    path: str,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """
    Read a raster band, mask nodata to NaN, and downsample for plotting.

    Rasters larger than _PLOT_MAX_PX pixels are subsampled by a factor chosen
    to bring them below that threshold, preventing multi-GiB memory allocations
    when matplotlib renders the image.

    Args:
        path: Path to a GeoTIFF file.

    Returns:
        data:   2-D float array with nodata replaced by NaN.
        extent: (left, right, bottom, top) in the raster CRS, suitable for
                passing as ``extent`` to imshow.
    """
    import rasterio

    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        nodata = src.nodata
        b = src.bounds
        extent = (b.left, b.right, b.bottom, b.top)
    if nodata is not None:
        data[data == nodata] = np.nan
    total = data.shape[0] * data.shape[1]
    if total > _PLOT_MAX_PX:
        factor = math.ceil(math.sqrt(total / _PLOT_MAX_PX))
        data = data[::factor, ::factor]
    return data, extent


def _to_wgs84_grid(
    data: np.ndarray,
    transform,
    src_crs,
    dst_bounds: tuple[float, float, float, float] | None = None,
):
    """
    Reproject a 2-D array from (transform, src_crs) onto a north-up EPSG:4326 grid.

    Brings rasters that live on the model's metric (UTM) working grid into
    the same lon/lat reference frame as every other diagnostic map, so axes
    are consistently labelled in degrees. Nearest-neighbour resampling
    preserves exact source values (no blending across nodata / threshold
    boundaries).

    Args:
        dst_bounds: Optional (left, bottom, right, top) in EPSG:4326.  When
            supplied the destination grid is created to fill exactly these bounds
            (no rasterio padding), so the imshow extent matches the domain bbox
            precisely.  When omitted rasterio's default padded bounds are used.

    Returns:
        data_wgs84: 2-D float32 array on the reprojected grid (NaN = nodata).
        extent:     (left, right, bottom, top) in degrees, for imshow ``extent``.
    """
    from rasterio.warp import calculate_default_transform, reproject as rio_reproject
    from rasterio.enums import Resampling
    from rasterio.transform import array_bounds, from_bounds as rio_from_bounds

    h, w = data.shape
    src_bounds = array_bounds(h, w, transform)
    # Always derive pixel count from calculate_default_transform so the output
    # resolution is appropriate for the source data.
    default_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, "EPSG:4326", w, h, *src_bounds
    )
    if dst_bounds is not None:
        left, bottom, right, top = dst_bounds
        dst_transform = rio_from_bounds(left, bottom, right, top, dst_w, dst_h)
    else:
        dst_transform = default_transform
        left, bottom, right, top = array_bounds(dst_h, dst_w, dst_transform)

    dst = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
    rio_reproject(
        source=data.astype(np.float32),
        destination=dst,
        src_transform=transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.nearest,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst, (left, right, bottom, top)


def read_raster_reprojected_for_plot(
    path: str,
    dst_bounds: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """
    Read a raster band on its native (typically UTM) grid, reproject it to
    EPSG:4326, mask nodata to NaN, and downsample for plotting.

    Counterpart to read_raster_for_plot for rasters that are stored on the
    model's metric working grid — reprojecting here keeps every diagnostic
    map in the same lon/lat reference frame.

    Args:
        dst_bounds: Optional (left, bottom, right, top) in EPSG:4326.  Pass
            the domain WGS84 bbox to ensure the reprojected raster fills the
            plot bbox exactly (no padding-induced shift).
    """
    import rasterio

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
    if nodata is not None:
        data[data == nodata] = np.nan

    data, extent = _to_wgs84_grid(data, transform, crs, dst_bounds=dst_bounds)

    total = data.shape[0] * data.shape[1]
    if total > _PLOT_MAX_PX:
        factor = math.ceil(math.sqrt(total / _PLOT_MAX_PX))
        data = data[::factor, ::factor]
    return data, extent


def map_background(
    ax,
    bbox_poly: Polygon,
    osm_land_path: str,
    river_basins_path: str | None = None,
    margin_frac: float = 0.3,
) -> None:
    """
    Draw OSM land polygons, optional river basin outlines, and domain bbox on ax.

    OSM land and river basins are both loaded with a spatial filter to avoid
    reading the full global files.  The margin around the domain is proportional
    to the larger of the two bbox dimensions.

    Args:
        ax:                 Matplotlib Axes to draw on.
        bbox_poly:          Shapely Polygon of the domain bbox in WGS84.
        osm_land_path:      Path to the OSM land polygons shapefile.
        river_basins_path:  Optional path to a river basins shapefile.  When
                            provided, basin outlines are drawn over the land layer
                            with a transparent fill and dark-grey edge.
        margin_frac:        Fraction of the bbox span added as margin on each side.
    """
    lon_min, lat_min, lon_max, lat_max = bbox_poly.bounds
    margin = max(lon_max - lon_min, lat_max - lat_min) * margin_frac
    xmin, ymin = lon_min - margin, lat_min - margin
    xmax, ymax = lon_max + margin, lat_max + margin

    land = gpd.read_file(osm_land_path, bbox=(xmin, ymin, xmax, ymax), engine="pyogrio")
    if not land.empty:
        land.plot(ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=1)

    if river_basins_path is not None:
        basins = gpd.read_file(river_basins_path, bbox=(xmin, ymin, xmax, ymax))
        if not basins.empty:
            basins.plot(
                ax=ax,
                facecolor="none",
                edgecolor="#888888",
                linewidth=0.6,
                zorder=2,
            )

    bx, by = bbox_poly.exterior.xy
    ax.plot(bx, by, color="black", linewidth=1.5, zorder=10)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.grid(True, alpha=0.3, linewidth=0.5)


def _save(fig, output_path: str) -> None:
    """Save figure, create parent directory, and close."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Written: {output_path}")


# ── rule 03 plots ─────────────────────────────────────────────────────────────


def plot_topography(
    topo_path: str,
    bbox_poly: Polygon,
    osm_land_path: str,
    output_path: str,
) -> None:
    """
    Topography map: terrain colormap clipped to [_VMIN_ELEV, _VMAX_ELEV] m,
    with pixels above the upper threshold highlighted in a distinct colour.
    """
    data, extent = read_raster_for_plot(topo_path)
    thresh_mask = data >= _VMAX_ELEV
    data_under = np.where(~thresh_mask, data, np.nan)
    data_thresh = np.where(thresh_mask, 1.0, np.nan)

    fig, ax = plt.subplots(figsize=(9, 7))
    map_background(ax, bbox_poly, osm_land_path)
    im = ax.imshow(
        data_under,
        cmap=plt.get_cmap("terrain"),
        vmin=_VMIN_ELEV,
        vmax=_VMAX_ELEV,
        extent=extent,
        origin="upper",
        zorder=2,
    )
    if not np.all(np.isnan(data_thresh)):
        ax.imshow(
            data_thresh,
            cmap=mcolors.ListedColormap([_THRESH_COLOR]),
            vmin=0,
            vmax=2,
            extent=extent,
            origin="upper",
            zorder=3,
        )
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cb.set_label("Elevation (m)")
    ax.legend(
        handles=[
            Patch(color=_THRESH_COLOR, label=f"≥ {_VMAX_ELEV:.0f} m (DEM threshold)")
        ],
        loc="lower right",
        framealpha=0.9,
    )
    ax.set_title("Topography (DiluviumDEM)")
    _save(fig, output_path)


def plot_bathymetry(
    bathy_path: str,
    bbox_poly: Polygon,
    osm_land_path: str,
    output_path: str,
) -> None:
    """
    Bathymetry map: terrain colormap from _VMIN_ELEV to _VMAX_ELEV m (GEBCO).
    """
    data, extent = read_raster_for_plot(bathy_path)

    fig, ax = plt.subplots(figsize=(9, 7))
    map_background(ax, bbox_poly, osm_land_path)
    im = ax.imshow(
        data,
        cmap=plt.get_cmap("terrain"),
        vmin=_VMIN_ELEV,
        vmax=_VMAX_ELEV,
        extent=extent,
        origin="upper",
        zorder=2,
    )
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, extend="both")
    cb.set_label("Elevation / depth (m)")
    ax.set_title("Bathymetry (GEBCO)")
    _save(fig, output_path)


def plot_landuse(
    landuse_path: str,
    bbox_poly: Polygon,
    osm_land_path: str,
    output_path: str,
) -> None:
    """
    Categorical land-use map with Copernicus LC100 class colours and legend.
    """
    data, extent = read_raster_for_plot(landuse_path)
    present = sorted({int(v) for v in np.unique(data[~np.isnan(data)])})

    colors = [_LC_COLORS.get(c, "#808080") for c in present]
    code_to_idx = {c: i for i, c in enumerate(present)}
    data_idx = np.full(data.shape, np.nan)
    for c, i in code_to_idx.items():
        data_idx[data == c] = i

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(
        boundaries=range(len(present) + 1), ncolors=len(present)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    map_background(ax, bbox_poly, osm_land_path)
    ax.imshow(data_idx, cmap=cmap, norm=norm, extent=extent, origin="upper", zorder=2)
    legend_patches = [
        Patch(color=_LC_COLORS.get(c, "#808080"), label=_LC_NAMES.get(c, str(c)))
        for c in present
    ]
    ax.legend(
        handles=legend_patches, loc="lower right", fontsize=7, framealpha=0.9, ncol=2
    )
    ax.set_title("Land use (Copernicus LC100, 2019)")
    _save(fig, output_path)


def plot_roughness(
    roughness_path: str,
    bbox_poly: Polygon,
    osm_land_path: str,
    output_path: str,
) -> None:
    """
    Manning's n roughness map: viridis discrete colormap with labelled colorbar.
    """
    data, extent = read_raster_for_plot(roughness_path)
    unique_n = sorted({round(float(v), 6) for v in np.unique(data[~np.isnan(data)])})

    palette = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(unique_n)))
    cmap = mcolors.ListedColormap(palette)
    norm = mcolors.BoundaryNorm(
        boundaries=range(len(unique_n) + 1), ncolors=len(unique_n)
    )
    data_idx = np.full(data.shape, np.nan)
    for i, n_val in enumerate(unique_n):
        data_idx[np.isclose(data, n_val, atol=1e-5)] = i

    fig, ax = plt.subplots(figsize=(9, 7))
    map_background(ax, bbox_poly, osm_land_path)
    im = ax.imshow(
        data_idx, cmap=cmap, norm=norm, extent=extent, origin="upper", zorder=2
    )
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cb.set_ticks([i + 0.5 for i in range(len(unique_n))])
    cb.set_ticklabels([f"{n:.4f}" for n in unique_n])
    cb.set_label("Manning's n")
    ax.set_title("Surface roughness (Manning's n)")
    _save(fig, output_path)


def plot_elevation_merged(
    merged_path: str,
    bbox_poly: Polygon,
    osm_land_path: str,
    output_path: str,
    title_str: str,
    clip_elevation_m: float,
) -> None:
    """
    Merged DiluviumDEM+GEBCO elevation map, terrain colormap clipped to
    [-clip_elevation_m, clip_elevation_m] m, with pixels at or above
    clip_elevation_m highlighted in a distinct colour.

    Anything at or above clip_elevation_m is not reliable elevation data —
    it is either land-DEM input that was clamped at that threshold (03a
    clips topo_utm at clip_elevation_m) or the impassable-barrier fill value
    used to plug land-DEM gaps — so it is flagged outright rather than drawn
    with the terrain colormap.

    The raster (on the model's metric UTM grid) is reprojected to EPSG:4326
    for display so this map shares the same lon/lat reference frame as every
    other diagnostic plot.
    """
    data, extent = read_raster_reprojected_for_plot(
        merged_path, dst_bounds=bbox_poly.bounds
    )
    thresh_mask = data >= clip_elevation_m
    data_under = np.where(~thresh_mask, data, np.nan)
    data_thresh = np.where(thresh_mask, 1.0, np.nan)

    fig, ax = plt.subplots(figsize=(9, 7))
    map_background(ax, bbox_poly, osm_land_path)
    im = ax.imshow(
        data_under,
        cmap=plt.get_cmap("terrain"),
        vmin=-clip_elevation_m,
        vmax=clip_elevation_m,
        extent=extent,
        origin="upper",
        zorder=2,
    )
    if not np.all(np.isnan(data_thresh)):
        ax.imshow(
            data_thresh,
            cmap=mcolors.ListedColormap([_THRESH_COLOR]),
            vmin=0,
            vmax=2,
            extent=extent,
            origin="upper",
            zorder=3,
        )

    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, extend="min")
    cb.set_label("Elevation (m)")
    ax.legend(
        handles=[
            Patch(
                color=_THRESH_COLOR,
                label=f"≥ {clip_elevation_m:.0f} m (no elevation data / barrier)",
            )
        ],
        loc="lower right",
        framealpha=0.9,
    )
    ax.set_title(title_str)
    _save(fig, output_path)


def plot_zsini(
    zsini_path: str,
    bbox_poly: Polygon,
    osm_land_path: str,
    output_path: str,
) -> None:
    """
    Initial water-level mask (zsini): cells seeded with an initial water level
    of 0 m (open water at model start, written as 0.0 in 03a_get_elevation.py)
    vs land / outside-domain cells (nodata — dry at model start).

    The raster (on the model's metric UTM grid) is reprojected to EPSG:4326
    for display so this map shares the same lon/lat reference frame as every
    other diagnostic plot.
    """
    data, extent = read_raster_reprojected_for_plot(
        zsini_path, dst_bounds=bbox_poly.bounds
    )
    water_mask = ~np.isnan(data)
    n_water = int(water_mask.sum())

    fig, ax = plt.subplots(figsize=(9, 7))
    map_background(ax, bbox_poly, osm_land_path)
    ax.imshow(
        np.where(water_mask, 1.0, np.nan),
        cmap=mcolors.ListedColormap(["#3860D0"]),
        vmin=0,
        vmax=2,
        extent=extent,
        origin="upper",
        zorder=2,
    )

    ax.legend(
        handles=[
            Patch(
                color="#3860D0", label=f"Initial water (zsini = 0 m) — {n_water:,} px"
            ),
        ],
        loc="lower right",
        framealpha=0.9,
    )
    ax.set_title("Initial water level (zsini)")
    _save(fig, output_path)


def plot_elevation_blending(
    blend_weight: np.ndarray,
    merged: np.ndarray,
    gebco_original: np.ndarray,
    land_mask: np.ndarray,
    utm_crs_str: str,
    transform,
    osm_land_path: str,
    output_path: str,
) -> None:
    """
    Two-subplot diagnostic for the DiluviumDEM / GEBCO merging.

    Left subplot  — Blend weight (1 = DiluviumDEM, 0 = GEBCO).
    Right subplot — Elevation change vs original GEBCO (ocean pixels only):
                    diff = merged − GEBCO_original, with OSM land as background.

    The arrays live on the domain's metric UTM working grid (``transform``,
    ``utm_crs_str``); both are reprojected to EPSG:4326 for display so this
    map shares the same lon/lat reference frame as every other diagnostic plot.
    """
    from rasterio.crs import CRS

    utm_crs = CRS.from_string(utm_crs_str)
    diff = np.where(~land_mask, merged - gebco_original, np.nan)

    bw_wgs, extent = _to_wgs84_grid(blend_weight, transform, utm_crs)
    diff_wgs, _ = _to_wgs84_grid(diff, transform, utm_crs)

    # Downsample large arrays for plotting
    total = bw_wgs.shape[0] * bw_wgs.shape[1]
    factor = max(1, int(np.ceil(np.sqrt(total / _PLOT_MAX_PX))))
    bw_ds = bw_wgs[::factor, ::factor]
    diff_ds = diff_wgs[::factor, ::factor]

    left, right, bottom, top = extent
    land = gpd.read_file(osm_land_path, bbox=(left, bottom, right, top))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Subplot 1: Blend weight ───────────────────────────────────────────────
    ax = axes[0]
    if not land.empty:
        land.plot(ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=1)
    im1 = ax.imshow(
        bw_ds,
        cmap="RdYlBu_r",
        vmin=0,
        vmax=1,
        extent=extent,
        origin="upper",
        zorder=2,
    )
    cb1 = plt.colorbar(im1, ax=ax, fraction=0.03, pad=0.04)
    cb1.set_label("Blend weight")
    cb1.set_ticks([0, 0.5, 1])
    cb1.set_ticklabels(["0 (GEBCO)", "0.5", "1 (DiluviumDEM)"])
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Blend weight (DiluviumDEM → GEBCO transition)")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # ── Subplot 2: Elevation change vs GEBCO (ocean only) ────────────────────
    ax = axes[1]
    if not land.empty:
        land.plot(ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=1)
    vmax = (
        float(np.nanpercentile(np.abs(diff_ds[~np.isnan(diff_ds)]), 95))
        if not np.all(np.isnan(diff_ds))
        else 1.0
    )
    im2 = ax.imshow(
        diff_ds,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=extent,
        origin="upper",
        zorder=2,
    )
    cb2 = plt.colorbar(im2, ax=ax, fraction=0.03, pad=0.04, extend="both")
    cb2.set_label("Δ elevation (m)")
    ax.set_xlabel("Longitude (°)")
    ax.set_title(
        "Merged − GEBCO (ocean only)\npositive = DiluviumDEM raised bathymetry"
    )
    ax.grid(True, alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    _save(fig, output_path)


def plot_river_network(
    river_path: str,
    bbox_poly: Polygon,
    osm_land_path: str,
    output_path: str,
) -> None:
    """
    Clipped river network overlaid on OSM land background and domain bbox.
    """
    rivers = gpd.read_file(river_path)
    if rivers.crs and rivers.crs.to_epsg() != 4326:
        rivers = rivers.to_crs("EPSG:4326")

    fig, ax = plt.subplots(figsize=(9, 7))
    map_background(ax, bbox_poly, osm_land_path)
    if not rivers.empty:
        rivers.plot(ax=ax, color="steelblue", linewidth=0.8, zorder=5)
    ax.set_title("River network (clipped to model domain)")
    _save(fig, output_path)


# ── rule 04 plots ─────────────────────────────────────────────────────────────


def plot_domain_map(
    bbox_poly: Polygon,
    river_gdf: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    crossings: gpd.GeoDataFrame,
    has_glofas: np.ndarray,
    osm_land_path: str,
    output_path: str,
) -> None:
    """
    Map of the model domain, river network, surge stations, and river crossings.

    Crossing points are coloured green when matched to a GloFAS cell (active)
    and red when not matched (inactive).
    """
    lon_min, lat_min, lon_max, lat_max = bbox_poly.bounds
    margin = max(lon_max - lon_min, lat_max - lat_min) * 0.3
    map_bounds = (
        lon_min - margin,
        lat_min - margin,
        lon_max + margin,
        lat_max + margin,
    )

    land = gpd.read_file(osm_land_path, bbox=map_bounds)

    fig, ax = plt.subplots(figsize=(10, 8))
    if not land.empty:
        land.plot(ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=1)
    river_gdf.plot(ax=ax, color="steelblue", linewidth=0.8, zorder=2)

    bx, by = bbox_poly.exterior.xy
    ax.plot(bx, by, color="black", linewidth=2, zorder=3)
    stations.plot(
        ax=ax, color="darkorange", markersize=35, marker="^", zorder=4, alpha=0.8
    )

    # Exclusion-reason masks — derived from filter columns written by rule 04.
    # Falls back to a single inactive category when the columns are absent.
    n = len(crossings)
    active = has_glofas.astype(bool) if n > 0 else np.zeros(0, dtype=bool)
    if (
        n > 0
        and "enters_domain" in crossings.columns
        and "visible_on_grid" in crossings.columns
    ):
        enters = crossings["enters_domain"].fillna(False).astype(bool).values
        visible = crossings["visible_on_grid"].fillna(False).astype(bool).values
        mask_active = active
        mask_no_glofas = (
            ~active & enters & visible
        )  # wide enough, enters, but no GloFAS/EVA
        mask_too_narrow = (
            ~active & enters & ~visible
        )  # enters domain but below grid resolution
        mask_no_entry = ~active & ~enters  # no downstream reach found inside domain
    else:
        mask_active = active
        mask_no_glofas = ~active
        mask_too_narrow = np.zeros(n, dtype=bool)
        mask_no_entry = np.zeros(n, dtype=bool)

    _CROSS_KW: dict[str, Any] = dict(markersize=60, marker="x", zorder=5, linewidths=2)
    if mask_active.any():
        crossings[mask_active].plot(ax=ax, color="limegreen", markersize=60, zorder=5)
    if mask_no_glofas.any():
        crossings[mask_no_glofas].plot(ax=ax, color="#e41a1c", **_CROSS_KW)
    if mask_too_narrow.any():
        crossings[mask_too_narrow].plot(ax=ax, color="#984ea3", **_CROSS_KW)
    if mask_no_entry.any():
        crossings[mask_no_entry].plot(ax=ax, color="#ff7f00", **_CROSS_KW)

    ax.set_xlim(map_bounds[0], map_bounds[2])
    ax.set_ylim(map_bounds[1], map_bounds[3])
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Boundary forcing locations")
    ax.grid(True, alpha=0.3, linewidth=0.5)

    legend_handles = [
        Line2D([0], [0], color="black", linewidth=2, label="Model domain"),
        Line2D([0], [0], color="steelblue", linewidth=1.5, label="River network"),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="darkorange",
            markersize=9,
            label="Surge stations",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="limegreen",
            markersize=9,
            label="River crossing (active)",
        ),
    ]
    if mask_no_entry.any():
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="x",
                color="#ff7f00",
                markersize=9,
                markeredgewidth=2,
                label="Excluded: does not enter domain",
            )
        )
    if mask_too_narrow.any():
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="x",
                color="#984ea3",
                markersize=9,
                markeredgewidth=2,
                label="Excluded: too narrow for grid",
            )
        )
    if mask_no_glofas.any():
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="x",
                color="#e41a1c",
                markersize=9,
                markeredgewidth=2,
                label="Excluded: no GloFAS match / EVA failed",
            )
        )

    ax.legend(handles=legend_handles, loc="best", framealpha=0.9)
    _save(fig, output_path)


def plot_forcing_timeseries(
    surge_ds: xr.Dataset,
    river_ds: xr.Dataset,
    return_period: int,
    output_path: str,
) -> None:
    """
    Two-panel timeseries: surge water level (left) and river discharge (right).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    t_surge = surge_ds["time"].values
    wl = surge_ds["water_level"].values
    for i in range(wl.shape[0]):
        ax1.plot(t_surge, wl[i], color="steelblue", alpha=0.5, linewidth=0.8)
    ax1.set_xlabel("Time (hours since start)")
    ax1.set_ylabel("Water level (m)")
    ax1.set_title(f"Surge forcing — RP{return_period} ({wl.shape[0]} stations)")
    ax1.grid(True, alpha=0.3)

    t_river = river_ds["time"].values
    q = river_ds["discharge"].values
    active = river_ds["has_glofas"].values.astype(bool)
    for i in range(q.shape[0]):
        if active[i]:
            ax2.plot(t_river, q[i], color="teal", alpha=0.6, linewidth=0.8)
    ax2.set_xlabel("Time (hours since start)")
    ax2.set_ylabel("Discharge (m³ s⁻¹)")
    ax2.set_title(
        f"River discharge forcing ({int(active.sum())}/{len(active)} crossings with GloFAS)"
    )
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, output_path)


# ── rule 05 plots ─────────────────────────────────────────────────────────────


def plot_cleaned_network(
    rivers_orig: gpd.GeoDataFrame,
    rivers_clean: gpd.GeoDataFrame,
    bbox_poly: Polygon,
    osm_land_path: str,
    river_basins: str | None,
    output_path: str,
) -> None:
    """
    Map of kept (blue) vs removed (salmon) reaches after connectivity cleaning.
    """
    lon_min, lat_min, lon_max, lat_max = bbox_poly.bounds
    margin = max(lon_max - lon_min, lat_max - lat_min) * 0.3
    map_bounds = (
        lon_min - margin,
        lat_min - margin,
        lon_max + margin,
        lat_max + margin,
    )

    land = gpd.read_file(osm_land_path, bbox=map_bounds)
    kept_ids = set(rivers_clean["reach_id"].astype(str))

    orig_wgs = (
        rivers_orig.to_crs("EPSG:4326")
        if rivers_orig.crs.to_epsg() != 4326
        else rivers_orig
    )
    removed = orig_wgs[~orig_wgs["reach_id"].astype(str).isin(kept_ids)]
    kept = orig_wgs[orig_wgs["reach_id"].astype(str).isin(kept_ids)]

    fig, ax = plt.subplots(figsize=(9, 7))
    if not land.empty:
        land.plot(ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=1)
    # if river_basins is not None:
    #     basins = gpd.read_file(river_basins, bbox=(lon_min, lat_min, lon_max, lat_max))
    #     if not basins.empty:
    #         basins.plot(
    #             ax=ax, facecolor="none", edgecolor="#888888", linewidth=0.6, zorder=2,
    #         )
    if not removed.empty:
        removed.plot(ax=ax, color="salmon", linewidth=0.5, alpha=0.7, zorder=3)
    if not kept.empty:
        kept.plot(ax=ax, color="steelblue", linewidth=0.8, zorder=4)

    bx, by = bbox_poly.exterior.xy
    ax.plot(bx, by, color="black", linewidth=1.5, zorder=5)
    ax.set_xlim(map_bounds[0], map_bounds[2])
    ax.set_ylim(map_bounds[1], map_bounds[3])
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(f"River network cleaning: {len(kept)}/{len(orig_wgs)} reaches kept")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(
        handles=[
            Line2D(
                [0], [0], color="steelblue", linewidth=2, label=f"Kept ({len(kept)})"
            ),
            Line2D(
                [0], [0], color="salmon", linewidth=2, label=f"Removed ({len(removed)})"
            ),
        ],
        loc="best",
        framealpha=0.9,
    )
    _save(fig, output_path)


def plot_clean_network_discharge(
    rivers_wgs: gpd.GeoDataFrame,
    bbox_poly: Polygon,
    osm_land_path: str,
    output_path: str,
) -> None:
    """
    Clean river network map where both line thickness and colour encode
    accumulated bankfull discharge (log scale).  Reaches with no accumulated
    discharge are shown in light grey; active reaches use four quartile bins
    to set line width and a log-scaled Blues colormap for colour.
    """
    lon_min, lat_min, lon_max, lat_max = bbox_poly.bounds
    margin = max(lon_max - lon_min, lat_max - lat_min) * 0.3
    map_bounds = (
        lon_min - margin,
        lat_min - margin,
        lon_max + margin,
        lat_max + margin,
    )
    land = gpd.read_file(osm_land_path, bbox=map_bounds)

    col = "bankfull_discharge_acc"
    active = (
        rivers_wgs[rivers_wgs[col] > 0].copy()
        if col in rivers_wgs.columns
        else gpd.GeoDataFrame()
    )
    inactive = (
        rivers_wgs[rivers_wgs[col] <= 0].copy()
        if col in rivers_wgs.columns
        else rivers_wgs.copy()
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    if not land.empty:
        land.plot(ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=1)

    # Reaches with no discharge — thin grey background
    if not inactive.empty:
        inactive.plot(ax=ax, color="#cccccc", linewidth=0.3, zorder=2)

    if active.empty:
        ax.text(
            0.5,
            0.5,
            "No reaches with accumulated discharge",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="grey",
        )
    else:
        q = active[col]
        q25, q50, q75 = (
            float(q.quantile(0.25)),
            float(q.quantile(0.5)),
            float(q.quantile(0.75)),
        )
        log_norm = mcolors.LogNorm(vmin=max(float(q.min()), 1e-3), vmax=float(q.max()))
        cmap = plt.get_cmap("Blues")

        lw_bins = [
            (active[q <= q25], 0.5, f"≤ {q25:.1f}"),
            (active[(q > q25) & (q <= q50)], 1.1, f"{q25:.1f}–{q50:.1f}"),
            (active[(q > q50) & (q <= q75)], 2.0, f"{q50:.1f}–{q75:.1f}"),
            (active[q > q75], 3.2, f"> {q75:.1f}"),
        ]
        for gdf_bin, lw, _ in lw_bins:
            if not gdf_bin.empty:
                gdf_bin.plot(
                    ax=ax,
                    column=col,
                    cmap=cmap,
                    norm=log_norm,
                    linewidth=lw,
                    zorder=3,
                    legend=False,
                )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=log_norm)
        sm.set_array([])
        fig.colorbar(
            sm,
            ax=ax,
            shrink=0.55,
            pad=0.02,
            label="Accumulated bankfull discharge (m³ s⁻¹)",
        )

        width_handles = [
            Line2D([0], [0], color="steelblue", linewidth=lw, label=f"{label} m³/s")
            for _, lw, label in lw_bins
        ]
        ax.add_artist(
            ax.legend(
                handles=width_handles,
                title="Discharge (m³/s)",
                loc="lower left",
                framealpha=0.9,
                fontsize=8,
            )
        )

    bx, by = bbox_poly.exterior.xy
    ax.plot(bx, by, color="black", linewidth=1.5, zorder=5)
    ax.set_xlim(map_bounds[0], map_bounds[2])
    ax.set_ylim(map_bounds[1], map_bounds[3])
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(
        f"Clean river network — discharge ({len(active)}/{len(rivers_wgs)} active reaches)"
    )
    ax.grid(True, alpha=0.3, linewidth=0.5)
    _save(fig, output_path)


# ── rule 06 plots ─────────────────────────────────────────────────────────────


def plot_river_depth(
    rivers_wgs: gpd.GeoDataFrame,
    bbox_poly: Polygon,
    osm_land_path: str,
    output_path: str,
) -> None:
    """
    River network coloured by computed hydraulic depth (Blues colormap).
    """
    lon_min, lat_min, lon_max, lat_max = bbox_poly.bounds
    margin = max(lon_max - lon_min, lat_max - lat_min) * 0.3
    map_bounds = (
        lon_min - margin,
        lat_min - margin,
        lon_max + margin,
        lat_max + margin,
    )

    land = gpd.read_file(osm_land_path, bbox=map_bounds)

    fig, ax = plt.subplots(figsize=(9, 7))
    if not land.empty:
        land.plot(ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=1)
    if not rivers_wgs.empty and "rivdph" in rivers_wgs.columns:
        rivers_wgs.plot(
            ax=ax,
            column="rivdph",
            cmap="Blues",
            linewidth=0.8,
            zorder=3,
            legend=True,
            legend_kwds={"label": "Hydraulic depth (m)", "shrink": 0.6},
        )

    bx, by = bbox_poly.exterior.xy
    ax.plot(bx, by, color="black", linewidth=1.5, zorder=4)
    ax.set_xlim(map_bounds[0], map_bounds[2])
    ax.set_ylim(map_bounds[1], map_bounds[3])
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("River network — hydraulic depth")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    _save(fig, output_path)


def plot_hydraulic_relations(
    rivers_wgs: gpd.GeoDataFrame,
    output_path: str,
) -> None:
    """
    Two log-log scatter plots of channel width vs hydraulic depth.

    Left panel:  coloured by accumulated discharge (log₁₀, plasma colormap).
    Right panel: coloured by distance from outlet (dist_out, viridis colormap).
                 Omitted if 'dist_out' is not present in the GeoDataFrame.
    """
    cols = ["width", "bankfull_discharge_acc", "rivdph"]
    has_dist = "dist_out" in rivers_wgs.columns
    if has_dist:
        cols.append("dist_out")

    df = (
        rivers_wgs[cols]
        .copy()
        .dropna(subset=["width", "bankfull_discharge_acc", "rivdph"])
    )
    df = df[(df["width"] > 0) & (df["bankfull_discharge_acc"] > 0) & (df["rivdph"] > 0)]

    n_panels = 2 if has_dist else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    def _log_scatter(ax, color_col, cmap, cbar_label, title):
        if df.empty:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="grey",
            )
            ax.set_title(title + " — no data")
            return
        c_vals = df[color_col] if color_col == "dist_out" else np.log10(df[color_col])
        sc = ax.scatter(
            df["width"],
            df["rivdph"],
            c=c_vals,
            cmap=cmap,
            alpha=0.5,
            s=7,
            rasterized=True,
        )
        fig.colorbar(sc, ax=ax, label=cbar_label, shrink=0.85)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Channel width (m)")
        ax.set_ylabel("Hydraulic depth (m)")
        ax.set_title(f"{title}  ({len(df)} reaches)")
        ax.grid(True, which="both", alpha=0.3, linewidth=0.5)

    _log_scatter(
        axes[0],
        "bankfull_discharge_acc",
        "plasma",
        "log₁₀ discharge (m³ s⁻¹)",
        "Width vs depth — coloured by discharge",
    )
    if has_dist:
        _log_scatter(
            axes[1],
            "dist_out",
            "viridis_r",
            "Distance from outlet (m)",
            "Width vs depth — coloured by dist_out",
        )

    fig.tight_layout()
    _save(fig, output_path)


def plot_river_network_width_discharge(
    rivers_wgs: gpd.GeoDataFrame,
    bbox_poly: Polygon,
    osm_land_path: str,
    output_path: str,
    seed_reach_ids: set[str] | None = None,
) -> None:
    """
    River network map with line thickness proportional to channel width (four
    quartile bins).  Reaches are coloured by hydraulic depth.  Boundary entry
    reaches are shown as circle markers whose area scales with discharge.
    """
    lon_min, lat_min, lon_max, lat_max = bbox_poly.bounds
    margin = max(lon_max - lon_min, lat_max - lat_min) * 0.3
    map_bounds = (
        lon_min - margin,
        lat_min - margin,
        lon_max + margin,
        lat_max + margin,
    )

    land = gpd.read_file(osm_land_path, bbox=map_bounds)

    # Shared depth colormap across all width bins
    depth_col = "rivdph"
    vmin = float(rivers_wgs[depth_col].min())
    vmax = float(rivers_wgs[depth_col].max())
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("Blues")

    # Width quartile bins → linewidth mapping
    w = rivers_wgs["width"].fillna(1.0)
    q25, q50, q75 = (
        float(w.quantile(0.25)),
        float(w.quantile(0.5)),
        float(w.quantile(0.75)),
    )
    lw_bins = [
        (rivers_wgs[w <= q25], 0.4, f"≤ {q25:.0f} m"),
        (rivers_wgs[(w > q25) & (w <= q50)], 0.9, f"{q25:.0f}–{q50:.0f} m"),
        (rivers_wgs[(w > q50) & (w <= q75)], 1.7, f"{q50:.0f}–{q75:.0f} m"),
        (rivers_wgs[w > q75], 2.8, f"> {q75:.0f} m"),
    ]

    fig, ax = plt.subplots(figsize=(9, 7))
    if not land.empty:
        land.plot(ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=1)

    for gdf_bin, lw, _ in lw_bins:
        if not gdf_bin.empty:
            gdf_bin.plot(
                ax=ax,
                column=depth_col,
                cmap=cmap,
                norm=norm,
                linewidth=lw,
                zorder=3,
                legend=False,
            )

    # Manual depth colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02, label="Hydraulic depth (m)")

    # Width legend (placed first so marker legend can be appended via ax.legend())
    width_handles = [
        Line2D([0], [0], color="steelblue", linewidth=lw, label=label)
        for _, lw, label in lw_bins
    ]
    ax.add_artist(
        ax.legend(
            handles=width_handles,
            title="Channel width",
            loc="lower left",
            framealpha=0.9,
            fontsize=8,
        )
    )

    # Boundary entry markers — circle area proportional to discharge
    if seed_reach_ids is not None and "bankfull_discharge_acc" in rivers_wgs.columns:
        seed_gdf = rivers_wgs[
            rivers_wgs["reach_id"].astype(str).isin(seed_reach_ids)
        ].copy()
        if not seed_gdf.empty:
            # Centroids must be computed in a projected CRS, then reprojected
            # back to WGS84 for plotting on `rivers_wgs` axes — geopandas warns
            # that geographic-CRS centroids are inaccurate.
            pts = seed_gdf.geometry.to_crs(pick_utm_crs(seed_gdf)).centroid.to_crs(
                "EPSG:4326"
            )
            q_vals = seed_gdf["bankfull_discharge_acc"].fillna(0).to_numpy(dtype=float)
            q_max = float(q_vals.max()) if q_vals.max() > 0 else 1.0
            marker_sizes = 30 + 250 * (q_vals / q_max)
            ax.scatter(
                pts.x,
                pts.y,
                s=marker_sizes,
                c="darkorange",
                zorder=9,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5,
            )
            # Size-scaled legend entries
            for frac in [0.2, 0.6, 1.0]:
                ax.scatter(
                    [],
                    [],
                    s=30 + 250 * frac,
                    c="darkorange",
                    alpha=0.85,
                    edgecolors="white",
                    linewidths=0.5,
                    label=f"Q = {frac * q_max:.0f} m³/s",
                )

    # Discharge marker legend (only when seed markers were drawn)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles=handles,
            title="Entry discharge",
            loc="lower right",
            framealpha=0.9,
            fontsize=8,
        )

    bx, by = bbox_poly.exterior.xy
    ax.plot(bx, by, color="black", linewidth=1.5, zorder=5)
    ax.set_xlim(map_bounds[0], map_bounds[2])
    ax.set_ylim(map_bounds[1], map_bounds[3])
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("River network — width (line thickness) & discharge (markers)")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    _save(fig, output_path)


# ── output analysis plots (postprocessing of scenario runs) ──────────────────


def _overlay_layers(
    crs,
    bounds: tuple[float, float, float, float],
    domain_poly: Polygon,
    land_polygons_path: str,
    river_network_path: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load land outline, river network, and domain polygon, reprojected to ``crs``.

    Land polygons and the river network are stored in WGS84; ``bounds`` (the
    raster bounds in ``crs``) is reprojected to WGS84 to spatially filter both
    reads before reprojecting them to the raster CRS so they align with the
    metric axes.

    Args:
        crs:                Target CRS — pass the *reprojected* (EPSG:4326)
                            raster CRS so overlays land in the same lon/lat
                            frame as the displayed data.
        bounds:             Raster bounds (left, bottom, right, top) in ``crs``.
        domain_poly:        Domain polygon in WGS84 (from ``load_domain``).
        land_polygons_path: Path to the OSM land polygons geopackage.
        river_network_path: Path to the clipped river network geopackage.

    Returns:
        (land, rivers, domain_gdf) — GeoDataFrames reprojected to ``crs``.
    """
    from rasterio.warp import transform_bounds

    wgs84_bounds = transform_bounds(crs, "EPSG:4326", *bounds)

    land = gpd.read_file(land_polygons_path, bbox=wgs84_bounds)
    if not land.empty:
        land = land.to_crs(crs)

    # River network may be stored in UTM — bbox filter with WGS84 bounds would
    # return empty; load fully and reproject instead.
    rivers = gpd.read_file(river_network_path)
    if not rivers.empty and rivers.crs is not None and rivers.crs != crs:
        rivers = rivers.to_crs(crs)

    domain_gdf = gpd.GeoDataFrame(geometry=[domain_poly], crs="EPSG:4326").to_crs(crs)

    return land, rivers, domain_gdf


def _draw_overlays(
    ax,
    land: gpd.GeoDataFrame,
    rivers: gpd.GeoDataFrame,
    domain_gdf: gpd.GeoDataFrame,
    zorder: int,
) -> None:
    """Draw filled land background, river network, and domain boundary on ``ax``."""
    if not land.empty:
        land.plot(
            ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=zorder
        )
    if not rivers.empty:
        rivers.plot(ax=ax, color="steelblue", linewidth=0.6, zorder=zorder + 1)
    domain_gdf.boundary.plot(ax=ax, color="black", linewidth=1.5, zorder=zorder + 2)


_OVERLAY_LEGEND_HANDLES = [
    Line2D([0], [0], color="black", linewidth=1.5, label="Model domain"),
    Patch(color="#d9d9d9", edgecolor="#aaaaaa", label="Land"),
    Line2D([0], [0], color="steelblue", linewidth=1.5, label="River network"),
]


def plot_max_inundation_map(
    da_hmax: xr.DataArray,
    domain_poly: Polygon,
    land_polygons_path: str,
    river_network_path: str,
    output_path: str,
    basin_id: str = "",
    run_label: str = "",
) -> None:
    """
    Max inundation depth map for a SFINCS run (output of
    ``postprocessing.compute_max_inundation``), with the land outline, river
    network, and domain boundary overlaid.

    Args:
        da_hmax:            Max inundation depth DataArray; NaN = dry / outside
                            the land domain.  Must carry CRS metadata (``rio.crs``).
        domain_poly:        Domain polygon in WGS84 (from ``load_domain``).
        land_polygons_path: Path to the OSM land polygons geopackage.
        river_network_path: Path to the clipped river network geopackage.
        output_path:        Destination PNG path.
        basin_id:           Basin identifier for the plot title.
        run_label:          Optional run/scenario label for the plot title.
    """
    # Reproject from the model's metric UTM grid to EPSG:4326.  Use imshow with
    # explicit extent rather than da.plot() to guarantee north-up orientation.
    da_wgs = da_hmax.squeeze().rio.reproject("EPSG:4326")
    y_dim = da_wgs.rio.y_dim
    x_dim = da_wgs.rio.x_dim
    if da_wgs.dims[0] != y_dim:
        da_wgs = da_wgs.transpose(y_dim, x_dim)
    wgs_arr = da_wgs.values.astype(np.float32)
    ys = da_wgs.coords[y_dim].values
    if len(ys) >= 2 and float(ys[0]) < float(ys[-1]):
        wgs_arr = wgs_arr[::-1]

    land, rivers, domain_gdf = _overlay_layers(
        da_wgs.rio.crs,
        da_wgs.rio.bounds(),
        domain_poly,
        land_polygons_path,
        river_network_path,
    )
    wgs_left, wgs_bottom, wgs_right, wgs_top = da_wgs.rio.bounds()

    valid = wgs_arr[~np.isnan(wgs_arr)]
    vmax = float(np.percentile(valid, 99)) if len(valid) > 0 else 1.0
    vmax = max(vmax, 0.01)
    n_flooded = int(da_hmax.notnull().sum().item())

    lon_min, lat_min, lon_max, lat_max = domain_poly.bounds
    _margin = max(lon_max - lon_min, lat_max - lat_min) * 0.3

    fig, ax = plt.subplots(figsize=(10, 8))
    if not land.empty:
        land.plot(ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=1)
    im = ax.imshow(
        wgs_arr,
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        extent=(wgs_left, wgs_right, wgs_bottom, wgs_top),
        origin="upper",
        aspect="auto",
        zorder=2,
    )
    cbar = fig.colorbar(im, ax=ax, extend="max", fraction=0.03, pad=0.04)
    cbar.set_label("Max inundation depth (m)")
    if not rivers.empty:
        rivers.plot(ax=ax, color="steelblue", linewidth=0.6, zorder=3)
    domain_gdf.boundary.plot(ax=ax, color="black", linewidth=1.5, zorder=4)

    ax.set_xlim(lon_min - _margin, lon_max + _margin)
    ax.set_ylim(lat_min - _margin, lat_max + _margin)
    ax.set_aspect("equal")
    title = "Max inundation depth"
    if run_label:
        title = f"{title} — {run_label}"
    if basin_id:
        title = f"{title} | {basin_id}"
    ax.set_title(f"{title}\n{n_flooded:,} flooded land cells")
    ax.legend(
        handles=_OVERLAY_LEGEND_HANDLES, loc="lower right", framealpha=0.9, fontsize=8
    )
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    _save(fig, output_path)


def animate_flood_progression(
    da_h: xr.DataArray,
    domain_poly: Polygon,
    land_polygons_path: str,
    river_network_path: str,
    output_path: str,
    basin_id: str = "",
    run_label: str = "",
    fps: int = 4,
) -> None:
    """
    Animate the progression of land-surface flooding over a SFINCS run
    (output of ``postprocessing.compute_flood_progression``), saved as an MP4.

    The land outline, river network, and domain boundary are drawn once as a
    static background; an imshow layer of instantaneous water depth (already
    masked to non-water land-use areas) is then updated for each time step.

    Args:
        da_h:               Instantaneous land-surface water depth DataArray
                            with a ``time`` dimension; NaN = dry / water / outside
                            domain.  Must carry CRS metadata (``rio.crs``).
        domain_poly:        Domain polygon in WGS84 (from ``load_domain``).
        land_polygons_path: Path to the OSM land polygons geopackage.
        river_network_path: Path to the clipped river network geopackage.
        output_path:        Destination MP4 path.
        basin_id:           Basin identifier for the plot title.
        run_label:          Optional run/scenario label for the plot title.
        fps:                Frames per second of the output video.
    """
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    if "time" not in da_h.dims:
        log.warning(
            "animate_flood_progression: da_h has no 'time' dimension — skipping"
        )
        Path(output_path).touch()
        return

    # Reproject from the model's metric UTM grid to EPSG:4326 so this animation
    # shares the same lon/lat reference frame as every other diagnostic plot.
    da_wgs = da_h.rio.reproject("EPSG:4326")
    land, rivers, domain_gdf = _overlay_layers(
        da_wgs.rio.crs,
        da_wgs.rio.bounds(),
        domain_poly,
        land_polygons_path,
        river_network_path,
    )

    vals = da_wgs.values
    valid = vals[~np.isnan(vals)]
    vmax = float(np.percentile(valid, 99)) if len(valid) > 0 else 1.0
    vmax = max(vmax, 0.01)

    x, y = da_wgs["x"].values, da_wgs["y"].values
    extent = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
    origin = "upper" if y[0] > y[-1] else "lower"

    lon_min, lat_min, lon_max, lat_max = domain_poly.bounds
    _margin = max(lon_max - lon_min, lat_max - lat_min) * 0.3

    fig, ax = plt.subplots(figsize=(10, 8))
    if not land.empty:
        land.plot(ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=1)
    if not rivers.empty:
        rivers.plot(ax=ax, color="steelblue", linewidth=0.6, zorder=3)
    domain_gdf.boundary.plot(ax=ax, color="black", linewidth=1.5, zorder=4)

    im = ax.imshow(
        np.full(da_wgs.shape[1:], np.nan),
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        extent=extent,
        origin=origin,
        zorder=2,
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, extend="max")
    cb.set_label("Water depth (m)")
    ax.set_xlim(lon_min - _margin, lon_max + _margin)
    ax.set_ylim(lat_min - _margin, lat_max + _margin)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(
        handles=_OVERLAY_LEGEND_HANDLES, loc="lower right", framealpha=0.9, fontsize=8
    )

    title_prefix = "Flood progression"
    if run_label:
        title_prefix = f"{title_prefix} — {run_label}"
    if basin_id:
        title_prefix = f"{title_prefix} | {basin_id}"
    title_artist = ax.set_title(title_prefix)

    times = da_wgs["time"].values
    n_frames = da_wgs.sizes["time"]

    def _update(i):
        im.set_data(da_wgs.isel(time=i).values)
        title_artist.set_text(
            f"{title_prefix}\n{np.datetime_as_string(times[i], unit='m')}"
        )
        return im, title_artist

    anim = FuncAnimation(fig, _update, frames=n_frames, blit=False)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=FFMpegWriter(fps=fps))
    plt.close(fig)
    log.info(f"Written: {output_path}")


def plot_geoid_offset(
    offset_arr: np.ndarray,
    offset_transform,
    wgs84_bounds: tuple[float, float, float, float],
    osm_land_path: str,
    output_path: str,
) -> None:
    """
    Geoid offset N_EGM2008 − N_GOCO06s over the domain (WGS84 lon/lat).

    Shows the step-1 correction field before it is resampled to the DEM grid.
    The global array is clipped to the domain view via axis limits; land polygons
    are overlaid in grey so the ocean pattern stands out.
    """
    h, w = offset_arr.shape
    lon_left = offset_transform.c
    lat_top = offset_transform.f
    lon_right = lon_left + w * offset_transform.a
    lat_bottom = lat_top + h * offset_transform.e  # e is negative (south)
    extent = (lon_left, lon_right, lat_bottom, lat_top)

    lon_min, lat_min, lon_max, lat_max = wgs84_bounds
    land = gpd.read_file(osm_land_path, bbox=(lon_min, lat_min, lon_max, lat_max))

    total = h * w
    factor = max(1, int(math.ceil(math.sqrt(total / _PLOT_MAX_PX))))
    arr_ds = offset_arr[::factor, ::factor]

    valid = arr_ds[~np.isnan(arr_ds)]
    vmax = float(np.percentile(np.abs(valid), 99)) if len(valid) > 0 else 1.0
    vmax = max(vmax, 0.001)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(
        arr_ds,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=extent,
        origin="upper",
        zorder=1,
    )
    if not land.empty:
        land.plot(ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=2)
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, extend="both")
    cb.set_label("Geoid offset (m)  [N_EGM2008 − N_GOCO06s]")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Geoid offset: N_EGM2008 − N_GOCO06s (step 1 input)")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    _save(fig, output_path)


def plot_mdt_ocean(
    mdt_np: np.ndarray,
    mdt_transform,
    wgs84_bounds: tuple[float, float, float, float],
    osm_land_path: str,
    output_path: str,
) -> None:
    """
    MDT HYBRID-CNES-CLS22 over ocean (raw values, NaN over land, WGS84 lon/lat).

    Shows the step-2 subtraction field in its raw state, before inverse-distance
    extrapolation over land.  NaN land pixels are transparent so only the ocean
    signal is visible; the land polygon outline is drawn for geographic context.
    """
    h, w = mdt_np.shape
    lon_left = mdt_transform.c
    lat_top = mdt_transform.f
    lon_right = lon_left + w * mdt_transform.a
    lat_bottom = lat_top + h * mdt_transform.e
    extent = (lon_left, lon_right, lat_bottom, lat_top)

    lon_min, lat_min, lon_max, lat_max = wgs84_bounds
    land = gpd.read_file(osm_land_path, bbox=(lon_min, lat_min, lon_max, lat_max))

    total = h * w
    factor = max(1, int(math.ceil(math.sqrt(total / _PLOT_MAX_PX))))
    arr_ds = mdt_np[::factor, ::factor]

    valid = arr_ds[~np.isnan(arr_ds)]
    vmax = float(np.percentile(np.abs(valid), 99)) if len(valid) > 0 else 1.0
    vmax = max(vmax, 0.001)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(
        arr_ds,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=extent,
        origin="upper",
        zorder=1,
    )
    if not land.empty:
        land.plot(ax=ax, color="none", edgecolor="#555555", linewidth=0.5, zorder=2)
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, extend="both")
    cb.set_label("MDT (m above GOCO06s geoid)")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(
        "MDT HYBRID-CNES-CLS22 — ocean only (step 2 input, before land extrapolation)"
    )
    ax.grid(True, alpha=0.3, linewidth=0.5)
    _save(fig, output_path)


def plot_datum_correction_delta(
    delta: np.ndarray,
    utm_crs_str: str,
    wgs84_bounds: tuple[float, float, float, float],
    osm_land_path: str,
    output_path: str,
) -> None:
    """
    Diverging raster plot of Δ elevation = MSL_elevation − EGM2008_elevation.

    Shows the spatial pattern of the vertical datum correction applied to
    DiluviumDEM: geoid shift (N_EGM2008 − N_GOCO06s) + MDT subtraction.
    Only land pixels are shown (where DiluviumDEM had valid data); ocean is grey.
    Axes are in projected UTM metres so the pattern can be compared with the
    elevation and zsini maps.
    """
    from pyproj import Transformer

    lon_min, lat_min, lon_max, lat_max = wgs84_bounds
    h, w = delta.shape

    try:
        plot_crs = gpd.GeoSeries(
            gpd.points_from_xy([lon_min, lon_max], [lat_min, lat_max]),
            crs="EPSG:4326",
        ).estimate_utm_crs()
    except Exception:
        from pyproj import CRS

        plot_crs = CRS.from_user_input(utm_crs_str)

    _trans = Transformer.from_crs("EPSG:4326", plot_crs, always_xy=True)
    _west, _south = _trans.transform(lon_min, lat_min)
    _east, _north = _trans.transform(lon_max, lat_max)
    extent = (_west, _east, _south, _north)

    total = h * w
    factor = max(1, int(math.ceil(math.sqrt(total / _PLOT_MAX_PX))))
    delta_ds = delta[::factor, ::factor]

    land = gpd.read_file(osm_land_path, bbox=(lon_min, lat_min, lon_max, lat_max))
    valid = delta_ds[~np.isnan(delta_ds)]
    vmax = float(np.percentile(np.abs(valid), 95)) if len(valid) > 0 else 1.0
    vmax = max(vmax, 0.01)

    fig, ax = plt.subplots(figsize=(9, 7))
    if not land.empty:
        land.to_crs(plot_crs).plot(
            ax=ax,
            color="#d9d9d9",
            edgecolor="#aaaaaa",
            linewidth=0.3,
            zorder=1,
        )
    im = ax.imshow(
        delta_ds,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=extent,
        origin="upper",
        zorder=2,
    )
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, extend="both")
    cb.set_label("Δ elevation (m)  [MSL − EGM2008]")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(
        "Vertical datum correction — DiluviumDEM\n"
        "(EGM2008  →  GOCO06s geoid  →  local MSL via MDT_CNES-CLS22)"
    )
    ax.grid(True, alpha=0.3, linewidth=0.5)
    _save(fig, output_path)


# ── rule 09 plots ─────────────────────────────────────────────────────────────


def plot_max_inundation_depth(
    hmax: np.ndarray,
    extent: tuple[float, float, float, float],
    epsg: int | str,
    output_path: str,
    basin_id: str = "",
    osm_land_path: str | None = None,
) -> None:
    """
    Max inundation depth map from SFINCS spinup (Blues colormap, dry cells masked).

    Args:
        hmax:          2-D float array of max inundation depth (m); 0 = dry, NaN = outside domain.
        extent:        (xmin, xmax, ymin, ymax) in the model CRS (UTM metres).
        epsg:          EPSG code of the model CRS (from sfincs.inp) — the grid is
                       assumed north-up and is reprojected to EPSG:4326 for display
                       so this map shares the same lon/lat frame as every other
                       diagnostic plot.
        output_path:   Destination PNG path.
        basin_id:      Basin identifier for the plot title.
        osm_land_path: Optional path to OSM land polygons; when provided the land
                       background and domain outline are drawn via map_background,
                       giving the same consistent layout as all other diagnostic maps.
    """
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    from shapely.geometry import box as _box

    active = ~np.isnan(hmax)
    flooded = active & (hmax > 0)
    n_active = int(active.sum())
    n_flooded = int(flooded.sum())
    frac = n_flooded / n_active if n_active > 0 else 0.0

    plot_data = np.where(flooded, hmax, np.nan)
    vmax = (
        float(np.nanpercentile(plot_data[~np.isnan(plot_data)], 99))
        if n_flooded > 0
        else 1.0
    )
    vmax = max(vmax, 0.01)

    xmin, xmax, ymin, ymax = extent
    h, w = plot_data.shape
    transform = from_origin(xmin, ymax, (xmax - xmin) / w, (ymax - ymin) / h)
    plot_data_wgs, wgs_extent = _to_wgs84_grid(
        plot_data, transform, CRS.from_epsg(int(epsg))
    )
    wgs_left, wgs_right, wgs_bottom, wgs_top = wgs_extent

    fig, ax = plt.subplots(figsize=(9, 7))
    if osm_land_path is not None:
        map_background(
            ax, _box(wgs_left, wgs_bottom, wgs_right, wgs_top), osm_land_path
        )
    im = ax.imshow(
        plot_data_wgs,
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        extent=wgs_extent,
        origin="upper",
        zorder=2,
    )
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, extend="max")
    cb.set_label("Max inundation depth (m)")
    title = "Max inundation depth — spinup baseline"
    if basin_id:
        title = f"{title} | {basin_id}"
    ax.set_title(
        f"{title}\n{n_flooded:,} / {n_active:,} active cells flooded ({frac:.1%})"
    )
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    _save(fig, output_path)


# ── rule 10 plots ─────────────────────────────────────────────────────────────


def plot_inundation_check(
    da_hmax: xr.DataArray,
    threshold_m: float,
    n_flooded: int,
    n_land: int,
    land_polygons_path: str,
    river_network_path: str,
    output_path: str,
    basin_id: str = "",
) -> None:
    """
    Sanity check: fraction of land pixels exceeding an inundation threshold.

    Left panel — map of max inundation depth using da_hmax.plot(), which reads
                 the DataArray's own x/y coordinates so north-up orientation is
                 handled automatically regardless of y-axis direction; the land
                 outline and clean river network are overlaid for context.
    Right panel — bar chart of flooded vs dry pixel counts.

    Args:
        da_hmax:            DataArray output of downscale_floodmap; NaN = dry /
                            outside domain.  Must carry CRS metadata
                            (``rio.crs``) and have 1-D x and y coordinate arrays.
        threshold_m:        hmin passed to downscale_floodmap (for labelling only).
        n_flooded:          Pre-computed flooded land pixel count (da_hmax not-null).
        n_land:             Pre-computed total land pixel count (dep not-null).
        land_polygons_path: Path to the OSM land polygons geopackage (WGS84),
                            drawn as an outline overlay.
        river_network_path: Path to the clean river network geopackage (WGS84),
                            drawn as an overlay.
        output_path:        Destination PNG path.
        basin_id:           Basin identifier for the plot title.
    """
    frac = n_flooded / n_land if n_land > 0 else 0.0
    n_dry = n_land - n_flooded

    # Reproject from the model's metric UTM grid to EPSG:4326.  Use explicit
    # imshow with extent rather than da.plot() to guarantee north-up orientation
    # regardless of how downscale_floodmap orders the spatial dimensions.
    da_wgs = da_hmax.squeeze().rio.reproject("EPSG:4326")
    y_dim = da_wgs.rio.y_dim
    x_dim = da_wgs.rio.x_dim
    if da_wgs.dims[0] != y_dim:
        da_wgs = da_wgs.transpose(y_dim, x_dim)
    wgs_arr = da_wgs.values.astype(np.float32)
    # rio.reproject returns y-decreasing (north at row 0); flip if south-up.
    ys = da_wgs.coords[y_dim].values
    if len(ys) >= 2 and float(ys[0]) < float(ys[-1]):
        wgs_arr = wgs_arr[::-1]

    wgs_left, wgs_bottom, wgs_right, wgs_top = da_wgs.rio.bounds()
    _margin = max(wgs_right - wgs_left, wgs_top - wgs_bottom) * 0.3

    # Land polygons are stored in WGS84; use bounds for spatial filter.
    land = gpd.read_file(
        land_polygons_path, bbox=(wgs_left, wgs_bottom, wgs_right, wgs_top)
    )
    # River network may be in UTM: load fully then reproject.
    rivers = gpd.read_file(river_network_path)
    if not rivers.empty and rivers.crs is not None and rivers.crs.to_epsg() != 4326:
        rivers = rivers.to_crs("EPSG:4326")

    valid = wgs_arr[~np.isnan(wgs_arr)]
    vmax = float(np.percentile(valid, 99)) if len(valid) > 0 else 1.0
    vmax = max(vmax, 0.01)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 1]}
    )

    # Left: land background, then inundation raster, then rivers on top.
    if not land.empty:
        land.plot(ax=ax1, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=1)
    im = ax1.imshow(
        wgs_arr,
        cmap="Blues",
        vmin=threshold_m,
        vmax=vmax,
        extent=(wgs_left, wgs_right, wgs_bottom, wgs_top),
        origin="upper",
        aspect="auto",
        zorder=2,
    )
    cbar = fig.colorbar(im, ax=ax1, extend="max", fraction=0.03, pad=0.04)
    cbar.set_label(f"Max inundation depth (m, ≥ {threshold_m} m)")
    if not rivers.empty:
        rivers.plot(ax=ax1, color="steelblue", linewidth=0.6, zorder=3)
    ax1.set_xlim(wgs_left - _margin, wgs_right + _margin)
    ax1.set_ylim(wgs_bottom - _margin, wgs_top + _margin)
    ax1.set_aspect("equal")
    ax1.set_xlabel("Longitude (°)")
    ax1.set_ylabel("Latitude (°)")
    ax1.set_title(
        f"Max inundation depth (hmin = {threshold_m} m)\n"
        f"{n_flooded:,} / {n_land:,} land pixels flooded  ({frac:.2%})"
    )
    ax1.legend(
        handles=[
            Patch(color="#2171b5", label=f"Flooded > {threshold_m} m ({n_flooded:,})"),
            Patch(
                color="#d9d9d9", edgecolor="#aaaaaa", label=f"Dry / shallow ({n_dry:,})"
            ),
            Line2D([0], [0], color="steelblue", linewidth=1.5, label="River network"),
        ],
        loc="lower right",
        framealpha=0.9,
    )
    ax1.grid(True, alpha=0.3, linewidth=0.5)

    # Right: bar chart
    bars = ax2.bar(
        ["Flooded", "Dry"],
        [n_flooded, n_dry],
        color=["#2171b5", "#d9d9d9"],
        edgecolor="black",
        linewidth=0.6,
    )
    for bar, val in zip(bars, [n_flooded, n_dry]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax2.set_ylabel("Land pixel count")
    ax2.set_title(f"Flooded fraction\n{frac:.2%}")
    ax2.grid(True, alpha=0.3, axis="y", linewidth=0.5)

    suptitle = "Sanity check — inundation ratio (baseline spinup)"
    if basin_id:
        suptitle = f"{suptitle} | {basin_id}"
    fig.suptitle(f"{suptitle}\nThreshold: {threshold_m} m", fontsize=12)
    fig.tight_layout()
    _save(fig, output_path)


def plot_longitudinal_profile(
    rivers_wgs: gpd.GeoDataFrame,
    seed_reach_ids: set[str],
    output_path: str,
) -> None:
    """
    Two-panel longitudinal profile from the most upstream seed reach to the
    ocean outlet.

    Panel 1 tracks accumulated discharge; panel 2 tracks hydraulic depth.
    Both panels share the same x-axis (distance from the entry seed reach).
    At bifurcations the line branches into one segment per downstream reach,
    showing how discharge and depth are distributed across channels.  At
    confluences from other upstream sources an abrupt rise in the profile
    is expected.
    """
    # ── reach lookups ─────────────────────────────────────────────────────────

    def _norm(x) -> str | None:
        s = str(x).strip()
        if s.lower() in ("nan", "none", "<na>", ""):
            return None
        try:
            return str(int(float(s)))
        except (ValueError, OverflowError):
            return s if s else None

    def _parse_dn(raw) -> list[str]:
        s = str(raw).strip().strip("[]")
        if not s or s.lower() in ("nan", "none", "<na>"):
            return []
        result = []
        for token in s.split(","):
            n = _norm(token.strip())
            if n:
                result.append(n)
        return result

    rids: list[str] = [
        r for r in (_norm(x) for x in rivers_wgs["reach_id"]) if r is not None
    ]
    valid_set: set[str] = set(rids)

    dist_of: dict[str, float] = {}
    q_of: dict[str, float] = {}
    dph_of: dict[str, float] = {}
    dn_of: dict[str, list[str]] = {}

    for rid, dist, q, dph, dn_raw in zip(
        rids,
        rivers_wgs["dist_out"],
        rivers_wgs["bankfull_discharge_acc"],
        rivers_wgs["rivdph"],
        rivers_wgs["rch_id_dn"],
    ):
        if rid is None:
            continue
        try:
            dist_of[rid] = float(dist)
        except (ValueError, TypeError):
            dist_of[rid] = 0.0
        try:
            q_of[rid] = float(q)
        except (ValueError, TypeError):
            q_of[rid] = 0.0
        try:
            dph_of[rid] = float(dph)
        except (ValueError, TypeError):
            dph_of[rid] = 0.0
        dn_of[rid] = [d for d in _parse_dn(dn_raw) if d in valid_set]

    # ── root: seed reach with largest dist_out ────────────────────────────────

    valid_seeds: set[str] = {
        s for s in (_norm(r) for r in seed_reach_ids) if s is not None
    }
    valid_seeds &= valid_set
    if not valid_seeds:
        log.warning("No valid seed reaches for longitudinal profile; skipping plot")
        Path(output_path).touch()
        return

    root_id = max(valid_seeds, key=lambda s: dist_of.get(s, 0.0))
    root_dist = dist_of[root_id]
    log.info(f"Longitudinal profile root: {root_id} (dist_out={root_dist:.0f} m)")

    # ── BFS to collect edges (parent, child) ──────────────────────────────────

    edges: list[tuple[str, str]] = []
    visited: set[str] = set()
    queue: list[str] = [root_id]
    while queue:
        rid = queue.pop(0)
        if rid in visited:
            continue
        visited.add(rid)
        for dn_id in dn_of.get(rid, []):
            edges.append((rid, dn_id))
            if dn_id not in visited:
                queue.append(dn_id)

    if not edges:
        log.warning("No downstream edges found for longitudinal profile; skipping plot")
        Path(output_path).touch()
        return

    log.info(f"Longitudinal profile: {len(visited)} reaches, {len(edges)} edges")

    # ── plot ──────────────────────────────────────────────────────────────────

    alpha = max(0.12, min(0.85, 40.0 / len(edges)))
    lw = 0.5 if len(edges) > 200 else 1.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for parent_id, child_id in edges:
        x1 = root_dist - dist_of.get(parent_id, 0.0)
        x2 = root_dist - dist_of.get(child_id, 0.0)
        ax1.plot(
            [x1, x2],
            [q_of.get(parent_id, 0), q_of.get(child_id, 0)],
            color="steelblue",
            alpha=alpha,
            linewidth=lw,
        )
        ax2.plot(
            [x1, x2],
            [dph_of.get(parent_id, 0), dph_of.get(child_id, 0)],
            color="seagreen",
            alpha=alpha,
            linewidth=lw,
        )

    for ax in (ax1, ax2):
        ax.axvline(
            0,
            color="darkorange",
            linewidth=1.0,
            linestyle="--",
            alpha=0.7,
            label="Entry point",
        )
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(loc="best", framealpha=0.9, fontsize=8)

    ax1.set_ylabel("Accumulated discharge (m³ s⁻¹)")
    ax1.set_title(
        f"Longitudinal profile — root reach {root_id} "
        f"({len(visited)} reaches, {len(edges)} edges)"
    )
    ax2.set_ylabel("Hydraulic depth (m)")
    ax2.set_xlabel("Distance from entry point (m)")

    fig.tight_layout()
    _save(fig, output_path)
