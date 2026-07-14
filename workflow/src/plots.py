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
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes as _rio_shapes
from rasterio.warp import calculate_default_transform, transform_geom as _transform_geom
import rioxarray  # noqa: F401  — registers the .rio accessor used for reprojection
import xarray as xr
import xugrid as xu
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import Polygon, shape as _shape

from src.geometry import pick_utm_crs

matplotlib.use("Agg")

log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

_PLOT_DPI = 100
_PLOT_MAX_PX = 2_000_000  # downsample rasters larger than this before rendering

_VMIN_ELEV = -30.0
_VMAX_ELEV = 80.0
_THRESH_COLOR = "#FF4500"  # used for pixels above _VMAX_ELEV in topography plots

# Diverging land/sea colormap: shades of blue below 0 m, shades of brown above.
_BATHY_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "bathymetry",
    np.vstack(
        [
            plt.get_cmap("Blues_r")(np.linspace(0.0, 0.85, 128)),
            plt.get_cmap("YlOrBr")(np.linspace(0.15, 0.95, 128)),
        ]
    ),
)

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
    water_bodies_path: str | None = None,
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
        water_bodies_path:  Optional path to a landuse raster (UTM).  Pixels with
                            value 200 (permanent water body) are vectorised and
                            drawn as white patches over the land polygon so inland
                            water bodies are not miscoloured as land.
        margin_frac:        Fraction of the bbox span added as margin on each side.
    """
    lon_min, lat_min, lon_max, lat_max = bbox_poly.bounds
    margin = max(lon_max - lon_min, lat_max - lat_min) * margin_frac
    xmin, ymin = lon_min - margin, lat_min - margin
    xmax, ymax = lon_max + margin, lat_max + margin

    land = gpd.read_file(osm_land_path, bbox=(xmin, ymin, xmax, ymax), engine="pyogrio")
    if not land.empty:
        land.plot(ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=1)

    if water_bodies_path is not None:
        with rasterio.open(water_bodies_path) as _wb_src:
            _wb_arr = _wb_src.read(1)
            _wb_mask = (_wb_arr == 200).astype(np.uint8)
            if _wb_mask.any():
                _src_crs = _wb_src.crs.to_wkt()
                _geoms = [
                    _shape(_transform_geom(_src_crs, "EPSG:4326", geom))
                    for geom, val in _rio_shapes(
                        _wb_mask, mask=_wb_mask, transform=_wb_src.transform
                    )
                    if val == 1
                ]
                if _geoms:
                    gpd.GeoDataFrame(geometry=_geoms, crs="EPSG:4326").plot(
                        ax=ax, color="white", edgecolor="none", zorder=1.5
                    )

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


# ── static data plots (rules 04-06: protection levels, elevation, landuse, ────
# ── roughness, river network) ─────────────────────────────────────────────────


def plot_topography(
    topo_path: str,
    bbox_poly: Polygon,
    osm_land_path: str,
    output_path: str,
    water_bodies_path: str | None = None,
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
    map_background(ax, bbox_poly, osm_land_path, water_bodies_path=water_bodies_path)
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
    ax.set_title("Topography (FathomDEM)")
    _save(fig, output_path)


def plot_bathymetry(
    bathy_path: str,
    bbox_poly: Polygon,
    osm_land_path: str,
    output_path: str,
    water_bodies_path: str | None = None,
) -> None:
    """
    Bathymetry map: diverging blue (depth) / brown (elevation) colormap,
    centred at 0 m, spanning [_VMIN_ELEV, _VMAX_ELEV] m (GEBCO).
    """
    data, extent = read_raster_for_plot(bathy_path)

    fig, ax = plt.subplots(figsize=(9, 7))
    map_background(ax, bbox_poly, osm_land_path, water_bodies_path=water_bodies_path)
    im = ax.imshow(
        data,
        cmap=_BATHY_CMAP,
        norm=mcolors.TwoSlopeNorm(vmin=_VMIN_ELEV, vcenter=0, vmax=_VMAX_ELEV),
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
    water_bodies_path: str | None = None,
) -> None:
    """
    Categorical land-use map with Copernicus LC100 class colours and legend.

    landuse.tif is on the model's metric UTM working grid (reprojected onto
    elevation_merged.tif's exact grid in 03b) -- reproject to EPSG:4326 for
    display, nearest-neighbour to preserve exact class codes.
    """
    data, extent = read_raster_reprojected_for_plot(
        landuse_path, dst_bounds=bbox_poly.bounds
    )
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
    map_background(ax, bbox_poly, osm_land_path, water_bodies_path=water_bodies_path)
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
    water_bodies_path: str | None = None,
) -> None:
    """
    Manning's n roughness map: viridis discrete colormap with labelled colorbar.

    roughness.tif inherits landuse.tif's grid (the model's metric UTM working
    grid) -- reproject to EPSG:4326 for display, nearest-neighbour to preserve
    exact roughness values.
    """
    data, extent = read_raster_reprojected_for_plot(
        roughness_path, dst_bounds=bbox_poly.bounds
    )
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
    map_background(ax, bbox_poly, osm_land_path, water_bodies_path=water_bodies_path)
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
    water_bodies_path: str | None = None,
) -> None:
    """
    Merged FathomDEM+GEBCO elevation map, diverging blue (depth) / brown-
    orange (elevation) colormap centred at 0 m.

    vmin/vmax are fixed at -10 m / 30 m regardless of the data's actual
    range, so elevation maps are visually comparable across basins; values
    outside this range are clipped to the colormap's end colours by imshow
    (not masked/hidden).

    The raster (on the model's metric UTM grid) is reprojected to EPSG:4326
    for display so this map shares the same lon/lat reference frame as every
    other diagnostic plot.
    """
    data, extent = read_raster_reprojected_for_plot(
        merged_path, dst_bounds=bbox_poly.bounds
    )

    vmin, vmax = -10.0, 30.0

    fig, ax = plt.subplots(figsize=(9, 7))
    map_background(ax, bbox_poly, osm_land_path, water_bodies_path=water_bodies_path)
    im = ax.imshow(
        data,
        cmap=_BATHY_CMAP,
        norm=mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax),
        extent=extent,
        origin="upper",
        zorder=2,
    )

    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cb.set_label("Elevation (m)")
    ax.set_title(title_str)
    _save(fig, output_path)


def plot_zsini(
    zsini_path: str,
    bbox_poly: Polygon,
    osm_land_path: str,
    output_path: str,
    water_bodies_path: str | None = None,
) -> None:
    """
    Initial water-level mask (zsini): cells seeded with an initial water level
    of 0 m (open water at model start) vs land / outside-domain cells (nodata
    — dry at model start).  Includes inland water bodies (landuse 200), which
    are overridden to the sea initial value.

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
    map_background(ax, bbox_poly, osm_land_path, water_bodies_path=water_bodies_path)
    ax.imshow(
        np.where(water_mask, 1.0, np.nan),
        cmap=mcolors.ListedColormap(["#3860D0"]),
        vmin=0,
        vmax=2,
        extent=extent,
        origin="upper",
        zorder=10,
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


def plot_protection_levels(
    delta_polygon_wgs84: Polygon,
    geogunit_raster_path: str,
    flopros_df,
    summary: dict,
    osm_land_path: str,
    output_path: str,
    margin_frac: float = 0.5,
    water_bodies_path: str | None = None,
) -> None:
    """
    Two-panel map (riverine / coastal) of FLOPROS design protection return
    period (years) for every geounit overlapping the delta polygon's bbox
    (plus a display margin), with the delta polygon outline overlaid and the
    dominant (largest-area) geounit's resolved RP stated in each title.

    A small windowed read of geogunit_raster_path clips to that bbox -- the
    full global raster is never loaded.

    Args:
        delta_polygon_wgs84:  Delta polygon (EPSG:4326).
        geogunit_raster_path: Path to the wri_geogunit_107 raster.
        flopros_df:           Output of protection_levels.load_flopros_table().
        summary:              Output of protection_levels.identify_dominant_protection()
                               for this same delta polygon -- supplies the
                               resolved (post-fallback/cap) RP values and
                               dominant unit's id/ISO for the titles.
        osm_land_path:        Path to the OSM land polygons (background).
        output_path:          Output PNG path.
        margin_frac:          Display margin as a fraction of the polygon's
                               bbox span (same convention as map_background).
    """
    import pandas as pd
    import shapely.vectorized
    import xarray as xr
    from shapely.geometry import box as _box

    bbox_poly = _box(*delta_polygon_wgs84.bounds)
    lon_min, lat_min, lon_max, lat_max = delta_polygon_wgs84.bounds
    margin = max(lon_max - lon_min, lat_max - lat_min) * margin_frac
    window_bounds = (
        lon_min - margin,
        lat_min - margin,
        lon_max + margin,
        lat_max + margin,
    )

    with xr.open_dataset(geogunit_raster_path) as ds:
        # lat/lon are both ascending in wri_geogunit_107 -- a plain slice()
        # works without descending-coordinate handling.
        sel = ds["Geogunits"].sel(
            lat=slice(window_bounds[1], window_bounds[3]),
            lon=slice(window_bounds[0], window_bounds[2]),
        )
        arr = sel.values.astype(np.float64)
        lon_vals = sel["lon"].values
        lat_vals = sel["lat"].values

    # arr's row 0 = lat_vals[0] (southernmost, ascending) -- origin="lower"
    # below matches that orientation directly, no flip needed.
    extent = (
        float(lon_vals.min()),
        float(lon_vals.max()),
        float(lat_vals.min()),
        float(lat_vals.max()),
    )

    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    inside = shapely.vectorized.contains(delta_polygon_wgs84, lon_grid, lat_grid)
    # xarray CF-decodes the raster's _FillValue to NaN on read, so
    # np.isfinite already excludes fill-value pixels.
    valid = inside & np.isfinite(arr)

    max_rp_for_scale = max(
        100.0,
        summary.get("riverine_rp_yr") or 0.0,
        summary.get("coastal_rp_yr") or 0.0,
    )
    norm = mcolors.LogNorm(vmin=1.0, vmax=max_rp_for_scale)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#d0d0d0")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    for ax, hazard, key in zip(
        axes, ("Riverine", "Coastal"), ("riverine_rp_yr", "coastal_rp_yr")
    ):
        map_background(
            ax, bbox_poly, osm_land_path, water_bodies_path=water_bodies_path
        )
        values_by_id = flopros_df[hazard].to_dict()
        rp_arr = np.full(arr.shape, np.nan, dtype=float)
        ids_in_window = np.unique(arr[valid].astype(np.int64))
        for gid in ids_in_window:
            val = values_by_id.get(int(gid))
            if val is not None and not pd.isna(val):
                rp_arr[valid & (arr.astype(np.int64) == gid)] = float(val)

        im = ax.imshow(
            rp_arr, cmap=cmap, norm=norm, extent=extent, origin="lower", zorder=2
        )
        bx, by = delta_polygon_wgs84.exterior.xy
        ax.plot(bx, by, color="blue", linewidth=1.5, zorder=11, label="Delta polygon")
        ax.set_title(
            f"{hazard} protection\nResolved RP used: {summary.get(key, float('nan')):.1f} yr "
            f"({summary.get(f'{hazard.lower()}_source', 'n/a')})\n"
            f"Dominant unit: {summary.get('dominant_iso') or 'n/a'} "
            f"(id={summary.get('dominant_geounit_id')}, "
            f"{100 * (summary.get('pixel_fraction') or 0):.0f}% of delta-polygon area)"
        )
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, extend="max")
    cb.set_label("Design protection return period (yr)")
    fig.suptitle("FLOPROS existing flood protection standard (WRI geogunit_107)")
    _save(fig, output_path)


def plot_global_protection_map(
    flopros_df,
    countries_gdf: gpd.GeoDataFrame,
    riverine_output_path: str,
    coastal_output_path: str,
    iso_column: str = "ISO_A3_EH",
    agg: str = "median",
) -> None:
    """
    Two separate world choropleths -- one riverine, one coastal -- of FLOPROS
    design flood protection return period (years), aggregated to country
    level. Both share the same (log) colorscale so they remain directly
    comparable even though they're saved as separate figures.

    Each country's value is the `agg` (default: median, robust to outlier
    sub-national units) of its Riverine/Coastal FLOPROS design RP across all
    its geounits that DO carry a value for that hazard. Countries with no
    FLOPROS value for a hazard anywhere are left NaN and rendered grey -- no
    default/fallback value is guessed here (unlike the per-basin pipeline's
    fallback chain in protection_levels.identify_dominant_protection, which
    exists to guarantee a usable RP for the forcing correction, not to
    faithfully represent "no known standard").

    Args:
        flopros_df:           Output of protection_levels.load_flopros_table()
                               -- must have an "ISO" column plus
                               "Riverine"/"Coastal".
        countries_gdf:         World country polygons (e.g. Natural Earth
                               admin_0_countries) with an ISO3 code column.
        riverine_output_path:  Output PNG path for the riverine map.
        coastal_output_path:   Output PNG path for the coastal map.
        iso_column:            Column in countries_gdf holding the ISO3 code
                               to join on (Natural Earth's "ISO_A3" is -99 for
                               a handful of countries, e.g. France/Norway --
                               "ISO_A3_EH" doesn't have that gap).
        agg:                   Aggregation applied within each country
                               ("median", "mean", "max", ...).
    """
    country_agg = flopros_df.groupby("ISO")[["Riverine", "Coastal"]].agg(agg)
    merged = countries_gdf.merge(
        country_agg, how="left", left_on=iso_column, right_index=True
    )

    finite_vals = merged[["Riverine", "Coastal"]].to_numpy(dtype=float)
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    max_rp = max(100.0, float(finite_vals.max())) if finite_vals.size else 100.0
    norm = mcolors.LogNorm(vmin=1.0, vmax=max_rp)
    cmap = plt.get_cmap("YlOrRd").copy()

    # World map extent is much wider than tall (360x145 deg); geopandas locks
    # the axes to equal aspect, so a figsize not matching that ratio leaves
    # large blank margins around the (aspect-shrunk) map that bbox_inches=
    # "tight" can't crop away (it's the axes' allocated box, not just its
    # drawn content). figsize is sized to match the map's own aspect ratio
    # plus fixed headroom for the title/colorbar.
    map_aspect = (85 - -60) / (180 - -180)  # lat span / lon span
    fig_w_in = 14.0
    fig_h_in = fig_w_in * map_aspect + 1.3  # + title/colorbar headroom

    output_paths = {"Riverine": riverine_output_path, "Coastal": coastal_output_path}
    for hazard, output_path in output_paths.items():
        fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), constrained_layout=True)
        merged.plot(
            column=hazard,
            ax=ax,
            cmap=cmap,
            norm=norm,
            missing_kwds={"color": "#d0d0d0", "label": "No FLOPROS data"},
            edgecolor="#888888",
            linewidth=0.2,
        )
        n_with_data = int(merged[hazard].notna().sum())
        ax.set_title(
            f"{hazard} flood protection standard\n"
            f"{n_with_data}/{len(merged)} countries with FLOPROS data ({agg} of sub-national units)"
        )
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(
            sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02, shrink=0.5
        )
        cbar.set_label("Design protection return period (years)")
        _save(fig, output_path)


def plot_river_network(
    river_path: str,
    bbox_poly: Polygon,
    osm_land_path: str,
    output_path: str,
    water_bodies_path: str | None = None,
) -> None:
    """
    Clipped river network overlaid on OSM land background and domain bbox.
    """
    rivers = gpd.read_file(river_path)
    if rivers.crs and rivers.crs.to_epsg() != 4326:
        rivers = rivers.to_crs("EPSG:4326")

    fig, ax = plt.subplots(figsize=(9, 7))
    map_background(ax, bbox_poly, osm_land_path, water_bodies_path=water_bodies_path)
    if not rivers.empty:
        rivers.plot(ax=ax, color="steelblue", linewidth=0.8, zorder=5)
    ax.set_title("River network (clipped to model domain)")
    _save(fig, output_path)


# ── rule 07 plots (get_boundary_forcings) ─────────────────────────────────────


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
    Map of the model domain, river network, surge stations, and river
    crossings.

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

    # Exclusion-reason masks — derived from filter columns written by rule 07.
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

    If the protection-level correction was applied (top-level
    protection_levels.enabled -- surge_ds/river_ds then carry
    'water_level_uncorrected'/'discharge_uncorrected' and
    'protection_level'/'protection_discharge'), each panel shows three
    layers per station/crossing instead of just the single effective
    series: the original (undefended) timeseries, the constant protection
    level/discharge subtracted, and the resulting effective (modelled)
    hydrograph -- so the size of the correction is visible directly,
    not just its end result. Falls back to the single-series plot
    (unchanged from before this correction existed) when disabled.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    surge_corrected = "water_level_uncorrected" in surge_ds
    river_corrected = "discharge_uncorrected" in river_ds

    t_surge = surge_ds["time"].values
    wl = surge_ds["water_level"].values
    if surge_corrected:
        wl_raw = surge_ds["water_level_uncorrected"].values
        protection_level = surge_ds["protection_level"].values
        for i in range(wl.shape[0]):
            ax1.plot(
                t_surge,
                wl_raw[i],
                color="grey",
                alpha=0.4,
                linewidth=0.8,
                linestyle="--",
            )
            ax1.hlines(
                protection_level[i],
                t_surge.min(),
                t_surge.max(),
                color="darkorange",
                alpha=0.6,
                linewidth=0.8,
                linestyle=":",
            )
            ax1.plot(t_surge, wl[i], color="steelblue", alpha=0.7, linewidth=0.8)
    else:
        for i in range(wl.shape[0]):
            ax1.plot(t_surge, wl[i], color="steelblue", alpha=0.5, linewidth=0.8)

    if surge_corrected:
        ax1.legend(
            handles=[
                Line2D([0], [0], color="grey", linestyle="--", label="Original"),
                Line2D(
                    [0],
                    [0],
                    color="darkorange",
                    linestyle=":",
                    label="Protection level",
                ),
                Line2D([0], [0], color="steelblue", label="Effective (modelled)"),
            ],
            loc="upper left",
            fontsize=8,
            framealpha=0.9,
        )

    ax1.set_xlabel("Time (hours since start)")
    ax1.set_ylabel("Water level (m, GOCO6s frame)")
    ax1.set_title(f"Surge forcing — RP{return_period} ({wl.shape[0]} stations)")
    ax1.grid(True, alpha=0.3)

    # ── per-station correction table ─────────────────────────────────────────
    # Shows the correction chain: rp_raw → −MDT → +SLR → peak(GOCO6s) → −prot → final_peak
    # protection_level in nc is stored in local MSL (raw, no MDT correction).
    # final_peak = peak(GOCO6s) − prot_raw = rp_raw − MDT + SLR − prot_raw (in GOCO6s).
    # baseline_m = mean(0 − MDT + SLR − prot) = mean over stations.
    has_raw = "rp_level_raw" in surge_ds
    has_mdt = "mdt" in surge_ds
    has_prot_ds = "protection_level" in surge_ds
    if has_raw and has_mdt:
        rp_levels = surge_ds["rp_level"].values  # peak in GOCO6s (= rp_raw − MDT + SLR)
        rp_levels_raw = surge_ds["rp_level_raw"].values  # raw COAST-RP (local MSL)
        mdts = surge_ds["mdt"].values
        slr_arr = (
            surge_ds["slr_m"].values
            if "slr_m" in surge_ds
            else np.zeros(len(rp_levels))
        )
        # protection_level stored as local-MSL raw value (no MDT correction)
        prot_arr = (
            surge_ds["protection_level"].values
            if has_prot_ds
            else np.zeros(len(rp_levels))
        )
        # final_peak = rp_raw − MDT + SLR − prot_raw (in GOCO6s)
        final_peaks = rp_levels - prot_arr

        if has_prot_ds:
            header = "Stn  rp_raw    −MDT    +SLR  peak(GOCO6s)  −prot  final_peak"
        else:
            header = "Stn  rp_raw    −MDT    +SLR  peak(GOCO6s)"
        rows = [header, "─" * len(header)]
        for i, (rl_raw, mdt_i, slr_i, rl, pr, fp) in enumerate(
            zip(rp_levels_raw, mdts, slr_arr, rp_levels, prot_arr, final_peaks)
        ):
            if has_prot_ds:
                rows.append(
                    f" {i + 1:2d}  {rl_raw:+7.3f}  {-mdt_i:+6.3f}  {slr_i:+6.3f}"
                    f"    {rl:+7.3f}  {-pr:+6.3f}    {fp:+7.3f}"
                )
            else:
                rows.append(
                    f" {i + 1:2d}  {rl_raw:+7.3f}  {-mdt_i:+6.3f}  {slr_i:+6.3f}    {rl:+7.3f}"
                )
        if has_prot_ds:
            rows.append("─" * len(header))
            bm = (
                float(surge_ds["baseline_m"].values)
                if "baseline_m" in surge_ds
                else float(np.mean(-mdts + slr_arr - prot_arr))
            )
            rows.append(f"  baseline_m (MWL (=0) − MDT + SLR − prot): {bm:+.4f} m")

        ax1.text(
            0.01,
            0.02,
            "\n".join(rows),
            transform=ax1.transAxes,
            fontsize=6.0,
            family="monospace",
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.85),
        )

    t_river = river_ds["time"].values
    q = river_ds["discharge"].values
    active = river_ds["has_glofas"].values.astype(bool)
    if river_corrected:
        q_raw = river_ds["discharge_uncorrected"].values
        protection_discharge = river_ds["protection_discharge"].values
        for i in range(q.shape[0]):
            if not active[i]:
                continue
            ax2.plot(
                t_river,
                q_raw[i],
                color="grey",
                alpha=0.4,
                linewidth=0.8,
                linestyle="--",
            )
            ax2.hlines(
                protection_discharge[i],
                t_river.min(),
                t_river.max(),
                color="darkorange",
                alpha=0.6,
                linewidth=0.8,
                linestyle=":",
            )
            ax2.plot(t_river, q[i], color="teal", alpha=0.7, linewidth=0.8)
        ax2.legend(
            handles=[
                Line2D([0], [0], color="grey", linestyle="--", label="Original"),
                Line2D(
                    [0],
                    [0],
                    color="darkorange",
                    linestyle=":",
                    label="Protection level",
                ),
                Line2D([0], [0], color="teal", label="Effective (modelled)"),
            ],
            loc="upper right",
            fontsize=8,
            framealpha=0.9,
        )
    else:
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


def plot_surge_corrections(
    stations: gpd.GeoDataFrame,
    output_path: str,
    protection_level_raw=None,
) -> None:
    """
    Diagnostic stacked-bar plot for all vertical corrections applied to COAST-RP
    storm-tide levels (MDT shift, SLR fingerprint, flood-protection subtraction).

    Left panel — four sub-bars per station, each anchored at 0 m (local MSL):
      1. rp_level_raw  (+ SLR stacked on top if nonzero) — steelblue/seagreen
      2. −MDT correction — darkorange; extends below 0 when MDT > 0, above 0 when MDT < 0
      3. −protection level (only when enabled) — mediumpurple; always below 0
      4. Net final peak = rp_raw − MDT + SLR − prot — navy

    Each correction has its own x sub-position so even tiny MDT bars are fully
    visible and cannot be obscured by the protection bar.

    Right panel: station locations coloured by the −MDT correction magnitude.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    n = len(stations)
    x = np.arange(n, dtype=float)

    raw = stations["rp_level_raw"].values
    mdt = (
        stations["mdt"].fillna(0.0).values if "mdt" in stations.columns else np.zeros(n)
    )
    slr = (
        stations["slr_m"].fillna(0.0).values
        if "slr_m" in stations.columns
        else np.zeros(n)
    )
    has_prot = protection_level_raw is not None
    prot_raw = np.asarray(protection_level_raw) if has_prot else np.zeros(n)

    mdt_corr = -mdt  # negative (below 0) when MDT > 0; positive (above 0) when MDT < 0

    # ── Sub-bar x positions (each component owns its own column) ─────────────
    # Layout (left → right): rp_raw | −MDT | −prot (if any) | net_peak
    n_bars = 3 + (1 if has_prot else 0)
    w = min(0.20, 0.85 / n_bars)
    g = 0.03
    half = (n_bars - 1) / 2.0 * (w + g)
    centers = np.array([i * (w + g) for i in range(n_bars)]) - half

    x_raw = x + centers[0]
    x_mdt = x + centers[1]
    if has_prot:
        x_prot = x + centers[2]
        x_net = x + centers[3]
    else:
        x_net = x + centers[2]

    # ── Bar 1: rp_level_raw + SLR ────────────────────────────────────────────
    ax1.bar(
        x_raw,
        raw,
        width=w,
        color="steelblue",
        label="rp_level_raw  (COAST-RP, local MSL)",
    )
    ax1.bar(x_raw, slr, width=w, bottom=raw, color="seagreen", label="+SLR fingerprint")

    # ── Bar 2: −MDT correction (each station its own column, anchored at 0) ──
    ax1.bar(
        x_mdt, mdt_corr, width=w, color="darkorange", label="−MDT  (local MSL → GOCO6s)"
    )

    # ── Bar 3: −protection (anchored at 0, independent of −MDT) ─────────────
    if has_prot:
        ax1.bar(
            x_prot,
            -prot_raw,
            width=w,
            color="mediumpurple",
            label="−protection level  (FLOPROS coastal, local MSL)",
        )

    # ── Bar 4: net final peak ─────────────────────────────────────────────────
    net_peak = raw - mdt + slr - prot_raw
    ax1.bar(
        x_net,
        net_peak,
        width=w,
        color="navy",
        alpha=0.75,
        label="Final peak  (rp_level_raw − MDT + SLR − protection)",
    )

    # ── Value annotations (black outside for small bars, white inside large) ──
    def _annotate_bar(ax, xs, vals, bottoms, fontsize=6.5, min_inside=0.12):
        for xi, v, b in zip(xs, vals, bottoms):
            if abs(v) < 1e-6:
                continue
            mid_y = b + v / 2.0
            outside = abs(v) < min_inside
            y_txt = (b + v + np.sign(v) * 0.02) if outside else mid_y
            va = ("bottom" if v > 0 else "top") if outside else "center"
            color = "black" if outside else "white"
            ax.text(
                xi,
                y_txt,
                f"{v:+.3f}",
                ha="center",
                va=va,
                fontsize=fontsize,
                color=color,
                clip_on=True,
            )

    _annotate_bar(ax1, x_raw, raw, np.zeros(n))
    if np.any(slr != 0):
        _annotate_bar(ax1, x_raw, slr, raw)
    _annotate_bar(ax1, x_mdt, mdt_corr, np.zeros(n))
    if has_prot:
        _annotate_bar(ax1, x_prot, -prot_raw, np.zeros(n))
    _annotate_bar(ax1, x_net, net_peak, np.zeros(n))

    ax1.axhline(
        0.0, color="black", linewidth=0.9, linestyle="--", label="0 m  (local MSL)"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(i + 1) for i in range(n)])
    ax1.set_xlabel("Station")
    ax1.set_ylabel("Water level relative to local MSL (m)")
    ax1.set_title(
        "Surge correction decomposition per station\n"
        "(bars left→right: rp_raw | −MDT | −prot | net peak)"
    )
    ax1.legend(fontsize=7, framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis="y")

    # ── Right panel: spatial map of −MDT correction ───────────────────────────
    sc = ax2.scatter(
        stations.geometry.x,
        stations.geometry.y,
        c=mdt_corr,
        cmap="coolwarm",
        s=60,
        edgecolor="k",
    )
    fig.colorbar(sc, ax=ax2, label="−MDT correction (m)")
    ax2.set_xlabel("Longitude (°)")
    ax2.set_ylabel("Latitude (°)")
    ax2.set_title("MDT correction per station  (−mdt, m)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, output_path)


# ── rule 08 plots (clean_river_network) ───────────────────────────────────────


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


# ── rules 09a-09b plots (add_river_depth, add_estuarine_depth) ────────────────


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


def plot_hydraulic_relations_with_estuarine(
    rivers_wgs: gpd.GeoDataFrame,
    output_path: str,
    L_e_m: float | None = None,
) -> None:
    """
    Two-panel hydraulic-relations plot that distinguishes depth-calculation origin.

    Left panel (log-log scatter, width vs depth):
        - Blue circles  — power-law only (rivdph_estuarine=False or no estuarine data)
        - Orange triangles — fully estuarine (Leuven 2018, blend_alpha=0)
        - Green squares — blend zone (linear mix, coloured by blend_alpha)

    Right panel (correlation, power-law depth vs final depth):
        Scatter for all estuarine/blend reaches showing rivdph_powerlaw (x)
        against the final rivdph (y), coloured by rivdph_blend_alpha
        (0=orange/estuarine, 1=blue/fluvial-end of blend). A 1:1 reference
        line is drawn. Only plotted when estuarine data are present.

    Falls back gracefully to the original two-panel discharge/dist_out layout
    when no estuarine columns are present (i.e. estuarine_depth.enabled=false),
    so the same function can serve both enabled and disabled modes.

    Args:
        rivers_wgs:  GeoDataFrame in EPSG:4326 from river_network_estuarine.gpkg.
        output_path: Destination PNG path.
        L_e_m:       Estuary length in metres (optional; drawn as a vertical
                     annotation in the correlation panel if supplied).
    """

    has_estuarine_cols = all(
        c in rivers_wgs.columns
        for c in ("rivdph_estuarine", "rivdph_powerlaw", "rivdph_blend_alpha")
    )
    estuarine_applied = has_estuarine_cols and rivers_wgs["rivdph_estuarine"].any()

    if not estuarine_applied:
        # Fallback: reproduce the original two-panel layout
        plot_hydraulic_relations(rivers_wgs=rivers_wgs, output_path=output_path)
        return

    cols_needed = [
        "width",
        "rivdph",
        "rivdph_powerlaw",
        "rivdph_estuarine",
        "rivdph_blend_alpha",
    ]
    df = rivers_wgs[cols_needed].copy().dropna(subset=["width", "rivdph"])
    df = df[(df["width"] > 0) & (df["rivdph"] > 0)]

    is_fluvial = ~df["rivdph_estuarine"].astype(bool)
    is_estuarine = df["rivdph_estuarine"].astype(bool) & (
        df["rivdph_blend_alpha"].fillna(0) == 0
    )
    is_blend = df["rivdph_estuarine"].astype(bool) & (
        df["rivdph_blend_alpha"].fillna(0) > 0
    )

    fig, (ax_width, ax_corr) = plt.subplots(1, 2, figsize=(14, 5))

    # ── left: width vs depth ─────────────────────────────────────────────
    kw = dict(s=18, linewidths=0.3, edgecolors="k")

    if is_fluvial.any():
        ax_width.scatter(
            df.loc[is_fluvial, "width"],
            df.loc[is_fluvial, "rivdph"],
            marker="o",
            color="steelblue",
            alpha=0.55,
            label="Power-law",
            **kw,
        )
    if is_estuarine.any():
        ax_width.scatter(
            df.loc[is_estuarine, "width"],
            df.loc[is_estuarine, "rivdph"],
            marker="^",
            color="darkorange",
            alpha=0.65,
            label="Estuarine (Leuven 2018)",
            **kw,
        )
    if is_blend.any():
        sc = ax_width.scatter(
            df.loc[is_blend, "width"],
            df.loc[is_blend, "rivdph"],
            marker="s",
            c=df.loc[is_blend, "rivdph_blend_alpha"],
            cmap="RdYlBu",
            vmin=0,
            vmax=1,
            alpha=0.75,
            label="Blend zone",
            **kw,
        )
        cb = fig.colorbar(sc, ax=ax_width, shrink=0.8)
        cb.set_label("Blend α  (0=estuarine, 1=fluvial)", fontsize=8)

    ax_width.set_xscale("log")
    ax_width.set_yscale("log")
    ax_width.set_xlabel("Channel width (m)")
    ax_width.set_ylabel("Hydraulic depth (m)")
    ax_width.set_title(f"Width vs depth by depth-model origin  ({len(df)} reaches)")
    ax_width.legend(fontsize=8, framealpha=0.9)
    ax_width.grid(True, which="both", alpha=0.3, linewidth=0.5)

    # ── right: correlation power-law vs final depth ──────────────────────
    df_est = df[df["rivdph_estuarine"].astype(bool)].dropna(subset=["rivdph_powerlaw"])
    df_est = df_est[df_est["rivdph_powerlaw"] > 0]

    if df_est.empty:
        ax_corr.text(
            0.5,
            0.5,
            "No estuarine reaches",
            ha="center",
            va="center",
            transform=ax_corr.transAxes,
            color="grey",
        )
    else:
        sc2 = ax_corr.scatter(
            df_est["rivdph_powerlaw"],
            df_est["rivdph"],
            c=df_est["rivdph_blend_alpha"].fillna(0),
            cmap="RdYlBu",
            vmin=0,
            vmax=1,
            s=22,
            alpha=0.7,
            linewidths=0.3,
            edgecolors="k",
        )
        cb2 = fig.colorbar(sc2, ax=ax_corr, shrink=0.8)
        cb2.set_label("Blend α  (0=estuarine, 1=fluvial)", fontsize=8)

        lims = [
            min(df_est["rivdph_powerlaw"].min(), df_est["rivdph"].min()) * 0.8,
            max(df_est["rivdph_powerlaw"].max(), df_est["rivdph"].max()) * 1.2,
        ]
        ax_corr.plot(lims, lims, "k--", linewidth=0.8, label="1 : 1")
        ax_corr.set_xlim(lims)
        ax_corr.set_ylim(lims)
        ax_corr.set_xscale("log")
        ax_corr.set_yscale("log")
        ax_corr.set_xlabel("Power-law depth (m)")
        ax_corr.set_ylabel("Final depth — estuarine / blend (m)")
        ax_corr.set_title(f"Power-law vs estuarine depth  ({len(df_est)} reaches)")
        ax_corr.legend(fontsize=8, framealpha=0.9)
        ax_corr.grid(True, which="both", alpha=0.3, linewidth=0.5)

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


# ── rule 13 plots (build_sfincs) ───────────────────────────────────────────────


def plot_refinement_zones(
    refinement_gdf: gpd.GeoDataFrame,
    domain_poly: Polygon,
    land_polygons_path: str,
    river_network_path: str,
    output_path: str,
    basin_id: str = "",
) -> None:
    """
    Quadtree refinement-zone diagnostic map: river and coastal buffer
    polygons colored by refinement level, with land outline, river network,
    and domain boundary overlaid.

    Args:
        refinement_gdf:     Output of ``quadtree_refinement.build_refinement_polygons``
                            (columns: geometry, refinement_level).
        domain_poly:        Domain polygon in WGS84 (from ``load_domain``).
        land_polygons_path: Path to the OSM land polygons geopackage.
        river_network_path: Path to the clipped river network geopackage.
        output_path:        Destination PNG path.
        basin_id:           Basin identifier for the plot title.
    """
    refinement_wgs = refinement_gdf.to_crs("EPSG:4326")
    bounds = refinement_wgs.total_bounds

    land, rivers, domain_gdf = _overlay_layers(
        "EPSG:4326", tuple(bounds), domain_poly, land_polygons_path, river_network_path
    )

    levels = sorted(refinement_gdf["refinement_level"].unique())
    cmap = plt.get_cmap("YlOrRd")
    norm = (
        mcolors.Normalize(vmin=min(levels), vmax=max(levels))
        if len(levels) > 1
        else None
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    _draw_overlays(ax, land, rivers, domain_gdf, zorder=1)
    for level in levels:
        color = cmap(norm(level)) if norm is not None else cmap(1.0)
        refinement_wgs[refinement_wgs["refinement_level"] == level].plot(
            ax=ax,
            facecolor=color,
            edgecolor="none",
            alpha=0.5,
            zorder=4,
            label=f"Refinement level {level}",
        )

    lon_min, lat_min, lon_max, lat_max = domain_poly.bounds
    _margin = max(lon_max - lon_min, lat_max - lat_min) * 0.3
    ax.set_xlim(lon_min - _margin, lon_max + _margin)
    ax.set_ylim(lat_min - _margin, lat_max + _margin)
    ax.set_aspect("equal")
    title = "Quadtree refinement zones"
    if basin_id:
        title = f"{title} | {basin_id}"
    ax.set_title(title)
    handles = _OVERLAY_LEGEND_HANDLES + [
        Patch(
            facecolor=cmap(norm(level) if norm is not None else 1.0),
            alpha=0.5,
            label=f"Refinement level {level}",
        )
        for level in levels
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.9, fontsize=8)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    _save(fig, output_path)


def reproject_max_for_plot(
    da: xr.DataArray,
    dst_crs: str = "EPSG:4326",
    max_px: int = _PLOT_MAX_PX,
    resampling: Resampling = Resampling.max,
) -> xr.DataArray:
    """
    Reproject ``da`` (in its own native/projected CRS) to ``dst_crs`` directly
    at a resolution coarse enough to keep the output spatial grid at or below
    ``max_px`` total pixels, aggregating with ``resampling`` (max by default)
    rather than reprojecting at native resolution and stride-decimating
    afterward.

    Reprojecting a fine-resolution SFINCS run raster (e.g. a quadtree run's
    subgrid, down to ~1.5 m pixels, or compute_max_inundation's own
    subgrid-resolution output) to WGS84 for a whole delta domain can produce
    tens of thousands of pixels per side. Two problems with doing that
    reproject at native resolution first and only downsampling afterward
    (this function's predecessor, _downsample_wgs_dataarray): (a) the
    reproject call itself still momentarily allocates the huge full-
    resolution intermediate array (observed: an 11627x19827 single-band
    array during reprojection, then a 4-channel float64 RGBA buffer during
    imshow -- ~6.9 GiB -- once handed to matplotlib), and (b) picking every
    Nth pixel post-hoc can alias away isolated peak values entirely, which
    matters for a MAX-inundation map. Reprojecting directly at the coarse
    target resolution with max-resampling avoids the large intermediate
    altogether and guarantees the true max within each output pixel survives
    the downsampling.

    ``calculate_default_transform`` is metadata-only (no pixel data is read
    or reprojected) and is used only to estimate the "natural" 1:1 output
    resolution, which is then coarsened (if needed) to hit the max_px budget.

    Non-spatial dimensions (e.g. ``time``) are reprojected slice-by-slice by
    rioxarray automatically and need no special handling here.
    """
    dst_transform, dst_width, dst_height = calculate_default_transform(
        da.rio.crs, dst_crs, da.rio.width, da.rio.height, *da.rio.bounds()
    )
    total_px = dst_width * dst_height
    factor = math.sqrt(total_px / max_px) if total_px > max_px else 1.0
    # dst_transform.a = the "natural" (1:1) pixel size in dst_crs units that
    # calculate_default_transform picked; scale it up by `factor` to hit the
    # max_px budget.
    target_res = dst_transform.a * factor
    return da.rio.reproject(dst_crs, resolution=target_res, resampling=resampling)


# ── run-output diagnostic plots (rules 14-15: run_spinup, sanity_checks) ──────


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
    da_wgs = reproject_max_for_plot(da_hmax.squeeze())
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


def _mesh_overlay_setup(
    da_mesh: xu.UgridDataArray,
    domain_poly: Polygon,
    land_polygons_path: str,
    river_network_path: str,
):
    """
    Common setup for mesh-native (quadtree) animations: the mesh's own native
    CRS/bounds, and the land/river/domain overlay layers reprojected to that
    CRS via ``_overlay_layers``.

    Plots the mesh directly in its native (projected, e.g. UTM) CRS rather
    than reprojecting to WGS84 like every raster-based plot in this module —
    xugrid has no simple ``.rio.reproject()`` equivalent for a mesh's
    topology (unlike a raster, reprojecting would mean rebuilding every cell
    polygon in the new CRS), so it's the overlays that get reprojected here
    instead, onto the mesh's own CRS.
    """
    crs = da_mesh.ugrid.grid.crs
    bounds = da_mesh.ugrid.total_bounds  # (xmin, ymin, xmax, ymax), native CRS
    land, rivers, domain_gdf = _overlay_layers(
        crs, bounds, domain_poly, land_polygons_path, river_network_path
    )
    return crs, bounds, land, rivers, domain_gdf


def animate_flood_progression(
    da_h: xr.DataArray | xu.UgridDataArray,
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

    For a REGULAR grid, the land outline, river network, and domain boundary
    are drawn once as a static background; an imshow layer of instantaneous
    water depth (already masked to non-water land-use areas) is then updated
    for each time step.

    For a QUADTREE run, ``da_h`` arrives as a mesh-native ``xu.UgridDataArray``
    (unmasked — see ``postprocessing.compute_flood_progression``) and is
    rendered directly as mesh cell polygons (``.ugrid.plot()``, a
    ``matplotlib.collections.PolyCollection``) updated per frame via
    ``set_array()`` — this never rasterizes the mesh, which for a large
    quadtree domain is what previously risked exhausting memory (see
    ``src.postprocessing``'s ``_coarsen_for_memory`` / mosaic history).

    Args:
        da_h:               Instantaneous land-surface water depth with a
                            ``time`` dimension; NaN = dry / water / outside
                            domain (regular-grid case only — the quadtree case
                            is unmasked). Regular-grid arrays must carry CRS
                            metadata (``rio.crs``); quadtree arrays must carry
                            mesh CRS metadata (``ugrid.grid.crs`` — see
                            ``postprocessing._ensure_ugrid_crs``).
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

    if isinstance(da_h, xu.UgridDataArray):
        _, bounds, land, rivers, domain_gdf = _mesh_overlay_setup(
            da_h, domain_poly, land_polygons_path, river_network_path
        )

        vals = da_h.values
        valid = vals[~np.isnan(vals)]
        vmax = float(np.percentile(valid, 99)) if len(valid) > 0 else 1.0
        vmax = max(vmax, 0.01)

        fig, ax = plt.subplots(figsize=(10, 8))
        _draw_overlays(ax, land, rivers, domain_gdf, zorder=1)
        coll = da_h.isel(time=0).ugrid.plot(
            ax=ax, cmap="Blues", vmin=0, vmax=vmax, zorder=2
        )
        cb = fig.colorbar(coll, ax=ax, fraction=0.03, pad=0.04, extend="max")
        cb.set_label("Water depth (m)")

        xmin, ymin, xmax, ymax = bounds
        _margin = max(xmax - xmin, ymax - ymin) * 0.3
        ax.set_xlim(xmin - _margin, xmax + _margin)
        ax.set_ylim(ymin - _margin, ymax + _margin)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.legend(
            handles=_OVERLAY_LEGEND_HANDLES,
            loc="lower right",
            framealpha=0.9,
            fontsize=8,
        )

        title_prefix = "Flood progression"
        if run_label:
            title_prefix = f"{title_prefix} — {run_label}"
        if basin_id:
            title_prefix = f"{title_prefix} | {basin_id}"
        title_artist = ax.set_title(title_prefix)

        times = da_h["time"].values
        n_frames = da_h.sizes["time"]

        def _update_mesh(i):
            coll.set_array(da_h.isel(time=i).values.ravel())
            title_artist.set_text(
                f"{title_prefix}\n{np.datetime_as_string(times[i], unit='m')}"
            )
            return coll, title_artist

        anim = FuncAnimation(fig, _update_mesh, frames=n_frames, blit=False)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(out_path), writer=FFMpegWriter(fps=fps))
        plt.close(fig)
        log.info(f"Written: {output_path}")
        return

    # Reproject from the model's metric UTM grid to EPSG:4326 so this animation
    # shares the same lon/lat reference frame as every other diagnostic plot.
    da_wgs = reproject_max_for_plot(da_h)
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


# ── rule 05a plots (get_elevation datum-correction diagnostics) ───────────────


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
    title: str = "Vertical datum correction\n(EGM2008 → GOCO06s geoid)",
    colorbar_label: str = "Δ elevation (m)  [GOCO06s − EGM2008]",
    vmax: float | None = None,
    symmetric: bool = True,
) -> None:
    """
    Diverging raster plot of a vertical datum correction delta (corrected − original).

    Generic for any single-step correction applied in 05a_get_elevation.py —
    the coastal DEM's geoid offset (EGM2008 → GOCO06s) or GEBCO's MDT
    subtraction (raw → GOCO06s); ``title``/``colorbar_label`` distinguish
    which. NaN pixels (including any caller has masked out, e.g. unchanged
    cells) are fully transparent, letting the grey land-polygon background
    show through. Axes are in projected UTM metres so the pattern can be
    compared with the elevation and zsini maps.

    By default (``symmetric=True``) the color scale is a symmetric ±vmax
    range, clipped to the 95th percentile of |delta| unless ``vmax`` is
    given explicitly (datum corrections are typically smooth and two-signed,
    so this keeps the bulk of the map well-contrasted), drawn with the
    diverging "RdBu_r" colormap. Set ``symmetric=False`` to instead use the
    data's own (possibly one-signed) min/max directly -- appropriate for a
    correction that only ever pushes one direction, where a symmetric ±vmax
    scale would waste half its range. ``vmax`` is ignored when
    ``symmetric=False``; the
    colormap also switches to a sequential "Reds"/"Reds_r" in that case,
    since RdBu_r's white midpoint would no longer fall at delta=0 once the
    range matches a skewed/one-signed data extent (small values would render
    almost fully saturated instead of faint).
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
    if symmetric:
        if vmax is None:
            vmax = float(np.percentile(np.abs(valid), 95)) if len(valid) > 0 else 1.0
        vmax = max(vmax, 0.01)
        vmin, vmax = -vmax, vmax
        extend = "both"
        cmap = "RdBu_r"
    else:
        if len(valid) > 0:
            vmin, vmax = float(valid.min()), float(valid.max())
        else:
            vmin, vmax = -0.01, 0.01
        if vmin == vmax:
            vmin, vmax = vmin - 0.01, vmax + 0.01
        extend = "neither"
        # One-signed data (e.g. a correction that only ever pushes one direction):
        # RdBu_r's white midpoint would no longer fall at zero once vmin/vmax match
        # the (skewed) data range, so a near-zero value would render almost fully
        # saturated instead of faint. A sequential map fixes that -- oriented so the
        # bound closer to zero (the *smaller*-magnitude end) is light and the more
        # extreme bound is dark, regardless of which sign dominates.
        cmap = "Reds_r" if abs(vmin) >= abs(vmax) else "Reds"

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
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        origin="upper",
        zorder=2,
    )
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, extend=extend)
    cb.set_label(colorbar_label)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    _save(fig, output_path)


# ── rule 15 plots (sanity_checks) ─────────────────────────────────────────────


def plot_inundation_check(
    da_hmax: xr.DataArray,
    threshold_m: float,
    n_flooded: int,
    n_land: int,
    land_polygons_path: str,
    river_network_path: str,
    output_path: str,
    basin_id: str = "",
    water_bodies_path: str | None = None,
    run_label: str = "baseline spinup",
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
        run_label:          Run/scenario label for the plot title (e.g.
                            "baseline spinup" or "event") — this function is
                            shared by rule 15 (spin-up sanity checks) and
                            rule 16 (main event run).
    """
    frac = n_flooded / n_land if n_land > 0 else 0.0
    n_dry = n_land - n_flooded

    # Reproject from the model's metric UTM grid to EPSG:4326.  Use explicit
    # imshow with extent rather than da.plot() to guarantee north-up orientation
    # regardless of how downscale_floodmap orders the spatial dimensions.
    da_wgs = reproject_max_for_plot(da_hmax.squeeze())
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
    if water_bodies_path is not None:
        with rasterio.open(water_bodies_path) as _wb_src:
            _wb_arr = _wb_src.read(1)
            _wb_mask = (_wb_arr == 200).astype(np.uint8)
            if _wb_mask.any():
                _src_crs = _wb_src.crs.to_wkt()
                _geoms = [
                    _shape(_transform_geom(_src_crs, "EPSG:4326", geom))
                    for geom, val in _rio_shapes(
                        _wb_mask, mask=_wb_mask, transform=_wb_src.transform
                    )
                    if val == 1
                ]
                if _geoms:
                    gpd.GeoDataFrame(geometry=_geoms, crs="EPSG:4326").plot(
                        ax=ax1, color="white", edgecolor="none", zorder=1.5
                    )
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

    suptitle = f"Sanity check — inundation ratio ({run_label})"
    if basin_id:
        suptitle = f"{suptitle} | {basin_id}"
    fig.suptitle(f"{suptitle}\nThreshold: {threshold_m} m", fontsize=12)
    fig.tight_layout()
    _save(fig, output_path)
