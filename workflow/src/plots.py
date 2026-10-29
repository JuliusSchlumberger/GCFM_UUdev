"""Summary plot functions for all workflow steps.

Every map's land background (``land_polygons_path``) is a vectorized WGS84
land mask, land use != 200 (the pipeline's open-sea code; see src.landuse).
From rule grid_align_landuse (09) on it is traced on the SFINCS grid
({basin}_land_mask_on_grid.gpkg); rules 03-09 and the analysis scripts use the native land-use
mask ({basin}_land_polygons.gpkg). No OSM data or web map tiles are used.
"""

from __future__ import annotations

import errno
import logging
import math
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401  — registers the .rio accessor used for reprojection
import xarray as xr
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from src.geometry import pick_utm_crs
from src.grid import footprint

matplotlib.use("Agg")

log = logging.getLogger(__name__)

_PLOT_DPI = 100
_PLOT_MAX_PX = 2_000_000  # downsample rasters larger than this before rendering
# One fixed depth scale for every inundation-depth map, so maps compare across
# scenarios and rounds; deeper cells saturate (extend="max").
_INUNDATION_DEPTH_CBAR_MAX_M = 5.0

# Blue below 0 m, brown above.
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
    # ESA WorldCover only: 10 stands in for LC100 111-126, 95 has no LC100 class.
    10: "Tree cover",
    95: "Mangroves",
    20: "Shrubland",
    30: "Herbaceous vegetation",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow / ice",
    80: "Inland water",
    90: "Herbaceous wetland",
    100: "Moss / lichen",
    111: "Closed forest, evergreen needleleaf",
    112: "Closed forest, evergreen broadleaf",
    113: "Closed forest, deciduous needleleaf",
    114: "Closed forest, deciduous broadleaf",
    115: "Closed forest, mixed",
    116: "Closed forest, unspecified",
    121: "Open forest, evergreen needleleaf",
    122: "Open forest, evergreen broadleaf",
    123: "Open forest, deciduous needleleaf",
    124: "Open forest, deciduous broadleaf",
    125: "Open forest, mixed",
    126: "Open forest, unspecified",
    200: "Sea",
}

_LC_COLORS: dict[int, str] = {
    0: "#808080",
    10: "#006400",  # WorldCover's own legend colours
    95: "#009678",
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


def imshow_on_grid(ax, data: np.ndarray, transform, **kwargs):
    """Draw a raster in its own CRS at its true position for any affine transform.

    The model grid is south-up, which imshow(extent=...) does not handle;
    the raster's affine is handed to matplotlib instead, so
    vector overlays in the same CRS line up. Sets axes limits to the footprint.
    """
    from matplotlib.transforms import Affine2D

    rows, cols = data.shape[:2]
    kwargs.setdefault("interpolation", "nearest")
    im = ax.imshow(data, extent=(0, cols, rows, 0), origin="upper", **kwargs)
    im.set_transform(
        Affine2D.from_values(
            transform.a, transform.d, transform.b, transform.e, transform.c, transform.f
        )
        + ax.transData
    )
    left, bottom, right, top = footprint(transform, (rows, cols)).bounds
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    return im


def _to_wgs84_grid(
    data: np.ndarray,
    transform,
    src_crs,
    dst_bounds: tuple[float, float, float, float] | None = None,
):
    # Nearest neighbour keeps exact source values. dst_bounds (EPSG:4326) fills
    # exactly that box instead of rasterio's padded bounds. Returns float32 data
    # (NaN = nodata) and the imshow extent (left, right, bottom, top) in degrees.
    from rasterio.enums import Resampling
    from rasterio.transform import array_bounds
    from rasterio.transform import from_bounds as rio_from_bounds
    from rasterio.warp import calculate_default_transform
    from rasterio.warp import reproject as rio_reproject

    h, w = data.shape
    src_bounds = footprint(transform, (h, w)).bounds
    # Pixel count from calculate_default_transform even when dst_bounds is given.
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
    """Read band 1, reproject to EPSG:4326 (nearest), nodata to NaN, downsample.

    Pass the domain WGS84 bbox as dst_bounds so the raster fills it exactly.
    Returns (data, extent) with extent (left, right, bottom, top) for imshow.
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


def add_land_background_to_geoaxes(ax, land_polygons_path: str, model_crs) -> None:
    """Draw the land mask underneath a cartopy GeoAxes in the model CRS.

    For the axes returned by hydromt_sfincs plot_basemap; replaces web map tiles.
    """
    land = gpd.read_file(land_polygons_path).to_crs(model_crs)
    if not land.empty:
        ax.add_geometries(
            land.geometry,
            crs=ax.projection,
            facecolor="#d9d9d9",
            edgecolor="#aaaaaa",
            linewidth=0.3,
            zorder=0,
        )


def map_background(
    ax,
    bbox_poly: Polygon,
    land_polygons_path: str,
    margin_frac: float = 0.3,
) -> None:
    """Draw the land mask and the WGS84 domain bbox.

    margin_frac is a fraction of the larger bbox span, added on each side.
    Open sea is not in the land mask, so it shows as the plain background.
    """
    lon_min, lat_min, lon_max, lat_max = bbox_poly.bounds
    margin = max(lon_max - lon_min, lat_max - lat_min) * margin_frac
    xmin, ymin = lon_min - margin, lat_min - margin
    xmax, ymax = lon_max + margin, lat_max + margin

    land = gpd.read_file(
        land_polygons_path, bbox=(xmin, ymin, xmax, ymax), engine="pyogrio"
    )
    if not land.empty:
        land.plot(ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=1)

    bx, by = bbox_poly.exterior.xy
    ax.plot(bx, by, color="black", linewidth=1.5, zorder=10)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.grid(True, alpha=0.3, linewidth=0.5)


_SAVE_ATTEMPTS = 5
_SAVE_RETRY_WAIT_S = 1.0
# Errnos of a transient Windows file lock; any other OSError (ENOSPC, ...) is final.
_LOCK_ERRNOS = (errno.EINVAL, errno.EACCES)


def save_figure(fig, output_path, *, close: bool = True, **savefig_kwargs) -> None:
    """Save fig, retrying while another process briefly locks the destination file.

    On Windows a PNG held open by antivirus, a previewer or Explorer's
    thumbnailer fails with OSError EINVAL ("Invalid argument"), not
    PermissionError; the locks are short, so spaced retries succeed. Creates
    the parent directory and closes the figure unless close=False.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, _SAVE_ATTEMPTS + 1):
        try:
            fig.savefig(path, **savefig_kwargs)
            break
        except OSError as exc:
            if exc.errno not in _LOCK_ERRNOS:
                if close:
                    plt.close(fig)
                raise
            if attempt == _SAVE_ATTEMPTS:
                if close:
                    plt.close(fig)
                raise OSError(
                    f"Could not write {path} after {_SAVE_ATTEMPTS} attempts "
                    f"({exc}). If this is Windows' 'Invalid argument', another "
                    f"process (antivirus, an image viewer, Explorer's thumbnailer) "
                    f"is holding the file open."
                ) from exc
            wait = _SAVE_RETRY_WAIT_S * attempt
            log.warning(
                f"Writing {path.name} failed ({exc.__class__.__name__}: {exc}); "
                f"retrying in {wait:.0f}s ({attempt}/{_SAVE_ATTEMPTS - 1})"
            )
            time.sleep(wait)
    if close:
        plt.close(fig)


def _save(fig, output_path: str) -> None:
    save_figure(fig, output_path, dpi=_PLOT_DPI, bbox_inches="tight")
    log.info(f"Written: {output_path}")


def plot_landuse(
    landuse_path: str,
    bbox_poly: Polygon,
    land_polygons_path: str,
    output_path: str,
) -> None:
    """Plot the categorical land-use map with class colours and legend.

    The title names the source from the raster's landuse_source tag (written
    by rule prepare_landuse, 02b); LC100 and ESA WorldCover share codes 20-100
    but not 10/95/111-126.
    """
    data, extent = read_raster_reprojected_for_plot(
        landuse_path, dst_bounds=bbox_poly.bounds
    )
    with rasterio.open(landuse_path) as _src:
        _source = _src.tags().get("landuse_source", "")
    _source_label = {
        "copernicus_lc100": "Copernicus LC100, 2019",
        "esa_worldcover": "ESA WorldCover 2021 v200, 10 m",
    }.get(_source, _source)
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
    map_background(ax, bbox_poly, land_polygons_path)
    ax.imshow(data_idx, cmap=cmap, norm=norm, extent=extent, origin="upper", zorder=2)
    legend_patches = [
        Patch(color=_LC_COLORS.get(c, "#808080"), label=_LC_NAMES.get(c, str(c)))
        for c in present
    ]
    ax.legend(
        handles=legend_patches, loc="lower right", fontsize=7, framealpha=0.9, ncol=2
    )
    ax.set_title("Land use" + (f" ({_source_label})" if _source_label else ""))
    _save(fig, output_path)


# Above this many distinct Manning values the roughness map goes continuous
# (a reclassified class raster holds ~10; an area-averaged one, tens of thousands).
_ROUGHNESS_MAX_DISCRETE_CLASSES = 24


def plot_roughness(
    roughness_path: str,
    bbox_poly: Polygon,
    land_polygons_path: str,
    output_path: str,
) -> None:
    """Plot the Manning's n roughness map.

    One colour per distinct value for a few classes; area-averaged n
    (src.landuse.aggregate_manning) has tens of thousands of values, so it gets
    a continuous 2nd-98th percentile scale (a per-value BoundaryNorm overflows
    its int16 index there).
    """
    data, extent = read_raster_reprojected_for_plot(
        roughness_path, dst_bounds=bbox_poly.bounds
    )
    valid = data[~np.isnan(data)]
    unique_n = sorted({round(float(v), 6) for v in np.unique(valid)})
    discrete = len(unique_n) <= _ROUGHNESS_MAX_DISCRETE_CLASSES

    fig, ax = plt.subplots(figsize=(9, 7))
    map_background(ax, bbox_poly, land_polygons_path)

    if discrete:
        palette = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(unique_n)))
        cmap = mcolors.ListedColormap(palette)
        norm = mcolors.BoundaryNorm(
            boundaries=range(len(unique_n) + 1), ncolors=len(unique_n)
        )
        plot_data = np.full(data.shape, np.nan)
        for i, n_val in enumerate(unique_n):
            plot_data[np.isclose(data, n_val, atol=1e-5)] = i
    else:
        cmap, plot_data = plt.get_cmap("viridis"), data
        lo, hi = np.percentile(valid, [2, 98]) if valid.size else (0.0, 1.0)
        norm = mcolors.Normalize(vmin=float(lo), vmax=float(max(hi, lo + 1e-6)))

    im = ax.imshow(
        plot_data, cmap=cmap, norm=norm, extent=extent, origin="upper", zorder=2
    )
    cb = plt.colorbar(
        im, ax=ax, fraction=0.03, pad=0.04, extend="neither" if discrete else "both"
    )
    if discrete:
        cb.set_ticks([i + 0.5 for i in range(len(unique_n))])
        cb.set_ticklabels([f"{n:.4f}" for n in unique_n])
    cb.set_label("Manning's n")
    ax.set_title(
        "Surface roughness (Manning's n)"
        + ("" if discrete else f"  —  area-averaged, {len(unique_n):,} distinct values")
    )
    _save(fig, output_path)


def plot_elevation_merged(
    merged_path: str,
    bbox_poly: Polygon,
    land_polygons_path: str,
    output_path: str,
    title_str: str,
) -> None:
    """Plot merged FathomDEM+GEBCO elevation on a diverging scale centred at 0 m.

    The scale is fixed at -10 m / 30 m so basins compare; values beyond it
    take the end colours.
    """
    data, extent = read_raster_reprojected_for_plot(
        merged_path, dst_bounds=bbox_poly.bounds
    )

    vmin, vmax = -10.0, 30.0

    fig, ax = plt.subplots(figsize=(9, 7))
    map_background(ax, bbox_poly, land_polygons_path)
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


def plot_sea_mask(
    sea_mask_path: str,
    bbox_poly: Polygon,
    land_polygons_path: str,
    output_path: str,
) -> None:
    """Plot the sea / inland-water mask (landuse 200; nodata = land or outside domain).

    This is a land/sea mask, not the initial water level; rule build_sfincs
    (13) turns it into zsini.tif (sea cells = calm_sea_level_m).
    """
    data, extent = read_raster_reprojected_for_plot(
        sea_mask_path, dst_bounds=bbox_poly.bounds
    )
    water_mask = ~np.isnan(data)
    n_water = int(water_mask.sum())

    fig, ax = plt.subplots(figsize=(9, 7))
    map_background(ax, bbox_poly, land_polygons_path)
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
            Patch(color="#3860D0", label=f"Sea / inland water — {n_water:,} px"),
        ],
        loc="lower right",
        framealpha=0.9,
    )
    ax.set_title("Sea / inland-water mask")
    _save(fig, output_path)


def plot_protection_levels(
    delta_polygon_wgs84: Polygon,
    geogunit_raster_path: str,
    flopros_df,
    summary: dict,
    land_polygons_path: str,
    output_path: str,
    margin_frac: float = 0.5,
) -> None:
    """Plot riverine and coastal FLOPROS design protection RP (years) around a delta.

    Shows geounits overlapping the delta bbox plus margin; titles give the
    dominant geounit's resolved (post-fallback/cap) RP from summary
    (protection_levels.identify_dominant_protection). Only a window of the
    wri_geogunit_107 raster is read.
    """
    import pandas as pd
    import shapely
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
        # lat/lon are both ascending in wri_geogunit_107.
        sel = ds["Geogunits"].sel(
            lat=slice(window_bounds[1], window_bounds[3]),
            lon=slice(window_bounds[0], window_bounds[2]),
        )
        arr = sel.values.astype(np.float64)
        lon_vals = sel["lon"].values
        lat_vals = sel["lat"].values

    # Row 0 is southernmost, hence origin="lower".
    extent = (
        float(lon_vals.min()),
        float(lon_vals.max()),
        float(lat_vals.min()),
        float(lat_vals.max()),
    )

    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    inside = shapely.contains_xy(delta_polygon_wgs84, lon_grid, lat_grid)
    # xarray decodes _FillValue to NaN, so isfinite drops fill pixels.
    valid = inside & np.isfinite(arr)

    max_rp_yr_for_scale = max(
        100.0,
        summary.get("river_protection_rp_yr") or 0.0,
        summary.get("surge_protection_rp_yr") or 0.0,
    )
    norm = mcolors.LogNorm(vmin=1.0, vmax=max_rp_yr_for_scale)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#d0d0d0")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    for ax, hazard, key in zip(
        axes,
        ("Riverine", "Coastal"),
        ("river_protection_rp_yr", "surge_protection_rp_yr"),
    ):
        map_background(ax, bbox_poly, land_polygons_path)
        values_by_id = flopros_df[hazard].to_dict()
        rp_yr_arr = np.full(arr.shape, np.nan, dtype=float)
        ids_in_window = np.unique(arr[valid].astype(np.int64))
        for gid in ids_in_window:
            val = values_by_id.get(int(gid))
            if val is not None and not pd.isna(val):
                rp_yr_arr[valid & (arr.astype(np.int64) == gid)] = float(val)

        im = ax.imshow(
            rp_yr_arr, cmap=cmap, norm=norm, extent=extent, origin="lower", zorder=2
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


def plot_river_network(
    river_path: str,
    bbox_poly: Polygon,
    land_polygons_path: str,
    output_path: str,
) -> None:
    """Plot the clipped river network over the land background and domain bbox."""
    rivers = gpd.read_file(river_path)
    if rivers.crs and rivers.crs.to_epsg() != 4326:
        rivers = rivers.to_crs("EPSG:4326")

    fig, ax = plt.subplots(figsize=(9, 7))
    map_background(ax, bbox_poly, land_polygons_path)
    if not rivers.empty:
        rivers.plot(ax=ax, color="steelblue", linewidth=0.8, zorder=5)
    ax.set_title("River network (clipped to model domain)")
    _save(fig, output_path)


def plot_domain_map(
    bbox_poly: Polygon,
    river_gdf: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    crossings: gpd.GeoDataFrame,
    has_glofas: np.ndarray,
    land_polygons_path: str,
    output_path: str,
) -> None:
    """Plot the model domain, river network, surge stations and river crossings.

    Crossing points: green active; inactive as red crosses (enters and visible
    on the grid but no GloFAS/EVA), purple (enters but narrower than the grid)
    or orange (no downstream reach inside the domain). Without rule 07d's filter
    columns every inactive crossing is red.
    """
    lon_min, lat_min, lon_max, lat_max = bbox_poly.bounds
    margin = max(lon_max - lon_min, lat_max - lat_min) * 0.3
    map_bounds = (
        lon_min - margin,
        lat_min - margin,
        lon_max + margin,
        lat_max + margin,
    )

    land = gpd.read_file(land_polygons_path, bbox=map_bounds)

    fig, ax = plt.subplots(figsize=(10, 8))
    if not land.empty:
        land.plot(ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=1)
    river_gdf.plot(ax=ax, color="steelblue", linewidth=0.8, zorder=2)

    bx, by = bbox_poly.exterior.xy
    ax.plot(bx, by, color="black", linewidth=2, zorder=3)
    stations.plot(
        ax=ax, color="darkorange", markersize=35, marker="^", zorder=4, alpha=0.8
    )

    # Exclusion reasons come from filter columns written by rule 07d; without
    # them all inactive crossings share one category.
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
    rp_yr: int | str | None,
    output_path: str,
) -> None:
    """Plot surge water level (left) and river discharge (right) forcing timeseries.

    The surge boundary is never reduced by the FLOPROS coastal protection
    level (a weir models it). When river_ds carries 'discharge_uncorrected' /
    'protection_discharge' (rule 07d does not write them), the river panel shows
    the undefended hydrograph, the constant protection discharge and the
    effective hydrograph; otherwise one series per crossing.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    river_corrected = "discharge_uncorrected" in river_ds

    t_surge = surge_ds["time"].values
    water_level_m = surge_ds["water_level"].values
    for i in range(water_level_m.shape[0]):
        ax1.plot(t_surge, water_level_m[i], color="steelblue", alpha=0.5, linewidth=0.8)

    ax1.set_xlabel("Time (hours since start)")
    ax1.set_ylabel("Water level (m, GOCO06s frame)")
    rp_label = (
        f"surge_rp_yr={rp_yr}"
        if rp_yr is None or isinstance(rp_yr, str)
        else f"RP{rp_yr}"
    )
    ax1.set_title(f"Surge forcing — {rp_label} ({water_level_m.shape[0]} stations)")
    ax1.grid(True, alpha=0.3)

    max_box = dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.85)
    if water_level_m.size:
        ax1.text(
            0.01,
            0.97,
            f"Max WL = {np.nanmax(water_level_m):.2f} m",
            transform=ax1.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox=max_box,
        )

    # Correction chain rp_raw → +MDT → peak(GOCO06s). water_level/storm_tide_at_rp_m are
    # MDT-only (src.surge.apply_slr_fingerprint); SLR is added at scenario
    # build time as slr_fingerprint × slr_m, so only the dimensionless
    # fingerprint is shown. The FLOPROS coastal level only floors the weir
    # crest (rule 12a) and is never netted out of this series.
    has_raw = "storm_tide_at_rp_raw_m" in surge_ds
    has_mdt = "mdt" in surge_ds
    if has_raw and has_mdt:
        rp_levels = surge_ds[
            "storm_tide_at_rp_m"
        ].values  # peak in GOCO06s (= rp_raw + MDT, MDT-only)
        rp_levels_raw = surge_ds[
            "storm_tide_at_rp_raw_m"
        ].values  # raw COAST-RP (local MSL)
        mdts = surge_ds["mdt"].values
        fingerprint_arr = (
            surge_ds["slr_fingerprint"].values
            if "slr_fingerprint" in surge_ds
            else np.full(len(rp_levels), np.nan)
        )

        header = "Stn  rp_raw    +MDT  peak(GOCO06s) SLR fingerprint"
        rows = [header, "─" * len(header)]
        for i, (rl_raw, mdt_i, fp_i, rl) in enumerate(
            zip(rp_levels_raw, mdts, fingerprint_arr, rp_levels)
        ):
            rows.append(
                f" {i + 1:2d}  {rl_raw:+7.3f}  {mdt_i:+6.3f}    {rl:+7.3f}       {fp_i:5.2f}"
            )
        rows.append("─" * len(header))
        bm = (
            float(surge_ds["calm_sea_level_m"].values)
            if "calm_sea_level_m" in surge_ds
            else float(np.mean(mdts))
        )
        rows.append(f"  calm_sea_level_m (MWL (=0) + MDT, MDT-only): {bm:+.4f} m")
        rows.append(
            "  (SLR added downstream: slr_fingerprint × configured slr_m target)"
        )

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
    if active.any():
        ax2.text(
            0.01,
            0.97,
            f"Max Q river = {np.nanmax(q[active]):,.0f} m³/s",
            transform=ax2.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox=max_box,
        )

    fig.tight_layout()
    _save(fig, output_path)


def plot_applied_forcing(
    surge_times: list,
    water_level_m: np.ndarray,
    river_times: list,
    discharge_m3s: np.ndarray,
    surge_rp_label: str,
    tstart_event,
    spinup_days: float,
    suptitle: str,
    output_path: str | Path,
) -> None:
    """Plot the boundary forcing a scenario model applies: water level and discharge.

    water_level_m is (station, time), discharge_m3s (crossing, time) for the
    active crossings. tstart_event (datetime or None) draws the event start.
    """
    import matplotlib.dates as mdates

    fig, (ax_wl, ax_q) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for i in range(water_level_m.shape[0]):
        ax_wl.plot(surge_times, water_level_m[i, :], linewidth=0.8, alpha=0.7)
    ax_wl.set_ylabel("Water level (m+ref)")
    ax_wl.set_title(f"Surge boundary — water level, {surge_rp_label} (all stations)")
    ax_wl.grid(True, alpha=0.3)

    n_cross = discharge_m3s.shape[0]
    for i in range(n_cross):
        ax_q.plot(river_times, discharge_m3s[i, :], linewidth=0.8, alpha=0.7)
    ax_q.set_ylabel("Discharge (m³/s)")
    ax_q.set_title(f"River discharge ({n_cross} active GloFAS crossing(s))")
    ax_q.grid(True, alpha=0.3)

    # Peak boxes: max over all stations/crossings and the full timeseries; the
    # river boundary is a discharge boundary, so its box reports max Q.
    max_box = dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.85)
    if water_level_m.size:
        ax_wl.text(
            0.01,
            0.97,
            f"Max WL = {np.nanmax(water_level_m):.2f} m",
            transform=ax_wl.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox=max_box,
        )
    if discharge_m3s.size:
        ax_q.text(
            0.01,
            0.97,
            f"Max Q river = {np.nanmax(discharge_m3s):,.0f} m³/s",
            transform=ax_q.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox=max_box,
        )

    if tstart_event is not None:
        for ax in (ax_wl, ax_q):
            ax.axvline(
                tstart_event,
                color="black",
                linewidth=1.2,
                linestyle="--",
                label=f"event start (after {spinup_days}d spinup)",
            )
            ax.legend(fontsize=8, loc="upper right")

    locator = mdates.AutoDateLocator()
    ax_q.xaxis.set_major_locator(locator)
    ax_q.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=150, bbox_inches="tight")


def plot_surge_corrections(
    stations: gpd.GeoDataFrame,
    output_path: str,
) -> None:
    """Plot the MDT correction of COAST-RP storm-tide levels per station.

    surge_forcing.nc levels are MDT-only; SLR is applied per scenario as the
    dimensionless slr_fingerprint × slr_m, so only the fingerprint is
    annotated. The FLOPROS coastal level (weir crest floor, rule 12a) has no bar.

    Left panel — three sub-bars per station, each anchored at 0 m (local MSL):
      1. storm_tide_at_rp_raw_m — steelblue
      2. +MDT correction — darkorange; extends above 0 when MDT > 0, below 0 when MDT < 0
      3. Net peak (MDT-only) = rp_raw + MDT — navy

    Right panel: station locations coloured by the +MDT correction magnitude.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    n = len(stations)
    x = np.arange(n, dtype=float)

    raw = stations["storm_tide_at_rp_raw_m"].values
    mdt = (
        stations["mdt"].fillna(0.0).values if "mdt" in stations.columns else np.zeros(n)
    )
    fingerprint = (
        stations["slr_fingerprint"].values
        if "slr_fingerprint" in stations.columns
        else np.full(n, np.nan)
    )

    mdt_corr = mdt  # MSL -> GOCO06s adds MDT: positive (above 0) when MDT > 0

    # Each component gets its own column so tiny MDT bars stay visible.
    n_bars = 3
    w = min(0.20, 0.85 / n_bars)
    g = 0.03
    half = (n_bars - 1) / 2.0 * (w + g)
    centers = np.array([i * (w + g) for i in range(n_bars)]) - half

    x_raw = x + centers[0]
    x_mdt = x + centers[1]
    x_net = x + centers[2]

    ax1.bar(
        x_raw,
        raw,
        width=w,
        color="steelblue",
        label="storm_tide_at_rp_raw_m  (COAST-RP, local MSL)",
    )

    ax1.bar(
        x_mdt,
        mdt_corr,
        width=w,
        color="darkorange",
        label="+MDT  (local MSL → GOCO06s)",
    )

    net_peak = raw + mdt
    ax1.bar(
        x_net,
        net_peak,
        width=w,
        color="navy",
        alpha=0.75,
        label="Net peak  (storm_tide_at_rp_raw_m + MDT, MDT-only)",
    )

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
    _annotate_bar(ax1, x_mdt, mdt_corr, np.zeros(n))
    _annotate_bar(ax1, x_net, net_peak, np.zeros(n))

    ax1.axhline(
        0.0, color="black", linewidth=0.9, linestyle="--", label="0 m  (local MSL)"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(i + 1) for i in range(n)])
    ax1.set_xlabel("Station")
    ax1.set_ylabel("Water level relative to local MSL (m)")
    ax1.set_title(
        "Surge correction decomposition per station (MDT-only; SLR applied "
        "downstream)\n(bars left→right: rp_raw | +MDT | net peak)"
    )
    ax1.legend(fontsize=7, framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis="y")

    sc = ax2.scatter(
        stations.geometry.x,
        stations.geometry.y,
        c=mdt_corr,
        cmap="coolwarm",
        s=60,
        edgecolor="k",
    )
    fig.colorbar(sc, ax=ax2, label="+MDT correction (m)")
    if np.any(~np.isnan(fingerprint)):
        for xi, yi, fp in zip(stations.geometry.x, stations.geometry.y, fingerprint):
            if not np.isnan(fp):
                ax2.annotate(
                    f"×{fp:.2f}",
                    (xi, yi),
                    fontsize=6,
                    xytext=(3, 3),
                    textcoords="offset points",
                )
    ax2.set_xlabel("Longitude (°)")
    ax2.set_ylabel("Latitude (°)")
    ax2.set_title(
        "MDT correction per station (+mdt, m)\n"
        "(annotations: SLR fingerprint ratio, × configured slr_m target)"
    )
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, output_path)


def plot_cleaned_network(
    rivers_orig: gpd.GeoDataFrame,
    rivers_clean: gpd.GeoDataFrame,
    bbox_poly: Polygon,
    land_polygons_path: str,
    output_path: str,
) -> None:
    """Plot kept (blue) vs removed (salmon) reaches after connectivity cleaning."""
    lon_min, lat_min, lon_max, lat_max = bbox_poly.bounds
    margin = max(lon_max - lon_min, lat_max - lat_min) * 0.3
    map_bounds = (
        lon_min - margin,
        lat_min - margin,
        lon_max + margin,
        lat_max + margin,
    )

    land = gpd.read_file(land_polygons_path, bbox=map_bounds)
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
    land_polygons_path: str,
    output_path: str,
) -> None:
    """Plot the clean network with width and colour by accumulated bankfull discharge.

    Reaches without discharge are light grey; others use four quartile width
    bins and a log-scaled Blues colormap.
    """
    lon_min, lat_min, lon_max, lat_max = bbox_poly.bounds
    margin = max(lon_max - lon_min, lat_max - lat_min) * 0.3
    map_bounds = (
        lon_min - margin,
        lat_min - margin,
        lon_max + margin,
        lat_max + margin,
    )
    land = gpd.read_file(land_polygons_path, bbox=map_bounds)

    col = "bankfull_discharge_acc_m3s"
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


# SWORD obstr_type -> label and colour of a structure marker.
_STRUCTURE_STYLE = {
    1: ("dam", "tab:red"),
    4: ("falls", "tab:red"),
    2: ("lock", "tab:orange"),
    3: ("low-head", "tab:orange"),
    0: ("level step only", "0.4"),
}


def plot_water_surface_profiles(
    surface,
    rivers: gpd.GeoDataFrame,
    seeds: list[str],
    output_path: str,
    title: str,
) -> None:
    """Plot the rule 10 water surface along each seed's widest downstream path.

    One panel per seed: FathomDEM pixels along the centreline, the water pixels
    the surface is built from (grey) and the skipped ones (islands, banks, sea;
    orange), node medians (black crosses), the smoothed surface (blue) and
    structure junctions coloured by the SWORD obstruction type of either reach
    (the surface is smoothed across them). The title gives the SWOT check of
    the path's reaches: surface slope / SWOT slope.

    Args:
        surface: src.water_surface.WaterSurface (with its centreline samples).
        rivers: The reaches the surface was built from ('reach_id', 'width',
            'rch_id_dn', optionally 'obstr_type').
        seeds: Seed reach ids, one panel each.
    """
    from src.river_network import normalize_reach_id, trace_widest_path

    net = surface.net
    obstr = {}
    if "obstr_type" in rivers.columns:
        obstr = dict(
            zip(
                rivers["reach_id"].map(normalize_reach_id),
                pd.to_numeric(rivers["obstr_type"], errors="coerce")
                .fillna(0)
                .astype(int),
            )
        )
    seeds = [s for s in seeds if s in net.reach_line] or []
    n = max(len(seeds), 1)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.6 * n), squeeze=False)
    if not seeds:
        axes[0, 0].text(
            0.5,
            0.5,
            "No seed reach",
            ha="center",
            va="center",
            transform=axes[0, 0].transAxes,
        )
    used_styles: set[int] = set()
    for ax, seed in zip(axes[:, 0], seeds):
        path = [r for r in trace_widest_path(rivers, seed) if r in net.reach_line]
        offset = 0.0
        for k, r in enumerate(path):
            along, value = surface.skipped.get(r, (np.array([]), np.array([])))
            ax.plot(
                (offset + along) / 1000,
                value,
                ".",
                ms=1.5,
                color="tab:orange",
                alpha=0.6,
                rasterized=True,
                label="other pixels (skipped)" if k == 0 else None,
            )
            along, value = surface.samples.get(r, (np.array([]), np.array([])))
            ax.plot(
                (offset + along) / 1000,
                value,
                ".",
                ms=1.5,
                color="0.72",
                rasterized=True,
                label="FathomDEM water pixels" if k == 0 else None,
            )
            ids, ch = net.reach_nodes[r], net.reach_chainage[r]
            ax.plot(
                (offset + ch) / 1000,
                surface.level_raw_m[ids],
                "x",
                ms=3,
                color="k",
                label="node median" if k == 0 else None,
            )
            ax.plot(
                (offset + ch) / 1000,
                surface.level_m[ids],
                "-",
                lw=1.5,
                color="tab:blue",
                label="smoothed surface" if k == 0 else None,
            )
            offset += net.reach_length[r]
            nxt = path[k + 1] if k + 1 < len(path) else None
            if nxt is not None and (r, nxt) in surface.structures:
                t = max(obstr.get(r, 0), obstr.get(nxt, 0))
                t = t if t in _STRUCTURE_STYLE and t > 0 else 0
                used_styles.add(t)
                ax.axvline(
                    offset / 1000,
                    color=_STRUCTURE_STYLE[t][1],
                    lw=0.9,
                    ls="--",
                    zorder=0,
                )
        finite = (
            np.concatenate([surface.level_m[net.reach_nodes[r]] for r in path])
            if path
            else np.array([0.0])
        )
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
        # Wide enough for the skipped pixels too (an island under the line);
        # percentiles keep single spikes from flattening the surface.
        pixels = np.concatenate(
            [
                np.asarray(s.get(r, ((), ()))[1], dtype=float)
                for s in (surface.samples, surface.skipped)
                for r in path
            ]
            or [np.array([])]
        )
        pixels = pixels[np.isfinite(pixels)]
        if pixels.size:
            lo = min(lo, float(np.percentile(pixels, 1)))
            hi = max(hi, float(np.percentile(pixels, 99)))
        pad = max(0.5, 0.1 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(0, max(offset, 1.0) / 1000)
        swot = surface.swot[surface.swot["reach_id"].isin(path)]
        swot_note = (
            f"; surface/SWOT slope median {swot['ratio'].median():.2f} over {len(swot)} reach(es)"
            if len(swot)
            else "; no reliable SWOT slope"
        )
        ax.set_title(
            f"seed {seed}: {len(path)} reach(es) along the widest arm{swot_note}",
            fontsize=9,
        )
        ax.set_ylabel("level (m, model datum)")
        ax.grid(alpha=0.3, lw=0.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [
        Line2D([0], [0], color=_STRUCTURE_STYLE[t][1], ls="--", lw=0.9)
        for t in sorted(used_styles)
    ]
    labels += [f"structure: {_STRUCTURE_STYLE[t][0]}" for t in sorted(used_styles)]
    axes[0, 0].legend(
        handles, labels, loc="upper right", fontsize=7, framealpha=0.9, ncol=2
    )
    axes[-1, 0].set_xlabel("distance from seed (km)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    _save(fig, output_path)


def plot_river_depth_width(
    segments: gpd.GeoDataFrame,
    land_mask_path: str,
    domain_outline,
    output_path: str,
    depth_label: str,
    title: str,
    lw_range: tuple[float, float] = (0.4, 6.0),
    value_col: str = "depth",
    value_range: tuple[float, float] | None = None,
    cmap_name: str = "viridis_r",
) -> None:
    """Plot the river network with colour = ``value_col`` and line width = channel width.

    segments are in the model CRS with ``value_col`` (m), 'width' (m, as used
    in the calculation) and optional 'fallback' (bool, depth on the slope
    floor; drawn dashed). value_range fixes the colour scale; None: 0 to the
    98th percentile. lw_range is the line width (points) of the narrowest and
    widest channel, scaled with sqrt(width) in between.
    """
    crs = segments.crs
    xmin, ymin, xmax, ymax = domain_outline.bounds
    pad = 0.03 * max(xmax - xmin, ymax - ymin)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_facecolor(
        "#e6f0f7"
    )  # open sea: light blue, so shallow (pale) channels stay visible
    land = gpd.read_file(land_mask_path).to_crs(crs)
    if not land.empty:
        land.plot(ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=1)
    for poly in getattr(domain_outline, "geoms", [domain_outline]):
        gx, gy = poly.exterior.xy
        ax.plot(gx, gy, color="black", linewidth=1.0, zorder=2)

    seg = segments[np.isfinite(segments["width"].to_numpy(dtype=float))]
    if seg.empty:
        ax.text(
            0.5,
            0.5,
            "No river segments",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="grey",
        )
    else:
        width = seg["width"].to_numpy(dtype=float)
        depth = seg[value_col].to_numpy(dtype=float)
        w_lo, w_hi = float(np.nanmin(width)), float(np.nanmax(width))

        def _lw(w):
            frac = np.sqrt(
                np.clip(
                    (np.asarray(w, dtype=float) - w_lo) / max(w_hi - w_lo, 1e-9),
                    0.0,
                    1.0,
                )
            )
            return lw_range[0] + (lw_range[1] - lw_range[0]) * frac

        if value_range is None:
            finite = np.isfinite(depth)
            vmax = float(np.nanpercentile(depth[finite], 98)) if finite.any() else 1.0
            norm = mcolors.Normalize(vmin=0.0, vmax=max(vmax, 0.1))
        else:
            norm = mcolors.Normalize(*value_range)
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#bbbbbb")
        fallback = (
            seg["fallback"].to_numpy(dtype=bool)
            if "fallback" in seg.columns
            else np.zeros(len(seg), dtype=bool)
        )
        # Wide channels first, so narrow ones stay visible on top.
        order = np.argsort(-width)
        for dashed in (False, True):
            idx = order[fallback[order] == dashed]
            if idx.size == 0:
                continue
            lines = [np.asarray(g.coords)[:, :2] for g in seg.geometry.to_numpy()[idx]]
            lc = LineCollection(
                lines,
                cmap=cmap,
                norm=norm,
                linewidths=_lw(width[idx]),
                linestyles=(0, (2, 1.5)) if dashed else "solid",
                capstyle="round" if not dashed else "butt",
                zorder=3,
            )
            lc.set_array(np.ma.masked_invalid(depth[idx]))
            ax.add_collection(lc)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        extend = "max" if value_range is None else "both"
        fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, extend=extend, label=depth_label)

        inner = [w for w in (100, 500, 2500) if 1.5 * w_lo < w < w_hi / 1.5]
        ticks = [round(w_lo), *inner, round(w_hi)]
        handles = [
            Line2D(
                [0], [0], color="#555555", linewidth=float(_lw(w)), label=f"{w:,.0f} m"
            )
            for w in ticks
        ]
        if fallback.any():
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="#555555",
                    linewidth=1.5,
                    linestyle=(0, (2, 1.5)),
                    label="depth on the slope floor",
                )
            )
        ax.legend(
            handles=handles,
            title="Channel width (used in the calculation)",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.13),
            ncol=len(handles),
            fontsize=8,
            framealpha=0.9,
        )

    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal")
    km = mticker.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}")
    ax.xaxis.set_major_formatter(km)
    ax.yaxis.set_major_formatter(km)
    ax.set_xlabel(f"x (km, {crs.to_string() if crs is not None else ''})")
    ax.set_ylabel("y (km)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    _save(fig, output_path)


def plot_river_depth(
    rivers_wgs: gpd.GeoDataFrame,
    bbox_poly: Polygon,
    land_polygons_path: str,
    output_path: str,
) -> None:
    """Plot the river network coloured by hydraulic depth ('river_depth_m', m)."""
    lon_min, lat_min, lon_max, lat_max = bbox_poly.bounds
    margin = max(lon_max - lon_min, lat_max - lat_min) * 0.3
    map_bounds = (
        lon_min - margin,
        lat_min - margin,
        lon_max + margin,
        lat_max + margin,
    )

    land = gpd.read_file(land_polygons_path, bbox=map_bounds)

    fig, ax = plt.subplots(figsize=(9, 7))
    if not land.empty:
        land.plot(ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=1)
    if not rivers_wgs.empty and "river_depth_m" in rivers_wgs.columns:
        rivers_wgs.plot(
            ax=ax,
            column="river_depth_m",
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


def _overlay_layers(
    crs,
    bounds: tuple[float, float, float, float],
    domain_poly: Polygon,
    land_polygons_path: str,
    river_network_path: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    # Returns (land, rivers, domain_gdf) in crs. Land is bbox-filtered with
    # bounds (in crs) transformed to WGS84; domain_poly is in WGS84.
    from rasterio.warp import transform_bounds

    wgs84_bounds = transform_bounds(crs, "EPSG:4326", *bounds)

    land = gpd.read_file(land_polygons_path, bbox=wgs84_bounds)
    if not land.empty:
        land = land.to_crs(crs)

    # The river network may be in UTM, where a WGS84 bbox filter returns nothing.
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
    if not land.empty:
        land.plot(
            ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=zorder
        )
    if not rivers.empty:
        rivers.plot(ax=ax, color="steelblue", linewidth=0.6, zorder=zorder + 1)
    domain_gdf.boundary.plot(ax=ax, color="black", linewidth=1.5, zorder=zorder + 2)


_OVERLAY_LEGEND_HANDLES = [
    Line2D([0], [0], color="black", linewidth=1.5, label="Model domain"),
    Patch(facecolor="#d9d9d9", edgecolor="#aaaaaa", label="Land"),
    Line2D([0], [0], color="steelblue", linewidth=1.5, label="River network"),
]


def plot_coastal_protection_weir(
    grid,
    diagnostics: dict,
    domain_poly: Polygon,
    land_polygons_path: str,
    river_network_path: str,
    output_path: str,
    basin_id: str = "",
    weir_gdf: gpd.GeoDataFrame | None = None,
) -> None:
    """Plot ocean, river channel and the coastal protection weir of rule 12a in UTM.

    grid is a src.protection_weir.GridArrays; diagnostics and weir_gdf come
    from build_coastal_protection_weir. The weir hugs the coast/riverbank (no
    inland offset). With weir_gdf ('elevation' per segment) the line is
    coloured by crest elevation, else drawn flat black.
    """
    ocean_mask = diagnostics["ocean_mask"]
    channel_mask = diagnostics["river_channel_mask"]
    weir_lines = diagnostics["weir_lines"]
    crest_elevation_m = diagnostics["crest_elevation_m"]

    transform = grid.transform
    bounds = footprint(transform, grid.shape).bounds

    land, rivers, domain_gdf = _overlay_layers(
        grid.crs, bounds, domain_poly, land_polygons_path, river_network_path
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    _draw_overlays(ax, land, rivers, domain_gdf, zorder=0)

    ocean_plot = np.where(ocean_mask, 1.0, np.nan)
    imshow_on_grid(
        ax,
        ocean_plot,
        transform,
        cmap=mcolors.ListedColormap(["steelblue"]),
        vmin=0,
        vmax=1,
        zorder=1,
    )
    channel_plot = np.where(channel_mask & ~ocean_mask, 1.0, np.nan)
    imshow_on_grid(
        ax,
        channel_plot,
        transform,
        cmap=mcolors.ListedColormap(["cornflowerblue"]),
        vmin=0,
        vmax=1,
        zorder=1.1,
    )

    has_crest_values = (
        weir_gdf is not None and not weir_gdf.empty and "elevation" in weir_gdf.columns
    )
    if has_crest_values:
        segments = [np.asarray(line.coords) for line in weir_gdf.geometry]
        crest_vals = weir_gdf["elevation"].to_numpy(dtype=float)
        lc = LineCollection(segments, cmap="turbo", zorder=3, linewidths=1.8)
        lc.set_array(crest_vals)
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label("Weir crest elevation (m)")
        weir_legend = Line2D(
            [0],
            [0],
            color="black",
            linewidth=1.5,
            label="Weir (colour = crest elevation)",
        )
    else:
        for line in weir_lines:
            x, y = line.xy
            ax.plot(x, y, color="black", linewidth=1.2, zorder=3)
        weir_legend = Line2D([0], [0], color="black", linewidth=1.5, label="Weir")

    legend_handles = _OVERLAY_LEGEND_HANDLES + [
        Patch(color="steelblue", label="Ocean"),
        Patch(color="cornflowerblue", label="River channel"),
        weir_legend,
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.9)

    title = (
        f"Coastal protection weir\ncrest={crest_elevation_m:+.2f} m, "
        f"{len(weir_lines)} segment(s)"
    )
    if basin_id:
        title = f"{title} | {basin_id}"
    ax.set_title(title)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    _save(fig, output_path)


def reproject_max_for_plot(
    da: xr.DataArray,
    dst_crs: str = "EPSG:4326",
    max_px: int = _PLOT_MAX_PX,
    resampling: Resampling = Resampling.max,
) -> xr.DataArray:
    """Reproject da to dst_crs at a resolution capped at max_px pixels, max-aggregated.

    Warping straight to the coarse grid avoids a multi-GiB full-resolution
    intermediate (subgrid output reaches ~1.5 m pixels) and keeps isolated
    peaks that stride-decimation would alias away. Georeferencing comes from
    hydromt's ``.raster`` accessor because rioxarray crops or raises on the
    south-up model grid. Returns a north-up DataArray with 1-D pixel-centre x/y
    (y descending); other dims (e.g. time) are reprojected slice by slice.
    """
    import hydromt  # noqa: F401 -- registers the .raster accessor
    from affine import Affine
    from rasterio.warp import reproject as rio_reproject

    y_dim, x_dim = da.raster.y_dim, da.raster.x_dim
    extra_dims = [d for d in da.dims if d not in (y_dim, x_dim)]
    da = da.transpose(*extra_dims, y_dim, x_dim)
    src_transform, src_crs = da.raster.transform, da.raster.crs
    height, width = da.sizes[y_dim], da.sizes[x_dim]

    natural_transform, natural_width, natural_height = calculate_default_transform(
        src_crs,
        dst_crs,
        width,
        height,
        *footprint(src_transform, (height, width)).bounds,
    )
    total_px = natural_width * natural_height
    factor = math.sqrt(total_px / max_px) if total_px > max_px else 1.0
    target_res = natural_transform.a * factor
    left, top = natural_transform.c, natural_transform.f
    dst_width = max(1, math.ceil(natural_width * natural_transform.a / target_res))
    dst_height = max(
        1, math.ceil(natural_height * abs(natural_transform.e) / target_res)
    )
    dst_transform = Affine(target_res, 0.0, left, 0.0, -target_res, top)

    src = np.asarray(da.values, dtype=np.float32)
    flat = src.reshape((-1, height, width))
    out = np.full((flat.shape[0], dst_height, dst_width), np.nan, dtype=np.float32)
    for i in range(flat.shape[0]):
        rio_reproject(
            source=flat[i],
            destination=out[i],
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=np.nan,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    # Resampling.max over all-NaN sources leaves the reduction's identity
    # (+/-inf) instead of nodata, which would poison downstream min/max.
    out[~np.isfinite(out)] = np.nan

    coords = {d: da[d] for d in extra_dims if d in da.coords}
    coords["y"] = top - (np.arange(dst_height) + 0.5) * target_res
    coords["x"] = left + (np.arange(dst_width) + 0.5) * target_res
    result = xr.DataArray(
        out.reshape(src.shape[:-2] + (dst_height, dst_width)),
        dims=(*extra_dims, "y", "x"),
        coords=coords,
        name=da.name,
    )
    result.rio.write_crs(dst_crs, inplace=True)
    return result


def plot_water_level_timeseries(
    t_days: np.ndarray,
    zs: np.ndarray,
    output_path: str,
    day_markers: list[tuple[float, str]] | None = None,
    station_labels: list[str] | None = None,
    basin_id: str = "",
    run_label: str = "",
    ylabel: str = "Water level (m)",
) -> None:
    """Plot water level over time at each observation point.

    t_days is (n_time,) in days, zs is (n_time, n_stations). day_markers adds
    (day, label) vertical lines, e.g. spin-up end. Spin-up convergence shows
    as flat lines near the right edge.
    """
    n_stations = zs.shape[1]
    labels = (
        station_labels
        if station_labels is not None
        else [f"obs {i + 1}" for i in range(n_stations)]
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    for i in range(n_stations):
        ax.plot(t_days, zs[:, i], lw=1.0, alpha=0.75, label=labels[i])

    for day, marker_label in day_markers or []:
        ax.axvline(day, color="red", linestyle="--", linewidth=1.2, label=marker_label)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel(ylabel)
    title_bits = [b for b in (basin_id, run_label) if b]
    prefix = f"{' | '.join(title_bits)}\n" if title_bits else ""
    ax.set_title(f"{prefix}Water level at {n_stations} observation point(s)")
    ax.legend(fontsize=7, ncol=max(1, n_stations // 5), loc="upper left")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=150, bbox_inches="tight")


def plot_crest_gap_map(
    crest_on_grid: np.ndarray,
    grid_transform,
    map_nc_path: str,
    weir_gdf: gpd.GeoDataFrame,
    rivers_utm: gpd.GeoDataFrame,
    discharge_x: np.ndarray,
    discharge_y: np.ndarray,
    output_path: str,
    basin_id: str = "",
    run_label: str = "",
    min_overtop_m: float = 0.01,
) -> None:
    """Plot weir crest minus simulated period-max water level (negative = overtopped).

    Only cells overtopped by more than min_overtop_m (the crest precision) are
    marked, counted and zoomed on; smaller exceedances are rounding.

    Evaluated at every finite cell of crest_on_grid (rule
    modelled_depth_estimation puts each weir edge's crest on its water-side
    cell). Panels: full domain, and a zoom on overtopped cells (else the crest
    area). Discharge points are the real model inflow points, not reach line
    starts. A SFINCS grid usually has a positive y-scale (row increases
    northward), so orientation is taken from grid_transform. Raises ValueError
    when no cell has a crest.
    """
    out_shape = crest_on_grid.shape
    rows, cols = np.where(np.isfinite(crest_on_grid))
    if len(rows) == 0:
        raise ValueError("plot_crest_gap_map: crest_on_grid has no finite cell")

    # zsmax (max over every computational timestep, as the calibration's
    # overtopping check uses) when written, else the max of hourly zs.
    with xr.open_dataset(map_nc_path) as ds:
        if "zsmax" in ds:
            zsmax = ds["zsmax"]
            period_max_zs_sim = zsmax.max(
                dim=[d for d in zsmax.dims if d not in ("n", "m")]
            ).values
        else:
            period_max_zs_sim = np.nanmax(ds["zs"].values, axis=0)
        map_grid_x = ds["x"].values
        map_grid_y = ds["y"].values

    valid_sfincs = np.isfinite(map_grid_x.ravel()) & np.isfinite(map_grid_y.ravel())
    sfincs_xy = np.column_stack(
        [map_grid_x.ravel()[valid_sfincs], map_grid_y.ravel()[valid_sfincs]]
    )
    sfincs_tree = cKDTree(sfincs_xy)
    raster_x, raster_y = rasterio.transform.xy(grid_transform, rows, cols)
    _dist, flat_idx = sfincs_tree.query(np.column_stack([raster_x, raster_y]))
    n_dim, m_dim = map_grid_x.shape
    orig_flat_idx = np.where(valid_sfincs)[0][flat_idx]
    n_idx, m_idx = np.unravel_index(orig_flat_idx, (n_dim, m_dim))
    zs_at_raster_cells = period_max_zs_sim[n_idx, m_idx]

    gap_grid = np.full(out_shape, np.nan, dtype=np.float32)
    gap_grid[rows, cols] = crest_on_grid[rows, cols] - zs_at_raster_cells

    overtopped_rows, overtopped_cols = np.where(gap_grid < -min_overtop_m)
    overtopped_x, overtopped_y = rasterio.transform.xy(
        grid_transform, overtopped_rows, overtopped_cols
    )

    def _plot_overlay(ax):
        if not rivers_utm.empty:
            rivers_utm.plot(ax=ax, color="black", linewidth=0.8, zorder=3)
        if not weir_gdf.empty:
            weir_gdf.plot(
                ax=ax, color="dimgray", linewidth=1.2, linestyle="--", zorder=4
            )
        if len(discharge_x):
            ax.scatter(
                discharge_x,
                discharge_y,
                marker="*",
                s=250,
                color="gold",
                edgecolors="black",
                linewidths=1,
                zorder=6,
                label="discharge point",
            )
        if len(overtopped_x):
            ax.scatter(
                overtopped_x,
                overtopped_y,
                s=120,
                facecolors="none",
                edgecolors="lime",
                linewidths=2.5,
                zorder=5,
                label=f"{len(overtopped_x)} cell(s) overtopped by > {min_overtop_m * 100:g} cm",
            )

    finite_gap = gap_grid[np.isfinite(gap_grid)]
    vmax = float(np.nanpercentile(np.abs(finite_gap), 99)) if len(finite_gap) else 1.0
    vmax = vmax if vmax > 0 else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    ax = axes[0]
    im = imshow_on_grid(
        ax,
        gap_grid,
        grid_transform,
        cmap="RdBu",
        vmin=-vmax,
        vmax=vmax,
        zorder=1,
    )
    plt.colorbar(
        im, ax=ax, label="crest - actual period-max zs (m)\n(negative = overtopped)"
    )
    _plot_overlay(ax)
    ax.legend(loc="upper right", fontsize=8)
    title_bits = [b for b in (basin_id, run_label) if b]
    prefix = f"{' | '.join(title_bits)}\n" if title_bits else ""
    ax.set_title(f"{prefix}Crest minus actual simulated water level")
    ax.set_aspect("equal")

    ax2 = axes[1]
    im2 = imshow_on_grid(
        ax2,
        gap_grid,
        grid_transform,
        cmap="RdBu",
        vmin=-vmax,
        vmax=vmax,
        zorder=1,
    )
    plt.colorbar(im2, ax=ax2, label="gap (m)")
    _plot_overlay(ax2)
    if len(overtopped_rows):
        pad_m = 15 * math.hypot(grid_transform.a, grid_transform.d)
        ax2.set_xlim(min(overtopped_x) - pad_m, max(overtopped_x) + pad_m)
        ax2.set_ylim(min(overtopped_y) - pad_m, max(overtopped_y) + pad_m)
        for x, y, r, c in zip(
            overtopped_x, overtopped_y, overtopped_rows, overtopped_cols
        ):
            ax2.annotate(
                f"({r},{c})\ngap={gap_grid[r, c]:.2f}m",
                (x, y),
                textcoords="offset points",
                xytext=(10, 10),
                color="black",
                fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8),
                zorder=7,
            )
        ax2.set_title(
            "Zoomed on the overtopped cell(s)\n(dashed = weir line, black = river network, star = discharge point)"
        )
    else:
        crest_xmin, crest_xmax = raster_x.min(), raster_x.max()
        crest_ymin, crest_ymax = raster_y.min(), raster_y.max()
        pad_m = 0.05 * max(crest_xmax - crest_xmin, crest_ymax - crest_ymin, 1.0)
        ax2.set_xlim(crest_xmin - pad_m, crest_xmax + pad_m)
        ax2.set_ylim(crest_ymin - pad_m, crest_ymax + pad_m)
        ax2.set_title(f"No cell overtopped by > {min_overtop_m * 100:g} cm this round")
    ax2.set_aspect("equal")

    fig.tight_layout()
    _save(fig, output_path)


def plot_wetland_dike_positions(
    landuse_on_grid: np.ndarray,
    grid_transform,
    weir_off: gpd.GeoDataFrame,
    weir_on: gpd.GeoDataFrame,
    ocean_wetland_mask: np.ndarray,
    output_path: str,
    active_on: bool,
    basin_id: str = "",
) -> None:
    """Plot the coastal dike with river_depth_modelling.unprotected_ocean_wetlands off vs on.

    Overlaid on open sea (200), sea-linked wetland/lagoon (moved outside the
    dike when on) and other wetland/lagoon (80/90, protected either way).
    Segments are matched by midpoint (< 1 m): grey in both, red OFF only,
    blue ON only. A zoom on the largest sea-linked patch is added if any.
    """
    from scipy.ndimage import label as _ndimage_label
    from src.landuse import LANDUSE_SEA_CODE, OCEAN_WETLAND_CLASSES

    shape = landuse_on_grid.shape
    classes = np.zeros(shape, dtype=np.uint8)
    classes[landuse_on_grid == LANDUSE_SEA_CODE] = 1
    classes[np.isin(landuse_on_grid, OCEAN_WETLAND_CLASSES)] = 2
    classes[ocean_wetland_mask] = 3
    class_colors = ["#e8e4d8", "#9ecae1", "#c7e9c0", "#31a354"]
    cmap = mcolors.ListedColormap(class_colors)

    def _midpoints(gdf: gpd.GeoDataFrame) -> np.ndarray:
        if gdf.empty:
            return np.zeros((0, 2))
        return np.array(
            [g.interpolate(0.5, normalized=True).coords[0][:2] for g in gdf.geometry]
        )

    def _in_other(mid_a: np.ndarray, mid_b: np.ndarray) -> np.ndarray:
        if len(mid_a) == 0 or len(mid_b) == 0:
            return np.zeros(len(mid_a), dtype=bool)
        dist, _ = cKDTree(mid_b).query(mid_a)
        return dist < 1.0

    mid_off, mid_on = _midpoints(weir_off), _midpoints(weir_on)
    off_shared, on_shared = _in_other(mid_off, mid_on), _in_other(mid_on, mid_off)
    off_only, on_only = weir_off[~off_shared], weir_on[~on_shared]
    shared = weir_off[off_shared]

    labels, n_patches = _ndimage_label(ocean_wetland_mask)
    has_zoom = n_patches > 0
    fig, axes = plt.subplots(
        1, 2 if has_zoom else 1, figsize=(20 if has_zoom else 11, 10)
    )
    axes = np.atleast_1d(axes)

    used_on, used_off = ("  <- used", "") if active_on else ("", "  <- used")
    for ax in axes:
        imshow_on_grid(ax, classes, grid_transform, cmap=cmap, vmin=0, vmax=3)
        for gdf, color, width in (
            (shared, "dimgray", 1.0),
            (off_only, "red", 1.8),
            (on_only, "blue", 1.8),
        ):
            if not gdf.empty:
                gdf.plot(ax=ax, color=color, linewidth=width, zorder=3)
        ax.set_aspect("equal")

    if has_zoom:
        biggest = np.argmax(np.bincount(labels.ravel())[1:]) + 1
        rows, cols = np.nonzero(labels == biggest)
        cell = abs(grid_transform.a)
        xs, ys = grid_transform * (cols + 0.5, rows + 0.5)
        pad = 15 * cell
        axes[1].set_xlim(np.min(xs) - pad, np.max(xs) + pad)
        axes[1].set_ylim(np.min(ys) - pad, np.max(ys) + pad)
        axes[1].set_title("Zoom: largest ocean-linked wetland patch")

    area_km2 = float(ocean_wetland_mask.sum()) * abs(grid_transform.determinant) / 1e6
    title_bits = [
        f"Basin {basin_id}" if basin_id else "",
        "coastal dike position, unprotected_ocean_wetlands off vs on",
    ]
    axes[0].set_title(
        " -- ".join(b for b in title_bits if b)
        + f"\nOFF: {weir_off.length.sum() / 1000:.1f} km, ON: {weir_on.length.sum() / 1000:.1f} km; "
        f"ocean-linked wetland moved outside the dike: {area_km2:.1f} km²"
    )
    handles = [
        Line2D(
            [0],
            [0],
            color="dimgray",
            linewidth=1.0,
            label="dike, same position with the switch OFF and ON",
        ),
        Line2D(
            [0],
            [0],
            color="red",
            linewidth=1.8,
            label=f"dike, switch OFF only (open-sea boundary){used_off}",
        ),
        Line2D(
            [0],
            [0],
            color="blue",
            linewidth=1.8,
            label=f"dike, switch ON only (landward of ocean-linked wetlands){used_on}",
        ),
        Patch(color=class_colors[1], label="open sea (200)"),
        Patch(color=class_colors[3], label="wetland/lagoon linked to the sea (80/90)"),
        Patch(color=class_colors[2], label="other wetland/lagoon (80/90)"),
        Patch(color=class_colors[0], label="other land"),
    ]
    axes[0].legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    _save(fig, output_path)


def plot_calibration_round_profiles(
    basin_id: str,
    seed: str,
    profiles_by_round: dict[int, pd.DataFrame],
    n_rounds: int,
    output_subplots_path: str,
    output_combined_path: str,
    round_titles: list[str] | None = None,
    water_level_labels: list[str] | None = None,
) -> None:
    """Plot bed, weir crest and water level along one seed-to-mouth path.

    One subplot per round plus an all-rounds overlay. profiles_by_round maps
    round_idx (0..n_rounds) to a DataFrame with 'along_path_m', 'dem_m',
    'river_depth_m', 'weir_crest_m', 'zs_m' (src.river_depth_calibration.
    gather_calibration_round_profile); the bed is dem_m - river_depth_m.
    water_level_labels names the level stored in 'zs_m' per round (default:
    period max); a subplot gets its own legend when its label differs from
    round 0's.
    """
    if round_titles is None:
        round_titles = [f"round {i}" for i in range(n_rounds + 1)]
    if water_level_labels is None:
        water_level_labels = ["water level, period max"] * (n_rounds + 1)
    fig, axes = plt.subplots(
        n_rounds + 1, 1, figsize=(11, 2.6 * (n_rounds + 1)), sharex=True, sharey=True
    )
    if n_rounds == 0:
        axes = [axes]
    for round_idx, ax in enumerate(axes):
        df = profiles_by_round[round_idx]
        ax.plot(
            df["along_path_m"],
            df["dem_m"],
            color="grey",
            lw=1.0,
            ls=":",
            label="natural DEM",
        )
        ax.plot(
            df["along_path_m"],
            df["dem_m"] - df["river_depth_m"],
            color="steelblue",
            lw=1.4,
            label="river bed (excavated)",
        )
        ax.plot(
            df["along_path_m"],
            df["weir_crest_m"],
            color="firebrick",
            lw=1.4,
            label="weir crest",
        )
        ax.plot(
            df["along_path_m"],
            df["zs_m"],
            color="black",
            lw=1.2,
            ls="-.",
            label=water_level_labels[round_idx],
        )
        ax.set_ylabel("Elevation (m)")
        ax.set_title(round_titles[round_idx], fontsize=9)
        ax.grid(True, alpha=0.3)
        if round_idx == 0 or water_level_labels[round_idx] != water_level_labels[0]:
            ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Distance along path from seed (m)")
    fig.suptitle(
        f"Basin {basin_id}, seed {seed}: bed/crest/water-level profile per round"
    )
    fig.tight_layout()
    save_figure(fig, output_subplots_path, dpi=150)

    fig, ax = plt.subplots(figsize=(12, 6))
    df0 = profiles_by_round[0]
    ax.plot(
        df0["along_path_m"],
        df0["dem_m"],
        color="black",
        lw=1.2,
        ls=":",
        label="natural DEM",
    )
    colors = cm.viridis(np.linspace(0.0, 0.9, n_rounds + 1))
    for round_idx in range(n_rounds + 1):
        df = profiles_by_round[round_idx]
        color = colors[round_idx]
        ax.plot(
            df["along_path_m"],
            df["dem_m"] - df["river_depth_m"],
            color=color,
            lw=1.3,
            ls="-",
            label=f"round {round_idx} bed",
        )
        ax.plot(
            df["along_path_m"],
            df["weir_crest_m"],
            color=color,
            lw=1.3,
            ls="--",
            label=f"round {round_idx} crest",
        )
        ax.plot(
            df["along_path_m"],
            df["zs_m"],
            color=color,
            lw=1.1,
            ls="-.",
            label=f"round {round_idx} zs",
        )
    ax.set_xlabel("Distance along path from seed (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(
        f"Basin {basin_id}, seed {seed}: bed (solid) / crest (dashed) / water level (dash-dot), all rounds"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    save_figure(fig, output_combined_path, dpi=150)


def draw_depth_map(
    ax,
    depth_m: np.ndarray,
    extent: tuple[float, float, float, float],
    land: gpd.GeoDataFrame,
    rivers: gpd.GeoDataFrame,
    vmax_m: float,
):
    """Draw a north-up depth raster over land polygons, with rivers on top.

    extent is (left, right, bottom, top) in the CRS of land and rivers.
    Returns the image for a colorbar.
    """
    if not land.empty:
        land.plot(ax=ax, color="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=1)
    im = ax.imshow(
        depth_m,
        cmap="Blues",
        vmin=0,
        vmax=vmax_m,
        extent=extent,
        origin="upper",
        aspect="auto",
        zorder=2,
    )
    if not rivers.empty:
        rivers.plot(ax=ax, color="steelblue", linewidth=0.6, zorder=3)
    return im


def plot_max_inundation_map(
    da_hmax: xr.DataArray,
    domain_poly: Polygon,
    land_polygons_path: str,
    river_network_path: str,
    output_path: str,
    basin_id: str = "",
    run_label: str = "",
) -> None:
    """Plot max inundation depth of a SFINCS run with land, rivers and domain overlaid.

    da_hmax (postprocessing.compute_max_inundation) uses NaN for dry or
    outside the land domain and must carry a CRS; domain_poly is in WGS84.
    """
    # imshow with explicit extent instead of da.plot() to guarantee north-up.
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

    vmax = _INUNDATION_DEPTH_CBAR_MAX_M
    n_flooded = int(da_hmax.notnull().sum().item())

    lon_min, lat_min, lon_max, lat_max = domain_poly.bounds
    _margin = max(lon_max - lon_min, lat_max - lat_min) * 0.3

    fig, ax = plt.subplots(figsize=(10, 8))
    im = draw_depth_map(
        ax, wgs_arr, (wgs_left, wgs_right, wgs_bottom, wgs_top), land, rivers, vmax
    )
    cbar = fig.colorbar(im, ax=ax, extend="max", fraction=0.03, pad=0.04)
    cbar.set_label("Max inundation depth (m)")
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


#: Colour scale per animate_flood_progression variable. Depth is pinned at 0
#: with the shared fixed maximum; water level can be negative relative to the
#: model datum and uses its true min/max so surge peaks are not clipped.
#: Without vmax, use_percentile=True clips to the 1st/99th percentile.
_FLOOD_ANIMATION_STYLE: dict[str, dict] = {
    "depth": {
        "cmap": "Blues",
        "label": "Water depth (m)",
        "vmin": 0.0,
        "vmax": _INUNDATION_DEPTH_CBAR_MAX_M,
        "use_percentile": True,
    },
    "level": {
        "cmap": "viridis",
        "label": "Water level (m)",
        "vmin": None,
        "vmax": None,
        "use_percentile": False,
    },
}


def _flood_animation_bounds(vals: np.ndarray, style: dict) -> tuple[float, float]:
    # isfinite, not ~isnan: Resampling.max can leave stray -inf pixels.
    valid = vals[np.isfinite(vals)]
    if style.get("vmax") is not None:
        vmin = (
            style["vmin"]
            if style["vmin"] is not None
            else (float(valid.min()) if len(valid) else 0.0)
        )
        return vmin, float(style["vmax"])
    if not len(valid):
        return (style["vmin"] if style["vmin"] is not None else 0.0), 1.0
    if style["use_percentile"]:
        vmax = float(np.percentile(valid, 99))
        vmin = (
            float(np.percentile(valid, 1)) if style["vmin"] is None else style["vmin"]
        )
    else:
        vmax = float(valid.max())
        vmin = float(valid.min()) if style["vmin"] is None else style["vmin"]
    if style["vmin"] is not None:
        vmax = max(vmax, 0.01)
    return vmin, vmax


def animate_flood_progression(
    da_h: xr.DataArray,
    domain_poly: Polygon,
    land_polygons_path: str,
    river_network_path: str,
    output_path: str,
    basin_id: str = "",
    run_label: str = "",
    fps: int = 4,
    variable: str = "depth",
) -> None:
    """Write an MP4 of a run's instantaneous water depth or level over time.

    da_h (postprocessing.compute_flood_progression) has a time dimension and a
    CRS; for variable="depth" NaN marks dry, water or outside the domain, for
    "level" it is unmasked. Raises ValueError without a time dimension.
    Requires ffmpeg.
    """
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    # libx264's default CRF (23) quantizes away the pale floodplain signal
    # against the grey land; -crf 15 is near visually lossless.
    def _make_writer():
        return FFMpegWriter(fps=fps, extra_args=["-crf", "15", "-pix_fmt", "yuv420p"])

    if variable not in _FLOOD_ANIMATION_STYLE:
        raise ValueError(
            f"variable must be one of {list(_FLOOD_ANIMATION_STYLE)}, got {variable!r}"
        )
    style = _FLOOD_ANIMATION_STYLE[variable]
    title_word = "Flood progression" if variable == "depth" else "Water level"

    if "time" not in da_h.dims:
        raise ValueError(
            f"animate_flood_progression: da_h has no 'time' dimension (dims {da_h.dims})"
        )

    da_wgs = reproject_max_for_plot(da_h)
    land, rivers, domain_gdf = _overlay_layers(
        da_wgs.rio.crs,
        da_wgs.rio.bounds(),
        domain_poly,
        land_polygons_path,
        river_network_path,
    )

    vals = da_wgs.values
    vmin, vmax = _flood_animation_bounds(vals, style)

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
        cmap=style["cmap"],
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        origin=origin,
        zorder=2,
    )
    extend = "max" if style["vmin"] is not None else "both"
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, extend=extend)
    cb.set_label(style["label"])
    ax.set_xlim(lon_min - _margin, lon_max + _margin)
    ax.set_ylim(lat_min - _margin, lat_max + _margin)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(
        handles=_OVERLAY_LEGEND_HANDLES, loc="lower right", framealpha=0.9, fontsize=8
    )

    title_prefix = title_word
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
    anim.save(str(out_path), writer=_make_writer())
    plt.close(fig)
    log.info(f"Written: {output_path}")


def plot_geoid_offset(
    offset_arr: np.ndarray,
    offset_transform,
    wgs84_bounds: tuple[float, float, float, float],
    land_polygons_path: str,
    output_path: str,
) -> None:
    """Plot the geoid offset N_EGM2008 − N_GOCO06s over the domain (WGS84 lon/lat).

    The step-1 correction field, before resampling to the DEM grid.
    """
    h, w = offset_arr.shape
    lon_left = offset_transform.c
    lat_top = offset_transform.f
    lon_right = lon_left + w * offset_transform.a
    lat_bottom = lat_top + h * offset_transform.e  # e is negative (south)
    extent = (lon_left, lon_right, lat_bottom, lat_top)

    lon_min, lat_min, lon_max, lat_max = wgs84_bounds
    land = gpd.read_file(land_polygons_path, bbox=(lon_min, lat_min, lon_max, lat_max))

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
    land_polygons_path: str,
    output_path: str,
) -> None:
    """Plot raw MDT HYBRID-CNES-CLS22 over ocean (NaN over land, WGS84 lon/lat).

    The GEBCO MSL -> GOCO06s correction field (added to GEBCO), before
    inverse-distance extrapolation over land.
    """
    h, w = mdt_np.shape
    lon_left = mdt_transform.c
    lat_top = mdt_transform.f
    lon_right = lon_left + w * mdt_transform.a
    lat_bottom = lat_top + h * mdt_transform.e
    extent = (lon_left, lon_right, lat_bottom, lat_top)

    lon_min, lat_min, lon_max, lat_max = wgs84_bounds
    land = gpd.read_file(land_polygons_path, bbox=(lon_min, lat_min, lon_max, lat_max))

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
    wgs84_bounds: tuple[float, float, float, float],
    land_polygons_path: str,
    output_path: str,
    title: str = "Vertical datum correction\n(EGM2008 → GOCO06s geoid)",
    colorbar_label: str = "Δ elevation (m)  [GOCO06s − EGM2008]",
    vmax: float | None = None,
    symmetric: bool = True,
    note: str | None = None,
) -> None:
    """Plot a vertical datum correction delta (corrected − original) in UTM metres.

    For the single-step corrections of 06_get_elevation.py: the coastal DEM's
    geoid offset (EGM2008 → GOCO06s) or GEBCO's +MDT (local MSL → GOCO06s).
    NaN pixels are transparent. symmetric=True uses ±vmax (default the 95th
    percentile of |delta|) on RdBu_r; symmetric=False uses the data min/max on
    a sequential Reds map and ignores vmax, for one-signed corrections.
    """
    from pyproj import Transformer

    lon_min, lat_min, lon_max, lat_max = wgs84_bounds
    h, w = delta.shape

    plot_crs = pick_utm_crs(
        gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([lon_min, lon_max], [lat_min, lat_max]),
            crs="EPSG:4326",
        )
    )

    _trans = Transformer.from_crs("EPSG:4326", plot_crs, always_xy=True)
    _west, _south = _trans.transform(lon_min, lat_min)
    _east, _north = _trans.transform(lon_max, lat_max)
    extent = (_west, _east, _south, _north)

    total = h * w
    factor = max(1, int(math.ceil(math.sqrt(total / _PLOT_MAX_PX))))
    delta_ds = delta[::factor, ::factor]

    land = gpd.read_file(land_polygons_path, bbox=(lon_min, lat_min, lon_max, lat_max))
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
        # With a skewed range RdBu_r's white midpoint leaves zero; a sequential
        # map keeps the bound closer to zero light and the extreme bound dark.
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
    if note:
        ax.text(
            0.0,
            -0.1,
            note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            style="italic",
            color="#444444",
        )
    _save(fig, output_path)


def plot_inundation_check(
    da_hmax: xr.DataArray,
    threshold_m: float,
    n_flooded: int,
    n_land: int,
    land_polygons_path: str,
    river_network_path: str,
    output_path: str,
    basin_id: str = "",
    run_label: str = "baseline spinup",
) -> None:
    """Plot the fraction of land pixels above an inundation threshold (rules 13b, 15).

    Left panel — map of max inundation depth drawn with ax.imshow and an
                 explicit extent from the DataArray's x/y coordinates, so it is
                 north-up whatever the y-axis direction; the land outline and
                 clean river network are overlaid for context.
    Right panel — bar chart of flooded vs dry pixel counts.

    da_hmax (downscale_floodmap) uses NaN for dry or outside the domain and
    must carry a CRS; threshold_m is its hmin, used for labels only. n_flooded
    and n_land are precomputed land pixel counts.
    """
    frac = n_flooded / n_land if n_land > 0 else 0.0
    n_dry = n_land - n_flooded

    # Explicit imshow extent instead of da.plot() guarantees north-up however
    # downscale_floodmap orders the spatial dimensions.
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

    land = gpd.read_file(
        land_polygons_path, bbox=(wgs_left, wgs_bottom, wgs_right, wgs_top)
    )
    # The river network may be in UTM, where a WGS84 bbox filter returns nothing.
    rivers = gpd.read_file(river_network_path)
    if not rivers.empty and rivers.crs is not None and rivers.crs.to_epsg() != 4326:
        rivers = rivers.to_crs("EPSG:4326")

    vmax = _INUNDATION_DEPTH_CBAR_MAX_M

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 1]}
    )

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
