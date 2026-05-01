"""Marine buffer construction and GEBCO tile clipping utilities.

Clips GEBCO bathymetry tiles to the buffered bounding boxes of delta basins,
masks out land pixels using OSM land polygons, and applies a depth cutoff.
Designed to run in parallel using a process pool for the tile-clipping step.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
from geopandas import GeoDataFrame
from rasterio.crs import CRS as RioCRS
from rasterio.merge import merge
from shapely import wkt
from shapely.geometry import box, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.validation import make_valid

from src.input_processing.utils.util_basin_bboxes import basins_to_buffered_bboxes
from src.utils.setup_logger import setup_logging

_LOG = setup_logging("marine_buffer")


# ---------------------------------------------------------------------------
# Land polygon loading and filtering
# ---------------------------------------------------------------------------


def load_land_polygons(
    shp_path: str,
    config: dict,  # type: ignore[type-arg]
    bbox: list[float] | None = None,
) -> GeoDataFrame:
    """Load OSM land polygons, optionally filtered to a bounding box.

    Reprojects to the project standard CRS if needed. Exits with an error
    if no polygons are found.

    Args:
        shp_path: Path to the OSM land polygons shapefile.
        config: Project config dict with key ``CRS.standard``.
        bbox: Optional spatial filter as ``[minx, miny, maxx, maxy]`` in the
            CRS of the shapefile.

    Returns:
        GeoDataFrame of land polygons in the project standard CRS.

    Raises:
        SystemExit: If no land polygons are found after loading.
    """
    if bbox:
        _LOG.info("Loading land polygons with spatial filter: %s", bbox)
        gdf = gpd.read_file(shp_path, bbox=bbox)
    else:
        _LOG.info(
            "Loading full global land polygon dataset (may take a while): %s", shp_path
        )
        gdf = gpd.read_file(shp_path)

    if gdf.empty:
        raise SystemExit(
            "No land polygons found. Check your shapefile path or try a different bbox."
        )

    target_epsg: int = int(config["CRS"]["standard"])
    if gdf.crs is None:
        gdf = gdf.set_crs(f"EPSG:{target_epsg}")
    elif gdf.crs.to_epsg() != target_epsg:
        gdf = gdf.to_crs(f"EPSG:{target_epsg}")

    _LOG.info("Loaded %d land polygons.", len(gdf))
    return gdf


def _prefilter_land_polygons(
    land_gdf: GeoDataFrame,
    delta_polygons_path: str,
    buffer_deg: float,
    config: dict,  # type: ignore[type-arg]
) -> GeoDataFrame:
    """Pre-filter OSM land polygons to those intersecting any delta bounding box.

    Replaces the dissolved-union approach with per-delta bounding boxes to
    avoid materialising a complex global geometry. Land polygons are kept if
    they intersect the union of all buffered delta bounding boxes.

    Args:
        land_gdf: Full (or bbox-filtered) OSM land polygon GeoDataFrame.
        delta_polygons_path: Path to the GeoPackage with delta basin polygons.
        buffer_deg: Buffer distance in degrees baked into each bounding box by
            :func:`basins_to_buffered_bboxes`.
        config: Project config dict with keys ``CRS.standard`` and
            ``DomainSchema.delta_id_lbl``.

    Returns:
        Filtered GeoDataFrame containing only the land polygons that intersect
        at least one buffered delta bounding box.
    """
    _LOG.info(
        "Pre-filtering land polygons against delta bboxes (buffer=%.4f deg).",
        buffer_deg,
    )
    delta_gdf = gpd.read_file(delta_polygons_path)

    target_epsg: int = int(config["CRS"]["standard"])
    if delta_gdf.crs is None:
        delta_gdf = delta_gdf.set_crs(f"EPSG:{target_epsg}")
    elif delta_gdf.crs.to_epsg() != target_epsg:
        delta_gdf = delta_gdf.to_crs(f"EPSG:{target_epsg}")

    delta_id_col: str = config["DomainSchema"]["delta_id_lbl"]
    bbox_gdf = basins_to_buffered_bboxes(delta_gdf, buffer_deg, delta_id_col)
    _LOG.debug("Built %d buffered bounding boxes.", len(bbox_gdf))

    # Union of simple rectangles is cheap — no complex polygon arithmetic.
    bbox_union = unary_union(bbox_gdf.geometry)

    n_before = len(land_gdf)
    land_gdf = cast(
        GeoDataFrame, land_gdf[land_gdf.geometry.intersects(bbox_union)].copy()
    )
    _LOG.info(
        "Pre-filter: %d → %d land polygons (%d removed).",
        n_before,
        len(land_gdf),
        n_before - len(land_gdf),
    )
    return land_gdf


# ---------------------------------------------------------------------------
# Marine buffer construction
# ---------------------------------------------------------------------------


def build_marine_buffer(
    land_gdf: GeoDataFrame,
    bbox: list[float] | None,
    buffer_deg: float,
    buffer_crs: str,
    buffer_fname: str,
    n_workers: int,
    config: dict,  # type: ignore[type-arg]
    delta_polygons_path: str,
) -> GeoDataFrame:
    """Return the pre-filtered land polygons used to mask GEBCO tiles.

    Replaces the previous union-and-buffer approach. The returned GeoDataFrame
    is passed to :func:`serialize_clipping` where land pixels are masked out
    per tile and a depth cutoff is applied, avoiding any monolithic geometry
    operations.

    Args:
        land_gdf: OSM land polygon GeoDataFrame in the project CRS.
        bbox: Optional bounding box ``[minx, miny, maxx, maxy]`` to clip land
            polygons before returning.
        buffer_deg: Buffer distance in degrees used by
            :func:`_prefilter_land_polygons` when building delta bounding boxes.
        buffer_crs: CRS string for the output GeoDataFrame.
        buffer_fname: Output path to write the land mask GeoPackage (for
            inspection/reuse).
        n_workers: Unused; retained for interface compatibility.
        config: Project config dict with keys ``CRS.standard`` and
            ``DomainSchema.delta_id_lbl``.
        delta_polygons_path: Path to the delta basin GeoPackage used for
            pre-filtering.

    Returns:
        GeoDataFrame of pre-filtered land polygons in ``buffer_crs``.
    """
    land_gdf = _prefilter_land_polygons(
        land_gdf, delta_polygons_path, buffer_deg, config
    )

    if bbox:
        _LOG.info("Clipping land polygons to bbox: %s", bbox)
        bbox_geom = box(bbox[0], bbox[1], bbox[2], bbox[3])
        land_gdf = land_gdf.copy()
        land_gdf["geometry"] = land_gdf.geometry.intersection(bbox_geom)
        land_gdf = cast(GeoDataFrame, land_gdf[~land_gdf.geometry.is_empty].copy())

    _LOG.info("Reprojecting %d land polygons to %s ...", len(land_gdf), buffer_crs)
    land_gdf = land_gdf.to_crs(buffer_crs)

    land_gdf.to_file(Path(buffer_fname), driver="GPKG")
    _LOG.info("Land mask written to: %s", buffer_fname)

    return land_gdf


# ---------------------------------------------------------------------------
# GEBCO tile clipping
# ---------------------------------------------------------------------------


def _clip_tile(
    args: tuple[str, str, int, float, str, str],
) -> str | None:
    """Clip a GEBCO tile to the delta bbox union, mask land, and apply a depth cutoff.

    Writes a temporary clipped tile to ``tmp_dir``. Tiles that do not
    intersect the bbox union, or are entirely masked after land removal or the
    depth cutoff, are skipped.

    Args:
        args: Tuple of ``(tif_path, bbox_wkt, crs_epsg, depth_cutoff,
            tmp_dir, land_wkt)`` where ``bbox_wkt`` is the WKT of the union
            of all delta bounding boxes used to clip the tile, and ``land_wkt``
            is the WKT of the unioned land polygons used to mask non-marine
            pixels.

    Returns:
        Path to the clipped temporary tile as a string, or None if skipped.
    """
    tif_path, bbox_wkt, crs_epsg, depth_cutoff, tmp_dir, land_wkt = args
    bbox_geom: BaseGeometry = wkt.loads(bbox_wkt)
    land_geom: BaseGeometry = wkt.loads(land_wkt)
    tile_crs_obj = RioCRS.from_epsg(crs_epsg)

    try:
        with rasterio.open(tif_path) as src:
            tile_crs = src.crs

            if tile_crs_obj != tile_crs:
                clip_geom: BaseGeometry = cast(
                    BaseGeometry,
                    gpd.GeoDataFrame(geometry=[bbox_geom], crs=tile_crs_obj)
                    .to_crs(tile_crs)
                    .geometry[0],
                )
                land_geom = cast(
                    BaseGeometry,
                    gpd.GeoDataFrame(geometry=[land_geom], crs=tile_crs_obj)
                    .to_crs(tile_crs)
                    .geometry[0],
                )
            else:
                clip_geom = bbox_geom

            if not clip_geom.intersects(box(*src.bounds)):
                return None

            # Use the raster's own nodata sentinel, or -9999 for int16 tiles
            # that cannot represent NaN. Convert to float32 immediately after.
            src_nodata = src.nodata if src.nodata is not None else -9999
            data, transform = rasterio.mask.mask(
                src, [mapping(clip_geom)], crop=True, nodata=src_nodata, filled=True
            )
            data = data.astype(np.float32)
            data[data == src_nodata] = np.nan

            if not np.any(np.isfinite(data)):
                return None

            # Mask land pixels — intersect with clip window first to avoid
            # allocating the full raster and to prevent shape mismatches.
            intersection: BaseGeometry = clip_geom.intersection(land_geom)
            if not intersection.is_empty:
                land_clipped, _ = rasterio.mask.mask(
                    src,
                    [mapping(intersection)],
                    crop=True,
                    nodata=src_nodata,
                    filled=True,
                )
                land_clipped = land_clipped.astype(np.float32)
                land_clipped[land_clipped == src_nodata] = np.nan
                # Finite pixels in land_clipped overlap land — mask them out.
                data[np.isfinite(land_clipped)] = np.nan

            # Mask by depth cutoff
            data[data > depth_cutoff] = np.nan

            if not np.any(np.isfinite(data)):
                return None

            tmp_path = Path(tmp_dir) / f"clipped_{Path(tif_path).stem}.tif"
            meta = src.meta.copy()
            meta.update(
                {
                    "driver": "GTiff",
                    "dtype": "float32",
                    "height": data.shape[1],
                    "width": data.shape[2],
                    "transform": transform,
                    "nodata": np.nan,
                    "compress": "deflate",
                    "predictor": 3,
                    "tiled": True,
                    "blockxsize": 512,
                    "blockysize": 512,
                }
            )
            with rasterio.open(tmp_path, "w", **meta) as dst:
                dst.write(data)
            return str(tmp_path)

    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Failed clipping {tif_path}: {exc}", stacklevel=2)
        return None


def merge_tiles(clipped_paths: list[str], output_path: Path) -> None:
    """Open all clipped tiles and merge them into a single GeoTIFF.

    Args:
        clipped_paths: List of paths to clipped tile GeoTIFFs.
        output_path: Destination path for the merged output GeoTIFF.
    """
    _LOG.info("Merging %d clipped tiles → %s", len(clipped_paths), output_path)
    sources = [rasterio.open(p) for p in clipped_paths]

    try:
        mosaic, transform = merge(sources, nodata=np.nan)

        meta = sources[0].meta.copy()
        meta.update(
            {
                "driver": "GTiff",
                "dtype": "float32",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "nodata": np.nan,
                "compress": "deflate",
                "predictor": 3,
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "bigtiff": "IF_SAFER",
            }
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(mosaic)

        valid = int(np.sum(np.isfinite(mosaic)))
        _LOG.info("Output size  : %d x %d px", mosaic.shape[2], mosaic.shape[1])
        _LOG.info("Valid pixels : %d", valid)
        _LOG.info(
            "Value range  : %.1f m  to  %.1f m",
            float(np.nanmin(mosaic)),
            float(np.nanmax(mosaic)),
        )

    finally:
        for src in sources:
            src.close()


def serialize_clipping(
    marine_gdf: GeoDataFrame,
    gebco_dir: Path,
    depth_buffer: float,
    n_workers: int,
    config_dict: dict,  # type: ignore[type-arg]
) -> list[str]:
    """Clip all GEBCO tiles in *gebco_dir* to the delta bboxes in parallel.

    Serialises the bbox union and land mask geometries to WKT for safe
    transfer to worker processes. Each tile is clipped to the bbox union,
    land pixels are masked, and the depth cutoff is applied. Tiles that do
    not intersect the bbox union or are fully masked are skipped.

    Args:
        marine_gdf: GeoDataFrame of pre-filtered land polygons as returned by
            :func:`build_marine_buffer`. Used both to derive the clipping bbox
            and as the land mask.
        gebco_dir: Directory to search recursively for GEBCO GeoTIFF files.
        depth_buffer: Depth cutoff in metres; pixels shallower than this value
            are masked.
        n_workers: Maximum number of parallel worker processes.
        config_dict: Project config dict with key ``CRS.standard``.

    Returns:
        List of paths to the clipped temporary tile files.

    Raises:
        SystemExit: If no GeoTIFF files are found in ``gebco_dir``, or if no
            tiles remain after clipping.
    """
    crs_epsg: int = config_dict["CRS"]["standard"]

    # Bbox union: the envelope covering all land polygons — used to clip tiles.
    bbox_union = unary_union(marine_gdf.geometry.envelope)
    bbox_wkt: str = bbox_union.wkt

    # Land union: used per-tile to mask non-marine pixels.
    _LOG.info("Unioning %d land polygon(s) for land mask ...", len(marine_gdf))
    land_union = make_valid(unary_union(marine_gdf.geometry))
    land_wkt: str = land_union.wkt

    tifs = sorted(gebco_dir.rglob("*.tif")) + sorted(gebco_dir.rglob("*.tiff"))
    if not tifs:
        raise SystemExit(f"No GeoTIFF files found in {gebco_dir}")

    _LOG.info(
        "Clipping %d GEBCO tile(s) (depth cutoff > %.1f m, %d workers) ...",
        len(tifs),
        depth_buffer,
        n_workers,
    )

    tmp_dir = gebco_dir / "_tmp_clipped_tiles"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tile_args: list[tuple[str, str, int, float, str, str]] = [
        (str(f), bbox_wkt, crs_epsg, depth_buffer, str(tmp_dir), land_wkt) for f in tifs
    ]

    clipped_paths: list[str] = []
    skipped = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_clip_tile, a): a[0] for a in tile_args}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result is None:
                skipped += 1
            else:
                clipped_paths.append(result)

            if i % 10 == 0 or i == len(tifs):
                _LOG.info(
                    "  %4d / %d tiles processed | %d kept, %d skipped",
                    i,
                    len(tifs),
                    len(clipped_paths),
                    skipped,
                )

    _LOG.info(
        "Clipping complete: %d tile(s) kept, %d skipped (no overlap or fully masked).",
        len(clipped_paths),
        skipped,
    )

    if not clipped_paths:
        raise SystemExit(
            "No GEBCO tiles remained after clipping. "
            "Check that your GEBCO and delta polygon files cover the same region."
        )

    return clipped_paths
