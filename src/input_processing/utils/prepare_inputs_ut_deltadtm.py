"""Fundamental functions to preprocess the Delta-DTM data.

Uses the modified data from Seegers & Minderhoud which already applies the
geoid correction. Includes functions to select and store files from the entire
set of Delta-DTM tiles relevant for the selected delta polygons (with a 1 deg
buffer), and a function to combine all selected tiles into one dataset.

These functions could be adjusted to work from a file list and download URLs
as provided by Seegers & Minderhoud, rather than an existing directory.
"""

import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from src.utils.setup_logger import setup_logging
from src.input_processing.utils.util_basin_bboxes import basins_to_buffered_bboxes

_LOG = setup_logging("preprocess_deltaDTM")

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

# Matches coordinate tags like: N00E006, S30W030, N51E004, S03W120
COORD_RE = re.compile(
    r"(?P<lat_hem>[NS])(?P<lat_deg>\d{2})"
    r"(?P<lon_hem>[EW])(?P<lon_deg>\d{2,3})"
)


def _parse_tile_bbox(
    filename: str,
    tile_extent: int | float,
) -> tuple[float, float, float, float] | None:
    """Parse a filename and return the tile bounding box.

    Extracts the coordinate tag (e.g. ``N00E006``) from the filename and
    converts it to a ``(minx, miny, maxx, maxy)`` bounding box in WGS-84
    degrees. The coordinate in the filename is treated as the SW corner of
    the tile.

    Args:
        filename: Tile filename to parse, e.g. ``"DeltaDTM_N00E006.tif"``.
        tile_extent: Width and height of the tile in degrees.

    Returns:
        Bounding box as ``(minx, miny, maxx, maxy)``, or None if no
        coordinate tag could be found in the filename.
    """
    m = COORD_RE.search(filename)
    if not m:
        _LOG.debug("No coordinate tag found in filename: %s", filename)
        return None

    lat = int(m.group("lat_deg"))
    lon = int(m.group("lon_deg"))

    if m.group("lat_hem") == "S":
        lat = -lat
    if m.group("lon_hem") == "W":
        lon = -lon

    # Coordinate in filename is the SW corner.
    if m.group("lat_hem") == "N":
        miny, maxy = lat, lat + tile_extent
    else:
        miny, maxy = lat - tile_extent, lat

    if m.group("lon_hem") == "E":
        minx, maxx = lon, lon + tile_extent
    else:
        minx, maxx = lon - tile_extent, lon

    return (minx, miny, maxx, maxy)


# ---------------------------------------------------------------------------
# Tile selection
# ---------------------------------------------------------------------------


def _select_tiles_one_delta(
    tile_dir: Path,
    study_bbox: tuple[float, float, float, float],
    tile_extent: int | float,
) -> list[Path]:
    """Return all tile paths in *tile_dir* that intersect *study_bbox*.

    Scans all ``.tif`` and ``.tiff`` files under ``tile_dir`` recursively,
    parses the coordinate tag from each filename, and selects those whose
    bounding box intersects the study area.

    Args:
        tile_dir: Root directory to search for tile files.
        study_bbox: Bounding box to intersect against, as
            ``(minx, miny, maxx, maxy)`` in WGS-84 degrees.
        tile_extent: Width and height of each tile in degrees.

    Returns:
        List of paths to tiles that intersect the study bounding box.
    """
    study_geom = box(*study_bbox)
    tifs = sorted(tile_dir.rglob("*.tif")) + sorted(tile_dir.rglob("*.tiff"))
    _LOG.debug("Scanning %d filenames in: %s", len(tifs), tile_dir)

    selected: list[Path] = []
    unparseable: list[str] = []

    for f in tifs:
        bbox = _parse_tile_bbox(f.name, tile_extent)
        if bbox is None:
            unparseable.append(f.name)
            continue
        if study_geom.intersects(box(*bbox)):
            selected.append(f)

    if unparseable:
        _LOG.warning(
            "%d file(s) had no parseable coordinate tag — first 3: %s",
            len(unparseable),
            unparseable[:3],
        )

    _LOG.debug(
        "Selected %d / %d tiles for bbox %s", len(selected), len(tifs), study_bbox
    )
    return selected


def combine_tiles_all_deltas(
    tile_dir: Path,
    delta_polygons_path: Path,
    buffer_deg: float,
    config: dict,  # type: ignore[type-arg]
    tile_extent: int | float,
) -> list[Path]:
    """Collect all tile paths that overlap any delta polygon in the dataset.

    Reads the delta polygon GeoPackage, computes a buffered bounding box per
    unique delta ID, and selects the tiles from ``tile_dir`` that intersect
    each box. Duplicate paths (tiles shared between deltas) are deduplicated
    before returning.

    Args:
        tile_dir: Directory containing the Delta-DTM tile files.
        delta_polygons_path: Path to the GeoPackage with delta basin polygons.
        buffer_deg: Buffer in degrees to add around each delta bounding box.
        config: Project config dict with keys ``CRS.standard`` and
            ``DomainSchema.delta_id_lbl``.
        tile_extent: Width and height of each tile in degrees.

    Returns:
        Deduplicated list of paths to all tiles that overlap at least one
        delta polygon.
    """
    _LOG.info("Loading delta polygons: %s", delta_polygons_path)
    polygons_gdf = gpd.read_file(delta_polygons_path)

    target_epsg: int = config["CRS"]["standard"]
    if polygons_gdf.crs is None or polygons_gdf.crs.to_epsg() != target_epsg:
        _LOG.debug("Reprojecting delta polygons to EPSG:%d.", target_epsg)
        polygons_gdf = polygons_gdf.to_crs(target_epsg)

    delta_id_col: str = config["DomainSchema"]["delta_id_lbl"]
    polygon_bboxes = basins_to_buffered_bboxes(polygons_gdf, buffer_deg, delta_id_col)
    _LOG.info(
        "Processing %d delta(s) from column '%s'.", len(polygon_bboxes), delta_id_col
    )

    all_selected: list[Path] = []
    for _, row in polygon_bboxes.iterrows():
        delta_name = row[delta_id_col]

        geom = getattr(row, "geometry")

        if not isinstance(geom, BaseGeometry):
            _LOG.warning("Invalid geometry for delta '%s'", delta_name)
            continue

        study_bbox: tuple[float, float, float, float] = geom.bounds

        selected = _select_tiles_one_delta(tile_dir, study_bbox, tile_extent)

        if not selected:
            _LOG.warning(
                "Delta '%s': no overlapping tiles found (bbox=%s).",
                delta_name,
                study_bbox,
            )
        else:
            _LOG.debug("Delta '%s': %d tile(s) selected.", delta_name, len(selected))

        all_selected.extend(selected)

    unique_selected = list(set(all_selected))
    _LOG.info(
        "Tile selection complete: %d total, %d unique.",
        len(all_selected),
        len(unique_selected),
    )
    return unique_selected


# ---------------------------------------------------------------------------
# Tile copying
# ---------------------------------------------------------------------------


def _copy_file(args: tuple[Path, Path]) -> str:
    """Copy a single file to its destination, creating parent dirs as needed.

    Args:
        args: Tuple of ``(src, dst)`` paths.

    Returns:
        The source filename (used for progress reporting by the caller).
    """
    src, dst = args
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return src.name


def copy_tiles(selected: list[Path], output_dir: Path, workers: int) -> None:
    """Copy selected tiles to *output_dir* using a thread pool.

    Already-present files are overwritten (``shutil.copy2`` semantics).
    Progress is logged every 10 files and on the final file.

    Args:
        selected: List of source tile paths to copy.
        output_dir: Destination directory; created if it does not exist.
        workers: Number of parallel copy threads.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_args: list[tuple[Path, Path]] = [
        (src, output_dir / src.name) for src in selected
    ]
    total = len(copy_args)
    _LOG.info("Copying %d file(s) to: %s", total, output_dir)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_copy_file, a) for a in copy_args]
        for future in as_completed(futures):
            done += 1
            if done % 10 == 0 or done == total:
                _LOG.info("  Copied %5d / %d", done, total)
