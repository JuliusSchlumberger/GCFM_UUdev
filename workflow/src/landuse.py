"""
Per-basin land-use source preparation (rule prepare_landuse, 02b), its
resampling onto the working grids, and roughness aggregation.

THE land-use classification every later rule reads is the per-basin
{basin}_landuse_source.tif this module builds: one raster, clipped to the
domain bbox, in WGS84 at the chosen source's own resolution, carrying
PIPELINE CODES -- the source's own class codes plus 200 for open sea, the
single sea/land criterion the whole pipeline is built on (sea_mask, both
land masks, the coastal-protection weir's ocean mask, zsini, the
water-level boundary).

Two sources, selected by config landuse.source:

  copernicus_lc100  Copernicus Global Land Service LC100, 100 m, catalogue
                    entry 'land_use'. Already carries 200 = open sea, so
                    the raster is only windowed to the domain -- codes pass
                    through unchanged.

  esa_worldcover    ESA WorldCover 2021 v200, 10 m, catalogue entry
                    'land_use_esa_worldcover'. Has NO sea class: open sea,
                    lagoons, rivers and lakes are all class 80 ("permanent
                    water bodies"), so 200 has to be derived -- see
                    _derive_sea_mask below.

Sea from WorldCover: the sea is the water you reach by swimming in from
outside the domain, and the rivers are where that stops.
  1. barrier = the basin's own clipped river network (rule 06) buffered to
     the reaches' own SWORD width, minus anything LC100 calls open sea
     (SWORD lines run out past the shoreline at the mouths, and a barrier
     there would cut a notch into the open sea).
  2. sea = connected components of WorldCover water (80) MINUS that
     barrier which (a) touch the domain window's border -- the sea enters
     the window from outside -- and (b) contain at least one pixel LC100
     calls open sea (200). (b) is the guard against an inland lake or an
     upstream river that happens to reach the border: on the Mississippi,
     38 water bodies touch the border and only 3 hold LC100 sea.
  3. plus WorldCover nodata where LC100 says sea -- open water beyond the
     WorldCover tile footprint.
Everything else keeps its WorldCover code, so lagoons cut off from the sea
by land stay 80, the river channel upstream of its mouth stays 80, and the
10 m coastline (beaches, spits, jetties: 10 km2 on basin 2433835 that LC100
called sea) stays land. Note that water genuinely open to the sea IS sea
here, including large sea-connected bays and lakes (Lake Pontchartrain).

This replaced a distance-capped variant (2026-09-14) that grew the sea from
LC100's own sea cells and cut it off beyond sea_fringe_m: that cap sliced
through physically continuous water, leaving nearshore sea classified as
inland water -- 47 km2 of it on basin 2433835, and 5,781 km2 on the
Mississippi, where no distance threshold separates sea from river and marsh
(raising it to 3.2 km still left 7,407 km2 and started leaking up-river).

MEMORY: at 10 m a delta-sized window is huge -- 2.0 GIGApixels for the
Mississippi's 466x323 km bbox (25M for the Ebro's 39x41 km). Nothing here
ever holds a full fine-resolution window, let alone the int32 label array
connected components would need for it (8.2 GB for the Mississippi):
  - the sea derivation (connected components + the rasterized river
    barrier) runs on a COARSENED working grid, capped at
    MAX_WORKING_PIXELS, and its result is applied back to the fine grid by
    intersecting with the fine water mask -- so the 10 m coastline
    survives, only the connectivity geometry is resolved coarsely;
  - every raster written here is streamed in row blocks capped at
    MAX_BLOCK_PIXELS.
Peak memory is therefore set by those two caps, not by the delta's size.

Roughness: Manning's n is a CONTINUOUS quantity, so it is never taken from
an upscaled class raster. write_roughness_raster converts the land-use
SOURCE to n at the source's own resolution and AREA-AVERAGES those values
onto the target grid (aggregate_manning). Class rasters are upscaled by
mode (most frequent class) instead -- see warp_landuse_to_grid.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from src.raster import reproject_nan_aware

log = logging.getLogger(__name__)

# THE sea code of this pipeline's own land-use convention (LC100's own
# "open sea" class; derived for WorldCover, which has no sea class).
SEA_CODE = 200
# Wetland/lagoon classes that may be left OUTSIDE the coastal protection
# when river_depth_modelling.unprotected_ocean_wetlands is on: herbaceous
# wetland (90) and permanent water bodies (80) in both schemes, plus
# WorldCover's own mangroves (95), which are intertidal by definition.
OCEAN_WETLAND_CLASSES = (80, 90, 95)

LANDUSE_NODATA = 255  # the per-basin product's nodata, both sources
ROUGHNESS_NODATA = -9999.0

SOURCES = ("copernicus_lc100", "esa_worldcover")

WORLDCOVER_WATER = 80  # "permanent water bodies" -- sea AND inland water
WORLDCOVER_NODATA = 0  # "no data" in the tiles; mapped to LANDUSE_NODATA
# ESA WorldCover v200 classes (10 m). 20-100 share both their code and
# their meaning with LC100; 10 replaces LC100's 111-126 forest classes and
# 95 (mangroves) has no LC100 equivalent -- both need a roughness lookup row.
WORLDCOVER_CLASSES = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built-up",
    60: "bare_sparse_vegetation",
    70: "snow_ice",
    80: "permanent_water_bodies",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}

# Working grid for the sea morphology: 60M px keeps the int32 label array
# at ~240 MB (the Mississippi's own 2.0 Gpx window would need 8.2 GB).
MAX_WORKING_PIXELS = 60_000_000
# Row-block budget for every streamed read/write below (~20M px: 20 MB as
# uint8, 80 MB as float32, before the warp's own temporaries).
MAX_BLOCK_PIXELS = 20_000_000

_GTIFF_CREATION = {
    "compress": "deflate",
    "tiled": True,
    "blockxsize": 512,
    "blockysize": 512,
}


# ── small helpers ────────────────────────────────────────────────────────────


def _pixel_size_m(transform, lat_deg: float) -> float:
    """
    Pixel size in metres of a WGS84 raster, taking the LARGER (N-S) of the
    two dimensions: a square degree-sized pixel is narrower E-W by
    cos(latitude) (at 40.7N, a 10 m WorldCover pixel is 9.3 m N-S but 7.0 m
    E-W). Used to report the working grid's own resolution; the larger of
    the two is the conservative choice for anything measured in whole
    pixels.
    """
    ns = float(abs(transform.e)) * 110_574.0
    ew = float(abs(transform.a)) * 111_320.0 * max(math.cos(math.radians(lat_deg)), 0.1)
    return max(ns, ew)


def _integer_window(src, wgs84_bounds) -> Window:
    """Window covering wgs84_bounds, snapped outwards to whole pixels."""
    win = from_bounds(*wgs84_bounds, transform=src.transform)
    return Window(
        col_off=math.floor(win.col_off),
        row_off=math.floor(win.row_off),
        width=math.ceil(win.width),
        height=math.ceil(win.height),
    )


def _row_blocks(height: int, width: int, max_pixels: int = MAX_BLOCK_PIXELS):
    """Yield (row_off, n_rows) row blocks holding at most max_pixels each."""
    rows = max(1, int(max_pixels // max(width, 1)))
    for row0 in range(0, height, rows):
        yield row0, min(rows, height - row0)


def _read_window(path: str | Path, wgs84_bounds: tuple[float, float, float, float]):
    """Windowed read of a WGS84 raster; returns (array, transform, nodata)."""
    with rasterio.open(path) as src:
        window = _integer_window(src, wgs84_bounds)
        arr = src.read(1, window=window, boundless=True, fill_value=src.nodata or 0)
        return arr, src.window_transform(window), src.nodata


def _lc100_on_grid(lc100_path, transform, shape) -> np.ndarray:
    """LC100 nearest-resampled onto an arbitrary WGS84 grid (pure upsampling
    at 10-60 m, so every fine pixel simply inherits the coarse class it
    falls in -- no class is invented).
    """
    out = np.full(shape, LANDUSE_NODATA, dtype=np.uint8)
    with rasterio.open(lc100_path) as lc_src:
        reproject(
            source=rasterio.band(lc_src, 1),
            destination=out,
            dst_transform=transform,
            dst_crs=lc_src.crs,
            resampling=Resampling.nearest,
        )
    return out


# ── sea derivation (ESA WorldCover) ──────────────────────────────────────────


def _river_barrier_mask(
    river_network_path: str | Path,
    transform,
    shape: tuple[int, int],
    lc100_sea: np.ndarray,
    width_factor: float,
    min_width_m: float,
) -> np.ndarray:
    """
    Rasterized river channels, the barrier that stops the sea fill at the
    river mouths.

    Each reach is buffered to half its own SWORD ``width`` (times
    ``width_factor``, at least ``min_width_m``) so the strip spans the
    channel rather than merely tracing its centreline -- an unbuffered line
    leaves water either side of it and the fill simply walks around it.
    Buffering happens in a metric CRS, then the result is rasterized onto
    the working grid.

    Anything LC100 calls open sea is removed from the barrier: SWORD's
    lines continue past the shoreline at the mouths, and a barrier out
    there would cut a notch into the open sea.
    """
    import geopandas as gpd
    from rasterio.features import rasterize

    rivers = gpd.read_file(river_network_path)
    if rivers.empty:
        log.warning(
            f"{Path(river_network_path).name} holds no reach -- no river barrier, so "
            f"the sea fill is limited only by land and by the LC100-sea guard."
        )
        return np.zeros(shape, dtype=bool)

    rivers_m = rivers.to_crs(rivers.estimate_utm_crs())
    widths = (
        rivers_m["width"].astype(float).fillna(0.0)
        if "width" in rivers_m.columns
        else pd.Series(np.zeros(len(rivers_m)))
    )
    radius = np.maximum(
        0.5 * widths.to_numpy() * float(width_factor), float(min_width_m)
    )
    buffered = gpd.GeoSeries(rivers_m.buffer(radius), crs=rivers_m.crs).to_crs(
        "EPSG:4326"
    )

    barrier = rasterize(
        [
            (geom, 1)
            for geom in buffered.geometry
            if geom is not None and not geom.is_empty
        ],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)
    barrier &= ~lc100_sea
    log.info(
        f"River barrier: {len(rivers)} reach(es), width x{width_factor} "
        f"(min {min_width_m:.0f} m) -> {int(barrier.sum()):,} working-grid cell(s)"
    )
    return barrier


def _derive_sea_mask(
    water: np.ndarray,
    nodata_mask: np.ndarray,
    lc100_sea: np.ndarray,
    barrier: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Open-sea mask on the working grid: water components that reach the
    window border AND hold LC100 sea, with the river barrier removed first.
    See this module's docstring.
    """
    from scipy import ndimage

    labels, n_components = ndimage.label(water & ~barrier)
    border = np.zeros(labels.shape, dtype=bool)
    border[0, :], border[-1, :], border[:, 0], border[:, -1] = True, True, True, True

    at_border = set(np.unique(labels[border]).tolist()) - {0}
    with_lc100_sea = set(np.unique(labels[lc100_sea]).tolist()) - {0}
    sea_labels = sorted(at_border & with_lc100_sea)
    if not sea_labels:
        log.warning(
            f"_derive_sea_mask: none of the {n_components:,} water bodies both reaches "
            f"the domain border ({len(at_border):,} do) and holds an LC100 open-sea pixel "
            f"({len(with_lc100_sea):,} do) -- inland basin, or the sources disagree; no "
            f"cell will be classified as open sea."
        )
    sea = np.isin(labels, sea_labels)
    del labels
    sea |= nodata_mask & lc100_sea

    stats = {
        "working_water_px": int(water.sum()),
        "working_lc100_sea_px": int(lc100_sea.sum()),
        "working_barrier_px": int(barrier.sum()),
        "working_sea_px": int(sea.sum()),
        "water_bodies": int(n_components),
        "bodies_at_border": len(at_border),
        "bodies_kept_as_sea": len(sea_labels),
    }
    return sea, stats


def _write_worldcover_source(
    worldcover_path,
    lc100_path,
    wgs84_bounds,
    out_path,
    river_network_path,
    river_barrier_width_factor,
    river_barrier_min_width_m,
) -> tuple[dict, dict]:
    """Stream the WorldCover-derived per-basin land use to out_path."""
    with rasterio.open(worldcover_path) as wc_src:
        window = _integer_window(wc_src, wgs84_bounds)
        h, w = int(window.height), int(window.width)
        transform_fine = wc_src.window_transform(window)
        wc_nodata = wc_src.nodata if wc_src.nodata is not None else WORLDCOVER_NODATA
        lat_mid = 0.5 * (wgs84_bounds[1] + wgs84_bounds[3])

        # 1. coarsened working grid for the morphology only
        factor = max(1, math.ceil(math.sqrt(h * w / MAX_WORKING_PIXELS)))
        ch, cw = math.ceil(h / factor), math.ceil(w / factor)
        transform_coarse = transform_fine * Affine.scale(w / cw, h / ch)
        wc_coarse = wc_src.read(
            1,
            window=window,
            out_shape=(ch, cw),
            resampling=Resampling.mode,
            boundless=True,
            fill_value=wc_nodata,
        )
        px_m_fine = _pixel_size_m(transform_fine, lat_mid)
        px_m_work = _pixel_size_m(transform_coarse, lat_mid)
        if factor > 1:
            log.info(
                f"Sea morphology on a coarsened working grid: {w}x{h} px @ "
                f"{px_m_fine:.1f} m -> {cw}x{ch} px @ {px_m_work:.1f} m (factor "
                f"{factor}, {MAX_WORKING_PIXELS / 1e6:.0f}M px budget); the 10 m "
                f"coastline is preserved by intersecting the result with the fine "
                f"water mask."
            )

        unexpected = (
            set(np.unique(wc_coarse).tolist())
            - set(WORLDCOVER_CLASSES)
            - {int(wc_nodata)}
        )
        if unexpected:
            log.warning(
                f"Codes outside the WorldCover legend {sorted(unexpected)} -- passed "
                f"through unchanged; check the tiles/VRT (they have no roughness "
                f"lookup row either)."
            )

        lc_sea_coarse = (
            _lc100_on_grid(lc100_path, transform_coarse, (ch, cw)) == SEA_CODE
        )
        barrier_coarse = (
            _river_barrier_mask(
                river_network_path,
                transform_coarse,
                (ch, cw),
                lc_sea_coarse,
                river_barrier_width_factor,
                river_barrier_min_width_m,
            )
            if river_network_path is not None
            else np.zeros((ch, cw), dtype=bool)
        )
        sea_coarse, stats = _derive_sea_mask(
            wc_coarse == WORLDCOVER_WATER,
            wc_coarse == wc_nodata,
            lc_sea_coarse,
            barrier_coarse,
        )
        del wc_coarse, barrier_coarse
        stats.update(
            working_factor=factor,
            working_resolution_m=round(px_m_work, 2),
            source_resolution_m=round(px_m_fine, 2),
        )

        # 2. stream the fine raster, applying the coarse sea mask
        meta = {
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 1,
            "height": h,
            "width": w,
            "crs": "EPSG:4326",
            "transform": transform_fine,
            "nodata": LANDUSE_NODATA,
            **_GTIFF_CREATION,
        }
        counts = {
            "sea_px": 0,
            "inland_water_px": 0,
            "land_inside_lc100_sea_px": 0,
            "nodata_px": 0,
        }
        col_idx = np.minimum(np.arange(w) // factor, cw - 1)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **meta) as dst:
            for row0, nrows in _row_blocks(h, w):
                blk = wc_src.read(
                    1,
                    window=Window(window.col_off, window.row_off + row0, w, nrows),
                    boundless=True,
                    fill_value=wc_nodata,
                )
                row_idx = np.minimum(np.arange(row0, row0 + nrows) // factor, ch - 1)
                sea_up = sea_coarse[row_idx][:, col_idx]
                lc_sea_up = lc_sea_coarse[row_idx][:, col_idx]

                water = blk == WORLDCOVER_WATER
                nodata_mask = blk == wc_nodata
                # The fine water mask is what keeps the 10 m coastline: a
                # pixel is sea only where WorldCover itself says water.
                sea = (water & sea_up) | (nodata_mask & lc_sea_up)

                codes = blk.astype(np.uint8, copy=True)
                codes[nodata_mask & ~sea] = LANDUSE_NODATA
                codes[sea] = SEA_CODE
                dst.write(codes, 1, window=Window(0, row0, w, nrows))

                counts["sea_px"] += int(sea.sum())
                counts["inland_water_px"] += int((water & ~sea).sum())
                counts["land_inside_lc100_sea_px"] += int(
                    (~water & ~nodata_mask & lc_sea_up).sum()
                )
                counts["nodata_px"] += int((codes == LANDUSE_NODATA).sum())
            dst.update_tags(landuse_source="esa_worldcover", sea_code=str(SEA_CODE))
    stats.update(counts)
    return meta, stats


def write_landuse_source(
    source: str,
    wgs84_bounds: tuple[float, float, float, float],
    lc100_path: str | Path,
    out_path: str | Path,
    worldcover_path: str | Path | None = None,
    river_network_path: str | Path | None = None,
    river_barrier_width_factor: float = 1.0,
    river_barrier_min_width_m: float = 30.0,
) -> tuple[dict, dict]:
    """
    Write the per-basin land-use raster in pipeline codes for ``wgs84_bounds``.

    Args:
        source:          "copernicus_lc100" or "esa_worldcover".
        wgs84_bounds:    (lon_min, lat_min, lon_max, lat_max) clip extent.
        lc100_path:      Global LC100 raster -- the product itself for
                         "copernicus_lc100", the SEA REFERENCE for
                         "esa_worldcover" (needed either way).
        out_path:        Destination GeoTIFF.
        worldcover_path: WorldCover VRT mosaic (esa_worldcover only).
        river_network_path: This basin's own clipped river network (rule
                         get_river_network, 06) -- the barrier that stops
                         the sea fill at the river mouths (esa_worldcover
                         only; without it the sea runs up every channel).
        river_barrier_width_factor, river_barrier_min_width_m:
                         barrier width per reach = max(width/2 * factor,
                         min_width_m).

    Returns:
        (meta, stats): the written raster's GeoTIFF meta and a dict of
        diagnostics for logging.
    """
    if source not in SOURCES:
        raise ValueError(f"landuse.source must be one of {SOURCES}, got {source!r}")

    if source == "esa_worldcover":
        if worldcover_path is None:
            raise ValueError("source 'esa_worldcover' needs worldcover_path")
        if river_network_path is None:
            log.warning(
                "No river_network_path: the sea fill has no river barrier, so it will "
                "run up every channel connected to the sea."
            )
        return _write_worldcover_source(
            worldcover_path,
            lc100_path,
            wgs84_bounds,
            out_path,
            river_network_path,
            river_barrier_width_factor,
            river_barrier_min_width_m,
        )

    # LC100: one window, already in pipeline codes (100 m -- 20M px even for
    # the largest delta bbox, so no streaming needed).
    arr, transform, nodata = _read_window(lc100_path, wgs84_bounds)
    codes = arr.astype(np.uint8, copy=True)
    if nodata is not None and int(nodata) != LANDUSE_NODATA:
        codes[arr == nodata] = LANDUSE_NODATA
    meta = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "height": codes.shape[0],
        "width": codes.shape[1],
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": LANDUSE_NODATA,
        **_GTIFF_CREATION,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(codes, 1)
        dst.update_tags(landuse_source=source, sea_code=str(SEA_CODE))
    stats = {
        "sea_px": int((codes == SEA_CODE).sum()),
        "nodata_px": int((codes == LANDUSE_NODATA).sum()),
        "source_resolution_m": round(
            _pixel_size_m(transform, 0.5 * (wgs84_bounds[1] + wgs84_bounds[3])), 2
        ),
    }
    return meta, stats


# ── resampling onto the working grids ────────────────────────────────────────


def _source_block_window(
    src, dst_transform, dst_crs, nrows: int, ncols: int, margin_px: int = 2
):
    """Source window (+margin) covering one destination block's footprint."""
    left, bottom, right, top = array_bounds(nrows, ncols, dst_transform)
    wgs = transform_bounds(dst_crs, src.crs, left, bottom, right, top, densify_pts=21)
    win = _integer_window(src, wgs)
    return Window(
        win.col_off - margin_px,
        win.row_off - margin_px,
        win.width + 2 * margin_px,
        win.height + 2 * margin_px,
    )


def warp_landuse_to_grid(
    landuse_source_path: str | Path, ref_meta: dict, out_path: str | Path
) -> dict:
    """
    Resample the land-use source onto ref_meta's grid by MODE (most frequent
    class), streamed in row blocks.

    Mode, not nearest: once the source is finer than the target (ESA
    WorldCover's 10 m against a ~30 m elevation grid) nearest keeps one
    arbitrary source pixel per cell and discards the rest. Classes stay
    categorical either way -- the continuous quantity derived from them,
    Manning's n, is area-averaged instead (write_roughness_raster).

    Returns:
        Dict of class -> pixel count in the written raster.
    """
    meta = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "height": ref_meta["height"],
        "width": ref_meta["width"],
        "crs": ref_meta["crs"],
        "transform": ref_meta["transform"],
        "nodata": LANDUSE_NODATA,
        **_GTIFF_CREATION,
    }
    histogram: dict[int, int] = {}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with (
        rasterio.open(landuse_source_path) as src,
        rasterio.open(out_path, "w", **meta) as dst,
    ):
        src_nodata = src.nodata if src.nodata is not None else LANDUSE_NODATA
        # Carry the source's own identity along, so every plot of a derived
        # raster can name it (src.plots.plot_landuse).
        dst.update_tags(
            **{
                k: v
                for k, v in src.tags().items()
                if k in ("landuse_source", "sea_code")
            }
        )
        for row0, nrows in _row_blocks(meta["height"], meta["width"]):
            block_transform = meta["transform"] * Affine.translation(0, row0)
            win = _source_block_window(
                src, block_transform, meta["crs"], nrows, meta["width"]
            )
            block = np.full((nrows, meta["width"]), LANDUSE_NODATA, dtype=np.uint8)
            reproject(
                source=src.read(1, window=win, boundless=True, fill_value=src_nodata),
                destination=block,
                src_transform=src.window_transform(win),
                src_crs=src.crs,
                dst_transform=block_transform,
                dst_crs=meta["crs"],
                src_nodata=src_nodata,
                dst_nodata=LANDUSE_NODATA,
                resampling=Resampling.mode,
            )
            dst.write(block, 1, window=Window(0, row0, meta["width"], nrows))
            values, counts = np.unique(block, return_counts=True)
            for v, c in zip(values.tolist(), counts.tolist()):
                histogram[int(v)] = histogram.get(int(v), 0) + int(c)
    return histogram


# ── roughness ────────────────────────────────────────────────────────────────


def landuse_to_manning(
    landuse: np.ndarray, lu_to_n: dict[int, float], nodata: int = LANDUSE_NODATA
) -> tuple[np.ndarray, set[int]]:
    """
    Land-use codes -> Manning's n (float32, NaN where unmapped/nodata).

    Returns:
        (n, unmapped_codes) -- unmapped codes are NaN in the output, so they
        neither contribute to nor bias an area average.
    """
    n = np.full(landuse.shape, np.nan, dtype=np.float32)
    for code, value in lu_to_n.items():
        if value > 0:  # the lookup carries -999 for "no data" rows
            n[landuse == code] = value
    unmapped = set(np.unique(landuse).tolist()) - set(lu_to_n) - {int(nodata)}
    return n, unmapped


def read_roughness_lookup(lookup_path: str | Path) -> dict[int, float]:
    """Land-use code -> Manning's n from the lookup CSV."""
    lookup = pd.read_csv(lookup_path)
    return dict(
        zip(
            lookup["copernicus_worldcover"].astype(int),
            lookup["manning_n"].astype(float),
        )
    )


def aggregate_manning(
    n_fine: np.ndarray,
    src_transform,
    src_crs,
    dst_shape: tuple[int, int],
    dst_transform,
    dst_crs,
    method: str = "area_mean",
    resampling=Resampling.average,
) -> np.ndarray:
    """
    Area-average Manning's n from a fine grid onto a target grid.

    The alternative to upscaling the CLASS raster and reclassifying it:
    a 70 m cell that is 60% water and 40% built-up gets its dominant class'
    0.02 that way, against 0.05 here. On basin 2433835, 30% of 70 m cells
    contain more than one class, and in 43% of those the two differ by more
    than 0.01.

    Args:
        method: "area_mean"  -- area-weighted mean of n (default).
                "conveyance" -- area-weighted HARMONIC mean, i.e. the
                composite n of parallel strips carrying flow at equal depth
                (1/n_eff = sum f_i/n_i), matching what hydromt's subgrid
                tables do per cell. Biased towards the smoothest pixels
                (half water + half built-up gives 0.033, not 0.060), so it
                is not the default.
        resampling: Resampling.average (default) genuinely area-averages,
                which is only meaningful when the source is FINER than the
                target. Callers upsampling a coarser source (LC100's 100 m
                onto a 31 m grid) must pass Resampling.nearest: averaging
                there blends neighbouring classes into intermediate n along
                every class boundary, inventing sub-pixel detail the source
                does not have.

    Returns:
        float32 array of shape dst_shape, NaN where no valid source pixel
        contributes.
    """
    if method not in ("area_mean", "conveyance"):
        raise ValueError(f"method must be 'area_mean' or 'conveyance', got {method!r}")

    # Averaging 1/n and inverting = harmonic mean of n. reproject_nan_aware
    # weights by valid-pixel coverage, so nodata neither pulls the mean
    # towards zero nor spreads into cells that do have data.
    field = (
        n_fine
        if method == "area_mean"
        else np.where(np.isnan(n_fine), np.nan, 1.0 / n_fine)
    )
    out = reproject_nan_aware(
        field.astype(np.float32),
        src_transform,
        src_crs,
        dst_shape,
        dst_transform,
        dst_crs,
        resampling=resampling,
    )
    if method == "conveyance":
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(out > 0, 1.0 / out, np.nan).astype(np.float32)
    return out


def write_roughness_raster(
    landuse_source_path: str | Path,
    lookup_path: str | Path,
    ref_meta: dict,
    out_path: str | Path,
    method: str = "area_mean",
    refine_to_source: bool = False,
) -> tuple[set[int], dict]:
    """
    Manning's n area-averaged from the land-use SOURCE at its own (possibly
    much finer) resolution onto ref_meta's grid -- never reclassified from
    an already-upscaled class raster. Streamed in row blocks.

    Args:
        refine_to_source: if True, write on ref_meta's grid SUBDIVIDED by an
            integer factor so the cells are about the size of the source's
            own pixels (rule get_roughness, 05c: ~10 m under ESA WorldCover
            instead of the ~31 m elevation grid, since hydromt samples this
            raster per SUBGRID pixel -- 7 m at nr_subgrid_pixels=10 -- and
            averages it by conveyance itself). The factor is 1, i.e. the
            reference grid unchanged, whenever the source is coarser
            (LC100's 100 m) -- upsampling a class raster adds no
            information. Keeping it an exact subdivision of the reference
            grid preserves CRS and alignment with elevation/subgrid.
        method: see aggregate_manning.

    Returns:
        (unmapped_codes, info) -- codes present in the source but missing
        from the lookup (NaN in the output), and the grid actually written.
    """
    lu_to_n = read_roughness_lookup(lookup_path)
    ref_res_m = abs(ref_meta["transform"].a)

    with rasterio.open(landuse_source_path) as src:
        src_bounds = src.bounds
        src_res_m = _pixel_size_m(
            src.transform, 0.5 * (src_bounds.bottom + src_bounds.top)
        )
        factor = 1
        if refine_to_source and src_res_m > 0:
            factor = max(1, int(round(ref_res_m / src_res_m)))

        height, width = ref_meta["height"] * factor, ref_meta["width"] * factor
        transform = ref_meta["transform"] * Affine.scale(1 / factor, 1 / factor)
        # Average only when the source really is finer than the target;
        # upsampling a coarser source (LC100's 100 m onto a 31 m grid) with
        # averaging would blend classes into intermediate n along every
        # boundary, inventing detail LC100 does not have. Nearest there
        # reproduces the plain class-value reclassification exactly.
        target_res_m = ref_res_m / factor
        downsampling = src_res_m <= target_res_m
        resampling = Resampling.average if downsampling else Resampling.nearest
        meta = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "height": height,
            "width": width,
            "crs": ref_meta["crs"],
            "transform": transform,
            "nodata": ROUGHNESS_NODATA,
            **_GTIFF_CREATION,
        }
        src_nodata = int(src.nodata) if src.nodata is not None else LANDUSE_NODATA

        unmapped: set[int] = set()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **meta) as dst:
            for row0, nrows in _row_blocks(height, width):
                block_transform = transform * Affine.translation(0, row0)
                win = _source_block_window(
                    src, block_transform, meta["crs"], nrows, width
                )
                lu_block = src.read(
                    1, window=win, boundless=True, fill_value=src_nodata
                )
                n_fine, missing = landuse_to_manning(
                    lu_block, lu_to_n, nodata=src_nodata
                )
                unmapped |= missing
                n_block = aggregate_manning(
                    n_fine,
                    src.window_transform(win),
                    src.crs,
                    (nrows, width),
                    block_transform,
                    meta["crs"],
                    method=method,
                    resampling=resampling,
                )
                dst.write(
                    np.where(np.isfinite(n_block), n_block, ROUGHNESS_NODATA).astype(
                        np.float32
                    ),
                    1,
                    window=Window(0, row0, width, nrows),
                )
    info = {
        "resolution_m": round(target_res_m, 2),
        "refine_factor": factor,
        "shape": (height, width),
        "source_resolution_m": round(src_res_m, 2),
        "method": method if downsampling else "nearest (source coarser than the grid)",
    }
    return unmapped, info
