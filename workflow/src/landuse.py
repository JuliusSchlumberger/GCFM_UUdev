"""Build the per-basin land-use source (rule prepare_landuse), warp it to grids, derive roughness.

{basin}_landuse_source.tif is THE land-use classification every later rule
reads: WGS84, the source's own resolution, clipped to the domain bbox, in
pipeline codes -- the source's class codes plus 200 = open sea, the single
sea/land criterion of the pipeline (sea_mask, land masks, the coastal weir's
ocean mask, zsini, the water-level boundary). Sources (config landuse.source):

  copernicus_lc100  LC100, 100 m, catalogue 'land_use'; already has 200 = open
                    sea, codes pass through unchanged.
  esa_worldcover    WorldCover 2021 v200, 10 m, catalogue
                    'land_use_esa_worldcover'; no sea class (sea, lagoons,
                    rivers, lakes are all 80), so 200 is derived:
    1. barrier = the basin's clipped river network (rule 06) buffered to the
       reaches' SWORD width, minus LC100 open sea (SWORD lines run past the
       shoreline at the mouths).
    2. sea = connected components of WorldCover water (80) minus the barrier
       that touch the window border AND contain an LC100 open-sea pixel (200);
       the latter guards against inland lakes/rivers reaching the border.
    3. plus WorldCover nodata where LC100 says sea (beyond the tile footprint).
  Everything else keeps its WorldCover code: cut-off lagoons and upstream
  channels stay 80, the 10 m coastline stays land; water genuinely open to the
  sea (e.g. Lake Pontchartrain) is sea.

Memory: the sea derivation runs on a coarsened working grid capped at
MAX_WORKING_PIXELS and is applied back to the fine grid by intersecting with
the fine water mask (the 10 m coastline survives); every raster is streamed
in blocks capped at MAX_BLOCK_PIXELS.

Roughness: Manning's n is continuous, so it is converted at the source
resolution and area-averaged onto the target grid (aggregate_manning); class
rasters are upscaled by mode (warp_landuse_to_grid).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from src.raster import reproject_nan_aware

log = logging.getLogger(__name__)

# LC100's "open sea" class; derived for WorldCover, which has no sea class.
SEA_CODE = 200
# Classes that may be left outside the coastal protection when
# river_depth_modelling.unprotected_ocean_wetlands is on: herbaceous wetland
# (90), permanent water bodies (80), WorldCover mangroves (95, intertidal).
OCEAN_WETLAND_CLASSES = (80, 90, 95)

LANDUSE_NODATA = 255  # per-basin product's nodata, both sources
ROUGHNESS_NODATA = -9999.0

SOURCES = ("copernicus_lc100", "esa_worldcover")

WORLDCOVER_WATER = 80  # "permanent water bodies" -- sea AND inland water
WORLDCOVER_NODATA = 0  # "no data" in the tiles; mapped to LANDUSE_NODATA
# ESA WorldCover v200 classes (10 m). 20-100 share code and meaning with
# LC100; 10 stands for LC100's 111-126 forest classes and 95 (mangroves) has
# no LC100 equivalent -- both need a roughness lookup row.
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

# Sea-morphology working grid: 60M px keeps the int32 label array at ~240 MB
# (the Mississippi's 2.0 Gpx 10 m window would need 8.2 GB).
MAX_WORKING_PIXELS = 60_000_000
# Block budget for every streamed read/write (~20M px: 20 MB as uint8,
# 80 MB as float32, before the warp's own temporaries).
MAX_BLOCK_PIXELS = 20_000_000

_GTIFF_CREATION = {
    "compress": "deflate",
    "tiled": True,
    "blockxsize": 512,
    "blockysize": 512,
}


def _pixel_size_m(transform, lat_deg: float) -> float:
    # Larger of N-S and E-W (E-W shrinks by cos(lat)): conservative for
    # anything measured in whole pixels.
    ns = float(abs(transform.e)) * 110_574.0
    ew = float(abs(transform.a)) * 111_320.0 * max(math.cos(math.radians(lat_deg)), 0.1)
    return max(ns, ew)


def _integer_window(src, wgs84_bounds) -> Window:
    win = from_bounds(*wgs84_bounds, transform=src.transform)
    return Window(
        col_off=math.floor(win.col_off),
        row_off=math.floor(win.row_off),
        width=math.ceil(win.width),
        height=math.ceil(win.height),
    )


def _row_blocks(height: int, width: int, max_pixels: int = MAX_BLOCK_PIXELS):
    rows = max(1, int(max_pixels // max(width, 1)))
    for row0 in range(0, height, rows):
        yield row0, min(rows, height - row0)


def _square_tiles(
    height: int,
    width: int,
    dst_res_m: float,
    src_res_m: float,
    max_source_pixels: int = MAX_BLOCK_PIXELS,
):
    # Square tiles, not row strips: on a rotated destination grid the
    # axis-aligned source box around a full-width strip is nearly the whole
    # domain; around a square it is at most 2x the square's area (the factor 2).
    ratio = max(dst_res_m, 1e-9) / max(src_res_m, 1e-9)  # source px per destination px
    side = max(1, int(math.sqrt(max_source_pixels / 2.0) / max(ratio, 1.0)))
    for row0 in range(0, height, side):
        for col0 in range(0, width, side):
            yield row0, col0, min(side, height - row0), min(side, width - col0)


def _read_window(path: str | Path, wgs84_bounds: tuple[float, float, float, float]):
    with rasterio.open(path) as src:
        window = _integer_window(src, wgs84_bounds)
        arr = src.read(1, window=window, boundless=True, fill_value=src.nodata or 0)
        return arr, src.window_transform(window), src.nodata


def _lc100_on_grid(lc100_path, transform, shape) -> np.ndarray:
    # Nearest: pure upsampling at 10-60 m, so no class is invented.
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


def _river_barrier_mask(
    river_network_path: str | Path,
    transform,
    shape: tuple[int, int],
    lc100_sea: np.ndarray,
    width_factor: float,
    min_width_m: float,
) -> np.ndarray:
    # Buffered to the channel width: an unbuffered centreline leaves water on
    # either side and the sea fill walks around it.
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
    with rasterio.open(worldcover_path) as wc_src:
        window = _integer_window(wc_src, wgs84_bounds)
        h, w = int(window.height), int(window.width)
        transform_fine = wc_src.window_transform(window)
        wc_nodata = wc_src.nodata if wc_src.nodata is not None else WORLDCOVER_NODATA
        lat_mid = 0.5 * (wgs84_bounds[1] + wgs84_bounds[3])

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
                # Sea only where WorldCover itself says water: keeps the 10 m coastline.
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
    """Write the per-basin land-use raster in pipeline codes; return (meta, stats).

    wgs84_bounds: (lon_min, lat_min, lon_max, lat_max). lc100_path is the
    product for "copernicus_lc100" and the sea reference for "esa_worldcover"
    (needed either way). river_network_path (esa_worldcover only, rule
    get_river_network): barrier stopping the sea fill at the river mouths;
    without it the sea runs up every channel. Barrier buffer radius per reach
    = max(width/2 * river_barrier_width_factor, river_barrier_min_width_m).
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

    # LC100 is already in pipeline codes; at 100 m even the largest delta bbox
    # is ~20M px, so no streaming.
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


def _source_block_window(
    src, dst_transform, dst_crs, nrows: int, ncols: int, margin_px: int = 2
):
    # Corner-based, not rasterio's array_bounds: that one returns south > north
    # for an unrotated SOUTH-UP block (the unrotated model grid).
    xs, ys = zip(
        *(dst_transform * c for c in [(0, 0), (ncols, 0), (ncols, nrows), (0, nrows)])
    )
    wgs = transform_bounds(
        dst_crs, src.crs, min(xs), min(ys), max(xs), max(ys), densify_pts=21
    )
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
    """Resample the land-use source onto ref_meta's grid by mode; return class -> pixel count.

    Mode, not nearest: with a finer source (WorldCover 10 m vs a ~30 m grid)
    nearest keeps one arbitrary source pixel per cell.
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
        # Carry the source identity so plots of derived rasters can name it.
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


def landuse_to_manning(
    landuse: np.ndarray, lu_to_n: dict[int, float], nodata: int = LANDUSE_NODATA
) -> tuple[np.ndarray, set[int]]:
    """Map land-use codes to Manning's n; return (n, unmapped_codes).

    n is float32, NaN where unmapped/nodata, so those cells do not bias an
    area average.
    """
    n = np.full(landuse.shape, np.nan, dtype=np.float32)
    for code, value in lu_to_n.items():
        if value > 0:  # the lookup carries -999 for "no data" rows
            n[landuse == code] = value
    unmapped = set(np.unique(landuse).tolist()) - set(lu_to_n) - {int(nodata)}
    return n, unmapped


def read_roughness_lookup(lookup_path: str | Path) -> dict[int, float]:
    """Read the land-use code -> Manning's n lookup CSV."""
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
    """Area-average Manning's n from a fine grid onto a target grid.

    method: "area_mean" (area-weighted mean of n) or "conveyance"
    (area-weighted harmonic mean, 1/n_eff = sum f_i/n_i: parallel strips at
    equal depth, as hydromt's subgrid tables do per cell; biased towards the
    smoothest pixels -- half water + half built-up gives 0.033, not 0.060).
    resampling: Resampling.average is only meaningful for a source finer than
    the target; upsampling a coarser source (LC100 100 m onto 31 m) needs
    Resampling.nearest, else classes blend into invented intermediate n.
    Returns float32, NaN where no valid source pixel contributes.
    """
    if method not in ("area_mean", "conveyance"):
        raise ValueError(f"method must be 'area_mean' or 'conveyance', got {method!r}")

    # reproject_nan_aware weights by valid-pixel coverage, so nodata neither
    # pulls the mean towards zero nor spreads into cells with data.
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
    """Write Manning's n area-averaged from the land-use source onto ref_meta's grid.

    refine_to_source: write on ref_meta's grid subdivided by an integer
    factor to about the source pixel size (~10 m under WorldCover, since
    hydromt samples this raster per subgrid pixel and averages by conveyance
    itself); factor 1 when the source is coarser (LC100 100 m). An exact
    subdivision keeps CRS and alignment with elevation/subgrid.
    Returns (codes missing from the lookup, NaN in the output; grid info).
    """
    lu_to_n = read_roughness_lookup(lookup_path)
    # hypot, not abs(a): the reference grid may be rotated (the model grid is).
    ref_res_m = math.hypot(ref_meta["transform"].a, ref_meta["transform"].d)

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
        # Average only for a finer source; for a coarser one nearest reproduces
        # the plain class-value reclassification (see aggregate_manning).
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
            # Square tiles: the target may be the rotated model grid (see _square_tiles).
            for row0, col0, nrows, ncols in _square_tiles(
                height, width, target_res_m, src_res_m
            ):
                block_transform = transform * Affine.translation(col0, row0)
                win = _source_block_window(
                    src, block_transform, meta["crs"], nrows, ncols
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
                    (nrows, ncols),
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
                    window=Window(col0, row0, ncols, nrows),
                )
    info = {
        "resolution_m": round(target_res_m, 2),
        "refine_factor": factor,
        "shape": (height, width),
        "source_resolution_m": round(src_res_m, 2),
        "method": method if downsampling else "nearest (source coarser than the grid)",
    }
    return unmapped, info
