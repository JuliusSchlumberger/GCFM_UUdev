"""
build_esa_worldcover_vrt.py — build the ESA WorldCover VRT from downloaded tiles.

The `land_use_esa_worldcover` entry in config/data_catalogue.yml points at a
single file:

    <raw_data_root>/ESA_Worldcover/esa_worldcover_2021_v200.vrt

That .vrt does not ship with the tiles: it is a small XML index listing every
tile and its extent, so GDAL can treat the 3x3 degree GeoTIFFs as one seamless
raster and open only the tiles overlapping the window being read. It stores
paths to tiles on *this* machine, so it has to be generated locally -- exactly
like tools/build_deltadtm_mask_vrt.py does for the DeltaDTM mask.

Re-run this after downloading more tiles; the VRT is just an index, rebuilding
is cheap. Only finished `*_Map.tif` tiles are picked up -- partial downloads
(e.g. `..._Map.tif.a1B2c3`) are ignored.

Used by rule prepare_landuse (02b) when config landuse.source == esa_worldcover.

Usage:
    python tools/build_esa_worldcover_vrt.py                 # default paths
    python tools/build_esa_worldcover_vrt.py --check-deltas  # + per-delta coverage report
    python tools/build_esa_worldcover_vrt.py --dry-run
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

import yaml

# Catalogue-declared properties of this dataset, kept here so the VRT we build
# is guaranteed consistent with what data_catalogue.yml promises downstream.
NODATA = 0  # raster_properties.nodata_value ("no data" class)
EXPECTED_DTYPE = "uint8"  # Byte
EXPECTED_EPSG = 4326
TILE_DEG = 3  # WorldCover tiles are 3x3 degrees
TILE_GLOB = "ESA_WorldCover_*_Map.tif"
TILE_RE = re.compile(r"_(?P<ns>[NS])(?P<lat>\d{2})(?P<ew>[EW])(?P<lon>\d{3})_Map\.tif$")

CATALOGUE_NAME = "land_use_esa_worldcover"


def catalogue_root_and_relpath(repo_root: Path, name: str) -> tuple[Path, str]:
    """Read meta.root and a dataset's file_path straight from the catalogue."""
    cat = yaml.safe_load(
        (repo_root / "config" / "data_catalogue.yml").read_text(encoding="utf-8")
    )
    # GCFM_RAW_DATA_ROOT overrides the committed root — same rule as 00_common.smk.
    root = Path(os.environ.get("GCFM_RAW_DATA_ROOT") or cat["meta"]["root"])
    for ds in cat["datasets"]:
        if ds["name"] == name:
            return root, ds["file_path"]
    raise KeyError(f"{name} not found in config/data_catalogue.yml")


def find_tiles(tile_dir: Path) -> list[Path]:
    tiles = sorted(p for p in tile_dir.glob(TILE_GLOB) if p.is_file())
    if not tiles:
        raise FileNotFoundError(
            f"No ESA WorldCover tiles ({TILE_GLOB}) found in {tile_dir}\n"
            f"Download the WorldCover 2021 v200 Map tiles into that folder first."
        )
    return tiles


def tile_name(lat: int, lon: int) -> str:
    """SW-corner degrees -> tile file name (lat/lon snapped to the 3-degree grid)."""
    ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
    return f"ESA_WorldCover_10m_2021_v200_{ns}{abs(lat):02d}{ew}{abs(lon):03d}_Map.tif"


def tiles_for_bounds(
    bounds: tuple[float, float, float, float],
) -> list[tuple[int, int]]:
    """SW corners of every 3x3 degree tile overlapping (lon_min, lat_min, lon_max, lat_max)."""
    lon_min, lat_min, lon_max, lat_max = bounds
    out = []
    lat0 = int(math.floor(lat_min / TILE_DEG) * TILE_DEG)
    lon0 = int(math.floor(lon_min / TILE_DEG) * TILE_DEG)
    for lat in range(lat0, int(math.ceil(lat_max / TILE_DEG) * TILE_DEG), TILE_DEG):
        for lon in range(lon0, int(math.ceil(lon_max / TILE_DEG) * TILE_DEG), TILE_DEG):
            out.append((lat, lon))
    return out


def check_delta_coverage(
    repo_root: Path, tile_dir: Path, buffer_deg: float = 0.05
) -> None:
    """Report delta polygons whose bbox is not fully covered by downloaded tiles."""
    import geopandas as gpd

    root, relpath = catalogue_root_and_relpath(repo_root, "delta_polygons")
    deltas = gpd.read_file(root / relpath).to_crs("EPSG:4326")
    id_col = "BasinID2" if "BasinID2" in deltas.columns else deltas.columns[0]
    print(f"\nchecking tile coverage of {len(deltas)} delta polygon(s) from {relpath}:")
    incomplete = 0
    for row in deltas.itertuples(index=False):
        geom = row.geometry
        b = geom.bounds
        bounds = (
            b[0] - buffer_deg,
            b[1] - buffer_deg,
            b[2] + buffer_deg,
            b[3] + buffer_deg,
        )
        missing = [
            tile_name(lat, lon)
            for lat, lon in tiles_for_bounds(bounds)
            if not (tile_dir / tile_name(lat, lon)).is_file()
        ]
        basin = getattr(row, id_col, "?")
        if missing:
            incomplete += 1
            print(
                f"  basin {basin}: MISSING {len(missing)} tile(s): {', '.join(missing)}"
            )
        else:
            print(f"  basin {basin}: complete")
    if incomplete:
        print(
            f"\n  {incomplete} delta(s) not fully covered. Ocean gaps are harmless "
            f"(WorldCover has no open-ocean tiles; rule prepare_landuse falls back to "
            f"LC100's own sea class there), but a MISSING LAND tile would silently "
            f"become nodata."
        )


def build_vrt(tiles: list[Path], out_vrt: Path) -> None:
    """
    Build the VRT with GDAL, falling back to the gdalbuildvrt CLI. Nearest
    neighbour is implicit and correct: these are categorical class codes, which
    must never be averaged or blended.
    """
    out_vrt.parent.mkdir(parents=True, exist_ok=True)

    try:
        from osgeo import gdal
    except ImportError:
        gdal = None

    if gdal is not None:
        gdal.UseExceptions()
        opts = gdal.BuildVRTOptions(
            srcNodata=NODATA, VRTNodata=NODATA, resolution="highest"
        )
        vrt = gdal.BuildVRT(str(out_vrt), [str(t) for t in tiles], options=opts)
        if vrt is None:
            raise RuntimeError("gdal.BuildVRT returned None")
        vrt.FlushCache()
        vrt = None  # closes and writes the file
        return

    import subprocess

    listing = out_vrt.parent / "_vrt_tile_list.txt"
    listing.write_text("\n".join(str(t) for t in tiles), encoding="utf-8")
    cmd = [
        "gdalbuildvrt",
        "-input_file_list",
        str(listing),
        "-srcnodata",
        str(NODATA),
        "-vrtnodata",
        str(NODATA),
        str(out_vrt),
    ]
    print("  (osgeo not importable — falling back to CLI)")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    listing.unlink(missing_ok=True)


def validate(out_vrt: Path) -> None:
    """Open the finished VRT and check it matches what the catalogue promises."""
    import rasterio

    with rasterio.open(out_vrt) as src:
        print(f"\n  VRT opens OK: {out_vrt}")
        print(f"    size    : {src.width} x {src.height}")
        print(f"    crs     : {src.crs}")
        print(f"    dtype   : {src.dtypes[0]}")
        print(f"    nodata  : {src.nodata}")
        print(f"    res     : {src.res}")
        print(f"    bounds  : {tuple(round(b, 3) for b in src.bounds)}")

        problems = []
        if src.crs is None or src.crs.to_epsg() != EXPECTED_EPSG:
            problems.append(f"CRS is {src.crs}, catalogue says EPSG:{EXPECTED_EPSG}")
        if src.dtypes[0] != EXPECTED_DTYPE:
            problems.append(
                f"dtype is {src.dtypes[0]}, catalogue says Byte/{EXPECTED_DTYPE}"
            )
        if src.nodata != NODATA:
            problems.append(f"nodata is {src.nodata}, catalogue says {NODATA}")

        if problems:
            print("\n  WARNING — mismatch against data_catalogue.yml:")
            for p in problems:
                print(f"    - {p}")
        else:
            print("\n  Matches the catalogue entry (EPSG:4326, Byte, nodata=0).")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    root, relpath = catalogue_root_and_relpath(repo_root, CATALOGUE_NAME)
    default_out = root / relpath

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--tile-dir",
        type=Path,
        default=default_out.parent,
        help=f"folder holding the downloaded WorldCover tiles (default: {default_out.parent})",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help=f"output .vrt path (default: {default_out})",
    )
    ap.add_argument(
        "--check-deltas",
        action="store_true",
        help="also report, per delta polygon, which tiles are still missing",
    )
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    print(f"catalogue root : {root}")
    print(f"tile dir       : {args.tile_dir}")
    print(f"output vrt     : {args.out}")

    tiles = find_tiles(args.tile_dir)
    partial = [
        p.name for p in args.tile_dir.iterdir() if p.is_file() and ".tif." in p.name
    ]
    print(f"\nfound {len(tiles)} finished tile(s); first few:")
    for t in tiles[:5]:
        print(f"  {t.name}")
    if partial:
        print(f"  ({len(partial)} partial download(s) ignored, e.g. {partial[0]})")

    if args.check_deltas:
        check_delta_coverage(repo_root, args.tile_dir)

    # Keeping the VRT in the tile folder lets GDAL store RELATIVE paths, so the
    # folder stays movable; elsewhere, absolute paths get baked in.
    if args.out.parent != args.tile_dir:
        print(
            "\n  NOTE: the .vrt is not in the tile folder, so GDAL will bake in "
            "ABSOLUTE tile paths.\n  Moving or renaming the tile folder later will break it."
        )

    if args.dry_run:
        print("\n(dry-run — nothing written)")
        return 0

    print("\nbuilding VRT…")
    build_vrt(tiles, args.out)
    validate(args.out)
    print(
        "\nDone. Set landuse.source: esa_worldcover in config/config.yml to use it "
        "(rule prepare_landuse, 02b)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
