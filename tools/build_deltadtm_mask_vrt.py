"""
build_deltadtm_mask_vrt.py — build the DeltaDTM mask VRT from downloaded tiles.

The `deltadtm_mask` entry in config/data_catalogue.yml points at a single file:

    <raw_data_root>/DeltaDTM_masks/deltadtm_mask.vrt

That .vrt does not ship with the tiles. It is a small XML index that lists every
tile and its georeferenced extent, so GDAL can treat thousands of 1x1 degree
GeoTIFFs as one seamless global raster and open only the tiles that overlap the
window being read. Because it stores paths to tiles on *this* machine, it has to
be generated locally — that is what your colleague meant by "it's a lookup table".

Used by rule 05a (get_elevation) via src.raster.clip_ocean_from_topo, purely to
find ocean pixels (mask value 1) so FathomDEM's bad seaward extension can be
dropped before the GEBCO merge.

Usage:
    python tools/build_deltadtm_mask_vrt.py                    # default paths
    python tools/build_deltadtm_mask_vrt.py --tile-dir D:/somewhere/else
    python tools/build_deltadtm_mask_vrt.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

# Catalogue-declared properties of this dataset. Kept here so the VRT we build
# is guaranteed consistent with what data_catalogue.yml promises downstream.
NODATA = 255  # raster_properties.nodata_value
EXPECTED_DTYPE = "uint8"  # raster_properties.data_type: Byte
EXPECTED_EPSG = 4326


def catalogue_root_and_relpath(repo_root: Path) -> tuple[Path, str]:
    """Read meta.root and the deltadtm_mask file_path straight from the catalogue."""
    cat = yaml.safe_load(
        (repo_root / "config" / "data_catalogue.yml").read_text(encoding="utf-8")
    )
    # GCFM_RAW_DATA_ROOT overrides the committed root — same rule as 00_common.smk.
    root = Path(os.environ.get("GCFM_RAW_DATA_ROOT") or cat["meta"]["root"])
    for ds in cat["datasets"]:
        if ds["name"] == "deltadtm_mask":
            return root, ds["file_path"]
    raise KeyError("deltadtm_mask not found in config/data_catalogue.yml")


def find_tiles(tile_dir: Path) -> list[Path]:
    tiles = sorted(p for p in tile_dir.rglob("*.tif") if p.is_file())
    if not tiles:
        raise FileNotFoundError(
            f"No .tif tiles found under {tile_dir}\n"
            f"Download the DeltaDTM *mask* tiles into that folder first "
            f"(note: the mask is a separate product from the DeltaDTM elevation tiles)."
        )
    return tiles


def build_vrt(tiles: list[Path], out_vrt: Path) -> None:
    """
    Build the VRT with GDAL. Tries the Python bindings first, falls back to the
    gdalbuildvrt CLI. Nearest-neighbour is implicit and correct here — this is a
    categorical class mask (0/1/2/3), so it must never be averaged or blended.
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

    # Fallback: CLI. Use a file list — thousands of paths will blow the shell limit.
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
            print("\n  Matches the catalogue entry (EPSG:4326, Byte, nodata=255).")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    root, relpath = catalogue_root_and_relpath(repo_root)
    default_out = root / relpath

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--tile-dir",
        type=Path,
        default=default_out.parent,
        help=f"folder holding the downloaded mask .tif tiles (default: {default_out.parent})",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help=f"output .vrt path (default: {default_out})",
    )
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    print(f"catalogue root : {root}")
    print(f"tile dir       : {args.tile_dir}")
    print(f"output vrt     : {args.out}")

    tiles = find_tiles(args.tile_dir)
    print(f"\nfound {len(tiles)} tile(s); first few:")
    for t in tiles[:5]:
        print(f"  {t.name}")

    # Keeping the VRT in the same folder as the tiles lets GDAL store RELATIVE
    # paths, so the whole folder stays movable. If it sits elsewhere, absolute
    # paths get baked in and moving the tiles silently breaks the mosaic.
    try:
        args.out.parent.relative_to(args.tile_dir)
        same_tree = True
    except ValueError:
        same_tree = args.tile_dir == args.out.parent
    if not same_tree:
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
        "\nDone. Re-run the pipeline; rule 05a (get_elevation) should now find the mask."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
