"""
merge_shapefiles.py
-------------------
Scans a folder of zipped directories containing shapefiles,
groups them by a level identifier in the directory name (e.g. lev04, lev05),
and merges each group into a single output shapefile.

Usage:
    python merge_shapefiles.py --input_dir /path/to/zips --output_dir /path/to/output

Dependencies:
    pip install geopandas
"""

import argparse
import re
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Compiled once at module level — no shell escaping issues
LEVEL_RE = re.compile(r"lev\d+", re.IGNORECASE)


def find_level_identifier(name: str) -> str | None:
    """Extract the first level identifier (e.g. 'lev04', 'lev05') from a filename stem."""
    match = LEVEL_RE.search(name)
    return match.group(0).lower() if match else None


def find_shapefiles_in_dir(directory: Path) -> list[Path]:
    """Return all .shp files found recursively under `directory`."""
    return list(directory.rglob("*.shp"))


def merge_and_save(gdfs: list[gpd.GeoDataFrame], output_path: Path) -> None:
    """Concatenate a list of GeoDataFrames and write to a shapefile."""
    if not gdfs:
        print("  [!] No data to merge — skipping.")
        return

    merged = gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True),
        crs=gdfs[0].crs,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(output_path)
    print(f"  [✓] Saved {len(merged)} features → {output_path}")


def process(input_dir: Path, output_dir: Path) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    zip_files = sorted(input_dir.glob("*.zip"))
    if not zip_files:
        print(f"No .zip files found in: {input_dir}")
        return

    print(f"Found {len(zip_files)} zip file(s) in {input_dir}\n")

    # Group zip files by their level identifier
    groups: dict[str, list[Path]] = defaultdict(list)
    ungrouped: list[Path] = []

    for zf in zip_files:
        level = find_level_identifier(zf.stem)
        if level:
            groups[level].append(zf)
        else:
            ungrouped.append(zf)

    if ungrouped:
        print(
            f"[!] {len(ungrouped)} zip(s) had no level identifier and will be skipped:"
        )
        for f in ungrouped:
            print(f"    {f.name}")
        print()

    # Process each level group
    for level, zips in sorted(groups.items()):
        print(f"── Level: {level}  ({len(zips)} zip(s))")
        gdfs: list[gpd.GeoDataFrame] = []

        for zf in zips:
            print(f"   Extracting: {zf.name}")
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                try:
                    with zipfile.ZipFile(zf, "r") as zhandle:
                        zhandle.extractall(tmp_path)
                except zipfile.BadZipFile:
                    print(f"   [!] Bad zip file, skipping: {zf.name}")
                    continue

                shapefiles = find_shapefiles_in_dir(tmp_path)
                if not shapefiles:
                    print(f"   [!] No .shp files found inside {zf.name}, skipping.")
                    continue

                for shp in shapefiles:
                    try:
                        gdf = gpd.read_file(shp)
                        gdf["_source_zip"] = zf.name
                        gdf["_source_shp"] = shp.name
                        gdfs.append(gdf)
                        print(f"      → Read {len(gdf)} features from {shp.name}")
                    except Exception as exc:
                        print(f"      [!] Could not read {shp.name}: {exc}")

        output_path = output_dir / f"hybas_global_merged_{level}.shp"
        merge_and_save(gdfs, output_path)
        print()

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge shapefiles from zipped directories, partitioned by level identifier."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=Path,
        help="Folder containing the .zip files.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Folder where merged shapefiles will be saved.",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")

    process(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
