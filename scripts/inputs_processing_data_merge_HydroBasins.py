r"""Merge shapefiles from zipped directories, grouped by Pfafstetter level.

Scans a folder of zipped directories containing HydroBasins shapefiles,
groups them by the level identifier in the directory name (e.g. ``lev04``,
``lev05``), and merges each group into a single output shapefile.

Example:
    Run from the command line::

        python scripts/data_merge_HydroBasins.py \\
            --input_dir /path/to/zips \\
            --output_dir /path/to/output

    Or call programmatically::

        >>> from scripts.data_merge_HydroBasins import process
        >>> from pathlib import Path
        >>> process(Path("data/zips"), Path("output/"))
"""

from __future__ import annotations

import argparse
import re
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Compiled once at module level — no shell-escaping issues.
LEVEL_RE: re.Pattern[str] = re.compile(r"lev\d+", re.IGNORECASE)


def find_level_identifier(name: str) -> str | None:
    """Extract the first level identifier from a filename stem.

    Matches patterns such as ``lev04``, ``Lev05``, or ``LEV07`` anywhere in
    *name* and returns the match normalised to lowercase.

    Args:
        name: The filename stem (without extension) to search, e.g.
            ``"hybas_global_lev04_v1c"``.

    Returns:
        The first matching level string in lowercase (e.g. ``"lev04"``), or
        None if no level identifier is found.

    Example:
        >>> find_level_identifier("hybas_global_lev04_v1c")
        'lev04'
        >>> find_level_identifier("some_other_file") is None
        True
    """
    match = LEVEL_RE.search(name)
    return match.group(0).lower() if match else None


def find_shapefiles_in_dir(directory: Path) -> list[Path]:
    """Return all ``.shp`` files found recursively under *directory*.

    Args:
        directory: Root directory to search. All subdirectories are included
            in the recursive scan.

    Returns:
        List of :class:`~pathlib.Path` objects for every ``.shp`` file found
        under *directory*. Returns an empty list if none are found.

    Example:
        >>> shapefiles = find_shapefiles_in_dir(Path("/tmp/extracted"))
        >>> print(len(shapefiles))
        3
    """
    return list(directory.rglob("*.shp"))


def merge_and_save(gdfs: list[gpd.GeoDataFrame], output_path: Path) -> None:
    """Concatenate *gdfs* and write the result to a shapefile.

    Creates the parent directory of *output_path* if it does not already
    exist. Does nothing if *gdfs* is empty.

    Args:
        gdfs: List of GeoDataFrames to concatenate. All frames are assumed to
            share the same CRS; the CRS of the first frame is used for the
            merged output.
        output_path: Full path (including filename) of the output shapefile.
            The ``.shp`` extension is expected.

    Returns:
        None. The merged shapefile is written to *output_path*.

    Example:
        >>> merge_and_save([gdf1, gdf2], Path("output/hybas_global_merged_lev04.shp"))
          [✓] Saved 1842 features → output/hybas_global_merged_lev04.shp
    """
    if not gdfs:
        print("  [!] No data to merge — skipping.")
        return

    merged: gpd.GeoDataFrame = gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True),
        crs=gdfs[0].crs,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(output_path)
    print(f"  [✓] Saved {len(merged)} features → {output_path}")


def process(input_dir: Path, output_dir: Path) -> None:
    """Scan *input_dir* for zip files, group by level, and merge each group.

    For each distinct level identifier found across all zip filenames, all
    shapefiles from that group are extracted to a temporary directory, read
    into GeoDataFrames, and merged into a single output shapefile in
    *output_dir*. Zip files with no level identifier are logged and skipped.

    Args:
        input_dir: Directory containing the ``.zip`` files to process.
        output_dir: Directory where merged shapefiles will be saved. Created
            automatically if it does not exist.

    Returns:
        None. One shapefile per level group is written to *output_dir*.

    Raises:
        SystemExit: Not raised directly; individual bad zip files and
            unreadable shapefiles are logged and skipped rather than aborting.

    Example:
        >>> process(Path("data/hydrobasins_zips"), Path("output/merged"))
        Found 8 zip file(s) in data/hydrobasins_zips
        ── Level: lev04  (2 zip(s))
           Extracting: hybas_na_lev04_v1c.zip
              → Read 921 features from hybas_na_lev04_v1c.shp
        ...
        Done.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    zip_files: list[Path] = sorted(input_dir.glob("*.zip"))
    if not zip_files:
        print(f"No .zip files found in: {input_dir}")
        return

    print(f"Found {len(zip_files)} zip file(s) in {input_dir}\n")

    # Group zip files by their level identifier.
    groups: dict[str, list[Path]] = defaultdict(list)
    ungrouped: list[Path] = []

    for zf in zip_files:
        level: str | None = find_level_identifier(zf.stem)
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

    # Process each level group.
    for level, zips in sorted(groups.items()):
        print(f"── Level: {level}  ({len(zips)} zip(s))")
        gdfs: list[gpd.GeoDataFrame] = []

        for zf in zips:
            print(f"   Extracting: {zf.name}")
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path: Path = Path(tmpdir)
                try:
                    with zipfile.ZipFile(zf, "r") as zhandle:
                        zhandle.extractall(tmp_path)
                except zipfile.BadZipFile:
                    print(f"   [!] Bad zip file, skipping: {zf.name}")
                    continue

                shapefiles: list[Path] = find_shapefiles_in_dir(tmp_path)
                if not shapefiles:
                    print(f"   [!] No .shp files found inside {zf.name}, skipping.")
                    continue

                for shp in shapefiles:
                    try:
                        gdf: gpd.GeoDataFrame = gpd.read_file(shp)
                        gdf["_source_zip"] = zf.name
                        gdf["_source_shp"] = shp.name
                        gdfs.append(gdf)
                        print(f"      → Read {len(gdf)} features from {shp.name}")
                    except Exception as exc:
                        print(f"      [!] Could not read {shp.name}: {exc}")

        output_path: Path = output_dir / f"hybas_global_merged_{level}.shp"
        merge_and_save(gdfs, output_path)
        print()

    print("Done.")


def main() -> None:
    r"""Parse command-line arguments and run the shapefile merge pipeline.

    Defines two required arguments — ``--input_dir`` and ``--output_dir`` —
    validates that the input directory exists, and delegates to
    :func:`process`.

    Returns:
        None.

    Raises:
        SystemExit: If ``--input_dir`` does not point to an existing directory,
            or if required arguments are missing (raised by argparse).

    Example:
        Run from the command line::

            python scripts/data_merge_HydroBasins.py \\
                --input_dir data/zips \\
                --output_dir output/merged
    """
    parser = argparse.ArgumentParser(
        description=(
            "Merge shapefiles from zipped directories, partitioned by level identifier."
        )
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
