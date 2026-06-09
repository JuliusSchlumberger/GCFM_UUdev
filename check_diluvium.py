"""
check_diluvium.py -- Audit, download, and extract DiluviumDEM zip archives.

Two checks, covering all cells in the five_deg_grid shapefile:

  1. Missing zips:  5-degree zip archives not yet in DiluviumDEM/zips/
                    -> downloaded automatically from Zenodo
  2. Missing tiles: downloaded zips whose .tif contents are not yet extracted
                    to DiluviumDEM/tiles/
                    -> extracted automatically

Usage
-----
  python check_diluvium.py            # check + download + extract
  python check_diluvium.py --dry-run  # check only, no downloads or extraction
"""

from __future__ import annotations

import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import yaml

# -- Config --------------------------------------------------------------------

ZENODO_RECORD_ID = "8384665"
_CATALOGUE = Path(__file__).parent / "config" / "data_catalogue.yml"


def _load_paths() -> tuple[Path, Path, Path]:
    with open(_CATALOGUE) as f:
        cat = yaml.safe_load(f)
    root = Path(cat["meta"]["root"])
    ds = {d["name"]: d for d in cat["datasets"]}
    tiles_dir = root / ds["diluvium_dem"]["file_path"]
    diluvium_root = tiles_dir.parent
    return (
        diluvium_root / "five_deg_grid" / "five_deg_grid_1degIDs_filtered.shp",
        diluvium_root / "zips",
        tiles_dir,
    )


FIVE_DEG_GRID_PATH, ZIPS_DIR, TILES_DIR = _load_paths()


# -- Helpers -------------------------------------------------------------------


def required_zip_ids(grid: gpd.GeoDataFrame) -> set[str]:
    """All 5-degree cell IDs present in the shapefile."""
    return set(grid["grid"].tolist())


def tiles_in_zip(zip_path: Path) -> list[str]:
    """Bare .tif filenames contained in a zip archive."""
    with zipfile.ZipFile(zip_path) as zf:
        return [Path(name).name for name in zf.namelist() if name.endswith(".tif")]


def fetch_zenodo_index() -> dict[str, tuple[str, int]]:
    """Return {zip_stem: (download_url, size_bytes)} from the Zenodo record."""
    api_url = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
    print(f"  Fetching Zenodo file index from {api_url} ...")
    with urllib.request.urlopen(api_url, timeout=30) as resp:
        data = json.loads(resp.read())
    index = {
        Path(f["key"]).stem: (f["links"]["self"], int(f["size"]))
        for f in data.get("files", [])
        if f["key"].endswith(".zip")
    }
    print(f"  {len(index)} zip file(s) listed on Zenodo")
    return index


def download_zip(url: str, dest: Path, size_bytes: int) -> None:
    """Download url to dest with an inline progress indicator."""
    tmp = dest.with_suffix(".tmp")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            downloaded = 0
            with open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(512 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if size_bytes:
                        pct = 100 * downloaded / size_bytes
                        print(
                            f"\r    {downloaded / 1e6:.1f} / {size_bytes / 1e6:.1f} MB"
                            f"  ({pct:.0f}%)",
                            end="",
                            flush=True,
                        )
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.rename(dest)
    print(f"\r    {size_bytes / 1e6:.1f} MB  -- done" + " " * 20)


# -- Main ----------------------------------------------------------------------


def main(dry_run: bool = False) -> None:
    print(f"Five-deg grid  : {FIVE_DEG_GRID_PATH}")
    print(f"Zips dir       : {ZIPS_DIR}")
    print(f"Tiles dir      : {TILES_DIR}")
    if dry_run:
        print("(dry-run: no files will be downloaded)")
    print()

    grid = gpd.read_file(FIVE_DEG_GRID_PATH).to_crs("EPSG:4326")
    print(f"Loaded {len(grid)} 5-degree grid cell(s)")

    required = required_zip_ids(grid)
    print(f"Required 5-degree cells : {len(required)}")

    existing_zips = {p.stem for p in ZIPS_DIR.glob("*.zip")}
    existing_tiles = {p.name for p in TILES_DIR.glob("DiluviumDEM_*.tif")}

    # -- 1. Missing zip files --------------------------------------------------
    missing_zips = sorted(required - existing_zips)

    print(f"\n{'=' * 60}")
    print(f"  MISSING ZIPS  ({len(missing_zips)} of {len(required)} required)")
    print(f"{'=' * 60}")
    if missing_zips:
        for name in missing_zips:
            print(f"  {name}.zip")

        if not dry_run:
            print()
            zenodo = fetch_zenodo_index()
            n_ok = n_skip = n_fail = 0
            for name in missing_zips:
                if name not in zenodo:
                    print(f"  WARNING: {name}.zip not found on Zenodo -- skipping")
                    n_skip += 1
                    continue
                url, size = zenodo[name]
                print(f"  Downloading {name}.zip  ({size / 1e6:.1f} MB) ...")
                try:
                    download_zip(url, ZIPS_DIR / f"{name}.zip", size)
                    n_ok += 1
                except Exception as exc:
                    print(f"\n  ERROR downloading {name}.zip: {exc}")
                    n_fail += 1
            print(
                f"\n  Downloads: {n_ok} succeeded, {n_skip} not on Zenodo, "
                f"{n_fail} failed"
            )
    else:
        print("  All required zip files are present.")

    # -- 2. Downloaded zips with missing tiles ---------------------------------
    # Refresh after potential downloads
    existing_zips = {p.stem for p in ZIPS_DIR.glob("*.zip")}
    downloaded_required = sorted(required & existing_zips)
    needs_unzip: list[tuple[str, list[str]]] = []

    for zip_id in downloaded_required:
        zip_path = ZIPS_DIR / f"{zip_id}.zip"
        try:
            tile_names = tiles_in_zip(zip_path)
        except Exception as exc:
            print(f"\nWARNING: could not read {zip_path.name}: {exc}")
            continue
        missing = [t for t in tile_names if t not in existing_tiles]
        if missing:
            needs_unzip.append((zip_id, missing))

    total_missing_tiles = sum(len(m) for _, m in needs_unzip)
    print(f"\n{'=' * 60}")
    print(
        f"  ZIPS NEEDING EXTRACTION  "
        f"({len(needs_unzip)} zip(s), {total_missing_tiles} tile(s) missing)"
    )
    print(f"{'=' * 60}")
    if needs_unzip:
        for zip_id, missing in needs_unzip:
            print(f"\n  {zip_id}.zip  -- {len(missing)} tile(s) not yet extracted:")
            for t in missing:
                print(f"    {t}")

        if not dry_run:
            TILES_DIR.mkdir(parents=True, exist_ok=True)
            n_ok = n_fail = 0
            print()
            for zip_id, missing in needs_unzip:
                zip_path = ZIPS_DIR / f"{zip_id}.zip"
                print(f"  Extracting {zip_id}.zip  ({len(missing)} tile(s)) ...")
                try:
                    with zipfile.ZipFile(zip_path) as zf:
                        names_in_zip = {
                            Path(n).name: n for n in zf.namelist() if n.endswith(".tif")
                        }
                        for tile in missing:
                            if tile not in names_in_zip:
                                print(f"    WARNING: {tile} not found in archive")
                                continue
                            data = zf.read(names_in_zip[tile])
                            (TILES_DIR / tile).write_bytes(data)
                            print(f"    {tile}  ({len(data) / 1e6:.1f} MB)")
                    n_ok += 1
                except Exception as exc:
                    print(f"  ERROR extracting {zip_id}.zip: {exc}")
                    n_fail += 1
            print(f"\n  Extraction: {n_ok} zip(s) succeeded, {n_fail} failed")
    else:
        print("  All downloaded zips are fully extracted.")

    # -- Summary ---------------------------------------------------------------
    existing_zips = {p.stem for p in ZIPS_DIR.glob("*.zip")}
    existing_tiles = {p.name for p in TILES_DIR.glob("DiluviumDEM_*.tif")}
    print(f"\n{'-' * 60}")
    print("  Summary")
    print(f"{'-' * 60}")
    print(f"  Required 5-degree cells : {len(required)}")
    print(
        f"  Downloaded zips         : {len(existing_zips & required)}"
        f"  (+ {len(existing_zips - required)} outside scope)"
    )
    print(f"  Missing zips            : {len(missing_zips)}")
    print(
        f"  Zips needing extraction : {len(needs_unzip)}"
        f"  ({total_missing_tiles} tile(s) total)"
    )
    print(f"  Tiles on disk           : {len(existing_tiles)}")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
