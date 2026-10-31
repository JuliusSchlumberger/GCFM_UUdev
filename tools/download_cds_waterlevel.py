"""Download and unzip hourly GTSM reanalysis water levels from the Copernicus CDS.

Needs CDS API credentials (cdsapi); resumable: extracted years are skipped.
Exits non-zero if any archive was corrupt.
"""

import sys
import zipfile
from pathlib import Path

import cdsapi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))
from src.io import site_setting  # noqa: E402

DATASET = "sis-water-level-change-timeseries-cmip6"
VARIABLES = ["storm_surge_residual", "total_water_level"]
YEARS = [str(y) for y in range(1950, 2025)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]

DELETE_ZIP_AFTER_EXTRACT = False


def extract(zip_path: Path, dest: Path) -> bool:
    """Extract zip_path into dest; return False if the archive is corrupt."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()  # CRC check on every member
            if bad is not None:
                print(f"  corrupt member {bad} in {zip_path.name}")
                return False
            zf.extractall(dest)
            print(f"  extracted {len(zf.namelist())} files -> {dest}")
        return True
    except zipfile.BadZipFile:
        print(f"  {zip_path.name} is not a valid zip")
        return False


def main() -> None:
    # raw_data_root is a site setting; see config/site.example.yml.
    out_dir = Path(site_setting("raw_data_root")) / "GTSM_storm_tide_hourly"
    out_dir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()
    failed: list[str] = []

    for var in VARIABLES:
        extract_dir = out_dir / var
        extract_dir.mkdir(parents=True, exist_ok=True)

        for year in YEARS:
            target = out_dir / f"{var}_{year}.zip"
            done_marker = extract_dir / f".{var}_{year}.extracted"

            if done_marker.exists():
                print(f"skip {target.name} (already extracted)")
                continue

            # Temporary name: an interrupted transfer never looks like a finished zip.
            if not target.exists():
                tmp = target.with_suffix(".zip.part")
                request = {
                    "variable": [var],
                    "experiment": "reanalysis",
                    "temporal_aggregation": ["hourly"],
                    "year": [year],
                    "month": MONTHS,
                    "version": ["v3"],
                }
                print(f"requesting {target.name}")
                client.retrieve(DATASET, request).download(str(tmp))
                tmp.replace(target)

            print(f"extracting {target.name}")
            if not extract(target, extract_dir):
                # Removed so the next run fetches it again.
                target.unlink(missing_ok=True)
                failed.append(target.name)
                continue

            done_marker.touch()
            if DELETE_ZIP_AFTER_EXTRACT:
                target.unlink()

    if failed:
        print(
            f"{len(failed)} archive(s) corrupt, re-run to fetch again: {', '.join(failed)}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
