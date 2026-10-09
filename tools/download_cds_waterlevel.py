import os
import zipfile
import cdsapi
from pathlib import Path

dataset = "sis-water-level-change-timeseries-cmip6"
# Machine-specific path, read from the GCFM_RAW_DATA_ROOT environment variable.
# Set it once in PowerShell, then restart your terminal (see
# CONTRIBUTING.md "Local machine paths"):
#   [Environment]::SetEnvironmentVariable("GCFM_RAW_DATA_ROOT", "D:\your\raw_data\path", "User")
out_dir = Path(os.environ["GCFM_RAW_DATA_ROOT"]) / "GTSM_storm_tide_hourly"
out_dir.mkdir(parents=True, exist_ok=True)

variables = ["storm_surge_residual", "total_water_level"]
years = [str(y) for y in range(1950, 2025)]  # extend range as needed
months = [f"{m:02d}" for m in range(1, 13)]

# Set True to remove each zip once its contents are extracted and verified.
DELETE_ZIP_AFTER_EXTRACT = False

client = cdsapi.Client()


def extract(zip_path: Path, dest: Path) -> bool:
    """Extract zip_path into dest. Returns False if the archive is corrupt."""
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


for var in variables:
    extract_dir = out_dir / var
    extract_dir.mkdir(parents=True, exist_ok=True)

    for year in years:
        target = out_dir / f"{var}_{year}.zip"
        done_marker = extract_dir / f".{var}_{year}.extracted"

        if done_marker.exists():
            print(f"skip {target.name} (already extracted)")
            continue

        # Download to a temporary name so an interrupted transfer is never
        # mistaken for a finished zip on the next run.
        if not target.exists():
            tmp = target.with_suffix(".zip.part")
            request = {
                "variable": [var],
                "experiment": "reanalysis",
                "temporal_aggregation": ["hourly"],
                "year": [year],
                "month": months,
                "version": ["v3"],
            }
            print(f"requesting {target.name}")
            client.retrieve(dataset, request).download(str(tmp))
            tmp.replace(target)

        print(f"extracting {target.name}")
        if not extract(target, extract_dir):
            # Remove the bad archive so it is fetched again on the next run.
            target.unlink(missing_ok=True)
            continue

        done_marker.touch()
        if DELETE_ZIP_AFTER_EXTRACT:
            target.unlink()
