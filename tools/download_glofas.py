"""Download GloFAS v4 consolidated daily river discharge, one netCDF per year.

Needs CDS API credentials (cdsapi); resumable: finished years are skipped.
Exits non-zero if any year failed.
"""

import os
import sys

import cdsapi

DATASET = "cems-glofas-historical"
YEARS = [str(y) for y in range(1979, 2026)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]


def main() -> None:
    # GCFM_RAW_DATA_ROOT is machine-specific; see CONTRIBUTING.md "Local machine paths".
    output_dir = os.path.join(os.environ["GCFM_RAW_DATA_ROOT"], "GloFAS")
    os.makedirs(output_dir, exist_ok=True)

    client = cdsapi.Client()
    failed: list[str] = []

    for year in YEARS:
        final_path = os.path.join(output_dir, f"glofas_{year}.nc")
        tmp_path = final_path + ".part"

        if os.path.exists(final_path):
            print(f"Skipping {year} (already downloaded)")
            continue

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        request = {
            "system_version": ["version_4_0"],
            "hydrological_model": ["lisflood"],
            "product_type": ["consolidated"],
            "variable": ["river_discharge_in_the_last_24_hours"],
            "hyear": [year],
            "hmonth": MONTHS,
            "hday": DAYS,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }

        print(f"Downloading {year} ...")
        try:
            client.retrieve(DATASET, request).download(tmp_path)
            os.rename(tmp_path, final_path)  # atomic: only "complete" once renamed
            print(f"Done: {final_path}")
        # cdsapi raises a bare Exception for API errors; keep going with the next year.
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            print(f"Failed {year}: {e}")
            failed.append(year)

    if failed:
        print(f"{len(failed)} year(s) failed: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
