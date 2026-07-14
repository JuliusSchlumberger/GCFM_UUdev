import cdsapi
import os

OUTPUT_DIR = r"D:\GCFM_UU\raw_data\GloFAS"
os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset = "cems-glofas-historical"

years = [str(y) for y in range(1979, 2026)]
months = [f"{m:02d}" for m in range(1, 13)]
days = [f"{d:02d}" for d in range(1, 32)]

client = cdsapi.Client()

for year in years:
    final_path = os.path.join(OUTPUT_DIR, f"glofas_{year}.nc")
    tmp_path = final_path + ".part"

    # Resume: skip anything already completed
    if os.path.exists(final_path):
        print(f"Skipping {year} (already downloaded)")
        continue

    # Clean up any leftover partial file from a previous interrupt
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    request = {
        "system_version": ["version_4_0"],
        "hydrological_model": ["lisflood"],
        "product_type": ["consolidated"],
        "variable": ["river_discharge_in_the_last_24_hours"],
        "hyear": [year],
        "hmonth": months,
        "hday": days,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    print(f"Downloading {year} ...")
    try:
        client.retrieve(dataset, request).download(tmp_path)
        os.rename(tmp_path, final_path)  # atomic: only "complete" once renamed
        print(f"Done: {final_path}")
    except Exception as e:
        # Remove the partial file so it's re-attempted next run
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print(f"Failed {year}: {e}")
