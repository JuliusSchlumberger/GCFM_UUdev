"""
Bulk TIF Downloader
Downloads all TIF files listed in a CSV to a target directory (e.g. an external hard drive).
- Auto-detects the URL column in the CSV
- Skips already-downloaded files (safe to re-run)
- Logs failed URLs to failed_downloads.csv
"""

import csv
import os
import sys
import time
import requests
from pathlib import Path
from urllib.parse import urlparse

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
CSV_FILE = (
    "../src/input_processing/data/DeltaDTM/DeltaDTM_v1.1.csv"  # Path to your CSV file
)
OUTPUT_DIR = "D:\GCFM_UU\data\DeltaDTM"  # ← Change to your external drive path
FAILED_LOG = "failed_downloads.csv"
TIMEOUT = 30  # seconds per request
CHUNK_SIZE = 1024 * 1024  # 1 MB
# ──────────────────────────────────────────────────────────────────────────────


def detect_url_column(fieldnames):
    for name in fieldnames:
        if any(
            kw in name.lower() for kw in ["url", "link", "href", "path", "download"]
        ):
            return name
    return None


def filename_from_url(url, index):
    name = os.path.basename(urlparse(url).path)
    return name if (name and name.lower().endswith(".tif")) else f"file_{index:05d}.tif"


def download_file(url, dest_path):
    try:
        with requests.get(url, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
        return True, None
    except Exception as e:
        return False, str(e)


def run_download():
    csv_path = Path(CSV_FILE)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path.resolve()}")
        sys.exit(1)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    downloaded = skipped = errors = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        url_col = detect_url_column(reader.fieldnames or [])

        if not url_col:
            first_row = next(reader, None)
            if first_row:
                for col, val in first_row.items():
                    if val.startswith("http"):
                        url_col = col
                        f.seek(0)
                        reader = csv.DictReader(f)
                        break

        if not url_col:
            print("ERROR: Cannot detect URL column. Set url_col manually.")
            sys.exit(1)

        print(f"URL column: '{url_col}'")
        rows = list(reader)
        total = len(rows)
        print(f"Found {total} URLs → {output_dir.resolve()}\n")

        for i, row in enumerate(rows, start=1):
            url = row.get(url_col, "").strip()
            if not url:
                skipped += 1
                continue

            filename = filename_from_url(url, i)
            dest = output_dir / filename

            if dest.exists() and dest.stat().st_size > 0:
                print(f"[{i}/{total}] Skip (exists): {filename}")
                skipped += 1
                continue

            print(f"[{i}/{total}] {filename} ...", end=" ", flush=True)
            success, err = download_file(url, dest)

            if success:
                print(f"OK ({dest.stat().st_size / 1_048_576:.1f} MB)")
                downloaded += 1
            else:
                print(f"FAILED: {err}")
                if dest.exists():
                    dest.unlink()
                failed.append({"url": url, "error": err})
                errors += 1

            time.sleep(0.1)

    if failed:
        with open(FAILED_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["url", "error"])
            writer.writeheader()
            writer.writerows(failed)
        print(f"\nFailed URLs → {FAILED_LOG}")

    print("\n── Summary ─────────────")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped:    {skipped}")
    print(f"  Failed:     {errors}")
    print(f"  Total:      {total}")


if __name__ == "__main__":
    main()
