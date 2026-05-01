"""Bulk TIF downloader for the DeltaDTM dataset.

Downloads all TIF files listed in a CSV to a target directory, for example
an external hard drive. Designed to be safe to re-run: already-downloaded
files are skipped and failed URLs are logged to a separate CSV.

Features:
    - Auto-detects the URL column in the CSV.
    - Skips already-downloaded files based on filename and non-zero size.
    - Logs failed URLs to ``failed_downloads.csv`` for later retry.

Example:
    Run directly from the command line::

        python download_DeltaDTM_data.py

    Or call programmatically::

        from download_DeltaDTM_data import run_download
        run_download()
"""

from __future__ import annotations

import csv
import os
import time
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

import requests

from src.utils.setup_logger import setup_logging

_LOG = setup_logging("download_DeltaDTM_data")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CSV_FILE: str = "../src/input_processing/data/DeltaDTM/DeltaDTM_v1.1.csv"
OUTPUT_DIR: str = r"D:\GCFM_UU\data\DeltaDTM"
FAILED_LOG: str = "failed_downloads.csv"
TIMEOUT: int = 30  # seconds per request
CHUNK_SIZE: int = 1024 * 1024  # 1 MB per chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def detect_url_column(fieldnames: Sequence[str]) -> str | None:
    """Return the first CSV column name that looks like a URL field.

    Matches column names containing any of the keywords ``url``, ``link``,
    ``href``, ``path``, or ``download`` (case-insensitive).

    Args:
        fieldnames: List of column names from the CSV header row.

    Returns:
        The first matching column name, or None if no match is found.

    Example:
        >>> detect_url_column(["id", "download_url", "name"])
        'download_url'
        >>> detect_url_column(["id", "area", "name"]) is None
        True
    """
    for name in fieldnames:
        if any(
            kw in name.lower() for kw in ["url", "link", "href", "path", "download"]
        ):
            return name
    return None


def filename_from_url(url: str, index: int) -> str:
    """Derive a local filename from a URL, falling back to an index-based name.

    Extracts the final path component of the URL. If the component does not
    end with ``.tif``, a zero-padded fallback name is returned instead.

    Args:
        url: The download URL to extract a filename from.
        index: 1-based row index used to construct the fallback filename when
            the URL does not contain a usable name.

    Returns:
        A filename string ending in ``.tif``.

    Example:
        >>> filename_from_url("https://example.com/data/delta_01.tif", 1)
        'delta_01.tif'
        >>> filename_from_url("https://example.com/data/", 3)
        'file_00003.tif'
    """
    name: str = os.path.basename(urlparse(url).path)
    return name if (name and name.lower().endswith(".tif")) else f"file_{index:05d}.tif"


def download_file(url: str, dest_path: Path) -> tuple[bool, str | None]:
    """Download a single file from *url* and write it to *dest_path*.

    Streams the response in chunks of ``CHUNK_SIZE`` bytes to avoid loading
    large TIF files into memory. The destination file is created only if the
    request succeeds; partial files are left for the caller to clean up.

    Args:
        url: The URL to download from.
        dest_path: The local path to write the downloaded file to.

    Returns:
        A tuple of ``(success, error)`` where *success* is True if the file
        was downloaded without error, and *error* is the error message string
        if the download failed, or None if it succeeded.

    Example:
        >>> success, err = download_file("https://example.com/file.tif", Path("out/file.tif"))
        >>> if not success:
        ...     print(f"Download failed: {err}")
    """
    try:
        with requests.get(url, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, str(e)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_download() -> None:
    """Download all TIF files listed in the configured CSV file.

    Reads ``CSV_FILE``, auto-detects the URL column, and downloads each
    file to ``OUTPUT_DIR``. Already-present non-empty files are skipped.
    Failed URLs are written to ``FAILED_LOG`` for later retry.

    Raises:
        SystemExit: If ``CSV_FILE`` does not exist, or if no URL column can
            be detected in the CSV header.

    Example:
        >>> run_download()  # downloads all TIFs listed in CSV_FILE
    """
    csv_path: Path = Path(CSV_FILE)
    if not csv_path.exists():
        _LOG.error("CSV not found: %s", csv_path.resolve())
        raise SystemExit(1)

    output_dir: Path = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    failed: list[dict[str, str]] = []
    downloaded: int = 0
    skipped: int = 0
    errors: int = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        url_col: str | None = detect_url_column(reader.fieldnames or [])

        # Fallback: scan first row for any value starting with "http"
        if not url_col:
            _LOG.debug("Header scan found no URL column — scanning first data row.")
            first_row = next(reader, None)
            if first_row:
                for col, val in first_row.items():
                    if val.startswith("http"):
                        url_col = col
                        f.seek(0)
                        reader = csv.DictReader(f)
                        break

        if not url_col:
            _LOG.error("Cannot detect URL column. Set url_col manually.")
            raise SystemExit(1)

        _LOG.info("URL column detected: '%s'", url_col)
        rows: list[dict[str, str]] = list(reader)
        total: int = len(rows)
        _LOG.info("Found %d URLs → %s", total, output_dir.resolve())

        t0 = time.time()

        for i, row in enumerate(rows, start=1):
            url: str = row.get(url_col, "").strip()
            if not url:
                _LOG.warning("Row %d: empty URL — skipping.", i)
                skipped += 1
                continue

            filename: str = filename_from_url(url, i)
            dest: Path = output_dir / filename

            if dest.exists() and dest.stat().st_size > 0:
                _LOG.debug("[%d/%d] Already exists, skipping: %s", i, total, filename)
                skipped += 1
                continue

            _LOG.info("[%d/%d] Downloading: %s", i, total, filename)
            success, err = download_file(url, dest)

            if success:
                size_mb = dest.stat().st_size / 1_048_576
                _LOG.info("[%d/%d] OK — %.1f MB: %s", i, total, size_mb, filename)
                downloaded += 1
            else:
                _LOG.warning("[%d/%d] FAILED: %s — %s", i, total, filename, err)
                if dest.exists():
                    dest.unlink()
                failed.append({"url": url, "error": err or "unknown"})
                errors += 1

            time.sleep(0.1)

        elapsed = time.time() - t0
        _LOG.info(
            "Download loop finished in %.1f s — "
            "downloaded: %d, skipped: %d, failed: %d, total: %d",
            elapsed,
            downloaded,
            skipped,
            errors,
            total,
        )

    if failed:
        with open(FAILED_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["url", "error"])
            writer.writeheader()
            writer.writerows(failed)
        _LOG.warning("%d failed URL(s) written to: %s", len(failed), FAILED_LOG)
