"""Remap a Copernicus ESA WorldCover landcover raster to Manning's N roughness values.

Uses a CSV lookup table to map ESA codes to Manning's N values.

Key design choices:
    - Processes the 1.7 GB raster in chunks (windows) so it never loads
      the full dataset into memory.
    - Tree cover spans multiple ESA codes (110-127) which are all mapped
      to a single N value via a code range in the CSV.
    - Unknown codes are set to NaN and reported at the end.
"""

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from src.utils.setup_logger import setup_logging

_LOG = setup_logging("roughness_from_landcover")

RANGE_RE = re.compile(r"\[(\d+);(\d+)]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as ``mm:ss`` or ``h:mm:ss``.

    Args:
        seconds: Duration in seconds.

    Returns:
        Human-readable string in ``mm:ss`` or ``h:mm:ss`` format.
    """
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Parse lookup table
# ---------------------------------------------------------------------------


def parse_lookup(csv_path: Path) -> dict[int, float]:
    """Parse a Manning's N lookup table from a CSV file into a code-to-value mapping.

    Reads a CSV file containing ESA WorldCover landcover codes and their
    corresponding Manning's N roughness values. Supports single codes and
    code ranges (e.g. ``[110;127]``), expanding ranges into individual integer
    mappings. Codes with a value of -999 are treated as nodata and mapped to
    NaN.

    Args:
        csv_path: Path to the CSV file containing at least the columns
            ``esa_worldcover`` and ``N``.

    Returns:
        Dictionary mapping ESA WorldCover integer codes to Manning's N values.
        Codes marked as nodata in the input are mapped to NaN.
    """
    _LOG.info("Parsing lookup table: %s", csv_path)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    lookup: dict[int, float] = {}
    n_ranges = 0
    n_nodata = 0

    for _, row in df.iterrows():
        code_str = str(row["esa_worldcover"].item()).strip()
        n_val = float(row["N"].item())
        n_mapped = np.nan if n_val == -999 else n_val

        if np.isnan(n_mapped):
            n_nodata += 1

        m = RANGE_RE.match(code_str)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            _LOG.debug(
                "Range entry [%d;%d] → N=%.4f (%d codes)", lo, hi, n_val, hi - lo + 1
            )
            for code in range(lo, hi + 1):
                lookup[code] = n_mapped
            n_ranges += 1
        else:
            try:
                lookup[int(code_str)] = n_mapped
            except ValueError:
                _LOG.warning("Could not parse ESA code '%s' — skipping.", code_str)

    _LOG.info(
        "Lookup parsed: %d total code mappings (%d range entries, %d nodata entries)",
        len(lookup),
        n_ranges,
        n_nodata,
    )
    return lookup


def build_lut_array(lookup: dict[int, float], max_code: int) -> np.ndarray:
    """Build a fast numpy lookup array indexed by ESA code.

    Args:
        lookup: Dictionary mapping ESA codes to Manning N values.
        max_code: Size of the output array; codes >= max_code are skipped.

    Returns:
        Float32 array of length ``max_code`` where ``lut[code]`` gives the
        Manning N value, or NaN for unmapped codes.
    """
    _LOG.info("Building LUT array (max_code=%d) ...", max_code)
    lut = np.full(max_code, np.nan, dtype=np.float32)
    n_skipped = 0

    for code, n_val in lookup.items():
        if 0 <= code < max_code:
            lut[code] = n_val
        else:
            _LOG.warning("ESA code %d exceeds max_code=%d — skipping.", code, max_code)
            n_skipped += 1

    n_valid = int(np.sum(~np.isnan(lut)))
    _LOG.info(
        "LUT array ready: %d valid entries, %d NaN slots, %d codes skipped (out of range)",
        n_valid,
        max_code - n_valid,
        n_skipped,
    )
    return lut


# ---------------------------------------------------------------------------
# Remap raster
# ---------------------------------------------------------------------------


def remap_landcover(
    landcover_path: Path,
    lut: np.ndarray,
    output_path: Path,
    chunk_size: int,
) -> set[int]:
    """Read the landcover raster in row chunks and write Manning N values.

    Applies the LUT to each chunk and writes a float32 GeoTIFF. Pixels
    whose ESA code is not present in the LUT are written as NaN.

    Args:
        landcover_path: Path to the input ESA WorldCover GeoTIFF.
        lut: LUT array as returned by :func:`build_lut_array`.
        output_path: Destination path for the Manning N GeoTIFF.
        chunk_size: Number of raster rows to process per chunk.

    Returns:
        Set of ESA codes encountered in the raster that had no LUT entry.
    """
    _LOG.info("Opening landcover raster: %s", landcover_path)

    with rasterio.open(landcover_path) as src:
        _LOG.info("  Size    : %d × %d px", src.width, src.height)
        _LOG.info("  CRS     : %s", src.crs)
        _LOG.info("  Nodata  : %s", src.nodata)
        _LOG.info("  Dtype   : %s", src.dtypes[0])
        total_pixels = src.width * src.height
        _LOG.info("  Total px: %s", f"{total_pixels:,}")

        meta = src.meta.copy()
        meta.update(
            {
                "dtype": "float32",
                "nodata": np.nan,
                "compress": "deflate",
                "predictor": 3,
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "bigtiff": "IF_SAFER",
            }
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        _LOG.info("Output path: %s", output_path)

        unknown_codes: set[int] = set()
        n_chunks = (src.height + chunk_size - 1) // chunk_size
        _LOG.info("Processing %d chunks (chunk_size=%d rows) ...", n_chunks, chunk_size)

        pixels_processed = 0
        t0 = time.time()
        t_last_log = t0

        with rasterio.open(output_path, "w", **meta) as dst:
            for chunk_i, row_off in enumerate(range(0, src.height, chunk_size), 1):
                rows_this_chunk = min(chunk_size, src.height - row_off)
                window = Window(0, row_off, src.width, rows_this_chunk)  # type: ignore[call-arg]

                lc = src.read(1, window=window)
                pixels_processed += lc.size

                new_unknown: set[int] = set()
                for c in np.unique(lc):
                    if 0 <= c < len(lut):
                        if np.isnan(lut[c]) and c not in (0, int(src.nodata or -1)):
                            new_unknown.add(int(c))
                    else:
                        new_unknown.add(int(c))

                first_seen = new_unknown - unknown_codes
                if first_seen:
                    _LOG.warning(
                        "Chunk %d: new unknown ESA codes encountered: %s",
                        chunk_i,
                        sorted(first_seen),
                    )
                unknown_codes |= new_unknown

                lc_clipped = np.clip(lc, 0, len(lut) - 1)
                roughness = lut[lc_clipped]

                if src.nodata is not None:
                    roughness[lc == src.nodata] = np.nan

                dst.write(roughness.astype(np.float32), 1, window=window)

                t_now = time.time()
                elapsed = t_now - t0
                should_log = (
                    chunk_i % 50 == 0
                    or chunk_i == n_chunks
                    or (t_now - t_last_log) >= 30
                )
                if should_log:
                    pct = chunk_i / n_chunks
                    remaining = (elapsed / pct) * (1 - pct) if pct > 0 else 0.0
                    px_per_s = pixels_processed / elapsed if elapsed > 0 else 0.0
                    _LOG.info(
                        "Chunk %5d / %d  |  %5.1f %%  |  elapsed %s  |  ETA %s"
                        "  |  %.1f Mpx/s  |  unknown codes so far: %d",
                        chunk_i,
                        n_chunks,
                        pct * 100,
                        _fmt_duration(elapsed),
                        _fmt_duration(remaining),
                        px_per_s / 1e6,
                        len(unknown_codes),
                    )
                    t_last_log = t_now

        total_elapsed = time.time() - t0
        _LOG.info(
            "Remapping complete in %s (%.1f Mpx/s overall)",
            _fmt_duration(total_elapsed),
            total_pixels / total_elapsed / 1e6,
        )

    if unknown_codes:
        _LOG.warning(
            "%d unknown ESA code(s) set to NaN in output: %s",
            len(unknown_codes),
            sorted(unknown_codes),
        )
    else:
        _LOG.info("No unknown ESA codes encountered.")

    return unknown_codes
