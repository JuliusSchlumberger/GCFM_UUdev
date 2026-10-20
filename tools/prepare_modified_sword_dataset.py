"""Build the modified SWORD v17c dataset by applying manual per-reach width fixes.

Writes SWORD_global_v17c_unpublished_modified.gpkg (catalogue "river_network",
the default river network) as a fresh copy of SWORD_global_v17c_unpublished.gpkg
("river_network_original"), overwriting it: WIDTH_CHANGES is the only source of
corrections. A reach whose original width differs from expected_original_width
by more than WIDTH_MATCH_TOLERANCE_M is skipped and reported (likely a wrong
reach_id or changed source data).

The files are ~1.3 GB GeoPackages, too slow to load with GeoPandas: the copy is
an OS file copy, reads use plain sqlite3 with a temporary reach_id index, and
UPDATEs go through `ogrinfo -dialect SQLite` because the GeoPackage's rtree
triggers call SpatiaLite functions (ST_IsEmpty, ST_MinX, ...) that plain
sqlite3 lacks ("no such function: ST_IsEmpty"), even for attribute-only
updates. Only the width column is changed, never geometry.

Usage:
    conda run -n hmt_sfincs_dev python prepare_modified_sword_dataset.py
"""

import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

OGRINFO = Path(sys.prefix) / "Library" / "bin" / "ogrinfo.exe"
if not OGRINFO.exists():
    _which = shutil.which("ogrinfo")
    if _which is None:
        raise FileNotFoundError(
            f"ogrinfo not found at {OGRINFO} or on PATH -- required to apply "
            f"attribute updates to the GeoPackage (see module docstring)."
        )
    OGRINFO = Path(_which)

# GCFM_RAW_DATA_ROOT is machine-specific; see CONTRIBUTING.md "Local machine paths".
SWORD_DIR = Path(os.environ["GCFM_RAW_DATA_ROOT"]) / "SWORD"
ORIGINAL_PATH = SWORD_DIR / "SWORD_global_v17c_unpublished.gpkg"
MODIFIED_PATH = SWORD_DIR / "SWORD_global_v17c_unpublished_modified.gpkg"
TABLE = "global_edges"

# Absorbs floored expected widths (438.67 m actual vs. 438 expected is applied).
WIDTH_MATCH_TOLERANCE_M = 1.0

# reach_id -> (expected_original_width_m, new_width_m, delta, name)
WIDTH_CHANGES: dict[str, tuple[float, float, str, str]] = {
    "74230100061": (1020, 270, "Mississippi", ""),
    "74230100101": (432, 250, "Mississippi", ""),
    "74230100091": (438, 200, "Mississippi", ""),
    "23261000101": (249, 200, "Rhine-Meuse", "Beneden Merwede"),
    "23261001751": (367, 450, "Rhine-Meuse", "Nieuwe Merwede"),
    "23261000055": (490, 150, "Rhine-Meuse", "Hollandse Ijssel"),
    "23250801081": (139, 80, "Rhine-Meuse", "Meuse"),
    "23261001795": (189, 110, "Rhine-Meuse", "Afgedamde Maas"),
    "23250800915": (271, 90, "Rhine-Meuse", "Donge"),
}


def main() -> None:
    """Copy the original SWORD file and apply WIDTH_CHANGES to the copy."""
    if not ORIGINAL_PATH.exists():
        raise FileNotFoundError(f"Original SWORD file not found: {ORIGINAL_PATH}")

    t0 = time.time()
    print(f"Copying {ORIGINAL_PATH.name} -> {MODIFIED_PATH.name} ...", flush=True)
    shutil.copy2(ORIGINAL_PATH, MODIFIED_PATH)
    print(f"  done ({time.time() - t0:.1f}s)", flush=True)

    # Read everything and close before writing: each ogrinfo call opens its
    # own connection and must not contend with a Python-side lock.
    con = sqlite3.connect(MODIFIED_PATH)
    cur = con.cursor()

    print("Building temporary index on reach_id ...", flush=True)
    t1 = time.time()
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_reach_id_tmp ON {TABLE}(reach_id)")
    con.commit()
    print(f"  done ({time.time() - t1:.1f}s)", flush=True)

    current_widths: dict[str, float | None] = {}
    for reach_id in WIDTH_CHANGES:
        cur.execute(f"SELECT width FROM {TABLE} WHERE reach_id = ?", (int(reach_id),))
        row = cur.fetchone()
        current_widths[reach_id] = row[0] if row is not None else None

    print("Dropping temporary index ...", flush=True)
    cur.execute("DROP INDEX IF EXISTS idx_reach_id_tmp")
    con.commit()
    con.close()

    applied, skipped = [], []
    for reach_id, (expected_orig, new_width, delta, name) in WIDTH_CHANGES.items():
        label = f"{reach_id} ({delta}{f' / {name}' if name else ''})"
        current_width = current_widths[reach_id]
        if current_width is None:
            print(f"  NOT FOUND: {label} -- skipped")
            skipped.append((reach_id, "not found"))
            continue
        if abs(current_width - expected_orig) > WIDTH_MATCH_TOLERANCE_M:
            print(
                f"  MISMATCH: {label}: actual width={current_width:.2f} differs from "
                f"expected_original={expected_orig} by more than {WIDTH_MATCH_TOLERANCE_M:.0f}m "
                f"-- skipped, NOT modified"
            )
            skipped.append((reach_id, f"width mismatch: actual={current_width:.2f}"))
            continue
        sql = f"UPDATE {TABLE} SET width = {float(new_width)} WHERE reach_id = {int(reach_id)}"
        proc = subprocess.run(
            [str(OGRINFO), "-dialect", "SQLite", "-sql", sql, str(MODIFIED_PATH)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"  FAILED: {label}: ogrinfo UPDATE failed: {proc.stderr.strip()}")
            skipped.append((reach_id, f"UPDATE failed: {proc.stderr.strip()}"))
            continue
        print(f"  OK: {label}: width {current_width:.2f} -> {new_width}")
        applied.append(reach_id)

    print(
        f"\n{len(applied)}/{len(WIDTH_CHANGES)} width change(s) applied, "
        f"{len(skipped)} skipped, in {time.time() - t0:.1f}s total"
    )
    if skipped:
        print("Skipped reaches (not modified) -- resolve and re-run:")
        for reach_id, reason in skipped:
            print(f"  {reach_id}: {reason}")


if __name__ == "__main__":
    main()
