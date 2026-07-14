"""
prepare_modified_sword_dataset.py — builds the "modified" SWORD v17c dataset
(data catalogue entry "river_network", the pipeline's default river network
source) from the unmodified original ("river_network_original"), by applying
a documented dictionary of manual per-reach attribute corrections.

ALWAYS rebuilds SWORD_global_v17c_unpublished_modified.gpkg completely fresh
from SWORD_global_v17c_unpublished.gpkg, overwriting whatever was there
before -- this dictionary is the SOLE, authoritative source of truth for
every correction; there is no incremental/preserve-prior-edits mode. Add new
corrections as new WIDTH_CHANGES entries (or new dicts alongside it, e.g.
MAIN_SIDE_CHANGES, following the same pattern) and re-run.

Before applying each width change, the reach's CURRENT width (i.e. its
value in the unmodified original, since the file is freshly copied from it
first) is compared against that entry's expected_original_width, within
WIDTH_MATCH_TOLERANCE_M (default 1 m -- absorbs expected_original_width
values that were floored/truncated rather than rounded). A mismatch beyond
that tolerance means the reach_id likely doesn't refer to what was intended
(wrong ID, or the source data changed upstream) -- these are flagged and
SKIPPED, never applied silently, so a bad reach_id can't silently corrupt
the wrong reach.

Implementation note: the two SWORD files are ~1.3 GB GeoPackages (SQLite);
reading either fully into GeoPandas to change a handful of attribute values
was multiple minutes just to COUNT(*) the rows on this machine's disk. This
script instead: (1) copies the file at the OS level (no GDAL/GeoPandas
parsing), (2) adds a temporary index on reach_id via plain sqlite3 so the
~9-20 point lookups/updates don't each require a full-table scan, (3) reads
current values via plain sqlite3 (fast, no triggers involved), but applies
the actual UPDATEs via `ogrinfo -dialect SQLite` (GDAL's own SQL engine)
rather than raw sqlite3 -- this GeoPackage's rtree/feature-count maintenance
triggers call SpatiaLite-style functions (ST_IsEmpty, ST_MinX, ...) that
plain Python sqlite3 has no definition for (confirmed: a raw sqlite3 UPDATE
fails with "no such function: ST_IsEmpty" even though only the width column,
never geometry, is being changed -- the trigger's WHEN clause still gets
evaluated). GDAL's SQLite dialect has these functions registered, matching
how GDAL itself created the file and its triggers in the first place.
Geometry is never touched by these UPDATEs, only the width column.

Usage:
    conda run -n hmt_sfincs_dev python prepare_modified_sword_dataset.py
"""

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

SWORD_DIR = Path("D:/GCFM_UU/raw_data/SWORD")
ORIGINAL_PATH = SWORD_DIR / "SWORD_global_v17c_unpublished.gpkg"
MODIFIED_PATH = SWORD_DIR / "SWORD_global_v17c_unpublished_modified.gpkg"
TABLE = "global_edges"

# Max |actual - expected_original_width_m| before a reach is flagged as a
# mismatch and skipped -- e.g. 438.67 m actual vs. an expected_original of
# 438 (a truncated/floored value) is within tolerance and still applied.
WIDTH_MATCH_TOLERANCE_M = 1.0

# reach_id -> (expected_original_width_m, new_width_m, delta, name)
# expected_original_width_m is checked against the reach's actual current
# width (within WIDTH_MATCH_TOLERANCE_M) before the change is applied -- a
# mismatch beyond that tolerance is flagged and skipped rather than applied
# blindly.
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
    if not ORIGINAL_PATH.exists():
        raise FileNotFoundError(f"Original SWORD file not found: {ORIGINAL_PATH}")

    t0 = time.time()
    print(f"Copying {ORIGINAL_PATH.name} -> {MODIFIED_PATH.name} ...", flush=True)
    shutil.copy2(ORIGINAL_PATH, MODIFIED_PATH)
    print(f"  done ({time.time() - t0:.1f}s)", flush=True)

    # ── phase 1: build a temp index, then read every current value, then
    # close the connection entirely -- so the write phase below (each
    # ogrinfo call opens its own connection) never contends with a
    # long-lived Python-side connection/lock on the same file. ──────────────
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

    # ── phase 2: verify + apply, one ogrinfo subprocess (its own connection)
    # per reach. ─────────────────────────────────────────────────────────────
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
