"""
diagnose_landuse_correction.py — Visual inspection aid for the "ocean pockets
enclosed by the coastal weir" landuse correction (rule modelled_depth_
estimation / its empirical sibling): writes a small categorical GeoTIFF
comparing {basin_id}_landuse.tif (raw) against {basin_id}_landuse_corrected.tif
(rule 10's own output, see src.raster.relabel_landuse_from_grid_mask) so the
relabeled pixels can be loaded straight into QGIS/GIS software alongside the
weir gpkg for a quick visual check.

Output raster values:
    0   Land / other (unchanged in both rasters)
    1   Ocean (landuse==200 in both -- real, still-open sea)
    2   Relabeled (200 in the raw file -> 80 in the corrected file -- the
        weir-enclosed pockets this whole mechanism targets)

Usage:
    conda run -n hmt_sfincs_dev python tests/diagnose_landuse_correction.py [basin_id]
    conda run -n hmt_sfincs_dev python tests/diagnose_landuse_correction.py 2433835
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
BASIN_ID = sys.argv[1] if len(sys.argv) > 1 else "2433835"

with open(REPO_ROOT / "config" / "config.yml") as fh:
    config = yaml.safe_load(fh)
RESULTS_DIR = Path(config["results_dir"])

basin_dir = RESULTS_DIR / BASIN_ID / "preprocessing_inputs" / "domain"
raw_path = basin_dir / f"{BASIN_ID}_landuse.tif"
corrected_path = basin_dir / f"{BASIN_ID}_landuse_corrected.tif"

for p in (raw_path, corrected_path):
    if not p.exists():
        raise FileNotFoundError(
            f"{p} does not exist -- rule modelled_depth_estimation (or its empirical "
            f"sibling) must have run for basin {BASIN_ID!r} first."
        )

with rasterio.open(raw_path) as src:
    raw = src.read(1)
    profile = src.profile.copy()

with rasterio.open(corrected_path) as src:
    corrected = src.read(1)

if raw.shape != corrected.shape:
    raise ValueError(
        f"Shape mismatch: raw {raw.shape} vs corrected {corrected.shape} -- "
        f"landuse_corrected.tif should be a pixel-for-pixel copy of landuse.tif."
    )

changed = (raw == 200) & (corrected == 80)
still_ocean = (raw == 200) & (corrected == 200)

out = np.zeros(raw.shape, dtype=np.uint8)
out[still_ocean] = 1
out[changed] = 2

n_changed = int(changed.sum())
n_still_ocean = int(still_ocean.sum())
cell_area_km2 = abs(profile["transform"].a * profile["transform"].e) / 1e6
log.info(f"Basin {BASIN_ID}: raw landuse: {raw_path}")
log.info(f"Basin {BASIN_ID}: corrected landuse: {corrected_path}")
log.info(
    f"Relabeled (weir-enclosed) pockets: {n_changed:,} pixel(s) "
    f"({n_changed * cell_area_km2:.4f} km2)"
)
log.info(
    f"Still-open ocean (unchanged, landuse==200 in both): {n_still_ocean:,} pixel(s) "
    f"({n_still_ocean * cell_area_km2:.4f} km2)"
)
if n_changed == 0:
    log.warning(
        "No pixels were relabeled -- either this basin's weir doesn't enclose any "
        "ocean-classified pockets, or rule modelled_depth_estimation hasn't picked up "
        "the fix yet (rerun it first)."
    )

profile.update(dtype="uint8", nodata=None, count=1)
out_path = (
    Path(__file__).resolve().parent / f"{BASIN_ID}_landuse_correction_diagnostic.tif"
)
with rasterio.open(out_path, "w", **profile) as dst:
    dst.write(out, 1)
log.info(f"Diagnostic raster written: {out_path}")
log.info("Values: 0=land/other, 1=still-open ocean (200), 2=relabeled pocket (200->80)")
