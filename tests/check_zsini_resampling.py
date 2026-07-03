"""
check_zsini_resampling.py — Interactive check: does switching
sf.initial_conditions.create()'s reproj_method from "average" to "nearest"
actually stop mostly-land coastal cells from starting "wet"?

Replicates hydromt_sfincs's initial_conditions.create() (mask_nodata, then
reproject_like at the SFINCS grid resolution) for both resampling methods,
then plots bed elevation plus the two "starts wet" (zsini > bed) masks side
by side, so the claim can be checked visually rather than taken on faith.

This opens an interactive matplotlib window (plt.show()) -- nothing is saved.

Usage:
    conda run -n hmt_sfincs_dev python tests/check_zsini_resampling.py [basin_id]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.warp import reproject

REPO_ROOT = Path(__file__).resolve().parents[1]
with open(REPO_ROOT / "config" / "config.yml") as fh:
    config = yaml.safe_load(fh)
RESULTS_DIR = Path(config["results_dir"])
DST_RES = float(config["sfincs"]["grid"]["resolution"])

basin_id = sys.argv[1] if len(sys.argv) > 1 else "2444235"
basin_dir = RESULTS_DIR / basin_id / "inputs" / "domain"
zsini_path = basin_dir / f"{basin_id}_zsini.tif"
elev_path = basin_dir / f"{basin_id}_elevation_merged.tif"


def resample(
    path: Path, method: Resampling, dst_transform, width: int, height: int, crs
):
    with rasterio.open(path) as src:
        arr = src.read(1)
        nodata = src.nodata
        transform = src.transform
        src_crs = src.crs
    if nodata is not None:
        arr = np.where(arr == np.float32(nodata), np.nan, arr).astype(np.float32)
    dst = np.full((height, width), np.nan, dtype=np.float32)
    reproject(
        source=arr,
        destination=dst,
        src_transform=transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=method,
    )
    return dst


with rasterio.open(zsini_path) as src:
    bounds = src.bounds
    crs = src.crs

width = int(np.ceil((bounds.right - bounds.left) / DST_RES))
height = int(np.ceil((bounds.top - bounds.bottom) / DST_RES))
dst_transform = rasterio.transform.from_origin(
    bounds.left, bounds.top, DST_RES, DST_RES
)

elev_dst = resample(elev_path, Resampling.average, dst_transform, width, height, crs)
zsini_avg = resample(zsini_path, Resampling.average, dst_transform, width, height, crs)
zsini_near = resample(zsini_path, Resampling.nearest, dst_transform, width, height, crs)

with rasterio.open(zsini_path) as src:
    src_is_sea = (src.read(1) == np.float32(0.0)).astype(np.float32)
    src_transform = src.transform
sea_frac = np.full((height, width), np.nan, dtype=np.float32)
reproject(
    source=src_is_sea,
    destination=sea_frac,
    src_transform=src_transform,
    src_crs=crs,
    dst_transform=dst_transform,
    dst_crs=crs,
    src_nodata=None,
    dst_nodata=np.nan,
    resampling=Resampling.average,
)

wet_avg = np.isfinite(zsini_avg) & np.isfinite(elev_dst) & (zsini_avg > elev_dst)
wet_near = np.isfinite(zsini_near) & np.isfinite(elev_dst) & (zsini_near > elev_dst)
# the artifact: a cell that "starts wet" despite being MOSTLY land by area --
# real open-sea cells (sea_frac >= 0.5) are expected/correct to start wet and
# would otherwise swamp the plot, so they're shown separately from the artifact.
artifact_avg = wet_avg & (sea_frac < 0.5)
artifact_near = wet_near & (sea_frac < 0.5)

print(f"Basin {basin_id}: {width}x{height} cells @ {DST_RES:.0f} m")
print(
    f"'starts wet' cells -- average resampling:  {int(wet_avg.sum())} (of which majority-land: {int(artifact_avg.sum())})"
)
print(
    f"'starts wet' cells -- nearest resampling:   {int(wet_near.sum())} (of which majority-land: {int(artifact_near.sum())})"
)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

im0 = axes[0].imshow(elev_dst, cmap="terrain", vmin=-5, vmax=5)
axes[0].set_title("Bed elevation (m)")
fig.colorbar(im0, ax=axes[0], fraction=0.04, label="m")

axes[1].imshow(elev_dst, cmap="gray", vmin=-5, vmax=5)
ys, xs = np.where(wet_avg & ~artifact_avg)
axes[1].scatter(xs, ys, s=6, c="steelblue", label="starts wet, majority sea (expected)")
ys, xs = np.where(artifact_avg)
axes[1].scatter(xs, ys, s=20, c="red", label="starts wet, MAJORITY LAND (artifact)")
axes[1].set_title(f"reproj_method='average' (artifact n={int(artifact_avg.sum())})")
axes[1].legend(loc="upper right", fontsize=7)

axes[2].imshow(elev_dst, cmap="gray", vmin=-5, vmax=5)
ys, xs = np.where(wet_near & ~artifact_near)
axes[2].scatter(xs, ys, s=6, c="steelblue", label="starts wet, majority sea (expected)")
ys, xs = np.where(artifact_near)
axes[2].scatter(xs, ys, s=20, c="red", label="starts wet, MAJORITY LAND (artifact)")
axes[2].set_title(f"reproj_method='nearest' (artifact n={int(artifact_near.sum())})")
axes[2].legend(loc="upper right", fontsize=7)

fig.suptitle(
    f"Basin {basin_id}: zsini > bed elevation at t=0 -- red = majority-land cells that start wet anyway"
)
fig.tight_layout()
plt.show()
