"""
plot_main_grid.py — Read an already-built SFINCS model's main computational
grid DEM (bed elevation, "dep") for one basin, write it out as a standalone
GeoTIFF, and plot it.

This is the elevation on the MAIN (coarse) computational grid SFINCS actually
solves hydrodynamics on (sfincs.dep) — not the finer subgrid reference
raster used for the subgrid volume/wet-fraction tables (dep_subgrid.tif,
see src.postprocessing.get_bed_level for that one instead).

Requires rule build_sfincs to have already run for the given basin_id
(reads results/{basin_id}/sfincs/sfincs.inp + sfincs.msk/.ind/.dep directly,
does not rebuild anything).

Usage:
    python tests/plot_main_grid.py <basin_id>
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))
from hydromt_sfincs import SfincsModel

REPO_ROOT = Path(__file__).resolve().parents[1]

if len(sys.argv) != 2:
    raise SystemExit("Usage: python tests/plot_main_grid.py <basin_id>")
basin_id = sys.argv[1]

with open(REPO_ROOT / "config" / "config.yml") as f:
    config = yaml.safe_load(f)
results_dir = Path(config["results_dir"])
sfincs_root = results_dir / str(basin_id) / "sfincs"
if not (sfincs_root / "sfincs.inp").exists():
    raise FileNotFoundError(
        f"No built SFINCS model found at {sfincs_root} "
        "(run rule build_sfincs for this basin_id first)."
    )

out_dir = REPO_ROOT / "figs" / "main_grid"
out_dir.mkdir(parents=True, exist_ok=True)
out_tif = out_dir / f"{basin_id}_grid_dem.tif"
out_png = out_dir / f"{basin_id}_grid_dem.png"

sf = SfincsModel(root=str(sfincs_root), mode="r")
sf.config.read()
sf.grid.read()

dep_da = sf.grid.data["dep"]
mask_da = sf.grid.data["mask"]

dep_da.raster.to_raster(str(out_tif), driver="GTiff", dtype="float32")
print(f"Written: {out_tif}")

# ── plot ──────────────────────────────────────────────────────────────────────
# Mask out inactive cells for a cleaner plot (still written to the tif as-is,
# unmasked — only the plot hides them).
dep_plot = dep_da.where(mask_da > 0)

fig, ax = plt.subplots(figsize=(8, 8))
im = dep_plot.plot(ax=ax, cmap="terrain", add_colorbar=False)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Elevation (m)")
resolution = float(sf.config.get("dx"))
ax.set_title(
    f"SFINCS main grid DEM — basin {basin_id}\n"
    f"{dep_da.sizes['y']}×{dep_da.sizes['x']} px @ {resolution:g} m"
)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(out_png, dpi=150)
plt.close(fig)
print(f"Plot written: {out_png}")

active_dep = dep_da.values[mask_da.values > 0]
print(
    f"Active-cell DEM stats: n={active_dep.size:,}, "
    f"min={np.nanmin(active_dep):.2f} m, max={np.nanmax(active_dep):.2f} m, "
    f"mean={np.nanmean(active_dep):.2f} m"
)
