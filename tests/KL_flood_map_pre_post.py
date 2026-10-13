"""
Max flood depth of one adaptation strategy for one event, modelled (PRE) and
postprocessed (POST) side by side on the same colour scale, plus a third panel
showing where the two agree and disagree. PRE is the reference (same definitions and
scores as KL_csi.py):

    hit          wet in PRE and in POST
    miss         wet in PRE, dry in POST  (flooding POST does not account for)
    false alarm  dry in PRE, wet in POST  (flooding POST adds that PRE does not have)
    dry in both  land that neither map floods (an agreement too)
"""

import os
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))
from src.plots import read_raster_reprojected_for_plot  # noqa: E402
from KL_csi import THRESHOLD, contingency, land_mask, wet_map  # noqa: E402

BASIN_ID = "2433835"
# Machine-specific path, read from the GCFM_RESULTS_DIR environment variable.
# Set it once in PowerShell, then restart your terminal (see
# CONTRIBUTING.md "Local machine paths"):
#   [Environment]::SetEnvironmentVariable("GCFM_RESULTS_DIR", "D:\your\results\path", "User")
BASIN_ROOT = Path(os.environ["GCFM_RESULTS_DIR"]) / BASIN_ID
RUNS = BASIN_ROOT / "runs"
DOMAIN = BASIN_ROOT / "preprocessing_inputs" / "domain"

EVENT = "river_500"  # coast_100, river_500, compound_100c_500r
STRATEGY = "accommodate_09"  # <measure>_<scale>
VMAX = 2.0  # m, top of the colour scale; deeper cells (e.g. the river channel) saturate

PANELS = [("PRE", "Modelled (re-run)"), ("POST", "Postprocessed (not re-run)")]
AGREE_COLORS = ["#b8c4d0", "#e66101", "#5e3c99"]  # hit, miss, false alarm
DRY_BOTH_COLOR = (
    "#dcebd5"  # land dry in both, drawn as the land colour of the third panel
)


def tif_path(method):
    return (
        RUNS / EVENT / "adaptation" / method.lower() / STRATEGY / "max_flood_depth.tif"
    )


# ------------------------------------------------ hits / misses / false alarms / dry (km2)
wet_pre, transform = wet_map(tif_path("PRE"))
wet_post, _ = wet_map(tif_path("POST"))
c = contingency(wet_pre, wet_post, land_mask(tif_path("PRE")), transform)
del wet_pre, wet_post
print(
    f"{EVENT} {STRATEGY}: hits {c['hits_km2']:.2f} km2, misses {c['misses_km2']:.2f} km2, "
    f"false alarms {c['false_alarms_km2']:.2f} km2, dry in both {c['dry_both_km2']:.2f} km2 | "
    f"CSI {c['CSI']:.2f}, agree {c['agree']:.2f}, HSS {c['HSS']:.2f}"
)

# ------------------------------------------------ maps (reprojected for plotting)
arrays = {}
bounds = None  # (left, bottom, right, top) of the first panel, reused so all align
for method, _ in PANELS:
    arr, (left, right, bottom, top) = read_raster_reprojected_for_plot(
        str(tif_path(method)), dst_bounds=bounds
    )
    bounds = bounds or (left, bottom, right, top)
    arrays[method] = arr
left, bottom, right, top = bounds

wet_p, wet_q = arrays["PRE"] > THRESHOLD, arrays["POST"] > THRESHOLD
agree = np.zeros(wet_p.shape, dtype=np.uint8)
agree[wet_p & wet_q] = 1
agree[wet_p & ~wet_q] = 2
agree[~wet_p & wet_q] = 3
agree = np.ma.masked_equal(agree, 0)  # dry in both -> transparent

margin = max(right - left, top - bottom) * 0.03
land = gpd.read_file(
    DOMAIN / f"{BASIN_ID}_land_polygons.gpkg",
    bbox=(left - margin, bottom - margin, right + margin, top + margin),
)
rivers = gpd.read_file(DOMAIN / f"{BASIN_ID}_river_network_clean.gpkg")
if rivers.crs is not None and rivers.crs.to_epsg() != 4326:
    rivers = rivers.to_crs("EPSG:4326")

fig, axes = plt.subplots(
    1, 3, figsize=(20, 6.5), facecolor="white", layout="constrained"
)
for ax in axes:
    land_color = DRY_BOTH_COLOR if ax is axes[2] else "#d9d9d9"
    land.plot(ax=ax, color=land_color, edgecolor="#aaaaaa", linewidth=0.3, zorder=1)

for ax, (method, title) in zip(axes[:2], PANELS):
    im = ax.imshow(
        arrays[method],
        cmap="Blues",
        vmin=0,
        vmax=VMAX,
        extent=(left, right, bottom, top),
        origin="upper",
        aspect="auto",
        zorder=2,
    )
    ax.set_title(title)
fig.colorbar(
    im,
    ax=axes[:2],
    extend="max",
    fraction=0.04,
    pad=0.02,
    shrink=0.75,
    label="Max flood depth (m)",
)

axes[2].imshow(
    agree,
    cmap=ListedColormap(AGREE_COLORS),
    norm=BoundaryNorm([0.5, 1.5, 2.5, 3.5], 3),
    extent=(left, right, bottom, top),
    origin="upper",
    aspect="auto",
    zorder=2,
)
axes[2].set_title(
    f"Agreement: {c['agree']:.0%} of cells | CSI {c['CSI']:.2f} | HSS {c['HSS']:.2f}"
)
axes[2].legend(
    handles=[
        Patch(
            facecolor=AGREE_COLORS[0],
            label=f"Hit: flooded in both ({c['hits_km2']:.1f} km²)",
        ),
        Patch(
            facecolor=DRY_BOTH_COLOR,
            edgecolor="#aaaaaa",
            label=f"Dry in both ({c['dry_both_km2']:.1f} km²)",
        ),
        Patch(
            facecolor=AGREE_COLORS[1],
            label=f"Miss: modelled only ({c['misses_km2']:.1f} km²)",
        ),
        Patch(
            facecolor=AGREE_COLORS[2],
            label=f"False alarm: postprocessed only ({c['false_alarms_km2']:.1f} km²)",
        ),
    ],
    loc="lower right",
    framealpha=0.9,
    fontsize=8,
)

for ax in axes:
    rivers.plot(ax=ax, color="steelblue", linewidth=0.6, zorder=3)
    ax.set_xlim(left - margin, right + margin)
    ax.set_ylim(bottom - margin, top + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°)")
axes[0].set_ylabel("Latitude (°)")
fig.suptitle(f"{EVENT} - {STRATEGY}")

out = RUNS / f"fig_flood_map_pre_post_{EVENT}_{STRATEGY}.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Wrote {out}")
