"""
Agreement of the postprocessed (POST) flood map with the modelled (PRE, re-run) flood
map, per flood event and strategy (measure x scale). PRE is the reference. Cells are
only wet (depth > THRESHOLD) or dry; dry-in-both is counted on land cells (see land_mask).

    hits          wet in PRE and in POST
    misses        wet in PRE, dry in POST  (POST under-credits the measure)
    false alarms  dry in PRE, wet in POST  (POST over-credits the measure)
    dry in both   dry in PRE and in POST   (correct negatives: agreeing there is no flood)

    CSI    hits / (hits + misses + false alarms)         ignores the dry-in-both cells
    agree  (hits + dry in both) / all land cells         counts them, but dominated by dry land
    HSS    Heidke skill score: 2(hits*dry - misses*fa) / ((h+m)(m+d) + (h+fa)(fa+d))
           counts them but discounts the agreement expected by chance: 1 = perfect,
           0 = no better than chance, < 0 = worse than chance

A strategy is compared only if it has a max_flood_depth.tif under both
<event>/adaptation/pre/<strategy>/ and <event>/adaptation/post/<strategy>/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib as mpl

BASIN = Path(r"D:\GCFM_UU\results\2433835")
RUNS = BASIN / "runs"
EVENTS = ["coast_100", "river_500", "compound_100c_500r"]
SCALES = ["04", "09", "1"]  # low -> high implementation scale
THRESHOLD = 0.05  # m, wet/dry cutoff (the pipeline's hmin)
# land domain, as the pipeline's flood metrics define it (compute_max_inundation):
# cells with a valid bed level that are not open sea (sea mask == 1)
DEP_SUBGRID = BASIN / "sfincs_skeleton" / "subgrid" / "dep_subgrid.tif"
SEA_MASK = (
    BASIN / "preprocessing_inputs" / "domain" / "2433835_zsini_sea_cells_on_grid.tif"
)
OUT_CSV = RUNS / "csi_pre_vs_post.csv"


def wet_map(path):
    with rasterio.open(path) as src:
        # nodata (-9999) is below the threshold, so it counts as dry
        return src.read(1) > THRESHOLD, src.transform


def land_mask(like_path):
    """Land cells on the grid of the flood map at `like_path`."""
    with rasterio.open(like_path) as like:
        shape, transform, crs = like.shape, like.transform, like.crs

    def on_grid(path):
        out = np.full(shape, np.nan, dtype="float32")
        with rasterio.open(path) as src:
            reproject(
                rasterio.band(src, 1),
                out,
                src_nodata=src.nodata,
                dst_transform=transform,
                dst_crs=crs,
                dst_nodata=np.nan,
                resampling=Resampling.nearest,
            )
        return out

    return np.isfinite(on_grid(DEP_SUBGRID)) & (on_grid(SEA_MASK) != 1.0)


def contingency(wet_pre, wet_post, land, transform):
    """Hits / misses / false alarms / dry-in-both (km2) and the scores.

    Wet cells count wherever they are (also on land reclaimed from the sea by NbS, which
    the baseline sea mask still calls sea); dry-in-both counts baseline land cells only.
    """
    h = int((wet_pre & wet_post).sum())
    m = int((wet_pre & ~wet_post).sum())
    f = int((~wet_pre & wet_post).sum())
    d = int((~wet_pre & ~wet_post & land).sum())
    km2 = abs(transform.a * transform.e) / 1e6  # area of one cell
    denom = (h + m) * (m + d) + (h + f) * (f + d)
    return dict(
        hits_km2=h * km2,
        misses_km2=m * km2,
        false_alarms_km2=f * km2,
        dry_both_km2=d * km2,
        CSI=h / (h + m + f) if h + m + f else float("nan"),
        agree=(h + d) / (h + m + f + d),
        HSS=2 * (h * d - m * f) / denom if denom else float("nan"),
    )


if __name__ == "__main__":
    rows = []
    land = None
    for event in EVENTS:
        pre_dir = RUNS / event / "adaptation" / "pre"
        post_dir = RUNS / event / "adaptation" / "post"
        pre = {p.parent.name for p in pre_dir.glob("*/max_flood_depth.tif")}
        post = {p.parent.name for p in post_dir.glob("*/max_flood_depth.tif")}
        for name in sorted(pre ^ post):
            print(
                f"{event}: {name} only has {'PRE' if name in pre else 'POST'} -- skipped"
            )

        for name in sorted(pre & post):
            measure, scale = name.rsplit("_", 1)
            wet_pre, tf_pre = wet_map(pre_dir / name / "max_flood_depth.tif")
            wet_post, tf_post = wet_map(post_dir / name / "max_flood_depth.tif")
            if wet_pre.shape != wet_post.shape or tf_pre != tf_post:
                raise ValueError(
                    f"{event}/{name}: PRE and POST are not on the same grid"
                )
            if land is None:  # one grid for every run
                land = land_mask(pre_dir / name / "max_flood_depth.tif")
                print(
                    f"Land domain: {land.sum() * abs(tf_pre.a * tf_pre.e) / 1e6:.2f} km2"
                )
            rows.append(
                dict(event=event, scale=scale, measure=measure)
                | contingency(wet_pre, wet_post, land, tf_pre)
            )

    df = pd.DataFrame(rows)
    scale_order = SCALES + sorted(set(df.scale) - set(SCALES))
    df["event"] = pd.Categorical(df.event, EVENTS, ordered=True)
    df["scale"] = pd.Categorical(df.scale, scale_order, ordered=True)
    df = df.sort_values(
        ["event", "scale", "measure"],
        key=lambda s: s.str.lower() if s.dtype == object else s,
    )
    df.to_csv(OUT_CSV, index=False)

    for event, g in df.groupby("event", observed=True):
        print(f"\n# {event}  (PRE = reference, wet > {THRESHOLD} m, areas in km2)")
        print(g.drop(columns="event").round(2).to_string(index=False))
    print(f"\nWrote {OUT_CSV}")


# ------------------------------------------------------------------------------------------
# Heat map with numbers
df = pd.read_csv(OUT_CSV)
df["bias"] = (df.hits_km2 + df.false_alarms_km2) / (df.hits_km2 + df.misses_km2)

measures = [
    "retreat",
    "accommodate",
    "NbS_protect_open",
    "grey_protect_open",
    "protect_closed",
    "advance",
]
m_lab = [
    "Retreat",
    "Accommodate",
    "Nbs Protect Open",
    "Grey Protect Open",
    "Protect Closed",
    "Advance",
]
events = ["coast_100", "river_500", "compound_100c_500r"]
e_lab = ["Coast\n100", "River\n500", "Compound\n100C 500R"]
scales = [1, 9, 4]  # full potential -> highly ineffective
scale_lab = {
    4: "Highly\nineffective",
    9: "Marginally\nineffective",
    1: "Full\npotential",
}

colors = ["#e57e7f", "#e9b1b5", "#fbe2d5", "#b4e4d9", "#7ad3b6"]  # low -> high
bounds = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
cmap = ListedColormap(colors)
norm = BoundaryNorm(bounds, cmap.N)

fig, axes = plt.subplots(1, 3, figsize=(10, 4.6), sharey=True)
e_title = ["Coast 100", "River 500", "Compound 100C 500R"]
for ax, e, t in zip(axes, events, e_title):
    sub = df[df.event == e].set_index(["measure", "scale"])
    Z = np.array([[sub.loc[(m, s), "CSI"] for s in scales] for m in measures])
    B = np.array([[sub.loc[(m, s), "bias"] for s in scales] for m in measures])
    ax.imshow(Z, cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(measures)):
        for j in range(len(scales)):
            z, b = Z[i, j], B[i, j]
            mark = "▲" if b > 1.25 else ("▼" if b < 0.8 else "")
            ax.text(
                j,
                i,
                f"{z:.2f}{' ' + mark if mark else ''}",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )
    ax.set_xticks(range(3))
    ax.set_xticklabels([scale_lab[s] for s in scales], fontsize=8.5)
    ax.set_title(t, fontsize=11)
    ax.set_xticks(np.arange(-0.5, 3), minor=True)
    ax.set_yticks(np.arange(-0.5, 6), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
axes[0].set_yticks(range(6))
axes[0].set_yticklabels(m_lab, fontsize=10)

cb = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axes,
    fraction=0.025,
    pad=0.02,
    spacing="uniform",
)
cb.set_ticks([(a + b) / 2 for a, b in zip(bounds[:-1], bounds[1:])])
cb.set_ticklabels(["<0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-1"])
cb.ax.tick_params(length=0, labelsize=9)
cb.outline.set_visible(False)
cb.set_label("CSI (modelled vs postprocessed max flood depth)", fontsize=9)
fig.text(
    0.02,
    -0.02,
    "▲ risk overestimated by postprocessed         "
    "▼ risk underestimated by postprocessed",
    fontsize=8.5,
)
fig.savefig(RUNS / "fig_csi_heatmap_numbers.png", dpi=300, bbox_inches="tight")


# ------------------------------------------------------------------------------------------
# No numbers but same pot:
df = pd.read_csv(OUT_CSV)
df["bias"] = (df.hits_km2 + df.false_alarms_km2) / (df.hits_km2 + df.misses_km2)

measures = [
    "retreat",
    "accommodate",
    "NbS_protect_open",
    "grey_protect_open",
    "protect_closed",
    "advance",
]
m_lab = [
    "Retreat",
    "Accommodate",
    "Nbs Protect Open",
    "Grey Protect Open",
    "Protect Closed",
    "Advance",
]
events = ["coast_100", "river_500", "compound_100c_500r"]
e_lab = ["Coast\n100", "River\n500", "Compound\n100C 500R"]
scales = [1, 9, 4]  # full potential -> highly ineffective
scale_lab = {
    4: "Highly\nineffective",
    9: "Marginally\nineffective",
    1: "Full\npotential",
}

colors = ["#e57e7f", "#e9b1b5", "#fbe2d5", "#b4e4d9", "#7ad3b6"]  # low -> high
bounds = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
cmap = ListedColormap(colors)
norm = BoundaryNorm(bounds, cmap.N)

fig, axes = plt.subplots(1, 3, figsize=(10, 4.6), sharey=True)
e_title = ["Coast 100", "River 500", "Compound 100C 500R"]
for ax, e, t in zip(axes, events, e_title):
    sub = df[df.event == e].set_index(["measure", "scale"])
    Z = np.array([[sub.loc[(m, s), "CSI"] for s in scales] for m in measures])
    B = np.array([[sub.loc[(m, s), "bias"] for s in scales] for m in measures])
    ax.imshow(Z, cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(measures)):
        for j in range(len(scales)):
            z, b = Z[i, j], B[i, j]
            mark = "▲" if b > 1.25 else ("▼" if b < 0.8 else "")
            if mark:
                ax.text(
                    j, i, mark, ha="center", va="center", fontsize=12, color="black"
                )
    ax.set_xticks(range(3))
    ax.set_xticklabels([scale_lab[s] for s in scales], fontsize=8.5)
    ax.set_title(t, fontsize=11)
    ax.set_xticks(np.arange(-0.5, 3), minor=True)
    ax.set_yticks(np.arange(-0.5, 6), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
axes[0].set_yticks(range(6))
axes[0].set_yticklabels(m_lab, fontsize=10)

cb = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axes,
    fraction=0.025,
    pad=0.02,
    spacing="uniform",
)
cb.set_ticks([(a + b) / 2 for a, b in zip(bounds[:-1], bounds[1:])])
cb.set_ticklabels(["<0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-1"])
cb.ax.tick_params(length=0, labelsize=9)
cb.outline.set_visible(False)
cb.set_label("CSI (modelled vs postprocessed max flood depth)", fontsize=9)
fig.text(
    0.02,
    -0.02,
    "▲ risk overestimated by postprocessed          "
    "▼ risk underestimated by postprocessed",
    fontsize=8.5,
)
fig.savefig(RUNS / "fig_csi_heatmap_no_numbers.png", dpi=300, bbox_inches="tight")
