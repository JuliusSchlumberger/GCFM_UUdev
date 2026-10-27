"""
Agreement of the postprocessed (POST) flood map with the modelled (PRE, re-run) flood
map, per flood event and strategy (measure x scale). PRE is the reference. Cells are
only wet (depth > THRESHOLD) or dry; dry-in-both is counted only on land cells (see
land_mask) that are wet in the no-adaptation baseline (<event>/visuals/max_flood_depth.tif),
so it means both agree the measure protects the cell, not that it never floods.

    hits          wet in PRE and in POST
    misses        wet in PRE, dry in POST  (POST over-credits the measure)
    false alarms  dry in PRE, wet in POST  (POST under-credits the measure)
    dry in both   dry in PRE and in POST   (protected in both: agreeing on the risk reduction)

    CSI    hits / (hits + misses + false alarms)         ignores the dry-in-both cells
    agree  (hits + dry in both) / all compared cells     counts them; high for weak measures,
                                                         where most cells stay wet in both
    HSS    Heidke skill score: 2(hits*dry - misses*fa) / ((h+m)(m+d) + (h+fa)(fa+d))
           counts them but discounts the agreement expected by chance: 1 = perfect,
           0 = no better than chance, < 0 = worse than chance

A strategy is compared only if it has a max_flood_depth.tif under both
<event>/adaptation/pre/<strategy>/ and <event>/adaptation/post/<strategy>/.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib as mpl

# Machine-specific path, read from the GCFM_RESULTS_DIR environment variable.
# Set it once in PowerShell, then restart your terminal (see
# CONTRIBUTING.md "Local machine paths"):
#   [Environment]::SetEnvironmentVariable("GCFM_RESULTS_DIR", "D:\your\results\path", "User")
BASIN = Path(os.environ["GCFM_RESULTS_DIR"]) / "2433835"
RUNS = BASIN / "runs"
EVENTS = ["coast_100_03SLR", "river_100_03SLR", "compound_100_03SLR"]
EVENT_TITLES = ["Coastal flood", "River flood", "Compound flood"]  # one per EVENTS
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


def contingency(wet_pre, wet_post, domain, transform):
    """Hits / misses / false alarms / dry-in-both (km2) and the scores.

    Wet cells count wherever they are (also on land reclaimed from the sea by NbS, which
    the baseline sea mask still calls sea, and on cells a measure floods that were dry in
    the baseline); dry-in-both counts `domain` cells only (land wet in the baseline).
    """
    h = int((wet_pre & wet_post).sum())
    m = int((wet_pre & ~wet_post).sum())
    f = int((~wet_pre & wet_post).sum())
    d = int((~wet_pre & ~wet_post & domain).sum())
    km2 = abs(transform.a * transform.e) / 1e6  # area of one cell
    denom = (h + m) * (m + d) + (h + f) * (f + d)
    return dict(
        hits_km2=h * km2,
        misses_km2=m * km2,
        false_alarms_km2=f * km2,
        protected_both_km2=d * km2,
        CSI=h / (h + m + f) if h + m + f else float("nan"),
        agree=(h + d) / (h + m + f + d),
        HSS=2 * (h * d - m * f) / denom if denom else float("nan"),
    )


MEASURES = [
    "retreat",
    "accommodate",
    "NbS_protect_open",
    "grey_protect_open",
    "protect_closed",
    "advance",
]
MEASURE_LABELS = [
    "Retreat",
    "Accommodate",
    "Nbs Protect Open",
    "Grey Protect Open",
    "Protect Closed",
    "Advance",
]
HEATMAP_SCALES = ["1", "09", "04"]  # full potential -> highly ineffective
SCALE_LABELS = {
    "04": "Highly\nineffective",
    "09": "Marginally\nineffective",
    "1": "Full\npotential",
}
CSI_COLORS = ["#e57e7f", "#e9b1b5", "#fbe2d5", "#b4e4d9", "#7ad3b6"]  # low -> high
CSI_BOUNDS = [0, 0.3, 0.5, 0.7, 0.9, 1.0]


def plot_heatmap(df, numbers, out):
    """CSI per event x measure x scale; ▲/▼ mark where POST over/under-floods PRE.

    numbers=True writes the CSI value in each cell, otherwise only the bias marks.
    """
    df = df.assign(
        scale=df.scale.astype(str),
        bias=(df.hits_km2 + df.false_alarms_km2) / (df.hits_km2 + df.misses_km2),
    )
    cmap = ListedColormap(CSI_COLORS)
    norm = BoundaryNorm(CSI_BOUNDS, cmap.N)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4.6), sharey=True)
    for ax, e, t in zip(axes, EVENTS, EVENT_TITLES):
        sub = df[df.event == e].set_index(["measure", "scale"])
        Z = np.array(
            [[sub.loc[(m, s), "CSI"] for s in HEATMAP_SCALES] for m in MEASURES]
        )
        B = np.array(
            [[sub.loc[(m, s), "bias"] for s in HEATMAP_SCALES] for m in MEASURES]
        )
        ax.imshow(Z, cmap=cmap, norm=norm, aspect="auto")
        for i in range(len(MEASURES)):
            for j in range(len(HEATMAP_SCALES)):
                z, b = Z[i, j], B[i, j]
                mark = "▲" if b > 1.25 else ("▼" if b < 0.8 else "")
                if numbers:
                    text, size = f"{z:.2f}{' ' + mark if mark else ''}", 9
                else:
                    text, size = mark, 12
                if text:
                    ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        fontsize=size,
                        color="black",
                    )
        ax.set_xticks(range(3))
        ax.set_xticklabels([SCALE_LABELS[s] for s in HEATMAP_SCALES], fontsize=8.5)
        ax.set_title(t, fontsize=11)
        ax.set_xticks(np.arange(-0.5, 3), minor=True)
        ax.set_yticks(np.arange(-0.5, 6), minor=True)
        ax.grid(which="minor", color="white", lw=1.5)
        ax.tick_params(which="both", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_yticks(range(6))
    axes[0].set_yticklabels(MEASURE_LABELS, fontsize=10)

    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        fraction=0.025,
        pad=0.02,
        spacing="uniform",
    )
    cb.set_ticks([(a + b) / 2 for a, b in zip(CSI_BOUNDS[:-1], CSI_BOUNDS[1:])])
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
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


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
        # no-adaptation flood map: dry-in-both only counts where this one is wet
        wet_base, tf_base = wet_map(RUNS / event / "visuals" / "max_flood_depth.tif")

        for name in sorted(pre & post):
            measure, scale = name.rsplit("_", 1)
            wet_pre, tf_pre = wet_map(pre_dir / name / "max_flood_depth.tif")
            wet_post, tf_post = wet_map(post_dir / name / "max_flood_depth.tif")
            if wet_pre.shape != wet_post.shape or tf_pre != tf_post:
                raise ValueError(
                    f"{event}/{name}: PRE and POST are not on the same grid"
                )
            if wet_base.shape != wet_pre.shape or tf_base != tf_pre:
                raise ValueError(
                    f"{event}/{name}: baseline and PRE are not on the same grid"
                )
            if land is None:  # one grid for every run
                land = land_mask(pre_dir / name / "max_flood_depth.tif")
                print(
                    f"Land domain: {land.sum() * abs(tf_pre.a * tf_pre.e) / 1e6:.2f} km2"
                )
            rows.append(
                dict(event=event, scale=scale, measure=measure)
                | contingency(wet_pre, wet_post, land & wet_base, tf_pre)
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

    # one heat map with the CSI written in each cell, one with only the bias marks
    plot_heatmap(df, numbers=True, out=RUNS / "fig_csi_heatmap_numbers.png")
    plot_heatmap(df, numbers=False, out=RUNS / "fig_csi_heatmap_no_numbers.png")
