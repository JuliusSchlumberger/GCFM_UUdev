"""
KL_hmin_sensitivity_figure.py -- How much the flooding depth threshold (hmin)
moves each strategy's residual risk, from the CSV KL_hmin_sensitivity.py writes.

Same layout as the scale-response figures in KL_results_figures.py (strategies
x flood types, residual against implementation scale): the lines are the
pipeline's own threshold (0.05 m), and a red bar at each point spans the
residual over every threshold. Residual = metric of the adapted run / metric of
the baseline at the SAME threshold, so the bars show whether the conclusions
move, not the absolute numbers.

Post-processed results at 0 m don't exist (see KL_hmin_sensitivity.py), so
their bars only span 0.05-0.1 m.

Output: <results>/<basin>/hmin_sensitivity/fig_hmin_sensitivity_<metric>.png
"""

import os
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASIN_ID = "2433835"
OUT_DIR = Path(os.environ["GCFM_RESULTS_DIR"]) / BASIN_ID / "hmin_sensitivity"
CSV = OUT_DIR / "metrics_hmin_sensitivity.csv"
REF_HMIN = 0.05  # the pipeline's own threshold: the lines
RANGE_COLOR = "#d62728"  # bars: range of the residual over the thresholds
RANGE_OFFSET = {"pre": -0.07, "post": 0.07}  # x nudge so the two bars don't overlap

# --- shared with KL_results_figures.py (scale-response figures) ---------------
EVENT_LABELS = {
    "coast_100_03SLR": "Coastal flood",
    "river_100_03SLR": "River flood",
    "compound_100_03SLR": "Compound flood",
}
MEASURE_ORDER = [
    "advance",
    "protect_closed",
    "grey_protect_open",
    "NbS_protect_open",
    "accommodate",
    "retreat",
]
STYLE = {
    "pre": dict(fmt="-o", color="#534eb8", label="Modelled (re-run)"),
    "post": dict(fmt="--s", color="#38b394", label="Postprocessed (not re-run)"),
}
METRIC_INFO = [
    ("flooded_area_km2", "residual flood extent", (0, 2)),
    ("urban_exposed_km2", "residual urban exposure", (0, 2)),
    ("volume_m3", "residual flood volume", (0, 3)),
    ("mean_depth_m", "residual mean flood depth", (0, 2)),
]
SCALES = ["04", "09", "1"]
X_LABELS = [
    "No adaptation",
    "Highly ineffective",
    "Marginally ineffective",
    "Full potential",
]
ABOVE_ONE_SQUEEZE = 0.4  # y-axis: linear 0-1, log-compressed above 1


def label(s):
    return s.replace("_", " ").title()


def _y_forward(y):
    y = np.asarray(y, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(y <= 1, y, 1 + ABOVE_ONE_SQUEEZE * np.log10(y))


def _y_inverse(x):
    x = np.asarray(x, dtype=float)
    with np.errstate(over="ignore"):
        return np.where(x <= 1, x, 10 ** ((x - 1) / ABOVE_ONE_SQUEEZE))


# ------------------------------------------------------------------------------


df = pd.read_csv(
    CSV, sep=None, engine="python"
)  # "," from the script, ";" once re-saved in Excel
df["scale"] = df.strategy.str.extract(r"_(04|09|1)$")[0]
df["measure"] = df.strategy.str.replace(r"_(04|09|1)$", "", regex=True)
events = [e for e in EVENT_LABELS if e in set(df.scenario)]
measures = [m for m in MEASURE_ORDER if m in set(df.measure)]
# thresholds each approach actually has results for (post: none below 0.05 m)
range_hmins = {
    ap: sorted(df.loc[(df.method == ap) & df.flooded_area_km2.notna(), "hmin"].unique())
    for ap in STYLE
}

for METRIC, LABEL, YLIM in METRIC_INFO:
    # residual at each threshold, against the baseline at the same threshold
    base = df[df.method == "baseline"].set_index(["scenario", "hmin"])[METRIC]
    runs = df[df.method != "baseline"].copy()
    runs["resid"] = (
        runs[METRIC]
        / base.reindex(pd.MultiIndex.from_frame(runs[["scenario", "hmin"]])).values
    )

    # extra width on the right for the legend (as in KL_results_figures.py)
    LEGEND_W = 2.0
    fig_w = 3 * len(events) + LEGEND_W
    fig, axes = plt.subplots(
        len(measures),
        len(events),
        figsize=(fig_w, 2.7 * len(measures)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for i, ms in enumerate(measures):
        for j, e in enumerate(events):
            A = axes[i, j]
            g = runs[(runs.scenario == e) & (runs.measure == ms)]

            for ap, s in STYLE.items():
                piv = (
                    g[g.method == ap]
                    .pivot_table(index="scale", columns="hmin", values="resid")
                    .reindex(SCALES)
                )
                ref = (
                    piv[REF_HMIN]
                    if REF_HMIN in piv
                    else pd.Series(np.nan, index=SCALES)
                )
                # scale 0 = no adaptation: residual 1 at every threshold
                A.plot(
                    X_LABELS,
                    [1.0] + ref.tolist(),
                    s["fmt"],
                    color=s["color"],
                    label=s["label"],
                    zorder=3,
                )
                # red bar per scale: residual range over every threshold
                x = np.arange(1, len(X_LABELS)) + RANGE_OFFSET[ap]
                lo, hi = piv.min(axis=1).values, piv.max(axis=1).values
                ok = np.isfinite(lo) & np.isfinite(hi)
                A.vlines(x[ok], lo[ok], hi[ok], color=RANGE_COLOR, lw=1.8, zorder=4)
                for y in (lo[ok], hi[ok]):  # caps
                    A.plot(x[ok], y, "_", color=RANGE_COLOR, ms=7, mew=1.8, zorder=4)

            A.set_yscale("function", functions=(_y_forward, _y_inverse))
            A.set_ylim(*YLIM)
            A.axhspan(1, YLIM[1], color="crimson", alpha=0.07)
            # only 1 and the top inside the thin band above 1, so its labels don't collide
            ticks = [0, 0.25, 0.5, 0.75, 1, YLIM[1]]
            A.set_yticks(ticks)
            A.set_yticklabels([f"{t:g}" for t in ticks], fontsize=11)
            A.tick_params(labelsize=11, labelrotation=45)
            # end each rotated label at its tick, so neighbouring labels don't overlap
            plt.setp(A.get_xticklabels(), ha="right", rotation_mode="anchor")
            if i == 0:
                A.set_title(EVENT_LABELS[e], fontsize=13)
            if j == 0:
                A.set_ylabel(LABEL, fontsize=12)
                A.annotate(
                    label(ms),
                    xy=(-0.5, 0.5),
                    xycoords="axes fraction",
                    fontsize=14,
                    rotation=90,
                    va="center",
                    ha="center",
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(
        Line2D([], [], color=RANGE_COLOR, ls="none", marker="|", ms=14, mew=1.8)
    )
    labels.append(
        "Range over flooding thresholds ("
        + ", ".join(
            f"{s['label'].split(' (')[0].lower()} {range_hmins[ap][0]:g}-{range_hmins[ap][-1]:g} m"
            for ap, s in STYLE.items()
        )
        + ")"
    )
    handles.append(Patch(facecolor="crimson", alpha=0.1, edgecolor="none"))
    labels.append("Residual risk")
    # one column on the right, long labels wrapped to keep it narrow
    right = 1 - LEGEND_W / fig_w
    fig.legend(
        handles,
        [textwrap.fill(t, 16, break_on_hyphens=False) for t in labels],
        loc="center left",
        bbox_to_anchor=(right, 0.5),
        ncol=1,
        frameon=False,
        fontsize=11,
        labelspacing=1.2,
    )
    plt.tight_layout(rect=[0, 0, right, 1])
    out = OUT_DIR / f"fig_hmin_sensitivity_{METRIC}.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")
