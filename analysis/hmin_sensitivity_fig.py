"""Plot how the flooding depth threshold (hmin) moves each strategy's residual risk.

Reads hmin_sensitivity.py's CSV. Same layout as pre_post_response_figures.py's scale-response
figures (strategies x flood types, residual against implementation scale): the
lines are the pipeline's own threshold (config metrics.hmin), and a red bar at
each point spans the residual over every threshold. Residual = metric of the
adapted run / metric of the baseline at the same threshold, so the bars show
whether the conclusions move, not the absolute numbers. Post-processed results
below metrics.hmin do not exist, so their bars span only metrics.hmin and up.

Output: <results_dir>/<basin>/figures/analysis/90_hmin_sensitivity_fig_<metric>.png
"""

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from _common import (
    DEFAULT_BASIN_ID,
    REPO_ROOT,
    analysis_dir,
    analysis_path,
    label,
    script_parser,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pre_post_response_figures import (
    EVENT_LABELS,
    MEASURE_ORDER,
    METRIC_INFO,
    SCALES,
    STYLE,
    X_LABELS,
    style_residual_axis,
)

RANGE_COLOR = "#d62728"  # bars: range of the residual over the thresholds
RANGE_OFFSET = {"PRE": -0.07, "POST": 0.07}  # x nudge so the two bars don't overlap
LEGEND_W = 2.0  # inches on the right for the legend


def plot_metric(df, metric, ylabel, ylim, ref_hmin, basin_id):
    """Write one strategies x events figure of residual `metric` against scale."""
    events = [e for e in EVENT_LABELS if e in set(df.scenario)]
    measures = [m for m in MEASURE_ORDER if m in set(df.measure)]
    # thresholds each approach has results for (post: none below ref_hmin)
    range_hmins = {
        ap: sorted(
            df.loc[(df.approach == ap) & df.flooded_area_km2.notna(), "hmin"].unique()
        )
        for ap in STYLE
    }

    # residual at each threshold, against the baseline at the same threshold
    base = df[df.method == "baseline"].set_index(["scenario", "hmin"])[metric]
    runs = df[df.method != "baseline"].copy()
    runs["resid"] = (
        runs[metric]
        / base.reindex(pd.MultiIndex.from_frame(runs[["scenario", "hmin"]])).values
    )

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
            ax = axes[i, j]
            g = runs[(runs.scenario == e) & (runs.measure == ms)]
            for ap, s in STYLE.items():
                piv = (
                    g[g.approach == ap]
                    .pivot_table(index="scale", columns="hmin", values="resid")
                    .reindex(SCALES)
                )
                ref = (
                    piv[ref_hmin]
                    if ref_hmin in piv
                    else pd.Series(np.nan, index=SCALES)
                )
                # scale 0 = no adaptation: residual 1 at every threshold
                ax.plot(
                    X_LABELS,
                    [1.0] + ref.tolist(),
                    s["fmt"],
                    color=s["color"],
                    label=s["label"],
                    zorder=3,
                )
                x = np.arange(1, len(X_LABELS)) + RANGE_OFFSET[ap]
                lo, hi = piv.min(axis=1).values, piv.max(axis=1).values
                ok = np.isfinite(lo) & np.isfinite(hi)
                ax.vlines(x[ok], lo[ok], hi[ok], color=RANGE_COLOR, lw=1.8, zorder=4)
                for y in (lo[ok], hi[ok]):  # caps
                    ax.plot(x[ok], y, "_", color=RANGE_COLOR, ms=7, mew=1.8, zorder=4)

            style_residual_axis(ax, ylim)
            if i == 0:
                ax.set_title(EVENT_LABELS[e], fontsize=13)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=12)
                ax.annotate(
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
            f"{s['label'].split(' (')[0].lower()} "
            f"{range_hmins[ap][0]:g}-{range_hmins[ap][-1]:g} m"
            for ap, s in STYLE.items()
            if range_hmins[ap]
        )
        + ")"
    )
    handles.append(Patch(facecolor="crimson", alpha=0.1, edgecolor="none"))
    labels.append("Residual risk")
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
    out = analysis_path(basin_id, __file__, f"{metric}.png")
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


def build_parser():
    """Return the command-line parser."""
    parser = script_parser(__doc__)
    parser.add_argument(
        "--basin-id",
        default=DEFAULT_BASIN_ID,
        help=f"basin id (default: {DEFAULT_BASIN_ID})",
    )
    return parser


def main():
    """Write one hmin-sensitivity figure per metric in METRIC_INFO."""
    args = build_parser().parse_args()
    basin_id = str(args.basin_id)
    cfg = yaml.safe_load(
        (REPO_ROOT / "config" / "config.yml").read_text(encoding="utf-8")
    )
    ref_hmin = float(cfg["metrics"]["hmin"])

    csv = analysis_path(basin_id, "hmin_sensitivity.py", "metrics.csv")
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} missing; run analysis/hmin_sensitivity.py first "
            f"(output folder {analysis_dir(basin_id)})"
        )
    df = pd.read_csv(csv, sep=None, engine="python")  # ";" once re-saved in Excel
    df["approach"] = df["method"].str.upper()  # PRE | POST | BASELINE
    df["scale"] = df.strategy.str.extract(r"_(04|09|1)$")[0]
    df["measure"] = df.strategy.str.replace(r"_(04|09|1)$", "", regex=True)

    for metric, ylabel, ylim in METRIC_INFO:
        plot_metric(df, metric, ylabel, ylim, ref_hmin, basin_id)


if __name__ == "__main__":
    main()
