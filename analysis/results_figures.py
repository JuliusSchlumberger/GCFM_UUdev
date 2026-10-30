"""Plot POST vs PRE agreement of adaptation measures for the results section.

Reads metrics_comparison.csv; risk reduction is relative to the matching event's
baseline row. Also plots measure rankings and a basin comparison of return-period
curves (the single-basin curves and forced hydrographs are in domain_figures.py).
Writes to <results_dir>/2433835/figures/analysis/ (90_results_figures_<quantity>.png);
the basin comparison of return-period curves to <results_dir>/figures/analysis/.
"""

import textwrap

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import analysis_dir, analysis_path, label, script_parser, site_path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASIN_ID = "2433835"
METRICS = ["flooded_area_km2", "urban_exposed_km2", "mean_depth_m", "volume_m3"]
SCALES = ["04", "09", "1"]  # strategy-name suffix, low -> high implementation scale

EVENT_ORDER = ["coast_100_03SLR", "river_100_03SLR", "compound_100_03SLR"]
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
    "PRE": dict(fmt="-o", color="#534eb8", label="Modelled (re-run)"),
    "POST": dict(fmt="--s", color="#38b394", label="Postprocessed (not re-run)"),
}
METRIC_INFO = [  # metric, y label, y limits
    ("flooded_area_km2", "residual flood extent", (0, 2)),
    ("urban_exposed_km2", "residual urban exposure", (0, 2)),
    ("volume_m3", "residual flood volume", (0, 3)),
    ("mean_depth_m", "residual mean flood depth", (0, 2)),
]
PLOT_SCALES = ["0"] + SCALES  # "0" = no-adaptation baseline (ratio == 1)
SCALE_LABELS = {
    "0": "No adaptation",
    "04": "Highly ineffective",
    "09": "Marginally ineffective",
    "1": "Full potential",
}
X_LABELS = [SCALE_LABELS[s] for s in PLOT_SCALES]

# y-axis: linear from 0 to 1 (the reduction range), log-compressed above 1 so the
# residual-risk band (> 1) stays a thin strip: each factor of 10 above 1 takes
# ABOVE_ONE_SQUEEZE of the height that 0-1 takes.
ABOVE_ONE_SQUEEZE = 0.4
# POST-PRE differences in residual ratio smaller than this are left unshaded
SHADE_TOL = 0.01
SHADE_ALPHA = 0.13

RANK_METRIC = "urban_exposed_km2"
TIE_DECIMALS = 2  # residual ratios equal at this precision share a rank ("=")
# a reduction smaller than this fraction of the baseline (incl. an increase) counts as
# "no effect", so all such measures tie for last; 0 = rank on the exact ratios
NO_EFFECT_TOL = 0.05
SCALE_MARKER = {"04": ("s", "#d62728"), "09": ("^", "#f0a30a"), "1": ("o", "#2ca02c")}

SCATTER_EVENTS = ["coast_100_03SLR"]
SCATTER_BAND = 1
SCATTER_DOT_SIZES = (40, 90, 150)  # one per scale in SCALES: small (04) -> big (1)

BUMP_METRICS = ["flooded_area_km2", "urban_exposed_km2", "volume_m3", "mean_depth_m"]
BUMP_SCALE = "1"
BUMP_APPROACHES = {"PRE": "Modelled rank", "POST": "Postprocessed rank"}
BUMP_NUDGE = 0.12  # rank units between tied lines
# categorical slots 1-6 of the dataviz reference palette, in MEASURE_ORDER
MEASURE_STYLE = dict(
    zip(
        MEASURE_ORDER,
        zip(
            ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
            ["o", "s", "^", "D", "v", "P"],
        ),
    )
)

COMPARE_BASINS = ["2433835", "3279946"]
BASIN_LABELS = {"3279946": "Chao Phraya (3279946)"}  # default: basin_id
BASIN_COLORS = ["#2a9d8f", "#e76f51", "#8338ec", "#ffb703"]  # cycled if > 4 basins


def pretty(s):
    """Return a measure name with spaces and only the first letter upper-cased."""
    s = s.replace("_", " ")
    return s[0].upper() + s[1:]  # keeps "NbS" (str.capitalize() would give "Nbs")


def event_label(e):
    """Return the figure title of a flood event."""
    return EVENT_LABELS.get(e, label(e))


def y_forward(y):
    """Map residual ratios to the y-axis: linear to 1, log-compressed above."""
    y = np.asarray(y, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(y <= 1, y, 1 + ABOVE_ONE_SQUEEZE * np.log10(y))


def y_inverse(x):
    """Invert y_forward."""
    x = np.asarray(x, dtype=float)
    with np.errstate(over="ignore"):
        return np.where(x <= 1, x, 10 ** ((x - 1) / ABOVE_ONE_SQUEEZE))


def style_residual_axis(ax, ylim):
    """Apply the residual-ratio y-axis, residual-risk band and tick layout."""
    ax.set_yscale("function", functions=(y_forward, y_inverse))
    ax.set_ylim(*ylim)
    ax.axhspan(1, ylim[1], color="crimson", alpha=0.07)
    # only 1 and the top inside the thin band above 1, so its labels don't collide
    ticks = [0, 0.25, 0.5, 0.75, 1, ylim[1]]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks], fontsize=11)
    ax.tick_params(labelsize=11, labelrotation=45)
    # end each rotated label at its tick, so neighbouring labels don't overlap
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")


def shade_divergence(ax, modelled, post, n=200):
    """Fill between the PRE and POST curves in the colour of the higher one.

    POST colour where POST is above PRE (POST underestimates effectiveness),
    PRE colour where below. Each segment between categorical x positions is
    interpolated in y_forward space, so the fill follows the plotted lines;
    segments with a NaN end stay unshaded.
    """
    pre_t = y_forward(modelled)
    post_t = y_forward(post)
    xp = np.arange(len(pre_t))
    # one grid over all segments, so each colour is one polygon (no seams at points)
    x = np.linspace(0, xp[-1], n * xp[-1] + 1)
    pre_y = y_inverse(np.interp(x, xp, pre_t))
    post_y = y_inverse(np.interp(x, xp, post_t))
    diff = post_y - pre_y
    for where, ap in ((diff > SHADE_TOL, "POST"), (diff < -SHADE_TOL, "PRE")):
        ax.fill_between(
            x,
            pre_y,
            post_y,
            where=where,
            color=STYLE[ap]["color"],
            alpha=SHADE_ALPHA,
            linewidth=0,
            zorder=0,
        )


def rank_by_residual(resid):
    """Return best-to-worst row order and rank; ties share the lowest rank."""
    effective = resid.where(resid < 1 - NO_EFFECT_TOL, 1.0)
    out = pd.DataFrame({"resid": resid, "val": effective.round(TIE_DECIMALS)})
    out = out.sort_values(["val", "resid"])
    out["row"] = np.arange(1, len(out) + 1)
    out["rank"] = out["val"].rank(method="min").astype(int)
    out["tied"] = out["val"].duplicated(keep=False)
    return out


def kendalls_w(vals):
    """Return the tie-corrected Kendall's W of a measures x metrics table."""
    vals = vals.dropna()
    n, m = vals.shape
    r = vals.rank(method="average")
    s = ((r.sum(axis=1) - m * (n + 1) / 2) ** 2).sum()
    t = sum((c**3 - c).sum() for c in (vals[col].value_counts() for col in vals))
    denom = m**2 * (n**3 - n) - m * t
    return 12 * s / denom if denom > 0 else np.nan


def load_comparison(csv):
    """Return (baseline rows by scenario, adapted rows with event/approach/scale)."""
    # "," from the pipeline, ";" once re-saved in Excel
    raw = pd.read_csv(csv, sep=None, engine="python")
    baselines = raw[raw.method == "baseline"].set_index("scenario")

    d = raw[raw.method != "baseline"].copy()
    d["event"] = d["scenario"]
    d["approach"] = d["method"].str.upper()  # PRE | POST
    d["scale"] = d.strategy.str.extract(r"_(04|09|1)$")[0]
    d["short"] = d.strategy.str.replace(r"_(04|09|1)$", "", regex=True)
    d["scale"] = pd.Categorical(d.scale, SCALES, ordered=True)
    if d["scale"].isna().any():
        raise ValueError("found a strategy name without a recognised _04/_09/_1 suffix")
    # a fully-protected run has flooded_area == 0 and NaN depth; depth is 0 by definition
    d[["mean_depth_m", "max_depth_m"]] = d[["mean_depth_m", "max_depth_m"]].fillna(0)
    return baselines, d


def plot_scale_response(d, baselines, events, measures):
    """Plot residual ratio against implementation scale, measures x events, per metric."""
    side_w = 2.0  # inches kept free on the right for an added graphic; 0 = none
    legend_h = 0.7  # inches above the column titles for the one-row legend
    for metric, ylabel, ylim in METRIC_INFO:
        d["ratio"] = d[metric] / d.event.map(baselines[metric])

        fig_w = 3 * len(events) + side_w
        fig_h = 2.7 * len(measures)
        fig, axes = plt.subplots(
            len(measures),
            len(events),
            figsize=(fig_w, fig_h),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for i, ms in enumerate(measures):
            for j, e in enumerate(events):
                ax = axes[i, j]
                g = d[(d.event == e) & (d.short == ms)]
                curves = {}
                for ap, s in STYLE.items():
                    gg = g[g.approach == ap].set_index("scale").reindex(SCALES)
                    ratios = [1.0] + gg["ratio"].tolist()  # scale 0 = no adaptation
                    ax.plot(
                        X_LABELS,
                        ratios,
                        s["fmt"],
                        color=s["color"],
                        label=s["label"],
                        zorder=3,
                    )
                    curves[ap] = ratios
                shade_divergence(ax, curves["PRE"], curves["POST"])
                style_residual_axis(ax, ylim)
                if i == 0:
                    ax.set_title(event_label(e), fontsize=13)
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
        for ap, text in (("POST", "underestimates"), ("PRE", "overestimates")):
            handles.append(
                Patch(facecolor=STYLE[ap]["color"], alpha=SHADE_ALPHA, edgecolor="none")
            )
            labels.append(f"Post-processed {text} effectiveness")
        handles.append(Patch(facecolor="crimson", alpha=0.1, edgecolor="none"))
        labels.append("Residual risk")
        right = 1 - side_w / fig_w
        top = 1 - legend_h / fig_h
        fig.legend(
            handles,
            [textwrap.fill(t, 16, break_on_hyphens=False) for t in labels],
            loc="upper center",
            bbox_to_anchor=(right / 2, 1),
            ncol=len(handles),
            frameon=False,
            fontsize=11,
        )
        plt.tight_layout(rect=[0, 0, right, top])
        fig.savefig(
            analysis_path(BASIN_ID, __file__, f"scale_response_{metric}.png"), dpi=150
        )
        plt.close(fig)

    print(
        f"\nWrote outputs to {analysis_dir(BASIN_ID)}: "
        "90_results_figures_scale_response_<metric>.png"
    )


def plot_return_period_comparison(results_root):
    """Overlay every COMPARE_BASINS basin's return-period curves, one colour each."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")

    for basin_id, color in zip(COMPARE_BASINS, BASIN_COLORS * len(COMPARE_BASINS)):
        forcing_dir = results_root / basin_id / "preprocessing_inputs" / "forcing"
        basin_label = BASIN_LABELS.get(basin_id, basin_id)

        # decode_times=False: units "hours since simulation start" have no
        # reference date for xarray's CF-time decoder.
        with xr.open_dataset(
            forcing_dir / "river_forcing.nc", decode_times=False
        ) as rds:
            active = rds["has_glofas"].values.astype(bool)
            rp_table = rds["discharge_rp_table"].values[active]
            i_main = np.nanargmax(rp_table[:, -1])
            ax1.plot(
                rds["rp_yr"].values, rp_table[i_main], color=color, label=basin_label
            )

        with xr.open_dataset(
            forcing_dir / "surge_forcing.nc", decode_times=False
        ) as sds:
            rp_yr = sds["rp_yr"].values
            for row in sds["storm_tide_rp_table"].values:
                ax2.plot(
                    rp_yr, row, color=color, lw=1, alpha=0.6, marker="o", markersize=2
                )
        # one legend entry per basin, not per station
        ax2.plot([], [], color=color, label=basin_label)

    ax1.set_xscale("log")
    ax1.set_xlabel("Return period (years)")
    ax1.set_ylabel("Discharge (m3/s)")
    ax1.set_title("River discharge (dominant crossing)")
    ax1.legend(fontsize=8)

    ax2.set_xscale("log")
    ax2.set_xlabel("Return period (years)")
    ax2.set_ylabel("Water level (m)")
    ax2.set_title("Coastal water level (per station)")
    ax2.legend(fontsize=8)

    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)

    fig.suptitle("Flood hazard return-period curves -- basin comparison")
    fig.tight_layout()
    out = analysis_path(
        None,
        __file__,
        "return_period_curves_comparison_" + "_vs_".join(COMPARE_BASINS) + ".png",
    )
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_rank_scatter(d, events):
    """Plot PRE rank against POST rank per measure and scale, one figure per event.

    Ranks are within each scale; the shaded band marks +-SCATTER_BAND ranks.
    """
    unknown = [e for e in SCATTER_EVENTS if e not in events]
    if unknown:
        raise ValueError(
            f"SCATTER_EVENTS {unknown} not in the results table; have {events}"
        )

    for e in SCATTER_EVENTS:
        frames = []
        for sc in SCALES:
            piv = d[(d.event == e) & (d.scale == sc)].pivot_table(
                index="short", columns="approach", values="resid", observed=True
            )
            if not {"PRE", "POST"} <= set(piv.columns) or piv.dropna().empty:
                continue
            w = piv.dropna()
            L, R = rank_by_residual(w["PRE"]), rank_by_residual(w["POST"])
            frames.append(
                pd.DataFrame(
                    {"scale": sc, "PRE_rank": L["rank"], "POST_rank": R["rank"]}
                )
                .rename_axis("measure")
                .reset_index()
            )
        if not frames:
            print(f"\n# {e}: no measure has both PRE and POST -- rank scatter skipped")
            continue
        pts = pd.concat(frames, ignore_index=True)
        pts["diff"] = pts["POST_rank"] - pts["PRE_rank"]

        n_out = int((pts["diff"].abs() > SCATTER_BAND).sum())
        print(
            f"\n# {e}: rank scatter ({RANK_METRIC}, ranks within each scale, 1 = lowest)"
        )
        print(pts.sort_values(["scale", "PRE_rank", "measure"]).to_string(index=False))
        print(f"  {n_out} of {len(pts)} dots lie outside the +-{SCATTER_BAND} band")

        n = int(max(pts["PRE_rank"].max(), pts["POST_rank"].max()))
        lim = np.array([0.5, n + 0.5])
        fig, ax = plt.subplots(figsize=(9.5, 7.5), facecolor="white")
        ax.fill_between(
            lim,
            lim - SCATTER_BAND,
            lim + SCATTER_BAND,
            color="#efefef",
            zorder=0,
            label=f"Within ±{SCATTER_BAND} rank{'s' * (SCATTER_BAND != 1)}",
        )
        ax.plot(lim, lim, color="#c8c8c8", lw=1, zorder=1)
        for i, (sc, size) in enumerate(zip(SCALES, SCATTER_DOT_SIZES)):
            g = pts[pts.scale == sc]
            # smaller dots on top, so dots at the same spot nest like rings
            ax.scatter(
                g["PRE_rank"],
                g["POST_rank"],
                s=size,
                color=SCALE_MARKER[sc][1],
                edgecolor="white",
                linewidth=0.8,
                zorder=3 + len(SCALES) - i,
                label=SCALE_LABELS[sc],
            )

        # one stacked label per spot: left of dots on/above the diagonal, else right
        for (x, y), g in pts.groupby(["PRE_rank", "POST_rank"]):
            names = "\n".join(pretty(m) for m in dict.fromkeys(g["measure"]))
            left = y >= x and x > 1.3
            ax.annotate(
                names,
                (x, y),
                xytext=(-9 if left else 9, 0),
                textcoords="offset points",
                ha="right" if left else "left",
                va="center",
                fontsize=8,
            )

        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal")
        ax.set_xticks(range(1, n + 1))
        ax.set_yticks(range(1, n + 1))
        ax.set_xlabel("Modelled rank")
        ax.set_ylabel("Postprocessed rank")
        ax.set_title(f"Rank scatter - {event_label(e)}", fontsize=11)
        ax.legend(loc="upper left", frameon=False, fontsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        note = "Ties share the lowest rank."
        if NO_EFFECT_TOL > 0:
            note += (
                f" Reductions below {NO_EFFECT_TOL:.0%} of baseline count as no effect."
            )
        fig.text(0.5, 0.01, note, ha="center", fontsize=7, color="gray")
        fig.subplots_adjust(left=0.08, right=0.82, bottom=0.10, top=0.94)
        out = analysis_path(BASIN_ID, __file__, f"rank_scatter_{e}_{RANK_METRIC}.png")
        fig.savefig(out, dpi=150, facecolor="white")
        plt.close(fig)
        print(f"Wrote {out.name} to {out.parent}")


def plot_rank_bump(d, baselines, events, measures):
    """Plot each measure's rank at BUMP_SCALE under every metric, approaches x events.

    Crossing lines show where the choice of metric reorders the strategies.
    Tied measures share a rank and are nudged apart by BUMP_NUDGE. Each panel
    notes Kendall's W across the metrics (1 = every metric gives the same order).
    """
    frames = []
    for e in events:
        for ap in BUMP_APPROACHES:
            g = d[(d.event == e) & (d.scale == BUMP_SCALE) & (d.approach == ap)]
            g = g.set_index("short")
            for m in BUMP_METRICS if not g.empty else []:
                r = rank_by_residual((g[m] / baselines.loc[e, m]).dropna())
                frames.append(
                    r.assign(event=e, approach=ap, metric=m)
                    .rename_axis("measure")
                    .reset_index()
                )
    if not frames:
        print(f"\n# no runs at scale {BUMP_SCALE} -- rank bump chart skipped")
        return
    bump = pd.concat(frames, ignore_index=True)
    # nudge tied lines apart in MEASURE_ORDER, so a tie held over several metrics
    # stays parallel
    bump["order"] = bump.measure.map(MEASURE_ORDER.index)
    bump = bump.sort_values("order")
    tie = bump.groupby(["event", "approach", "metric", "rank"])["order"]
    bump["y"] = (
        bump["rank"] + (tie.cumcount() - (tie.transform("size") - 1) / 2) * BUMP_NUDGE
    )

    ranks = bump.pivot_table(
        index=["event", "measure"], columns=["metric", "approach"], values="rank"
    )[BUMP_METRICS]
    print(
        f"\n# Rank per metric at scale {BUMP_SCALE} (1 = lowest residual), "
        "modelled (PRE) vs postprocessed (POST)"
    )
    print(ranks.to_string())
    same = ranks.xs("PRE", axis=1, level=1) == ranks.xs("POST", axis=1, level=1)
    print("\n# Measures whose POST rank equals the PRE rank, per event and metric")
    print(same.groupby("event").sum().astype(int).to_string())

    metric_short = {
        m: textwrap.fill(lbl.removeprefix("residual ").capitalize(), 10)
        for m, lbl, _ in METRIC_INFO
    }
    fig_h = 2.9 * len(BUMP_APPROACHES) + 0.9
    fig, axes = plt.subplots(
        len(BUMP_APPROACHES),
        len(events),
        figsize=(3.4 * len(events), fig_h),
        sharex=True,
        sharey=True,
        squeeze=False,
        facecolor="white",
    )
    x = np.arange(len(BUMP_METRICS))
    for i, (ap, ap_label) in enumerate(BUMP_APPROACHES.items()):
        for j, e in enumerate(events):
            ax = axes[i, j]
            p = bump[(bump.event == e) & (bump.approach == ap)]
            if p.empty:
                ax.set_axis_off()
                continue
            for ms, g in p.groupby("measure"):
                color, marker = MEASURE_STYLE[ms]
                ax.plot(
                    x,
                    g.set_index("metric").reindex(BUMP_METRICS)["y"],
                    color=color,
                    lw=2,
                    marker=marker,
                    ms=7,
                    mec="white",
                    mew=1,
                    zorder=3,
                )
            w = kendalls_w(p.pivot(index="measure", columns="metric", values="val"))
            ax.set_title(f"W = {w:.2f}", loc="right", fontsize=8, color="#52514e")
            if i == 0:
                ax.set_title(event_label(e), loc="left", fontsize=11)
            if j == 0:
                ax.set_ylabel(ap_label, fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels([metric_short[m] for m in BUMP_METRICS], fontsize=9)
            ax.set_ylim(len(measures) + 0.5, 0.5)  # rank 1 on top
            ax.set_yticks(range(1, len(measures) + 1))
            ax.grid(color="#e6e6e6", lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

    handles = [
        Line2D([], [], color=c, marker=mk, lw=2, ms=7, mec="white", mew=1)
        for c, mk in (MEASURE_STYLE[ms] for ms in measures)
    ]
    fig.legend(
        handles,
        [pretty(ms) for ms in measures],
        loc="upper center",
        ncol=len(measures),
        frameon=False,
        fontsize=9,
    )
    note = (
        f"{SCALE_LABELS[BUMP_SCALE]} scale; rank 1 = lowest residual. Ties share the"
        " lowest rank. W = Kendall's coefficient of concordance across metrics."
    )
    if NO_EFFECT_TOL > 0:
        note += f" Reductions below {NO_EFFECT_TOL:.0%} of baseline count as no effect."
    fig.text(
        0.5,
        0.005,
        textwrap.fill(note, 150),
        ha="center",
        va="bottom",
        fontsize=7,
        color="gray",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1 - 0.45 / fig_h])
    out = analysis_path(BASIN_ID, __file__, f"rank_bump_metrics_{BUMP_SCALE}.png")
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out.name} to {out.parent}")


def build_parser():
    """Return the command-line parser (no arguments besides --help)."""
    return script_parser(__doc__)


def main():
    """Run all results figures for basin 2433835."""
    build_parser().parse_args()

    # results_dir is a site setting; see config/site.example.yml.
    results_root = site_path("results_dir")
    baselines, d = load_comparison(
        results_root / BASIN_ID / "runs" / "metrics_comparison.csv"
    )

    print(
        "\n# Rows sharing an identical metric vector (i.e. the approach cannot "
        "distinguish two designs):"
    )
    for (e, a), g in d.groupby(["event", "approach"]):
        n_dup = g.duplicated(METRICS, keep=False).sum()
        print(f"  {e:20s} {a:5s}: {n_dup:2d} / {len(g)}")

    events = sorted(d.event.unique(), key=EVENT_ORDER.index)
    measures = sorted(d.short.unique(), key=MEASURE_ORDER.index)

    plot_scale_response(d, baselines, events, measures)
    plot_return_period_comparison(results_root)

    d["resid"] = d[RANK_METRIC] / d.event.map(baselines[RANK_METRIC])  # 1 = no change
    plot_rank_scatter(d, events)
    plot_rank_bump(d, baselines, events, measures)


if __name__ == "__main__":
    main()
