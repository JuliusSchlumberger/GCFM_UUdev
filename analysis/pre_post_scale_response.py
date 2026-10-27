"""
POST- vs PRE-processing agreement analysis for flood adaptation measures.

Step 1: split `strategy` into measure name + scale suffix
Step 2: convert absolute metrics to % risk reduction vs the matching event baseline
Step 3: agreement diagnostics (figures in results section)

"""

import os
import textwrap
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Machine-specific path, read from the GCFM_RESULTS_DIR environment variable.
# Set it once in PowerShell, then restart your terminal (see
# CONTRIBUTING.md "Local machine paths"):
#   [Environment]::SetEnvironmentVariable("GCFM_RESULTS_DIR", "D:\your\results\path", "User")
OUT_DIR = Path(os.environ["GCFM_RESULTS_DIR"]) / "2433835/runs"
CSV = OUT_DIR / "metrics_comparison.csv"
METRICS = ["flooded_area_km2", "urban_exposed_km2", "mean_depth_m", "volume_m3"]
SCALES = ["04", "09", "1"]  # raw strategy-name suffix, low -> high implementation scale

# ----------------------------------------------------------------- 1. parse
raw = pd.read_csv(
    CSV, sep=None, engine="python"
)  # "," from the pipeline, ";" once re-saved in Excel
baselines = raw[raw.method == "baseline"].set_index("scenario")

d = raw[raw.method != "baseline"].copy()
d["event"] = d["scenario"]
d["approach"] = d["method"].str.upper()  # PRE | POST

d["scale"] = d.strategy.str.extract(r"_(04|09|1)$")[0]
d["measure"] = d.strategy.str.replace(r"_(04|09|1)$", "", regex=True)
d["short"] = d["measure"]
d["scale"] = pd.Categorical(d.scale, SCALES, ordered=True)
assert d["scale"].notna().all(), (
    "found a strategy name without a recognised _04/_09/_1 suffix"
)

# a fully-protected run has flooded_area == 0 and NaN depth; depth is 0 by definition
d[["mean_depth_m", "max_depth_m"]] = d[["mean_depth_m", "max_depth_m"]].fillna(0)

# # ------------------------------------------- 2. % reduction vs event baseline
for m in METRICS:
    b = d.event.map(baselines[m])
    d["red_" + m] = 100 * (b - d[m]) / b
# d.to_csv(OUT_DIR / "tidy_scenarios.csv", index=False)

wide = d.pivot_table(
    index=["event", "short", "scale"],
    columns="approach",
    values=["red_" + m for m in METRICS],
    observed=True,
)
for m in METRICS:
    wide[("err_" + m, "")] = wide[("red_" + m, "POST")] - wide[("red_" + m, "PRE")]
# wide.to_csv(OUT_DIR / "post_vs_pre_wide.csv")

# ------------------------------------------------------ 3a. does POST resolve scale?
print(
    "\n# Rows sharing an identical metric vector (i.e. the approach cannot "
    "distinguish two designs):"
)
cols = ["flooded_area_km2", "urban_exposed_km2", "mean_depth_m", "volume_m3"]
for (e, a), g in d.groupby(["event", "approach"]):
    print(f"  {e:14s} {a:5s}: {g.duplicated(cols, keep=False).sum():2d} / {len(g)}")

# # ------------------------------------------------------ 3b. Spearman per event
# print("\n# Spearman rho, POST vs PRE ranking of all measure-scale combinations")
# rows = []
# for e, g in d.groupby("event"):
#     p = g.pivot_table(
#         index=["measure", "scale"],
#         columns="approach",
#         values=["red_" + m for m in METRICS],
#         observed=True,
#     )
#     for m in METRICS:
#         r, pv = spearmanr(p[("red_" + m, "POST")], p[("red_" + m, "PRE")])
#         rows.append(dict(event=e, metric=m, rho=round(r, 2), p=round(pv, 4)))
# sp = pd.DataFrame(rows)
# print(sp.pivot(index="metric", columns="event", values="rho").to_string())
# sp.to_csv(OUT_DIR / "spearman_by_event_metric.csv", index=False)

# # ------------------------------------------- 3c. binary agreement (tune threshold)
# THRESH = 25.0  # % reduction in urban exposure counted as "effective"
# u = wide["red_urban_exposed_km2"].reset_index()
# u["POST_eff"], u["PRE_eff"] = u.POST > THRESH, u.PRE > THRESH
# print(f"\n# Confusion matrix, 'effective' = >{THRESH:.0f}% reduction in urban exposure")
# print(
#     pd.crosstab(u.PRE_eff, u.POST_eff, rownames=["PRE (benchmark)"], colnames=["POST"])
# )
# u["error"] = u.POST - u.PRE
# print("\n# Disagreements, sorted by magnitude")
# print(
#     u[u.POST_eff != u.PRE_eff]
#     .reindex(u.error.abs().sort_values(ascending=False).index)
#     .dropna(subset=["error"])[["event", "short", "scale", "PRE", "POST", "error"]]
#     .round(1)
#     .to_string(index=False)
# )

# ------------------------------------------------------------------ 4. figures
# scale-response figures: POST-PRE differences in residual ratio smaller than this
# are left unshaded
SHADE_TOL = 0.01
SHADE_ALPHA = 0.13  # opacity of that shading (and its legend patches)

EVENT_ORDER = ["coast_100_03SLR", "river_100_03SLR", "compound_100_03SLR"]
EVENT_LABELS = {
    "coast_100_03SLR": "Coastal flood",
    "river_100_03SLR": "River flood",
    "compound_100_03SLR": "Compound flood",
}
events = sorted(d.event.unique(), key=EVENT_ORDER.index)
MEASURE_ORDER = [
    "advance",
    "protect_closed",
    "grey_protect_open",
    "NbS_protect_open",
    "accommodate",
    "retreat",
]
measures = sorted(d.short.unique(), key=MEASURE_ORDER.index)

STYLE = {
    "PRE": dict(fmt="-o", color="#534eb8", label="Modelled (re-run)"),
    "POST": dict(fmt="--s", color="#38b394", label="Postprocessed (not re-run)"),
}


def label(s):
    return s.replace("_", " ").title()


def event_label(e):
    """Figure title for a flood event: EVENT_LABELS, else the tidied scenario name."""
    return EVENT_LABELS.get(e, label(e))


METRIC_INFO = [
    ("flooded_area_km2", "residual flood extent", (0, 2), [0, 0.25, 0.5, 1, 1.5, 2]),
    (
        "urban_exposed_km2",
        "residual urban exposure",
        (0, 2),
        [0, 0.25, 0.5, 1, 2],
    ),
    ("volume_m3", "residual flood volume", (0, 3), [0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3]),
    ("mean_depth_m", "residual mean flood depth", (0, 2), [0, 0.25, 0.5, 1, 1.5, 2]),
]

PLOT_SCALES = ["0"] + SCALES  # prepend the no-adaptation baseline (ratio == 1)
SCALE_LABELS = {
    "0": "No adaptation",
    "04": "Highly ineffective",
    "09": "Marginally ineffective",
    "1": "Full potential",
}
X_LABELS = [SCALE_LABELS[s] for s in PLOT_SCALES]

# y-axis: linear from 0 to 1 (the reduction range), log-compressed above 1 so the
# residual-risk band (> 1) stays a thin strip. ABOVE_ONE_SQUEEZE sets how thin:
# each factor of 10 above 1 takes this many "units" of height (0-1 is one unit).
ABOVE_ONE_SQUEEZE = 0.4


def _y_forward(y):
    y = np.asarray(y, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(y <= 1, y, 1 + ABOVE_ONE_SQUEEZE * np.log10(y))


def _y_inverse(x):
    x = np.asarray(x, dtype=float)
    with np.errstate(over="ignore"):
        return np.where(x <= 1, x, 10 ** ((x - 1) / ABOVE_ONE_SQUEEZE))


def shade_divergence(ax, modelled, post, n=200):
    """
    Fill between the PRE (modelled) and POST curves: POST colour where POST
    is above PRE (POST underestimates effectiveness), PRE colour where it is
    below. Interpolates each segment between neighbouring categorical x
    positions in the _y_forward space, so the fill follows the plotted lines,
    and leaves segments with a NaN at either end unshaded.
    """
    pre_t = _y_forward(modelled)
    post_t = _y_forward(post)
    xp = np.arange(len(pre_t))
    # one grid over all segments, so each colour is one polygon (no seams at the
    # points); np.interp gives NaN on any segment with a NaN end, left unshaded
    x = np.linspace(0, xp[-1], n * xp[-1] + 1)
    pre_y = _y_inverse(np.interp(x, xp, pre_t))
    post_y = _y_inverse(np.interp(x, xp, post_t))
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


for METRIC, LABEL, YLIM, YTICKS in METRIC_INFO:
    BL = baselines[METRIC].to_dict()
    d["ratio"] = d[METRIC] / d.event.map(BL)

    # blank strip on the right (inches) kept free for a small added graphic; on A4
    # this tall figure is limited by page height, so the extra width doesn't shrink
    # it further. 0 = no strip
    SIDE_W = 2.0
    LEGEND_H = 0.7  # inches above the column titles for the one-row legend
    fig_w = 3 * len(events) + SIDE_W
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
            A = axes[i, j]
            g = d[(d.event == e) & (d.short == ms)]

            curves = {}
            for ap, s in STYLE.items():
                gg = g[g.approach == ap].set_index("scale").reindex(SCALES)
                ratios = [1.0] + gg["ratio"].tolist()  # scale 0 = no adaptation
                A.plot(
                    X_LABELS,
                    ratios,
                    s["fmt"],
                    color=s["color"],
                    label=s["label"],
                    zorder=3,
                )
                curves[ap] = ratios
            shade_divergence(A, curves["PRE"], curves["POST"])

            A.set_yscale("function", functions=(_y_forward, _y_inverse))
            A.set_ylim(*YLIM)
            # A.axhline(1, c="crimson", lw=0.8)
            A.axhspan(1, YLIM[1], color="crimson", alpha=0.07)
            # only 1 and the top inside the thin band above 1, so its labels don't collide
            ticks = [0, 0.25, 0.5, 0.75, 1, YLIM[1]]
            A.set_yticks(ticks)
            A.set_yticklabels([f"{t:g}" for t in ticks], fontsize=11)
            A.tick_params(labelsize=11, labelrotation=45)
            # end each rotated label at its tick, so neighbouring labels don't overlap
            plt.setp(A.get_xticklabels(), ha="right", rotation_mode="anchor")
            if i == 0:
                A.set_title(event_label(e), fontsize=13)
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
    for ap, text in (("POST", "underestimates"), ("PRE", "overestimates")):
        handles.append(
            Patch(facecolor=STYLE[ap]["color"], alpha=SHADE_ALPHA, edgecolor="none")
        )
        labels.append(f"Post-processed {text} effectiveness")
    handles.append(Patch(facecolor="crimson", alpha=0.1, edgecolor="none"))
    labels.append("Residual risk")
    # one row across the top of the panels, long labels wrapped to keep it narrow
    right = 1 - SIDE_W / fig_w
    top = 1 - LEGEND_H / fig_h
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
    fig.savefig(OUT_DIR / f"fig_scale_response_{METRIC}.png", dpi=150)
    plt.close(fig)

print(
    f"\nWrote outputs to {OUT_DIR}: tidy_scenarios.csv, post_vs_pre_wide.csv, "
    "spearman_by_event_metric.csv, fig_scale_response_<metric>.png"
)

# Combined plot of hydrographs -- same two panels as fig_return_period_curves.png
# (study_area_figures.py: river discharge RP curve | coastal water level RP curves) but
# with every basin in COMPARE_BASINS overlaid on the same axes, one colour per basin, so
# deltas can be compared directly. Opens each basin's own river_forcing.nc/
# surge_forcing.nc.
COMPARE_BASINS = ["2433835", "3279946"]  # add/remove basin_ids to compare others
BASIN_LABELS = {"3279946": "Chao Phraya (3279946)"}  # optional override; else basin_id
BASIN_COLORS = ["#2a9d8f", "#e76f51", "#8338ec", "#ffb703"]  # cycled if > 4 basins
RESULTS_ROOT = OUT_DIR.parent.parent  # .../results (parent of every {basin_id}/ dir)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")

for basin_id, color in zip(COMPARE_BASINS, BASIN_COLORS * len(COMPARE_BASINS)):
    forcing_dir = RESULTS_ROOT / basin_id / "preprocessing_inputs" / "forcing"
    basin_label = BASIN_LABELS.get(basin_id, basin_id)

    with xr.open_dataset(forcing_dir / "river_forcing.nc", decode_times=False) as rds:
        active = rds["has_glofas"].values.astype(bool)
        rp_table = rds["discharge_rp_table"].values[active]
        i_main = np.nanargmax(rp_table[:, -1])
        ax1.plot(
            rds["return_period"].values,
            rp_table[i_main],
            color=color,
            label=basin_label,
        )

    with xr.open_dataset(forcing_dir / "surge_forcing.nc", decode_times=False) as sds:
        table_rp = sds["table_rp"].values
        storm_tide_table = sds["storm_tide_rp_table"].values
        for i in range(storm_tide_table.shape[0]):
            ax2.plot(
                table_rp,
                storm_tide_table[i],
                color=color,
                lw=1,
                alpha=0.6,
                marker="o",
                markersize=2,
            )
    # one legend-only proxy line per basin on ax2 (station curves above carry no label,
    # so the legend doesn't grow to one entry per station)
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
out_compare = RESULTS_ROOT / (
    "fig_return_period_curves_comparison_" + "_vs_".join(COMPARE_BASINS) + ".png"
)
fig.savefig(out_compare, dpi=150, facecolor="white")
plt.close(fig)
print(f"Wrote {out_compare}")


# Ranking -----------------------------------------------------------------------------------------------------------------------
# Rank the measures by residual urban exposure (1 = lowest) under the modelled
# (PRE) and postprocessed (POST) approach. Everything is computed from the numbers in
# metrics_comparison.csv: <metric> of the run / <metric> of the event's baseline row.
RANK_METRIC = "urban_exposed_km2"
TIE_DECIMALS = 2  # residual ratios equal at this precision share a rank ("=")
# a reduction smaller than this fraction of the baseline (incl. an increase) counts as
# "no effect", so all such measures tie for last; 0 = rank on the exact ratios
NO_EFFECT_TOL = 0.05
SCALE_MARKER = {"04": ("s", "#d62728"), "09": ("^", "#f0a30a"), "1": ("o", "#2ca02c")}

d["resid"] = d[RANK_METRIC] / d.event.map(baselines[RANK_METRIC])  # 1 = no change


def pretty(s):
    s = s.replace("_", " ")
    return s[0].upper() + s[1:]  # keeps "NbS" (str.capitalize() would give "Nbs")


def rank_by_residual(resid):
    """Best-to-worst row order plus rank; ties share the lowest rank."""
    effective = resid.where(resid < 1 - NO_EFFECT_TOL, 1.0)
    out = pd.DataFrame({"resid": resid, "val": effective.round(TIE_DECIMALS)})
    out = out.sort_values(["val", "resid"])
    out["row"] = np.arange(1, len(out) + 1)
    out["rank"] = out["val"].rank(method="min").astype(int)
    out["tied"] = out["val"].duplicated(keep=False)
    return out


# Rank scatter ------------------------------------------------------------------------------------------------------------------
# Modelled rank (x) vs postprocessed rank (y) of every measure, one dot per measure and
# scale, coloured by scale. Ranks are within each scale, using RANK_METRIC /
# TIE_DECIMALS / NO_EFFECT_TOL from the ranking settings above. Dots on the diagonal keep
# their rank; the shaded band marks +-SCATTER_BAND ranks.
SCATTER_EVENTS = ["coast_100_03SLR"]  # any of coast_100, river_500, compound_100c_500r
SCATTER_BAND = 1
SCATTER_DOT_SIZES = (40, 90, 150)  # one per scale in SCALES: small (04) -> big (1)

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
            pd.DataFrame({"scale": sc, "PRE_rank": L["rank"], "POST_rank": R["rank"]})
            .rename_axis("measure")
            .reset_index()
        )
    if not frames:
        print(f"\n# {e}: no measure has both PRE and POST -- rank scatter skipped")
        continue
    pts = pd.concat(frames, ignore_index=True)
    pts["diff"] = pts["POST_rank"] - pts["PRE_rank"]

    n_out = int((pts["diff"].abs() > SCATTER_BAND).sum())
    print(f"\n# {e}: rank scatter ({RANK_METRIC}, ranks within each scale, 1 = lowest)")
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

    # one label per spot (tied / repeated measures are stacked), on the left for dots on
    # or above the diagonal and on the right for dots below it (or hugging the y-axis),
    # to keep neighbours and the axis apart
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
        note += f" Reductions below {NO_EFFECT_TOL:.0%} of baseline count as no effect."
    fig.text(0.5, 0.01, note, ha="center", fontsize=7, color="gray")
    fig.subplots_adjust(left=0.08, right=0.82, bottom=0.10, top=0.94)
    out = OUT_DIR / f"fig_rank_scatter_{e}_{RANK_METRIC}.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out.name} to {OUT_DIR}")


# Metric bump chart -------------------------------------------------------------------------------------------------------------
# Rank of every measure at BUMP_SCALE under each risk metric, one line per measure, so
# crossing lines show where the choice of metric reorders the strategies. Rows are the
# modelled (PRE) and postprocessed (POST) approach, columns events; ranks are within each
# panel, using TIE_DECIMALS / NO_EFFECT_TOL from the ranking settings above. Tied measures
# share a rank and are nudged apart by BUMP_NUDGE. Each panel notes Kendall's W across the
# metrics (1 = every metric gives the same order).
BUMP_METRICS = ["flooded_area_km2", "urban_exposed_km2", "volume_m3", "mean_depth_m"]
BUMP_SCALE = "1"  # full potential
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
METRIC_SHORT = {
    m: textwrap.fill(lbl.removeprefix("residual ").capitalize(), 10)
    for m, lbl, *_ in METRIC_INFO
}


def kendalls_w(vals):
    """Tie-corrected Kendall's W of a measures x metrics table of ranking values."""
    vals = vals.dropna()
    n, m = vals.shape
    r = vals.rank(method="average")
    s = ((r.sum(axis=1) - m * (n + 1) / 2) ** 2).sum()
    t = sum((c**3 - c).sum() for c in (vals[col].value_counts() for col in vals))
    denom = m**2 * (n**3 - n) - m * t
    return 12 * s / denom if denom > 0 else np.nan


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
bump = pd.concat(frames, ignore_index=True)
# nudge tied lines apart in MEASURE_ORDER, so a tie held over several metrics stays parallel
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

fig_w, fig_h = 3.4 * len(events), 2.9 * len(BUMP_APPROACHES) + 0.9
fig, axes = plt.subplots(
    len(BUMP_APPROACHES),
    len(events),
    figsize=(fig_w, fig_h),
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
        ax.set_xticklabels([METRIC_SHORT[m] for m in BUMP_METRICS], fontsize=9)
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
    f"{SCALE_LABELS[BUMP_SCALE]} scale; rank 1 = lowest residual. Ties share the lowest"
    " rank. W = Kendall's coefficient of concordance across metrics."
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
out = OUT_DIR / f"fig_rank_bump_metrics_{BUMP_SCALE}.png"
fig.savefig(out, dpi=150, facecolor="white")
plt.close(fig)
print(f"Wrote {out.name} to {OUT_DIR}")
