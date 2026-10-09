"""
POST- vs PRE-processing agreement analysis for flood adaptation measures.

Step 1: split `strategy` into measure name + scale suffix
Step 2: convert absolute metrics to % risk reduction vs the matching event baseline
Step 3: agreement diagnostics (figures in results section)

"""

from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
from hydromt_sfincs import SfincsModel
import matplotlib
import matplotlib.dates as mdates

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT_DIR = Path(r"D:\GCFM_UU\results\2433835_V2_paper\runs")
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
EVENT_ORDER = ["coast_100", "river_500", "compound_100c_500r"]
events = sorted(d.event.unique(), key=EVENT_ORDER.index)
# case-insensitive: plain sorted() puts "NbS_..." before "grey_..." (uppercase
# N sorts before lowercase g in ASCII), not the intended alphabetical order
measures = sorted(d.short.unique(), key=str.lower)

STYLE = {
    "PRE": dict(fmt="-o", color="#0072B2", label="Modelled (re-run)"),
    "POST": dict(fmt="--s", color="#E69F00", label="Postprocessed (not re-run)"),
}


def label(s):
    return s.replace("_", " ").title()


METRIC_INFO = [
    ("flooded_area_km2", "residual flood extent", (0, 2), [0, 0.25, 0.5, 1, 1.5, 2]),
    (
        "urban_exposed_km2",
        "residual urban exposure",
        (0, 3),
        [0, 0.25, 0.5, 1, 2, 3],
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

for METRIC, LABEL, YLIM, YTICKS in METRIC_INFO:
    BL = baselines[METRIC].to_dict()
    d["ratio"] = d[METRIC] / d.event.map(BL)

    fig, axes = plt.subplots(
        len(measures),
        len(events),
        figsize=(3 * len(events), 2.7 * len(measures)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for i, ms in enumerate(measures):
        for j, e in enumerate(events):
            A = axes[i, j]
            g = d[(d.event == e) & (d.short == ms)]

            for ap, s in STYLE.items():
                gg = g[g.approach == ap].set_index("scale").reindex(SCALES)
                ratios = [1.0] + gg["ratio"].tolist()  # scale 0 = no adaptation
                A.plot(X_LABELS, ratios, s["fmt"], color=s["color"], label=s["label"])

            A.set_yscale("symlog", linthresh=0.05)
            A.set_ylim(*YLIM)
            # A.axhline(1, c="crimson", lw=0.8)
            A.axhspan(1, YLIM[1], color="crimson", alpha=0.07)
            A.set_yticks(YTICKS)
            A.set_yticklabels([f"{t:g}" for t in YTICKS], fontsize=7)
            A.tick_params(labelsize=7, labelrotation=45)
            if i == 0:
                A.set_title(label(e), fontsize=9)
            if j == 0:
                A.set_ylabel(LABEL, fontsize=8)
                A.annotate(
                    label(ms),
                    xy=(-0.42, 0.5),
                    xycoords="axes fraction",
                    fontsize=11,
                    rotation=90,
                    va="center",
                    ha="center",
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Patch(facecolor="crimson", alpha=0.1, edgecolor="none"))
    labels.append("Residual risk")
    axes[0, 0].legend(handles, labels, fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_DIR / f"fig_scale_response_{METRIC}.png", dpi=150)

print(
    f"\nWrote outputs to {OUT_DIR}: tidy_scenarios.csv, post_vs_pre_wide.csv, "
    "spearman_by_event_metric.csv, fig_scale_response_<metric>.png"
)

# Time series plots
BASIN_ROOT = OUT_DIR.parent  # .../results/<basin_id>
FORCING_DIR = BASIN_ROOT / "preprocessing_inputs" / "forcing"
# decode_times=False: "time"'s units ("hours since simulation start") is a
# relative offset, not a real calendar date -- xarray's CF-time decoder
# chokes trying to parse "simulation start" as a reference date.
river_ds = xr.open_dataset(FORCING_DIR / "river_forcing.nc", decode_times=False)
surge_ds = xr.open_dataset(FORCING_DIR / "surge_forcing.nc", decode_times=False)

# the single most significant river crossing (largest high-RP discharge),
# same selection basis 07_get_boundary_forcings.py's own diagnostic plot uses
active = river_ds["has_glofas"].values.astype(bool)
rp_table = river_ds["discharge_rp_table"].values[active]
i_main = np.nanargmax(rp_table[:, -1])

# plot 1: all return periods
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="white")

ax1.plot(river_ds["return_period"].values, rp_table[i_main], color="#1b4965")
ax1.set_xscale("log")
ax1.set_xlabel("Return period (years)")
ax1.set_ylabel("Discharge (m3/s)")
ax1.set_title("River discharge")

table_rp = surge_ds["table_rp"].values
storm_tide_table = surge_ds["storm_tide_rp_table"].values
lons, lats = surge_ds["longitude"].values, surge_ds["latitude"].values
for i in range(storm_tide_table.shape[0]):
    ax2.plot(
        table_rp,
        storm_tide_table[i],
        marker="o",
        markersize=3,
        label=f"Station {i} ({lats[i]:.2f}N, {lons[i]:.2f}E)",
    )
ax2.set_xscale("log")
ax2.set_xlabel("Return period (years)")
ax2.set_ylabel("Water level (m)")
ax2.set_title("Coastal water level")
# ax2.legend(fontsize=7)

for ax in (ax1, ax2):
    ax.grid(alpha=0.3)

# fig.suptitle("Flood hazard return-period curves")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_return_period_curves.png", dpi=150, facecolor="white")
plt.close(fig)


# -------------------------------------------------------------------------------
#  plot 2: forced hydrograph -- the REAL, actually-built forcing for one
# scenario's own SFINCS run, read straight off its sfincs.bzs/sfincs.dis via
# hydromt_sfincs (real calendar time, RP/protection-floor/SLR/multiplier all
# already resolved). NOT river_forcing.nc's/surge_forcing.nc's own stored
# 'discharge'/'water_level' -- those are a basin-level, scenario-independent
# preview at a fixed diagnostic RP, never what a specific scenario like
# coast_500 actually ran with (same fix 13_build_sfincs.py's own 05_forcing.png
# already applies -- see its comment above the forcing-plot block).
# 2x3 grid: one column per scenario, water level on top / discharge on
# bottom, y-axis shared within each row so magnitudes are directly
# comparable across scenarios (coast_500's surge peak vs. compound_500's,
# river_500's discharge peak vs. compound_500's, etc.)
SCENARIOS = ["coast_100", "river_500", "compound_100c_500r"]
SCENARIO_COLOR = "#14b5dd"
# if these scenarios' own slr_m (config/scenarios.yml, per-scenario) is 0.0,
# the plotted water level IS the actual built forcing, with no SLR -- shown
# as a flat reference line AT y=SLR_M (an illustrative SLR magnitude, not a
# baseline+SLR shift) on the coastal scenarios only, so the flood peak's
# height can be read directly against it (river_500 has near-baseline surge
# to begin with, so this reference isn't the interesting comparison there).
SLR_M = 0.5
SLR_SCENARIOS = {"coast_100"}  # , "compound_100c_500r"}

fig, axes = plt.subplots(
    2,
    len(SCENARIOS),
    figsize=(3.2 * len(SCENARIOS), 5.5),
    sharex=True,
    sharey="row",
    facecolor="white",
)

for j, scen in enumerate(SCENARIOS):
    scenario_root = BASIN_ROOT / "runs" / scen / "sfincs"
    mod = SfincsModel(root=str(scenario_root), mode="r")

    wl_comp = mod.get_component("water_level")
    wl_comp.read()
    wl = wl_comp.data  # dims (time, index), var 'bzs'
    wl_now = wl["bzs"].mean(dim="index").values
    axes[0, j].plot(wl.time.values, wl_now, color=SCENARIO_COLOR, lw=1.8)

    if scen in SLR_SCENARIOS:
        axes[0, j].axhline(
            SLR_M,
            color=SCENARIO_COLOR,
            lw=1.5,
            linestyle=":",
            label=f"{SLR_M:g} m SLR",
        )
        axes[0, j].legend(fontsize=7, frameon=False, loc="upper left")

    dis_comp = mod.get_component("discharge_points")
    dis_comp.read()
    dis = dis_comp.data  # dims (time, index), var 'dis'
    axes[1, j].plot(
        dis.time.values, dis["dis"].mean(dim="index").values, color="#2411ce", lw=1.8
    )

    for i in (0, 1):
        axes[i, j].grid(alpha=0.3)
        axes[i, j].spines[["top", "right"]].set_visible(False)

    axes[0, j].set_title(scen)

axes[0, 0].set_ylabel("Water level (m)")
axes[1, 0].set_ylabel("Discharge (m³/s)")

# sharex=True only links the axis range, not tick-label visibility, and
# AutoDateLocator's default tick density (fine in 13_build_sfincs.py's own
# single wide 10" figure) is too cluttered at this grid's ~3" column width --
# force 12-hourly ticks instead and rotate labels to keep them legible.

locator = mdates.HourLocator(byhour=[0, 12])
formatter = mdates.ConciseDateFormatter(locator)
for ax in axes.flat:
    ax.label_outer()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", labelrotation=45)
    # "2000" is just this pipeline's dummy simulation reference year, not a
    # meaningful date -- drop ConciseDateFormatter's repeated offset text
    ax.xaxis.get_offset_text().set_visible(False)

fig.suptitle("Forcing hydrographs")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_forced_hydrograph.png", dpi=200, facecolor="white")
plt.close(fig)

river_ds.close()
surge_ds.close()
print(f"Wrote fig_return_period_curves.png, fig_forced_hydrograph.png to {OUT_DIR}")

# Combined plot of hydrographs -- same two panels as fig_return_period_curves.png
# (river discharge RP curve | coastal water level RP curves) but with every basin in
# COMPARE_BASINS overlaid on the same axes, one colour per basin, so deltas can be
# compared directly. Independent of OUT_DIR/the single-basin run above -- reopens each
# basin's own river_forcing.nc/surge_forcing.nc fresh (those were already closed above).
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


# Ranking plots -----------------------------------------------------------------------------------------------------------------
# Rank the measures by residual urban exposure (1 = lowest) under the modelled
# (PRE) and postprocessed (POST) approach -- one figure per event, one line per
# measure linking its two ranks. Everything is computed from the numbers in
# metrics_comparison.csv: <metric> of the run / <metric> of the event's baseline row.
RANK_EVENTS = ["coast_100"]  # any of coast_100, river_500, compound_100c_500r
RANK_METRIC = "urban_exposed_km2"
RANK_SCALE = "09"  # "1" = full potential; "09" / "04" also work
TIE_DECIMALS = 2  # residual ratios equal at this precision share a rank ("=")
# a reduction smaller than this fraction of the baseline (incl. an increase) counts as
# "no effect", so all such measures tie for last; 0 = rank on the exact ratios
NO_EFFECT_TOL = 0.05
SCALE_MARKER = {"04": ("s", "#d62728"), "09": ("^", "#f0a30a"), "1": ("o", "#2ca02c")}
MOVED_STYLE = dict(color="#1f2d3d", lw=2.2)  # rank changes by >= 2
STAY_STYLE = dict(color="#b5d4ae", lw=1.6)  # rank changes by <= 1

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


unknown = [e for e in RANK_EVENTS if e not in events]
if unknown:
    raise ValueError(f"RANK_EVENTS {unknown} not in the results table; have {events}")

marker, mcolor = SCALE_MARKER[RANK_SCALE]
for e in RANK_EVENTS:
    piv = d[(d.event == e) & (d.scale == RANK_SCALE)].pivot_table(
        index="short", columns="approach", values="resid", observed=True
    )
    if not {"PRE", "POST"} <= set(piv.columns) or piv.dropna().empty:
        print(
            f"\n# {e}: no measure has both PRE and POST at scale {RANK_SCALE} -- skipped"
        )
        continue
    w = piv.dropna()
    L, R = rank_by_residual(w["PRE"]), rank_by_residual(w["POST"])

    tab = pd.DataFrame(
        {
            "PRE": L["resid"],
            "PRE_rank": L["rank"],
            "POST": R["resid"],
            "POST_rank": R["rank"],
        }
    ).loc[L.index]
    tab["shift"] = tab["POST_rank"] - tab["PRE_rank"]
    print(
        f"\n# {e}: {RANK_METRIC} / baseline ({baselines.at[e, RANK_METRIC]:.2f}), "
        f"scale {RANK_SCALE}; rank 1 = lowest, rows ordered by PRE"
    )
    print(tab.round(3).to_string())

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(measures) + 1.8), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(len(measures) + 0.5, 0.5)  # rank 1 at the top
    ax.axis("off")
    ax.set_title(label(e), fontsize=11, pad=26)
    ax.text(
        0,
        1,
        "Modelled (reference)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=STYLE["PRE"]["color"],
        fontweight="bold",
    )
    ax.text(
        1,
        1,
        "Postprocessed",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=STYLE["POST"]["color"],
        fontweight="bold",
    )
    ax.plot([0, 0.40], [0.5, 0.5], color=STYLE["PRE"]["color"], lw=1.5, clip_on=False)
    ax.plot([0.60, 1], [0.5, 0.5], color=STYLE["POST"]["color"], lw=1.5, clip_on=False)

    for m in w.index:
        yl, yr = L.at[m, "row"], R.at[m, "row"]
        moved = abs(L.at[m, "rank"] - R.at[m, "rank"]) >= 2
        ax.plot(
            [0.42, 0.58],
            [yl, yr],
            zorder=2 if moved else 1,
            **(MOVED_STYLE if moved else STAY_STYLE),
        )

        rank_l = ("=" if L.at[m, "tied"] else "") + str(L.at[m, "rank"])
        rank_r = ("=" if R.at[m, "tied"] else "") + str(R.at[m, "rank"])
        ax.text(0.00, yl, rank_l, ha="left", va="center", fontsize=9, color="gray")
        ax.plot(0.055, yl, marker=marker, color=mcolor, ms=6, ls="none", clip_on=False)
        ax.text(0.08, yl, pretty(m), ha="left", va="center", fontsize=10)
        ax.text(
            0.40,
            yl,
            f"{L.at[m, 'resid']:.2f}",
            ha="right",
            va="center",
            fontsize=9,
            color="gray",
        )

        ax.text(
            0.60,
            yr,
            f"{R.at[m, 'resid']:.2f}",
            ha="left",
            va="center",
            fontsize=9,
            color="gray",
        )
        ax.text(0.66, yr, rank_r, ha="left", va="center", fontsize=9, color="gray")
        ax.plot(0.735, yr, marker=marker, color=mcolor, ms=6, ls="none", clip_on=False)
        ax.text(0.76, yr, pretty(m), ha="left", va="center", fontsize=10)

    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                marker=marker,
                color=mcolor,
                ls="none",
                ms=6,
                label=SCALE_LABELS[RANK_SCALE],
            ),
            Line2D([], [], label="Moved 2 or more ranks", **MOVED_STYLE),
            Line2D([], [], label="Within ±1 rank", **STAY_STYLE),
            Line2D([], [], ls="none", label="= tied rank"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    if NO_EFFECT_TOL > 0:
        fig.text(
            0.5,
            0.01,
            f"Reductions below {NO_EFFECT_TOL:.0%} of baseline count as "
            "no effect and share a rank.",
            ha="center",
            fontsize=7,
            color="gray",
        )
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    out = OUT_DIR / f"fig_rank_comparison_{e}_{RANK_METRIC}_scale{RANK_SCALE}.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out.name} to {OUT_DIR}")


# Rank scatter ------------------------------------------------------------------------------------------------------------------
# Modelled rank (x) vs postprocessed rank (y) of every measure, one dot per measure and
# scale, coloured by scale. Ranks are within each scale, exactly as in the slope chart
# above (same RANK_METRIC / TIE_DECIMALS / NO_EFFECT_TOL). Dots on the diagonal keep
# their rank; the shaded band marks +-SCATTER_BAND ranks.
SCATTER_EVENTS = ["coast_100"]  # any of coast_100, river_500, compound_100c_500r
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
    ax.set_title(f"Rank scatter - {label(e)}", fontsize=11)
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
