"""
POST- vs PRE-processing agreement analysis for flood adaptation measures.

Step 1: split `strategy` into measure name + scale suffix
Step 2: convert absolute metrics to % risk reduction vs the matching event baseline
Step 3: agreement diagnostics (figures in results section)

"""

from pathlib import Path
import numpy as np
import xarray as xr
from hydromt_sfincs import SfincsModel
import matplotlib
import matplotlib.dates as mdates

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(r"D:\GCFM_UU\results\2433835\runs")
CSV = OUT_DIR / "metrics_comparison.csv"
METRICS = ["flooded_area_km2", "urban_exposed_km2", "mean_depth_m", "volume_m3"]
SCALES = ["04", "09", "1"]  # raw strategy-name suffix, low -> high implementation scale

# # ----------------------------------------------------------------- 1. parse
# raw = pd.read_csv(CSV)
# baselines = raw[raw.method == "baseline"].set_index("scenario")

# d = raw[raw.method != "baseline"].copy()
# d["event"] = d["scenario"]
# d["approach"] = d["method"].str.upper()  # PRE | POST

# d["scale"] = d.strategy.str.extract(r"_(04|09|1)$")[0]
# d["measure"] = d.strategy.str.replace(r"_(04|09|1)$", "", regex=True)
# d["short"] = d["measure"]
# d["scale"] = pd.Categorical(d.scale, SCALES, ordered=True)
# assert d["scale"].notna().all(), (
#     "found a strategy name without a recognised _04/_09/_1 suffix"
# )

# # a fully-protected run has flooded_area == 0 and NaN depth; depth is 0 by definition
# d[["mean_depth_m", "max_depth_m"]] = d[["mean_depth_m", "max_depth_m"]].fillna(0)

# # # ------------------------------------------- 2. % reduction vs event baseline
# for m in METRICS:
#     b = d.event.map(baselines[m])
#     d["red_" + m] = 100 * (b - d[m]) / b
# # d.to_csv(OUT_DIR / "tidy_scenarios.csv", index=False)

# wide = d.pivot_table(
#     index=["event", "short", "scale"],
#     columns="approach",
#     values=["red_" + m for m in METRICS],
#     observed=True,
# )
# for m in METRICS:
#     wide[("err_" + m, "")] = wide[("red_" + m, "POST")] - wide[("red_" + m, "PRE")]
# # wide.to_csv(OUT_DIR / "post_vs_pre_wide.csv")

# # ------------------------------------------------------ 3a. does POST resolve scale?
# print(
#     "\n# Rows sharing an identical metric vector (i.e. the approach cannot "
#     "distinguish two designs):"
# )
# cols = ["flooded_area_km2", "urban_exposed_km2", "mean_depth_m", "volume_m3"]
# for (e, a), g in d.groupby(["event", "approach"]):
#     print(f"  {e:14s} {a:5s}: {g.duplicated(cols, keep=False).sum():2d} / {len(g)}")

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

# # ------------------------------------------------------------------ 4. figures
# events = sorted(d.event.unique())
# measures = sorted(d.short.unique())

# STYLE = {
#     "PRE": dict(fmt="-o", color="#1b4965", label="PRE (re-run)"),
#     "POST": dict(fmt="--s", color="#e07a5f", label="POST (post-processed)"),
# }


# def label(s):
#     return s.replace("_", " ").title()


# METRIC_INFO = [
#     ("flooded_area_km2", "residual flood extent", (0, 2), [0, 0.25, 0.5, 1, 1.5, 2]),
#     (
#         "urban_exposed_km2",
#         "residual urban exposure",
#         (0, 8),
#         [0, 0.25, 0.5, 1, 2, 4, 8],
#     ),
#     ("volume_m3", "residual flood volume", (0, 2), [0, 0.25, 0.5, 1, 1.5, 2]),
#     ("mean_depth_m", "residual mean flood depth", (0, 2), [0, 0.25, 0.5, 1, 1.5, 2]),
# ]

# for METRIC, LABEL, YLIM, YTICKS in METRIC_INFO:
#     BL = baselines[METRIC].to_dict()
#     d["ratio"] = d[METRIC] / d.event.map(BL)

#     fig, axes = plt.subplots(
#         len(events),
#         len(measures),
#         figsize=(3 * len(measures), 2.7 * len(events)),
#         sharex=True,
#         sharey=True,
#         squeeze=False,
#     )

#     for i, e in enumerate(events):
#         for j, ms in enumerate(measures):
#             A = axes[i, j]
#             g = d[(d.event == e) & (d.short == ms)]

#             for ap, s in STYLE.items():
#                 gg = g[g.approach == ap].set_index("scale").reindex(SCALES)
#                 A.plot(
#                     SCALES, gg["ratio"], s["fmt"], color=s["color"], label=s["label"]
#                 )

#             A.set_yscale("symlog", linthresh=0.05)
#             A.set_ylim(*YLIM)
#             A.axhline(1, c="crimson", lw=0.8)
#             A.axhspan(1, YLIM[1], color="crimson", alpha=0.06)
#             A.set_yticks(YTICKS)
#             A.set_yticklabels([f"{t:g}" for t in YTICKS], fontsize=7)
#             A.tick_params(labelsize=7, labelrotation=45)
#             if i == 0:
#                 A.set_title(label(ms), fontsize=9)
#             if j == 0:
#                 A.set_ylabel(f"{label(e)}\n{LABEL}", fontsize=8)

#     axes[0, 0].legend(fontsize=8)
#     fig.suptitle(
#         f"{LABEL.capitalize()} relative to the no-adaptation baseline "
#         "(1 = no change, 0 = eliminated, >1 = worse)"
#     )
#     plt.tight_layout()
#     fig.savefig(OUT_DIR / f"fig_scale_response_{METRIC}.png", dpi=150)

# print(
#     f"\nWrote outputs to {OUT_DIR}: tidy_scenarios.csv, post_vs_pre_wide.csv, "
#     "spearman_by_event_metric.csv, fig_scale_response_<metric>.png"
# )

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
SCENARIOS = ["coast_500", "river_500", "compound_500"]
SCENARIO_COLOR = "#1b4965"
# current config (config.yml boundary_forcings.surge.slr.slr_m) is 0.0, so
# the plotted water level IS the actual built forcing, with no SLR -- shown
# as a flat reference line AT y=SLR_M (an illustrative SLR magnitude, not a
# baseline+SLR shift) on the coastal scenarios only, so the flood peak's
# height can be read directly against it (river_500 has near-baseline surge
# to begin with, so this reference isn't the interesting comparison there).
SLR_M = 0.5
SLR_SCENARIOS = {"coast_500", "compound_500"}

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
        dis.time.values, dis["dis"].mean(dim="index").values, color="#e07a5f", lw=1.8
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
