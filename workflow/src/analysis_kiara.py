"""
POST- vs PRE-processing agreement analysis for flood adaptation measures.

Step 1: split `strategy` into measure name + scale suffix
Step 2: convert absolute metrics to % risk reduction vs the matching event baseline
Step 3: agreement diagnostics (scatter, signed error, confusion matrix, Spearman)

Run:  python analysis_kiara.py
"""

from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(r"D:\GCFM_UU\results\2433835\runs")
CSV = OUT_DIR / "metrics_comparison.csv"
METRICS = ["flooded_area_km2", "urban_exposed_km2", "mean_depth_m", "volume_m3"]
SCALES = ["04", "09", "1"]  # raw strategy-name suffix, low -> high implementation scale

# ----------------------------------------------------------------- 1. parse
raw = pd.read_csv(CSV)
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

# ------------------------------------------- 2. % reduction vs event baseline
for m in METRICS:
    b = d.event.map(baselines[m])
    d["red_" + m] = 100 * (b - d[m]) / b

d.to_csv(OUT_DIR / "tidy_scenarios.csv", index=False)

wide = d.pivot_table(
    index=["event", "short", "scale"],
    columns="approach",
    values=["red_" + m for m in METRICS],
    observed=True,
)
for m in METRICS:
    wide[("err_" + m, "")] = wide[("red_" + m, "POST")] - wide[("red_" + m, "PRE")]
wide.to_csv(OUT_DIR / "post_vs_pre_wide.csv")

# ------------------------------------------------------ 3a. does POST resolve scale?
print(
    "\n# Rows sharing an identical metric vector (i.e. the approach cannot "
    "distinguish two designs):"
)
cols = ["flooded_area_km2", "urban_exposed_km2", "mean_depth_m", "volume_m3"]
for (e, a), g in d.groupby(["event", "approach"]):
    print(f"  {e:14s} {a:5s}: {g.duplicated(cols, keep=False).sum():2d} / {len(g)}")

# ------------------------------------------------------ 3b. Spearman per event
print("\n# Spearman rho, POST vs PRE ranking of all measure-scale combinations")
rows = []
for e, g in d.groupby("event"):
    p = g.pivot_table(
        index=["measure", "scale"],
        columns="approach",
        values=["red_" + m for m in METRICS],
        observed=True,
    )
    for m in METRICS:
        r, pv = spearmanr(p[("red_" + m, "POST")], p[("red_" + m, "PRE")])
        rows.append(dict(event=e, metric=m, rho=round(r, 2), p=round(pv, 4)))
sp = pd.DataFrame(rows)
print(sp.pivot(index="metric", columns="event", values="rho").to_string())
sp.to_csv(OUT_DIR / "spearman_by_event_metric.csv", index=False)

# ------------------------------------------- 3c. binary agreement (tune threshold)
THRESH = 25.0  # % reduction in urban exposure counted as "effective"
u = wide["red_urban_exposed_km2"].reset_index()
u["POST_eff"], u["PRE_eff"] = u.POST > THRESH, u.PRE > THRESH
print(f"\n# Confusion matrix, 'effective' = >{THRESH:.0f}% reduction in urban exposure")
print(
    pd.crosstab(u.PRE_eff, u.POST_eff, rownames=["PRE (benchmark)"], colnames=["POST"])
)
u["error"] = u.POST - u.PRE
print("\n# Disagreements, sorted by magnitude")
print(
    u[u.POST_eff != u.PRE_eff]
    .reindex(u.error.abs().sort_values(ascending=False).index)
    .dropna(subset=["error"])[["event", "short", "scale", "PRE", "POST", "error"]]
    .round(1)
    .to_string(index=False)
)

# ------------------------------------------------------------------ 4. figures
events = sorted(d.event.unique())
measures = sorted(d.short.unique())

STYLE = {
    "PRE": dict(fmt="-o", color="#1b4965", label="PRE (re-run)"),
    "POST": dict(fmt="--s", color="#e07a5f", label="POST (post-processed)"),
}


def label(s):
    return s.replace("_", " ").title()


METRIC_INFO = [
    ("flooded_area_km2", "residual flood extent", (0, 2), [0, 0.25, 0.5, 1, 1.5, 2]),
    (
        "urban_exposed_km2",
        "residual urban exposure",
        (0, 8),
        [0, 0.25, 0.5, 1, 2, 4, 8],
    ),
    ("volume_m3", "residual flood volume", (0, 2), [0, 0.25, 0.5, 1, 1.5, 2]),
    ("mean_depth_m", "residual mean flood depth", (0, 2), [0, 0.25, 0.5, 1, 1.5, 2]),
]

for METRIC, LABEL, YLIM, YTICKS in METRIC_INFO:
    BL = baselines[METRIC].to_dict()
    d["ratio"] = d[METRIC] / d.event.map(BL)

    fig, axes = plt.subplots(
        len(events),
        len(measures),
        figsize=(3 * len(measures), 2.7 * len(events)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for i, e in enumerate(events):
        for j, ms in enumerate(measures):
            A = axes[i, j]
            g = d[(d.event == e) & (d.short == ms)]

            for ap, s in STYLE.items():
                gg = g[g.approach == ap].set_index("scale").reindex(SCALES)
                A.plot(
                    SCALES, gg["ratio"], s["fmt"], color=s["color"], label=s["label"]
                )

            A.set_yscale("symlog", linthresh=0.05)
            A.set_ylim(*YLIM)
            A.axhline(1, c="crimson", lw=0.8)
            A.axhspan(1, YLIM[1], color="crimson", alpha=0.06)
            A.set_yticks(YTICKS)
            A.set_yticklabels([f"{t:g}" for t in YTICKS], fontsize=7)
            A.tick_params(labelsize=7, labelrotation=45)
            if i == 0:
                A.set_title(label(ms), fontsize=9)
            if j == 0:
                A.set_ylabel(f"{label(e)}\n{LABEL}", fontsize=8)

    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"{LABEL.capitalize()} relative to the no-adaptation baseline "
        "(1 = no change, 0 = eliminated, >1 = worse)"
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"fig_scale_response_{METRIC}.png", dpi=150)

print(
    f"\nWrote outputs to {OUT_DIR}: tidy_scenarios.csv, post_vs_pre_wide.csv, "
    "spearman_by_event_metric.csv, fig_scale_response_<metric>.png"
)
