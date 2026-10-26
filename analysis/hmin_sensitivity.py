"""
KL_hmin_sensitivity.py -- Sensitivity of the flood metrics to the flooding depth
threshold (config metrics.hmin, 0.05 m in the pipeline).

Recomputes the flood maps and risk metrics at every threshold in HMINS, without
touching any pipeline output:
  baseline + pre  flood map recomputed from each run's own SFINCS zsmax, once with
                  hmin=0, then thresholded per hmin with `> hmin` -- exactly what
                  hydromt's downscale_floodmap does (hmax.where(hmax > hmin)).
  post            the existing post flood maps (built at 0.05 m) thresholded per
                  hmin. Below 0.05 m they can't be recomputed without rerunning
                  the post-processing, so those rows are NaN. The threshold is
                  applied to the FINAL depth, so pixels a measure lowered below
                  it (nbs_land_reclamation) drop out -- the pipeline itself
                  still counts those as flooded.
Only the metrics change: the attribution mask and the post-processing decisions
stay the pipeline's own.

Output: <results>/<basin>/hmin_sensitivity/metrics_hmin_sensitivity.csv, same
columns as runs/metrics_comparison.csv plus `hmin`.

Usage:
  python tests/KL_hmin_sensitivity.py                    # all SCENARIOS
  python tests/KL_hmin_sensitivity.py river_100_03SLR    # just these scenarios
"""

import os
import re
import sys
from pathlib import Path

import pandas as pd
import rioxarray as rxr
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "workflow"))
from src.postprocessing import compute_max_inundation, compute_risk_metrics  # noqa: E402

BASIN_ID = "2433835"
HMINS = [0.0, 0.05, 0.1]
SCENARIOS = ["coast_100_03SLR", "river_100_03SLR", "compound_100_03SLR"]
URBAN_CODE = 50  # config metrics.urban_landuse_code
SAVE_MAPS = False  # True: also write every thresholded max_flood_depth.tif
PIPELINE_HMIN = 0.05  # threshold the existing post maps were built with

# Machine-specific paths, read from environment variables (see
# CONTRIBUTING.md "Local machine paths")
BASIN_ROOT = Path(os.environ["GCFM_RESULTS_DIR"]) / BASIN_ID
RAW_ROOT = Path(os.environ["GCFM_RAW_DATA_ROOT"])
RUNS_DIR = BASIN_ROOT / "runs"
DOMAIN_DIR = BASIN_ROOT / "preprocessing_inputs" / "domain"
OUT_DIR = BASIN_ROOT / "hmin_sensitivity"

SKELETON_ROOT = BASIN_ROOT / "sfincs_skeleton"
SEA_MASK = DOMAIN_DIR / f"{BASIN_ID}_zsini_sea_cells_on_grid.tif"
LANDUSE = DOMAIN_DIR / f"{BASIN_ID}_landuse.tif"
DELTA_POLYGON = DOMAIN_DIR / f"{BASIN_ID}_delta_polygon.gpkg"

with open(REPO / "config" / "adaptation_strategies.yml") as f:
    STRATEGY_FAMILIES = yaml.safe_load(f)


def retention_zone(strategy):
    """Retention zone to exclude from the metrics, as rule 17 does for a pre run
    with water_retention / water_retention_greening; None otherwise.
    """
    family = re.sub(r"_(04|09|1)$", "", strategy)
    measures = (STRATEGY_FAMILIES.get(family) or {}).get("measures", {})
    m = measures.get("water_retention", measures.get("water_retention_greening"))
    return RAW_ROOT / m["locations"] if m else None


def metrics_rows(da_hmax, da_dep, landuse, hmins, out_dir, exclude=None, **labels):
    """One metrics row per threshold, from a flood map holding every depth > 0."""
    rows = []
    for h in hmins:
        da_h = da_hmax.where(da_hmax > h)
        if SAVE_MAPS:
            p = OUT_DIR / f"hmin_{h:g}" / out_dir / "max_flood_depth.tif"
            p.parent.mkdir(parents=True, exist_ok=True)
            da_h.raster.to_raster(str(p), dtype="float32", nodata=-9999.0)
        m = compute_risk_metrics(
            da_h,
            da_dep,
            landuse,
            DELTA_POLYGON,
            urban_code=URBAN_CODE,
            exclude_geom_path=exclude,
        )
        rows.append({"basin_id": BASIN_ID, **labels, "hmin": h, **m})
    return rows


scenarios = sys.argv[1:] or SCENARIOS
rows = []
for sc in scenarios:
    # baseline -- inputs as rule compute_flood_metrics (17)
    print(f"\n{sc}: baseline")
    da_hmax, da_dep = compute_max_inundation(
        RUNS_DIR / sc / "sfincs", SKELETON_ROOT, SEA_MASK, hmin=0.0
    )
    base_dep = da_dep  # the land grid post's own metrics use (as 18b_adapt_post.py)
    rows += metrics_rows(
        da_hmax,
        da_dep,
        LANDUSE,
        HMINS,
        Path(sc),
        scenario=sc,
        strategy="no strategy",
        method="baseline",
    )

    # pre -- inputs as rule adapt_flood_metrics_pre (18a)
    for run in sorted(
        (RUNS_DIR / sc / "adaptation" / "pre").glob("*/sfincs/sfincs_map.nc")
    ):
        st, skel = run.parents[1].name, run.parents[1] / "sfincs_skeleton"
        print(f"{sc}: pre  {st}")
        da_hmax, da_dep = compute_max_inundation(
            run.parent, skel, skel / "sea_mask.tif", hmin=0.0
        )
        rows += metrics_rows(
            da_hmax,
            da_dep,
            skel / "retreat_landuse.tif",
            HMINS,
            Path(sc) / "pre" / st,
            exclude=retention_zone(st),
            scenario=sc,
            strategy=st,
            method="pre",
        )

    # post -- existing maps (built at PIPELINE_HMIN), metrics as 18b_adapt_post.py
    post_hmins = [h for h in HMINS if h >= PIPELINE_HMIN]
    for tif in sorted(
        (RUNS_DIR / sc / "adaptation" / "post").glob("*/max_flood_depth.tif")
    ):
        st = tif.parent.name
        print(f"{sc}: post {st}")
        da_hmax = rxr.open_rasterio(tif, masked=True).squeeze(drop=True)
        retreat_lu = tif.parent / "retreat_landuse.tif"
        rows += metrics_rows(
            da_hmax,
            base_dep,
            retreat_lu if retreat_lu.exists() else LANDUSE,
            post_hmins,
            Path(sc) / "post" / st,
            scenario=sc,
            strategy=st,
            method="post",
        )
        rows += [
            {
                "basin_id": BASIN_ID,
                "scenario": sc,
                "strategy": st,
                "method": "post",
                "hmin": h,
            }
            for h in HMINS
            if h < PIPELINE_HMIN
        ]

df = pd.DataFrame(rows)
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_csv = OUT_DIR / "metrics_hmin_sensitivity.csv"
df.to_csv(out_csv, index=False)
print(f"\nWrote {out_csv} ({len(df)} rows)")
if any(h < PIPELINE_HMIN for h in HMINS):
    print(
        f"Note: post rows below {PIPELINE_HMIN} m are NaN -- the post flood maps only "
        f"exist at {PIPELINE_HMIN} m (rerun post-processing to get them)."
    )

# Check: at the pipeline's own threshold every row should equal the pipeline's
# metrics_comparison.csv. Known exception: post measures that lower depths
# (nbs_land_reclamation) leave pixels below the threshold that the pipeline
# still counts as flooded, while this script thresholds the final depth.
METRICS = ["flooded_area_km2", "urban_exposed_km2", "mean_depth_m", "volume_m3"]
main = pd.read_csv(RUNS_DIR / "metrics_comparison.csv", sep=None, engine="python")
chk = df[df.hmin == PIPELINE_HMIN].merge(
    main, on=["scenario", "strategy", "method"], suffixes=("", "_main")
)
for m in METRICS:
    chk["diff_" + m] = chk[m] - chk[m + "_main"]
bad = chk[(chk[["diff_" + m for m in METRICS]].abs() > 0).any(axis=1)]
print(
    f"Check vs runs/metrics_comparison.csv at hmin={PIPELINE_HMIN}: "
    f"{len(chk) - len(bad)} of {len(chk)} rows identical"
)
if len(bad):
    print("Differing rows (this script minus the pipeline):")
    cols = ["scenario", "method", "strategy"] + ["diff_" + m for m in METRICS]
    print(bad[cols].to_string(index=False))

# Summary: baseline metrics per threshold, relative to the pipeline's threshold
base = df[df.method == "baseline"].set_index(["scenario", "hmin"])[METRICS[:2]]
ref = base.xs(PIPELINE_HMIN, level="hmin")
print("\nBaseline flooded area / urban exposure per threshold (ratio to 0.05 m):")
print((base / ref.reindex(base.index, level="scenario")).round(3).to_string())
