"""Recompute flood metrics at several flooding depth thresholds (hmin).

Leaves every pipeline output untouched; the pipeline's own threshold is
config metrics.hmin.
  baseline + pre  flood map recomputed from each run's own SFINCS zsmax with
                  hmin=0, then thresholded per hmin with `> hmin`, as
                  hydromt's downscale_floodmap does (hmax.where(hmax > hmin)).
  post            the existing post flood maps (built at metrics.hmin)
                  thresholded per hmin. Below metrics.hmin they cannot be
                  recomputed without rerunning the post-processing, so those
                  rows are NaN. The threshold applies to the final depth, so
                  pixels a measure lowered below it (nbs_land_reclamation) drop
                  out, while the pipeline still counts them as flooded.
Only the metrics change: the attribution mask and the post-processing
decisions stay the pipeline's own.

Output: <results_dir>/<basin>/figures/analysis/90_hmin_sensitivity_metrics.csv,
the columns of runs/metrics_comparison.csv plus `hmin`; hmin_sensitivity_fig.py
plots it.
"""

import re
import sys

import pandas as pd
import rioxarray as rxr
import yaml
from _common import (
    DEFAULT_BASIN_ID,
    REPO_ROOT,
    analysis_dir,
    analysis_path,
    script_parser,
    site_path,
)

sys.path.insert(0, str(REPO_ROOT / "workflow"))
from src.adaptation_strategies import resolve_locations  # noqa: E402
from src.postprocessing import (  # noqa: E402
    compute_max_inundation,
    compute_risk_metrics,
    read_land_bed_level,
)
from src.raster import FLOAT_NODATA  # noqa: E402

HMINS = [0.0, 0.05, 0.1]
SCENARIOS = ["coast_100_03SLR", "river_100_03SLR", "compound_100_03SLR"]
METRICS = ["flooded_area_km2", "urban_exposed_km2", "mean_depth_m", "volume_m3"]


def retention_zone(strategy, strategy_families, basin_root, basin_id):
    """Return the water_retention zone a pre run excludes from its metrics, or None.

    As rule adapt_flood_metrics_pre: the zone is the measure's locations file
    under adaptation_data_root.
    """
    family = re.sub(r"_(04|09|1)$", "", strategy)
    family_def = strategy_families.get(family)
    if not family_def or "water_retention" not in family_def.get("measures", {}):
        return None
    resolved = resolve_locations(family_def, basin_root / "adaptation_lines", basin_id)
    return (
        site_path("adaptation_data_root")
        / (resolved["measures"]["water_retention"]["locations"])
    )


def metrics_rows(
    da_hmax, da_dep, landuse, hmins, ctx, map_dir=None, exclude=None, **labels
):
    """Return one metrics row per threshold, from a flood map holding every depth > 0."""
    rows = []
    for h in hmins:
        da_h = da_hmax.where(da_hmax > h)
        if map_dir is not None:
            p = ctx["maps_root"] / f"hmin_{h:g}" / map_dir / "max_flood_depth.tif"
            p.parent.mkdir(parents=True, exist_ok=True)
            da_h.raster.to_raster(str(p), dtype="float32", nodata=FLOAT_NODATA)
        m = compute_risk_metrics(
            da_h,
            da_dep,
            landuse,
            ctx["delta_polygon"],
            urban_code=ctx["urban_code"],
            exclude_geom_path=exclude,
        )
        rows.append({"basin_id": ctx["basin_id"], **labels, "hmin": h, **m})
    return rows


def scenario_rows(sc, ctx, strategy_families, save_maps):
    """Return the metrics rows of one scenario's baseline, pre and post runs."""
    runs_dir = ctx["basin_root"] / "runs"
    pipeline_hmin = ctx["pipeline_hmin"]

    # baseline: inputs as rule compute_flood_metrics
    print(f"\n{sc}: baseline")
    base_dep = read_land_bed_level(
        ctx["basin_root"] / "sfincs_skeleton" / "land_bed_level.nc"
    )
    da_hmax, _ = compute_max_inundation(runs_dir / sc / "sfincs", base_dep, hmin=0.0)
    if da_hmax is None:
        raise RuntimeError(f"{sc}: zsmax unavailable -- did the event run finish?")
    rows = metrics_rows(
        da_hmax,
        base_dep,
        ctx["landuse"],
        HMINS,
        ctx,
        map_dir=sc if save_maps else None,
        scenario=sc,
        strategy="no strategy",
        method="baseline",
    )

    # pre: inputs as rule adapt_flood_metrics_pre
    for run in sorted(
        (runs_dir / sc / "adaptation" / "pre").glob("*/sfincs/sfincs_map.nc")
    ):
        st, skel = run.parents[1].name, run.parents[1] / "sfincs_skeleton"
        print(f"{sc}: pre  {st}")
        da_dep = read_land_bed_level(skel / "land_bed_level.nc")
        da_hmax, _ = compute_max_inundation(run.parent, da_dep, hmin=0.0)
        if da_hmax is None:
            raise RuntimeError(f"{sc} pre {st}: zsmax unavailable")
        rows += metrics_rows(
            da_hmax,
            da_dep,
            skel / "retreat_landuse.tif",
            HMINS,
            ctx,
            map_dir=f"{sc}/pre/{st}" if save_maps else None,
            exclude=retention_zone(
                st, strategy_families, ctx["basin_root"], ctx["basin_id"]
            ),
            scenario=sc,
            strategy=st,
            method="pre",
        )

    # post: existing maps (built at pipeline_hmin), metrics as 17c_adapt_post.py
    post_hmins = [h for h in HMINS if h >= pipeline_hmin]
    for tif in sorted(
        (runs_dir / sc / "adaptation" / "post").glob("*/max_flood_depth.tif")
    ):
        st = tif.parent.name
        print(f"{sc}: post {st}")
        # masked=True: measures write a literal nodata that would otherwise count as flooded
        da_hmax = rxr.open_rasterio(tif, masked=True).squeeze(drop=True)
        retreat_lu = tif.parent / "retreat_landuse.tif"
        rows += metrics_rows(
            da_hmax,
            base_dep,
            retreat_lu if retreat_lu.exists() else ctx["landuse"],
            post_hmins,
            ctx,
            map_dir=f"{sc}/post/{st}" if save_maps else None,
            scenario=sc,
            strategy=st,
            method="post",
        )
        rows += [
            {
                "basin_id": ctx["basin_id"],
                "scenario": sc,
                "strategy": st,
                "method": "post",
                "hmin": h,
            }
            for h in HMINS
            if h < pipeline_hmin
        ]
    return rows


def check_against_pipeline(df, comparison_csv, pipeline_hmin):
    """Print how many rows at the pipeline's threshold match metrics_comparison.csv.

    Known exception: post measures that lower depths (nbs_land_reclamation)
    leave pixels below the threshold that the pipeline still counts as flooded.
    """
    main_df = pd.read_csv(comparison_csv, sep=None, engine="python")
    chk = df[df.hmin == pipeline_hmin].merge(
        main_df, on=["scenario", "strategy", "method"], suffixes=("", "_main")
    )
    for m in METRICS:
        chk["diff_" + m] = chk[m] - chk[m + "_main"]
    bad = chk[(chk[["diff_" + m for m in METRICS]].abs() > 0).any(axis=1)]
    print(
        f"Check vs runs/metrics_comparison.csv at hmin={pipeline_hmin}: "
        f"{len(chk) - len(bad)} of {len(chk)} rows identical"
    )
    if len(bad):
        print("Differing rows (this script minus the pipeline):")
        cols = ["scenario", "method", "strategy"] + ["diff_" + m for m in METRICS]
        print(bad[cols].to_string(index=False))


def build_parser():
    """Return the command-line parser."""
    parser = script_parser(__doc__)
    parser.add_argument(
        "scenarios",
        nargs="*",
        default=SCENARIOS,
        help=f"scenario name(s) (default: {' '.join(SCENARIOS)})",
    )
    parser.add_argument(
        "--basin-id",
        default=DEFAULT_BASIN_ID,
        help=f"basin id (default: {DEFAULT_BASIN_ID})",
    )
    parser.add_argument(
        "--save-maps",
        action="store_true",
        help="also write every thresholded max_flood_depth.tif to "
        "figures/analysis/90_hmin_sensitivity_maps/",
    )
    return parser


def main():
    """Recompute the metrics at every threshold in HMINS and write the CSV."""
    args = build_parser().parse_args()
    basin_id = str(args.basin_id)

    cfg = yaml.safe_load(
        (REPO_ROOT / "config" / "config.yml").read_text(encoding="utf-8")
    )
    strategy_families = yaml.safe_load(
        (REPO_ROOT / "config" / "adaptation_strategies.yml").read_text(encoding="utf-8")
    )
    # results_dir is a site setting; see config/site.example.yml.
    basin_root = site_path("results_dir") / basin_id
    domain_dir = basin_root / "preprocessing_inputs" / "domain"
    ctx = {
        "basin_id": basin_id,
        "basin_root": basin_root,
        "pipeline_hmin": float(cfg["metrics"]["hmin"]),
        "urban_code": int(cfg["metrics"]["urban_landuse_code"]),
        "landuse": domain_dir / f"{basin_id}_landuse_channels.tif",
        "delta_polygon": domain_dir / f"{basin_id}_delta_polygon.gpkg",
        "maps_root": analysis_dir(basin_id) / "90_hmin_sensitivity_maps",
    }

    if ctx["pipeline_hmin"] not in HMINS:
        raise ValueError(
            f"HMINS {HMINS} must include metrics.hmin {ctx['pipeline_hmin']}"
        )

    rows = []
    for sc in args.scenarios:
        rows += scenario_rows(sc, ctx, strategy_families, args.save_maps)

    df = pd.DataFrame(rows)
    out_csv = analysis_path(basin_id, __file__, "metrics.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(df)} rows)")
    pipeline_hmin = ctx["pipeline_hmin"]
    if any(h < pipeline_hmin for h in HMINS):
        print(
            f"Note: post rows below {pipeline_hmin} m are NaN -- the post flood maps "
            f"only exist at {pipeline_hmin} m (rerun post-processing to get them)."
        )

    check_against_pipeline(
        df, basin_root / "runs" / "metrics_comparison.csv", pipeline_hmin
    )

    base = df[df.method == "baseline"].set_index(["scenario", "hmin"])[METRICS[:2]]
    ref = base.xs(pipeline_hmin, level="hmin")
    print(
        f"\nBaseline flooded area / urban exposure per threshold "
        f"(ratio to {pipeline_hmin:g} m):"
    )
    print((base / ref.reindex(base.index, level="scenario")).round(3).to_string())


if __name__ == "__main__":
    main()
