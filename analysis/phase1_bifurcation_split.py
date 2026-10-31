"""Phase 1.5: SFINCS discharge split at the bifurcations of the forcing paths vs the width share.

`run`: copies the basin's round-1 calibration model to a scratch folder
(GCFM_PHASE1_SCRATCH, default <results_dir>/../scratch_phase1/<basin>/round1_crs),
adds cross-section lines across each arm of every bifurcation on the seeds'
widest paths (src.river_depth_calibration.bifurcation_cross_sections, the
helper behind river_depth_modelling.crosssections) as sfincs.crs + crsfile,
and reruns SFINCS there. The production round-1 folder is not touched.
`compare`: end-of-run discharge through each arm (sfincs_his.nc
crosssection_discharge, absolute value) as a share of its bifurcation, against
the width share src.river_network.accumulate_discharge gave the calibration
discharge (q_calib of the depth-estimated network).

Writes per basin 90_phase1_bifurcation_split_arms.csv to
<results_dir>/<basin>/figures/analysis/ and phase1_bifurcation_split_summary.csv to
<results_dir>/figures/analysis/.

Usage: python analysis/phase1_bifurcation_split.py run|compare [basin_id ...]
"""

import logging
import os
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from _common import analysis_path, script_parser, site_path
from _phase1 import PHASE1_BASINS, basin_paths, reach_q_dict, write_summary

log = logging.getLogger(__name__)


def scratch_dir(basin_id: str) -> Path:
    root = os.environ.get("GCFM_PHASE1_SCRATCH") or str(
        site_path("results_dir").parent / "scratch_phase1"
    )
    return Path(root) / basin_id / "round1_crs"


def cross_sections(basin_id: str) -> gpd.GeoDataFrame:
    from src.grid import load_grid_def
    from src.river_depth_calibration import bifurcation_cross_sections
    from src.river_network import normalize_reach_id

    p = basin_paths(basin_id)
    rivers = gpd.read_file(p["rivers_clean"]).to_crs(
        load_grid_def(p["sfincs_grid"])["crs"]
    )
    seeds = sorted(
        {
            normalize_reach_id(r)
            for r, s in zip(rivers["reach_id"], rivers["is_seed"])
            if bool(s)
        }
        - {None}
    )
    return bifurcation_cross_sections(rivers, seeds)


def run(basin_id: str, threads: int) -> None:
    from hydromt_sfincs import utils
    from src.io import site_setting
    from src.sfincs_run import run_sfincs_subprocess

    crs = cross_sections(basin_id)
    if crs.empty:
        log.warning(f"{basin_id}: no bifurcation on the seeds' paths -- nothing to run")
        return
    src_dir, dst = basin_paths(basin_id)["round1"], scratch_dir(basin_id)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src_dir,
        dst,
        ignore=shutil.ignore_patterns("sfincs_map.nc", "sfincs_his.nc", "*.rst"),
    )
    utils.write_geoms(
        dst / "sfincs.crs",
        utils.gdf2linestring(crs[["name", "geometry"]]),
        stype="crs",
        fmt="%11.1f",
    )
    crs.to_file(dst / "cross_sections.gpkg", driver="GPKG")
    inp = (dst / "sfincs.inp").read_text().splitlines()
    inp = [line for line in inp if not line.strip().lower().startswith("crsfile")]
    (dst / "sfincs.inp").write_text(
        "\n".join(inp + ["crsfile              = sfincs.crs", ""])
    )
    log.info(f"{basin_id}: {len(crs)} cross-section(s); running SFINCS in {dst}")
    run_sfincs_subprocess(
        site_setting("sfincs_exe"),
        dst,
        6 * 3600,
        log,
        label=f"round 1 + crs ({basin_id})",
        n_threads=threads,
    )


def compare(basin_id: str) -> list[dict]:
    p = basin_paths(basin_id)
    dst = scratch_dir(basin_id)
    crs = gpd.read_file(dst / "cross_sections.gpkg")
    with xr.open_dataset(dst / "sfincs_his.nc", decode_times=False) as ds:
        q_end = np.abs(np.asarray(ds["crosssection_discharge"].values, dtype=float)[-1])
        names = (
            [
                str(n).strip()
                for n in np.asarray(ds["crosssection_name"].values).astype(str)
            ]
            if "crosssection_name" in ds
            else list(crs["name"])
        )
    q_by_name = dict(zip(names, q_end))
    depth = gpd.read_file(p["rivers_depth"])
    q_calib = reach_q_dict(depth, depth["q_calib"].to_numpy(float))
    crs["q_sfincs_m3s"] = crs["name"].map(q_by_name)
    crs["q_width_m3s"] = crs["arm_reach"].map(q_calib)
    crs["share_sfincs"] = crs["q_sfincs_m3s"] / crs.groupby("parent_reach")[
        "q_sfincs_m3s"
    ].transform("sum")
    crs["share_width"] = crs["q_width_m3s"] / crs.groupby("parent_reach")[
        "q_width_m3s"
    ].transform("sum")
    crs["abs_error"] = (crs["share_sfincs"] - crs["share_width"]).abs()
    out = crs.drop(columns="geometry")
    out.to_csv(
        analysis_path(basin_id, __file__, "arms.csv"), index=False, float_format="%.4g"
    )
    return [{"basin_id": basin_id, **r} for r in out.to_dict("records")]


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("action", choices=["run", "compare"])
    parser.add_argument("basin_ids", nargs="*", default=PHASE1_BASINS)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    if args.action == "run":
        for b in args.basin_ids:
            run(str(b), args.threads)
        return 0
    rows = [r for b in args.basin_ids for r in compare(str(b))]
    print(
        write_summary(rows, analysis_path(None, __file__, "summary.csv")).to_string(
            index=False
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
