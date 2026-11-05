"""Phase 2c: search for a SFINCS setting stable above alpha 0.25 with advection off.

`synthetic`: the Phase 2b trench (phase2_advection_channel.build_case),
advection 0, from the two cases unstable at alpha 0.5 (aligned at 3000 m3/s,
diagonal at 500 m3/s). One factor at a time from alpha 0.5: alpha, discharge
ramp time, a sudden bed drop mid-channel, inflow position and spread, and the
SFINCS stability keywords theta, wiggle_suppression and viscosity. Per run,
from 10-minute maps of the centreline cells: the largest 10-minute level step
after the ramp, the number of step reversals (consecutive steps over
REVERSAL_M in opposite directions), the end-level error against normal depth
(bed drops added upstream) and the wall time.
`event`: reruns a finished event of <results_dir> (restart from its spin-up)
in scratch with sfincs.inp keywords set (--set key=value ...).
`event-compare`: per rerun against the pipeline run: largest hourly level
step and cells stepping more than STEP_FLAG_M in an hour, cells reversing,
the zsmax difference and the change in wet cells, and the wall time.

Scratch under $GCFM_PLAN_SCRATCH/p2c (default <results_dir>/../scratch_plan/p2c).
Writes phase2c_alpha_stability_synthetic_summary.csv and _synthetic.png to
<results_dir>/figures/analysis/, and per basin
90_phase2c_alpha_stability_event_<scenario>_summary.csv and .png.

Usage:
  python analysis/phase2c_alpha_stability.py synthetic [--workers 4 --threads 3]
  python analysis/phase2c_alpha_stability.py event 2433835 river_100_03SLR --name alpha0.5 --set alpha=0.5
  python analysis/phase2c_alpha_stability.py event-compare 2433835 river_100_03SLR
"""

import logging
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from _common import REPO_ROOT, analysis_path, script_parser, site_path
from _phase1 import git_hash, load_config, write_summary

log = logging.getLogger("alpha_stability")

BASES = {"aligned_q3000": ("aligned", 3000.0), "diagonal_q500": ("diagonal", 500.0)}
DROP_AT_M = 10_000.0
SYNTHETIC_VARIANTS = {
    "alpha0.25": {"alpha": 0.25},
    "alpha0.35": {"alpha": 0.35},
    "alpha0.4": {"alpha": 0.4},
    "alpha0.45": {"alpha": 0.45},
    "alpha0.5": {},
    "ramp6h": {"ramp_hr": 6.0, "run_hr": 18.0},
    "ramp24h": {"ramp_hr": 24.0, "run_hr": 36.0},
    "drop0.5m": {"bed_drops": ((DROP_AT_M, 0.5),)},
    "drop1m": {"bed_drops": ((DROP_AT_M, 1.0),)},
    "drop2m": {"bed_drops": ((DROP_AT_M, 2.0),)},
    "drop1m_alpha0.25": {"alpha": 0.25, "bed_drops": ((DROP_AT_M, 1.0),)},
    "inflow_2km": {"inflow_s_m": 2000.0},
    "inflow_3cells": {"n_inflow": 3},
    "theta0.9": {"extra_config": {"theta": 0.9}},
    "theta0.8": {"extra_config": {"theta": 0.8}},
    "wiggle": {"extra_config": {"wiggle_suppression": 1}},
    "viscosity0": {"extra_config": {"viscosity": 0}},
    **{
        f"dx{dx}_alpha{a:g}": {"alpha": a, "dx_m": float(dx)}
        for dx in (50, 200)
        for a in (0.25, 0.35, 0.5)
    },
}
SUBGRID_PIXEL_M = 10.0
MAP_STEP_S = 600
REVERSAL_M = 0.05
UNSTABLE_STEP_M = 0.5
STEP_FLAG_M = 2.0
WET_M = 0.15


def scratch_root() -> Path:
    base = os.environ.get("GCFM_PLAN_SCRATCH") or (
        site_path("results_dir").parent / "scratch_plan"
    )
    return Path(base) / "p2c"


def _centreline_cells(r: dict) -> pd.DataFrame:
    col = np.floor((r["xy"][:, 0] - r["x0"]) / r["dx_m"]).astype(int)
    row = np.floor((r["xy"][:, 1] - r["y0"]) / r["dx_m"]).astype(int)
    cells = (
        pd.DataFrame({"row": row, "col": col, "s_m": r["s"]})
        .groupby(["row", "col"], sort=False)["s_m"]
        .median()
        .reset_index()
    )
    inside = (cells["s_m"] >= 0) & (cells["s_m"] <= r["length_m"])
    return cells[inside].sort_values("s_m").reset_index(drop=True)


def _reversals(steps: np.ndarray) -> int:
    big = np.where(np.abs(steps) > REVERSAL_M, np.sign(steps), 0.0)
    return int(((big[1:] * big[:-1]) < 0).sum())


def _spec(variant: str) -> tuple[float, dict]:
    spec = dict(SYNTHETIC_VARIANTS[variant])
    alpha = spec.pop("alpha", 0.5)
    spec["extra_config"] = {"dtmapout": MAP_STEP_S, **spec.get("extra_config", {})}
    return alpha, spec


def build_synthetic(root: Path, base: str, variant: str, cfg: dict) -> dict:
    import copy

    from phase2_advection_channel import build_case

    layout, q = BASES[base]
    alpha, spec = _spec(variant)
    dx_m = spec.pop("dx_m", None)
    if dx_m is not None:
        cfg = copy.deepcopy(cfg)
        cfg["sfincs"]["grid"]["resolution_m"] = dx_m
        nr = round(dx_m / SUBGRID_PIXEL_M)
        # hydromt needs an even pixel count per cell; halve the pixel if odd.
        cfg["sfincs"]["subgrid"]["nr_subgrid_pixels"] = nr if nr % 2 == 0 else 2 * nr
    if root.exists():
        shutil.rmtree(root)
    return build_case(root, layout, q, 0, alpha, cfg, **spec)


def run_synthetic(root: Path, threads: int) -> float:
    from src.sfincs_run import run_sfincs_subprocess

    t0 = time.perf_counter()
    run_sfincs_subprocess(
        site_path("sfincs_exe"), root, 3600, log, label=root.name, n_threads=threads
    )
    return time.perf_counter() - t0


def synthetic_metrics(
    root: Path, r: dict, base: str, variant: str, wall_s: float
) -> tuple[dict, pd.DataFrame]:
    from phase2_advection_channel import BED_SLOPE, RAMP_HR

    alpha, spec = _spec(variant)
    cells = _centreline_cells(r)
    with xr.open_dataset(root / "sfincs_map.nc") as ds:
        zs = ds["zs"].values[:, cells["row"], cells["col"]]
    hours = np.arange(zs.shape[0]) * MAP_STEP_S / 3600.0
    steps = np.diff(zs, axis=0)
    ramp_hr = spec.get("ramp_hr", RAMP_HR)
    after = hours[1:] > ramp_hr + 2.0
    upstream_raise = np.zeros(len(cells))
    for s_d, d in spec.get("bed_drops", ()):
        upstream_raise += np.where(cells["s_m"] < s_d, d, 0.0)
    level_normal = (
        BED_SLOPE * (r["length_m"] - cells["s_m"]) + r["h_n"] + upstream_raise
    )
    err = zs[-1] - level_normal.to_numpy()
    row = {
        "base": base,
        "variant": variant,
        "alpha": alpha,
        "max_step_after_ramp_m": float(np.nanmax(np.abs(steps[after]))),
        "reversals": sum(_reversals(steps[:, j]) for j in range(steps.shape[1])),
        "error_median_m": float(np.nanmedian(err)),
        "error_max_m": float(np.nanmax(np.abs(err))),
        "wall_s": wall_s,
    }
    row["unstable"] = bool(
        row["max_step_after_ramp_m"] > UNSTABLE_STEP_M or row["error_max_m"] > 2.0
    )
    mid = int(np.argmin(np.abs(cells["s_m"].to_numpy() - r["length_m"] / 2)))
    series = pd.DataFrame(
        {"hour": hours, "zs_mid_m": zs[:, mid] - level_normal.iloc[mid]}
    ).assign(base=base, variant=variant)
    return row, series


def synthetic(workers: int, threads: int) -> pd.DataFrame:
    cfg = load_config()
    run_dir = scratch_root() / "synthetic"
    jobs = [(b, v, run_dir / f"{b}_{v}") for b in BASES for v in SYNTHETIC_VARIANTS]
    # hydromt_sfincs_patch swaps a module global while writing: build serially.
    models = [build_synthetic(root, b, v, cfg) for b, v, root in jobs]
    with ThreadPoolExecutor(workers) as ex:
        walls = list(ex.map(lambda j: run_synthetic(j[2], threads), jobs))
    out = [
        synthetic_metrics(root, r, b, v, w)
        for (b, v, root), r, w in zip(jobs, models, walls)
    ]
    rows = [o[0] for o in out]
    df = write_summary(rows, analysis_path(None, __file__, "synthetic_summary.csv"))
    _plot_synthetic(df, pd.concat([o[1] for o in out]), git_hash())
    return df


def _plot_synthetic(df: pd.DataFrame, series: pd.DataFrame, tag: str) -> None:
    variants = list(SYNTHETIC_VARIANTS)
    fig, axes = plt.subplots(2, len(BASES), figsize=(15, 9), squeeze=False)
    for j, base in enumerate(BASES):
        d = df[df["base"] == base].set_index("variant").reindex(variants)
        ax = axes[0, j]
        y = np.arange(len(variants))
        colors = ["tab:red" if u else "tab:blue" for u in d["unstable"]]
        ax.barh(y, d["max_step_after_ramp_m"].clip(lower=1e-4), color=colors)
        ax.set_xscale("log")
        ax.axvline(UNSTABLE_STEP_M, color="k", ls=":", lw=0.8)
        ax.set_yticks(
            y,
            [
                f"{v} ({int(n)} rev, {w:.0f} s)"
                for v, n, w in zip(variants, d["reversals"], d["wall_s"])
            ],
            fontsize=7,
        )
        ax.invert_yaxis()
        ax.set_xlabel("largest 10-min level step after the ramp (m); red: unstable")
        ax.set_title(f"{base}", fontsize=9)
        ax.grid(alpha=0.3)
        ax = axes[1, j]
        for v in variants:
            s = series[(series["base"] == base) & (series["variant"] == v)]
            ax.plot(s["hour"], s["zs_mid_m"], lw=0.8, label=v)
        ax.set_ylim(-3, 6)
        ax.set_xlabel("hour")
        ax.set_ylabel("mid-channel level - normal-depth level (m)")
        ax.grid(alpha=0.3)
    axes[1, -1].legend(fontsize=6, ncol=2)
    fig.suptitle(
        f"phase2 2c synthetic trench, advection 0, one factor from alpha 0.5 | {tag}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(analysis_path(None, __file__, "synthetic.png"), dpi=120)
    plt.close(fig)


def _event_dirs(basin_id: str, scenario: str) -> tuple[Path, Path]:
    src = site_path("results_dir") / basin_id / "runs" / scenario / "sfincs"
    return src, scratch_root() / basin_id / scenario


def event(
    basin_id: str, scenario: str, name: str, settings: list[str], threads: int
) -> None:
    src, out = _event_dirs(basin_id, scenario)
    d = out / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in ("sfincs.bnd", "sfincs.bzs", "sfincs.dis", "sfincs.src"):
        if (src / f).exists():
            shutil.copy(src / f, d / f)
    inp = (src / "sfincs.inp").read_text()

    def relink(m: re.Match) -> str:
        target = (src / m.group(2).strip()).resolve()
        return f"{m.group(1)}= {Path(os.path.relpath(target, d)).as_posix()}"

    inp = re.sub(r"^(\w+\s+)= (\.\.[\\/].*)$", relink, inp, flags=re.M)
    for kv in settings:
        key, value = kv.split("=", 1)
        line = f"{key:<21}= {value}"
        if re.search(rf"^{key}\s+=", inp, flags=re.M):
            inp = re.sub(rf"^{key}\s+= .*$", line, inp, flags=re.M)
        else:
            inp = inp.rstrip("\n") + "\n" + line + "\n"
    (d / "sfincs.inp").write_text(inp)
    env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
    t0 = time.perf_counter()
    with open(d / "stdout.txt", "w") as f:
        code = subprocess.run(
            [str(site_path("sfincs_exe"))],
            cwd=d,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        ).returncode
    (d / "wall_s.txt").write_text(f"{time.perf_counter() - t0:.0f}\n")
    if code:
        raise SystemExit(f"{name}: SFINCS exit {code}, see {d / 'stdout.txt'}")


def _event_metrics(d: Path, inflow_nm: tuple[int, int]) -> tuple[dict, dict]:
    with xr.open_dataset(d / "sfincs_map.nc") as ds:
        zs = ds["zs"].values
        zb = ds["zb"].values
        zsmax = ds["zsmax"].max("timemax").values
    steps = np.diff(zs, axis=0)
    absmax = np.nanmax(np.abs(steps), axis=0)
    up = np.nanmax(steps, axis=0)
    down = -np.nanmin(steps, axis=0)
    n, m = inflow_nm
    return (
        {
            "max_hourly_step_m": float(np.nanmax(absmax)),
            "cells_step_gt_2m": int(np.nansum(absmax > STEP_FLAG_M)),
            "cells_reversing_gt_1m": int(np.nansum((up > 1.0) & (down > 1.0))),
            "inflow_max_hourly_step_m": float(np.nanmax(np.abs(steps[:, n, m]))),
            "inflow_zsmax_m": float(np.nanmax(zs[:, n, m])),
            "wet_cells": int(np.nansum(zsmax - zb > WET_M)),
        },
        {"zsmax": zsmax, "zb": zb, "inflow_zs": zs[:, n, m]},
    )


def event_compare(basin_id: str, scenario: str) -> pd.DataFrame:
    src, out = _event_dirs(basin_id, scenario)
    inp = dict(
        (k.strip(), v.strip())
        for k, v in (
            line.split("=", 1)
            for line in (src / "sfincs.inp").read_text().splitlines()
            if "=" in line
        )
    )
    x0, y0, dx = float(inp["x0"]), float(inp["y0"]), float(inp["dx"])
    xs, ys = np.loadtxt(src / "sfincs.src", usecols=(0, 1), ndmin=2)[0]
    inflow_nm = (int((ys - y0) // dx), int((xs - x0) // dx))
    names = [
        "pipeline",
        *sorted(p.name for p in out.iterdir() if (p / "sfincs_map.nc").exists()),
    ]
    rows, fields = [], {}
    for name in names:
        d = src if name == "pipeline" else out / name
        row, f = _event_metrics(d, inflow_nm)
        wall = d / "wall_s.txt"
        row = {
            "run": name,
            **row,
            "wall_s": float(wall.read_text()) if wall.exists() else np.nan,
        }
        rows.append(row)
        fields[name] = f
    ref = fields["pipeline"]
    for row in rows:
        f = fields[row["run"]]
        wet = (np.maximum(f["zsmax"], ref["zsmax"]) - ref["zb"]) > WET_M
        diff = (f["zsmax"] - ref["zsmax"])[wet]
        row["zsmax_diff_median_m"] = float(np.nanmedian(diff))
        row["zsmax_absdiff_p99_m"] = float(np.nanpercentile(np.abs(diff), 99))
        row["wet_cells_rel"] = row["wet_cells"] / rows[0]["wet_cells"]
    df = write_summary(
        rows, analysis_path(basin_id, __file__, f"event_{scenario}_summary.csv")
    )
    _plot_event(basin_id, scenario, fields, git_hash())
    return df


def _plot_event(basin_id: str, scenario: str, fields: dict, tag: str) -> None:
    others = [n for n in fields if n != "pipeline"]
    fig, axes = plt.subplots(
        1, 1 + len(others), figsize=(5 * (1 + len(others)), 5), squeeze=False
    )
    ax = axes[0, 0]
    for name, f in fields.items():
        ax.plot(f["inflow_zs"], lw=0.8, label=name)
    ax.set_xlabel("hour")
    ax.set_ylabel("level at the inflow cell (m)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ref = fields["pipeline"]
    for ax, name in zip(axes[0, 1:], others):
        diff = fields[name]["zsmax"] - ref["zsmax"]
        im = ax.imshow(diff, origin="lower", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        ax.set_title(f"zsmax {name} - pipeline", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.7, label="m")
    fig.suptitle(
        f"phase2 2c | {basin_id} | {scenario}: reruns against the pipeline run | {tag}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(analysis_path(basin_id, __file__, f"event_{scenario}.png"), dpi=110)
    plt.close(fig)


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("action", choices=["synthetic", "event", "event-compare"])
    parser.add_argument("basin_id", nargs="?", default="2433835")
    parser.add_argument("scenario", nargs="?", default="river_100_03SLR")
    parser.add_argument("--name")
    parser.add_argument("--set", nargs="*", default=[], dest="settings")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    sys.path.insert(0, str(REPO_ROOT / "workflow"))
    with pd.option_context("display.width", 250, "display.max_columns", None):
        if args.action == "synthetic":
            print(synthetic(args.workers, args.threads).round(3).to_string(index=False))
        elif args.action == "event":
            if not args.name:
                parser.error("event needs --name")
            event(args.basin_id, args.scenario, args.name, args.settings, args.threads)
        else:
            print(
                event_compare(args.basin_id, args.scenario)
                .round(3)
                .to_string(index=False)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
