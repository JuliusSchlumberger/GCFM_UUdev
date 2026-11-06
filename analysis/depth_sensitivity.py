"""Flood extent, volume and urban exposure of event scenarios with the burned river depth scaled.

`run`: for one basin and scale, copies the basin's preprocessing_inputs to
<runs_root>/depth_<scale>/ and runs Snakemake there (GCFM_RESULTS_DIR) with
river_depth_modelling.depth_scale = scale and the bifurcation check reporting
only (a --configfile overlay) and --forcerun modelled_depth_estimation, so rule
11 (burn and crest calibration) and the build (skeleton, spin-up, event runs,
metrics) run on the scaled bed. Production results are not touched.
`summary`: each scale's flood metrics against the production run in
<results_dir>, per basin and scenario.
`figure`: the summary as one figure across basins and scenarios.

runs_root defaults to <results_dir>/../depth_sensitivity_runs. The summary and
figure go to <results_dir>/figures/analysis/depth_sensitivity/.

Usage:
    python analysis/depth_sensitivity.py run 2433835 --scale 0.7 --scenarios coast_100_03SLR river_100_03SLR
    python analysis/depth_sensitivity.py summary 2433835 3279946 --scales 0.7 1.3 --scenarios coast_100_03SLR river_100_03SLR
    python analysis/depth_sensitivity.py figure
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _basin_inputs import git_hash
from _common import REPO_ROOT, script_parser, site_path

METRICS = {
    "flooded_area_km2": "flooded area",
    "volume_m3": "flood volume",
    "urban_exposed_km2": "urban area exposed",
}


def out_dir() -> Path:
    d = site_path("results_dir") / "figures" / "analysis" / "depth_sensitivity"
    d.mkdir(parents=True, exist_ok=True)
    return d


def runs_root(path: str | None) -> Path:
    return (
        Path(path)
        if path
        else site_path("results_dir").parent / "depth_sensitivity_runs"
    )


def scale_root(root: Path, scale: float) -> Path:
    return root / f"depth_{scale:g}"


def run(
    basin_id: str, scale: float, scenarios: list[str], cores: int, root: Path
) -> int:
    dest = scale_root(root, scale)
    src = site_path("results_dir") / basin_id / "preprocessing_inputs"
    if (dest / basin_id).exists():
        shutil.rmtree(dest / basin_id)
    shutil.copytree(src, dest / basin_id / "preprocessing_inputs")
    overlay = dest / "depth_scale.yml"
    # A scaled bed can shift the bifurcation split: report it, do not stop.
    overlay.write_text(
        "river_processing:\n  river_depth_modelling:\n"
        f"    depth_scale: {scale}\n"
        "    bifurcation_check: {max_share_error: 1.0}\n"
    )
    env = {**os.environ, "GCFM_RESULTS_DIR": dest.as_posix()}
    cmd = [
        "snakemake",
        "build",
        "--cores",
        str(cores),
        # the calling suite holds this working directory's lock
        "--nolock",
        "--configfile",
        "config/config.yml",
        str(overlay),
        "--forcerun",
        "modelled_depth_estimation",
        "--config",
        f"target_basins=[{basin_id}]",
        f"target_scenarios={scenarios!r}",
    ]
    with open(dest / f"snakemake_{basin_id}.log", "w") as log:
        return subprocess.run(
            cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
        ).returncode


def metrics(results_root: Path, basin_id: str, scenario: str) -> dict:
    p = results_root / basin_id / "runs" / scenario / "metrics" / "flood_metrics.csv"
    if not p.exists():
        raise FileNotFoundError(f"no flood metrics for {basin_id} {scenario}: {p}")
    return pd.read_csv(p).iloc[0].to_dict()


def summary(
    basin_ids: list[str], scales: list[float], scenarios: list[str], root: Path
) -> pd.DataFrame:
    rows = []
    for basin_id in basin_ids:
        for scenario in scenarios:
            base = metrics(site_path("results_dir"), basin_id, scenario)
            for scale in [1.0, *scales]:
                m = (
                    base
                    if scale == 1.0
                    else metrics(scale_root(root, scale), basin_id, scenario)
                )
                row = {"basin_id": basin_id, "scenario": scenario, "depth_scale": scale}
                for key in METRICS:
                    row[key] = float(m[key])
                    row[f"{key}_rel"] = (
                        float(m[key]) / float(base[key]) if base[key] else np.nan
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def figure(table: pd.DataFrame, path: Path) -> None:
    groups = list(table.groupby(["basin_id", "scenario"], sort=False).groups)
    scales = sorted(s for s in table["depth_scale"].unique() if s != 1.0)
    markers = {s: m for s, m in zip(scales, ["v", "^", "s", "D"])}
    fig, axes = plt.subplots(
        1, len(METRICS), figsize=(5 * len(METRICS), 4.2), sharey=True
    )
    for ax, (key, label) in zip(axes, METRICS.items()):
        for i, (basin, scen) in enumerate(groups):
            g = table[(table["basin_id"] == basin) & (table["scenario"] == scen)]
            rel = g.set_index("depth_scale")[f"{key}_rel"]
            ax.plot([i, i], [rel.min(), rel.max()], color="0.6", lw=3, zorder=1)
            for s in scales:
                ax.plot(
                    i,
                    rel[s],
                    markers[s],
                    color="tab:red" if s > 1 else "tab:blue",
                    ms=8,
                    zorder=2,
                    label=f"depth x {s:g}" if i == 0 else None,
                )
            base = g.loc[g["depth_scale"] == 1.0, key].iloc[0]
            text = (
                f"{base / 1e9:.2f} km3"
                if key == "volume_m3"
                else f"{base:.0f} km2"
                if base >= 10
                else f"{base:.1f} km2"
            )
            ax.annotate(
                text,
                (i, 1.0),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=7,
                va="center",
            )
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{b}\n{s}" for b, s in groups], fontsize=7)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, 2)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("relative to the production bed (x 1)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "depth_sensitivity: burned river depth scaled; baseline value labelled | "
        + git_hash(),
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("action", choices=["run", "summary", "figure"])
    parser.add_argument("basin_ids", nargs="*", default=[])
    parser.add_argument("--scale", type=float, help="run: the depth scale")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.7, 1.3])
    parser.add_argument("--scenarios", nargs="+", default=["default"])
    parser.add_argument("--cores", type=int, default=14)
    parser.add_argument("--runs-root", default=None)
    args = parser.parse_args()
    root = runs_root(args.runs_root)
    csv = out_dir() / "summary.csv"
    if args.action == "run":
        if args.scale is None or len(args.basin_ids) != 1:
            parser.error("run takes one basin_id and --scale")
        return run(args.basin_ids[0], args.scale, args.scenarios, args.cores, root)
    if args.action == "summary":
        table = summary(args.basin_ids, args.scales, args.scenarios, root)
        table.to_csv(csv, index=False, float_format="%.4g")
        print(table.round(3).to_string(index=False))
        return 0
    figure(
        pd.read_csv(csv, dtype={"basin_id": str}), out_dir() / "depth_sensitivity.png"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
