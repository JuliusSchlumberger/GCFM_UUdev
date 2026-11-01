"""Phase 6.1: how long the walled calibration run needs to settle, and what settling changes.

Three inputs per basin: a long round-1 run (run length 3 x L^2/D) under
<scratch>/p61_long, the results with the walled run length
(<scratch>/phase1_baseline_snapshot) and with the settle run length
(results_dir). <scratch> is GCFM_PLAN_SCRATCH (default
<results_dir>/../scratch_plan).

Figure 1, one row per basin: at the river-bed profile nodes, the 12 h mean
water level of the long run minus its end level (median and 90 % band), with
the walled and settle run ends; the crests are set from the levels a
calibration run stops on. Figure 2, one row per basin: crest elevation
(walled vs settle) and the change in maximum flood depth per scenario.

Writes phase6_settle_time_{settle,crest_depth}.png/.csv to <results_dir>/figures/analysis/.

Usage: python analysis/phase6_settle_time.py [basin_id ...] [--scenarios coast_100_03SLR river_100_03SLR]
"""

import os
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import yaml
from _common import REPO_ROOT, analysis_path, script_parser, site_path
from check_river_bed import _git_hash
from peak_lag import DEFAULT_BASINS
from scipy.spatial import cKDTree

MEAN_WINDOW_HR = 12
TOL_M = 0.05


def scratch_root() -> Path:
    return Path(
        os.environ.get("GCFM_PLAN_SCRATCH")
        or site_path("results_dir").parent / "scratch_plan"
    )


def _inputs(root: Path, basin_id: str) -> Path:
    return root / basin_id / "preprocessing_inputs"


def ramp_plus_margin_hr() -> float:
    with open(REPO_ROOT / "config" / "config.yml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["river_processing"]["river_depth_modelling"]
    return float(cfg["ramp_hr"]) + float(cfg["calibration_margin_hr"])


def node_levels(root: Path, basin_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Return hours and the trailing 12 h mean water level at the profile nodes (time, node)."""
    inputs = _inputs(root, basin_id)
    nodes = pd.read_csv(
        inputs
        / "visuals"
        / "11_calibration"
        / "river_bed"
        / f"{basin_id}_river_bed_nodes.csv"
    )
    map_nc = inputs / "depth_crest_calibration" / "round1" / "sfincs_map.nc"
    with xr.open_dataset(map_nc) as ds:
        gx, gy = ds["x"].values, ds["y"].values
        ok = np.isfinite(gx)
        _, i = cKDTree(np.c_[gx[ok], gy[ok]]).query(nodes[["x", "y"]].to_numpy())
        n, m = np.unravel_index(np.flatnonzero(ok.ravel())[i], gx.shape)
        zs = ds["zs"].values[:, n, m]
        t = (ds["time"].values - ds["time"].values[0]) / np.timedelta64(1, "h")
    zs = zs[:, np.isfinite(zs).all(axis=0)]
    k = max(1, int(round(MEAN_WINDOW_HR / float(np.median(np.diff(t))))))
    c = np.cumsum(np.vstack([np.zeros((1, zs.shape[1])), zs]), axis=0)
    return t[k - 1 :], (c[k:] - c[:-k]) / k


def settle_basin(long_root: Path, after_root: Path, basin_id: str) -> dict:
    t, z = node_levels(long_root, basin_id)
    deficit = z[-1] - z
    off = np.abs(deficit) > TOL_M
    last_off = np.where(
        off.any(axis=0), t[off.shape[0] - 1 - np.argmax(off[::-1], axis=0)], 0.0
    )
    check = pd.read_csv(
        _inputs(after_root, basin_id)
        / "depth_crest_calibration"
        / "round1"
        / "steady_check.csv"
    ).iloc[0]
    run_walled = float(np.ceil(check["t_walled_hr"] + ramp_plus_margin_hr()))
    run_settle = float(check["run_hr"])
    row = {
        "basin_id": basin_id,
        "t_walled_hr": float(check["t_walled_hr"]),
        "t_settle_hr": float(check["t_settle_hr"]),
        "run_walled_hr": run_walled,
        "run_settle_hr": run_settle,
        "t_steady_node_p50_hr": float(np.percentile(last_off, 50)),
        "t_steady_node_p90_hr": float(np.percentile(last_off, 90)),
    }
    for name, h in (("walled", run_walled), ("settle", run_settle)):
        d = deficit[min(int(np.searchsorted(t, h)), len(t) - 1)]
        row[f"deficit_at_{name}_end_p50_m"] = float(np.median(d))
        row[f"deficit_at_{name}_end_p90_m"] = float(np.percentile(d, 90))
    row["_series"] = (t, deficit)
    return row


def crest_depth_basin(
    before_root: Path, after_root: Path, basin_id: str, scenarios: list[str]
) -> dict:
    weir = f"{basin_id}_coastal_protection_weir.gpkg"
    crest = {
        name: gpd.read_file(_inputs(root, basin_id) / "domain" / weir)[
            "elevation"
        ].to_numpy(float)
        for name, root in (("walled", before_root), ("settle", after_root))
    }
    row = {"basin_id": basin_id, "_crest": crest, "_ddepth": {}}
    for name in ("walled", "settle"):
        row[f"crest_{name}_p50_m"] = float(np.median(crest[name]))
        row[f"crest_{name}_p90_m"] = float(np.percentile(crest[name], 90))
    for sc in scenarios:
        metrics = {
            name: pd.read_csv(
                root / basin_id / "runs" / sc / "metrics" / "flood_metrics.csv"
            ).iloc[0]
            for name, root in (("walled", before_root), ("settle", after_root))
        }
        for col in ("flooded_area_km2", "urban_exposed_km2", "volume_km3"):
            for name in ("walled", "settle"):
                row[f"{sc}_{col}_{name}"] = float(metrics[name][col])
        depth = {}
        for name, root in (("walled", before_root), ("settle", after_root)):
            with rasterio.open(
                root / basin_id / "runs" / sc / "visuals" / "max_flood_depth.tif"
            ) as src:
                a = src.read(1, masked=True).filled(0.0).astype(float)
                extent = src.bounds
            depth[name] = np.nan_to_num(a)
        row["_ddepth"][sc] = (depth["settle"] - depth["walled"], extent)
    return row


def plot_settle(rows: list[dict], tag: str, path: Path) -> None:
    fig, axes = plt.subplots(1, len(rows), figsize=(7 * len(rows), 4.2), squeeze=False)
    for ax, r in zip(axes[0], rows):
        t, d = r["_series"]
        ax.fill_between(
            t,
            np.percentile(d, 10, axis=1),
            np.percentile(d, 90, axis=1),
            color="tab:blue",
            alpha=0.2,
            label="p10-p90 of nodes",
        )
        ax.plot(t, np.median(d, axis=1), color="tab:blue", lw=1.5, label="median node")
        for name, color in (("walled", "tab:red"), ("settle", "tab:green")):
            h = r[f"run_{name}_hr"]
            ax.axvline(
                h,
                color=color,
                ls="--",
                label=f"{name} run end {h:.0f} h: median {r[f'deficit_at_{name}_end_p50_m']:+.2f} m, "
                f"p90 {r[f'deficit_at_{name}_end_p90_m']:+.2f} m",
            )
        ax.axhline(TOL_M, color="0.5", lw=0.8, ls=":")
        ax.set_yscale("symlog", linthresh=0.05)
        ax.set_xlabel("hours from run start")
        ax.set_ylabel(f"end level - {MEAN_WINDOW_HR} h mean level (m)")
        ax.set_title(
            f"{r['basin_id']}: T_walled {r['t_walled_hr']:.1f} h, L^2/D {r['t_settle_hr']:.1f} h, "
            f"nodes steady p50/p90 {r['t_steady_node_p50_hr']:.0f}/{r['t_steady_node_p90_hr']:.0f} h",
            fontsize=9,
        )
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"phase 6.1: calibration run settling at the river-bed nodes | {tag}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_crest_depth(
    rows: list[dict], scenarios: list[str], tag: str, path: Path
) -> None:
    ncol = 1 + len(scenarios)
    fig, axes = plt.subplots(
        len(rows), ncol, figsize=(5.2 * ncol, 4.6 * len(rows)), squeeze=False
    )
    for ax_row, r in zip(axes, rows):
        ax = ax_row[0]
        lo = min(np.percentile(v, 1) for v in r["_crest"].values())
        hi = max(np.percentile(v, 99) for v in r["_crest"].values())
        bins = np.linspace(lo, hi, 60)
        for name, color in (("walled", "tab:red"), ("settle", "tab:green")):
            ax.hist(
                r["_crest"][name],
                bins=bins,
                histtype="step",
                color=color,
                label=f"{name}: median {r[f'crest_{name}_p50_m']:.2f} m",
            )
        ax.set_xlabel("weir crest elevation (m)")
        ax.set_ylabel("weir segments")
        ax.set_title(f"{r['basin_id']}: weir crests", fontsize=9)
        ax.legend(fontsize=7)
        for ax, sc in zip(ax_row[1:], scenarios):
            dd, b = r["_ddepth"][sc]
            im = ax.imshow(
                np.where(dd == 0, np.nan, dd),
                cmap="RdBu_r",
                vmin=-1,
                vmax=1,
                extent=(b.left, b.right, b.bottom, b.top),
            )
            ax.set_title(
                f"{sc}: max depth settle - walled\nflooded "
                f"{r[f'{sc}_flooded_area_km2_walled']:.0f} -> {r[f'{sc}_flooded_area_km2_settle']:.0f} km2",
                fontsize=9,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.7, label="m")
    fig.suptitle(
        f"phase 6.1: crests and flood depth, walled vs settle run length | {tag}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    )


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("basin_ids", nargs="*", default=DEFAULT_BASINS)
    parser.add_argument(
        "--scenarios", nargs="+", default=["coast_100_03SLR", "river_100_03SLR"]
    )
    args = parser.parse_args()
    scratch = scratch_root()
    after = site_path("results_dir")
    before = scratch / "phase1_baseline_snapshot"
    tag = _git_hash()
    basins = [str(b) for b in args.basin_ids]

    settle = [settle_basin(scratch / "p61_long", after, b) for b in basins]
    plot_settle(settle, tag, analysis_path(None, __file__, "settle.png"))
    crest = [crest_depth_basin(before, after, b, args.scenarios) for b in basins]
    plot_crest_depth(
        crest, args.scenarios, tag, analysis_path(None, __file__, "crest_depth.png")
    )

    with pd.option_context("display.width", 250, "display.max_columns", None):
        for name, rows in (("settle", settle), ("crest_depth", crest)):
            df = _table(rows)
            df.to_csv(
                analysis_path(None, __file__, f"{name}.csv"),
                index=False,
                float_format="%.3f",
            )
            print(df.round(2).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
