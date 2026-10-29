"""Diagnose the river surface, slope, depth and bed of rule 10 and rule 11 round 0 per basin.

Re-runs rule 10's surface (``src.water_surface.build_water_surface``) and the
round-0 bed (``src.river_bed.estimate_river_bed_from_files``) on a basin's rule
08-09 outputs with the current code and config, no SFINCS, writing nothing
under results_dir. Writes per basin to the output directory: metrics JSON,
node table, bed jumps with their cause, SWOT slope ratios per reach, the
surface GPKG, and one profile figure per forcing-point mainstem. The rule-10
surface on disk is kept as ``surface10_m`` for before/after comparison.

Usage:
    python analysis/check_river_bed.py [basin_id ...] [--out-dir DIR]
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from _common import DEFAULT_BASIN_ID, REPO_ROOT, script_parser, site_path
from _upstream_profile import (
    forward_levels,
    mainstem_profile,
    scaled_discharge,
    surface_mismatch,
)

JUMP_M = 1.0
LEVEL_TOL_M = 0.1


def _config() -> dict:
    import yaml

    return yaml.safe_load(
        (REPO_ROOT / "config" / "config.yml").read_text(encoding="utf-8")
    )


def _git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def _sample(path: Path, crs, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    import rasterio
    from pyproj import Transformer

    with rasterio.open(path) as src:
        xs, ys = Transformer.from_crs(crs, src.crs, always_xy=True).transform(x, y)
        v = np.array([s[0] for s in src.sample(zip(xs, ys))], dtype=float)
        if src.nodata is not None:
            v[v == src.nodata] = np.nan
    return v


def run_round0(basin_id: str, cfg: dict, out_dir: Path):
    """Return the round-0 bed, rivers, forcing table, calm sea level and paths of a basin."""
    import geopandas as gpd
    import rasterio
    import xarray as xr
    from src.domain import load_domain
    from src.grid import load_grid_def
    from src.io import catalogue_entry, load_catalogue, raw_input_path
    from src.river_bed import estimate_river_bed_from_files
    from src.river_network import normalize_reach_id, outflow_crossing_mask
    from src.surge import read_calm_sea_level_m
    from src.water_surface import (
        build_water_surface,
        node_sampler,
        sample_centrelines,
        write_water_surface,
    )

    prep = site_path("results_dir") / basin_id / "preprocessing_inputs"
    dom, frc = prep / "domain", prep / "forcing"
    rp = cfg["river_processing"]
    rdm = rp["river_depth_modelling"]
    cat = load_catalogue(REPO_ROOT / cfg["data_catalogue"])
    sword = [
        raw_input_path(cat, "river_network").format(region=r)
        for r in catalogue_entry(cat, "river_network")["regions"]
    ]
    crs = load_grid_def(dom / f"{basin_id}_sfincs_grid.json")["crs"]
    wgs84_bounds, _, _ = load_domain(
        dom / "domain_bbox.json", dom / f"{basin_id}_domain.gpkg"
    )
    with xr.open_dataset(frc / "surge_forcing.nc", decode_times=False) as ds:
        sea = read_calm_sea_level_m(ds)
    rivers = gpd.read_file(dom / f"{basin_id}_river_network_clean.gpkg")
    paths = {
        "water_surface_disk": dom / f"{basin_id}_river_water_surface.gpkg",
        "water_surface": out_dir / f"{basin_id}_river_water_surface.gpkg",
        "elevation_merged": dom / f"{basin_id}_elevation_merged.tif",
    }
    merged, landuse = paths["elevation_merged"], dom / f"{basin_id}_landuse.tif"
    with rasterio.open(merged) as src:
        dem_crs = src.crs
    surface, _ = build_water_surface(
        rivers.to_crs(dem_crs),
        lambda lines: sample_centrelines(
            lines, merged, landuse, int(cfg["landuse"]["water_code"])
        ),
        rp["water_surface"],
        sea_level_m=sea,
        sample_nodes=node_sampler(
            merged, dom / f"{basin_id}_landuse_channels.tif", dem_crs
        ),
        gebco_floor_m=sea - float(cfg["terrain"]["gebco_max_depth_m"]),
    )
    write_water_surface(surface, paths["water_surface"], dem_crs)
    bed, _q, _seeds, _q_mean = estimate_river_bed_from_files(
        rivers,
        frc / "river_forcing.nc",
        paths["water_surface"],
        sword,
        wgs84_bounds,
        crs,
        sea,
        cfg["terrain"]["gebco_min_depth_m"],
        rp["channel_manning_n"],
        rp["water_surface"],
        rdm["bed"],
    )
    with xr.open_dataset(frc / "river_forcing.nc", decode_times=False) as ds:
        active = ds["has_glofas"].values.astype(bool) & ~outflow_crossing_mask(
            ds, rivers
        )
        forcing = pd.DataFrame(
            {
                "reach_id": [
                    normalize_reach_id(r) for r in ds["inside_reach_id"].values[active]
                ],
                "q_mean_m3s": ds["mean_discharge_m3s"].values[active].astype(float),
                "q_bf_m3s": ds["bankfull_discharge_m3s"].values[active].astype(float),
            }
        )
    return bed, rivers, forcing, sea, crs, paths


def node_table(bed, sea: float, crs, paths: dict) -> pd.DataFrame:
    """Return rule 11's node table plus the rule-10 surface on disk, the DEM and per-node flags."""
    from scipy.spatial import cKDTree
    from src.water_surface import read_water_surface

    t = bed.node_table()
    t["reach_id"] = t["reach_id"].astype(str)
    ws, _ = read_water_surface(paths["water_surface_disk"])
    _, k = cKDTree(np.c_[ws.geometry.x, ws.geometry.y]).query(np.c_[t["x"], t["y"]])
    t["surface10_m"] = ws["level_m"].to_numpy()[k]
    t["dem_m"] = _sample(
        paths["elevation_merged"], crs, t["x"].to_numpy(), t["y"].to_numpy()
    )
    t["below_sea"] = t["level_m"] < sea - 1e-6
    t["bed_above_dem"] = t["bed_m"] > t["dem_m"]
    return t


def bed_jumps(bed, t: pd.DataFrame) -> pd.DataFrame:
    """Return node pairs along each reach where the bed changes by more than JUMP_M, with the cause."""
    rows = []
    for r in bed.net.reach_ids:
        ids = np.asarray(bed.net.reach_nodes[r])
        for a, b in zip(ids[:-1], ids[1:]):
            dz = t.at[b, "bed_m"] - t.at[a, "bed_m"]
            if abs(dz) <= JUMP_M:
                continue
            cause = []
            if max(t.at[a, "depth_slope"], t.at[b, "depth_slope"]) > 2 * min(
                t.at[a, "depth_slope"], t.at[b, "depth_slope"]
            ):
                cause.append("slope")
            if max(t.at[a, "width_m"], t.at[b, "width_m"]) > 1.2 * min(
                t.at[a, "width_m"], t.at[b, "width_m"]
            ):
                cause.append("width")
            if t.at[a, "bathymetry"] != t.at[b, "bathymetry"]:
                cause.append("bathymetry")
            if abs(t.at[b, "level_m"] - t.at[a, "level_m"]) > 0.5:
                cause.append("level")
            rows.append(
                {
                    "reach_id": r,
                    "node_up": a,
                    "node_dn": b,
                    "x": t.at[b, "x"],
                    "y": t.at[b, "y"],
                    "dist_to_sea_km": t.at[b, "dist_to_sea_m"] / 1000,
                    "dbed_m": dz,
                    "cause": "+".join(cause) or "other",
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "reach_id",
            "node_up",
            "node_dn",
            "x",
            "y",
            "dist_to_sea_km",
            "dbed_m",
            "cause",
        ],
    )


def swot_ratios(bed, rivers, t: pd.DataFrame) -> pd.DataFrame:
    """Return per reach with a reliable SWOT slope the ratio of the depth, surface and on-disk rule-10 slopes to it."""
    from src.river_network import normalize_reach_id

    if not {"slope_obs_p50", "slope_obs_reliable"} <= set(rivers.columns):
        return pd.DataFrame()
    sw = rivers.assign(reach_id=rivers["reach_id"].map(normalize_reach_id)).set_index(
        "reach_id"
    )
    rows = []
    for r in bed.net.reach_ids:
        s, ok = sw.at[r, "slope_obs_p50"], sw.at[r, "slope_obs_reliable"]
        ids = np.asarray(bed.net.reach_nodes[r])
        ch = np.asarray(bed.net.reach_chainage[r])
        if not (ok == 1 and np.isfinite(s) and s > 0) or len(ids) < 3:
            continue
        s10 = -np.polyfit(ch, t.loc[ids, "surface10_m"].to_numpy(), 1)[0]
        rows.append(
            {
                "reach_id": r,
                "swot_slope": s,
                "ratio_depth_slope": float(np.median(t.loc[ids, "depth_slope"])) / s,
                "ratio_surface_slope": float(np.median(t.loc[ids, "slope"])) / s,
                "ratio_surface10_disk_slope": s10 / s,
            }
        )
    return pd.DataFrame(rows)


def metrics(
    bed,
    t: pd.DataFrame,
    jumps: pd.DataFrame,
    swot: pd.DataFrame,
    mismatch: pd.DataFrame,
) -> dict:
    from src.river_bed import forward_deviation_summary

    d2 = [
        np.diff(t.loc[np.asarray(bed.net.reach_nodes[r]), "bed_m"].to_numpy(), 2)
        for r in bed.net.reach_ids
    ]
    d2 = (
        np.concatenate([v for v in d2 if v.size])
        if any(v.size for v in d2)
        else np.array([np.nan])
    )
    inf = t["informative"].astype(bool)
    out = {
        "nodes": int(len(t)),
        "bed_roughness_m": float(np.nanstd(d2)),
        "bed_jumps_gt_1m": int(len(jumps)),
        "bed_jump_causes": jumps["cause"].value_counts().to_dict()
        if len(jumps)
        else {},
        "level_minus_fathom_informative_m": {
            "median": float(np.median((t["level_m"] - t["fathom_level_m"])[inf]))
            if inf.any()
            else None,
            "p90_abs": float(
                np.percentile(np.abs(t["level_m"] - t["fathom_level_m"])[inf], 90)
            )
            if inf.any()
            else None,
        },
        "level_minus_surface10_disk_m": {
            "median": float(np.median(t["level_m"] - t["surface10_m"])),
            "p5": float(np.percentile(t["level_m"] - t["surface10_m"], 5)),
            "p95": float(np.percentile(t["level_m"] - t["surface10_m"], 95)),
        },
        "depth_slope_p10_p90": [
            float(v) for v in np.percentile(t["depth_slope"], [10, 90])
        ],
        "share_on_slope_floor": float(t["on_floor"].mean()),
        "nodes_below_mouth_level": int(t["below_sea"].sum()),
        "share_bed_above_dem": float(np.nanmean(t["bed_above_dem"])),
        "swot_reaches": int(len(swot)),
    }
    out["network_forward"] = forward_deviation_summary(bed)
    for col in (
        "ratio_depth_slope",
        "ratio_surface_slope",
        "ratio_surface10_disk_slope",
    ):
        if len(swot):
            r = swot[col].to_numpy()
            out[f"swot_{col}"] = {
                "median": float(np.median(r)),
                "share_within_x2": float(np.mean((r > 0.5) & (r < 2))),
            }
    for _, m in mismatch.iterrows():
        out[f"forward_misfit_{m['discharge']}"] = {
            k: float(m[k])
            for k in (
                "mean_m",
                "p10_m",
                "p90_m",
                "max_abs_m",
                "at_forcing_point_m",
                "share_within_tol",
            )
        }
    return out


def plot_mainstem(
    prof: pd.DataFrame,
    fwd: dict,
    sea: float,
    swot: pd.DataFrame,
    title: str,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = prof["dist_from_mouth_m"] / 1000
    fig, ax = plt.subplots(
        3, 1, figsize=(11, 10), sharex=True, height_ratios=[3, 2, 1.5]
    )
    ax[0].plot(
        x, prof["fathom_level_m"], ".", color="0.6", ms=3, label="FathomDEM median"
    )
    ax[0].plot(
        x,
        prof["surface10_m"],
        "-",
        color="tab:cyan",
        label="rule-10 surface on disk (before)",
    )
    ax[0].plot(
        x,
        prof["level_m"],
        "-",
        color="tab:blue",
        lw=2,
        label="surface (rule 10, current code)",
    )
    ax[0].plot(
        x,
        prof["forward_level_m"],
        ":",
        color="tab:orange",
        lw=1.5,
        label="forward, network, design Q",
    )
    ax[0].plot(x, prof["bed_m"], "-", color="k", label="bed")
    ax[0].plot(x, prof["dem_m"], "-", color="tab:brown", lw=0.6, label="DEM (merged)")
    for label, eta in fwd.items():
        ax[0].plot(x, eta, "--", lw=1, label=f"forward {label}, calm sea")
    ax[0].axhline(sea, color="tab:red", lw=0.8, label="calm sea level")
    ax[0].set_ylabel("m (model datum)")
    ax[0].legend(fontsize=7, ncol=2)
    ax[1].semilogy(
        x, np.clip(prof["slope"], 1e-7, None), color="tab:blue", label="surface slope"
    )
    ax[1].semilogy(
        x, prof["depth_slope"], color="k", lw=0.8, label="depth slope (floored)"
    )
    s10 = np.clip(
        np.gradient(
            prof["surface10_m"].to_numpy(), prof["dist_from_mouth_m"].to_numpy()
        ),
        1e-7,
        None,
    )
    ax[1].semilogy(x, s10, color="tab:cyan", lw=0.8, label="rule-10 surface gradient")
    if len(swot):
        s = prof["reach_id"].map(swot.set_index("reach_id")["swot_slope"])
        ax[1].semilogy(x, s, "o", color="tab:green", ms=3, label="SWOT (reliable)")
    ax[1].set_ylabel("slope (m/m)")
    ax[1].legend(fontsize=7)
    ax[2].plot(x, prof["width_m"], color="tab:purple", label="width")
    ax2 = ax[2].twinx()
    ax2.plot(x, prof["level_m"] - prof["bed_m"], color="k", lw=0.8, label="depth")
    ax[2].set_ylabel("width (m)")
    ax2.set_ylabel("depth (m)")
    ax[2].set_xlabel("distance from mouth (km)")
    ax[2].invert_xaxis()
    fig.suptitle(title)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def check_basin(basin_id: str, out_dir: Path, cfg: dict) -> dict:
    from src.river_network import build_downstream_adjacency

    bed, rivers, forcing, sea, crs, paths = run_round0(basin_id, cfg, out_dir)
    t = node_table(bed, sea, crs, paths)
    jumps = bed_jumps(bed, t)
    swot = swot_ratios(bed, rivers, t)
    downstream = build_downstream_adjacency(rivers)
    n = float(cfg["river_processing"]["channel_manning_n"])
    design = cfg["river_processing"]["river_depth_modelling"]["bed"]["design_discharge"]
    tag = _git_hash()
    mism = []
    for fp in forcing.drop_duplicates("reach_id").itertuples():
        if fp.reach_id not in set(t["reach_id"]):
            continue
        prof = mainstem_profile(t, downstream, fp.reach_id)
        prof = prof.merge(
            t[["node", "depth_slope", "surface10_m", "dem_m", "forward_level_m"]],
            on="node",
            how="left",
        )
        q_design = fp.q_bf_m3s if design == "bankfull" else fp.q_mean_m3s
        qs = {f"Q_design ({design})": q_design, "Q_mean": fp.q_mean_m3s}
        mm = surface_mismatch(prof, qs, sea, n, tol_m=LEVEL_TOL_M).assign(
            forcing_reach=fp.reach_id
        )
        mism.append(mm)
        fwd = {
            k: forward_levels(prof, scaled_discharge(prof, q), sea, n)
            for k, q in qs.items()
        }
        plot_mainstem(
            prof,
            fwd,
            sea,
            swot,
            f"river bed check {basin_id}, mainstem of {fp.reach_id} ({tag})",
            out_dir / f"{basin_id}_mainstem_{fp.reach_id}.png",
        )
    mismatch = pd.concat(mism, ignore_index=True) if mism else pd.DataFrame()
    agg = (
        mismatch.groupby("discharge")[
            [
                "mean_m",
                "p10_m",
                "p90_m",
                "max_abs_m",
                "at_forcing_point_m",
                "share_within_tol",
            ]
        ]
        .median()
        .reset_index()
        if len(mismatch)
        else mismatch
    )
    m = metrics(bed, t, jumps, swot, agg)
    m.update(
        {
            "basin_id": basin_id,
            "git": tag,
            "design_discharge": design,
            "forcing_points": int(len(mism)),
        }
    )
    t.to_csv(out_dir / f"{basin_id}_nodes.csv", index=False)
    jumps.to_csv(out_dir / f"{basin_id}_jumps.csv", index=False)
    swot.to_csv(out_dir / f"{basin_id}_swot.csv", index=False)
    mismatch.to_csv(out_dir / f"{basin_id}_forward_misfit.csv", index=False)
    (out_dir / f"{basin_id}_metrics.json").write_text(
        json.dumps(m, indent=1), encoding="utf-8"
    )
    return m


def main() -> None:
    parser = script_parser(__doc__)
    parser.add_argument(
        "basin_ids", nargs="*", default=[DEFAULT_BASIN_ID], help="basin ids"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "figs" / "river_bed_check"
    )
    args = parser.parse_args()
    sys.path.insert(0, str(REPO_ROOT / "workflow"))
    from src.log import setup_logging

    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(args.out_dir / "check_river_bed.log"))
    cfg = _config()
    rows = []
    for b in args.basin_ids:
        m = check_basin(str(b), args.out_dir, cfg)
        rows.append(
            {k: v for k, v in m.items() if not isinstance(v, dict | list)}
            | {
                f"{k}.{j}": w
                for k, v in m.items()
                if isinstance(v, dict)
                for j, w in v.items()
                if not isinstance(w, dict)
            }
        )
    summary = pd.DataFrame(rows).set_index("basin_id").T
    summary.to_csv(args.out_dir / "summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
