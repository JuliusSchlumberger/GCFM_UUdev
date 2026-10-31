"""Phase 1.8: SWOT coverage in SWORD v17c and the datum of its water-surface elevations.

Coverage: SWORD nodes of the basin's cleaned network with n_obs >= 10 and a
finite wse_obs_p50, per reach and per basin; reaches with slope_obs_reliable.
Datum: on nodes upstream of the upstream-boundary-check influence end
(largest x_inf_km of the basin's reference rows in
figs/upstream_boundary_check/<basin>_forcing_points.csv, distance to sea from
the rule-11 node table), the median of wse_obs_p50 - static wse (MERIT,
EGM2008) and of wse_obs_p50 - rule-10 level, the latter both on GOCO06s and
converted to EGM2008 with src.raster.compute_geoid_offset_arr (level_EGM2008 =
level_GOCO06s - offset, as rule 06 adds the offset to FathomDEM). SWORD nodes
are matched to the nearest network node within 500 m.

Writes per basin 90_phase1_swot_coverage_reaches.csv to
<results_dir>/<basin>/figures/analysis/ and phase1_swot_coverage_summary.csv to
<results_dir>/figures/analysis/.

Usage: python analysis/phase1_swot_coverage.py [basin_id ...]
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from _common import REPO_ROOT, analysis_path, script_parser
from _phase1 import PHASE1_BASINS, basin_paths, load_config, pct, write_summary
from pyproj import Transformer
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)
MIN_OBS = 10
MATCH_M = 500.0


def _sword(
    cfg: dict, layer: str, bounds, reach_ids: set[int], columns: list[str]
) -> pd.DataFrame:
    from src.io import catalogue_entry, load_catalogue, raw_input_path

    cat = load_catalogue(REPO_ROOT / cfg["data_catalogue"])
    parts = []
    for region in catalogue_entry(cat, "river_network")["regions"]:
        path = raw_input_path(cat, "river_network").format(region=region)
        df = gpd.read_file(path, layer=layer, bbox=bounds)
        if len(df):
            parts.append(
                df[df["reach_id"].astype("int64").isin(reach_ids)][
                    columns + ["geometry"]
                ]
            )
    return (
        pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    )


def influence_end_km(basin_id: str) -> float:
    path = (
        REPO_ROOT
        / "figs"
        / "upstream_boundary_check"
        / f"{basin_id}_forcing_points.csv"
    )
    if not path.exists():
        return float("nan")
    t = pd.read_csv(path)
    ref = t[t["reference"].astype(bool)] if "reference" in t else t
    return float((ref if len(ref) else t)["x_inf_km"].max())


def geoid_offset_at(cfg: dict, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    import rasterio
    from src.io import load_catalogue, raw_input_path
    from src.raster import compute_geoid_offset_arr

    cat = load_catalogue(REPO_ROOT / cfg["data_catalogue"])
    arr, transform, crs = compute_geoid_offset_arr(
        raw_input_path(cat, "goco06s"), raw_input_path(cat, "egm2008_geoid")
    )
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    rows, cols = rasterio.transform.rowcol(transform, x, y)
    rows, cols = np.clip(rows, 0, arr.shape[0] - 1), np.clip(cols, 0, arr.shape[1] - 1)
    return arr[rows, cols].astype(float)


def coverage_basin(basin_id: str, cfg: dict) -> dict:
    from src.domain import load_domain
    from src.grid import load_grid_def

    p = basin_paths(basin_id)
    rivers = gpd.read_file(p["rivers_clean"])
    reach_ids = set(rivers["reach_id"].astype("int64"))
    bounds, _, _ = load_domain(p["domain_bbox"], p["domain_gpkg"])
    nodes = _sword(
        cfg,
        "nodes",
        bounds,
        reach_ids,
        ["node_id", "reach_id", "wse", "n_obs", "wse_obs_p50"],
    )
    reaches = _sword(
        cfg,
        "reaches",
        bounds,
        reach_ids,
        ["reach_id", "slope_obs_reliable", "slope_obs_p50"],
    )
    nodes["observed"] = (
        pd.to_numeric(nodes["n_obs"], errors="coerce") >= MIN_OBS
    ) & np.isfinite(pd.to_numeric(nodes["wse_obs_p50"], errors="coerce"))
    per_reach = (
        nodes.groupby("reach_id")
        .agg(n_nodes=("observed", "size"), share_observed=("observed", "mean"))
        .reset_index()
    )
    rel = reaches.assign(
        reliable=reaches["slope_obs_reliable"]
        .astype(str)
        .str.lower()
        .isin(["1", "true"])
    )
    per_reach = per_reach.merge(
        rel[["reach_id", "reliable", "slope_obs_p50"]], on="reach_id", how="left"
    )
    per_reach.to_csv(
        analysis_path(basin_id, __file__, "reaches.csv"),
        index=False,
        float_format="%.6g",
    )

    # datum, on observed nodes upstream of the influence end
    t = pd.read_csv(p["node_table"])
    crs = load_grid_def(p["sfincs_grid"])["crs"]
    lon, lat = nodes.geometry.x.to_numpy(), nodes.geometry.y.to_numpy()
    nx, ny = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    dist, idx = cKDTree(t[["x", "y"]].to_numpy()).query(np.column_stack([nx, ny]))
    x_inf = influence_end_km(basin_id)
    up = (
        (dist <= MATCH_M)
        & (t["dist_to_sea_m"].to_numpy()[idx] > 1000.0 * x_inf)
        & nodes["observed"].to_numpy()
    )
    wse_obs = pd.to_numeric(nodes["wse_obs_p50"], errors="coerce").to_numpy(float)
    level_goco = t["level_m"].to_numpy(float)[idx]
    offset = geoid_offset_at(cfg, lon, lat)
    return {
        "basin_id": basin_id,
        "n_sword_nodes": int(len(nodes)),
        "share_nodes_observed": float(nodes["observed"].mean())
        if len(nodes)
        else np.nan,
        "n_reaches": int(len(per_reach)),
        "share_reaches_observed_any": float((per_reach["share_observed"] > 0).mean())
        if len(per_reach)
        else np.nan,
        "n_reaches_slope_reliable": int(rel["reliable"].sum()),
        "influence_end_km": x_inf,
        "n_datum_nodes": int(up.sum()),
        "median_obs_minus_static_wse_m": pct(
            (wse_obs - pd.to_numeric(nodes["wse"], errors="coerce").to_numpy(float))[
                up
            ],
            50,
        ),
        "median_obs_minus_rule10_goco_m": pct((wse_obs - level_goco)[up], 50),
        "median_obs_minus_rule10_egm_m": pct((wse_obs - (level_goco - offset))[up], 50),
        "median_geoid_offset_m": pct(offset[up], 50),
    }


def main() -> int:
    parser = script_parser(__doc__)
    parser.add_argument("basin_ids", nargs="*", default=PHASE1_BASINS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    rows = [coverage_basin(str(b), cfg) for b in args.basin_ids]
    print(
        write_summary(rows, analysis_path(None, __file__, "summary.csv")).to_string(
            index=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
