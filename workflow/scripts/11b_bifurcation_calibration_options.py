"""Plot discharge partitioning at each bifurcation without vs with the manual SWORD widths.

Run by rule test_bifurcation_calibration_options (diagnostic only). Rebuilds
the cleaned network from SWORD v17c twice, without and with
config/sword_width_corrections.csv, using the rule 06 + 08 cleaning functions
and the production width-based bifurcation split. Only widths differ, so
river_forcing.nc seed crossings, adjacency and bifurcations are taken once and
reused for both.

Two panels where canonical 'width' differs in the neighbourhood, else one.
Line width ∝ accumulated discharge (shared scale). Labels:
"immediate% / bifurcation%" with immediate% = Q(reach)/Q(single upstream
reach) and bifurcation% = Q(reach)/Q(bifurcation reach); bifurcation% alone
at confluences.
"""

import math
from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.domain import load_domain
from src.geometry import pick_utm_crs
from src.log import setup_logging
from src.plots import save_figure
from src.profiling import ScriptProfiler
from src.river_forcing import load_forcing_crossings, snap_crossings_to_reaches
from src.sword import apply_width_corrections, read_sword_reaches, read_width_corrections
from src.river_network import (
    _as_linestring,
    accumulate_discharge,
    build_downstream_adjacency,
    collect_downstream_main_paths,
    enforce_mouth_width_monotonic,
    fix_tjunction_tails,
    identify_delta_outflow_points,
    normalize_channel_widths,
    normalize_reach_id,
    remove_reaches_with_missing_width,
)

log = setup_logging(snakemake.log[0])
profiler = ScriptProfiler(snakemake)

basin_id = snakemake.wildcards.basin_id
plot_dir = Path(snakemake.output.plot_dir)
plot_dir.mkdir(parents=True, exist_ok=True)

DISCHARGE_VARIABLE = snakemake.params.discharge_variable
N_STEPS = 3
WIDTH_DIFF_TOLERANCE_M = 0.01  # below this |width_orig - width_mod| is floating-point noise

SWORD_REGIONS = list(snakemake.input.sword_regions)
WIDTH_CORRECTIONS = read_width_corrections(snakemake.input.width_corrections)


def clip_and_preclean(
    corrected: bool,
    wgs84_bounds: tuple[float, float, float, float],
    delta_polygon: gpd.GeoDataFrame,
    seed_q: dict[str, float],
) -> gpd.GeoDataFrame:
    """Clip SWORD to the domain and apply the width-independent cleaning.

    Mirrors rule 06 (manual widths only if ``corrected``) plus the non-width
    part of rule 08; width normalisation, mouth-width monotonicity and
    discharge accumulation are left to the caller.
    """
    rivers = read_sword_reaches(SWORD_REGIONS, wgs84_bounds)
    if corrected:
        rivers = apply_width_corrections(rivers, WIDTH_CORRECTIONS)
    rivers = fix_tjunction_tails(rivers)

    reachable = collect_downstream_main_paths(rivers, set(seed_q.keys()))
    rivers_clean = rivers[
        rivers["reach_id"].apply(normalize_reach_id).isin(reachable)
    ].copy()
    rivers_clean["is_seed"] = (
        rivers_clean["reach_id"].apply(normalize_reach_id).isin(set(seed_q.keys()))
    )
    log.info(f"  clipped {len(rivers)} reach(es) -> {len(rivers_clean)} reachable from seed(s)")

    rivers_clean, outflow_points = identify_delta_outflow_points(rivers_clean, delta_polygon)
    log.info(f"  delta-outline outflow points: {len(outflow_points)}")

    rivers_clean = remove_reaches_with_missing_width(rivers_clean)
    return rivers_clean


def apply_base_width_rule(rivers_clean: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Apply the production width treatment (normalised widths, monotonic mouth width)."""
    rivers = normalize_channel_widths(rivers_clean)
    return enforce_mouth_width_monotonic(rivers)


def compute_discharge(
    rivers: gpd.GeoDataFrame, seed_q: dict[str, float]
) -> gpd.GeoDataFrame:
    """Run discharge accumulation and attach the result as 'bankfull_discharge_acc_m3s'."""
    adjacency = build_downstream_adjacency(rivers)
    q = accumulate_discharge(rivers, seed_q, adjacency)
    rivers = rivers.copy()
    rivers["bankfull_discharge_acc_m3s"] = q
    return rivers


def downstream_within_steps(adjacency: dict, root: str, n_steps: int) -> list[str]:
    """Return reach_ids within n_steps downstream hops of root, root included."""
    visited = {root}
    frontier = [root]
    for _ in range(n_steps):
        next_frontier = []
        for reach_id in frontier:
            for dn in adjacency.get(reach_id, []):
                if dn not in visited:
                    visited.add(dn)
                    next_frontier.append(dn)
        frontier = next_frontier
        if not frontier:
            break
    return list(visited)


def discharge_to_linewidth(q: float, q_max: float, min_lw: float = 1.0, max_lw: float = 10.0) -> float:
    """Map discharge to a line width, scaling with sqrt(q / q_max)."""
    if not np.isfinite(q) or q <= 0 or not np.isfinite(q_max) or q_max <= 0:
        return min_lw
    return min_lw + (max_lw - min_lw) * np.sqrt(min(q, q_max) / q_max)


def utm_epsg_to_cartopy_crs(utm_epsg: str) -> ccrs.CRS:
    """Convert an 'EPSG:326xx'/'EPSG:327xx' UTM code to a cartopy UTM CRS."""
    epsg = int(utm_epsg.split(":")[1])
    if 32601 <= epsg <= 32660:
        return ccrs.UTM(epsg - 32600, southern_hemisphere=False)
    if 32701 <= epsg <= 32760:
        return ccrs.UTM(epsg - 32700, southern_hemisphere=True)
    raise ValueError(f"Unexpected UTM EPSG code: {utm_epsg}")


def annotation_label(
    reach_id: str, root_reach_id: str, q_series: pd.Series, upstream_adj: dict[str, list[str]]
) -> str | None:
    """Return 'immediate% / bifurcation%', or 'bifurcation%' at confluences; None if undefined."""
    q_root = float(q_series.get(root_reach_id, np.nan))
    q_reach_id = float(q_series.get(reach_id, np.nan))
    if not np.isfinite(q_root) or q_root <= 0 or not np.isfinite(q_reach_id):
        return None
    bifurcation_pct = q_reach_id / q_root * 100

    ups = upstream_adj.get(reach_id, [])
    if len(ups) == 1:
        q_parent = float(q_series.get(ups[0], np.nan))
        if np.isfinite(q_parent) and q_parent > 0:
            immediate_pct = q_reach_id / q_parent * 100
            return f"{immediate_pct:.0f}% / {bifurcation_pct:.0f}%"
    return f"{bifurcation_pct:.0f}%"


wgs84_bounds, _domain_crs, domain_poly = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)
delta_polygon = gpd.read_file(snakemake.input.delta_polygon)

crossings = load_forcing_crossings(snakemake.input.river_forcing, discharge_variable=DISCHARGE_VARIABLE)
seed_q = snap_crossings_to_reaches(crossings)
if not seed_q:
    log.warning(f"Basin {basin_id}: no seed reach(es) with valid discharge -- nothing to plot")
else:
    log.info(f"Basin {basin_id}: {len(seed_q)} seed reach(es), reused across both SWORD sources")

    log.info(f"Basin {basin_id}: SWORD without manual widths")
    rivers_orig = clip_and_preclean(False, wgs84_bounds, delta_polygon, seed_q)
    rivers_orig = apply_base_width_rule(rivers_orig)
    rivers_orig = compute_discharge(rivers_orig, seed_q)

    log.info(f"Basin {basin_id}: SWORD with manual widths")
    rivers_mod = clip_and_preclean(True, wgs84_bounds, delta_polygon, seed_q)
    rivers_mod = apply_base_width_rule(rivers_mod)
    rivers_mod = compute_discharge(rivers_mod, seed_q)

    # Geometry and topology are shared, so adjacency, bifurcations and plotted
    # geometry come from the original network only.
    utm_crs = pick_utm_crs(rivers_orig)
    cartopy_crs = utm_epsg_to_cartopy_crs(utm_crs)
    land_utm = gpd.read_file(snakemake.input.land_polygons).to_crs(utm_crs)

    def _prep(rivers: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        rivers_utm = rivers.to_crs(utm_crs)
        rivers_utm["reach_id_norm"] = rivers_utm["reach_id"].apply(normalize_reach_id)
        return rivers_utm.set_index("reach_id_norm", drop=False)

    rivers_orig_utm = _prep(rivers_orig)
    rivers_mod_utm = _prep(rivers_mod)

    adjacency = build_downstream_adjacency(rivers_orig_utm)
    upstream_adj: dict[str, list[str]] = {reach_id: [] for reach_id in adjacency}
    for reach_id, dns in adjacency.items():
        for dn in dns:
            upstream_adj.setdefault(dn, []).append(reach_id)

    bifurcation_ids = [reach_id for reach_id, dn in adjacency.items() if len(dn) >= 2]
    log.info(f"Basin {basin_id}: {len(rivers_orig_utm)} reaches, {len(bifurcation_ids)} bifurcation(s)")

    q_by_scenario = {
        "Original": rivers_orig_utm["bankfull_discharge_acc_m3s"],
        "Modified": rivers_mod_utm["bankfull_discharge_acc_m3s"],
    }

    for root_reach_id in bifurcation_ids:
        neighborhood_ids = downstream_within_steps(adjacency, root_reach_id, N_STEPS)
        if not all(reach_id in rivers_orig_utm.index for reach_id in neighborhood_ids):
            log.warning(
                f"Basin {basin_id}, bifurcation {root_reach_id}: neighborhood reach missing "
                f"from network -- skipping"
            )
            continue

        # Canonical 'width' drives accumulate_discharge, so compare that, not
        # the raw SWORD attribute.
        width_differs = any(
            abs(float(rivers_orig_utm.loc[reach_id, "width"]) - float(rivers_mod_utm.loc[reach_id, "width"]))
            > WIDTH_DIFF_TOLERANCE_M
            for reach_id in neighborhood_ids
        )
        panels = dict(q_by_scenario) if width_differs else {"Modified": q_by_scenario["Modified"]}
        n_panels = len(panels)

        q_max = max(
            q_series.loc[neighborhood_ids].replace([np.inf, -np.inf], np.nan).fillna(0.0).max()
            for q_series in panels.values()
        )
        if not np.isfinite(q_max) or q_max <= 0:
            q_max = 1.0

        neighborhood_geoms = [rivers_orig_utm.loc[reach_id].geometry for reach_id in neighborhood_ids]
        minx = min(g.bounds[0] for g in neighborhood_geoms)
        miny = min(g.bounds[1] for g in neighborhood_geoms)
        maxx = max(g.bounds[2] for g in neighborhood_geoms)
        maxy = max(g.bounds[3] for g in neighborhood_geoms)
        margin = max(maxx - minx, maxy - miny) * 0.15 + 50.0
        extent = (minx - margin, maxx + margin, miny - margin, maxy + margin)
        neighborhood_cx = (minx + maxx) / 2
        neighborhood_cy = (miny + maxy) / 2

        fig = plt.figure(figsize=(8 * n_panels, 8))
        for i, (label, q_series) in enumerate(panels.items()):
            ax = fig.add_subplot(1, n_panels, i + 1, projection=cartopy_crs)
            ax.set_extent(extent, crs=cartopy_crs)
            # Land mask from rule get_land_polygons (land use != 200, native
            # raster): this rule runs before the SFINCS grid exists.
            if not land_utm.empty:
                ax.add_geometries(
                    land_utm.geometry, crs=cartopy_crs,
                    facecolor="#d9d9d9", edgecolor="#aaaaaa", linewidth=0.3, zorder=0,
                )

            for reach_id in neighborhood_ids:
                line = _as_linestring(rivers_orig_utm.loc[reach_id].geometry)
                if line is None:
                    continue
                q_val = float(q_series.get(reach_id, 0.0))
                lw = discharge_to_linewidth(q_val, q_max)
                color = "black" if reach_id == root_reach_id else "steelblue"
                ax.plot(
                    *line.xy, color=color, linewidth=lw, solid_capstyle="round",
                    transform=cartopy_crs, zorder=2,
                )

                if reach_id == root_reach_id:
                    continue
                label_text = annotation_label(reach_id, root_reach_id, q_series, upstream_adj)
                if label_text is not None:
                    mid = line.interpolate(0.5, normalized=True)
                    # Offset radially from the neighbourhood centroid so labels
                    # on close-running branches separate; a leader line links back.
                    dx, dy = mid.x - neighborhood_cx, mid.y - neighborhood_cy
                    dist = math.hypot(dx, dy)
                    offset_pts = (dx / dist * 22, dy / dist * 22) if dist > 0 else (0, 22)
                    ax.annotate(
                        label_text,
                        (mid.x, mid.y),
                        xytext=offset_pts,
                        textcoords="offset points",
                        fontsize=9,
                        color="darkred",
                        ha="center",
                        va="center",
                        zorder=4,
                        xycoords=cartopy_crs._as_mpl_transform(ax),
                        bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.8),
                        arrowprops=dict(arrowstyle="-", color="darkred", linewidth=0.6, alpha=0.7, shrinkA=0, shrinkB=2),
                    )

            ax.set_title(label, fontsize=11)

        diff_note = (
            "original vs. modified (width differs in this neighborhood)"
            if width_differs
            else "original and modified are identical here -- single panel"
        )
        fig.suptitle(
            f"Basin {basin_id} — bifurcation at reach {root_reach_id} ({diff_note})\n"
            f"line width ∝ accumulated discharge"
            f"{' (shared scale across panels)' if n_panels > 1 else ''}; "
            f"labels = immediate-upstream % / bifurcation-relative % "
            f"(bifurcation-relative % alone at confluences)",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        out_path = plot_dir / f"{root_reach_id}.png"
        save_figure(fig, out_path, dpi=150)
        log.info(f"Written: {out_path}")

profiler.stop()
log.info("Done")
