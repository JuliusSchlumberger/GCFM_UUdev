"""
12b_bifurcation_calibration_options.py — compares discharge partitioning at
every bifurcation between the ORIGINAL and MODIFIED SWORD river networks, by
rebuilding the cleaned river network FROM RAW SWORD (not reading the
pre-built river_network_clean.gpkg) via the actual pipeline functions
(mirrors rules get_river_network (06) + clean_river_network (08)), once per
source:

  1. Original SWORD (data catalogue "river_network_original",
     SWORD/SWORD_global_v17c_unpublished.gpkg).
  2. Modified SWORD (data catalogue "river_network" -- this is now the
     pipeline's default source, SWORD/SWORD_global_v17c_unpublished_
     modified.gpkg -- specific reaches manually corrected, e.g. width, based
     on Google Earth imagery and expert knowledge).

Both are cleaned the same way and discharge is split with the same current
production rule (pure width-based bifurcation splitting) -- the ONLY thing
that can differ between them is which reaches were manually corrected.

Rebuilding from raw SWORD requires every non-width-dependent cleaning step
rule 08 performs (fix_tjunction_tails, downstream-reachability filtering
from the basin's seed crossings, delta-outflow-point identification,
missing-width handling) in addition to rule 06's domain-bbox clip -- see
clip_and_preclean() below. river_forcing.nc (rule 07's output, needed for
seed crossings/discharges) is REUSED UNCHANGED for both SWORD sources:
crossing detection and GloFAS discharge matching depend only on reach
geometry and location, which SWORD's product description confirms unchanged
between the two files (same feature count, same schema; only specific
reaches' attribute values were manually adjusted, not geometry or
rch_id_up/rch_id_dn topology) -- this also means both sources share the same
geometry/adjacency, so bifurcations are identified once (from the original
network) and reused for both.

For each bifurcation, the two sources' canonical 'width' is compared across
every reach in the local neighborhood (see downstream_within_steps): if any
differ, a 2-panel figure (original | modified) is produced so the actual
effect of the correction is visible; if none differ, the two sources give
an identical result there, so just ONE panel is produced instead of a
redundant duplicate. In either case, each reach's line width is
proportional to its accumulated discharge (a SHARED scale across panels
when there are two); every downstream reach in the local neighborhood is
annotated with up to two percentages:
  - immediate-upstream %: Q(reach) / Q(its own immediate upstream reach),
    recomputed at each hop -- shown only where the reach has exactly one
    upstream neighbour (undefined at confluences).
  - bifurcation-relative %: Q(reach) / Q(the bifurcation reach itself) --
    a FIXED denominator for every reach downstream of that split, however
    many hops away, so it stays comparable along the whole distributary
    branch and is well-defined even at confluences (unlike the
    immediate-upstream %).
Format: "immediate% / bifurcation%" (e.g. "39% / 24%"), or just
"bifurcation%" alone where the immediate-upstream % is undefined.

Diagnostic-only side branch: only needs rule 06's raw river network sources
(read directly from the data catalogue, not rule 06's own output) and rule
07's river_forcing.nc -- nothing downstream depends on this rule's output.
One figure per bifurcation, written to
visuals/bifurcation_calibration_options/{upstream_reach_id}.png.
"""

import math
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point, box as shapely_box

from src.domain import load_domain
from src.geometry import pick_utm_crs
from src.log import setup_logging
from src.profiling import ScriptProfiler
from src.river_forcing import load_forcing_crossings, snap_crossings_to_reaches
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

N_ITERATIONS = int(snakemake.params.n_iterations)
MIN_WIDTH_M = float(snakemake.params.min_width_m)
DISCHARGE_VARIABLE = snakemake.params.discharge_variable
N_STEPS = 3
WIDTH_DIFF_TOLERANCE_M = 0.01  # min |width_orig - width_mod| in a bifurcation's neighborhood
                                # before it's considered "actually different" (vs. floating-point noise)

ORIGINAL_SWORD_PATH = snakemake.input.river_network_original
MODIFIED_SWORD_PATH = snakemake.input.river_network


def clip_and_preclean(
    global_path: str,
    wgs84_bounds: tuple[float, float, float, float],
    delta_polygon: gpd.GeoDataFrame,
    seed_q: dict[str, float],
) -> gpd.GeoDataFrame:
    """Clip a raw global river network to the domain bbox and apply every
    cleaning step that does NOT depend on which width column is chosen
    (mirrors rule get_river_network (06) + the non-width part of rule
    clean_river_network (08): fix_tjunction_tails, downstream-reachability
    filtering from seed_q, is_seed marking, delta-outflow-point
    identification, missing-width handling). Width normalisation, mouth-
    width monotonicity, and discharge accumulation are scenario-specific
    and applied by the caller afterward."""
    clip_gdf = gpd.GeoDataFrame(geometry=[shapely_box(*wgs84_bounds)], crs="EPSG:4326")
    probe_crs = gpd.read_file(global_path, rows=0).crs
    bbox = (
        clip_gdf.to_crs(probe_crs).total_bounds
        if probe_crs is not None and probe_crs != clip_gdf.crs
        else wgs84_bounds
    )
    river_gdf = gpd.read_file(global_path, bbox=tuple(bbox), engine="pyogrio")
    clip_src = (
        clip_gdf if river_gdf.crs is None or river_gdf.crs == clip_gdf.crs
        else clip_gdf.to_crs(river_gdf.crs)
    )
    rivers = gpd.clip(river_gdf, clip_src).copy()
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
    """Production width treatment: normalize_channel_widths (fix swapped
    width/max_width, canonical 'width') + enforce_mouth_width_monotonic."""
    rivers = normalize_channel_widths(rivers_clean)
    return enforce_mouth_width_monotonic(rivers)


def compute_discharge(
    rivers: gpd.GeoDataFrame, seed_q: dict[str, float]
) -> gpd.GeoDataFrame:
    """Run discharge accumulation and attach the result as 'bankfull_discharge_acc'."""
    adjacency = build_downstream_adjacency(rivers)
    q = accumulate_discharge(rivers, seed_q, adjacency, N_ITERATIONS, MIN_WIDTH_M)
    rivers = rivers.copy()
    rivers["bankfull_discharge_acc"] = q
    return rivers


def downstream_within_steps(adjacency: dict, root: str, n_steps: int) -> list[str]:
    """BFS downstream from root; returns reach_ids reachable within n_steps hops (incl. root)."""
    visited = {root}
    frontier = [root]
    for _ in range(n_steps):
        next_frontier = []
        for rid in frontier:
            for dn in adjacency.get(rid, []):
                if dn not in visited:
                    visited.add(dn)
                    next_frontier.append(dn)
        frontier = next_frontier
        if not frontier:
            break
    return list(visited)


def discharge_to_linewidth(q: float, q_max: float, min_lw: float = 1.0, max_lw: float = 10.0) -> float:
    if not np.isfinite(q) or q <= 0 or not np.isfinite(q_max) or q_max <= 0:
        return min_lw
    return min_lw + (max_lw - min_lw) * np.sqrt(min(q, q_max) / q_max)


def utm_epsg_to_cartopy_crs(utm_epsg: str) -> ccrs.CRS:
    """Convert an 'EPSG:326xx'/'EPSG:327xx' UTM code (as returned by
    src.geometry.pick_utm_crs) to the equivalent cartopy CRS, so plotted data
    (already reprojected to the same UTM CRS via pyproj) lines up exactly
    with an OSM tile background fetched in that projection."""
    epsg = int(utm_epsg.split(":")[1])
    if 32601 <= epsg <= 32660:
        return ccrs.UTM(epsg - 32600, southern_hemisphere=False)
    if 32701 <= epsg <= 32760:
        return ccrs.UTM(epsg - 32700, southern_hemisphere=True)
    raise ValueError(f"Unexpected UTM EPSG code: {utm_epsg}")


def auto_zoomlevel(extent: tuple[float, float, float, float], lat_deg: float) -> int:
    """Same core formula hydromt_sfincs's own plot_basemap(bmap=...) uses to
    pick an OSM tile zoom level from a projected-CRS extent and latitude."""
    earth_circumference_m = 2 * np.pi * 6378137
    tile_size_m = max(extent[1] - extent[0], extent[3] - extent[2]) / 4
    zoom = int(np.log2(earth_circumference_m * abs(np.cos(np.radians(lat_deg))) / tile_size_m))
    return min(17, max(10, zoom))


def annotation_label(
    rid: str, root_rid: str, q_series: pd.Series, upstream_adj: dict[str, list[str]]
) -> str | None:
    """'immediate% / bifurcation%', or just 'bifurcation%' where the
    immediate-upstream % is undefined (confluences, >1 upstream neighbour)."""
    q_root = float(q_series.get(root_rid, np.nan))
    q_rid = float(q_series.get(rid, np.nan))
    if not np.isfinite(q_root) or q_root <= 0 or not np.isfinite(q_rid):
        return None
    bifurcation_pct = q_rid / q_root * 100

    ups = upstream_adj.get(rid, [])
    if len(ups) == 1:
        q_parent = float(q_series.get(ups[0], np.nan))
        if np.isfinite(q_parent) and q_parent > 0:
            immediate_pct = q_rid / q_parent * 100
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

    # ── Original SWORD ──────────────────────────────────────────────────────
    log.info(f"Basin {basin_id}: original SWORD")
    rivers_orig = clip_and_preclean(ORIGINAL_SWORD_PATH, wgs84_bounds, delta_polygon, seed_q)
    rivers_orig = apply_base_width_rule(rivers_orig)
    rivers_orig = compute_discharge(rivers_orig, seed_q)

    # ── Modified SWORD (pipeline default) ───────────────────────────────────
    log.info(f"Basin {basin_id}: modified SWORD")
    rivers_mod = clip_and_preclean(MODIFIED_SWORD_PATH, wgs84_bounds, delta_polygon, seed_q)
    rivers_mod = apply_base_width_rule(rivers_mod)
    rivers_mod = compute_discharge(rivers_mod, seed_q)

    # ── reproject to a shared UTM CRS; geometry/topology are identical
    # between the two sources (confirmed: only attribute values were
    # modified), so bifurcation detection, adjacency, and plotted geometry
    # are all built once from the original network and reused for the
    # modified one -- only 'width' and the discharge values can differ. ──
    utm_crs = pick_utm_crs(rivers_orig)
    cartopy_crs = utm_epsg_to_cartopy_crs(utm_crs)

    def _prep(rivers: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        rivers_utm = rivers.to_crs(utm_crs)
        rivers_utm["reach_id_norm"] = rivers_utm["reach_id"].apply(normalize_reach_id)
        return rivers_utm.set_index("reach_id_norm", drop=False)

    rivers_orig_utm = _prep(rivers_orig)
    rivers_mod_utm = _prep(rivers_mod)

    adjacency = build_downstream_adjacency(rivers_orig_utm)
    upstream_adj: dict[str, list[str]] = {rid: [] for rid in adjacency}
    for rid, dns in adjacency.items():
        for dn in dns:
            upstream_adj.setdefault(dn, []).append(rid)

    bifurcation_ids = [rid for rid, dn in adjacency.items() if len(dn) >= 2]
    log.info(f"Basin {basin_id}: {len(rivers_orig_utm)} reaches, {len(bifurcation_ids)} bifurcation(s)")

    q_by_scenario = {
        "Original": rivers_orig_utm["bankfull_discharge_acc"],
        "Modified": rivers_mod_utm["bankfull_discharge_acc"],
    }

    for root_rid in bifurcation_ids:
        neighborhood_ids = downstream_within_steps(adjacency, root_rid, N_STEPS)
        if not all(rid in rivers_orig_utm.index for rid in neighborhood_ids):
            log.warning(
                f"Basin {basin_id}, bifurcation {root_rid}: neighborhood reach missing "
                f"from network -- skipping"
            )
            continue

        # Only show both panels if the correction actually touched something
        # in this neighborhood -- comparing the canonical 'width' (the value
        # that actually drives accumulate_discharge), not the raw SWORD
        # attribute, so this reflects exactly what could visibly differ.
        width_differs = any(
            abs(float(rivers_orig_utm.loc[rid, "width"]) - float(rivers_mod_utm.loc[rid, "width"]))
            > WIDTH_DIFF_TOLERANCE_M
            for rid in neighborhood_ids
        )
        panels = dict(q_by_scenario) if width_differs else {"Modified": q_by_scenario["Modified"]}
        n_panels = len(panels)

        q_max = max(
            q_series.loc[neighborhood_ids].replace([np.inf, -np.inf], np.nan).fillna(0.0).max()
            for q_series in panels.values()
        )
        if not np.isfinite(q_max) or q_max <= 0:
            q_max = 1.0

        # ── extent + OSM tile zoom level for this bifurcation's neighborhood ──
        neighborhood_geoms = [rivers_orig_utm.loc[rid].geometry for rid in neighborhood_ids]
        minx = min(g.bounds[0] for g in neighborhood_geoms)
        miny = min(g.bounds[1] for g in neighborhood_geoms)
        maxx = max(g.bounds[2] for g in neighborhood_geoms)
        maxy = max(g.bounds[3] for g in neighborhood_geoms)
        margin = max(maxx - minx, maxy - miny) * 0.15 + 50.0
        extent = (minx - margin, maxx + margin, miny - margin, maxy + margin)
        neighborhood_cx = (minx + maxx) / 2
        neighborhood_cy = (miny + maxy) / 2
        centroid_wgs = (
            gpd.GeoSeries([Point((minx + maxx) / 2, (miny + maxy) / 2)], crs=utm_crs)
            .to_crs("EPSG:4326")
            .iloc[0]
        )
        zoomlevel = auto_zoomlevel(extent, centroid_wgs.y)

        fig = plt.figure(figsize=(8 * n_panels, 8))
        for i, (label, q_series) in enumerate(panels.items()):
            ax = fig.add_subplot(1, n_panels, i + 1, projection=cartopy_crs)
            ax.set_extent(extent, crs=cartopy_crs)
            try:
                ax.add_image(cimgt.OSM(), zoomlevel)
            except Exception as e:
                log.warning(f"Basin {basin_id}, bifurcation {root_rid}: OSM basemap fetch failed ({e}) -- continuing without it")

            for rid in neighborhood_ids:
                line = _as_linestring(rivers_orig_utm.loc[rid].geometry)
                if line is None:
                    continue
                q_val = float(q_series.get(rid, 0.0))
                lw = discharge_to_linewidth(q_val, q_max)
                color = "black" if rid == root_rid else "steelblue"
                ax.plot(
                    *line.xy, color=color, linewidth=lw, solid_capstyle="round",
                    transform=cartopy_crs, zorder=2,
                )

                if rid == root_rid:
                    continue
                label_text = annotation_label(rid, root_rid, q_series, upstream_adj)
                if label_text is not None:
                    mid = line.interpolate(0.5, normalized=True)
                    # Push the label radially outward from the neighborhood's
                    # centroid rather than placing it exactly on the line --
                    # reaches on different branches sit on different sides of
                    # the centroid, so this tends to separate their labels
                    # even where the branches themselves run close together
                    # (the raw on-line placement collided there). A thin
                    # leader line ties the label back to its actual reach.
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
            f"Basin {basin_id} — bifurcation at reach {root_rid} ({diff_note})\n"
            f"line width ∝ accumulated discharge"
            f"{' (shared scale across panels)' if n_panels > 1 else ''}; "
            f"labels = immediate-upstream % / bifurcation-relative % "
            f"(bifurcation-relative % alone at confluences)",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        out_path = plot_dir / f"{root_rid}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        log.info(f"Written: {out_path}")

profiler.stop()
log.info("Done")
