from collections import deque
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
from shapely.geometry import box as shapely_box
from shapely.ops import transform as shp_transform

from src.domain import load_domain
from src.log import setup_logging
from src.plots import plot_river_network
from src.profiling import ScriptProfiler

log = setup_logging(snakemake.log[0])

profiler = ScriptProfiler(snakemake)
gpd_read_file = profiler.wrap(gpd.read_file)
gpd_clip      = profiler.wrap(gpd.clip)

wgs84_bounds, domain_crs, domain_poly = load_domain(
    snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg
)
log.info(f"Domain WGS84 bounds: {wgs84_bounds}")

# ── 1. Load and clip river network ────────────────────────────────────────────

clip_gdf = gpd.GeoDataFrame(geometry=[shapely_box(*wgs84_bounds)], crs="EPSG:4326")

probe_crs = gpd.read_file(snakemake.input.global_river_network, rows=0).crs
if probe_crs is not None and probe_crs != clip_gdf.crs:
    bbox = clip_gdf.to_crs(probe_crs).total_bounds
else:
    bbox = wgs84_bounds

river_gdf = gpd_read_file(
    snakemake.input.global_river_network, bbox=tuple(bbox), engine="pyogrio"
)
clip_src = clip_gdf if river_gdf.crs is None or river_gdf.crs == clip_gdf.crs else clip_gdf.to_crs(river_gdf.crs)
clipped_rivers = gpd_clip(river_gdf, clip_src).copy()
log.info(f"Clipped to {len(clipped_rivers)} reach(es)")


# ── 2. Sample max elevation along each reach ──────────────────────────────────

def _norm_reach_id(x) -> str | None:
    s = str(x).strip()
    if s.lower() in ("nan", "none", "<na>", ""):
        return None
    try:
        return str(int(float(s)))
    except (ValueError, OverflowError):
        return s or None


def _parse_dn_ids(raw) -> list[str]:
    """Parse rch_id_dn to a list of normalised reach-ID strings."""
    s = str(raw).strip().strip("[]")
    if not s or s.lower() in ("nan", "none", "<na>"):
        return []
    result = []
    for token in s.split(","):
        nid = _norm_reach_id(token.strip())
        if nid:
            result.append(nid)
    return result


def _sample_max_elevation(geom_wgs84, src, transformer, nodata) -> float:
    """Return the max valid DEM elevation sampled along *geom_wgs84*."""
    geom_proj = shp_transform(transformer.transform, geom_wgs84)
    length = geom_proj.length
    # Sample at ~100 m intervals; minimum 2 points (start + end).
    n_pts = max(2, int(length / 100) + 1)
    pts = [geom_proj.interpolate(t, normalized=True) for t in np.linspace(0, 1, n_pts)]
    coords = [(p.x, p.y) for p in pts]
    vals = np.array([v[0] for v in src.sample(coords, masked=False)], dtype=float)
    if nodata is not None:
        vals[vals == nodata] = np.nan
    valid = vals[np.isfinite(vals)]
    return float(valid.max()) if valid.size > 0 else np.nan


with rasterio.open(snakemake.input.elevation_merged) as dem_src:
    dem_crs    = dem_src.crs
    dem_nodata = dem_src.nodata
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", dem_crs, always_xy=True
    )
    elevations = np.array([
        _sample_max_elevation(geom, dem_src, transformer, dem_nodata)
        for geom in clipped_rivers.geometry
    ])

clipped_rivers["max_elevation"] = elevations
n_valid = int(np.isfinite(elevations).sum())
log.info(
    f"Elevation sampled: {n_valid}/{len(clipped_rivers)} reaches have a finite value "
    f"(min={np.nanmin(elevations):.1f} m, max={np.nanmax(elevations):.1f} m)"
)


# ── 3. Enforce monotone downstream decrease via topological sort ──────────────
# Build a DAG from the clipped network.  For each reach (node), rch_id_dn
# lists its immediate downstream neighbours.  A single upstream→downstream
# pass then suffices: when reach U is processed, we cap every downstream
# reach D's elevation to min(elev[D], elev[U]).  At confluences (multiple
# upstream reaches) the downstream reach receives the minimum of all upstream
# caps — exactly the correct behaviour.

reach_ids = clipped_rivers["reach_id"].values
id_to_idx = {}
for i, rid in enumerate(reach_ids):
    nid = _norm_reach_id(rid)
    if nid:
        id_to_idx[nid] = i

# Build adjacency (upstream → downstream neighbours *within the clipped network*)
# and in-degree count for Kahn's algorithm.
dn_col    = "rch_id_dn" if "rch_id_dn" in clipped_rivers.columns else None
adjacency = {i: [] for i in range(len(clipped_rivers))}
in_degree = [0] * len(clipped_rivers)

if dn_col is not None:
    for i, raw_dn in enumerate(clipped_rivers[dn_col]):
        for dn_id in _parse_dn_ids(raw_dn):
            j = id_to_idx.get(dn_id)
            if j is not None and j != i:
                adjacency[i].append(j)
                in_degree[j] += 1
else:
    log.warning("Column 'rch_id_dn' not found — skipping monotone enforcement")

# Kahn's BFS topological sort
elev = elevations.copy()
queue = deque(i for i, deg in enumerate(in_degree) if deg == 0)
n_processed = 0

while queue:
    u = queue.popleft()
    n_processed += 1
    for v in adjacency[u]:
        # Cap downstream elevation
        if np.isfinite(elev[u]) and (not np.isfinite(elev[v]) or elev[v] > elev[u]):
            elev[v] = elev[u]
        in_degree[v] -= 1
        if in_degree[v] == 0:
            queue.append(v)

n_capped = int(np.sum(elev < elevations))
n_cycle  = len(clipped_rivers) - n_processed   # non-zero if graph has a cycle
if n_cycle:
    log.warning(
        f"Topological sort did not reach {n_cycle} reach(es) — possible cycle "
        f"in rch_id_dn; those reaches keep their original elevation"
    )
log.info(
    f"Monotone enforcement: {n_capped} reach elevation(s) capped to upstream value"
)

clipped_rivers["max_elevation"] = elev


# ── 4. Write output ───────────────────────────────────────────────────────────

Path(snakemake.output.spec_river_network).parent.mkdir(parents=True, exist_ok=True)
clipped_rivers.to_file(snakemake.output.spec_river_network, driver="GPKG")
log.info(f"Written: {snakemake.output.spec_river_network} ({len(clipped_rivers)} reaches)")

plot_river_network(
    snakemake.output.spec_river_network, domain_poly,
    snakemake.input.land_polygons, snakemake.output.plot_river_network,
    water_bodies_path=snakemake.input.spec_landuse,
)
profiler.stop()
log.info("Done")
