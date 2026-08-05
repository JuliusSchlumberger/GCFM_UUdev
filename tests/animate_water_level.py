"""
animate_water_level.py — Regenerate the flood-progression animation for an
already-run SFINCS spin-up or event run, as either water DEPTH (the
02_flood_animation.mp4 rule 15/16 output) or water LEVEL.

Reuses postprocessing.compute_flood_progression / plots.animate_flood_progression
directly (same functions rule 15/16 call) via their `variable="depth"|"level"`
parameter -- does not re-run SFINCS, only regenerates the animation from the
run's existing sfincs_map.nc.

Requires the run to have already produced sfincs_map.nc with a time-varying
`zs` (spin-up: rule run_spinup with dtmapout set; event: rule run_event).

Usage:
    conda run -n hmt_sfincs_dev python tests/animate_water_level.py <basin_id> [run_label] [variable]

    run_label: "spinup" (default) or "event"
    variable:  "depth" or "level" (default) -- pass "depth" to just re-run
               the existing rule-15/16 animation without going through Snakemake.

Output:
    results/{basin_id}/visuals/model_runs/{spinup|main_run}/03_water_level_animation.mp4
    (depth re-runs overwrite the existing 02_flood_animation.mp4 instead)
"""

import sys
from pathlib import Path
from typing import cast

import geopandas as gpd
import yaml
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))
from src.plots import animate_flood_progression
from src.postprocessing import compute_flood_progression

REPO_ROOT = Path(__file__).resolve().parents[1]

if len(sys.argv) < 2:
    raise SystemExit(
        "Usage: python tests/animate_water_level.py <basin_id> [run_label] [variable]"
    )
basin_id = sys.argv[1]
run_label = sys.argv[2] if len(sys.argv) > 2 else "spinup"
variable = sys.argv[3] if len(sys.argv) > 3 else "level"
if run_label not in ("spinup", "event"):
    raise SystemExit(f"run_label must be 'spinup' or 'event', got {run_label!r}")
if variable not in ("depth", "level"):
    raise SystemExit(f"variable must be 'depth' or 'level', got {variable!r}")

with open(REPO_ROOT / "config" / "config.yml") as f:
    config = yaml.safe_load(f)
results_dir = Path(config["results_dir"])
animation_fps = int(config["sfincs"]["sanity_checks"]["animation_fps"])

domain_dir = results_dir / basin_id / "preprocessing_inputs" / "domain"
sfincs_root = results_dir / basin_id / "sfincs"
run_dir = sfincs_root / "spinup" if run_label == "spinup" else sfincs_root
visuals_subdir = "spinup" if run_label == "spinup" else "main_run"
visuals_dir = results_dir / basin_id / "visuals" / "model_runs" / visuals_subdir

sfincs_map_path = run_dir / "sfincs_map.nc"
if not sfincs_map_path.exists():
    raise FileNotFoundError(
        f"{sfincs_map_path} not found -- run rule "
        f"{'run_spinup' if run_label == 'spinup' else 'run_event'} for basin {basin_id} first."
    )

landuse_path = domain_dir / f"{basin_id}_landuse.tif"
land_polygons_path = domain_dir / f"{basin_id}_land_polygons.gpkg"
river_network_path = domain_dir / f"{basin_id}_river_network_clean.gpkg"
domain_gpkg_path = domain_dir / f"{basin_id}_domain.gpkg"

domain_gdf = gpd.read_file(domain_gpkg_path)
if domain_gdf.crs is not None and domain_gdf.crs.to_epsg() != 4326:
    domain_gdf = domain_gdf.to_crs("EPSG:4326")
_union = domain_gdf.geometry.union_all()
domain_poly = cast(
    Polygon, _union if isinstance(_union, Polygon) else _union.convex_hull
)

print(f"=== Basin {basin_id} | run={run_label} | variable={variable} ===")
print(f"Reading: {sfincs_map_path}")

da_h = compute_flood_progression(run_dir, landuse_path, variable=variable)
if da_h is None:
    raise SystemExit(
        "compute_flood_progression returned None -- no time-varying 'zs' "
        f"(and, for variable='depth', 'zb') in {sfincs_map_path}."
    )

out_name = (
    "02_flood_animation.mp4" if variable == "depth" else "03_water_level_animation.mp4"
)
out_path = visuals_dir / out_name
animate_flood_progression(
    da_h,
    domain_poly,
    str(land_polygons_path),
    str(river_network_path),
    str(out_path),
    basin_id=basin_id,
    run_label=run_label,
    fps=animation_fps,
    variable=variable,
)
print(f"Written: {out_path}")
