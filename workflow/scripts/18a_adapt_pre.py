"""Apply a preprocessing adaptation strategy to a basin's SFINCS skeleton.

Run by rule adapt_apply_pre. Applies each measure (e.g. offshore barriers, dike
rings, pumps, retreat) in adaptation_strategies.yml order and writes the
adapted model to adapted_root. Only touched components (weirs, drainage
structures, subgrid) are rewritten; other geometry files are forwarded by
relative reference. sfincs.msk and the subgrid rasters read by postprocessing
are copied instead, since those readers resolve paths against the model root.
"""

import json
import shutil
from pathlib import Path
from hydromt_sfincs import SfincsModel

from src.adaptation_method_pre import dispatch_rules
from src.adaptation_scaling import resolve_forcing_params
from src.hydromt_sfincs_patch import use_float32_subgrid_write
from src.log import setup_logging
from src.sfincs_run import parse_sfincs_inp, forward_geometry_files, write_weir_file

log = setup_logging(snakemake.log[0])
use_float32_subgrid_write()

# 0. params and inputs
skeleton_inp_path = Path(snakemake.input.skeleton_inp)
baseline_flood_map_path = Path(snakemake.input.baseline_flood_map)

measures_def = snakemake.params.measures_def
adaptation_root = Path(snakemake.params.adaptation_root)
skeleton_root = Path(snakemake.params.skeleton_root)
adapted_root = Path(snakemake.params.adapted_root)

# Resolve params derived from this scenario's baseline run (max_coastal_waterlevel,
# max_discharge, max_river_waterlevel, current_protection).
strategy_def = resolve_forcing_params(
    snakemake.params.strategy_def,
    baseline_sfincs_root=snakemake.params.baseline_sfincs_root,
    baseline_map_nc=snakemake.input.baseline_sfincs_map_nc,
    skeleton_weir=snakemake.input.skeleton_weir,
    adaptation_root=adaptation_root,
    log=log,
)

# 1. Read skeleton model
adapted_root.mkdir(parents=True, exist_ok=True)
skeleton_cfg = parse_sfincs_inp(skeleton_root / "sfincs.inp")

shutil.copy2(skeleton_root / "sfincs.msk", adapted_root / "sfincs.msk")

sf = SfincsModel(root = str(skeleton_root), mode = "r")
sf.read()
sf.root.set(str(adapted_root), mode="w+")
log.info(f"Skeleton loaded from {skeleton_root}, writes redirected to {adapted_root}")


# 2. sfincs.inp key owned by each component
COMPONENT_FILEKEY = {
    "weirs": "weirfile",
    "drainage_structures": "drnfile",
    "subgrid": "sbgfile",  
    # "storage_volume": "volfile", # used for water_retention_greening 
}

COMPONENT_OF_MEASURE = {
    "offshore_barrier": ("weirs",),
    "pumps": ("drainage_structures",),
    "nbs_land_reclamation": ("subgrid",),
    "water_retention": ("subgrid", "weirs"),
    # "water_retention_greening": ("storage_volume",), # unused, see adaptation_method_pre.py
    "river_levee": ("weirs",),
    "coastal_levee": ("weirs",),
    # "dike_ring" : ("weirs",),  # unused, see adaptation_method_pre.py
    "retreat": ("subgrid",),
    "urban_raising": ("subgrid",),
}

# 3. Apply measures in adaptation_strategies.yml order
touched = set()
applied = set()  # a measure can skip itself (water_retention)
for measure_type, raw_params in strategy_def["measures"].items():
    measure_def = measures_def[measure_type]
    resolved = {
        k: (str(adaptation_root /v) if k in ("locations", "dep_subgrid") and isinstance(v, str) else v)
        for k, v in raw_params.items()
    }
    # retreat/urban_raising pick eligible cells from the baseline flood map
    flood_map_path = str(baseline_flood_map_path) if measure_type in ("retreat", "urban_raising") else None

    if measure_type in ("retreat", "nbs_land_reclamation", "water_retention", "urban_raising"):
        # Never strategy-configured: always the basin's built elevation/roughness.
        resolved["dep_subgrid"] = str(skeleton_root / "subgrid" / "dep_subgrid.tif")
        resolved["roughness_native_path"] = str(snakemake.input.roughness_native)

    if measure_type in ("water_retention"): # also "water_retention_greening" here if used
        # baseline_excess_volume comes from rule attribution_mask, per basin x scenario.
        with open(snakemake.input.baseline_excess_volume) as fh:
            resolved["baseline_excess_volume"] = json.load(fh)["baseline_excess_volume"]
        if resolved["baseline_excess_volume"] == 0:
            # No river/compound flooding: applying would dig nothing but still
            # add the weir ring. Not in `applied`, so handled as absent below.
            log.warning(
                "Skipping water_retention: baseline_excess_volume is 0 (no river/compound "
                "flooding in this scenario) -- nothing to store"
            )
            continue

    if measure_type == "water_retention":
        # apply_water_retention patches a copy of the spin-up restart so the
        # excavated zone's water level follows the lowered bed. sfincs.ind is
        # basin-level: elevation changes do not alter the active-cell order.
        resolved["rst_path"] = str(snakemake.input.rstart)
        resolved["ind_path"] = str(skeleton_root / "sfincs.ind")

    if measure_type in ("retreat", "nbs_land_reclamation", "urban_raising"):
        resolved["landuse_path"] = str(snakemake.input.landuse)

    if measure_type in ("retreat", "nbs_land_reclamation"):
        # Only land-use-reclassifying measures look up the new class's Manning's n.
        resolved["lu_roughness_lookup_path"] = str(snakemake.input.lu_roughness_lookup)
        # apply_NbS_land_reclamation defaults to "foreshore_landuse.tif"; the
        # rule output and adapt_flood_metrics_pre expect "retreat_landuse.tif".
        if measure_type == "nbs_land_reclamation":
            resolved["out_path"] = "retreat_landuse.tif"
            # An updated sea_mask.tif (reclaimed cells cleared to land) is
            # written so postprocessing does not mask the reclaimed land as sea.
            resolved["sea_mask_path"] = str(snakemake.input.sea_mask)

    log.info(f"Applying measure {measure_type} with params {resolved}")
    
    sf = dispatch_rules(measure_type, 
                        sf, 
                        measure_def, 
                        flood_map_path=flood_map_path, 
                        method = "preprocessing", 
                        **resolved)
    touched.update(COMPONENT_OF_MEASURE[measure_type])
    applied.add(measure_type)

# 3b. adapt_flood_metrics_pre reads retreat_landuse.tif so urban metrics reflect
# the reclassification; copy the raw landuse when no measure wrote it.
# TODO: this copies to all paths, see if i can change
LANDUSE_WRITING_MEASURES = {"retreat", "nbs_land_reclamation"}
retreat_landuse_path = adapted_root / "retreat_landuse.tif"
if not (LANDUSE_WRITING_MEASURES & applied):
    shutil.copy2(snakemake.input.landuse, retreat_landuse_path)

# 3c. Same for the sea mask.
SEA_MASK_WRITING_MEASURES = {"nbs_land_reclamation"}
sea_mask_path = adapted_root / "sea_mask.tif"
if not (SEA_MASK_WRITING_MEASURES & applied):
    shutil.copy2(snakemake.input.sea_mask, sea_mask_path)

# 3d. Same for the spin-up restart.
RESTART_WRITING_MEASURES = {"water_retention"}
rst_path_local = adapted_root / Path(snakemake.input.rstart).name
if not (RESTART_WRITING_MEASURES & applied):
    shutil.copy2(snakemake.input.rstart, rst_path_local)

# 4. Write only touched components
if "weirs" in touched:
    sf.weirs.write()
    write_weir_file(sf)  # crests at 1 cm, not hydromt_sfincs' 0.1 m
if "drainage_structures" in touched:
    sf.drainage_structures.write()
if "storage_volume" in touched:
    # data_vars=["vol"]: an unrestricted sf.grid.write() rewrites every grid
    # map. It also writes sfincs.ind unconditionally; harmless, the mask is
    # unchanged.
    sf.grid.write(data_vars=["vol"])
if "subgrid" in touched:
    sf.subgrid.write()
else:
    # src.postprocessing.get_bed_level reads adapted_root/subgrid/dep_subgrid.tif
    # directly, not the forwarded sbgfile; without a copy it silently falls
    # back to the coarse zb grid.
    src_subgrid_dir = skeleton_root / "subgrid"
    dst_subgrid_dir = adapted_root / "subgrid"
    for fname in ("dep_subgrid.tif", "manning_subgrid.tif"):
        src_file = src_subgrid_dir / fname
        if src_file.exists():
            dst_subgrid_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_subgrid_dir / fname)

# 5. Forward untouched file entries from the skeleton by relative path
exclude_keys = {COMPONENT_FILEKEY[c] for c in touched}
geometry_lines = forward_geometry_files(skeleton_cfg, skeleton_root, adapted_root, exclude=exclude_keys)
scalar_ines = [f"{k:<20} = {v}" for k, v in skeleton_cfg.items() if not k.endswith("file")]

for key in exclude_keys:
    val = sf.config.get(key)
    if val: 
        geometry_lines.append(f"{key:<20} = {val}")

with open(adapted_root / "sfincs.inp", "w") as fh:
    fh.write("\n".join(scalar_ines + geometry_lines) + "\n")
