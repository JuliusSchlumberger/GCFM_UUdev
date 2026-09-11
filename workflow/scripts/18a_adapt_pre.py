"""
Apply a preprocessing adaptation strategy to a skeleton SFINCS model.

Loads the skeleton SFINCS model built for a basin, applies each measure
defined in the given adaptation strategy (e.g. offshore barriers, dike
rings, pumps, retreat) in the order specified in
adaptation_strategies.yml, and writes out a new "adapted" SFINCS model
under adapted_root. Only the components actually modified by the applied
measures (weirs, drainage structures, subgrid) are re-written; all
untouched geometry files are forwarded from the skeleton by relative
reference to avoid duplicating unchanged data. Special-cased files that
hydromt_sfincs always resolves relative to the model root (sfincs.msk,
subgrid rasters used by postprocessing) are physically copied into
adapted_root instead of forwarded, since reference-based forwarding is
not respected when reading them.
"""

import json
import shutil
from pathlib import Path
from hydromt_sfincs import SfincsModel

from src.adaptation_method_pre import dispatch_rules
from src.log import setup_logging
from src.sfincs_run import parse_sfincs_inp, forward_geometry_files

log = setup_logging(snakemake.log[0])

# 0. load in params and input
skeleton_inp_path = Path(snakemake.input.skeleton_inp)
baseline_flood_map_path = Path(snakemake.input.baseline_flood_map)

strategy_def = snakemake.params.strategy_def
measures_def = snakemake.params.measures_def
adaptation_root = Path(snakemake.params.adaptation_root)
skeleton_root = Path(snakemake.params.skeleton_root)
adapted_root = Path(snakemake.params.adapted_root)

# 1. Read original skeleton sfincs model
adapted_root.mkdir(parents=True, exist_ok=True)
skeleton_cfg = parse_sfincs_inp(skeleton_root / "sfincs.inp")

# hydromt_sfincs's RegularGrid.read() reads every other "*file" entry (dep/
# manning/ini/subgrid/indexfile) via config.get(key, fallback=..., abs_path=True),
# which respects whatever path is actually written in sfincs.inp -- including a
# forwarded "../.." reference. The mask is the one exception: it's read via
# config.get_set_file_variable("mskfile", "sfincs.msk"), which IGNORES the
# configured mskfile value and always resolves to "<model root>/sfincs.msk"
# (existence unchecked, warning suppressed on a full read) -- confirmed against
# the installed hydromt_sfincs v2.0.0rc2 source. So unlike every other geometry
# file, sfincs.msk must be a REAL file physically inside adapted_root, never
# just a forwarded reference, or any downstream hydromt read of this root
# (e.g. adapt_build_forcing_pre's own SfincsModel(root=adapted_root)) silently
# gets an empty mask -- which then breaks subgrid reading and
# sf.water_level.create()'s boundary-cell lookup.
shutil.copy2(skeleton_root / "sfincs.msk", adapted_root / "sfincs.msk")

sf = SfincsModel(root = str(skeleton_root), mode = "r")
sf.read()
sf.root.set(str(adapted_root), mode="w+")
log.info(f"Skeleton loaded from {skeleton_root}, writes redirected to {adapted_root}")


# 2. which sfincs.inp key each measure's component owns
COMPONENT_FILEKEY = {
    "weirs": "weirfile",
    "drainage_structures": "drnfile",
    "subgrid": "sbgfile",  # TODO: check if this is correct
    # storage_volume lives on the coarse regular grid (mod.grid.data["vol"]),
    # not subgrid -- a genuinely new component this script never wrote
    # before apply_water_retention_greening.
    "storage_volume": "volfile",
}

COMPONENT_OF_MEASURE = {
    "offshore_barrier": ("weirs",),
    "pumps": ("drainage_structures",),
    "nbs_land_reclamation": ("subgrid",),
    # water_retention touches BOTH: subgrid (the excavated DEM/roughness) and
    # weirs (the containing ring around the zone boundary, see its own
    # docstring) -- without "weirs" here, sf.weirs.write() below never runs,
    # so the newly-created weir stays in-memory only and the untouched
    # skeleton's own original weirfile gets forwarded by reference instead,
    # silently discarding it (confirmed: a run showed byte-identical results
    # to a pre-weir run because of exactly this).
    "water_retention": ("subgrid", "weirs"),
    # water_retention_greening (storage_volume) is the SAME class of bug risk
    # -- without "storage_volume" here, sf.grid.write() below never runs, so
    # the newly-created volfile stays in-memory only. Never needs dep_subgrid/
    # roughness_native_path (it doesn't touch subgrid or elevation at all).
    "water_retention_greening": ("storage_volume",),
    "river_levee": ("weirs",),
    "coastal_levee": ("weirs",),
    "dike_ring" : ("weirs",),  # unused -- see adaptation_method_pre.py, replaced by urban_raising
    "retreat": ("subgrid",),
    "urban_raising": ("subgrid",),
}

# 3. Apply each measure in the strategy in the order given in adaptation_strategies.yml 
touched = set()
for measure_type, raw_params in strategy_def["measures"].items():
    measure_def = measures_def[measure_type]
    resolved = {
        k: (str(adaptation_root /v) if k in ("locations", "dep_subgrid") and isinstance(v, str) else v)
        for k, v in raw_params.items()
    }
    # NOTE: retreat/urban_raising specifically need the baseline flood map to
    # determine which cells are eligible (retreated / raised)
    flood_map_path = str(baseline_flood_map_path) if measure_type in ("retreat", "urban_raising") else None
    if measure_type in ("retreat", "nbs_land_reclamation", "water_retention", "urban_raising"):
        # dep_subgrid/roughness_native_path are never strategy-configured (see
        # adaptation_strategies.yml's own comment) -- these measures always
        # reuse this basin's own already-built elevation/roughness, never a
        # separately-authored raw-data file.
        resolved["dep_subgrid"] = str(skeleton_root / "subgrid" / "dep_subgrid.tif")
        resolved["roughness_native_path"] = str(snakemake.input.roughness_native)
    if measure_type in ("water_retention", "water_retention_greening"):
        # baseline_excess_volume is never strategy-configured either -- rule
        # attribution_mask (18c) computes it once per basin x scenario (see
        # water_retention_excess_volume_input's own docstring) and this is the
        # only place that reads the resulting JSON. Both water_retention
        # variants are sized against this SAME fixed reference.
        with open(snakemake.input.baseline_excess_volume) as fh:
            resolved["baseline_excess_volume"] = json.load(fh)["baseline_excess_volume"]
    if measure_type == "water_retention":
        # rst_path/ind_path are never strategy-configured -- apply_water_retention
        # patches a COPY of the basin's own spin-up restart so the excavated
        # zone's initial water level matches the new (lower) bed instead of
        # inheriting the original terrain's own absolute water level (see its
        # own docstring). ind_path is basin-level (skeleton_root, not
        # adapted_root) -- the active-cell layout/order is unaffected by any
        # strategy's own elevation changes.
        resolved["rst_path"] = str(snakemake.input.rstart)
        resolved["ind_path"] = str(skeleton_root / "sfincs.ind")
    if measure_type in ("retreat", "nbs_land_reclamation", "urban_raising"):
        # landuse_path is needed by anything that must identify which cells
        # are urban (retreat, urban_raising) or reclassifies/samples land use
        # (nbs_land_reclamation).
        resolved["landuse_path"] = str(snakemake.input.landuse)
    if measure_type in ("retreat", "nbs_land_reclamation"):
        # lu_roughness_lookup_path is needed only by the measures that
        # actually RECLASSIFY land use and must look up the new class's own
        # Manning's n (water_retention never changes roughness at all;
        # urban_raising changes elevation only -- the cell is still urban,
        # so roughness is passed through unchanged, never looked up).
        resolved["lu_roughness_lookup_path"] = str(snakemake.input.lu_roughness_lookup)
        # apply_NbS_land_reclamation defaults out_path to "foreshore_landuse.tif",
        # but adapt_flood_metrics_pre (17_flood_metrics.py) and the rule's own
        # declared Snakemake output both expect "retreat_landuse.tif" -- override
        # so the reclassed lulc lands where downstream actually looks for it.
        # (apply_retreat already defaults to "retreat_landuse.tif" itself.)
        if measure_type == "nbs_land_reclamation":
            resolved["out_path"] = "retreat_landuse.tif"
            # basin's own corrected open-sea mask -- apply_NbS_land_reclamation
            # writes an updated copy (reclaimed cells cleared to land) to
            # sea_mask.tif under adapted_root, since compute_max_inundation
            # (17_flood_metrics.py) and compute_flood_progression
            # (16_run_event.py) both mask flood depth to NaN wherever this
            # file reads "sea" -- without an update, they'd keep masking out
            # the newly reclaimed land, even though SFINCS itself simulates
            # it correctly.
            resolved["sea_mask_path"] = str(snakemake.input.sea_mask)

    log.info(f"Applying measure {measure_type} with params {resolved}")
    
    sf = dispatch_rules(measure_type, 
                        sf, 
                        measure_def, 
                        flood_map_path=flood_map_path, 
                        method = "preprocessing", 
                        **resolved)
    touched.update(COMPONENT_OF_MEASURE[measure_type])

# 3b. apply_retreat writes a reclassified adapted_root/retreat_landuse.tif
# (urban cells it retreats relabeled to target_code) -- adapt_flood_metrics_pre
# reads THIS file (not the basin's own raw landuse.tif) so urban_exposed_km2/
# urban_area_km2 reflect the retreat, not the pre-retreat urban footprint.
# For strategies without retreat, no such file gets written -- copy the raw
# landuse through unchanged so the output is always present either way.
LANDUSE_WRITING_MEASURES = {"retreat", "nbs_land_reclamation"}
retreat_landuse_path = adapted_root / "retreat_landuse.tif"
if not (LANDUSE_WRITING_MEASURES & set(strategy_def["measures"])):
    shutil.copy2(snakemake.input.landuse, retreat_landuse_path)

# 3c. same pattern for the sea mask: apply_NbS_land_reclamation writes an
# updated adapted_root/sea_mask.tif (reclaimed cells cleared to land). For
# strategies that don't reclaim land, no such file gets written -- copy the
# basin's own static sea mask through unchanged so the output declared in
# 18a_adapt_pre.smk is always present either way.
SEA_MASK_WRITING_MEASURES = {"nbs_land_reclamation"}
sea_mask_path = adapted_root / "sea_mask.tif"
if not (SEA_MASK_WRITING_MEASURES & set(strategy_def["measures"])):
    shutil.copy2(snakemake.input.sea_mask, sea_mask_path)

# 3d. same pattern for the restart: apply_water_retention writes a patched
# adapted_root/<rst filename> (zs shifted to match the excavated bed, see
# its own docstring). For strategies that don't touch the DEM this way, no
# such file gets written -- copy the basin's own spin-up restart through
# unchanged so the output declared in 18a_adapt_pre.smk (and read by
# adapt_build_forcing_pre/adapt_run_event_pre below) is always present.
RESTART_WRITING_MEASURES = {"water_retention"}
rst_path_local = adapted_root / Path(snakemake.input.rstart).name
if not (RESTART_WRITING_MEASURES & set(strategy_def["measures"])):
    shutil.copy2(snakemake.input.rstart, rst_path_local)

# 4. Write only touched components - untocuhe ones stay forwarded by reference below, never duplicated
if "weirs" in touched:
    sf.weirs.write()
if "drainage_structures" in touched:
    sf.drainage_structures.write()
if "storage_volume" in touched:
    # storage_volume lives on the coarse regular grid (mod.grid.data["vol"]),
    # a component this script never wrote before -- restrict to data_vars=
    # ["vol"] so this doesn't ALSO rewrite dep/mask/manning/ini (confirmed:
    # an unrestricted sf.grid.write() call writes every grid map present,
    # duplicating unchanged geometry this measure never touched). Also
    # writes a local sfincs.ind (RegularGrid.write()'s own write_ind() call
    # is unconditional) -- harmless, since mask is unchanged and the
    # forwarded mskfile reference below still resolves to the identical mask
    # this index was computed from.
    sf.grid.write(data_vars=["vol"])
if "subgrid" in touched:
    sf.subgrid.write()
else:
    # subgrid untouched: the sfincs.inp sbgfile entry gets forwarded by
    # relative reference below (SFINCS itself reads it fine from
    # skeleton_root), but src.postprocessing.get_bed_level looks for
    # dep_subgrid.tif as a REAL file directly under this run's own
    # adapted_root/subgrid/ (it never resolves sfincs.inp's forwarded
    # reference) -- so flood metrics for this strategy would otherwise
    # silently fall back to the coarse zb grid. Physically copy the
    # skeleton's own reference rasters through, same pattern as the
    # sfincs.msk copy above.
    src_subgrid_dir = skeleton_root / "subgrid"
    dst_subgrid_dir = adapted_root / "subgrid"
    for fname in ("dep_subgrid.tif", "manning_subgrid.tif"):
        src_file = src_subgrid_dir / fname
        if src_file.exists():
            dst_subgrid_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_subgrid_dir / fname)

# 5. forward every untouched file entry from skeleton by relative path (like 13_build_sfincs.py), excluding the 
# keys this run just wrote 
exclude_keys = {COMPONENT_FILEKEY[c] for c in touched}
geometry_lines = forward_geometry_files(skeleton_cfg, skeleton_root, adapted_root, exclude=exclude_keys)
scalar_ines = [f"{k:<20} = {v}" for k, v in skeleton_cfg.items() if not k.endswith("file")]

# append the touched components own just-written filen names (same as 13_build_sfincs.py for the bndfile/bzsfile/etc.)
for key in exclude_keys:
    val = sf.config.get(key)
    if val: 
        geometry_lines.append(f"{key:<20} = {val}")

with open(adapted_root / "sfincs.inp", "w") as fh:
    fh.write("\n".join(scalar_ines + geometry_lines) + "\n")
