# Rule: the skeleton's bed level on the flood-depth grid, NaN on open sea
# (src.postprocessing.land_bed_level), written once per basin and read by every
# rule that turns a SFINCS map into flood depths on the baseline skeleton.

rule land_bed_level:
    input:
        # Written with subgrid/dep_subgrid.tif, which the script reads.
        sfincs_subgrid = results_path("{basin_id}/sfincs_skeleton/sfincs_subgrid.nc"),
        # Corrected sea/land classification (rule 11, grid-aligned): cells the
        # final weir protects are land.
        sea_mask       = results_path("{basin_id}/preprocessing_inputs/domain/{basin_id}_zsini_sea_cells_on_grid.tif"),
    output:
        land_bed_level = results_path("{basin_id}/sfincs_skeleton/land_bed_level.nc"),
    params:
        skeleton_root = lambda wildcards: results_path(f"{wildcards.basin_id}/sfincs_skeleton"),
    threads: rule_threads("land_bed_level")
    log:
        "logs/{basin_id}/12b_land_bed_level.log"
    script:
        "../scripts/12b_land_bed_level.py"
