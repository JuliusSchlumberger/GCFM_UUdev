"""Write the skeleton's bed level on the flood-depth grid, NaN on open sea; run by rule land_bed_level."""

from src.log import setup_logging
from src.postprocessing import land_bed_level, write_land_bed_level

log = setup_logging(snakemake.log[0])

da = land_bed_level(snakemake.params.skeleton_root, snakemake.input.sea_mask)
write_land_bed_level(da, snakemake.output.land_bed_level)
log.info(
    f"Written: {snakemake.output.land_bed_level} ({da.shape[1]} x {da.shape[0]} px, "
    f"{int(da.notnull().sum()):,} land px)"
)
