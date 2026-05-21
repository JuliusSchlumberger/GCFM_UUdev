import logging

from src.io import load_catalogue, catalogue_entry, read_geometry


logging.basicConfig(filename=snakemake.log[0], level=logging.INFO)
log = logging.getLogger(__name__)

CATALOGUE = load_catalogue(snakemake.config["data_catalogue"])

basin_id = int(snakemake.wildcards.basin_id)
log.info(f"Extracting basin {basin_id}")

deltas = read_geometry(snakemake.input.delta_polygons)
attribute_name = catalogue_entry(CATALOGUE, "delta_polygons")["attributes"][0]["name"]

basin = deltas[deltas[attribute_name] == basin_id]

if basin.empty:
    raise ValueError(
        f"BasinID2 {basin_id} not found in {snakemake.input.delta_polygons}"
    )

basin.to_file(snakemake.output.specific_delta, driver="GPKG")
log.info(f"Wrote {snakemake.output.specific_delta} ({len(basin)} features)")
