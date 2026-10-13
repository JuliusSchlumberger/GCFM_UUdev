"""
07a_select_surge_stations.py -- The COAST-RP stations a basin's sea boundary
is forced at, as a small text file.

Exactly rule get_boundary_forcings' (07) own selection
(src.surge.select_surge_stations: expanding search radius around the domain,
de-duplicated, capped), written out on its own so the expensive GTSM
extraction (rule extract_gtsm_series) depends on THIS file and nothing
else. The file is plain JSON with rounded values, so re-running this rule on
an unchanged domain rewrites identical content, and Snakemake (which
compares the checksum of small input files, not just their timestamp) does
not re-trigger the extraction -- only a change in the selected stations
does.

Output: surge_stations.json -- {"stations": [{"lon", "lat", "dist_km"}, ...]},
        nearest first.
"""

import json
from pathlib import Path

import geopandas as gpd

from src.domain import load_domain
from src.log import setup_logging
from src.surge import select_surge_stations

log = setup_logging(snakemake.log[0])

_, domain_crs, domain_poly = load_domain(snakemake.input.spec_basins_meta, snakemake.input.domain_gpkg)
domain_utm = gpd.GeoDataFrame(geometry=[domain_poly], crs="EPSG:4326").to_crs(domain_crs)

stations = select_surge_stations(
    snakemake.input.surge_data,
    domain_utm,
    domain_crs,
    min_stations=snakemake.params.min_surge_stations,
    max_stations=snakemake.params.max_surge_stations,
    search_radii_km=snakemake.params.search_radii_km,
    dedupe_radius_km=snakemake.params.surge_dedupe_radius_km,
)
out = {
    "stations": [
        {"lon": round(float(g.x), 6), "lat": round(float(g.y), 6), "dist_km": round(float(d) / 1000.0, 3)}
        for g, d in zip(stations.geometry, stations["dist_m"])
    ]
}
Path(snakemake.output.surge_stations).parent.mkdir(parents=True, exist_ok=True)
with open(snakemake.output.surge_stations, "w") as f:
    json.dump(out, f, indent=2)
log.info(
    f"Written: {snakemake.output.surge_stations} ({len(out['stations'])} station(s), "
    f"farthest {max(s['dist_km'] for s in out['stations']):.1f} km)"
)
