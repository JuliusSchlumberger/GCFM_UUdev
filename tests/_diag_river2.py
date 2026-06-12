"""Extended river network diagnostic."""

import sys
import geopandas as gpd
import netCDF4 as nc
from pathlib import Path

basin = sys.argv[1] if len(sys.argv) > 1 else "2444235"
results = Path(r"D:\GCFM_UU\results") / basin / "inputs"

# --- 1. river_network.gpkg columns and reach_ids ---
rnet_path = results / "domain" / "river_network.gpkg"
print(f"=== {rnet_path} ===")
gdf = gpd.read_file(rnet_path)
print(f"CRS: {gdf.crs}")
print(f"Columns: {list(gdf.columns)}")
print(f"Rows: {len(gdf)}")
if "reach_id" in gdf.columns:
    print(f"reach_id sample: {gdf['reach_id'].head(3).tolist()}")
if "rch_id_dn" in gdf.columns:
    print(f"rch_id_dn sample: {gdf['rch_id_dn'].head(3).tolist()}")

# --- 2. river_forcing.nc crossings ---
forcing_path = results / "forcing" / "river_forcing.nc"
print(f"\n=== {forcing_path} ===")
if forcing_path.exists():
    ds = nc.Dataset(forcing_path)
    for var in ds.variables:
        v = ds.variables[var]
        print(f"  {var}: shape={v.shape} dims={v.dimensions}")
        if v.size < 20:
            print(f"    values: {v[:].tolist()}")
    ds.close()
else:
    print("  NOT FOUND")

# --- 3. domain_crs for this basin ---
domain_path = results / "domain" / "domain.gpkg"
if domain_path.exists():
    domain_gdf = gpd.read_file(domain_path)
    print(f"\nDomain CRS: {domain_gdf.crs}")
