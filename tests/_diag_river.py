"""Diagnostic: compare SWORD geom centroids vs. x/y attribute columns."""

import sys
import geopandas as gpd
import numpy as np

path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else r"D:\GCFM_UU\results\2444235\inputs\domain\river_network.gpkg"
)
gdf = gpd.read_file(path)

print(f"CRS          : {gdf.crs}")
print(f"Rows         : {len(gdf)}")
print(f"Geom types   : {gdf.geom_type.value_counts().to_dict()}")

if "x" in gdf.columns and "y" in gdf.columns:
    # Compare x/y attribute vs. geometry centroid
    cx = gdf.geometry.centroid.x.values
    cy = gdf.geometry.centroid.y.values
    ax = gdf["x"].values.astype(float)
    ay = gdf["y"].values.astype(float)
    dx = np.abs(cx - ax)
    dy = np.abs(cy - ay)
    print(f"\nx vs centroid.x  |  max diff: {dx.max():.6f}  mean diff: {dx.mean():.6f}")
    print(f"y vs centroid.y  |  max diff: {dy.max():.6f}  mean diff: {dy.mean():.6f}")
    # Show worst offenders
    worst = np.argsort(dx + dy)[-5:]
    print("\n5 rows with largest x/y vs. centroid discrepancy:")
    for i in worst:
        print(
            f"  row {i}: attr=({ax[i]:.5f}, {ay[i]:.5f})  centroid=({cx[i]:.5f}, {cy[i]:.5f})"
        )
else:
    print("\nNo x/y attribute columns found")

# Print a sample of raw geometry coordinates
print("\nFirst 3 reach geometries (first & last vertex):")
for i, row in gdf.head(3).iterrows():
    geom = row.geometry
    if geom.geom_type == "MultiLineString":
        parts = list(geom.geoms)
        first, last = list(parts[0].coords)[0], list(parts[-1].coords)[-1]
        n_vertices = sum(len(part.coords) for part in parts)
    else:
        coords = list(geom.coords)
        first, last, n_vertices = coords[0], coords[-1], len(coords)
    print(f"  row {i}: first={first}  last={last}  n_vertices={n_vertices}")
