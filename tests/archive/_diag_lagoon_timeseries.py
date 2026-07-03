import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))
from src.postprocessing import load_sfincs_output, get_bed_level

RUN_DIR = Path("D:/GCFM_UU/experiments/river_smoothing_sfincs/2444235/original/spinup")
SFINCS_ROOT = Path("D:/GCFM_UU/experiments/river_smoothing_sfincs/2444235/original")

mod = load_sfincs_output(RUN_DIR)
print("output vars:", list(mod.output.data.data_vars))

da_zs = mod.output.data["zs"]  # full water-level timeseries (time, mesh/grid)
print("zs dims:", da_zs.dims, da_zs.shape)

# locate the lagoon centroid (lon=19.405, lat=40.588, from the landuse=80 check)
crs = da_zs.rio.crs if hasattr(da_zs, "rio") else mod.grid.data.rio.crs
t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
x, y = t.transform(19.405, 40.588)
print(f"lagoon centroid in model CRS: x={x:.0f}, y={y:.0f}")

# nearest grid cell (regular grid -> x/y dims on the dep/zb DataArray)
da_dep = get_bed_level(mod, SFINCS_ROOT, include_subgrid=True)
bed_here = da_dep.sel(x=x, y=y, method="nearest")
print(f"bed elevation at lagoon centroid: {float(bed_here.values):.3f} m")

zs_here = da_zs.sel(x=x, y=y, method="nearest")
zs_vals = zs_here.compute().values
print(
    f"zs over time at lagoon centroid: min={np.nanmin(zs_vals):.3f}, max={np.nanmax(zs_vals):.3f}"
)
print(f"zs at t=0: {zs_vals[0]:.3f}")
print(f"zs at t=last: {zs_vals[-1]:.3f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(zs_vals, marker=".", label="water level (zs)")
ax.axhline(float(bed_here.values), color="brown", linestyle="--", label="bed level")
ax.set_xlabel("output step")
ax.set_ylabel("level (m)")
ax.set_title("Lagoon centroid: water level vs bed level over the spin-up")
ax.legend()
fig.savefig("tests/_diag_lagoon_timeseries.png", dpi=130)
plt.show()
