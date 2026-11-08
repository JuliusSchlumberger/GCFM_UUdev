"""Write the river and surge peak travel times to the main mouth and the compound lag; run by rule event_timing.

River: walled travel time (src.river_profile.walled_travel_time) at the routed
bankfull discharge from the main inflow (largest bankfull discharge) along its
mainstem, with the sea at rule 11's calibration storm tide. It matched the
SFINCS mouth-discharge plateau on 2433835 and 3279946 (docs/river_depth_gvf.md).
Surge: sqrt(g h) from the water-level boundary to that mainstem's mouth.
lag_hr = surge - river travel: the river peak forced that much before the
surge peak reaches the mouth with it.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from src.event_timing import sea_travel_hr
from src.log import setup_logging
from src.river_network import (
    build_downstream_adjacency,
    normalize_reach_id,
    outflow_crossing_mask,
)
from src.river_profile import mainstem_profile, walled_travel_time

log = setup_logging(snakemake.log[0])
basin_id = str(snakemake.params.basin_id)

rivers = gpd.read_file(snakemake.input.river_network)
with xr.open_dataset(snakemake.input.river_forcing, decode_times=False) as ds:
    active = ds["has_glofas"].values.astype(bool) & ~outflow_crossing_mask(ds, rivers)
    if not active.any():
        raise RuntimeError(f"{basin_id}: no active river crossing -- no river travel time")
    bankfull = ds["bankfull_discharge_m3s"].values[active]
    main = int(np.nanargmax(bankfull))
    main_reach = normalize_reach_id(ds["inside_reach_id"].values[active][main])

nodes = pd.read_csv(snakemake.input.river_bed_nodes)
nodes["reach_id"] = nodes["reach_id"].map(normalize_reach_id)
q_bankfull = dict(
    zip(
        rivers["reach_id"].map(normalize_reach_id),
        rivers["bankfull_discharge_acc_m3s"].to_numpy(float),
    )
)
prof = mainstem_profile(
    nodes, build_downstream_adjacency(rivers), main_reach, reach_q=q_bankfull
)

# Rule 11's walled run: its level at the mainstem's last node is the sea level it held.
walled_path = Path(snakemake.input.walled_travel)
walled = pd.read_csv(walled_path, dtype={"crossing_reach_id": str})
walled = walled[walled["crossing_reach_id"].map(normalize_reach_id) == main_reach]
if walled.empty:
    raise RuntimeError(f"{basin_id}: {walled_path.name} has no rows for main inflow reach {main_reach}")
sea_level_m = float(walled["level_m"].iloc[-1])

river = walled_travel_time(
    prof,
    prof["reach_id"].map(q_bankfull).to_numpy(float),
    sea_level_m,
    float(snakemake.params.channel_manning_n),
    forcing_hr=float(snakemake.params.ramp_hr),
    froude_cap=float(snakemake.params.froude_cap),
    regime_rT=tuple(snakemake.params.wave_regime_rT),
)
if river["froude_capped"].any():
    log.info(f"Froude cap ({snakemake.params.froude_cap:g}) holds the depth at {int(river['froude_capped'].sum())} of {len(river)} node(s)")
river_travel_hr = float(river["t_wave_hr"].iloc[-1])

gis = Path(snakemake.input.skeleton_inp).parent / "gis"
with rasterio.open(gis / "dep.tif") as r_dep, rasterio.open(gis / "mask.tif") as r_msk:
    dep = r_dep.read(1, masked=True).filled(np.nan).astype(float)
    msk = r_msk.read(1)
    mouth_rc = r_dep.index(float(prof["x"].iloc[-1]), float(prof["y"].iloc[-1]))
    dx_m = abs(r_dep.transform.a)
surge_travel_hr, sea_path_km = sea_travel_hr(dep, msk, sea_level_m, mouth_rc, dx_m)
if not np.isfinite(surge_travel_hr):
    raise RuntimeError(f"{basin_id}: no wet path from the water-level boundary to the main mouth")

out = pd.DataFrame(
    [
        {
            "basin_id": basin_id,
            "main_reach_id": main_reach,
            "river_path_km": float(prof["dist_from_mouth_m"].iloc[0]) / 1e3,
            "q_bankfull_main_m3s": float(bankfull[main]),
            "sea_level_m": sea_level_m,
            "river_travel_hr": river_travel_hr,
            "sea_path_km": sea_path_km,
            "surge_travel_hr": surge_travel_hr,
            "lag_hr": surge_travel_hr - river_travel_hr,
        }
    ]
)
Path(snakemake.output.event_timing).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(snakemake.output.event_timing, index=False)
log.info(
    f"Event timing: river peak {river_travel_hr:.1f} h over {out['river_path_km'].iloc[0]:.1f} km "
    f"(bankfull {bankfull[main]:.0f} m3/s), surge peak {surge_travel_hr:.2f} h over {sea_path_km:.1f} km; "
    f"compound lag {out['lag_hr'].iloc[0]:+.1f} h"
)
