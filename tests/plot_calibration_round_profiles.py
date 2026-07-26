"""
plot_calibration_round_profiles.py -- Diagnostic: how do the calibrated
river bed, weir crest, and period-max water level change round-by-round
during iterative calibration (rule modelled_depth_estimation,
10_depth_estimation_modelled.py)?

Thin CLI wrapper around the same functions modelled_depth_estimation calls
automatically at the end of its own round loop
(src.river_depth_calibration.gather_calibration_round_profile,
src.plots.plot_calibration_round_profiles) -- reads each round's own
calibration_state.csv directly (modelled_depth_estimation's real per-cell
ground truth: dem, rivdph, weir_crest, zs (period-maximum water level)),
not a reconstruction/replay of its own update formulas. Useful for
regenerating these plots without re-running the whole calibration (e.g.
after only the plotting code changed), or for inspecting a basin run
before modelled_depth_estimation has produced them itself.

Path selection uses src.river_network.trace_widest_path (starting at each
is_seed reach, always continuing onto the widest in-domain downstream
candidate at a bifurcation) rather than trace_seed_mainstem_paths' own
main_path_id/is_mainstem_edge tagging, since SWORD's is_delta_outflow
attribute can be unreliable -- it may be False even for a reach that is
the network's actual outlet.

Outputs (figs/calibration_round_profiles/), one set per seed reach found:
  {basin_id}_seed{seed}_profiles_subplots.png  -- one subplot per round
  {basin_id}_seed{seed}_profiles_combined.png  -- all rounds overlaid
  {basin_id}_seed{seed}_profiles.csv           -- underlying per-cell data

Usage:
    conda run -n hmt_sfincs_dev python tests/plot_calibration_round_profiles.py [basin_id]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow"))
from src.plots import plot_calibration_round_profiles
from src.river_depth_calibration import gather_calibration_round_profile
from src.river_network import normalize_reach_id

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
for _name in ("hydromt", "hydromt_sfincs"):
    logging.getLogger(_name).setLevel(logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGS_DIR = REPO_ROOT / "figs" / "calibration_round_profiles"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

with open(REPO_ROOT / "config" / "config.yml") as fh:
    config = yaml.safe_load(fh)
RESULTS_DIR = Path(config["results_dir"])
N_ROUNDS = int(
    config["river_processing"]["river_depth_modelling"]["n_correction_iterations"]
)


def process_basin(basin_id: str) -> None:
    basin_dir = RESULTS_DIR / basin_id / "inputs"
    calib_root = RESULTS_DIR / basin_id / "sfincs_calibration"
    network_path = (
        basin_dir / "domain" / f"{basin_id}_river_network_depth_estimated.gpkg"
    )
    elevation_path = basin_dir / "domain" / f"{basin_id}_elevation_conditioned.tif"

    if not network_path.exists() or not calib_root.exists():
        log.warning(f"basin {basin_id}: missing network or calib_root, skipping")
        return

    last_state_csv = calib_root / f"round{N_ROUNDS}" / "calibration_state.csv"
    if not last_state_csv.exists():
        log.warning(
            f"basin {basin_id}: {last_state_csv} not found -- config's "
            f"n_correction_iterations ({N_ROUNDS}) may not match what's on disk, "
            f"or rule modelled_depth_estimation hasn't been (re-)run yet"
        )
        return
    stale_state_csv = calib_root / f"round{N_ROUNDS + 1}" / "calibration_state.csv"
    if (
        stale_state_csv.exists()
        and stale_state_csv.stat().st_mtime > last_state_csv.stat().st_mtime
    ):
        log.warning(
            f"basin {basin_id}: round{N_ROUNDS + 1}/calibration_state.csv is NEWER than "
            f"round{N_ROUNDS}/calibration_state.csv -- config's n_correction_iterations "
            f"({N_ROUNDS}) looks stale vs. what's actually on disk; results below may not "
            f"match the current config"
        )

    rivers = gpd.read_file(network_path)
    with rasterio.open(elevation_path) as src:
        utm_crs = src.crs
    rivers_utm = rivers.to_crs(utm_crs)

    seeds = sorted(
        {
            normalize_reach_id(rid)
            for rid, s in zip(rivers["reach_id"], rivers["is_seed"])
            if bool(s) is True
        }
        - {None}
    )
    if not seeds:
        log.warning(f"basin {basin_id}: no is_seed reach found, skipping")
        return

    for seed in seeds:
        path_rids, profiles_by_round = gather_calibration_round_profile(
            calib_root, rivers_utm, seed, N_ROUNDS
        )
        plot_calibration_round_profiles(
            basin_id=basin_id,
            seed=seed,
            profiles_by_round=profiles_by_round,
            n_rounds=N_ROUNDS,
            output_subplots_path=str(
                FIGS_DIR / f"{basin_id}_seed{seed}_profiles_subplots.png"
            ),
            output_combined_path=str(
                FIGS_DIR / f"{basin_id}_seed{seed}_profiles_combined.png"
            ),
        )
        pd.concat(
            [df.assign(round=i) for i, df in profiles_by_round.items()],
            ignore_index=True,
        ).to_csv(FIGS_DIR / f"{basin_id}_seed{seed}_profiles.csv", index=False)
        log.info(
            f"basin {basin_id} seed {seed}: wrote profile plots "
            f"({N_ROUNDS} correction round(s), path of {len(path_rids)} reach(es))"
        )


def main() -> None:
    basin_id = sys.argv[1] if len(sys.argv) > 1 else "2433835"
    process_basin(basin_id)


if __name__ == "__main__":
    main()
