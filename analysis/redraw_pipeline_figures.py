"""Redraw a basin's pipeline figures (<results_dir>/<basin_id>/figures/) from saved outputs.

No model is built or run: each figure is drawn with the same src.plots call as
its rule, on the files that rule left behind. The coastal-weir map (rule 12a,
15_build_coastal_weir.png) needs the weir diagnostics held in memory during
the build and is not redrawn.

Usage: python analysis/redraw_pipeline_figures.py [basin_id]
(results_dir from config/site.yml or GCFM_RESULTS_DIR)
"""

import logging
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from _basin_inputs import calm_sea_level_m, load_config
from _common import add_basin_id_argument, script_parser, site_path
from src.domain import load_domain
from src.plots import (
    plot_crest_profiles,
    plot_initial_water_level,
    plot_max_inundation_map,
    plot_run_forcing,
    plot_skeleton_maps,
)
from src.postprocessing import (
    compute_max_inundation,
    land_bed_level,
    read_land_bed_level,
)
from src.sfincs_run import read_stability, stability_note

log = logging.getLogger(__name__)


def skeleton_jobs(root: Path, basin_id: str, figures: Path) -> dict[str, Callable]:
    skeleton = root / "sfincs_skeleton"
    land_mask = (
        root / "preprocessing_inputs" / "domain" / f"{basin_id}_land_mask_on_grid.gpkg"
    )

    def skeleton_maps() -> None:
        from hydromt_sfincs import SfincsModel

        sf = SfincsModel(
            root=str(skeleton),
            data_libs=[str(skeleton / "data_catalog_local.yml")],
            mode="r",
        )
        domain_dir = root / "preprocessing_inputs" / "domain"
        plot_skeleton_maps(
            sf.grid.data,
            sf.observation_points.gdf,
            load_domain(
                domain_dir / "domain_bbox.json", domain_dir / f"{basin_id}_domain.gpkg"
            )[2],
            str(land_mask),
            figures / "11_build_elevation.png",
            figures / "12_build_mask.png",
            figures / "13_build_roughness.png",
            basin_id=basin_id,
        )

    def zsini() -> None:
        from hydromt_sfincs import SfincsModel

        sf = SfincsModel(
            root=str(skeleton),
            data_libs=[str(skeleton / "data_catalog_local.yml")],
            mode="r",
        )
        domain_dir = root / "preprocessing_inputs" / "domain"
        plot_initial_water_level(
            skeleton / "zsini.tif",
            load_domain(
                domain_dir / "domain_bbox.json", domain_dir / f"{basin_id}_domain.gpkg"
            )[2],
            str(land_mask),
            figures / "14_build_zsini.png",
            calm_sea_level_m(basin_id),
            basin_id=basin_id,
            active_mask=sf.grid.data["mask"].values > 0,
        )

    return {"11-13 skeleton maps": skeleton_maps, "14 zsini": zsini}


def run_jobs(root: Path, basin_id: str, figures: Path) -> dict[str, Callable]:
    domain_dir = root / "preprocessing_inputs" / "domain"
    land_mask = str(domain_dir / f"{basin_id}_land_mask_on_grid.gpkg")
    rivers = str(domain_dir / f"{basin_id}_river_network_clean.gpkg")
    domain_poly = load_domain(
        domain_dir / "domain_bbox.json", domain_dir / f"{basin_id}_domain.gpkg"
    )[2]
    calib = root / "preprocessing_inputs" / "depth_crest_calibration"
    jobs: dict[str, Callable] = {}

    def forcing_and_depth(
        run_root: Path,
        label: str,
        hmax_of: Callable,
        hmin_m: float,
        stability: Path | None,
    ):
        def job() -> None:
            forcing_png = figures / f"20_forcing_timeseries__{label}.png"
            depth_png = figures / f"40_run_max_flood_depth__{label}.png"
            note = (
                stability_note(read_stability(stability))
                if stability and stability.exists()
                else None
            )
            spinup = label.startswith("spinup")
            # rule 13a labels every spin-up forcing figure "spinup" and flags it when unstable
            plot_run_forcing(
                run_root,
                forcing_png,
                basin_id,
                "spinup" if spinup else label,
                unstable_note=note if spinup else None,
            )
            da_hmax, _ = hmax_of(hmin_m)
            if da_hmax is None:
                raise RuntimeError(f"no zsmax in {run_root}")
            plot_max_inundation_map(
                da_hmax,
                domain_poly,
                land_mask,
                rivers,
                str(depth_png),
                basin_id=basin_id,
                run_label=label.replace("_slr", " slr "),
                hmin_m=hmin_m,
                unstable_note=note,
            )

        return job

    for round_root in sorted(calib.glob("round[1-9]")):
        label = f"calib_r{round_root.name.removeprefix('round')}"
        jobs[label] = forcing_and_depth(
            round_root,
            label,
            lambda h, r=round_root: compute_max_inundation(
                r, land_bed_level(r, calib / "ocean_mask_on_grid.tif"), hmin=h
            ),
            0.0,
            None,
        )

    skeleton_bed = root / "sfincs_skeleton" / "land_bed_level.nc"
    for spin_root in sorted((root / "spin_up").glob("slr_*")):
        label = f"spinup_slr{spin_root.name.removeprefix('slr_')}"
        jobs[label] = forcing_and_depth(
            spin_root,
            label,
            lambda h, r=spin_root: compute_max_inundation(
                r, read_land_bed_level(skeleton_bed), hmin=h
            ),
            0.0,
            spin_root / "stability.csv",
        )

    for run_root in sorted((root / "runs").glob("*/sfincs")):
        if not (run_root / "sfincs_map.nc").exists():
            continue
        label = run_root.parent.name
        jobs[label] = forcing_and_depth(
            run_root,
            label,
            lambda h, r=run_root: compute_max_inundation(
                r, read_land_bed_level(skeleton_bed), hmin=h
            ),
            float(load_config()["metrics"]["hmin"]),
            run_root / "stability.csv",
        )
    return jobs


def crest_profile_job(root: Path, basin_id: str, figures: Path) -> dict[str, Callable]:
    profile_dir = root / "preprocessing_inputs" / "visuals" / "11_calibration"

    def job() -> None:
        profiles = {}
        for csv in sorted(profile_dir.glob("round_profiles_seed*.csv")):
            df = pd.read_csv(csv)
            profiles[csv.stem.removeprefix("round_profiles_seed")] = df[
                df["round"] == df["round"].max()
            ]
        if not profiles:
            raise FileNotFoundError(f"no round_profiles_seed*.csv in {profile_dir}")
        plot_crest_profiles(basin_id, profiles, figures / "30_calib_crest_profile.png")

    return {"30 crest profile": job}


def main() -> None:
    parser = script_parser(__doc__)
    add_basin_id_argument(parser)
    basin_id = parser.parse_args().basin_id
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = site_path("results_dir") / basin_id
    figures = root / "figures"
    jobs = {
        **skeleton_jobs(root, basin_id, figures),
        **crest_profile_job(root, basin_id, figures),
        **run_jobs(root, basin_id, figures),
    }
    failed = []
    for name, job in jobs.items():
        try:
            job()
            print(f"redrawn: {name}")
        except Exception as exc:  # noqa: BLE001 -- report every failing figure, then exit non-zero
            failed.append(name)
            log.exception(f"failed: {name}: {exc}")
        finally:
            plt.close("all")
    print(
        f"{len(jobs) - len(failed)} of {len(jobs)} figure groups redrawn under {figures}"
    )
    if failed:
        sys.exit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
