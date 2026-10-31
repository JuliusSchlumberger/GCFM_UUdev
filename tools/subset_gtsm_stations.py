"""Read the hourly GTSM reanalysis once into the station subset rule extract_gtsm_series reads.

The stations are those behind every basin's surge_stations.json (rule
select_surge_stations), so run that rule for all basins first, e.g.
    snakemake --until select_surge_stations --cores 4
    python tools/subset_gtsm_stations.py
Rerun after adding a basin or changing a domain: rule extract_gtsm_series fails
on a station missing from the subset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "workflow"))
from src.gtsm import (  # noqa: E402
    build_station_subset,
    match_stations,
    station_coordinates,
)
from src.io import load_catalogue, raw_input_path, site_setting  # noqa: E402

STATIONS_FILE = "preprocessing_inputs/forcing/surge_stations.json"


def surge_station_files(results_dir: Path, basins: list[str] | None) -> list[Path]:
    """Return surge_stations.json of the given basins, else of every basin found, sorted."""
    if basins:
        files = [results_dir / b / STATIONS_FILE for b in basins]
        missing = [str(f) for f in files if not f.exists()]
        if missing:
            raise FileNotFoundError(
                f"run rule select_surge_stations first; missing: {missing}"
            )
        return files
    files = sorted(results_dir.glob(f"*/{STATIONS_FILE}"))
    if not files:
        raise FileNotFoundError(
            f"no */{STATIONS_FILE} under {results_dir}; run rule select_surge_stations first"
        )
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--basins", nargs="+", help="basin ids (default: every basin under results_dir)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="parallel monthly-file readers (default: 2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="subset file (default: catalogue gtsm_station_subset)",
    )
    args = parser.parse_args()

    config = yaml.safe_load(
        (REPO_ROOT / "config" / "config.yml").read_text(encoding="utf-8")
    )
    max_km = float(config["boundary_forcings"]["surge"]["hydrograph"]["max_match_km"])
    catalogue = load_catalogue(REPO_ROOT / "config" / "data_catalogue.yml")
    folders = {
        "waterlevel": raw_input_path(catalogue, "gtsm_total_water_level"),
        "surge": raw_input_path(catalogue, "gtsm_surge_residual"),
    }
    output = args.output or Path(raw_input_path(catalogue, "gtsm_station_subset"))

    files = surge_station_files(Path(site_setting("results_dir")), args.basins)
    _, gtsm_lon, gtsm_lat = station_coordinates(folders["surge"])
    station_idx: set[int] = set()
    for path in files:
        stations = json.loads(path.read_text())["stations"]
        idx, _ = match_stations(
            np.array([s["lon"] for s in stations]),
            np.array([s["lat"] for s in stations]),
            gtsm_lon,
            gtsm_lat,
            max_km,
        )
        station_idx.update(int(i) for i in idx)
    print(
        f"{len(station_idx)} GTSM station(s) from {len(files)} basin(s); reading every monthly file"
    )

    subset = build_station_subset(
        folders, np.array(sorted(station_idx)), n_workers=args.workers
    )
    subset.attrs["basins"] = " ".join(p.parents[2].name for p in files)
    n_time = subset.sizes["time"]
    encoding = {
        v: {"zlib": True, "complevel": 4, "chunksizes": (n_time, 1)}
        for v in ("waterlevel", "surge")
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".part")
    subset.to_netcdf(partial, encoding=encoding)
    partial.replace(output)
    print(
        f"wrote {output} ({subset.sizes['station']} stations, {n_time:,} hourly steps)"
    )


if __name__ == "__main__":  # the parallel readers re-import this file
    main()
