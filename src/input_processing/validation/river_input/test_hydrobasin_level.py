"""Workflow for building model domains across multiple Pfafstetter basin levels.

Iterates over the four configured Pfafstetter watershed levels (04–07) and
builds model domains for any level whose output file does not yet exist. This
allows the pipeline to be re-run safely without reprocessing already-completed
levels.

Example:
"""

from __future__ import annotations

from pathlib import Path

from src.input_processing.config.loader import config
from src.input_processing.utils.preprocess_01_ut_model_domains import create_model_domains


def compare_pfafstett_lvls() -> None:
    """Build model domains for each Pfafstetter level that has not yet been processed.

    Checks whether the output GeoPackage for each of the four Pfafstetter
    levels (04–07) already exists. If a file is present it is skipped; if it
    is absent, :func:`create_model_domains` is called to build and save it.

    The four levels correspond to increasingly coarse watershed delineations.
    Running all four allows downstream comparison of how the choice of
    Pfafstetter level affects the resulting model domains.

    Returns:
        None. Results are written to the output paths defined in
        ``config['filepaths']``.

    Raises:
        KeyError: If any of the expected keys (``river_basins_lvl04`` through
            ``river_basins_lvl07``, or their ``out_`` prefixed counterparts)
            are missing from the config.

    Example:
        >>> compare_pfafstett_lvls()
        Processing level river_basins_lvl04 ...
        Processing level river_basins_lvl06 ...
    """
    different_lvls: list[str] = [
        "river_basins_lvl04",
        "river_basins_lvl05",
        "river_basins_lvl06",
        "river_basins_lvl07",
    ]

    for lvl in different_lvls:
        if Path(config["filepaths"][lvl]).is_file():
            print(f"Skipping {lvl} — output already exists.")
            continue

        print(f"Processing level {lvl} ...")
        create_model_domains(config["filepaths"][f"out_{lvl}"])