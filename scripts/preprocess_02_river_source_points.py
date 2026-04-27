"""CLI entry point for extracting river source points for each delta domain.

Identifies the most-downstream GloFAS discharge cells that can serve as model
inflow points for each delta domain polygon. Uses three input datasets:

- **Delta polygons** (Edmonds et al. 2020): vectorised global delta extents
  characterised by four vertices. Used to identify which basins fall within
  each delta rather than to define the hydrologically relevant flood area
  directly.
- **HydroBasins** (Lehner et al. 2013): vectorised sub-basin boundary polygons
  at global scale, provided at 12 Pfafstetter coding levels. Level 6 is used
  as an intermediate resolution based on preliminary testing of overlap with
  the delta polygons.
- **DeltaDTM** (Pronk et al., modified by Seeger & Minderhoud): coastal
  topography dataset used to identify and clip coastal areas when deriving
  the inland domain boundary.

Example:
    Run from the command line::

        python scripts/preprocess_02_river_source_points.py

    Or call programmatically::

        >>> from scripts.preprocess_02_river_source_points import main
        >>> main()
"""

from __future__ import annotations

from src.input_processing.workflows.preprocess_02_wf_extract_river_points import (
    extract_points,
)


def main() -> None:
    """Run the river source point extraction pipeline with default configuration.

    Thin entry point that calls :func:`extract_points` with all file paths and
    settings read from the project config. Loads global datasets once, iterates
    over all delta domains, identifies the most-downstream GloFAS inflow points,
    and writes results to three output GeoPackage files.

    Returns:
        None. Output files are written to the paths defined in
        ``config['filepaths']``.

    Raises:
        FileNotFoundError: If the delta domains or basin domains GeoPackage
            cannot be found at the configured paths (propagated from
            :func:`extract_points`).

    Example:
        >>> main()
    """
    extract_points()


if __name__ == "__main__":
    main()
