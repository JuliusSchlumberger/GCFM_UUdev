"""CLI entry point for defining hydrological modelling domains per delta.

Builds model domains by overlapping global delta polygons with Pfafstetter
river basins and filtering by river intersection. Uses three input datasets:

- **Delta polygons** (Edmonds et al. 2020): vectorised global delta extents
  characterised by four vertices. Based on geomorphological identification,
  this dataset is not suited for directly determining hydrologically relevant
  flood modelling areas, but is used to identify which basins fall within each
  delta.
- **HydroBasins** (Lehner et al. 2013): vectorised sub-basin boundary polygons
  at global scale, provided at 12 Pfafstetter coding levels. Level 6 is used
  as an intermediate resolution based on preliminary testing of overlap with
  the delta polygons.
- **SWORD** (Altenau et al. 2021): global river network dataset. Deltas with
  no intersecting river reach are excluded.

Example:
    Run from the command line::

        python scripts/preprocess_01_model_domain.py

    Or call programmatically::

        >>> from scripts.preprocess_01_model_domain import main
        >>> main()
"""

from __future__ import annotations

from src.input_processing.workflows.preprocess_01_wf_model_domain import (
    create_domains_chosen_level,
)


def main() -> None:
    """Run the model domain building pipeline with default configuration.

    Thin entry point that calls :func:`create_domains_chosen_level` with all
    file paths and settings read from the project config. Builds basin domains
    for all delta polygons that pass the area and river-intersection filters,
    and writes three output GeoPackage files.

    Returns:
        None. Output files are written to the paths defined in
        ``config['filepaths']``.

    Raises:
        ValueError: If any required schema column is missing from the loaded
            delta or river GeoDataFrames (propagated from
            :func:`create_domains_chosen_level`).

    Example:
        >>> main()
    """
    create_domains_chosen_level()


if __name__ == "__main__":
    main()
