"""Command-line entry point for modifying delta mask polygons.

Delegates to :func:`modify_test_delta_masks`, which loads the Edmonds et al.
(2020) delta polygons and iteratively expands any polygon whose offshore edges
do not fully reach the coastline.

Example:
    Run from the command line::

        python scripts/input_modify_delta_masks.py

    Or call programmatically::

        >>> from scripts.input_modify_delta_masks import main
        >>> main()
"""

from __future__ import annotations

from src.input_processing.validation.river_input.test_delta_masks_modification import (
    modify_test_delta_masks,
)


def main() -> None:
    """Run the delta mask modification pipeline with default settings.

    Thin entry point that calls :func:`modify_test_delta_masks` with all
    arguments at their configured defaults. Loads the Edmonds et al. (2020)
    delta polygons, corrects any polygon whose offshore edges do not reach the
    coastline, and writes the corrected set to the configured output GeoPackage.

    Returns:
        None. Results are written to the output path defined in
        ``config['filepaths']['output']``.

    Raises:
        FileNotFoundError: If the delta polygon file or land-use raster cannot
            be found at the configured paths (propagated from
            :func:`modify_test_delta_masks`).

    Example:
        >>> main()
    """
    modify_test_delta_masks()


if __name__ == "__main__":
    main()
