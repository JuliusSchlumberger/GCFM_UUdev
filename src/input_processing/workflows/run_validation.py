"""Workflow script for running validation checks on input data products.

Collects all validation entry points in one place so the full validation
suite can be triggered with a single import. Currently includes:

- **test_GLOFAS**: compares the GloFAS v4.0 reanalysis dataset against other
  river data products (SWORD, Lin et al., GRIT) to verify spatial alignment
  for a chosen delta test case.

Todo:
    Convert this module into a CLI parser so each validation step can be
    called by name from the command line with configurable arguments.

Example:
    >>> from src.input_processing.workflows.run_validation import test_glofas
    >>> test_glofas("id_delta1")
"""

from __future__ import annotations

from src.input_processing.validation.river_input.test_GLOFAS import plot_glofas


def test_glofas(test_id: str) -> None:
    """Run the GloFAS spatial alignment check for a single delta test case.

    Loads the GloFAS reanalysis discharge, clips it to the delta extent, and
    produces a figure overlaying mean discharge with the Lin et al. and GRIT
    river networks. The figure is saved to the configured validation plots
    directory.

    Args:
        test_id: Key into ``config['Testcase']`` that maps to the numeric
            delta identifier, e.g. ``"id_delta1"``.

    Returns:
        None. The figure is saved to disk and displayed interactively by
        :func:`plot_glofas_lin`.

    Raises:
        KeyError: If *test_id* is not present in ``config['Testcase']``.

    Example:
        >>> test_glofas("id_delta1")
    """
    plot_glofas(test_id)
