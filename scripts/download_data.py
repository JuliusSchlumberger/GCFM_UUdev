"""Command-line entry point for downloading the DeltaDTM dataset.

Delegates to :func:`run_download`, which reads the configured CSV file,
auto-detects the URL column, and downloads all listed TIF files to the
configured output directory.

Example:
    Run from the command line::

        python scripts/download_data.py

    Or call programmatically::

        >>> from scripts.download_data import main
        >>> main()
"""

from __future__ import annotations

from src.input_processing.utils.download_DeltaDTM_data import run_download


def main() -> None:
    """Download all DeltaDTM TIF files listed in the configured CSV.

    Thin entry point that calls :func:`run_download` with no arguments,
    using all file paths and settings defined in
    :mod:`download_DeltaDTM_data`.

    Returns:
        None. Downloaded files are written to ``OUTPUT_DIR`` and any failures
        are logged to ``FAILED_LOG`` as defined in
        :mod:`download_DeltaDTM_data`.

    Raises:
        SystemExit: If the configured CSV file does not exist or no URL
            column can be detected (propagated from :func:`run_download`).

    Example:
        >>> main()
    """
    run_download()


if __name__ == "__main__":
    main()
