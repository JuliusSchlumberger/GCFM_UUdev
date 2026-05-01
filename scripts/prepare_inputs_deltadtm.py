"""Command-line entry point for selecting and copying Delta-DTM tiles."""

from src.input_processing.workflows.prepare_inputs_wf_deltadtm import (
    select_deltadtm_tiles,
)


def main() -> None:
    """Select and copy Delta-DTM tiles overlapping the configured study area."""
    select_deltadtm_tiles()


if __name__ == "__main__":
    main()
