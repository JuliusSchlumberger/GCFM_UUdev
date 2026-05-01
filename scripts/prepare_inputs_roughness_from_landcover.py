"""Command-line entry point for remapping land-cover to Manning's N roughness."""

from src.input_processing.workflows.prepare_inputs_wf_roughness_from_landcover import (
    get_roughness_from_landcover,
)


def main() -> None:
    """Remap ESA WorldCover land-cover to Manning's N roughness with default settings."""
    get_roughness_from_landcover()


if __name__ == "__main__":
    main()
