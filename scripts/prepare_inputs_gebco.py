"""Command-line entry point for clipping GEBCO tiles to delta regions."""

from src.input_processing.workflows.prepare_inputs_wf_gebco import clip_and_merge_gebco


def main() -> None:
    """Clip GEBCO bathymetry tiles to delta bounding boxes and merge with default settings."""
    clip_and_merge_gebco()


if __name__ == "__main__":
    main()
