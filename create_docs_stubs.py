"""Create all mkdocs documentation stub files.

Run this script once from the repo root to generate all the .md files
that mkdocs needs. Each file contains a single mkdocstrings directive
that pulls the docstrings from the corresponding Python module.

Usage:
    python create_docs_stubs.py
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Map of (docs_path, python_module_path) pairs
# ---------------------------------------------------------------------------

STUBS: list[tuple[str, str]] = [
    # Scripts
    ("docs/scripts/download_data.md", "scripts.download_data"),
    (
        "docs/scripts/data_merge_HydroBasins.md",
        "scripts.inputs_processing_data_merge_HydroBasins",
    ),
    ("docs/scripts/input_modify_delta_masks.md", "scripts.input_modify_delta_masks"),
    (
        "docs/scripts/preprocess_01_model_domain.md",
        "scripts.preprocess_01_model_domain",
    ),
    (
        "docs/scripts/preprocess_02_river_source_points.md",
        "scripts.preprocess_02_river_source_points",
    ),
    # Config
    (
        "docs/src/input_processing/config/loader.md",
        "src.input_processing.config.loader",
    ),
    # Utils
    (
        "docs/src/input_processing/utils/util_unify_typing_and_schema.md",
        "src.input_processing.utils.util_unify_typing_and_schema",
    ),
    (
        "docs/src/input_processing/utils/loading_files.md",
        "src.input_processing.utils.loading_files",
    ),
    (
        "docs/src/input_processing/utils/plotting.md",
        "src.input_processing.utils.plotting",
    ),
    (
        "docs/src/input_processing/utils/preprocess_01_ut_model_domains.md",
        "src.input_processing.utils.preprocess_01_ut_model_domains",
    ),
    (
        "docs/src/input_processing/utils/preprocess_02_ut_extract_river_points.md",
        "src.input_processing.utils.preprocess_02_ut_extract_river_points",
    ),
    (
        "docs/src/input_processing/utils/download_DeltaDTM_data.md",
        "src.input_processing.utils.download_DeltaDTM_data",
    ),
    # Utils - Validation
    (
        "docs/src/input_processing/utils/validation/modify_delta_masks.md",
        "src.input_processing.utils.validation.modify_delta_masks",
    ),
    # Workflows
    (
        "docs/src/input_processing/workflows/preprocess_01_wf_model_domain.md",
        "src.input_processing.workflows.preprocess_01_wf_model_domain",
    ),
    (
        "docs/src/input_processing/workflows/preprocess_02_wf_extract_river_points.md",
        "src.input_processing.workflows.preprocess_02_wf_extract_river_points",
    ),
    (
        "docs/src/input_processing/workflows/run_validation.md",
        "src.input_processing.workflows.run_validation",
    ),
    # Validation
    (
        "docs/src/input_processing/validation/river_input/test_GLOFAS.md",
        "src.input_processing.validation.river_input.test_GLOFAS",
    ),
    (
        "docs/src/input_processing/validation/river_input/test_delta_masks_modification.md",
        "src.input_processing.validation.river_input.test_delta_masks_modification",
    ),
    # Debugging
    (
        "docs/src/input_processing/debugging/debug_river_source_extraction.md",
        "src.input_processing.debugging.debug_river_source_extraction",
    ),
]


def title_from_path(module_path: str) -> str:
    """Convert a dotted module path to a human-readable title."""
    return module_path.split(".")[-1].replace("_", " ").title()


def create_stub(docs_path: str, module_path: str) -> None:
    """Create a single mkdocstrings stub file."""
    path = Path(docs_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    title = title_from_path(module_path)
    content = f"# {title}\n\n::: {module_path}\n"

    path.write_text(content, encoding="utf-8")
    print(f"  Created: {docs_path}")


def main() -> None:
    """Create all stub files."""
    print("Creating mkdocs stub files...\n")
    for docs_path, module_path in STUBS:
        create_stub(docs_path, module_path)
    print(f"\nDone — {len(STUBS)} files created.")
    print("Run 'mkdocs serve' to preview the documentation.")


if __name__ == "__main__":
    main()
