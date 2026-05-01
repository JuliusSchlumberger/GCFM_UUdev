"""CLI entry point for building a SFINCS hydrological model per delta.

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
"""

from __future__ import annotations

from src.modelling_framework.workflows.main_wf_01_build_model import build_model


def main() -> None:
    """Build a SFINCS model for the configured delta basin with debug plotting enabled."""
    build_model(debug_plotting=True)


if __name__ == "__main__":
    main()
