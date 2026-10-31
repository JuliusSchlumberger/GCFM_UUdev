"""Plot a basin's clipped SWORD river network over the land mask.

Reads <results_dir>/<basin_id>/preprocessing_inputs/domain/; writes
<results_dir>/<basin_id>/figures/analysis/90_plot_river_network_map.png. The
cleaned network is plotted by rule clean_river_network itself.

Usage:
    python analysis/plot_river_network.py [basin_id]
"""

import logging
import sys
from types import SimpleNamespace

from _common import (
    REPO_ROOT,
    add_basin_id_argument,
    analysis_path,
    script_parser,
    site_path,
)

sys.path.insert(0, str(REPO_ROOT / "workflow"))
from src.domain import load_domain
from src.plots import plot_river_network

log = logging.getLogger(__name__)


def rule_io(basin_id: str) -> SimpleNamespace:
    """Return the inputs, params and outputs the former Snakemake rule passed in."""
    dom = site_path("results_dir") / basin_id / "preprocessing_inputs" / "domain"
    return SimpleNamespace(
        input=SimpleNamespace(
            spec_basins_meta=str(dom / "domain_bbox.json"),
            domain_gpkg=str(dom / f"{basin_id}_domain.gpkg"),
            spec_river_network=str(dom / f"{basin_id}_river_network.gpkg"),
            land_polygons=str(dom / f"{basin_id}_land_polygons.gpkg"),
        ),
        output=SimpleNamespace(
            plot_river_network=str(analysis_path(basin_id, __file__, "map.png"))
        ),
        params=SimpleNamespace(),
        wildcards=SimpleNamespace(basin_id=basin_id),
    )


def run(snk: SimpleNamespace) -> None:
    """Run the diagnostic for one basin; snk holds input, output, params and wildcards."""

    _, _, domain_poly = load_domain(snk.input.spec_basins_meta, snk.input.domain_gpkg)
    plot_river_network(
        snk.input.spec_river_network,
        domain_poly,
        snk.input.land_polygons,
        snk.output.plot_river_network,
    )
    log.info(f"Written: {snk.output.plot_river_network}")


def main() -> None:
    parser = script_parser(__doc__)
    add_basin_id_argument(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(rule_io(str(args.basin_id)))


if __name__ == "__main__":
    main()
