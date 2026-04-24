"""
This CLI script allows to automatically run the code to define the modelling domains per delta. It uses the following
input data:
Delta-Polygons (Edmonds et al. 2020): vectorized extend of the deltas globally, characterized by 4 points. As dataset is based on geomorphological identificaiton, it is not suited to determine the hydrologically relevant area for flood modelling, but is used to identify relevant basins that fall within the delta.
HYDRO-Basins (Lehner et al. 2013): vectorized polygon layers depicting sub-basin boundaries at global scale. They are provided on 12 levels according to "Pfafstetter" coding system.
                                    We use level 6 as an intermediate resolution based on preliminary testing of overlapping the basins with the Delta Polygons.
Delta-DTM (Pronk et al., modified by Seeger & minderhoud): coastal topography, used to identify and clip the coastal areas.
"""

import argparse

# from src.input_processing.validation.river_input.test_hydrobasin_level import
from src.input_processing.validation.river_input.test_domain_qsources import (
    test_domain_qsource_flow,
)


def main():
    parser = argparse.ArgumentParser(description="Define the model domain")
    # parser.add_argument("--choice",
    #                     help="Select which parts of the input processing you want to operate. The following are currently implemented: 'validate_glofas', ...", required=True)
    # parser.add_argument("--delta", help="specify which delta to run scripts for, see /config/decision.yaml for options",
    #                     default="id_delta1")

    args = parser.parse_args()

    # --- Tests ---

    # Test: workflow and plot figure showing intermediate results
    # test_validate_process_map()

    # Test: Randomly plot a set of deltas and the extraction points:
    test_domain_qsource_flow()

    # get_statistics_delta_domains(debug_plot=False)    # TODO explore how the process of using basin boundaries for all deltas with distance to coastlina < 10km


if __name__ == "__main__":
    main()
