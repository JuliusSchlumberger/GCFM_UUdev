from src.input_processing.validation.river_input.test_delta_masks_modification import (
    modify_test_delta_masks,
)

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Modify Edmond et al. 2020 polygons to include the entire coastline of a delta"
    )
    # parser.add_argument("--choice",
    #                     help="Select which parts of the input processing you want to operate. The following are currently implemented: 'validate_glofas', ...", required=True)
    # parser.add_argument("--delta", help="specify which delta to run scripts for, see /config/decision.yaml for options",
    #                     default="id_delta1")

    args = parser.parse_args()
    modify_test_delta_masks()


if __name__ == "__main__":
    main()
