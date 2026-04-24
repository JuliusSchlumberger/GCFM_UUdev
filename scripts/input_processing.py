
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Select and run specific tasks for input processing or validation"
    )
    parser.add_argument(
        "--choice",
        help="Select which parts of the input processing you want to operate. The following are currently implemented: 'validate_glofas', ...",
        required=True,
    )
    parser.add_argument(
        "--delta",
        help="specify which delta to run scripts for, see /config/decision.yaml for options",
        default="id_delta1",
    )

    args = parser.parse_args()

    if args.choice == "validate_glofas":
        print(
            "Testing GloFAS dataset: mapping data against other dataproducts of SWORD, MERIT."
        )
        from src.input_processing.workflows.run_validation import test_glofas

        test_glofas(args.delta)
    else:
        print("Selected input processing task is not available")


if __name__ == "__main__":
    main()
