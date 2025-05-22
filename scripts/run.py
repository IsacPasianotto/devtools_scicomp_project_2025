"""Script which uses built packages to run the matmatmul code"""
import argparse
from logging import Logger
from pymatmatmul.utils import read_config, setup_logger, validate_config


logger: Logger = setup_logger("INFO")

def main(config_file: str):
    """
    Main function to run the matrix multiplication code.

    Args:
    - config_file (str): The name of the configuration file (without the extension).
    """

    logger.info("Reading configuration from: %s.yaml", config_file)
    kwargs = read_config(config_file)
    log_level = kwargs.get("logLevel", "INFO").upper()
    logger.setLevel(log_level)

    try:
        validate_config(kwargs)
    except (AttributeError, AssertionError, NotImplementedError) as e:
        logger.error("Configuration validation failed: %s", e, exc_info=True)
        exit(1)
    logger.debug("Configuration validated successfully.")
    logger.debug("Parsed config: %s", kwargs)

    if kwargs.get("genRandomMatrices"):
        logger.info("Generating random %dx%d &%dx%d matrices.",
                    kwargs["dimensions"]["A"][0], kwargs["dimensions"]["A"][1],
                    kwargs["dimensions"]["B"][0], kwargs["dimensions"]["B"][1])

    logger.info("TODO: Matrix multiplication operation would go here.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute the SUMMA algorithm to perform a matrix-matrix multiplication."
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="experiments/config",
        help="The name of the configuration file (without the extension).",
    )
    args = parser.parse_args()

    try:
        main(args.config_file)
    except Exception as e:
        logger.exception("The following error is occured, aborting the program: %s", e, exc_info=True)
        exit(1)
