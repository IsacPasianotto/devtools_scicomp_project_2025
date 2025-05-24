"""Script which uses built packages to run the matmatmul code"""
import argparse
import numpy as np
from logging import Logger
from mpi4py import MPI
from pymatmatmul.utils import read_config, setup_logger, validate_config
from pymatmatmul.mpi_utils import get_n_local, print_matrix
from pymatmatmul.gen_matrices import generate_random_matrix
from numpy.typing import NDArray

logger: Logger = setup_logger("INFO")

def main(config_file: str, comm: MPI.Comm) -> None:
    """
    Main function to run the matrix multiplication code.

    Args:
    - config_file (str): The name of the configuration file (without the extension).
    - comm (MPI.Comm): The MPI communicator object.

    Returns:
    - None
    """
    rank: int = comm.Get_rank()
    size: int = comm.Get_size()
    if rank == 0:
        logger.info("Reading configuration from: %s.yaml", config_file)
    kwargs = read_config(config_file)
    log_level = kwargs.get("logLevel", "INFO").upper()
    logger.setLevel(log_level)

    try:
        validate_config(kwargs)
    except (AttributeError, AssertionError, NotImplementedError) as e:
        logger.error("Configuration validation failed: %s", e, exc_info=True)
        exit(1)

    if rank == 0:
        logger.debug("Configuration validated successfully.")
        logger.debug("Parsed config: %s", kwargs)
        logger.info("Generating random %dx%d &%dx%d matrices.",
                    kwargs["dimensions"]["A"][0], kwargs["dimensions"]["A"][1],
                    kwargs["dimensions"]["B"][0], kwargs["dimensions"]["B"][1])

    local_A_rows: int = get_n_local(kwargs["dimensions"]["A"][0], size, rank)
    local_B_rows: int = get_n_local(kwargs["dimensions"]["B"][0], size, rank)
    logger.debug("Rank %d will handle %d rows of A and %d rows of B.", rank, local_A_rows, local_B_rows)

    if kwargs.get("genRandomMatrices"):
    # else case not implemented yet, but already thrown in the validation
        A_local: NDArray = generate_random_matrix(local_A_rows,
                                                  kwargs["dimensions"]["A"][1],
                                                  kwargs.get("generationMin"),
                                                  kwargs.get("generationMax"),
                                                  dtype=kwargs.get("dtype")
                                                  )
        B_local: NDArray = generate_random_matrix(local_B_rows,
                                                  kwargs["dimensions"]["B"][1],
                                                  kwargs.get("generationMin"),
                                                  kwargs.get("generationMax"),
                                                  dtype=kwargs.get("dtype")
                                                  )
        logger.debug("Rank %d has the following elements ofA:\n%s", rank, A_local)
        logger.debug("Rank %d has the following elements of B:\n%s", rank, B_local)

        # do it only for debug to not waste communication time in real run
        if log_level == "DEBUG":
            if rank == 0:
                logger.debug("------ Matrix A ---")
            print_matrix(A_local, rank, size, comm, logger)
            if rank == 0:
                logger.debug("------ Matrix B ---")
            print_matrix(B_local, rank, size, comm, logger)



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
        comm: MPI.Comm = MPI.COMM_WORLD
        rank: int = comm.Get_rank()
        size: int = comm.Get_size()

        if rank == 0:
            logger.info("MPI initialized successfully with %d processes.", size)
        logger.debug("I am %d of %d", rank, size)

        main(args.config_file, comm)
    except Exception as e:
        logger.exception("The following error is occured, aborting the program: %s", e, exc_info=True)
        exit(1)
