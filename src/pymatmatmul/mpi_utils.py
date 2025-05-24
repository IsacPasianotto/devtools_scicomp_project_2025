"""Utilities for MPI communication and matrix (de)composition for easier implementation of tests"""
from mpi4py import MPI
from mpi4py.util import dtlib
import numpy as np
from numpy.typing import NDArray
from logging import Logger


def get_n_local(Arows: int, world_size: int, rank: int) -> int:
    """
    Calculate the number of rows the passeshaped rank will recieve when splitting the matrix.
    In case of uneven division, the first ranks will receive one extra row.

    Args:
    - Arows (int): The total number of rows in the matrix.
    - world_size (int): The total number of ranks.
    - rank (int): The rank to calculate the number of rows for, commonly the current rank.

    Returns:
    - int: The number of rows assigned to the given rank.
    """
    return Arows // world_size + (1 if (rank < Arows % world_size) else 0)

def get_n_offset(Arows: int, world_size: int, rank: int) -> int:
    """
    Calculate the offset for the given rank w.r.t. the global matrix.

    Args:
    - Arows (int): The total number of rows in the matrix.
    - world_size (int): The total number of ranks.
    - rank (int): The rank to calculate the offset for, commonly the current rank.

    Returns:
    - int: The offset for the given rank.
    """
    min_rows: int = Arows // world_size
    remainder: int = Arows % world_size
    if rank < remainder:
        return rank * (min_rows + 1)
    return remainder * (min_rows + 1) + (rank - remainder) * min_rows


def matrix_from_root_to_ranks(
        A: NDArray,
        comm: MPI.Comm
        ) -> NDArray:
    """
    Split the given A matrix from the root rank to all other ranks breaking it by rows.
    The first ranks will receive one extra row in case of uneven division.

    Args:
    - A (np.ndarray): The matrix to be split.
    - comm (MPI.Comm): The MPI communicator.

    Returns:
    - np.ndarray: The local matrix assigned to the given rank.
    """
    world_size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        if world_size > A.shape[0]:
            raise ValueError("Number of ranks exceeds number of rows in A.")
        shape: tuple = A.shape
        dtype: np.dtype = A.dtype
        comm.bcast(shape, root=0)

        # Convert dtype to string for broadcasting
        comm.bcast(str(dtype), root=0)

        n_local: int = get_n_local(A.shape[0], world_size, rank)
        n_offset: int = get_n_offset(A.shape[0], world_size, rank)
        A_local: NDArray = A[n_offset:n_offset + n_local, :]

        for i in range(1, world_size):
            n_local_i: int = get_n_local(A.shape[0], world_size, i)
            n_offset_i: int = get_n_offset(A.shape[0], world_size, i)
            comm.Send([A[n_offset_i:n_offset_i + n_local_i, :], dtlib.from_numpy_dtype(dtype)], dest=i)
    else:
        shape: tuple = comm.bcast(None, root=0)
        dtype_str: str = comm.bcast(None, root=0)
        dtype: np.dtype = np.dtype(dtype_str)

        n_local: int = get_n_local(shape[0], world_size, rank)

        n_offset: int = get_n_offset(shape[0], world_size, rank)
        A_local: NDArray = np.empty((n_local, shape[1]), dtype=dtype)
        comm.Recv([A_local, dtlib.from_numpy_dtype(dtype)], source=0)

    return A_local


def gather_from_ranks_to_root(
        A_local: NDArray,
        comm: MPI.Comm
        ) -> NDArray:
    """
    Gather the local matrices from all ranks to the root rank.
    The root rank will concatenate the local matrices into a single matrix.
    The other ranks will return None.

    Args:
    - A_local (np.ndarray): The local matrix assigned to the given rank.
    - comm (MPI.Comm): The MPI communicator.
    Returns:
    - np.ndarray: The concatenated matrix on the root rank, None on other ranks.
    """
    rank = comm.Get_rank()
    A_gathered: NDArray = comm.gather(A_local, root=0)

    if rank == 0:
        A_full = np.concatenate(A_gathered, axis=0)
        return A_full
    return None

def print_matrix(
        A: NDArray,
        rank: int,
        world_size: int,
        comm: MPI.Comm,
        logger: Logger = None
        ) -> None:
    """
    Print the local matrix assigned to the given rank, in order fromrank 0 to rank n.
    The main purpose of this function is to visualize the local matrix assigned to
    each rank for debugging purposes.

    Args:
    - A (np.ndarray): The local matrix assigned to the given rank.
    - rank (int): The rank to print the matrix for.
    - world_size (int): The total number of ranks.
    - comm (MPI.Comm): The MPI communicator.
    - logger (Logger, optional): The logger to use for printing. If None, print to stdout.
    """
    fout = print if logger is None else logger.debug
    for i in range(world_size):
        comm.Barrier()
        if rank == i:
            fout("Rank %d matrix:" % rank)
            fout(A)
            fout("............................")
