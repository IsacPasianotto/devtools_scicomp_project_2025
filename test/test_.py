import pytest
import pymatmatmul.gen_matrices
import pymatmatmul.matmul
from mpi4py import MPI

import numpy as np
from pymatmatmul.mpi_utils import gather_from_ranks_to_root

from src.pymatmatmul.mpi_utils import matrix_from_root_to_ranks


@pytest.mark.mpi_skip()
def test_generation_matrices():
    A = pymatmatmul.gen_matrices.generate_random_matrices( n = 3,m= 4,
                                 generationMin = 0.0,generationMax = 1.0,
                                 dtype= np.float32)
    assert A.shape == (3, 4)

@pytest.mark.mpi(min_size=2)
def test_mat_mul():
    comm = MPI.COMM_WORLD
    comm.Barrier()
    rank = comm.Get_rank()
    size = comm.Get_size()
    A = np.ones((50,49))
    B = np.ones((49, 35))
    C_expected = A@B
    A_local = matrix_from_root_to_ranks(A, comm)
    B_local = matrix_from_root_to_ranks(B, comm)
    C_local = pymatmatmul.matmul.matmul(A_local,B_local,50,49,35) # C shape 6,6
    C_computed = gather_from_ranks_to_root(C_local, comm)
    if rank == 0:
        assert np.array_equal(C_expected, C_computed)
    assert True

@pytest.mark.mpi(min_size=2)
def test_split_easy():
    comm = MPI.COMM_WORLD
    comm.Barrier()
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        A = np.ones((size,5),dtype=np.double)
        for id,row in enumerate(A):
            A[id,:] =id*row
    else:
        A = None
    A_local = matrix_from_root_to_ranks(A, comm)
    comm.Barrier()
    assert np.sum((A_local == rank))  == 5

@pytest.mark.mpi(min_size=2)
def test_rebuild_easy():
    comm = MPI.COMM_WORLD
    comm.Barrier()
    rank = comm.Get_rank()
    size = comm.Get_size()
    A_local = rank*np.ones((1,5),dtype=np.double)
    A = gather_from_ranks_to_root(A_local, comm)
    comm.Barrier()
    if rank == 0:
        A_expected = np.ones((size,5),dtype=np.double)
        for id,row in enumerate(A_expected):
            A_expected[id,:] =id*row
        assert np.array_equal(A, A_expected)
    else:
        assert A is None
