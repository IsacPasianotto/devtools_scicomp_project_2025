import pytest
import pymatmatmul.gen_matrices
import pymatmatmul.matmul
from mpi4py import MPI

import numpy as np
from pymatmatmul.mpi_utils import gather_from_ranks_to_root
from pymatmatmul.mpi_utils import matrix_from_root_to_ranks
from pymatmatmul import SUPPORTED_DTYPES


@pytest.mark.mpi_skip()
def test_generation_matrices():
    A = pymatmatmul.gen_matrices.generate_random_matrix( n = 3,m= 4,
                                 generationMin = 0.0,generationMax = 1.0,
                                 dtype= np.float32)
    assert A.shape == (3, 4)
@pytest.mark.mpi_skip()
@pytest.mark.parametrize("f", [ pymatmatmul.matmul.matmul_naive,pymatmatmul.matmul.matmul_numpy,pymatmatmul.matmul.matmul_numbajit,pymatmatmul.matmul.matmul_numbaaot])
@pytest.mark.parametrize("m, n, p", [
    (50, 49, 35),
    (60, 40, 30),
    (100, 100, 100),
    (20, 50, 10),
])
@pytest.mark.parametrize('dtype', SUPPORTED_DTYPES)
def test_serial_matmul(f,m,n,p,dtype):
    A = pymatmatmul.gen_matrices.generate_random_matrix( n = n,m= m,
                                 generationMin = 0.0,generationMax = 10.0,
                                 dtype= dtype)
    B = pymatmatmul.gen_matrices.generate_random_matrix( n = m,m= p,
                                 generationMin = 0.0,generationMax = 10.0,
                                 dtype= dtype)
    C = f(A,B)
    assert np.allclose(C,A@B)
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("m, n, p", [
    (50, 49, 35),
    (60, 40, 30),
    (100, 100, 100),
    (20, 50, 10),
])
@pytest.mark.parametrize("backend", ["naive", "numpy", "numba-jit", "numba-aot"])
@pytest.mark.parametrize('dtype', SUPPORTED_DTYPES)
def test_mat_mul(m, n, p,backend,dtype):
    comm = MPI.COMM_WORLD
    comm.Barrier()
    rank = comm.Get_rank()

    A = np.ones((m, n),dtype=dtype)
    B = np.ones((n, p),dtype=dtype)
    C_expected = A @ B

    A_local = matrix_from_root_to_ranks(A, comm)
    B_local = matrix_from_root_to_ranks(B, comm)

    C_local = pymatmatmul.matmul.matmul(A_local, B_local, m, n, p,algorithm=backend)
    C_computed = gather_from_ranks_to_root(C_local, comm)

    if rank == 0:
        assert np.allclose(C_expected, C_computed), f"Matrix multiplication failed for ({m},{n}) x ({n},{p})" # CS101, avoid == on floating point number
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
