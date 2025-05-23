import pytest
import pymatmatmul.gen_matrices
import pymatmatmul.matmul
from mpi4py import MPI

import numpy as np

@pytest.mark.mpi_skip()
def test_generation_matrices():
    A = pymatmatmul.gen_matrices.generate_random_matrices( n = 3,m= 4,
                                 generationMin = 0.0,generationMax = 1.0,
                                 dtype= np.float32)
    assert A.shape == (3, 4)

@pytest.mark.mpi(min_size=2)
def test_mat_mul():
    A = np.ones((3,4),dtype=np.double) # global size: 6,4
    B = np.ones((2,6),dtype=np.double) # global size  4,6
    C = pymatmatmul.matmul.matmul(A,B,6,4,6)
    print(C)
    assert np.sum((C == 4)) == 18
@pytest.mark.mpi_skip()
def test_single_print():
    print("bbbb")
    assert True
