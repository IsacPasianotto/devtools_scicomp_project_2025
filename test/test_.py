import pytest
import pymatmatmul.gen_matrices
import pymatmatmul.matmul

import numpy as np

@pytest.mark.mpi_skip()
def test_generation_matrices():
    A = pymatmatmul.gen_matrices.generate_random_matrices( n = 3,m= 4,
                                 generationMin = 0.0,generationMax = 1.0,
                                 dtype= np.float32)
    assert A.shape == (3, 4)

@pytest.mark.mpi(min_size=2)
def test_multiple_prints():
    print("aaaa")
    assert True
@pytest.mark.mpi(min_size=2,max_size=2)
def test_mat_mul():
    A = np.empty((3,4)) # global size: 6,4
    B = np.empty((2,6)) # global size  4,6
    c = pymatmatmul.matmul.matmul(A,B,6,4,6)

@pytest.mark.mpi_skip()
def test_single_print():
    print("bbbb")
    assert True
