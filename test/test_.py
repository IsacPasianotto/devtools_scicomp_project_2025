import pytest
import pymatmatmul
import numpy as np

@pytest.mark.mpi_skip()
def test_generation_matrices():
    A,B = pymatmul.gen_matrices( n = 3,m= 4,p= 5,
                                 generationMin = 0.0,generationMax = 1.0,
                                 dtype= np.float)
    assert A.shape == (3,4)
    assert B.shape == (4,5)
@pytest.mark.mpi(min_size=2)
def test_multiple_prints():
    print("aaaa")
    assert True

@pytest.mark.mpi_skip()
def test_single_print():
    print("bbbb")
    assert True
