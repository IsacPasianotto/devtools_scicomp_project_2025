"""Generate randomly matrices"""
from typing import List, Any
import numpy as np
from numpy.typing import NDArray

def generate_random_matrices(
        n: int,
        m: int,
        generationMin: float = 0.0,
        generationMax: float = 1.0,
        dtype = np.double
        ) -> NDArray:
    """

    Generate two random matrices A and B of dimensions n x m and m x p respectively.

    Args:
    - n (int): Number of rows in matrix A.
    - m (int): Number of columns in matrix A
    - generationMin (float): Minimum value for the random number generation. Default is 0.0.
    - generationMax (float): Maximum value for the random number generation. Default is 1.0.
    - dtype (np.dtype): Type of the random number generation. Default is np.double.


    Returns:
    - A Randomly generated matrix A of dimensions n x m, composed of random floats if
        backend is 'python' or numpy array if backend is 'numpy'.
        backend is 'python' or numpy array if backend is 'numpy'.
    """

    return (generationMax-generationMin)*np.random.rand(n, m).astype(dtype)+generationMin
