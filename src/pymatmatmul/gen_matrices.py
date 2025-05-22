"""Generate randomly matrices"""
from typing import List, Any
import numpy as np

def generate_random_matrices(
        n: int,
        m: int,
        p: int,
        generationMin: float = 0.0,
        generationMax: float = 1.0,
        dtype = np.double
        ) -> Any:
    """
    Generate two random matrices A and B of dimensions n x m and m x p respectively.

    Args:
    - n (int): Number of rows in matrix A.
    - m (int): Number of columns in matrix A and number of rows in matrix B.
    - p (int): Number of columns in matrix B.
    - generationMin (float): Minimum value for the random number generation. Default is 0.0.
    - generationMax (float): Maximum value for the random number generation. Default is 1.0.
    - dtype (np.dtype): Type of the random number generation. Default is np.double.


    Returns:
    - A Randomly generated matrix A of dimensions n x m, composed of random floats if
        backend is 'python' or numpy array if backend is 'numpy'.
    - B Randomly generated matrix B of dimensions m x p, composed of random floats if
        backend is 'python' or numpy array if backend is 'numpy'.
    """

    return ((generationMax-generationMin)*np.random.rand(n, m).astype(dtype)+generationMin,
            (generationMax-generationMin)*np.random.rand(m, p).astype(dtype)+generationMin)

