"""Generate randomly matrices"""
from typing import List, Any, Callable
import numpy as np
from numpy.typing import NDArray


def generate_random_matrix(
        n: int,
        m: int,
        generationMin: float = 0.0,
        generationMax: float = 1.0,
        dtype = np.double
        ) -> NDArray:
    """

    Generate a randomly filled matrix A and B of dimensions n x m.

    Args:
    - n (int): Number of rows in matrix A.
    - m (int): Number of columns in matrix A
    - generationMin (float): Minimum value for the random number generation. Default is 0.0.
    - generationMax (float): Maximum value for the random number generation. Default is 1.0.
    - dtype (np.dtype): Type of the random number generation. Default is np.double.


    Returns:
    - A (NDArray): A randomly filled matrix of shape (n, m) with values between generationMin and generationMax.
    Raises:
    - ValueError: If n or m are not positive integers, or if dtype is unsupported.
    - ValueError: If generationMin is greater than generationMax.
    """
    if not isinstance(n, int) or not isinstance(m, int):
        raise ValueError("Error! passed non integer values for matrix dimensions.")
    if n <= 0 or m <= 0:
        raise ValueError("Error, tried to generate a matrix with non positive dimensions.")

    try:
        dtype: np.dtype = np.dtype(dtype)
    except TypeError:
        raise ValueError("Error! Unsupported dtype %s. Please use a valid numpy dtype." % dtype)

    def generate_float():
        return (generationMax - generationMin) * np.random.rand(n, m).astype(dtype) + generationMin
    def generate_int():
        return np.random.randint(generationMin, generationMax + 1, size=(n, m), dtype=dtype)

    kind_dispatch: dict[str, Callable] = {
        'f': generate_float,  # float
        'i': generate_int,    # int
        'u': generate_int     # unsigned int
    }

    if dtype.kind not in kind_dispatch:
        raise ValueError("Unsupported dtype kind '%s'. Only float and int are supported." % dtype.kind)

    return kind_dispatch[dtype.kind]()
