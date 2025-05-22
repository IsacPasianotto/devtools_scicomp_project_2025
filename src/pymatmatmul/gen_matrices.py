"""Generate randomly matrices"""
import random as rnd
from typing import List, Any


def generate_random_matrices(
        n: int,
        m: int,
        p: int,
        backend: str = 'python'
        generationMin: float = 0.0,
        generationMax: float = 1.0
        ) -> Any:
    """
    Generate two random matrices A and B of dimensions n x m and m x p respectively.

    Args:
    - n (int): Number of rows in matrix A.
    - m (int): Number of columns in matrix A and number of rows in matrix B.
    - p (int): Number of columns in matrix B.
    - generationMin (float): Minimum value for the random number generation. Default is 0.0.
    - generationMax (float): Maximum value for the random number generation. Default is 1.0.
    - backend (str): Backend to use for generating the matrices. Default is 'python'.
      available options are 'python' and 'numpy'.

    Returns:
    - A Randomly generated matrix A of dimensions n x m, composed of random floats if
        backend is 'python' or numpy array if backend is 'numpy'.
    - B Randomly generated matrix B of dimensions m x p, composed of random floats if
        backend is 'python' or numpy array if backend is 'numpy'.
    """
    pass


def gen_random_matrix_python(
        n: int,
        m: int
        generationMin: float = 0.0,
        generationMax: float = 1.0
        ) -> List[List[float]]:
    """
    Generate a random matrix of dimensions n x m using Python's built-in random module.

    Args:
    - n (int): Number of rows in the matrix.
    - m (int): Number of columns in the matrix.
    - generationMin (float): Minimum value for the random number generation. Default is 0.0.
    - generationMax (float): Maximum value for the random number generation. Default is 1.0.

    Returns:
    - List[List[float]]: Randomly generated matrix of dimensions n x m.
    """
    return [[rnd.uniform(generationMin, generationMax) for _ in range(m)] for _ in range(n)]
