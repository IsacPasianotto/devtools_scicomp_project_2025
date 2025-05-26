"""Code to compile AOT with numba the matmatmul function for performance."""
import numpy as np
from numba import njit, prange
from numba.pycc import CC


cc= CC('numba_compiled_matmul')


@cc.export('matmul_numbaaot_float64', 'float64[:,:](float64[:,:], float64[:,:])')
@njit(parallel=True)
def matmul_numbaaot(
        A: 'float64[:,:]',
        B: 'float64[:,:]'
        ) -> 'float64[:,:]':
    """
    Multiply two matrices A and B using numba AOT (Ahead-Of-Time) compilation for performance.
    Function compiled to handle float64 data types.

    Args:
    - A (np.ndarray): First matrix of shape (n, m).
    - B (np.ndarray): Second matrix of shape (m, p).

    Returns:
    - np.ndarray: Resulting matrix of shape (n, p) after multiplication using float64 data type.
    """
    n: int
    m: int
    p: int
    n, m = A.shape
    _, p = B.shape
    C: 'float64[:,:]' = np.zeros((n, p), dtype=A.dtype)
    for i in prange(n):
        for j in range(p):
            for k in range(m):
                C[i, j] += A[i, k] * B[k, j]
    return C


@cc.export('matmul_numbaaot_float32', 'float32[:,:](float32[:,:], float32[:,:])')
@njit(parallel=True)
def matmul_numbaaot_float32(
        A: 'float32[:,:]',
        B: 'float32[:,:]'
        ) -> 'float32[:,:]':
    """
    Multiply two matrices A and B using numba AOT (Ahead-Of-Time) compilation for performance.
    Function compiled to handle float32 data types.

    Args:
    - A (np.ndarray): First matrix of shape (n, m).
    - B (np.ndarray): Second matrix of shape (m, p).

    Returns:
    - np.ndarray: Resulting matrix of shape (n, p) after multiplication using float32 data type.
    """
    n: int
    m: int
    p: int
    n, m = A.shape
    _, p = B.shape
    C: 'float32[:,:]' = np.zeros((n, p), dtype=A.dtype)
    for i in prange(n):
        for j in range(p):
            for k in range(m):
                C[i, j] += A[i, k] * B[k, j]
    return C


@cc.export('matmul_numbaaot_int64', 'int64[:,:](int64[:,:], int64[:,:])')
@njit(parallel=True)
def matmul_numbaaot_int64(
        A: 'int64[:,:]',
        B: 'int64[:,:]'
        ) -> 'int64[:,:]':
    """
    Multiply two matrices A and B using numba AOT (Ahead-Of-Time) compilation for performance.
    Function compiled to handle int64 data types.

    Args:
    - A (np.ndarray): First matrix of shape (n, m).
    - B (np.ndarray): Second matrix of shape (m, p).

    Returns:
    - np.ndarray: Resulting matrix of shape (n, p) after multiplication using int64 data type.
    """
    n: int
    m: int
    p: int
    n, m = A.shape
    _, p = B.shape
    C: 'int64[:,:]' = np.zeros((n, p), dtype=A.dtype)
    for i in prange(n):
        for j in range(p):
            for k in range(m):
                C[i, j] += A[i, k] * B[k, j]
    return C

@cc.export('matmul_numbaaot_int32', 'int32[:,:](int32[:,:], int32[:,:])')
@njit(parallel=True)
def matmul_numbaaot_int32(
        A: 'int32[:,:]',
        B: 'int32[:,:]'
        ) -> 'int32[:,:]':
    """
    Multiply two matrices A and B using numba AOT (Ahead-Of-Time) compilation for performance.
    Function compiled to handle int32 data types.

    Args:
    - A (np.ndarray): First matrix of shape (n, m).
    - B (np.ndarray): Second matrix of shape (m, p).

    Returns:
    - np.ndarray: Resulting matrix of shape (n, p) after multiplication using int32 data type.
    """
    n: int
    m: int
    p: int
    n, m = A.shape
    _, p = B.shape
    C: 'int32[:,:]' = np.zeros((n, p), dtype=A.dtype)
    for i in prange(n):
        for j in range(p):
            for k in range(m):
                C[i, j] += A[i, k] * B[k, j]
    return C

@cc.export('matmul_numbaaot_int16', 'int16[:,:](int16[:,:], int16[:,:])')
@njit(parallel=True)
def matmul_numbaaot_int16(
        A: 'int16[:,:]',
        B: 'int16[:,:]'
        ) -> 'int16[:,:]':
    """
    Multiply two matrices A and B using numba AOT (Ahead-Of-Time) compilation for performance.
    Function compiled to handle int16 data types.

    Args:
    - A (np.ndarray): First matrix of shape (n, m).
    - B (np.ndarray): Second matrix of shape (m, p).

    Returns:
    - np.ndarray: Resulting matrix of shape (n, p) after multiplication using int16 data type.
    """
    n: int
    m: int
    p: int
    n, m = A.shape
    _, p = B.shape
    C: 'int16[:,:]' = np.zeros((n, p), dtype=A.dtype)
    for i in prange(n):
        for j in range(p):
            for k in range(m):
                C[i, j] += A[i, k] * B[k, j]
    return C


if __name__ == "__main__":
    cc.compile()
