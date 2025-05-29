"""Main module for the matmatmul package. Contains the main loop for the distributed matrix multiplication."""
import numpy as np
from mpi4py import MPI
from mpi4py.util import dtlib
from pymatmatmul.mpi_utils import get_n_local,get_n_offset
from numpy.typing import NDArray
from typing import Any, Callable
from numba import njit, prange
from pymatmatmul import AOT_BACKEND_DISPATCH,SUPPORTED_DTYPES
def matmul_naive(
        A: NDArray,
        B: NDArray
        ) -> NDArray:
    """
    Multiply two matrices A(n, m) and B(m, p) using the iterative method.
    Returns a matrix C(n, p) as the result of the multiplication.
    Note: This is a naive implementation which costs O(n * m * p) time complexity,
    hence it was implemented only to make comparison with other solution which are more efficient.

    Args:
    - A (np.ndarray): First matrix of shape (n, m).
    - B (np.ndarray): Second matrix of shape (m, p).
    Returns:
    - c (np.ndarray): Resulting matrix of shape (n, p) after multiplication.
    """
    n: int
    m: int
    p: int
    n, m = A.shape
    _, p = B.shape
    C: NDArray = np.zeros((n, p), dtype=A.dtype)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                C[i, j] += A[i, k] * B[k, j]
    return C


def matmul_numpy(
        A: NDArray,
        B: NDArray
        ) -> NDArray:
    """
    Multiply two matrices A and B using numpy's built-in function,
    which relies on underlying optimized libraries like BLAS or LAPACK.

    Args:
    - A (np.ndarray): First matrix of shape (n, m).
    - B (np.ndarray): Second matrix of shape (m, p).

    Returns:
    - np.ndarray: Resulting matrix of shape (n, p) after multiplication.
    """
    return np.matmul(A, B)

@njit(parallel=True)
def matmul_numbajit(
        A: NDArray,
        B: NDArray
        ) -> NDArray:
    """
    Multiply two matrices A and B using numba's JIT compilation for performance.

    Args:
    - A (np.ndarray): First matrix of shape (n, m).
    - B (np.ndarray): Second matrix of shape (m, p).

    Returns:
    - np.ndarray: Resulting matrix of shape (n, p) after multiplication.
    """
    n: int
    m: int
    p: int
    n, m = A.shape
    _, p = B.shape
    C: NDArray = np.zeros((n, p), dtype=A.dtype)
    for i in prange(n):
        for j in range(p):
            for k in range(m):
                C[i, j] += A[i, k] * B[k, j]
    return C

def matmul_numbaaot(
        A: NDArray,
        B: NDArray
        ) -> NDArray:
    """
    Multiply two matrices A and B using numba AOT (Ahead-Of-Time) compilation for performance.

    Args:
    - A (np.ndarray): First matrix of shape (n, m).
    - B (np.ndarray): Second matrix of shape (m, p).

    Returns:
    - np.ndarray: Resulting matrix of shape (n, p) after multiplication.
    """
    dtype: np.dtype = np.promote_types(A.dtype, B.dtype)
    if dtype not in SUPPORTED_DTYPES:
        raise NotImplementedError("Unsupported dtype %s. Supported dtypes are: %s" % (dtype, SUPPORTED_DTYPES))

    mm: Callable = AOT_BACKEND_DISPATCH.get(dtype)
    return mm(A, B)


def matmul(
        A: Any,
        B: Any,
        n_global: int,
        m_global: int,
        p_global: int,
        algorithm: str = "naive",
        dtype: Any = None,
        comm:MPI.Comm = None
        ) -> NDArray:
    """
    Function needed to be called by all MPI processes to perform a distributed matrix
    matrix multiplication.
    It implements a special case of the SUMMA algorithm, where each matrix is splitted
    only along rows (hence the number of column block is always 1).
    The function will try to cast the input matrices into numpy arrays with the given dtype
    if provided or try to promote the types of the input matrices.


    Args:
    - A (Any): First matrix to multiply, suggested to be a 2D array-like structure.
    - B (Any): Second matrix to multiply, suggested to be a 2D array-like structure.
    - n_global (int): Global number of rows in matrix A.
    - m_global (int): Global number of columns in matrix A and rows in matrix B.
    - p_global (int): Global number of columns in matrix B.
    - algorithm (str): Algorithm to use for the multiplicatiion. The default is "naive".
          available algorithms are: "naive", "numpy", "numba-jit" and "numba-aot.
    - dtype (Any, optional): Data type to cast the input matrices to. If None, it will
            promote the types of the input matrices.
    - comm (MPI.Comm, optional): MPI communicator to use. If not provided, a new MPI communicator
            will be istantiated using `MPI.COMM_WORLD`.

    Returns:
    - C (np.ndarray): Resulting matrix of shape (n_local, p_global) after multiplication,
    where n_local is the number of rows assigned to the current MPI process.

    Raises:
    - ValueError: If the input matrices are not 2D arrays, or if the shapes of the matrices
        are not compatible for multiplication, or if the algorithm is not supported.
    - RuntimeError: If there is an error converting the input matrices to numpy arrays.
    """

    try:
        A: NDArray = np.asarray(A,copy=None)
        B: NDArray = np.asarray(B,copy=None)
    except Exception as e:
        raise RuntimeError("Error converting input matrices to numpy arrays: %s" % e)

    if dtype is None:
        dtype: np.dtype = np.promote_types(A.dtype, B.dtype)

        #Check dtype and promote if necessary
        if A.dtype != B.dtype:
            A = A.astype(dtype)
            B = B.astype(dtype)
    else:
        try:
            dtype: np.dtype = np.dtype(dtype)
            A: NDArray = A.astype(dtype)
            B: NDArray = B.astype(dtype)
        except TypeError:
            raise ValueError("Error! The provided dtype: %s is not supported, or the conversion failed." % dtype)

    BACKEND_DISPATCH: dict[str, Callable] = {
        "naive": matmul_naive,
        "numpy": matmul_numpy,
        "numba-jit": matmul_numbajit,
        "numba-aot": matmul_numbaaot,
    }

    # Input validation
    if A.ndim != 2:
        raise ValueError("A must be a 2-dimensional array")
    if B.ndim != 2:
        raise ValueError("B must be a 2-dimensional array")
    if algorithm not in BACKEND_DISPATCH:
        raise ValueError("Algorithm %s not supported. Available algorithms are: %s" % (algorithm, BACKEND_DISPATCH.keys()))

    mm: Callable[NDArray, NDArray] = BACKEND_DISPATCH.get(algorithm)

    # MPI setup
    if comm is None:
        comm: MPI.Comm = MPI.COMM_WORLD
    if not isinstance(comm, MPI.Comm):
        raise TypeError("comm must be an instance of MPI.Comm, got %s" % type(comm))
    rank: int = comm.Get_rank()
    size: int = comm.Get_size()

    # Compute local sizes and offsets
    n_offset: int = get_n_offset(n_global, size, rank)
    n_loc: int = get_n_local(n_global, size, rank)
    m_offset: int = get_n_offset(m_global, size, rank)
    # Compute m_loc for each rank to avoid communicating them for the displacements
    m_locs: list[int] = [get_n_local(m_global, size, r) for r in range(size)]
    m_loc: int = m_locs[rank]

    C: NDArray = np.zeros(shape=(n_loc, p_global),
                 order='C', dtype=dtype)

    buffer: NDArray = np.empty(shape=(m_global,get_n_local(p_global,size,0)),
                      order='C', dtype=dtype)

    for k in range(size):

        p_offset_iter: int = get_n_offset(p_global, size, k)
        p_loc_iter: int = get_n_local(p_global, size, k)

        sendcounts: NDArray = np.array([m_locs[rank] * p_loc_iter for rank in range(size)])
        # use a view of the buffer to avoid copying data
        fit_buffer: NDArray = buffer.ravel()[:m_global*p_loc_iter].reshape((m_global, p_loc_iter))
        displacements: NDArray = np.insert(np.cumsum(sendcounts[:-1]), 0, 0)

        fit_buffer[m_offset:m_offset+m_loc,0:p_loc_iter] = np.ascontiguousarray(B[0:m_loc,p_offset_iter:p_offset_iter+p_loc_iter], dtype=dtype)

        # Gather the local blocks of B from all processes
        comm.Allgatherv( sendbuf=MPI.IN_PLACE,recvbuf=(fit_buffer, sendcounts, displacements, dtlib.from_numpy_dtype(dtype)))
        C[0:n_loc,p_offset_iter:p_offset_iter+p_loc_iter] = mm(A, fit_buffer)

    return C
