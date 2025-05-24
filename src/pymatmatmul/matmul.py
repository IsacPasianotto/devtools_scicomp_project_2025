"""Main module for the matmatmul package. Contains the main loop for the distributed matrix multiplication."""
import numpy as np
from mpi4py import MPI
from mpi4py.util import dtlib
from pymatmatmul.mpi_utils import get_n_local,get_n_offset
from pymatmatmul.utils import get_valid_backends

def matmul_base(A,B):
    return A@B


def matmul(A, B, n_global, m_global, p_global, algorithm="base"):
    ###################################
    # Input check                     #
    ###################################
    # Convert as np.array
    A = np.asarray(A,copy=None)
    B = np.asarray(B,copy=None)
    '''
    https://numpy.org/doc/2.1/reference/generated/numpy.asarray.html
    >>> a = np.array([1, 2], dtype=np.float32)
    >>> np.shares_memory(np.asarray(a, dtype=np.float32), a)
    True
    >>> np.shares_memory(np.asarray(a, dtype=np.float64), a)
    False
    '''
    dtype = np.promote_types(A.dtype, B.dtype)
    #Check dtype and promote if necessary
    if A.dtype != B.dtype:
        A = A.astype(dtype)
        B = B.astype(dtype)

    # Check shape, consider broadcast  if array or insert a new axis
    if A.ndim != 2:
        raise ValueError("A must be a 2-dimensional array")
    if B.ndim != 2:
        raise ValueError("B must be a 2-dimensional array")

    if algorithm not in get_valid_backends():
        raise ValueError("Algorithm %s not supported. Available algorithms are: %s" % (algorithm, get_valid_backends()))

    # TODO: remove this
    if algorithm in ["numpy", "numba"]:
        raise NotImplementedError("Algorithm %s not implemented yet. Available algorithms are: %s" % (algorithm, get_valid_backends()))

    # with python >= 3.10 we can move towards case switch
    if algorithm == "base":
        mm =  matmul_base
    # Init mpi
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_offset = get_n_offset(n_global, size, rank)
    n_loc = get_n_local(n_global, size, rank)
    m_offset =get_n_offset(m_global, size, rank)
    m_loc = get_n_local(m_global, size, rank)

    C = np.zeros(shape=(n_loc, p_global),
                 order='C', dtype=dtype)

    buffer = np.empty(shape=(m_global,get_n_local(p_global,size,0)),
                      order='C', dtype=dtype)

    for k in range(size):

        p_offset_iter = get_n_offset(p_global, size, k)
        p_loc_iter = get_n_local(p_global, size, k)

        sendcounts = np.array(comm.allgather(m_loc*p_loc_iter)) #TODO remove communication
        #use ravel instead of flatten, ravel is in place, flatten return a copy
        fit_buffer = buffer.ravel()[:m_global*p_loc_iter].reshape((m_global, p_loc_iter))
        displacements = np.insert(np.cumsum(sendcounts[:-1]), 0, 0)
        fit_buffer[m_offset:m_offset+m_loc,0:p_loc_iter] = np.ascontiguousarray(B[0:m_loc,p_offset_iter:p_offset_iter+p_loc_iter], dtype=dtype)
        comm.Allgatherv(
            sendbuf=MPI.IN_PLACE, # TODO consider in place MPI.IN_PLACE
            recvbuf=(fit_buffer, sendcounts, displacements, dtlib.from_numpy_dtype(dtype)  ) #TODO correct data type ->  mpi4py.util.dtlib.from_numpy_dtype(dtype)
        )
        C[0:n_loc,p_offset_iter:p_offset_iter+p_loc_iter]=mm(A, fit_buffer)
    return C
