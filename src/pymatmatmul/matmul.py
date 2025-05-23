import numpy as np
from mpi4py import MPI
from pymatmatmul.mpi_utils import get_n_local,get_n_offset
def matmul(A, B, n_global, m_global, p_global, algorithm="base"):
    #TO DO: CAPIRE COME

    A= np.ascontiguousarray(A)
    B= np.ascontiguousarray(B)

    #Sanitize shape
    assert True

    if algorithm == "base":
        mm =  matmul_base
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_local = get_n_local(n_global,size,rank)
    # TO DO aggiungere dtype
    C = np.zeros(shape=(n_local,p_global),
                 order='C',dtype=np.double)
    buffer = np.empty(shape=(m_global,get_n_local(p_global,size,0)),
                      order='C',dtype=np.double)

    n_offset = get_n_offset(n_global, size, rank)
    n_loc = get_n_local(n_global, size, rank)
    m_offset =get_n_offset(m_global, size, rank)
    m_loc = get_n_local(m_global, size, rank)

    p_loc = get_n_local(p_global, size, rank)
    p_offset = get_n_offset(p_global, size, rank)
    for k in range(size):


        p_offset_iter = get_n_offset(p_global, size, k)
        p_loc_iter = get_n_local(p_global, size, k)

        sendcounts = np.array(comm.allgather(m_loc*p_loc_iter)) #TODO remove communication

        displacements = np.insert(np.cumsum(sendcounts[:-1]), 0, 0)
        bufferino_contiguo = np.ascontiguousarray(B[0:m_loc,p_offset_iter:p_offset_iter+p_loc_iter], dtype=np.double)
        print(f"{rank=} {bufferino_contiguo.shape=} {sendcounts=}")

        comm.Allgatherv(
            sendbuf=bufferino_contiguo, # TODO consider in place
            recvbuf=(buffer, sendcounts, displacements, MPI.DOUBLE) #TODO correct data type
        )
        print(f"{rank=} {buffer}")
        C[0:n_loc,p_offset_iter:p_offset_iter+p_loc_iter]=mm(A,buffer)
    return C

def matmul_base(A,B):
    return A@B



