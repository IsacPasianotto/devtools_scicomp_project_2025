import numpy as np
from mpi4py import MPI
from mpi_utils import get_n_local,get_n_offset
def matmul(A, B, n_global, m_global, p_global, algorithm="base"):
    #TO DO: CAPIRE COME

    A= np.array(A)
    B= np.array(B)

    #Sanitize shape
    assert True

    if algorithm == "base":
        mm =  matmul_base
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_local = get_n_local(n_global,size,rank)
    # TO DO aggiungere dtype
    C = np.zeros(size=(n_local,p_global),
                 order='C')
    buffer = np.empty(size=(n_global,get_n_local(p_global,size,0)),
                      order='C')

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

        comm.Allgatherv(
            sendbuf=B[m_offset:m_offset+m_loc,p_offset_iter:p_offset_iter+p_loc_iter],
            recvbuf=(buffer, sendcounts, displacements, MPI.DOUBLE) #TODO correct data type
        )
        C[n_offset:n_offset+n_loc,p_offset:p_offset+p_loc]+=mm(A,B) #TODO insert a view


def matmul_base(A,B):
    pass



