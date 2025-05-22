from mpi4py import MPI
def get_n_local(n_global,world_size,rank):

    return n_global // world_size + (1 if (rank < n_global % world_size) else 0)
def get_n_offset(n_global,world_size,rank):

    return sum([get_n_local(n_global,world_size,rank) for k in range(rank)])