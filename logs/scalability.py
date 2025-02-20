from pyclassify.utilsnumba import distance_numba
from pyclassify.utils import distance_numpy, distance
import time
import matplotlib.pyplot as plt
import numpy as np


times = []
times_numba = []
times_numpy = []
def main():
    # loop over the number of dimension
    N = 10
    for d in range(1, N):
        # create two points with d dimensions
        point1 = np.random.rand(d)
        point2 = np.random.rand(d)


        # compute the distances
        tstart = time.time()
        distance(point1, point2)
        times.append(time.time() - tstart)

        tstart = time.time()
        distance_numba(point1, point2)
        times_numba.append(time.time() - tstart)

        tstart = time.time()
        distance_numpy(point1, point2)
        times_numpy.append(time.time() - tstart)

    plt.plot(range(1, N), times, label='plain')
    plt.plot(range(1, N), times_numba, label='numba')
    plt.plot(range(1, N), times_numpy, label='numpy')
    plt.xlabel('Number of dimensions')
    plt.ylabel('Time')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
