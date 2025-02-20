from math import sqrt
from line_profiler import profile
import numpy as np
from numba.pycc import CC
from numba import jit, prange


cc = CC('utilsnumba')
@profile
@cc.export('distance_numba', 'f8(f8[:], f8[:])')
@jit(nopython=True, parallel=True)
def distance_numba(point1, point2):
    dist = 0
    for i in prange(len(point1)):
        dist += (point1[i] - point2[i])**2
    return sqrt(dist)

cc.compile()
