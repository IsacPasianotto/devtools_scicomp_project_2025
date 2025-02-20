from .utils import distance, majority_vote, distance_numpy
from line_profiler import profile
import numpy as np
from .utilsnumba import distance_numba

class kNN():
    def __init__(self, k, backhand = 'plain'):
        if not isinstance(k, int):
            raise TypeError("k must be an integer number")
        if k < 1:
            raise RuntimeError("k must be greater than one")
        if backhand not in ['plain', 'numpy', 'numba']:
            raise RuntimeError("backhand must be either 'plain', 'numpy' or 'numba'")
        self.k = k
        self.backhand = backhand
        self.distance = distance if backhand == 'plain' else distance_numpy if backhand == 'numpy' else distance_numba

    @profile
    def _get_k_nearest_neighbors(self, X, y, x):
        distances = [self.distance(pt, x) for pt in X]
        srt_idx = sorted(range(len(distances)), key=distances.__getitem__)
        srt_idx = srt_idx[:self.k]
        return [y[i] for i in srt_idx]

    @profile
    def __call__(self, data, new_points):
        X, y = data[0], data[1]
        if self.backhand:
            X, new_points = np.array(X), np.array(new_points)
            y = np.array(y)
        knn = [self._get_k_nearest_neighbors(X, y, x) for x in new_points]
        prd = [majority_vote(i) for i in knn]
        self.predicted = prd
