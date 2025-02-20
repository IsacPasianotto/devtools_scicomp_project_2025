from .utils import distance, majority_vote

class kNN():
    def __init__(self, k):
        if not isinstance(k, int):
            raise TypeError("k must be an integer number")
        if k < 1:
            raise RuntimeError("k must be greater than one")
        self.k = k

    def _get_k_nearest_neighbors(self, X, y, x):
        distances = [distance(pt, x) for pt in X]
        srt_idx = sorted(range(len(distances)), key=distances.__getitem__)
        srt_idx = srt_idx[:self.k]
        return [y[i] for i in srt_idx]

    def __call__(self, data, new_points):
        
        X, y = data[0], data[1]
        knn = [self._get_k_nearest_neighbors(X, y, x) for x in new_points]
        prd = [majority_vote(i) for i in knn]
        self.predicted = prd
