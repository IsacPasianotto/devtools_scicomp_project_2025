from pyclassify.utils import distance, majority_vote
from pyclassify import kNN
import pytest

def test_distance():
    # some points
    point0 = [1,2,3,4]
    point1 = [2,1,4,3]
    point2 = [0,1,2,3]

    with pytest.raises(AssertionError):
        assert distance(point0, point1) == 3
        assert distance(point0, point1) == distance(point1, point0)
        assert distance(point0, point1) >= 0
        assert distance(point0, point0) == 0
        assert distance(point0, point2) < distance(point0, point1) + distance(point1, point2)

def test_majority_vote():
    ex = [1, 0, 0, 0]
    assert majority_vote(ex) == 0

def constructor():
    k = 5
    obj = kNN(k)
    assert obj.k == k
