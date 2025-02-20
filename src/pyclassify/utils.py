from math import sqrt
from statistics import mode
import yaml
import os
from line_profiler import profile
import numpy as np
from numba.pycc import CC
from numba import jit, prange

@profile
def distance(point1, point2):
    if len(point1) != len(point2):
        raise RuntimeError("Error: attempting to compute the distance of 2 points with different dimensions")
    dist = 0
    for i in range(len(point1)):
        dist += (point1[i] - point2[i])**2
    return sqrt(dist)

@profile
def distance_numpy(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

@profile
def majority_vote(neighbors):
    labels = list(set(neighbors))
    if len(labels) < 1:
        raise RuntimeError("Error, majority_vote methond applied to a list with less than 2 classes")
    return mode(neighbors)


def read_config(file):
   filepath = os.path.abspath(f'{file}.yaml')
   with open(filepath, 'r') as stream:
      kwargs = yaml.safe_load(stream)
   return kwargs

def read_file(filename):
    X, y = [], []

    with open(filename) as f:
        for line in f:
            values = line.split(',')
            y.append(0 if values[-1][0] == '0' else 1)
            X.append([float(i) for i in values[:-1]])
    return X, y
