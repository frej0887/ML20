import numpy as np


def entropy(L: np.array):
    s = 0
    unique, counts = np.unique(L, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    for l in unique:
        p = counts_dict[l]/len(L)
        s -= p*np.log2(p)
    return s


def gain(X: np.array, Y: np.array, a: int):
    right = []
    for i, v in enumerate(np.unique(X[:, a])):
        Sv = Y[X[:, a] == v]
        right.append(len(Sv)/len(X[:, 0])*entropy(Sv))
    return entropy(Y) - sum(right)
