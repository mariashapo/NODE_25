from __future__ import annotations
import numpy as np

def chebyshev_lobatto_nodes(N: int, t0: float, t1: float) -> np.ndarray:
    k = np.arange(N)
    x = np.cos(np.pi * k / (N - 1))[::-1]          # increasing
    t = (t1 + t0)/2 + (t1 - t0)/2 * x
    return t.astype(np.float64)

def chebyshev_diff_matrix(N: int, t0: float, t1: float) -> np.ndarray:
    k = np.arange(N)
    x = np.cos(np.pi * k / (N - 1))[::-1]
    c = np.ones(N); c[0] = c[-1] = 2.0
    c *= (-1.0) ** np.arange(N)
    X = np.tile(x, (N, 1))
    dX = X - X.T
    D = (np.outer(c, 1/c)) / (dX + np.eye(N))
    D = D - np.diag(D.sum(axis=1))
    D *= 2.0 / (t1 - t0)                           # scale to [t0,t1]
    return D.astype(np.float64)
