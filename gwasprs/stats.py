import numpy as np
import scipy.stats as stats
from jax import pmap

from . import linalg, utils

def unnorm_autocovariance(X: 'np.ndarray[(1, 1), np.floating]') -> 'np.ndarray[(1, 1), np.floating]':
    return linalg.mmdot(X, X)

def unnorm_covariance(X: 'np.ndarray[(1, 1), np.floating]', y: 'np.ndarray[(1,), np.floating]'):
    return linalg.mvdot(X, y)

def batched_unnorm_autocovariance(X: 'np.ndarray[(1, 1, 1), np.floating]', acceleration="single") -> 'np.ndarray[(1, 1, 1), np.floating]':
    if acceleration == "single":
        return linalg.batched_mmdot(X, X)
    elif acceleration == "pmap":
        pmap_func = pmap(linalg.batched_mmdot, in_axes = 3, out_axes = 3)
        ncores = utils.jax_cpu_cores()
        nsample, ndims, batch = X.shape
        minibatch, remainder = divmod(batch, ncores)
        A = np.reshape(X[:, :, :(minibatch*ncores)], (nsample, ndims, minibatch, ncores))
        Y = np.reshape(pmap_func(A, A), (ndims, ndims, -1))

        if remainder != 0:
            B = X[:, :, (minibatch*ncores):]
            Z = linalg.batched_mmdot(B, B)
            Y = np.concatenate((Y, Z), axis=2)
        return Y
    else:
        raise ValueError(f"{acceleration} acceleration is not supported.")

def batched_unnorm_covariance(X: 'np.ndarray[(1, 1, 1), np.floating]', y: 'np.ndarray[(1, 1), np.floating]'):
    return linalg.batched_mvdot(X, y)

def t_dist_pvalue(t_stat, df):
    return 2 * (1 - stats.t.cdf(np.abs(t_stat), df))

