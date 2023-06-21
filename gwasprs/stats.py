import numpy as np
import scipy.stats as stats

from . import linalg

def unnorm_autocovariance(X: 'np.ndarray[(1, 1), np.floating]') -> 'np.ndarray[(1, 1), np.floating]':
    return linalg.mmdot(X, X)

def unnorm_covariance(X: 'np.ndarray[(1, 1), np.floating]', y: 'np.ndarray[(1,), np.floating]'):
    return linalg.mvmul(X.T, y)

def batched_unnorm_autocovariance(X: 'np.ndarray[(1, 1, 1), np.floating]') -> 'np.ndarray[(1, 1, 1), np.floating]':
    return linalg.batched_mmdot(X, X)

def batched_unnorm_covariance(X: 'np.ndarray[(1, 1, 1), np.floating]', y: 'np.ndarray[(1, 1), np.floating]'):
    return linalg.batched_mvmul(X.T, y)

def t_dist_pvalue(t_stat, df):
    return 2 * (1 - stats.t.cdf(np.abs(t_stat), df))

