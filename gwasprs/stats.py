import numpy as np
import scipy.stats as stats

from .linalg import mvdot, matmul

def unnorm_autocovariance(X: 'np.ndarray[(1, 1), np.floating]') -> 'np.ndarray[(1, 1), np.floating]':
    return matmul(X.T, X)

def unnorm_covariance(X: 'np.ndarray[(1, 1), np.floating]', y: 'np.ndarray[(1,), np.floating]'):
    return mvdot(X.T, y)

def t_dist_pvalue(t_stat, df):
    return 2 * (1 - stats.t.cdf(np.abs(t_stat), df))

