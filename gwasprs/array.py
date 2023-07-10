from jax import numpy as jnp
from jax import random
import numpy as np
from . import linalg
from scipy.stats import ortho_group
from jax import scipy as jsp

def concat(xs):
    return jnp.concatenate(xs, axis=1)

def impute_with(X, val=0.0):
    return jnp.nan_to_num(X, copy=True, nan=val, posinf=None, neginf=None)

def simulate_genotype_matrix(key, shape=(10,30), r_mask=0.9, c_mask=0.9, impute=False, standardize=False):
    # simulate genotype matrix with NAs
    X = random.randint(key=key, minval=0, maxval=3, shape=shape).astype('float32')
    mask_ridx = random.choice(key=key, a=shape[0], shape=(int(X.size*0.9),))
    mask_cidx = random.choice(key=key, a=shape[1], shape=(int(X.size*0.9),))
    X = X.at[mask_ridx,mask_cidx].set(jnp.nan)

    if impute:
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])

    if standardize:
        X = (X-np.mean(X, axis=0))/np.nanstd(X, axis=0, ddof=1)
        X = np.delete(X, np.isnan(X[0]), axis=1)
        
    return X

def _subspace_iteration(A, G):
    H = linalg.mmdot(A.T, G)
    H, R = jsp.linalg.qr(H, mode='economic')
    G = linalg.mmdot(A, H)
    G, R = jsp.linalg.qr(G, mode='economic')
    return G, R

def simulate_eigenvectors(n, m, seed=42, iterations=10):
    A = linalg.randn(n, m, seed)
    G = ortho_group.rvs(dim=m)
    for _ in range(iterations):
        G, R = _subspace_iteration(A, G)
    return G, R