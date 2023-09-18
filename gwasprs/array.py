from typing import NewType

import numpy as np
import numpy.typing as npt
from scipy.stats import ortho_group
from jax import numpy as jnp
from jax import scipy as jsp
from jax import random

from . import linalg


Str1DArray = NewType("Str1DArray", npt.NDArray[np.byte])
IntNDArray = NewType("IntNDArray", npt.NDArray[np.int32])
FloatNDArray = NewType("FloatNDArray", npt.NDArray[np.float32])


def concat(xs, axis=1):
    return jnp.concatenate(xs, axis=axis)


def impute_with(X, val=0.0):
    return jnp.nan_to_num(X, copy=True, nan=val, posinf=None, neginf=None)


def expand_to_2dim(x, axis=-1):
    x = jnp.array(x)
    if jnp.ndim(x) == 1:
        x = jnp.expand_dims(x, axis)
    return x


def simulate_genotype_matrix(key, shape=(10,30), r_mask=0.9, c_mask=0.9, impute=False, standardize=False):
    # simulate genotype matrix with NAs
    X = random.randint(key=key, minval=0, maxval=3, shape=shape).astype('float32')
    mask_ridx = random.choice(key=key, a=shape[0], shape=(int(X.size*0.9),))
    mask_cidx = random.choice(key=key, a=shape[1], shape=(int(X.size*0.9),))
    X = X.at[mask_ridx,mask_cidx].set(jnp.nan)

    if impute:
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X = X.at[inds].set(np.take(col_mean, inds[1]))

    if standardize:
        X = (X-jnp.mean(X, axis=0))/jnp.nanstd(X, axis=0, ddof=1)
        X = jnp.delete(X, jnp.isnan(X[0]), axis=1)

    return X


def _subspace_iteration(A, G):
    H = linalg.mmdot(A.T, G)
    H, R = jsp.linalg.qr(H, mode='economic')
    G = linalg.mmdot(A, H)
    G, R = jsp.linalg.qr(G, mode='economic')
    return G, R


def simulate_eigenvectors(n, m, k, seed=42, iterations=10):
    A = linalg.randn(n, m, seed).T
    G = ortho_group.rvs(dim=n)
    for _ in range(iterations):
        G, R = _subspace_iteration(A, G)
    return G[:,:k], R[:,:k]