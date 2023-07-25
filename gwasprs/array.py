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

def genotype_matrix_input_formatter(As, edge_axis=None, sample_axis=None, snp_axis=None, transpose=False):
    """
    As is a list: please give sample_axis and snp_axis
    As is an array: shape is (1,1), give sample_axis and snp_axis
                    shape is (1,1,1), give all axes
    As accepts different sample sizes but only same snp numbers.
    """

    # To list of 2d arrays
    sample_axis -= 1
    snp_axis -= 1

    if isinstance(As, (tuple, list)):
        pass

    elif len(As.shape) == 2:
        As = [As]
        sample_axis += 1
        snp_axis += 1

    elif len(As.shape) == 3:
        As = np.split(As, As.shape[edge_axis], axis=edge_axis)

    elif len(As[0].shape) == 2:
        # Dealing with np.array with multiple sizes of matrix (dtype=object)
        As = [As[edge_idx] for edge_idx in range(len(As))]
    
    else:
        raise ValueError('Unsupported data type.')

    if sample_axis == 1:
        As = [np.moveaxis(As[edge_idx], [sample_axis, snp_axis], [0,1]) for edge_idx in range(len(As))]

    if transpose:
        As = [A.T for A in As]

    return As
