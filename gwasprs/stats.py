import numpy as np
import scipy.stats as stats
from jax import jit, vmap, pmap
from jax import numpy as jnp

from . import linalg, utils



def sum_of_square(A, mean):
    """Sum of square

    Make column-wise mean of A equal to 0 and calculate the column-wise sum of squares.
    original genotype_scaling_var_step

    Args:
        A (np.ndarray[(1, 1), np.floating]) : Genotype matrix with shape (samples, SNPs)
        mean (np.ndarray[(1,), np.floating]) : Vector
    
    Returns:
        (np.ndarray[(1, 1), np.floating]) : modified A matrix
        (np.ndarray[(1,), np.floating]) : sum of square vector
    """
    # Make the SNP mean = 0
    A = A - mean
    return A, jnp.sum(jnp.square(A), axis=0)

@jit
def normalize(norms, ortho):
    """Normalize the length of eigenvector

    Make the length of eigenvector equal to 1.
    original normalize_step

    Args:
        norms (np.ndarray[(1,), np.floating]) : Vector
        ortho (list of np.ndarray[(1,), np.floating]) : List

    Returns:
        (np.ndarray[(1, 1), np.floating]) : normalized eigenvectors as a matrix
    """
    ortho = jnp.asarray(ortho)
    norms = 1/jnp.sqrt(
        jnp.expand_dims(
            jnp.asarray(norms),
            -1
        )
    )
    return (norms * ortho).T

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


# PCA

def _vectorize(func, in_axes, out_axes):
    return vmap(jit(func), in_axes=in_axes, out_axes=out_axes)

def nansum(A):
    """Sum of matrix discarding NAs

    Perform nansum.
    original genotype_impute_step

    Args:
        A (np.ndarray[(1, 1), np.floating]) : Genotype matrix with shape (samples, SNPs)

    Returns
        (np.ndarray[(1,), np.floating]) : nansum of SNP as a vector
        (int) : sample count
    """
    snp_sum, sample_count = _vectorize(linalg.nansum, 1, 0)(A)
    return snp_sum, sample_count


def impute_and_local_mean(A, snp_mean):
    """Mean of imuted data

    Re-calculate mean of imputed data
    original genotype_scaling_mean_step

    Args:
        A (np.ndarray[(1, 1), np.floating]) : Genotype matrix with shape (samples, SNPs)
        snp_mean (np.ndarray[(1,), np.floating]) : mean of SNP as a vector
    
    Returns:
        (np.ndarray[(1, 1), np.floating]) : Imputed genotype matrix
        (np.ndarray[(1,), np.floating]) : sum of SNP as a vector
        (int) : sample count
    """
    # replace nan with means
    na_indices = jnp.where(jnp.isnan(A))
    A = A.at[na_indices].set(jnp.take(snp_mean, na_indices[1]))

    local_sum = _vectorize(jnp.sum, 1, 0)(A)
    local_count = A.shape[0]
    return A, local_sum, local_count


def aggregate_sums(local_sums, local_counts):
    """Aggregate local sums

    Collect local sums and sample counts and calculate the global mean
    original genotype_scaling_mean_step

    Args:
        local_sums (list of np.ndarray[(1,), np.floating]) : sum of SNP as a vector
        local_counts (list of integers) : sample count
    Return:
        (np.ndarray[(1,), np.floating]) : global mean of SNP
    """
    global_count = jnp.array(local_counts).sum(axis=0)
    global_mean = jnp.array(local_sums).sum(axis=0) / global_count

    return global_mean


def local_ssq(A, global_mean):
    """Calculate local sum of square
    
    Make column-wise mean of A equal to 0 and calculate the column-wise sum of squares.
    original genotype_scaling_var_step

    Args:
        A (np.ndarray[(1, 1), np.floating]) : Imputed genotype matrix
        global_mean (np.ndarray[(1,), np.floating]) : global mean of SNP
    
    Returns:
        (np.ndarray[(1, 1), np.floating]) : mean-shift genotype matrix
        (np.ndarray[(1,), np.floating]) : sum of square 
    """
    A, ssq = _vectorize(sum_of_square, (1,0), (1,0))(
        A,
        jnp.expand_dims(global_mean, -1)
    )

    return A, ssq


def aggregate_ssq(local_ssqs, local_counts):
    """Aggregate sums of square

    Collect local sums of square and delete SNPs with variance=0
    original genotype_scaling_var_step

    Args:
        local_ssqs (list of np.ndarray[(1,), np.floating]) : sum of square as a vector
        local_counts (list of integers) : sample counts

    Return:
        (np.ndarray[(1,), np.floating]) : global variance of SNP
        (np.ndarray[(1,), np.floating]) : boolean vector for whose variance = 0
    """
    global_count = jnp.array(local_counts).sum(axis=0)
    local_ssqs = jnp.array(local_ssqs)
    global_var = jnp.sum(local_ssqs, axis=0) / (global_count - 1)

    deleted = jnp.where(global_var == 0)[0]
    global_var = jnp.delete(global_var, deleted)

    return global_var, deleted


def standardize(A, global_var, deleted):
    """Standardize the preprocessed genotype matrix

    Perform the final step of standardization
    original genotype_standardization_step

    Args:
        A (np.ndarray[(1, 1), np.floating]) : Imputed genotype matrix with SNP mean=0
        global_var (np.ndarray[(1,), np.floating]) : global variance of SNP
        deleted (np.ndarray[(1,), np.floating]) : boolean vector for whose variance = 0
    
    Returns:
        np.ndarray[(1, 1), np.floating]) : Standardized genotype matrix
    """
    A = jnp.delete(A, deleted, axis=1)
    A = A / jnp.sqrt(global_var)

    return A
