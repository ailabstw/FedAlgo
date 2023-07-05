import numpy as np
import scipy.stats as stats
from jax import jit, pmap
from jax import numpy as jnp
import logging

from . import linalg, vectorize, utils



def sum_of_square(A, mean):
    # Make the SNP mean = 0
    A = A - mean
    return A, jnp.sum(jnp.square(A), axis=0)

@jit
def normalize(norms, ortho):
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

def federated_standardize(A):
    A = impute_with_zero(A)
    local_sum, local_count = impute_and_local_mean(A, snp_mean)
    global_mean, global_count = aggregate_sums([local_sum], [local_count])
    local_ssq = local_ssq(A, global_mean)
    global_var, deleted = aggregate_ssq([local_ssq], global_count)
    A = standardize(A, global_var, deleted)
    return A

def impute_with_zero(A):
    """
    original genotype_impute_step
    """
    snp_sum, sample_count = vectorize.vmap_jit(linalg.nansum, 1, 0)(A)
    logging.debug(f'SNP sum: {snp_sum}')
    logging.debug(f'Sample counts: {sample_count}')
    return snp_sum, sample_count


def impute_and_local_mean(A, snp_mean):
    """
    Re-calculate mean of imputed data
    original genotype_scaling_mean_step
    """
    # replace nan with means
    na_indices = jnp.where(jnp.isnan(A))
    A = A.at[na_indices].set(jnp.take(snp_mean, na_indices[1]))

    local_sum = vectorize.vmap_jit(jnp.sum, 1, 0)(A)
    local_count = A.shape[0]
    logging.debug(f'SNP sum: {local_sum}')
    logging.debug(f'Sample counts: {local_count}')
    return local_sum, local_count


def aggregate_sums(local_sums, local_counts):
    """
    original genotype_scaling_mean_step

    Args:
        mean_scalars
            a list containing SNP_sums and sample_count with format [(SNP_sums, sample_count),()...]
    Return:
        global_mean
            the feature means
    """
    global_count = jnp.array(local_counts).sum(axis=0)
    global_mean = jnp.array(local_sums).sum(axis=0) / global_count
    logging.debug(f'Total count: {global_count}')
    logging.debug(f'SNP mean: {global_mean}')

    return global_mean, global_count


def local_ssq(A, global_mean):
    """
    Calculate local sum of squares
    original genotype_scaling_var_step
    """
    _, ssq = vectorize.vmap_jit(stats.sum_of_square, (1,0), (1,0))(
        A,
        jnp.expand_dims(global_mean, -1)
    )

    return ssq


def aggregate_ssq(local_ssqs, global_count):
    """
    original genotype_scaling_var_step

    Args:
        local_ssqs
            a list containing np.arrays with format [snp_sqr_sums, snp_sqr_sums,...]
    Return:
        global_var
            the feature variances
        deleted
            the boolean vector for whose variance = 0
    """
    local_ssqs = jnp.array(local_ssqs)
    global_var = jnp.sum(local_ssqs, axis=0) / (global_count - 1)

    deleted = jnp.where(global_var == 0)[0]
    global_var = jnp.delete(global_var, deleted)

    logging.debug(f'SNP variance: {global_var}')

    return global_var, deleted


def standardize(A, global_var, deleted):
    """
    original genotype_standardization_step

    :param snp_vars: global variance from aggregator with probably reduced dimensions because of var==0 SNPs
    :param delete: full dimensions boolean vector
    """
    A = jnp.delete(A, deleted, axis=1)
    A = A / jnp.sqrt(global_var)
    logging.debug(f'Standardizes genotype matrix {A.shape}\n{A}')

    return A
