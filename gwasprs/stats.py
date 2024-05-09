import numpy as np
import scipy.stats as stats
from jax import jit, vmap, pmap
from jax import numpy as jnp
from jax import scipy as jsp
from jax.typing import ArrayLike
from . import linalg, utils


def make_mean_zero(A, mean):
    # Make the SNP mean = 0
    return A - mean


def sum_of_square(A):
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
    return jnp.sum(jnp.square(A), axis=0)


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
    norms = 1 / jnp.sqrt(jnp.expand_dims(jnp.asarray(norms), -1))
    return (norms * ortho).T


def unnorm_autocovariance(
    X: "np.ndarray[(1, 1), np.floating]",
) -> "np.ndarray[(1, 1), np.floating]":
    return linalg.mmdot(X, X)


def unnorm_covariance(
    X: "np.ndarray[(1, 1), np.floating]", y: "np.ndarray[(1,), np.floating]"
):
    return linalg.mvdot(X, y)


def batched_unnorm_autocovariance(
    X: "np.ndarray[(1, 1, 1), np.floating]", acceleration="single"
) -> "np.ndarray[(1, 1, 1), np.floating]":
    if acceleration == "single":
        return linalg.batched_mmdot(X, X)
    elif acceleration == "pmap":
        pmap_func = pmap(linalg.batched_mmdot, in_axes=0, out_axes=0)
        ncores = utils.jax_cpu_cores()
        batch, nsample, ndims = X.shape
        minibatch, remainder = divmod(batch, ncores)
        A = np.reshape(
            X[: (minibatch * ncores), :, :], (ncores, minibatch, nsample, ndims)
        )
        Y = np.reshape(pmap_func(A, A), (-1, ndims, ndims))

        if remainder != 0:
            B = X[(minibatch * ncores) :, :, :]
            Z = linalg.batched_mmdot(B, B)
            Y = np.concatenate((Y, Z), axis=0)
        return Y
    else:
        raise ValueError(f"{acceleration} acceleration is not supported.")


def batched_unnorm_covariance(
    X: "np.ndarray[(1, 1, 1), np.floating]", y: "np.ndarray[(1, 1), np.floating]"
):
    return linalg.batched_mvdot(X, y)


def blocked_unnorm_autocovariance(
    X: "np.ndarray[(1, 1), np.floating]",
) -> "np.ndarray[(1, 1), np.floating]":
    return linalg.mmdot(X, X)


def blocked_unnorm_covariance(
    X: "np.ndarray[(1, 1), np.floating]", y: "np.ndarray[(1,), np.floating]"
):
    return linalg.mvdot(X, y)


def t_dist_pvalue(t_stat, df):
    return 2 * (1 - stats.t.cdf(np.abs(t_stat), df))


# Logistic


@jit
def logistic_stats(beta, inv_hessian):
    std = jnp.sqrt(inv_hessian.diagonal())
    t_stat = beta / std
    p_value = 1 - jsp.stats.chi2.cdf(jnp.square(t_stat), 1)
    return t_stat, p_value


@jit
def batched_logistic_stats(beta, inv_hessian):
    return vmap(logistic_stats, (0, 0), (0, 0))(beta, inv_hessian)


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


# def impute_with_mean(A, mean):
#    na_indices = jnp.where(jnp.isnan(A))
#    A = A.at[na_indices].set(jnp.take(mean, na_indices[1]))
#    return A


def impute_with_mean(
    A: ArrayLike, means: ArrayLike, batch_size: int = 10000
) -> ArrayLike:
    """Fill na with mean
    The origin implementation may encounter the problem of too large indice for xla scatter
    Here use batch to solve the trouble.
    Noted that at function is only allow for jnp.array.
    """
    feat_num = len(means)
    _start = 0
    while _start < feat_num:
        _end = min(_start + batch_size, feat_num)
        na_indices = jnp.where(jnp.isnan(A[:, _start:_end]))
        na_indices = (na_indices[0], na_indices[1] + _start)
        A = A.at[na_indices].set(jnp.take(means, na_indices[1]))
        _start = _end

    return A


def sum_and_count(A):
    col_sum = _vectorize(jnp.sum, 1, 0)(A)
    count = A.shape[0]
    return col_sum, count


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
