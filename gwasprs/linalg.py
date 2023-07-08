from abc import ABCMeta

import numpy as np
import scipy.sparse.linalg as slinalg
from jax import jit, vmap
from jax import numpy as jnp
from jax import scipy as jsp
from jax import random as jrand
import logging

from . import vectorize


def nansum(A):
    snp_sum = jnp.nansum(A, axis=0)
    non_na_count = jnp.count_nonzero(~jnp.isnan(A))
    return snp_sum, non_na_count

@jit
def mvdot(X: 'np.ndarray[(1, 1), np.floating]', y: 'np.ndarray[(1,), np.floating]') -> 'np.ndarray[(1,), np.floating]':
    """Matrix-vector dot product

    Perform X.T * y.

    Args:
        X (np.ndarray[(1, 1), np.floating]): Matrix.
        y (np.ndarray[(1,), np.floating]): Vector.

    Returns:
        np.ndarray[(1,), np.floating]: Vector.
    """
    return vmap(jnp.vdot, (1, None), 0)(X, y)

@jit
def mvmul(X: 'np.ndarray[(1, 1), np.floating]', y: 'np.ndarray[(1,), np.floating]') -> 'np.ndarray[(1,), np.floating]':
    """Matrix-vector multiplication

    Perform X * y.

    Args:
        X (np.ndarray[(1, 1), np.floating]): Matrix.
        y (np.ndarray[(1,), np.floating]): Vector.

    Returns:
        np.ndarray[(1,), np.floating]: Vector.
    """
    return vmap(jnp.vdot, (0, None), 0)(X, y)

@jit
def mmdot(X: 'np.ndarray[(1, 1), np.floating]', Y: 'np.ndarray[(1, 1), np.floating]') -> 'np.ndarray[(1, 1), np.floating]':
    """Matrix-matrix dot product

    Perform X.T * Y.

    Args:
        X (np.ndarray[(1, 1), np.floating]): Matrix.
        Y (np.ndarray[(1, 1), np.floating]): Matrix.

    Returns:
        np.ndarray[(1, 1), np.floating]: Matrix.
    """
    return vmap(mvmul, (None, 1), 1)(X.T, Y)

@jit
def matmul(X: 'np.ndarray[(1, 1), np.floating]', Y: 'np.ndarray[(1, 1), np.floating]') -> 'np.ndarray[(1, 1), np.floating]':
    """Matrix multiplication

    Perform X * Y.

    Args:
        X (np.ndarray[(1, 1), np.floating]): Matrix.
        Y (np.ndarray[(1, 1), np.floating]): Matrix.

    Returns:
        np.ndarray[(1, 1), np.floating]: Matrix.
    """
    return vmap(mvmul, (None, 1), 1)(X, Y)

def gen_mvmul(y: np.ndarray):
    @jit
    def _mvmul(X: np.ndarray) -> np.ndarray:
        return vmap(jnp.vdot, (0, None), 0)(X, y)

    return _mvmul


def batched_vdot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Batched vector-vector dot product

    Perform x.T * y with batch on their last dimension.

    Args:
        x (np.ndarray[(1, 1), np.floating]): Batched vector.
        y (np.ndarray[(1, 1), np.floating]): Batched vector.

    Returns:
        np.ndarray[(1, 1), np.floating]: Batched vector.
    """
    return jnp.sum(x * y, axis=0)


@jit
def batched_mvdot(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Batched matrix-vector dot product

    Perform X.T * y with batch on their last dimension.

    Args:
        X (np.ndarray[(1, 1, 1), np.floating]): Batched matrix.
        y (np.ndarray[(1, 1), np.floating]): Batched vector.

    Returns:
        np.ndarray[(1, 1), np.floating]: Batched vector.
    """
    return vmap(mvdot, (2, 1), 1)(X, y)

@jit
def batched_mvmul(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Batched matrix-vector multiplication

    Perform X * y with batch on their last dimension.

    Args:
        X (np.ndarray[(1, 1, 1), np.floating]): Batched matrix.
        y (np.ndarray[(1, 1), np.floating]): Batched vector.

    Returns:
        np.ndarray[(1, 1), np.floating]: Batched vector.
    """
    return vmap(mvmul, (2, 1), 1)(X, y)

@jit
def batched_mmdot(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Batched matrix-matrix dot product

    Perform X.T * Y with batch on their last dimension.

    Args:
        X (np.ndarray[(1, 1, 1), np.floating]): Batched matrix.
        Y (np.ndarray[(1, 1, 1), np.floating]): Batched matrix.

    Returns:
        np.ndarray[(1, 1, 1), np.floating]: Batched matrix.
    """
    return vmap(mmdot, 2, 2)(X, Y)

@jit
def batched_matmul(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Batched matrix multiplication

    Perform X * Y with batch on their last dimension.

    Args:
        X (np.ndarray[(1, 1, 1), np.floating]): Batched matrix.
        Y (np.ndarray[(1, 1, 1), np.floating]): Batched matrix.

    Returns:
        np.ndarray[(1, 1, 1), np.floating]: Batched matrix.
    """
    return vmap(matmul, 2, 2)(X, Y)


@jit
def batched_diagonal(X: np.ndarray) -> np.ndarray:
    return vmap(jnp.diagonal, 2, 1)(X)


@jit
def batched_inv(X: np.ndarray) -> np.ndarray:
    return vmap(jnp.linalg.inv, 2, 2)(X)


def batched_cholesky(X: np.ndarray) -> np.ndarray:
    batch_size = X.shape[2]
    L = np.empty(X.shape)
    for b in range(batch_size):
        L.view()[:, :, b] = np.linalg.cholesky(X[:, :, b])
    return L


@jit
def batched_solve(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return vmap(jsp.linalg.solve, (2, 1), 1)(X, y)


@jit
def batched_solve_lower_triangular(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return vmap(lambda X, y: jsp.linalg.solve_triangular(X, y, lower=True), (2, 1), 1)(X, y)


@jit
def batched_solve_trans_lower_triangular(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return vmap(lambda X, y: jsp.linalg.solve_triangular(X, y, trans="T", lower=True), (2, 1), 1)(X, y)


class LinearSolver(object, metaclass=ABCMeta):
    def __init__(self) -> None:
        pass


class InverseSolver(LinearSolver):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, X: 'np.ndarray[(1, 1), np.floating]', y: 'np.ndarray[(1,), np.floating]'):
        # solve beta for X @ beta = y
        return jnp.linalg.solve(X, y)


class BatchedInverseSolver(LinearSolver):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, X: 'np.ndarray[(1, 1, 1), np.floating]', y: 'np.ndarray[(1, 1), np.floating]'):
        # solve beta for X @ beta = y
        return batched_solve(X, y)


class CholeskySolver(LinearSolver):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, X: 'np.ndarray[(1, 1), np.floating]', y: 'np.ndarray[(1,), np.floating]'):
        # L = Cholesky(X)
        L = jnp.linalg.cholesky(X)
        # solve Lz = y
        z = jsp.linalg.solve_triangular(L, y, lower=True)
        # solve Lt beta = z
        return jsp.linalg.solve_triangular(L, z, trans="T", lower=True)


class BatchedCholeskySolver(LinearSolver):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, X: 'np.ndarray[(1, 1, 1), np.floating]', y: 'np.ndarray[(1, 1), np.floating]'):
        # L = Cholesky(X)
        L = batched_cholesky(X)
        # solve Lz = y
        z = batched_solve_lower_triangular(L, y)
        # solve Lt beta = z
        return batched_solve_trans_lower_triangular(L, z)


class QRSolver(LinearSolver):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, X: 'np.ndarray[(1, 1), np.floating]', y: 'np.ndarray[(1,), np.floating]'):
        # Q, R = QR(X)
        Q, R = jnp.linalg.qr(X)
        # solve R beta = Qty
        return jsp.linalg.solve(R, mvdot(Q, y), lower=False)


@jit
def orthogonal_project(v, ortho, res):
    ortho = jnp.asarray(ortho)
    res = jnp.expand_dims(jnp.array(res), -1)
    projection = jnp.sum(res * ortho, axis=0)
    return v - projection

@jit
def svd(X):
    return jsp.linalg.svd(X, full_matrices=False)

@jit
def svd_cov_matrix(cov_matrices):
    cov_matrix = jnp.sum(jnp.asarray(cov_matrices),axis=0)
    U, S, Vt = jsp.linalg.svd(cov_matrix, full_matrices=False)
    return U

def randn(n, m, seed=42):
    return jrand.normal(key=jrand.PRNGKey(seed), shape=(n, m))


def check_eigenvector_convergence(current, previous, tolerance, required=None):
    """
    This function checks whether two sets of vectors are assymptotically collinear,
    up to a tolerance of epsilon.

    Args:
        current: The current eigenvector estimate
        previous: The eigenvector estimate from the previous iteration
        tolerance: The error tolerance for eigenvectors to be equal
        required: optional parameter for the number of eigenvectors required to have converged
    Returns: True if the required numbers of eigenvectors have converged to the given precision, False otherwise
                deltas, the current difference between the dot products
    """

    nr_converged = 0
    col = 0
    converged = False
    deltas = []
    if required is None:
        required = current.shape[1]
    while col < current.shape[1] and not converged:
        # check if the scalar product of the current and the previous eigenvectors
        # is 1, which means the vectors are 'parallel'
        delta = jnp.abs(jnp.sum(jnp.dot(jnp.transpose(current[:, col]), previous[:, col])))
        deltas.append(delta)
        if delta >= 1 - tolerance:
            nr_converged = nr_converged + 1
        if nr_converged >= required:
            converged = True
        col = col + 1
    return converged, deltas


def federated_svd(A, threshold, epsilon = 1e-9, max_iter = 15):
    current_iter = 1
    H_converged = False
    Hs = []
    P, A = local_svd_approximation(A, threshold)
    V = aggregate_svd_approximation([P], k1)
    G = init_G(A, V)
    prev_H = init_H(m, n)
    prev_H = update_local_H(A, G)

    while not H_converged and current_iter < max_iter:
        H = update_global_H(Hs)
        H_converged, delta_H = check_eigenvector_convergence(H, prev_H, epsilon)
        logging.debug(f'H converged status: {H_converged}')
        Hs.append(H)

        G = update_local_G(A, H)

        current_iter += 1

    H = decompose_H_stack(Hs)
    cov_matrix = local_cov_matrix(A, H)
    U = decompose_cov_matrices([cov_matrix], k2)
    norm = local_G_and_init_orthonormalization(P, U)


def local_svd_approximation(A, threshold):
    """
    Args:
        A (_type_): _description_
        threshold (_type_): threshold for preservation.
    """
    U, S, Vt = svd(A)
    V = Vt.T
    logging.debug(f'G matrix {V.shape}\n{V}')

    # determine information preserving level
    var = jnp.square(S)
    total_var = jnp.sum(var)
    var_sum, nd = 0, 0
    for i in range(len(var)):
        var_sum += var[i]
        pres_var = var_sum / total_var
        if pres_var > threshold:
            nd = i+1
            logging.debug(f'Preserved information ratio {pres_var}')
            break

    # Create reshaped diagnoal sigma matrix
    R = jnp.diag(S)[:nd,:nd]

    Vr = V[:, :nd]
    P = vectorize.fast_dot(jnp.sqrt(R), Vr.T)  # sqrt?

    # Transpose A for the next step > n_SNPs * n_samples
    logging.debug(f'Transpose A matrix >> n_SNPs * n_samples')
    A = A.T

    return P, A


def aggregate_svd_approximation(p_matrices, k1):
    """
    Algo4/3-5
    Perform SVD on matrices containing partial information of A

    original svd_approximation_step

    Args:
        p_matrices
            a list contains H matrices (A*G) with shape (k1, n_SNPs)
    Return:
        v_matrix
            an approximative matrix (H hat) lets clients get approximative local G
    """

    # P shape: (k1*n_clients, n_SNPs)
    P_matrix = jnp.concatenate(p_matrices,axis=0)
    logging.debug(f'After concat P matrix: {P_matrix.shape}\n{P_matrix}')

    # V shape: (n_SNPs, k1)
    if k1 < 120:
        logging.debug(f'Performing sparse SVD...')
        U, S, v_matrix = slinalg.svds(np.array(P_matrix), k=k1)
        v_matrix = jnp.array(v_matrix.T)
    else:
        logging.debug(f'Performing dense SVD...')
        U, S, v_matrix = svd(P_matrix)
        v_matrix = v_matrix.T
        nd = min(k1, U.shape[0], v_matrix.shape[0])
        v_matrix = v_matrix[:, :nd]

    logging.debug(f'V matrix: {v_matrix.shape}\n{v_matrix}')

    return v_matrix


def init_G(A, H):
    """
    original randomized_svd_init_step in client
    """
    # Get init G from approximate H (V_MATRIX)
    G = vectorize.fast_dot(A.T, H)
    return G


def init_H(m, n):
    """
    original randomized_svd_init_step in aggregator
    """
    prev_H = randn(m, n)
    return prev_H


def update_local_H(A, G):
    """
    original update_H_step in client
    """
    H = vectorize.fast_dot(A, G)
    return H


def update_global_H(Hs):
    """
    Algo2/10-11
    Update global H matrix

    original update_H_step in aggregator

    Args:
        h_matrices
            updated H matrices from clients
    Return:
        H_MATRIX
            updated and orthonormalized global H matrix
    """
    # H shape: (n_SNPs, k)
    H = jnp.sum(jnp.asarray(Hs),axis=0)
    Q, R = jsp.linalg.qr(H, mode='economic')
    logging.debug(f'After H orthonormalization: {H.shape}')

    return Q


def update_local_G(A, H):
    """
    original update_G_step in client
    """
    G = vectorize.fast_dot(A.T, H)
    return G


def decompose_H_stack(Hs):
    """
    original decompose_H_stack_step in aggregator
    """
    Hs = jnp.asarray(jnp.concatenate(Hs, axis=1))
    H, S, G = svd(Hs)
    logging.debug(f'After SVD, H matrix: {H.shape}')

    return H[:, :Hs.shape[1]-1]


def local_cov_matrix(A, H):
    """
    original calculate_cov_matrices_step in client
    """
    # P corresponds to A hat
    P = vectorize.fast_dot(H.T, A)
    logging.debug(f'P matrix {P.shape}\n{P}')

    # cov_matrix corresponds to M hat
    cov_matrix = vectorize.fast_dot(P, P.T)
    logging.debug(f'Covariance matrix {cov_matrix.shape}\n{cov_matrix}')

    return cov_matrix


def decompose_cov_matrices(cov_matrices, k):
    """
    original calculate_cov_matrices_step in aggregator
    """
    U = svd_cov_matrix(cov_matrices)[:, :k]
    return U


def local_G_and_init_orthonormalization(P, U):
    """
    original compute_local_G_step
    """
    G = mmdot(P, U)

    # Orthonormalization initialize
    ortho = [G[:, 0]]

    return G, jnp.vdot(G[:,0],G[:,0]), ortho

