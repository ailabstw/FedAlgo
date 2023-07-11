from abc import ABCMeta

import numpy as np
import scipy.sparse.linalg as slinalg
from jax import jit, vmap
from jax import numpy as jnp
from jax import scipy as jsp
from jax import random as jrand
import logging


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
    """Orthogonalize

    v - (summation of v's projections on i-1 orthogonalized eigenvectors)

    Args:
        v (np.ndarray[(1,), np.floating]) : ith eigenvector to be orthogonalized
        ortho (list of np.ndarray[(1,), np.floating]): i-1 orthogonalized eigenvectors with shape (n,)
        res (np.ndarray[(1,), np.floating]): residuals with shape (i-1,)
    
    Returns:
        (np.ndarray[(1,), np.floating]) : ith orthogonalized eigenvector
    """
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

def iteration_update(current_H, Hs, current_iteration, max_iterations):
    if current_iteration < max_iterations:
        Hs.append(current_H)
    return current_H, Hs, current_iteration+1


def init_H(m, k1):
    """Initial H matrix generation

    Generate random H matrix with shape (m, k1), 
    where m and k1 represent m SNPs and k1 latent dimensions respectively.
    original randomized_svd_init_step in aggregator

    Args:
        m (int) : number of SNPs
        k1 (int) : latent dimensions of H matrix in SVD and must be <= n samples
    
    Returns:
        (np.ndarray[(1,1), np.floating]) : random H matrix
    """
    prev_H = randn(m, k1)
    return prev_H


def update_local_H(A, G):
    """Update H matrix in edge

    H = AG, where H (m, k1), A (m, n) and G (n, k1)
    original update_H_step in client

    Args:
        A (np.ndarray[(1,1), np.floating]) : genotype matrix with shape (m, n), where m and n represent m SNPs and n samples respectively.
        G (np.ndarray[(1,1), np.floating]) : randomly generated and orthonormalized G matrix in the 1st step or updated G matrix during iterations.
    
    Returns:
        (np.ndarray[(1,1), np.floating]) : updated H matrix (m, k1)
    """
    H = mmdot(A.T, G)
    return H


def update_global_H(Hs):
    """Update H matrix in aggregator

    Algo2/10-11
    Update global H matrix by summation and orthonormalization of H matrix collected from edges.
    original update_H_step in aggregator

    Args:
        Hs (list of np.ndarray[(1,1), np.floating]) : H matrices collected from edges.

    Return:
        (np.ndarray[(1,1), np.floating]) : updated and orthonormalized H matrix (m, k1)
    """
    H = jnp.sum(jnp.asarray(Hs),axis=0)
    H, R = jsp.linalg.qr(H, mode='economic')

    return H


def update_local_G(A, H):
    """Update G matrix in edge

    G = AtH, where G (n, k1), At (n, m) and H (m, k1)
    original update_G_step in client

    Args:
        A (np.ndarray[(1,1), np.floating]) : genotype matrix with shape (m, n), where m and n represent m SNPs and n samples respectively.
        H (np.ndarray[(1,1), np.floating]) : global H matrix from aggregator with shape (m, k1)
    
    Returns:
        (np.ndarray[(1,1), np.floating]) : update G matrix with shape (n, k1)
    """
    G = mmdot(A, H)
    return G


def decompose_H_stack(Hs):
    """Stack H matrices from I iterations and decompose it

    Each H matrix is the updated H matrix during iterations.
    The shape of stacked H matrix (Hs) is (m, k1*I), where m is the number of SNPs, k1 is the latent dimensions 
    and I iterations depending on the convergence status and max iterations.
    original decompose_H_stack_step in aggregator

    Args:
        Hs (list of np.ndarray[(1,1), np.floating]) : H matrices from iterations
    
    Returns:
        (np.ndarray[(1,1), np.floating]) : decomposed H matrix from stacked H matrices with shape (m, k1*I), where m is the number of SNPs, k1 is the latent dimensions 
                                           and I iterations depending on the convergence status and max iterations.
    """
    Hs = jnp.asarray(jnp.concatenate(Hs, axis=1))
    H, S, G = svd(Hs)
    return H


def local_cov_matrix(A, H):
    """Calculate proxy matrix P and covariance matrix

    Algo5/4-5
    P is the proxy data matrix with shape (k1*I, n), where n is the number of samples, k1 is the latent dimensions and I iterations depending on the convergence status and max iterations.
    cov_matrix is covariance matrix with shape (k1*I, k1*I).

    Args:
        A (np.ndarray[(1,1), np.floating]) : genotype matrix with shape (m, n), where m and n represent m SNPs and n samples respectively.
        H (np.ndarray[(1,1), np.floating]) : H matrix decomposed from stacked H matrices with shape (m, k1*I), where m is the number of SNPs, k1 is the latent dimensions 
                                             and I iterations depending on the convergence status and max iterations.
    
    Returns:
        (np.ndarray[(1,1), np.floating]) : the proxy data matrix with shape (k1*I, n), where n is the number of samples, k1 is the latent dimensions and I iterations depending on the convergence status and max iterations.
        (np.ndarray[(1,1), np.floating]) : the inner prodcut of proxy data matrix with shape (k1*I, k1*I) as the covariance matrix.
    """
    P = mmdot(H, A) # P corresponds to A hat
    cov_matrix = mmdot(P.T, P.T) # cov_matrix corresponds to M hat
    return P, cov_matrix


def decompose_cov_matrices(cov_matrices, k2):
    """Decompose covariance matrix in aggregator

    Decompose covariance matrix collected from proxy data matrices for each edge.
    original calculate_cov_matrices_step in aggregator

    Args:
        cov_matrices (list of np.ndarray[(1,1), np.floating]) : the covariance matrices collected from edges with shape (k1*I, k1*I) for each matrix,
                                                                where k1 is the latent dimensions and I iterations depending on the convergence status and max iterations.
        k2 (int) : output latent dimensions of U matrix in SVD and must be <= k1*I.
    
    Returns:
        (np.ndarray[(1,1), np.floating]) : eigenvectors used for getting the G matrix in edges with shape (k1*I, k2)
    """
    U = svd_cov_matrix(cov_matrices)[:, :k2]
    return U


def local_G_and_init_orthonormalization(P, U):
    """Update G matrix and initialize orthonomalization

    Use proxy data matrix P (k1*I, n) and eigenvectors U (k1*I, k2) from aggregator to get the G matrix with shape (n, k2)
    original compute_local_G_step

    Args:
        P (np.ndarray[(1,1), np.floating]) : proxy data matrix from edge with shape (k1*I, n)
        U (np.ndarray[(1,1), np.floating]) : eigenvectors used for getting the G matrix in edge with shape (k1*I, k2)
    
    Returns:
        (np.ndarray[(1,1), np.floating]) : the G matrix with shape (n, k2)
        (int) : the norm of the first partial eigenvector (the rest is distributed on different edges)
        (the list of np.ndarray[(1,), np.floating]) : the list used to store partial eigenvectors in the downstream orthonormalization process
    """

    G = mmdot(P, U)

    # Orthonormalization initialize
    ortho = [G[:, 0]]

    return G, jnp.vdot(G[:,0],G[:,0]), ortho