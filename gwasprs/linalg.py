from abc import ABCMeta

import numpy as np
from jax import jit, vmap
from jax import numpy as jnp
from jax import scipy as jsp
from jax import random as jrand

from . import array, stats, project

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
    return jnp.sum(x * y, axis=1)


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
    return vmap(mvdot, (0, 0), 0)(X, y)

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
    return vmap(mvmul, (0, 0), 0)(X, y)

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
    return vmap(mmdot, 0, 0)(X, Y)

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
    return vmap(matmul, 0, 0)(X, Y)


@jit
def batched_diagonal(X: np.ndarray) -> np.ndarray:
    return vmap(jnp.diagonal, 0, 0)(X)


@jit
def batched_inv(X: np.ndarray) -> np.ndarray:
    return vmap(jnp.linalg.inv, 0, 0)(X)


def batched_cholesky(X: np.ndarray) -> np.ndarray:
    batch_size = X.shape[0]
    L = np.empty(X.shape)
    for b in range(batch_size):
        L.view()[b, :, :] = np.linalg.cholesky(X[b, :, :])
    return L


@jit
def batched_solve(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return vmap(jsp.linalg.solve, (0, 0), 0)(X, y)


@jit
def batched_solve_lower_triangular(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return vmap(lambda X, y: jsp.linalg.solve_triangular(X, y, lower=True), (0, 0), 0)(X, y)


@jit
def batched_solve_trans_lower_triangular(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return vmap(lambda X, y: jsp.linalg.solve_triangular(X, y, trans="T", lower=True), (0, 0), 0)(X, y)


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


def local_G_from_proxy_matrix(P, U):
    """Update G matrix and initialize orthonomalization

    Use proxy data matrix P (k1*I, n) and eigenvectors U (k1*I, k2) from aggregator to get the G matrix with shape (n, k2)
    original compute_local_G_step

    Args:
        P (np.ndarray[(1,1), np.floating]) : proxy data matrix from edge with shape (k1*I, n)
        U (np.ndarray[(1,1), np.floating]) : eigenvectors used for getting the G matrix in edge with shape (k1*I, k2)

    Returns:
        (np.ndarray[(1,1), np.floating]) : the G matrix with shape (n, k2)
    """
    G = mmdot(P, U)
    return G


def init_orthonormalization(M):
    """
    This function supports general usage without SVD process.

    Args:
        M (np.ndarray[(1,1), np.floating]) : The eigenvectors should be placed vertically as M[:,i].

    Returns:
        (np.floating) : the norm of the first partial eigenvector (the rest is distributed on different edges)
        (the list of np.ndarray[(1,), np.floating]) : the list used to store partial eigenvectors in the downstream orthonormalization process
    """
    ortho = [M[:, 0]]

    return jnp.vdot(M[:,0],M[:,0]), ortho


def eigenvec_concordance_estimation(GT, SIM, latent_axis=(1,1), decimal=5):

    if latent_axis[0] != 1:
        GT = GT.T
    if latent_axis[1] != 1:
        SIM = SIM.T

    if GT.shape != SIM.shape:
        raise ValueError(f'Inconcordance matrix shapes: {GT.shape} ground truth, {SIM.shape} simulation')

    def _report(A):
        I = np.identity(A.shape[0])
        if np.testing.assert_array_almost_equal(abs(A), I, decimal=decimal) is None:
            return ('PASSED', True)
        else:
            return (f'FAILED\ninner product:\n{A}', False)

    GT_orthonormal = _report(mmdot(GT,GT))
    SIM_orthonormal = _report(mmdot(SIM,SIM))
    concordance = _report(mmdot(GT,SIM))

    message = f"\
        =========== Concordance Estimation ===========\n\
        Ground Truth Orthonormal: {GT_orthonormal[0]}\n\
        Simulation Orthonormal: {SIM_orthonormal[0]}\n\
        Concordance between GT and SIM: {concordance[0]}\n\
        =============================================="

    print(message)

    return (GT_orthonormal[1], SIM_orthonormal[1], concordance[1], message)


def _get_shared_args(func, local_args):
    # Extract shared args across functions
    shared_args = func.__code__.co_varnames[0:5]
    return {arg:local_args.get(arg) for arg in shared_args if arg != 'formated'}

def _get_func_specific_args(func, kwargs):
    # Extract function-specific args
    func_args = func.__code__.co_varnames
    return {arg:kwargs.get(arg) for arg in func_args if arg in kwargs.keys()}

def federated_standardization(As, formated=False, edge_axis=None, sample_axis=None, snp_axis=None):
    if not formated:
        shared_args = _get_shared_args(federated_standardization, locals())
        As = array.genotype_matrix_input_formatter(**shared_args)

    # global mean without NAs
    local_sums, local_counts = [], []
    for edge_idx in range(len(As)):
        s, c = stats.nansum(As[edge_idx])
        local_sums.append(s)
        local_counts.append(c)
    GLOBAL_MEAN = stats.aggregate_sums(local_sums, local_counts)

    # global mean from imputed data
    local_sums, local_counts = [], []
    for edge_idx in range(len(As)):
        a, s, c = stats.impute_and_local_mean(As[edge_idx], GLOBAL_MEAN)
        local_sums.append(s)
        local_counts.append(c)
        As[edge_idx] = a
    GLOBAL_MEAN = stats.aggregate_sums(local_sums, local_counts)

    # global variance
    local_ssqs, local_counts = [], []
    for edge_idx in range(len(As)):
        a, ssq = stats.local_ssq(As[edge_idx], GLOBAL_MEAN)
        local_ssqs.append(ssq)
        local_counts.append(a.shape[0])
        As[edge_idx] = a
    GLOBAL_VAR, DELETE = stats.aggregate_ssq(local_ssqs, local_counts)

    # standardize
    std_As = []
    for edge_idx in range(len(As)):
        std_a = stats.standardize(As[edge_idx], GLOBAL_VAR, DELETE)
        std_As.append(std_a)

    return std_As


def federated_vertical_subspace_iteration(As, formated=False, edge_axis=None, sample_axis=None, snp_axis=None,
                                          transposed=False, k1=20, epsilon=1e-9, max_iterations=20):
    if not formated or not transposed:
        shared_args = _get_shared_args(federated_vertical_subspace_iteration, locals())
        As = array.genotype_matrix_input_formatter(transpose=True, **shared_args)

    # edge init
    local_Gs = []
    for edge_idx in range(len(As)):
        g = randn(As[edge_idx].shape[1], k1)
        g, r = jsp.linalg.qr(g, mode='economic')
        local_Gs.append(g)

    # aggregator init
    prev_H = init_H(As[0].shape[0], k1)
    current_iteration = 1
    H_converged = False
    Hs = []

    # Vertical subspace iterations
    while not H_converged and current_iteration < max_iterations:
        # Update global H
        hs = []
        for edge_idx in range(len(As)):
            h = update_local_H(As[edge_idx], local_Gs[edge_idx])
            hs.append(h)
        GLOBAL_H = update_global_H(hs)

        # Check convergence & save global H
        H_converged, _ = check_eigenvector_convergence(GLOBAL_H, prev_H, epsilon)
        prev_H, Hs, current_iteration = iteration_update(GLOBAL_H, Hs, current_iteration, max_iterations)

        # Update local G
        for edge_idx in range(len(As)):
            g = update_local_G(As[edge_idx], GLOBAL_H)
            local_Gs[edge_idx] = g

    return As, Hs, local_Gs


def federated_randomized_svd(As, formated=False, edge_axis=None, sample_axis=None, snp_axis=None,
                             transposed=False, Hs=None, local_Gs=None, k2=20, **kwargs):
    shared_args = _get_shared_args(federated_randomized_svd, locals())

    # If Hs or local_Gs is missing, generate it.
    # If Hs exists, check whether As is formated.
    if (Hs or local_Gs) is None:
        kwargs.update(shared_args)
        As, Hs, local_Gs = federated_vertical_subspace_iteration(**kwargs)
    elif not formated or not transposed:
        As = array.genotype_matrix_input_formatter(transpose=True, **shared_args)

    # Get the projection matrix
    GLOBAL_H = decompose_H_stack(Hs)

    # Form proxy data matrices to get proxy covariance matrices
    Ps, COVs = [], []
    for edge_idx in range(len(As)):
        p, cov = local_cov_matrix(As[edge_idx], GLOBAL_H)
        Ps.append(p)
        COVs.append(cov)
    GLOBAL_U = decompose_cov_matrices(COVs, k2)

    # Update G matrix and prepare for the orthonormalization
    for edge_idx in range(len(As)):
        g = local_G_from_proxy_matrix(Ps[edge_idx], GLOBAL_U)
        local_Gs[edge_idx] = g

    return GLOBAL_H, local_Gs


def final_H_update(As, local_Gs):
    hs = []
    for edge_idx in range(len(As)):
        h = update_local_H(As[edge_idx], local_Gs[edge_idx])
        hs.append(h)
    GLOBAL_H = update_global_H(hs)
    return GLOBAL_H


def federated_svd(As, formated=False, edge_axis=None, sample_axis=None, snp_axis=None, **kwargs):
    if not formated:
        shared_args = _get_shared_args(federated_svd, locals())
        As = array.genotype_matrix_input_formatter(**shared_args)
    shared_args = {'As':As, 'formated':True, 'edge_axis':0, 'sample_axis':1, 'snp_axis':2}

    # Standardization
    std_As = federated_standardization(**shared_args)
    std_As = array.genotype_matrix_input_formatter(std_As, 0, 1, 2, transpose=True)
    shared_args.update({'As':std_As, 'transposed':True})

    # Vertical subspace iterations
    vsi_kwargs = {**shared_args, **_get_func_specific_args(federated_vertical_subspace_iteration, kwargs)}
    As, Hs, local_Gs = federated_vertical_subspace_iteration(**vsi_kwargs)
    shared_args.update({'As':As})
    kwargs.update({'Hs':Hs, 'local_Gs':local_Gs})

    # Randomized SVD
    randomized_kwargs = {**shared_args, **_get_func_specific_args(federated_randomized_svd, kwargs)}
    GLOBAL_H, local_Gs = federated_randomized_svd(**randomized_kwargs)
    kwargs.update({'GLOBAL_H':GLOBAL_H, 'MTXs':local_Gs})

    # Gram-Schmidt Orthonormalization
    ortho_kwargs = {**_get_func_specific_args(project.federated_orthonormalization, kwargs)}
    local_Gs = project.federated_orthonormalization(**ortho_kwargs)

    # Final global H update
    GLOBAL_H = final_H_update(As, local_Gs)

    return local_Gs, GLOBAL_H


@jit
def logistic_predict(X, beta):
    """Logistic regression prediction

    Perform sigmoid(X*beta)

    Args:
        X (np.ndarray[(1, 1), np.floating]): Matrix.
        beta (np.ndarray[(1,), np.floating]): Vector.

    Returns:
        np.ndarray[(1,), np.floating]: Vector.
    """
    pred_y = 1 / (1 + jnp.exp(-mvmul(X,beta)))
    return pred_y

@jit
def logistic_residual(y, pred_y):
    """Residual calculation

    Perform y - predicted_y

    Args:
        y (np.ndarray[(1,), np.floating]): Vector.
        pred_y (np.ndarray[(1, 1), np.floating]): Vector.

    Returns:
        np.ndarray[(1, 1), np.floating]: Vector.
    """
    return jnp.expand_dims(y, -1) - pred_y

@jit
def logistic_gradient(X, residual):
    """Logistic gradient vector

    Perform X.T * (y - predicted_y)

    Args:
        X (np.ndarray[(1, 1), np.floating]): Matrix.
        residual (np.ndarray[(1, 1), np.floating]): Vector.

    Returns:
        np.ndarray[(1, 1), np.floating]: Vector.
    """
    return mvdot(X, residual)

@jit
def logistic_hessian(X, pred_y):
    """Logistic hessian matrix

    Perform jnp.multiply(X.T, (pred_y * (1 - pred_y)).T) * X

    Args:
        X (np.ndarray[(1, 1), np.floating]): Matrix.
        pred_y (np.ndarray[(1, 1), np.floating]): Vector.

    Returns:
        np.ndarray[(1, 1), np.floating]: Matrix.
    """
    return matmul(jnp.multiply(X.T, (pred_y * (1 - pred_y)).T), X)

@jit
def logistic_loglikelihood(X, y, pred_y):
    """Logistic log likelihood estimation

    Perform SUM(
        y * log(predicted_y + epsilon) +
        (1 - y) * log(1 - predicted_y + epsilon)
    )

    Args:
        X (np.ndarray[(1, 1), np.floating]): Matrix.
        y (np.ndarray[(1,), np.floating]): Vector.
        pred_y (np.ndarray[(1, 1), np.floating]): Vector.

    Returns:
        np.ndarray[(1,), np.floating]: float.
    """
    epsilon = jnp.finfo(float).eps
    return jnp.sum(y * jnp.log(pred_y + epsilon) + (1 - y) * jnp.log(1 - pred_y + epsilon))

@jit
def batched_logistic_predict(X, beta):
    """Batched logistic regression prediction

    Perform sigmoid(X*beta)

    Args:
        X (np.ndarray[(1, 1, 1), np.floating]): Batched matrix.
        beta (np.ndarray[(1, 1), np.floating]): Batched vector.

    Returns:
        np.ndarray[(1, 1), np.floating]: Batched vector.
    """
    return vmap(logistic_predict, (0,0), 0)(X, beta)

@jit
def batched_logistic_residual(y, pred_y):
    """Batched residual calculation

    Perform y - predicted_y

    Args:
        y (np.ndarray[(1, 1), np.floating]): Batched vector.
        pred_y (np.ndarray[(1, 1, 1), np.floating]): Batched vector.

    Returns:
        np.ndarray[(1, 1, 1), np.floating]: Batched vector.
    """
    return vmap(logistic_residual, (0,0), 0)(y, pred_y)

@jit
def batched_logistic_gradient(X, residual):
    """Batched logistic gradient vector

    Perform X.T * (y - predicted_y)

    Args:
        X (np.ndarray[(1, 1, 1), np.floating]): Batched matrix.
        residual (np.ndarray[(1, 1, 1), np.floating]): Batched vector.

    Returns:
        np.ndarray[(1, 1, 1), np.floating]: Batched vector.
    """
    return vmap(logistic_gradient, (0,0), 0)(X, residual)

@jit
def batched_logistic_hessian(X, pred_y):
    """Batched logistic hessian matrix

    Perform jnp.multiply(X.T, (pred_y * (1 - pred_y)).T) * X

    Args:
        X (np.ndarray[(1, 1, 1), np.floating]): Batched matrix.
        pred_y (np.ndarray[(1, 1, 1), np.floating]): Batched vector.

    Returns:
        np.ndarray[(1, 1, 1), np.floating]: Batched matrix.
    """
    return vmap(logistic_hessian, (0,0), 0)(X, pred_y)

@jit
def batched_logistic_loglikelihood(X, y, pred_y):
    """Batched logistic log likelihood estimation

    Perform SUM(
        y * log(predicted_y + epsilon) +
        (1 - y) * log(1 - predicted_y + epsilon)
    )

    Args:
        X (np.ndarray[(1, 1, 1), np.floating]): Batched matrix.
        y (np.ndarray[(1, 1), np.floating]): Batched vector.
        pred_y (np.ndarray[(1, 1, 1), np.floating]): Batched vector.

    Returns:
        np.ndarray[(1,), np.floating]: Batched vector.
    """
    return vmap(logistic_loglikelihood, (0,0,0), 0)(X, y, pred_y)





