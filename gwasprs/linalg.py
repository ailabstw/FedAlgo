from abc import ABCMeta

import numpy as np
from jax import jit, vmap
from jax import numpy as jnp
from jax import scipy as jsp

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
def batched_solve_triangular(X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
    return vmap(lambda X, y: jsp.linalg.solve_triangular(X, y, **kwargs), 2, 2)(X, y)


class LinearSolver(object, metaclass=ABCMeta):
    def __init__(self) -> None:
        pass


class InverseSolver(LinearSolver):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, X: 'np.ndarray[(1, 1), np.floating]', y: 'np.ndarray[(1,), np.floating]'):
        inv_X = jnp.linalg.inv(X)
        # beta = X^-1 y
        return mvmul(inv_X, y)


class BatchedInverseSolver(LinearSolver):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, X: 'np.ndarray[(1, 1, 1), np.floating]', y: 'np.ndarray[(1, 1), np.floating]'):
        inv_X = batched_inv(X)
        # beta = X^-1 y
        return batched_mvmul(inv_X, y)


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
        z = batched_solve_triangular(L, y, lower=True)
        # solve Lt beta = z
        return batched_solve_triangular(L, z, trans="T", lower=True)


class QRSolver(LinearSolver):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, X: 'np.ndarray[(1, 1), np.floating]', y: 'np.ndarray[(1,), np.floating]'):
        # Q, R = QR(X)
        Q, R = jnp.linalg.qr(X)
        # solve R beta = Qty
        return jsp.linalg.solve(R, mvdot(Q, y), lower=False)
