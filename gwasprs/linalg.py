import numpy as np
from jax import jit, vmap
from jax import numpy as jnp

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
