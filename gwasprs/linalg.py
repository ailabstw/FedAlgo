import numpy as np
from jax import jit, vmap
from jax import numpy as jnp

@jit
def mvdot(X: 'np.ndarray[(1, 1), np.floating]', y: 'np.ndarray[(1,), np.floating]') -> 'np.ndarray[(1,), np.floating]':
    return vmap(jnp.vdot, (0, None), 0)(X, y)

@jit
def mmdot(X: 'np.ndarray[(1, 1), np.floating]', Y: 'np.ndarray[(1, 1), np.floating]') -> 'np.ndarray[(1, 1), np.floating]':
    return vmap(mvdot, (None, 1), 1)(X.T, Y)

@jit
def matmul(X: 'np.ndarray[(1, 1), np.floating]', Y: 'np.ndarray[(1, 1), np.floating]') -> 'np.ndarray[(1, 1), np.floating]':
    return vmap(mvdot, (None, 1), 1)(X, Y)

def gen_mvdot(y: np.ndarray):
    @jit
    def _mvdot(X: np.ndarray) -> np.ndarray:
        return vmap(jnp.vdot, (0, None), 0)(X, y)
    
    return _mvdot

@jit
def batched_mvdot(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return vmap(mvdot, (2, 1), 1)(X, y)

@jit
def batched_mmdot(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return vmap(mmdot, 2, 2)(X, Y)

@jit
def batched_matmul(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return vmap(matmul, 2, 2)(X, Y)
