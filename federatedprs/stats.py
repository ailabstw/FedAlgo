import numpy as np
from jax import jit, vmap
from jax import numpy as jnp

@jit
def mvdot(X: np.ndarray[(1, 1), np.floating], y: np.ndarray[(1,), np.floating]) -> np.ndarray[(1,), np.floating]:
    return vmap(jnp.vdot, (0, None), 0)(X, y)

@jit
def matmul(X: np.ndarray[(1, 1), np.floating], Y: np.ndarray[(1, 1), np.floating]) -> np.ndarray[(1, 1), np.floating]:
    return vmap(mvdot, (None, 1), 1)(X, Y)

def gen_mvdot(y: np.ndarray):
    @jit
    def _mvdot(X: np.ndarray) -> np.ndarray:
        return vmap(jnp.vdot, (0, None), 0)(X, y)
    
    return _mvdot

def unnorm_autocovariance(X: np.ndarray[(1, 1), np.floating]) -> np.ndarray[(1, 1), np.floating]:
    return matmul(X.T, X)

def unnorm_covariance(X: np.ndarray[(1, 1), np.floating], y: np.ndarray[(1,), np.floating]):
    return mvdot(X.T, y)
