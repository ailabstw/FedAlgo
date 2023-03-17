import numpy as np
from jax import jit, vmap
from jax import numpy as jnp

def unnorm_autocovariance(X: np.ndarray[(1, 1), np.floating]) -> np.ndarray[(1, 1), np.floating]:
    XtX = jnp.dot(X.T, X)
    return XtX

@jit
def vmap_unnorm_autocovariance(X: np.ndarray[(1, 1), np.floating]) -> np.ndarray[(1, 1), np.floating]:
    return vmap(unnorm_autocovariance)(X)

def unnorm_covariance(X: np.ndarray[(1, 1), np.floating], y: np.ndarray[(1,), np.floating]):
    Xty = jnp.dot(X.T, y)
    return Xty

@jit
def vmap_unnorm_covariance(X: np.ndarray[(1, 1), np.floating], y: np.ndarray[(1,), np.floating]):
    return vmap(unnorm_covariance)(X)
