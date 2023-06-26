import numpy as np
from jax import numpy as jnp


def isnonnan(X: np.ndarray, axis=1):
    return jnp.sum(jnp.isnan(X), axis=axis) == 0


def get_mask(X: 'np.ndarray[(1, 1), np.floating]'):
    return jnp.expand_dims(isnonnan(X, axis=1), -1)


def nonnan_count(x, axis=0):
    """
    Number of non-NaN samples
    """
    return jnp.sum(jnp.logical_not(jnp.isnan(x)), axis=axis)
