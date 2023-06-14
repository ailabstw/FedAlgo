import numpy as np
from jax import numpy as jnp

def get_mask(X: 'np.ndarray[(1, 1), np.floating]'):
    return jnp.expand_dims(jnp.sum(jnp.isnan(X), axis = 1) == 0, -1)

def nonnan_count(x):
    """
    Number of non-NaN samples
    """
    return jnp.sum(jnp.logical_not(jnp.isnan(x)), axis = 0)
