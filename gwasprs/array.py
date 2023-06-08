from jax import numpy as jnp

def concat(xs):
    return jnp.concatenate(xs, axis=1)

def impute_with(X, val=0.0):
    return jnp.nan_to_num(X, copy=True, nan=val, posinf=None, neginf=None)
