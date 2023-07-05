from jax import jit, vmap
from jax import numpy as jnp

def vmap_jit(func, in_axes, out_axes):
    return vmap(jit(func), in_axes=in_axes, out_axes=out_axes)

def fast_dot(X1, X2):
    func = vmap_jit(jnp.dot, 0, 0)
    X1 = jnp.expand_dims(X1, axis=0)
    X2 = jnp.expand_dims(X2, axis=0)

    return func(X1, X2)[0]
