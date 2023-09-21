import numpy as np
import jax.numpy as jnp
from jax import jit, vmap

size = 3000
X = np.random.normal(size=(size, size)).astype(np.float32)
y = np.random.normal(size=(size,)).astype(np.float32)

%timeit jnp.dot(X, X.T).block_until_ready()
# 414 ms ± 15.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

def unnorm_autocovariance(X: np.ndarray[(1, 1), np.floating]) -> np.ndarray[(1, 1), np.floating]:
    XtX = jnp.dot(X.T, X)
    return XtX

%timeit unnorm_autocovariance(X)
# 508 ms ± 57.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

@jit
def vmap_unnorm_autocovariance(X: np.ndarray[(1, 1), np.floating]) -> np.ndarray[(1, 1), np.floating]:
    return vmap(unnorm_autocovariance)(X)

%timeit vmap_unnorm_autocovariance(X)
# 10.9 ms ± 98.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

def factory():
    @jit
    def vmap_unnorm_autocovariance(X: np.ndarray[(1, 1), np.floating]) -> np.ndarray[(1, 1), np.floating]:
        return vmap(unnorm_autocovariance)(X)
    
    return vmap_unnorm_autocovariance

f = factory()

%timeit f(X)
# 11.4 ms ± 950 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

def dot(X: np.ndarray[(1, 1), np.floating], y: np.ndarray[(1,), np.floating]) -> np.ndarray[(1, 1), np.floating]:
    Xty = jnp.dot(X.T, y)
    return Xty

%timeit dot(X, y)
# 30.1 ms ± 9.95 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

@jit
def vmap_dot(X: np.ndarray[(1, 1), np.floating], y: np.ndarray[(1,), np.floating]) -> np.ndarray[(1, 1), np.floating]:
    return vmap(dot)(X, y)

%timeit vmap_dot(X, y)
# 13.3 ms ± 2.77 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)

def gen_prod_vec(y: np.ndarray[(1,), np.floating]):
    @jit
    def vmap_dot(X: np.ndarray[(1, 1), np.floating]) -> np.ndarray[(1, 1), np.floating]:
        return vmap(dot)(X, y)
    
    return vmap_dot

prod_vec = gen_prod_vec(y)

%timeit prod_vec(X)
# 11 ms ± 281 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
