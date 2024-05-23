import numpy as np
from scipy.linalg import block_diag
import jax.numpy as jnp
import jax.scipy.linalg as jslinalg


N = 1000

A = np.random.rand(N, 1)
B = np.random.rand(N-1, 1)
C = np.random.rand(N-2, 1)

X = block_diag(A, B, C)
beta = np.random.rand(3)

X2 = jslinalg.block_diag(A, B, C)
beta2 = jnp.array(beta)

%timeit y = gwasprs.stats.blocked_unnorm_covariance(X.T, beta)
# 6.56 µs ± 1.32 µs per loop (mean ± std. dev. of 7 runs, 100,000 loops each)

%timeit y2 = gwasprs.stats.blocked_unnorm_covariance(X2.T, beta2)
# 277 µs ± 33.6 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

%timeit y2 = gwasprs.linalg.mvdot(X.T, beta)
# 34.6 µs ± 12.5 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)



%timeit gwasprs.stats.blocked_unnorm_autocovariance(X)
# 23 µs ± 1.01 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

%timeit gwasprs.linalg.mmdot(X2, X2)
# 37.8 µs ± 984 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

np.allclose(y, y2)

