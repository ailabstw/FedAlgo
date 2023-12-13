import pyperf
import numpy as np
import jax.numpy as jnp

import gwasprs


N = 8000
BLOCK_SIZES = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

runner = pyperf.Runner()

for block_size in BLOCK_SIZES:
    X = jnp.array(np.random.rand(block_size, N))
    y = jnp.array(np.random.rand(block_size))

    runner.bench_func(f'blocked_unnorm_covariance/block size({block_size})', gwasprs.stats.blocked_unnorm_covariance, X, y)
    runner.bench_func(f'blocked_unnorm_autocovariance/block size({block_size})', gwasprs.stats.blocked_unnorm_autocovariance, X)
