import numpy as np
import jax.numpy as jnp
from bed_reader import open_bed
import pandas as pd

from fedalgo import gwasprs


BATCH = 10

# read files

bfile_prefix = "/Users/yuehhua/workspace/DEMO/DEMO_CLF/demo_hg38"
my_bed = open_bed(f"{bfile_prefix}.bed")
metadata = pd.read_csv(f"{bfile_prefix}.csv")


# data

nsample = my_bed.iid_count
nsnp = my_bed.sid_count

genotype = my_bed.read()
covariates = covariates = metadata[["SEX", "AGE"]].values


# model

X = np.expand_dims(np.nan_to_num(genotype[:, 0:BATCH]).T, 1)
C = np.repeat(np.expand_dims(covariates.T, 0), BATCH, 0)
X = jnp.concatenate([X, C], 1)
beta = jnp.array(np.random.rand(X.shape[0], 3))

y_hat = gwasprs.stats.batched_unnorm_covariance(X, beta)

# BATCH = 10
# In [1]: %timeit gwasprs.stats.batched_unnorm_covariance(X, beta)
# 62 µs ± 23.5 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

# BATCH = 20
# In [4]: %timeit gwasprs.stats.batched_unnorm_covariance(X, beta)
# 158 µs ± 52.9 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

# BATCH = 50
# In [15]: %timeit gwasprs.stats.batched_unnorm_covariance(X, beta)
# 517 µs ± 67.5 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

# BATCH = 100
# In [19]: %timeit gwasprs.stats.batched_unnorm_covariance(X, beta)
# 766 µs ± 53.7 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

# BATCH = 200
# In [24]: %timeit gwasprs.stats.batched_unnorm_covariance(X, beta)
# 2.11 ms ± 70.8 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)

# BATCH = 500
# In [27]: %timeit gwasprs.stats.batched_unnorm_covariance(X, beta)
# 6.17 ms ± 215 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# BATCH = 1000
# In [3]: %timeit gwasprs.stats.batched_unnorm_covariance(X, beta)
# 15.5 ms ± 2.76 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)

# BATCH = 2000
# In [6]: %timeit gwasprs.stats.batched_unnorm_covariance(X, beta)
# 25.3 ms ± 4.17 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# BATCH = 3000
# In [9]: %timeit gwasprs.stats.batched_unnorm_covariance(X, beta)
# 36.3 ms ± 822 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)

# BATCH = 4000
# In [12]: %timeit gwasprs.stats.batched_unnorm_covariance(X, beta)
# 48.6 ms ± 1.69 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# BATCH = 5000
# In [15]: %timeit gwasprs.stats.batched_unnorm_covariance(X, beta)
# 141 ms ± 29.5 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

# BATCH = 7500
# In [18]: %timeit gwasprs.stats.batched_unnorm_covariance(X, beta)
# 278 ms ± 59.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# BATCH = 10000
# In [21]: %timeit gwasprs.stats.batched_unnorm_covariance(X, beta)
# 349 ms ± 109 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


cov = gwasprs.stats.batched_unnorm_autocovariance(X)

# BATCH = 10
# In [12]: %timeit gwasprs.stats.batched_unnorm_autocovariance(X)
# 5.09 s ± 598 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# BATCH = 20
# In [5]: %timeit gwasprs.stats.batched_unnorm_autocovariance(X)
# 14.4 s ± 3.24 s per loop (mean ± std. dev. of 7 runs, 1 loop each)

# BATCH = 50
# In [16]: %timeit gwasprs.stats.batched_unnorm_autocovariance(X)
# 31.2 s ± 3.93 s per loop (mean ± std. dev. of 7 runs, 1 loop each)

# BATCH = 100
# In [20]: %timeit gwasprs.stats.batched_unnorm_autocovariance(X)
# 2min 32s ± 56.9 s per loop (mean ± std. dev. of 7 runs, 1 loop each)

# BATCH = 200
# In [23]: %timeit gwasprs.stats.batched_unnorm_autocovariance(X)
# 2min 46s ± 1min 28s per loop (mean ± std. dev. of 7 runs, 1 loop each)

