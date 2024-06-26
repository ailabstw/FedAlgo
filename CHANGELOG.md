# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 24.05.23

### Changed

- Initial release of `fedalgo`, the renamed version of `gwasprs`, to make this repository more general to support any federated algorithms.

### Added
- Add federated Cox PH Regression `cox.py`.

### Migration Guide
To help users transition from `gwasprs` to `fedalgo`, follow these steps:

**Package Import:**
   Update all your import statements in the codebase.
   ```python
   # Old import
   import gwasprs

   # New import
   from fedalgo import gwasprs
   ```

## [0.21.0]

### Fixed

- Use the numpy to perform mean imputation, the output will be either `np.ndarray` or `jax.Array`.

### Added

- Add functions for performing federated LD pruning (`ld.py`).
- Add functions for plotting the manhattan and qq plots of GWAS result (`gwasplot.py`).

## [0.20.0]

### Fixed

- Fix the precision issue by replacing `cdf` with `sf` to calculate p-value.
- Change `jax_cpu_cores` to `jax_dev_count` for more general usage.
- Format coding style.

## [0.19.8]

### Fixed

- Fix the issue came from `orthonormalize` output.
- Fix the mean imputation issue when matrix too large.
- Fix the array type under the regression objects.

## [0.19.7]

### Added

- Add expected total steps in `GWASDataIterator`.

## [0.19.6]

### Added

- Return singular values in `orthonormalize()`.

## [0.19.5]

### Fixed

- Clean the implementation of logistic regression.

## [0.19.4]

### Fixed

- Update plink2 download url.

## [0.19.3]

### Fixed

- Return `BlockDiagonalMatrix` while indexing with interval.

## [0.19.2]

### Fixed

- Adjust the implementation of `create_snp_table`.

### Added

- Add `get_qc_metadata` for removing unnecessary bfile reading step.

## [0.19.1]

### Fixed

- Add array type checking in `array.py` and `aggregations.py`.

## [0.19.0]

### Added

- Support iterate array over rows or columns using `ArrayIterator`.

## [0.18.6]

### Fixed

- Add array type checking in `mask.py`.

## [0.18.5]

### Added

- `BlockDiagonalMatrix` supports initializing from list and dense matrix, `fromdense`, `fromlist` and `fromindex`.

### Fixed

- Replace `scipy.linalg.block_diag` with self-defining `block_diag` supporting n-dimensional arrays.

## [0.18.4]

### Added

- Add `tolist` to `BlockDiagonalMatrix`.

### Fixed

- Move `drop_missing_samples` to `kwargs` instead.
- Fix logics in `iterator.py`.

## [0.18.3]

### Added

- Add block diagonal matrix multiplication assertion.

## [0.18.2]

### Added

- Make `drop_missing_samples` optional in `GWASData`.

## [0.18.1]

### Fixed

- Fix the endless iteration of `SNPIterator`.
- Fix the additional step for two-layer iterations.

## [0.18.0]

### Added

- `Intersect` supports aggregate list of list.

## [0.17.0]

### Added

- Covariates filtering api follows the bfile filtering.

### Fixed

- Make qc codes easy to read.

## [0.16.7]

### Fixed

- Bug fix.

## [0.16.6]

### Fixed

- Make poetry to build plink2.

## [0.16.5]

### Fixed

- Fix the incorrect sizes of the outputs from `GWASDataIterator`.

## [0.16.4]

### Fixed

- Remove keeping dropped information.

## [0.16.3]

### Fixed

- Unit test over python 3.12.

## [0.16.2]

### Fixed

- Ensure dimension of `mse` and `XtX` are matched, otherwise align them together.
- Resolve frequent CI error due to numerical error.

## [0.16.1]

### Fixed

- Increase numerical stability for `inv` and `InverseSolver`.

## [0.16.0]

### Fixed

- Change `BlockedLinearRegression.sse(X, y, nobs)` to `BlockedLinearRegression.sse(X, y)`.

## [0.15.8]

### Fixed

- Increase numerical stability for `CholeskySolver`.

## [0.15.7]

### Added

- `BlockDiagonalMatrix` supports addition only for matrix which poses the same `blockshapes`.

### Fixed

- `SumUp` supports `BlockDiagonalMatrix`.

## [0.15.6]

### Added

- Support `BlockDiagonalMatrix.shape`.

### Fixed

- `CholeskySolver` supports `BlockDiagonalMatrix`.

## [0.15.5]

### Added

- Support `BlockDiagonalMatrix.append(BlockDiagonalMatrix)`.

### Fixed

- Deep copy arrays while construsting `BlockDiagonalMatrix`.

## [0.15.4]

### Fixed

- Improve `t_stats` and add mse.

## [0.15.3]

### Fixed

- Adapt `blocked_unnorm_autocovariance` and `blocked_unnorm_covariance` to `BlockDiagonalMatrix`.

## [0.15.2]

### Fixed

- `BlockedLinearRegression` integrates `BlockDiagonalMatrix`.

## [0.15.1]

### Fixed

- Fix `GWASData.subset`.
- Change dependency `polars` to `pandas`.

## [0.15.0]

### Added

- Add `BlockDiagonalMatrix`, `AbstractBlockDiagonalMatrix` and `BlockDiagonalMatrixIterator`.
- Support indexing, shape, matrix-vector multiplication (`linalg.mvmul`), matrix-vector dot product (`linalg.mvdot`), matrix-matrix multiplication (`linalg.matmul`) and inverse operation (`linalg.inv`) for `BlockDiagonalMatrix`.

## [0.14.0]

### Added

- Add `BlockedLinearRegression`.

## [0.13.2]

### Added

- `LinearRegression.predict` support sparse `X`.

## [0.13.1]

### Added

- `LinearRegression` support `include_bias` parameter.

## [0.13.0]

### Added

- `SumUp` support sparse array.

## [0.12.0]

### Added

- Add `GWASData` and `GWASDataIterator`.

## [0.11.0]

### Added

- Add `blocked_unnorm_autocovariance` and `blocked_unnorm_covariance`.

## [0.10.2]

### Fixed

- Refactor 4 interfaces for PCA algorithms.

## [0.10.1]

### Added

- Migrate array functions from RAFAEL and add a series of array types and `expand_to_2dim`.

## [0.10.0]

### Added

- Add `Intersect`.

## [0.9.1]

### Added

- Refactor QC.
- Add `qc_stats`, `calculate_allele_freq_hwe`, `calculate_homo_het_count`,
    `write_snp_list`, `get_histogram` and `get_obs_count`.
- Deprecate `cal_qc_client`.

## [0.9.0]

### Added

- Refactor loader.

## [0.8.1]

### Added

- Add `include_bias` for `BatchedLinearRegression`.

### Fixed

- Correct `dof` for `BatchedLinearRegression`.
- Fix `fit` construction for `BatchedLinearRegression`.

## [0.8.0]

### Added

- Add `SumUp` aggregation.

## [0.7.3]

### Fixed

- Fix batch axis for `batched_unnorm_autocovariance`.

## [0.7.2]

### Added

- Add `gwasprs.regression.add_bias`.

## [0.7.1]

### Fixed

- Match the requirement of logistic usecase in stepfl

## [0.7.0]

### Added

- Add batched logistic regression

### Fixed

- Set batch axis to 0 for all batch operations.

## [0.6.2]

### Added

- Extract 4 interfaces for PCA algorithms
- Add comparisons between plink, numpy and fed-svd

## [0.6.1]

### Fixed

- Match the requirement of PCA usecase in stepfl


## [0.6.0]

### Added

- Main update of all federated SVD process


## [0.5.4]

### Fixed

- Fix `plink2` not found completely and support different platform plink2.


## [0.5.3]

### Fixed

- Fix `plink2` not found.


## [0.5.2]

### Fixed

- swap in loader, remove self before BIM.
- add `--rm-dup 'force-first'` in qc stat calculation to deal with duplication problem
- chromosome name is set to `1..22` instead of `1..21`
- Some minor text adjustment
- (Batched)InverseSolver solve linear system bypassing inverse matrix


## [0.5.1]

### Fixed

- Switch nose to pytest.
- Add `setup_plink2` and collect to setup.py.


## [0.5.0]

### Added

- `batched_unnorm_autocovariance` and `BatchedLinearRegression` support `pmap`.


## [0.4.1]

### Fixed

- Fix `BatchedCholeskySolver`.


## [0.4.0]

### Added

- Add `batched_diagonal` to linalg.

### Fixed

- Fix `BatchedLinearRegression.t_stats`.


## [0.3.1]

### Fixed

- Fix `BatchedLinearRegression.predict`.


## [0.3.0]

### Added

- Add `batched_matmul`, `batched_mvdot`, `batched_mvdot`, `batched_mvmul`, `batched_inv`, `batched_cholesky`, `batched_solve_triangular`, `BatchedInverseSolver`, `BatchedCholeskySolver` to linalg.
- Add `batched_unnorm_autocovariance`, `batched_unnorm_covariance` to stats.
- Add `BatchedLinearRegression` to regression.
- Add `isnonnan` to mask.


## [0.2.1]

### Added

- Add test data for `test_qc` and `test_loader`.
- Add `get_snp_table` in `GwasDataLoader` to deal with duplicated snp.

### Fixed

- Qc, non-autosome snp will cause abnormal `.hardy` file (haploid will be removed, x saved as .hardy.x), implementation for filtering out non-autosome snp in loader
- Fix snp rename max allele size to 23
