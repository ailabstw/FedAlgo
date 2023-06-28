# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


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