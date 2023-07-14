# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
