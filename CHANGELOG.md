# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1]

### Added

- Add test data for `test_qc` and `test_loader`.
- Add `get_snp_table` in `GwasDataLoader` to deal with duplicated snp.

### Fixed

- Qc, non-autosome snp will cause abnormal `.hardy` file (haploid will be removed, x saved as .hardy.x), implementation for filtering out non-autosome snp in loader
- Fix snp rename max allele size to 23
