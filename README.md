# Federated-Algorithms
This repository contains low-level function to support federated analysis, now mainly for Federated Genome Wide Association Study (GWAS) and Federated Survival Analysis.

## Compilation

To support functionalities from plink, we should compile source files and install executable binaries.

### Manual

Execute scripts in `scripts` to compile external programs and install executables.

### Automatic

Automatic compilation and installation are recorded in `.gitlab-ci.yml` for CI and `setup.py` for packaging.


## References
- [Federated GWAS Regression & Mechanism] [sPLINK: a hybrid federated tool as a robust alternative to meta-analysis in genome-wide association studies](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02562-1)
- [Federated SVD] [Federated horizontally partitioned principal component analysis for biomedical applications](https://academic.oup.com/bioinformaticsadvances/article/2/1/vbac026/6574370?login=false)
- [Federated Cox PH Regression] [DC-COX: Data collaboration Cox proportional hazards model for privacy-preserving survival analysis on multiple parties](https://www.sciencedirect.com/science/article/pii/S1532046422002696?via%3Dihub)