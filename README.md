# Federated-PRS

## Install

Execute following command with network connection to AI Labs.

`pip install gwasprs --index-url https://gitlab-ci-token:fz91zDTrZV-1T1ysa2tv@gitlab.corp.ailabs.tw/api/v4/projects/3247/packages/pypi/simple`

## Compilation

To support functionalities from plink, we should compile source files and install executable binaries.

### Manual

Execute scripts in `scripts` to compile external programs and install executables.

### Automatic

Automatic compilation and installation are recorded in `.gitlab-ci.yml` for CI and `setup.py` for packaging.

## Development information

you can find token in docs


## References

- [sPLINK: a hybrid federated tool as a robust alternative to meta-analysis in genome-wide association studies](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02562-1)
- [Federated-SVD](https://arxiv.org/pdf/2205.12109.pdf)
