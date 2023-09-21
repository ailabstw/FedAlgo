from scipy.sparse import block_diag

from . import mask, array


def dropna_block_diag(genotype, covariates):
    As = []
    for i in range(genotype.shape[1]):
        A = mask.dropnan(array.concat((genotype[:, i:i+1], covariates)))
        As.append(A)
    return block_diag(As)