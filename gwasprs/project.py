from jax import numpy as jnp
from jax import scipy as jsp
import logging

from . import stats, linalg

# Aggregator

## gram_schmidt
    
def init_gram_schmidt(norms):
    """First step of the orthonormalization process 

    Calculate the norm of the first eigenvector
    original federated_gram_schmidt_init_step

    Args:
        norms (list of np.floating) : partial norms collected from edges

    Returns:
        (list of np.floating) : real norm of the first eigenvector
    """
    return [jnp.sum(jnp.asarray(norms))]

def aggregate_residuals(residuals):
    """Calculate the real residuals

    When orthogonalizing ith eigenvector, there are i-1 values in the residuals.
    original compute_residuals_step

    Args:
        residuals (the list of list of np.floating) : partial residuals collected from edges, the size of each vector depends on rank of eigenvector.
    
    Returns:
        (np.ndarray[(1,), np.floating]) : real residuals used for projecting ith eigenvector with shape (i-1,).
    """
    return jnp.sum(jnp.asarray(residuals), axis=0)

def aggregate_norms(norms):
    """Calculate the real norm of ith eigenvector

    After computing residuals for get the orthogonal vector, calculate the norm of ith eigenvector used for normalization.
    original norm_aggregation_step

    Args:
        norms (list of np.floating) : partial norms of ith eigenvector collected from edges.
    
    Returns:
        (np.floating) : real norm of ith eigenvector 
    """
    return jnp.sum(jnp.asarray(norms))

def update_H(h_matrices):
    """
    original final_H_update_step
    """
    H = jnp.sum(jnp.asarray(h_matrices), axis=0)
    # H shape: (n_SNPs, k)
    H, R = jsp.linalg.qr(H, mode='economic')
    return H


# client

## gram_schmidt

def compute_residuals_step(G, Ortho, eigen_idx, norms):
    """Calculate the local residuals

    When orthogonalizing ith eigenvector, there are i-1 values in the residuals.
    (u.v)/(u norm), where u is the orthogonalized vector and v is the eigenvector to be orthogonalized.

    Args:
        G (np.ndarray[(1,1), np.floating]) : The G matrix with shape (n, k2), where n and k2 represent the number of samples and the latent dimensions decided in gwasprs.linalg.decompose_cov_matrices step.
        Ortho (list of np.ndarray[(1,), np.floating]) : the list stores i-1 orthogonalized vectors (n,) of G matrix.
        eigen_idx (int) : the index represents ith eigenvector, e.g. 2nd eigenvector: eigen_idx=1.
    
    Returns:
        (list of np.floating) : i-1 residuals used for orthogonalized ith eigenvector.
    """
    residuals = []
    for res_idx in range(eigen_idx):
        u = Ortho[res_idx]
        v = G[:,eigen_idx]
        r = jnp.vdot(u,v)/norms[res_idx]
        residuals.append(r)
    
    return residuals

def orthogonalize_step(G, Ortho, eigen_idx, residuals):
    """Orthogonalize

    Orthogonalize ith eigenvector after collecting i-1 real residuals.

    Args:
        G (np.ndarray[(1,1), np.floating]) : The G matrix with shape (n, k2), where n and k2 represent the number of samples and the latent dimensions decided in gwasprs.linalg.decompose_cov_matrices step.
        Ortho (list of np.ndarray[(1,), np.floating]) : the list stores i-1 orthogonalized vectors (n,) of G matrix.
        eigen_idx (int) : the index represents ith eigenvector, e.g. 2nd eigenvector: eigen_idx=1.
        residuals (np.ndarray[(1,), np.floating]) : real residuals used for orthogonalization with shape (i-1,).
    
    Returns:
        (np.floating) : the partial norm of ith eigenvector.
    """
    u = linalg.orthogonal_project(G[:,eigen_idx], Ortho, residuals)
    Ortho.append(u)
    return jnp.vdot(u,u)

def normalize_step(norms, Ortho):
    """Normalization

    Make the norms of eigenvectors = 1

    Args:
        norms (list of np.floating) : i norms
        Ortho (list of np.ndarray[(1,), np.floating]) : i orthogonalized eigenvectors
    
    Returns:
        (np.ndarray[(1,1), np.floating]) : orthonormalized G matrix with shape (n, k2), where n and k2 represent the number of samples and the latent dimensions decided in gwasprs.linalg.decompose_cov_matrices step.
    """
    G = stats.normalize(norms, Ortho)
    return G
    