from jax import numpy as jnp
from jax import scipy as jsp
import logging

import stats, linalg, vectorize

# Aggregator

## gram_schmidt
    
def init_gram_schmidt(norms):
    """
    original federated_gram_schmidt_init_step
    """
    return [jnp.sum(jnp.asarray(norms))]

def residuals(residuals):
    """
    original compute_residuals_step
    """
    return jnp.sum(jnp.asarray(residuals), axis=0)

def aggregate_norms(norms):
    """
    original norm_aggregation_step
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

def compute_residuals_step(Ortho, G, eigen_idx, norms):
    residuals = []
    for res_idx in range(eigen_idx):
        u = Ortho[res_idx]
        v = G[:,eigen_idx]
        r = vectorize.fast_dot(u, v)/norms[res_idx]

        residuals.append(r)
    
    return residuals

def orthogonalize_step(G, Ortho, eigen_idx, residuals):
        u = linalg.orthogonal_project(G[:,eigen_idx], Ortho, residuals)
        Ortho.append(u)
        return vectorize.fast_dot(u, u)

def normalize_step(norms, Ortho):
    G = stats.normalize(norms, Ortho)
    logging.debug(f'G matrix {G.shape}\n{G}')
    return G
    