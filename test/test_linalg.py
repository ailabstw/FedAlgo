import unittest
import gwasprs
import numpy as np
import scipy.linalg as slinalg
from jax import random
import jax.numpy as jnp
import jax.scipy as jsp

class LinAlgTestCase(unittest.TestCase):

    def setUp(self):
        key = random.PRNGKey(758493)
        A = random.uniform(key, shape=(4, 4))
        A = jnp.expand_dims(A.T @ A, axis=0)
        X = np.array(
            [[[1,1,1],
              [2,2,2],
              [3,3,3],
              [4,4,4]]]
        )
        Y = np.array(
            [[[1, 1, 1],
              [2, 2, 2],
              [3, 3, 3]]]
        )
        y = np.array(
            [[[1],
              [2],
              [3]]]
        )
        self.A = np.concatenate((A, A), axis=0)
        self.X = np.concatenate((X, X), axis=0)
        self.Y = np.concatenate((Y, Y), axis=0)
        self.y = np.concatenate((y, y), axis=0)

    def tearDown(self):
        self.X = None
        self.Y = None
        self.y = None

    def test_batched_mvmul(self):
        result = gwasprs.linalg.batched_mvmul(self.X, self.y)
        ans = np.array(
            [[ 6, 12, 18, 24],
             [ 6, 12, 18, 24]]
        )
        np.testing.assert_array_equal(ans, result)

    def test_batched_matmul(self):
        result = gwasprs.linalg.batched_matmul(self.X, self.Y)
        ans = np.array(
            [[[ 6,  6,  6],
              [12, 12, 12],
              [18, 18, 18],
              [24, 24, 24]],

             [[ 6,  6,  6],
              [12, 12, 12],
              [18, 18, 18],
              [24, 24, 24]]]
        )
        np.testing.assert_array_equal(ans, result)

    def test_batched_mvdot(self):
        result = gwasprs.linalg.batched_mvdot(self.Y, self.y)
        ans = np.array(
            [[14, 14, 14],
             [14, 14, 14]]
        )
        np.testing.assert_array_equal(ans, result)

    def test_batched_mmdot(self):
        result = gwasprs.linalg.batched_mmdot(self.Y, self.Y)
        ans = np.array(
            [[[14, 14, 14],
              [14, 14, 14],
              [14, 14, 14]],

             [[14, 14, 14],
              [14, 14, 14],
              [14, 14, 14]]]
        )
        np.testing.assert_array_equal(ans, result)

    def test_batched_diagonal(self):
        d = np.array([1,2,3])
        D = np.diag(d)
        D = np.expand_dims(D, axis=0)
        D = np.concatenate((D, D), axis=0)
        result = gwasprs.linalg.batched_diagonal(D)

        d = np.expand_dims(d, axis=0)
        ans = np.concatenate((d, d), axis=0)
        np.testing.assert_array_equal(ans, result)

    def test_batched_cholesky(self):
        L = np.linalg.cholesky(self.A[0, :, :])
        result = gwasprs.linalg.batched_cholesky(self.A)

        L = np.expand_dims(L, axis=0)
        ans = np.concatenate((L, L), axis=0)
        np.testing.assert_array_equal(ans, result)

    def test_batched_cholesky_solver(self):
        y = np.random.randn(4)
        y = np.expand_dims(y, axis=0)
        y = np.concatenate((y, y), axis=0)
        result = gwasprs.linalg.BatchedCholeskySolver()(self.A, y)
        ans = slinalg.solve(self.A[0, :, :], y[0, :])
        ans = np.expand_dims(ans, axis=0)
        ans = np.concatenate((ans, ans), axis=0)
        norm = np.linalg.norm(result - ans)
        self.assertAlmostEqual(norm, 0, places=5)



class FederatedSVDTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        key = random.PRNGKey(758493)
        n_edges, n_samples, n_SNPs = 4, [30,50,40,60], 200
        self.k1, self.k2, self.max_iterations, self.epsilon = 20, 20, 20, 1e-9

        # init A : n_samples * n_SNPs
        self.As = [gwasprs.array.simulate_genotype_matrix(key, shape=(n_samples[edge_idx], n_SNPs)) for edge_idx in range(n_edges)]
        self.shared_args = {'As':self.As, 'formated':True, 'edge_axis':0, 'sample_axis':1, 'snp_axis':2}
        split_idx = [sum(n_samples[:edge_idx+1]) for edge_idx in range(n_edges)]

        # Standardization answer
        GLOBAL_A = np.concatenate(self.As, axis=0)
        mean = np.nanmean(GLOBAL_A, axis=0)
        na_idx = np.where(np.isnan(GLOBAL_A))
        GLOBAL_A = np.array(GLOBAL_A)
        GLOBAL_A[na_idx] = np.take(mean, na_idx[1])
        self.GLOBAL_A_ans = (GLOBAL_A-np.mean(GLOBAL_A,axis=0))/np.nanstd(GLOBAL_A,axis=0,ddof=1)
        self.GLOBAL_A_ans = np.delete(self.GLOBAL_A_ans, np.isnan(self.GLOBAL_A_ans[0]),axis=1)
        self.GLOBAL_As_ans = np.vsplit(self.GLOBAL_A_ans, split_idx)[:n_edges]

        # SVD answer
        self.GLOBAL_G_ans, s, self.GLOBAL_H_ans = jsp.linalg.svd(self.GLOBAL_A_ans, full_matrices=False)
        self.GLOBAL_G_ans = self.GLOBAL_G_ans[:,0:self.k2]
        self.GLOBAL_H_ans = self.GLOBAL_H_ans.T[:,0:self.k2]

        # Orthonormalization init
        G, R = gwasprs.array.simulate_eigenvectors(np.sum(n_samples), n_SNPs, self.k2)
        self.Gs = np.vsplit(G, split_idx)[:n_edges]

    
    def test_federated_standardization(self):
        result = np.concatenate(gwasprs.linalg.federated_standardization(**self.shared_args), axis=0)
        np.testing.assert_array_almost_equal(self.GLOBAL_A_ans, result, decimal=5)
    

    def test_federated_vertical_subspace_iteration(self):
        gwasprs.linalg.federated_vertical_subspace_iteration(**self.shared_args)


    def test_federated_randomized_svd_with_vsi(self):
        kwargs = {**self.shared_args, **{'k1':self.k1, 'epsilon':self.epsilon, 'max_iterations':self.max_iterations}}
        kwargs['As'] = self.GLOBAL_As_ans

        # Vertical subspace iterations, the output As (n_SNPs, n_samples)
        As, Hs, local_Gs = gwasprs.linalg.federated_vertical_subspace_iteration(**kwargs)

        # Randomized SVD
        kwargs.update({'As':As, 'Hs':Hs, 'local_Gs':local_Gs, 'k2':self.k2, 'transposed':True})
        result_H, result_Gs = gwasprs.linalg.federated_randomized_svd(**kwargs)

        # Final update of H
        result_H = gwasprs.linalg.final_H_update(As, result_Gs)

        # Evaluations
        result_G = np.concatenate(result_Gs, axis=0)
        result_G = jsp.linalg.qr(result_G, mode='economic')[0]

        gwasprs.linalg.eigenvec_concordance_estimation(self.GLOBAL_G_ans, result_G, decimal=3)
        gwasprs.linalg.eigenvec_concordance_estimation(self.GLOBAL_H_ans, result_H, decimal=3)

    
    def test_federated_randomized_svd_without_vsi(self):
        kwargs = {**self.shared_args, **{'k1':self.k1, 'epsilon':self.epsilon, 'max_iterations':self.max_iterations, 'k2':self.k2}}
        kwargs['As'] = self.GLOBAL_As_ans

        # Randomized SVD
        result_H, result_Gs = gwasprs.linalg.federated_randomized_svd(**kwargs)

        # Final update of H
        As = gwasprs.array.genotype_matrix_input_formatter(kwargs['As'], 0, 1, 2, transpose=True)
        result_H = gwasprs.linalg.final_H_update(As, result_Gs)

        # Evaluations
        result_G = np.concatenate(result_Gs, axis=0)
        result_G = jsp.linalg.qr(result_G, mode='economic')[0]

        gwasprs.linalg.eigenvec_concordance_estimation(self.GLOBAL_G_ans, result_G, decimal=3)
        gwasprs.linalg.eigenvec_concordance_estimation(self.GLOBAL_H_ans, result_H, decimal=3)
    

    def test_federated_gram_schmidt_orthonormalization_(self):
        gwasprs.project.federated_orthonormalization(self.Gs)
    

    def test_federated_svd(self):
        kwargs = {**self.shared_args, **{'k1':self.k1, 'epsilon':self.epsilon, 'max_iterations':self.max_iterations, 'k2':self.k2}}

        # Federated SVD
        result_Gs, result_H = gwasprs.linalg.federated_svd(**kwargs)

        # Evaluations
        result_G = np.concatenate(result_Gs, axis=0)

        gwasprs.linalg.eigenvec_concordance_estimation(self.GLOBAL_G_ans, result_G, decimal=4)
        gwasprs.linalg.eigenvec_concordance_estimation(self.GLOBAL_H_ans, result_H, decimal=4)