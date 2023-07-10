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
        A = jnp.expand_dims(A.T @ A, -1)
        X = np.array(
            [[[1], [1], [1]],
             [[2], [2], [2]],
             [[3], [3], [3]],
             [[4], [4], [4]]]
        )
        Y = np.array(
            [[[1], [1], [1]],
             [[2], [2], [2]],
             [[3], [3], [3]]]
        )
        y = np.array(
            [[1],
             [2],
             [3]]
        )
        self.A = np.concatenate((A, A), axis=2)
        self.X = np.concatenate((X, X), axis=2)
        self.Y = np.concatenate((Y, Y), axis=2)
        self.y = np.concatenate((y, y), axis=1)

    def tearDown(self):
        self.X = None
        self.Y = None
        self.y = None

    def test_batched_mvmul(self):
        result = gwasprs.linalg.batched_mvmul(self.X, self.y)
        ans = np.array(
            [[ 6,  6],
             [12, 12],
             [18, 18],
             [24, 24]]
        )
        np.testing.assert_array_equal(ans, result)

    def test_batched_matmul(self):
        result = gwasprs.linalg.batched_matmul(self.X, self.Y)
        ans = np.array(
            [[[ 6,  6],
              [ 6,  6],
              [ 6,  6]],
             [[12, 12],
              [12, 12],
              [12, 12]],
             [[18, 18],
              [18, 18],
              [18, 18]],
             [[24, 24],
              [24, 24],
              [24, 24]]]
        )
        np.testing.assert_array_equal(ans, result)

    def test_batched_mvdot(self):
        result = gwasprs.linalg.batched_mvdot(self.Y, self.y)
        ans = np.array(
            [[14, 14],
             [14, 14],
             [14, 14],]
        )
        np.testing.assert_array_equal(ans, result)

    def test_batched_mmdot(self):
        result = gwasprs.linalg.batched_mmdot(self.Y, self.Y)
        ans = np.array(
            [[[14, 14],
              [14, 14],
              [14, 14]],
             [[14, 14],
              [14, 14],
              [14, 14]],
             [[14, 14],
              [14, 14],
              [14, 14]],]
        )
        np.testing.assert_array_equal(ans, result)

    def test_batched_diagonal(self):
        d = np.array([1,2,3])
        D = np.diag(d)
        D = np.expand_dims(D, -1)
        D = np.concatenate((D, D), axis=2)
        result = gwasprs.linalg.batched_diagonal(D)

        d = np.expand_dims(d, -1)
        ans = np.concatenate((d, d), axis=1)
        np.testing.assert_array_equal(ans, result)

    def test_batched_cholesky(self):
        L = np.linalg.cholesky(self.A[:, :, 0])
        result = gwasprs.linalg.batched_cholesky(self.A)

        L = np.expand_dims(L, -1)
        ans = np.concatenate((L, L), axis=2)
        np.testing.assert_array_equal(ans, result)

    def test_batched_cholesky_solver(self):
        y = np.random.randn(4)
        y = np.expand_dims(y, -1)
        y = np.concatenate((y, y), axis=1)
        result = gwasprs.linalg.BatchedCholeskySolver()(self.A, y)
        ans = slinalg.solve(self.A[:, :, 0], y[:, 0])
        ans = np.expand_dims(ans, -1)
        ans = np.concatenate((ans, ans), axis=1)
        norm = np.linalg.norm(result - ans)
        self.assertAlmostEqual(norm, 0, places=5)

class FedSvdTestCase(unittest.TestCase):

    def setUp(self):
        key = random.PRNGKey(758493)
        self.n_clients = 3

        # Orthonormalization
        G, R = gwasprs.array.simulate_eigenvectors(30, 21)
        self.Ps = [G[i*7:(i+1)*7, :].T for i in range(self.n_clients)]
        self.U = R

        # Randomzied SVD
        # Edge
        A = gwasprs.array.simulate_genotype_matrix(key, shape=(30,60), impute=True, standardize=True)
        self.As = [A[i*10:(i+1)*10, :].T for i in range(self.n_clients)]
        n_samples = int(A.shape[0]/self.n_clients)
        self.Gs = [gwasprs.linalg.randn(n_samples, n_samples, i) for i in range(self.n_clients)]
        self.Gs = [jsp.linalg.qr(g, mode='economic')[0] for g in self.Gs]

        # Aggregator
        self.prev_H = gwasprs.linalg.init_H(60, n_samples)
        self.epsilon = 1e-9
        self.current_iteration = 1
        self.max_iterations = 20
        self.H_converged = False
        self.Hs = []
        self.k = 10

    
    def tearDown(self):
        self.Ps, self.U = None, None
        self.As, self.Gs, self.Hs = None, None, None
        self.prev_H = None

    def test_federated_gram_schmidt_orthonormalization(self):
        Gs = [gwasprs.linalg.mmdot(p, self.U) for p in self.Ps]
        G = np.concatenate(Gs, axis=0)
        ans, R = jsp.linalg.qr(G, mode='economic')
        
        # First eigenvector
        Gs, norms, orthos = [], [], []
        for p in self.Ps:
            g, norm, ortho = gwasprs.linalg.local_G_and_init_orthonormalization(p, self.U)
            Gs.append(g)
            norms.append(norm)
            orthos.append(ortho)
        NORMS = gwasprs.project.init_gram_schmidt(norms)

        # Rest
        for EIGEN_IDX in range(1,self.U.shape[1]):
            residuals = [gwasprs.project.compute_residuals_step(Gs[i], orthos[i], EIGEN_IDX, NORMS) for i in range(len(Gs))]
            RESIDUALS = gwasprs.project.residuals(residuals)
            norms = [gwasprs.project.orthogonalize_step(Gs[i], orthos[i], EIGEN_IDX, RESIDUALS) for i in range(len(Gs))]
            NORMS.append(gwasprs.project.aggregate_norms(norms))
        result = np.concatenate([gwasprs.project.normalize_step(NORMS, orthos[i]) for i in range(len(Gs))], axis=0)

        INNER_PRODUCTS = gwasprs.linalg.mmdot(ans, result)
        np.testing.assert_array_almost_equal(np.identity(INNER_PRODUCTS.shape[0]), abs(INNER_PRODUCTS), decimal=5)
    
    def test_federated_randomized_svd(self):
        A = np.concatenate(self.As, axis=1)
        u, s, ans = jsp.linalg.svd(A, full_matrices=False)

        # Algorithm 2 iterations
        while not self.H_converged and self.current_iteration < self.max_iterations:
            # Update global H
            Hs = [gwasprs.linalg.update_local_H(self.As[i], self.Gs[i]) for i in range(self.n_clients)]
            GLOBAL_H = gwasprs.linalg.update_global_H(Hs)

            # Check convergence & save H
            self.H_converged, _ = gwasprs.linalg.check_eigenvector_convergence(GLOBAL_H, self.prev_H, self.epsilon)
            self.prev_H, self.Hs, self.current_iteration = gwasprs.linalg.iteration_update(GLOBAL_H, self.Hs, self.current_iteration, self.max_iterations)

            # Update local G
            for i in range(self.n_clients):
                g = gwasprs.linalg.update_local_G(self.As[i], GLOBAL_H)
                self.Gs[i] = g
        
        # Algorithm 5 main steps
        GLOBAL_H = gwasprs.linalg.decompose_H_stack(self.Hs) 
        Ps, covs = [], []
        for a in self.As:
            p, cov = gwasprs.linalg.local_cov_matrix(a, GLOBAL_H)
            Ps.append(p)
            covs.append(cov)
        GLOBAL_U = gwasprs.linalg.decompose_cov_matrices(covs, self.k)
        for i in range(self.n_clients):
            g, norm, ortho = gwasprs.linalg.local_G_and_init_orthonormalization(Ps[i], GLOBAL_U)
            self.Gs[i] = g
        result = np.concatenate(self.Gs, axis=0)
        result, r = jsp.linalg.qr(result, mode='economic')

        INNER_PRODUCTS = gwasprs.linalg.mmdot(ans.T, result)[:self.k, :self.k]
        np.testing.assert_array_almost_equal(np.identity(INNER_PRODUCTS.shape[0]), abs(INNER_PRODUCTS), decimal=5)