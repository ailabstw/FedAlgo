import unittest
import gwasprs
import numpy as np
import scipy.linalg as slinalg
from jax import random
import jax.numpy as jnp

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
