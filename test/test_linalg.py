import unittest
import gwasprs
import numpy as np
from jax import random
import jax.numpy as jnp

class LinAlgTestCase(unittest.TestCase):

    def setUp(self):
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
        key = random.PRNGKey(758493)
        A = random.uniform(key, shape=(3, 4))
        X = A.T @ A
        L = np.linalg.cholesky(X)

        X = jnp.expand_dims(X, -1)
        X = np.concatenate((X, X), axis=2)
        result = gwasprs.linalg.batched_cholesky(X)

        L = np.expand_dims(L, -1)
        ans = np.concatenate((L, L), axis=2)
        np.testing.assert_array_equal(ans, result)
