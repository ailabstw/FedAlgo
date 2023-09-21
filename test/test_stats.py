import unittest
import numpy as np
from scipy.sparse import block_diag

import gwasprs


class CovarianceTestCase(unittest.TestCase):

    def setUp(self):
        self.d1 = 100
        self.d2 = 10
        self.X = np.random.rand(self.d1, self.d2)
        self.y = np.random.rand(self.d1)

    def tearDown(self):
        self.d1 = None
        self.d2 = None
        self.X = None
        self.y = None

    def test_unnorm_autocovariance(self):
        result = gwasprs.unnorm_autocovariance(self.X)
        ans = self.X.T @ self.X
        np.testing.assert_array_almost_equal(ans, result, decimal=5)

    def test_unnorm_covariance(self):
        result = gwasprs.unnorm_covariance(self.X, self.y)
        ans = self.X.T @ self.y
        np.testing.assert_array_almost_equal(ans, result, decimal=5)


class BlockCovarianceTestCase(unittest.TestCase):

    def setUp(self):
        self.d1 = 100
        self.d2 = 10
        self.A = np.random.rand(self.d1 - 1, self.d2)
        self.B = np.random.rand(self.d1 - 2, self.d2)
        self.C = np.random.rand(self.d1 - 3, self.d2)
        self.X = block_diag([self.A, self.B, self.C])
        self.y = np.random.rand(3 * self.d2)

    def tearDown(self):
        self.A = None
        self.B = None
        self.C = None
        self.X = None
        self.y = None

    def test_blocked_unnorm_autocovariance(self):
        result = gwasprs.stats.blocked_unnorm_autocovariance(self.X)
        ans = (self.X.T @ self.X).toarray()
        np.testing.assert_array_almost_equal(ans, result, decimal=5)

    def test_blocked_unnorm_covariance(self):
        result = gwasprs.stats.blocked_unnorm_covariance(self.X.T, self.y)
        ans = self.X @ self.y
        np.testing.assert_array_almost_equal(ans, result, decimal=5)
