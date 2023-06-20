import unittest
import gwasprs
import numpy as np

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
        np.testing.assert_array_almost_equal(self.X.T @ self.X, result, decimal=5)

    def test_unnorm_covariance(self):
        result = gwasprs.unnorm_covariance(self.X, self.y)
        np.testing.assert_array_almost_equal(np.dot(self.X.T, self.y), result, decimal=5)
