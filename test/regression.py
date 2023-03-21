import unittest
import federatedprs
import numpy as np

class FedLinearRegressionTestCase(unittest.TestCase):

    def setUp(self):
        self.n = 100
        self.dim = 10
        self.X = np.random.rand(self.dim, self.n)
        self.beta = np.random.rand(self.dim)
        self.model = federatedprs.FedLinearRegression(self.beta)

    def tearDown(self):
        self.n = None
        self.dim = None
        self.model = None
        self.X = None

    def test_predict(self):
        result = self.model.predict(self.X)
        np.testing.assert_array_almost_equal(np.dot(self.X.T, self.beta), result, decimal=5)
