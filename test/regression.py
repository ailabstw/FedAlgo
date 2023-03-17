import unittest
import federatedprs
import numpy as np

class FedLinearRegressionTestCase(unittest.TestCase):

    def setUp(self):
        self.n = 100
        self.dim = 10
        beta = np.random.rand(self.dim)
        self.model = federatedprs.FedLinearRegression(beta)
        self.X = np.random.rand(self.n, self.dim)

    def tearDown(self):
        self.n = None
        self.dim = None
        self.model = None
        self.X = None

    def test_predict(self):
        result = self.model.predict(self.X)
        self.assertEqual((self.n, ), result.shape)

    def test_vmap_predict(self):
        result = self.model.predict(self.X)
        self.assertEqual((self.n, ), result.shape)
