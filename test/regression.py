import unittest
import federatedprs
import numpy as np
from scipy.stats import norm

class LinearRegressionTestCase(unittest.TestCase):

    def setUp(self):
        self.n = 100
        self.dim = 10
        self.X = np.random.rand(self.n, self.dim)
        self.beta = np.random.rand(self.dim)
        self.y = np.dot(self.X, self.beta) + np.random.rand(self.n)
        self.model = federatedprs.LinearRegression(self.beta)

    def tearDown(self):
        self.n = None
        self.dim = None
        self.model = None
        self.X = None
        self.y = None

    def test_predict(self):
        result = self.model.predict(self.X)
        np.testing.assert_array_almost_equal(np.dot(self.X, self.beta), result, decimal=5)

    def test_fit(self):
        model = federatedprs.LinearRegression.fit(self.X, self.y)
        self.assertTrue(True)
        
    def test_residual(self):
        result = self.model.residual(self.X, self.y)
        np.testing.assert_array_almost_equal(self.y - self.model.predict(self.X), result, decimal=5)
    
    def test_sse(self):
        result = self.model.sse(self.X, self.y)
        resd = self.model.residual(self.X, self.y)
        np.testing.assert_array_almost_equal(np.vdot(resd.T, resd), result, decimal=5)

class LogisticRegressionTestCase(unittest.TestCase):

    def setUp(self):
        self.n = 100
        self.dim = 10
        self.X = np.random.rand(self.n, self.dim)
        self.beta = np.random.rand(self.dim)
        z = np.dot(self.X, self.beta)
        pred_y = norm.cdf(z - np.mean(z) + np.random.randn(self.n))
        binarize = lambda x: 1.0 if x > 0.5 else 0.0
        self.y = np.array(list(map(binarize, pred_y)))
        self.model = federatedprs.LogisticRegression(self.beta)

    def tearDown(self):
        self.n = None
        self.dim = None
        self.model = None
        self.X = None
        self.y = None

    def test_predict(self):
        result = self.model.predict(self.X)
        predicted_y = 1 / (1 + np.exp(-np.dot(self.X, self.beta)))
        predicted_y = np.expand_dims(predicted_y, -1)
        np.testing.assert_array_almost_equal(predicted_y, result, decimal=5)

    def test_fit(self):
        self.model.fit(self.X, self.y)
        self.assertTrue(True)
        
    def test_residual(self):
        result = self.model.residual(self.X, self.y)
        ans = np.expand_dims(self.y, -1) - self.model.predict(self.X)
        np.testing.assert_array_almost_equal(ans, result, decimal=5)
    
    def test_gradient(self):
        result = self.model.gradient(self.X, self.y)
        ans = np.dot(self.X.T, self.model.residual(self.X, self.y))
        np.testing.assert_array_almost_equal(ans, np.expand_dims(result, -1), decimal=5)
    
    def test_hessian(self):
        pass
    
    def test_loglikelihood(self):
        pass
    