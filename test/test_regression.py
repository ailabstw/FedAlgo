import unittest
import gwasprs
import gwasprs.linalg as linalg
import numpy as np
from scipy.stats import norm

class LinearRegressionTestCase(unittest.TestCase):

    def setUp(self):
        self.n = 100
        self.dim = 10
        self.X = np.random.rand(self.n, self.dim)
        self.beta = np.random.rand(self.dim)
        self.y = np.dot(self.X, self.beta) + np.random.rand(self.n)
        self.model = gwasprs.LinearRegression(self.beta)

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
        model = gwasprs.LinearRegression.fit(self.X, self.y)
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
        self.model = gwasprs.LogisticRegression(self.beta)

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


class BatchedLinearRegressionTestCase(unittest.TestCase):

    def setUp(self):
        self.n = 100
        self.dim = 10
        self.batch_size = 2
        self.X = np.random.rand(self.batch_size, self.n, self.dim)
        self.beta = np.random.rand(self.batch_size, self.dim)
        self.y = linalg.batched_mvmul(self.X, self.beta) + np.random.rand(self.batch_size, self.n)
        self.model = gwasprs.regression.BatchedLinearRegression(self.beta)

    def tearDown(self):
        self.n = None
        self.dim = None
        self.model = None
        self.X = None
        self.y = None

    def test_predict(self):
        result = self.model.predict(self.X)
        np.testing.assert_array_almost_equal(linalg.batched_mvmul(self.X, self.beta), result, decimal=5)

    def test_fit(self):
        model = gwasprs.regression.BatchedLinearRegression.fit(self.X, self.y, algo=linalg.BatchedInverseSolver())
        self.assertTrue(True)

    def test_residual(self):
        result = self.model.residual(self.X, self.y)
        np.testing.assert_array_almost_equal(self.y - self.model.predict(self.X), result, decimal=5)

    def test_sse(self):
        result = self.model.sse(self.X, self.y)
        resd = self.model.residual(self.X, self.y)
        ans = np.expand_dims(linalg.batched_vdot(resd, resd), 0)
        np.testing.assert_array_almost_equal(ans, result, decimal=5)

    def test_t_stats(self):
        sse = self.model.sse(self.X, self.y)
        print(f'sse: {sse.shape}\n{sse}')
        XtX = linalg.batched_mmdot(self.X, self.X)
        print(f'XtX: {XtX.shape}\n{XtX}')
        dof = self.model.dof(gwasprs.mask.nonnan_count(self.X))
        print(f'dof: {dof}')
        result = self.model.t_stats(sse, XtX, dof)
        print(result)
        print(result.shape)
        self.assertEqual((self.batch_size, self.dim), result.shape)


# class BatchedLogisticRegressionTestCase(unittest.TestCase):

#     def setUp(self):
#         self.n = 100
#         self.dim = 10
#         self.batch_size = 2
#         self.X = np.random.rand(self.n, self.dim, self.batch_size)
#         self.beta = np.random.rand(self.dim, self.batch_size)
#         z = linalg.batched_mvdot(self.X, self.beta)
#         pred_y = norm.cdf(z - np.mean(z) + np.random.randn(self.n, self.batch_size))
#         binarize = lambda x: 1.0 if x > 0.5 else 0.0
#         self.y = np.array(list(map(binarize, pred_y)))
#         self.model = gwasprs.regression.BatchedLogisticRegression(self.beta)
    
#     def tearDown(self):
#         self.n = None
#         self.dim = None
#         self.model = None
#         self.X = None
#         self.y = None

#     def test_predict(self):
#         result = self.model.predict(self.X)
#         predicted_y = 1 / (1 + np.exp(-linalg.batched_mvdot(self.X, self.beta)))
#         predicted_y = np.expand_dims(predicted_y, -1)
#         np.testing.assert_array_almost_equal(predicted_y, result, decimal=5)
    
#     def test_fit(self):
#         self.model.fit(self.X, self.y)
#         self.assertTrue(True)
    
#     def test_residual(self):
#         result = self.model.residual(self.X, self.y)
#         ans = np.expand_dims(self.y, -1) - self.model.predict(self.X)
#         np.testing.assert_array_almost_equal(ans, result, decimal=5)

#     def test_gradient(self):
#         result = self.model.gradient(self.X, self.y)
#         ans = linalg.batched_mvdot(self.X.T, self.model.residual(self.X, self.y))
#         np.testing.assert_array_almost_equal(ans, np.expand_dims(result, -1), decimal=5)

#     def test_hessian(self):
#         pass

#     def test_loglikelihood(self):
#         pass

    


    