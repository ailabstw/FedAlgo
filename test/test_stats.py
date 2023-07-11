import unittest
import gwasprs
import numpy as np
from jax import random

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

class StandardizationTestCase(unittest.TestCase):

    def setUp(self):
        key = random.PRNGKey(758493)
        self.X = gwasprs.array.simulate_genotype_matrix(key,shape=(21,30))
        self.As = [self.X[i*7:(i+1)*7, :] for i in range(3)]

    def tearDown(self):
        self.X, self.As = None, None

    def test_federated_standardization(self):
        mean = np.nanmean(self.X, axis=0)
        na_idx = np.where(np.isnan(self.X))
        self.X = np.array(self.X)
        self.X[na_idx] = np.take(mean, na_idx[1])
        ans = (self.X-np.mean(self.X,axis=0))/np.nanstd(self.X,axis=0,ddof=1)
        ans = np.delete(ans, np.isnan(ans[0]),axis=1)
        
        # global mean without NAs
        sums, counts = [], []
        for i in range(len(self.As)):
            s, c = gwasprs.stats.nansum(self.As[i])
            sums.append(s)
            counts.append(c)
        GLOBAL_MEAN, GLOBAL_COUNT = gwasprs.stats.aggregate_sums(sums, counts)

        # global mean from imputed data
        sums, counts = [], []
        for i in range(len(self.As)):
            a, s, c = gwasprs.stats.impute_and_local_mean(self.As[i], GLOBAL_MEAN)
            sums.append(s)
            counts.append(c)
            self.As[i] = a
        GLOBAL_MEAN, GLOBAL_COUNT = gwasprs.stats.aggregate_sums(sums, counts)

        # global variance
        ssqs = []
        for i in range(len(self.As)):
            a, ssq = gwasprs.stats.local_ssq(self.As[i], GLOBAL_MEAN)
            ssqs.append(ssq)
            self.As[i] = a
        GLOBAL_VAR, DELETE = gwasprs.stats.aggregate_ssq(ssqs, GLOBAL_COUNT)
        
        result = np.concatenate([gwasprs.stats.standardize(A, GLOBAL_VAR, DELETE) for A in self.As], axis=0)
        np.testing.assert_array_almost_equal(ans, result, decimal=5)




