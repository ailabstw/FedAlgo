import unittest
from test.test_linalg import MatmulTestCase
from test.test_regression import LinearRegressionTestCase, LogisticRegressionTestCase
from test.test_stats import CovarianceTestCase

if __name__ == '__main__':
    matmul_suite = unittest.TestLoader().loadTestsFromTestCase(MatmulTestCase)
    linear_suite = unittest.TestLoader().loadTestsFromTestCase(LinearRegressionTestCase)
    logist_suite = unittest.TestLoader().loadTestsFromTestCase(LogisticRegressionTestCase)
    cov_suite = unittest.TestLoader().loadTestsFromTestCase(CovarianceTestCase)
    alltests = unittest.TestSuite([matmul_suite, linear_suite, logist_suite, cov_suite])
    result = unittest.TextTestRunner().run(alltests)

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
