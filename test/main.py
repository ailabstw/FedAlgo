import unittest
from test.regression import LinearRegressionTestCase, LogisticRegressionTestCase
from test.stats import CovarianceTestCase

if __name__ == '__main__':
    linear_suite = unittest.TestLoader().loadTestsFromTestCase(LinearRegressionTestCase)
    logist_suite = unittest.TestLoader().loadTestsFromTestCase(LogisticRegressionTestCase)
    cov_suite = unittest.TestLoader().loadTestsFromTestCase(CovarianceTestCase)
    alltests = unittest.TestSuite([linear_suite, logist_suite, cov_suite])
    result = unittest.TextTestRunner().run(alltests)

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
