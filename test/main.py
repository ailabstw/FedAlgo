import unittest
from regression import LinearRegressionTestCase
from stats import CovarianceTestCase

if __name__ == '__main__':
    flr_suite = unittest.TestLoader().loadTestsFromTestCase(LinearRegressionTestCase)
    cov_suite = unittest.TestLoader().loadTestsFromTestCase(CovarianceTestCase)
    alltests = unittest.TestSuite([flr_suite, cov_suite])
    result = unittest.TextTestRunner().run(alltests)

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
