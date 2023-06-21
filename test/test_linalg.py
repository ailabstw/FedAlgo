import unittest
import gwasprs
import numpy as np

class LinAlgTestCase(unittest.TestCase):

    def setUp(self):
        X = np.array(
            [[[1], [1], [1]],
             [[2], [2], [2]],
             [[3], [3], [3]],
             [[4], [4], [4]]]
        )
        Y = np.array(
            [[[1], [1], [1]],
             [[2], [2], [2]],
             [[3], [3], [3]]]
        )
        y = np.array(
            [[1],
             [2],
             [3]]
        )
        self.X = np.concatenate((X, X), axis=2)
        self.Y = np.concatenate((Y, Y), axis=2)
        self.y = np.concatenate((y, y), axis=1)

    def tearDown(self):
        self.X = None
        self.Y = None
        self.y = None

    def test_batched_matmul(self):
        result = gwasprs.linalg.batched_matmul(self.X, self.Y)
        ans = np.array(
            [[[ 6,  6],
              [ 6,  6],
              [ 6,  6]],
             [[12, 12],
              [12, 12],
              [12, 12]],
             [[18, 18],
              [18, 18],
              [18, 18]],
             [[24, 24],
              [24, 24],
              [24, 24]]]
        )
        np.testing.assert_array_equal(ans, result)

    def test_batched_mvdot(self):
        result = gwasprs.linalg.batched_mvdot(self.X, self.y)
        ans = np.array(
            [[ 6,  6],
             [12, 12],
             [18, 18],
             [24, 24]]
        )
        np.testing.assert_array_equal(ans, result)

    def test_batched_mmdot(self):
        result = gwasprs.linalg.batched_mmdot(self.Y, self.Y)
        ans = np.array(
            [[[14, 14],
              [14, 14],
              [14, 14]],
             [[14, 14],
              [14, 14],
              [14, 14]],
             [[14, 14],
              [14, 14],
              [14, 14]],]
        )
        np.testing.assert_array_equal(ans, result)
