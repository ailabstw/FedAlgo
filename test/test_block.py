import unittest
import gwasprs
import gwasprs.linalg as linalg
import numpy as np
# from scipy.stats import norm


class BlockDiagonalTestCase(unittest.TestCase):

    def setUp(self):
        self.dim = 5
        self.Xs = [
            np.random.rand(7, self.dim),
            np.random.rand(11, self.dim),
            np.random.rand(13, self.dim),
        ]
        self.X = gwasprs.block.BlockDiagonalMatrix(self.Xs)
        self.y = np.random.rand(3 * self.dim)

    def tearDown(self):
        self.dim = None
        self.X = None
        self.y = None

    def test_ndim(self):
        self.assertEqual(2, self.X.ndim)

    def test_blocks(self):
        self.assertEqual(3, self.X.nblocks)
        self.assertEqual(self.X.nblocks, len(self.X.blocks))

        for (i, blk) in enumerate(self.X.blocks):
            np.testing.assert_array_almost_equal(self.Xs[i], blk, decimal=5)

        ans = [X.shape for X in self.Xs]
        self.assertEqual(ans, self.X.blockshapes)

        i = 1
        self.assertEqual(self.Xs[i].shape, self.X.blockshape(i))

    def test_append(self):
        result = gwasprs.block.BlockDiagonalMatrix(self.Xs[0:2])
        result.append(self.Xs[2])

        for (i, blk) in enumerate(result):
            np.testing.assert_array_almost_equal(self.Xs[i], blk, decimal=5)

    def test_append_block_diag(self):
        result = gwasprs.block.BlockDiagonalMatrix(self.Xs)
        Y = gwasprs.block.BlockDiagonalMatrix([
            np.random.rand(7, self.dim),
            np.random.rand(11, self.dim),
            np.random.rand(13, self.dim),
        ])
        result.append(Y)

        for (i, blk) in enumerate(self.Xs):
            np.testing.assert_array_almost_equal(blk, result[i], decimal=5)
        for (i, blk) in enumerate(Y):
            np.testing.assert_array_almost_equal(blk, result[i + len(self.Xs)], decimal=5)

    def test_mvdot(self):
        Xs = [
            np.random.rand(self.dim, 7),
            np.random.rand(self.dim, 11),
            np.random.rand(self.dim, 13),
        ]
        result = linalg.mvdot(gwasprs.block.BlockDiagonalMatrix(Xs), self.y)
        ans = np.concatenate([
            Xs[0].T @ self.y[0:self.dim],
            Xs[1].T @ self.y[self.dim:2*self.dim],
            Xs[2].T @ self.y[2*self.dim:3*self.dim],
        ])
        np.testing.assert_array_almost_equal(ans, result, decimal=5)

    def test_mvmul(self):
        result = linalg.mvmul(self.X, self.y)
        ans = np.concatenate([
            self.Xs[0] @ self.y[0:self.dim],
            self.Xs[1] @ self.y[self.dim:2*self.dim],
            self.Xs[2] @ self.y[2*self.dim:3*self.dim],
        ])
        np.testing.assert_array_almost_equal(ans, result, decimal=5)

    def test_mmdot(self):
        result = linalg.mmdot(self.X, self.X)
        ans = [X.T @ X for X in self.Xs]

        for i in range(result.nblocks):
            np.testing.assert_array_almost_equal(ans[i], result[i], decimal=5)

    def test_matmul(self):
        Ys = [
            np.random.rand(self.dim, 7),
            np.random.rand(self.dim, 11),
            np.random.rand(self.dim, 13),
        ]
        result = linalg.matmul(self.X, gwasprs.block.BlockDiagonalMatrix(Ys))
        ans = [X @ Y for (X, Y) in zip(self.Xs, Ys)]

        for i in range(result.nblocks):
            np.testing.assert_array_almost_equal(ans[i], result[i], decimal=5)

    def test_inv(self):
        cov = linalg.mmdot(self.X, self.X)
        result = linalg.inv(cov)

        for (i, blk) in enumerate(result):
            np.testing.assert_array_almost_equal(np.linalg.inv(cov[i]), blk, decimal=5)
