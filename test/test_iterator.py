import unittest
import numpy as np

import gwasprs


class IteratorTestCase(unittest.TestCase):

    def setUp(self):
        self.n_SNP = 56
        self.n_sample = 79
        self.snp_step = 15
        self.sample_step = 20

    def tearDown(self):
        pass

    def test_IndexIterator(self):
        iter = gwasprs.iterator.IndexIterator(self.n_SNP, step=self.snp_step)
        self.assertEqual(slice(0, self.snp_step, None), next(iter))
        self.assertEqual(slice(self.snp_step, 2*self.snp_step, None), next(iter))
        self.assertFalse(iter.is_end())

    def test_SNPIterator(self):
        iter = gwasprs.iterator.SNPIterator(self.n_SNP, step=self.snp_step)
        self.assertEqual(np.s_[0:self.snp_step, :], next(iter))
        self.assertEqual(np.s_[self.snp_step:2*self.snp_step, :], next(iter))

    def test_SNPIterator_samples(self):
        iter = gwasprs.iterator.SNPIterator(self.n_SNP, step=self.snp_step).samples(self.n_sample, step=self.sample_step)
        self.assertEqual(np.s_[0:self.snp_step, 0:self.sample_step], next(iter))
        self.assertEqual(np.s_[0:self.snp_step, self.sample_step:2*self.sample_step], next(iter))

    def test_SampleIterator(self):
        iter = gwasprs.iterator.SampleIterator(self.n_sample, step=self.sample_step)
        self.assertEqual(np.s_[:, 0:self.sample_step], next(iter))
        self.assertEqual(np.s_[:, self.sample_step:2*self.sample_step], next(iter))

    def test_SampleIterator_snps(self):
        iter = gwasprs.iterator.SampleIterator(self.n_sample, step=self.sample_step).snps(self.n_SNP, step=self.snp_step)
        self.assertEqual(np.s_[0:self.snp_step, 0:self.sample_step], next(iter))
        self.assertEqual(np.s_[self.snp_step:2*self.snp_step, 0:self.sample_step], next(iter))

