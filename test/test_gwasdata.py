import unittest
import os
import numpy as np
import pandas as pd
import gwasprs

bfile_path = os.path.join(os.getcwd(), 'data/test_bfile/hapmap1_100')
cov_path = os.path.join(os.getcwd(), 'data/test_bfile/hapmap1_100.cov')
pheno_path = os.path.join(os.getcwd(), 'data/test_bfile/hapmap1_100.pheno')


class GWASData_Standard_TestCase(unittest.TestCase):

    def setUp(self):
        # General data
        self.bed = gwasprs.reader.BedReader(bfile_path).read()
        self.bim = gwasprs.reader.BimReader(bfile_path).read()
        pheno = gwasprs.reader.PhenotypeReader(pheno_path, 'pheno').read()
        fam = gwasprs.reader.FamReader(bfile_path).read()
        self.fam = gwasprs.gwasdata.format_fam(fam, pheno)
        cov = gwasprs.reader.CovReader(cov_path).read()
        self.cov = gwasprs.gwasdata.format_cov(cov, self.fam)

        # GWASData object
        self.gwasdata = gwasprs.gwasdata.GWASData.read(bfile_path, cov_path, pheno_path, 'pheno')

    def tearDown(self):
        self.bed = None
        self.bim = None
        self.fam = None
        self.cov = None
        self.gwasdata = None

    def test_data_formation(self):
        self.gwasdata.subset()
        pd.testing.assert_frame_equal(self.fam, self.gwasdata.phenotype)
        pd.testing.assert_frame_equal(self.bim, self.gwasdata.snp)
        pd.testing.assert_frame_equal(self.cov, self.gwasdata.covariate)
        np.testing.assert_allclose(self.bed, self.gwasdata.genotype, equal_nan=True)

    def test_drop_missing_samples(self):
        missing_idx = list(set(gwasprs.gwasdata.get_mask_idx(self.fam)).union(gwasprs.gwasdata.get_mask_idx(self.cov)))
        keep_idx = list(set(self.fam.index).difference(missing_idx))
        self.fam = self.fam.iloc[keep_idx,:].reset_index(drop=True)
        self.cov = self.cov.iloc[keep_idx,:].reset_index(drop=True)
        # self.GT = self.bed.read()[keep_idx,:]

        self.gwasdata.subset()
        self.gwasdata.drop_missing_samples()

        pd.testing.assert_frame_equal(self.fam, self.gwasdata.phenotype)
        pd.testing.assert_frame_equal(self.cov, self.gwasdata.covariate)
        # np.testing.assert_allclose(self.GT, self.gwasdata.GT, equal_nan=True)

    def test_standard(self):
        missing_idx = list(set(gwasprs.gwasdata.get_mask_idx(self.fam)).union(gwasprs.gwasdata.get_mask_idx(self.cov)))
        keep_idx = list(set(self.fam.index).difference(missing_idx))
        self.fam = self.fam.iloc[keep_idx,:].reset_index(drop=True)
        self.cov = self.cov.iloc[keep_idx,:].reset_index(drop=True)
        # self.GT = self.bed.read()[keep_idx,:]

        self.gwasdata.standard()

        pd.testing.assert_frame_equal(self.fam, self.gwasdata.phenotype)
        pd.testing.assert_frame_equal(self.cov, self.gwasdata.covariate)
        # np.testing.assert_allclose(self.GT, self.gwasdata.GT, equal_nan=True)


class GWASDataIteratorTestCase(unittest.TestCase):

    def setUp(self):
        self.bedreader = gwasprs.reader.BedReader(bfile_path)
        self.bed = self.bedreader.read()
        self.n_SNP = self.bedreader.n_snp
        self.n_sample = self.bedreader.n_sample
        self.snp_default_step = 15
        self.sample_default_step = 11
        self.sample_step_list = [12, 10, 8, 20, 11] # 61
        self.snp_step_list = [23, 17, 2, 11, 14, 34] # 101

        pheno = gwasprs.reader.PhenotypeReader(pheno_path, 'pheno').read()
        fam = gwasprs.reader.FamReader(bfile_path).read()
        self.fam = gwasprs.gwasdata.format_fam(fam, pheno)
        cov = gwasprs.reader.CovReader(cov_path).read()
        self.cov = gwasprs.gwasdata.format_cov(cov, self.fam)
        self.bim = gwasprs.reader.BimReader(bfile_path).read()

    def tearDown(self):
        self.bed = None
        self.fam = None
        self.cov = None
        self.bim = None

    def test_sample_iterator(self):
        idx_iter = gwasprs.gwasdata.SampleIterator(self.n_sample, self.sample_default_step)
        dataiter = gwasprs.gwasdata.GWASDataIterator(bfile_path, None,
                                                    style="sample",
                                                    sample_step=self.sample_default_step)

        for idx in idx_iter:
            bed = self.bed[idx]
            bim = self.bim
            fam = self.fam.loc[idx[0]]
            cov = self.cov.loc[idx[0]]
            ans = gwasprs.gwasdata.GWASData(bed, fam, bim, cov)

            result = next(dataiter)

            self.assertEqual(ans, result)

    def test_snp_iterator(self):
        idx_iter = gwasprs.gwasdata.SNPIterator(self.n_SNP, self.snp_default_step)
        dataiter = gwasprs.gwasdata.GWASDataIterator(bfile_path, None,
                                                    style="snp",
                                                    snp_step=self.snp_default_step)

        for idx in idx_iter:
            bed = self.bed[idx]
            bim = self.bim.loc[idx[1]]
            fam = self.fam
            cov = self.cov
            ans = gwasprs.gwasdata.GWASData(bed, fam, bim, cov)

            result = next(dataiter)

            self.assertEqual(ans, result)

    # def test_sample_snp_get_chunk(self):
    #     iterator = gwasprs.gwasdata.SampleIterator(self.n_sample, self.sample_default_step).snps(self.n_SNP, self.snp_default_step)
    #     GWASDataIterator = gwasprs.gwasdata.GWASDataIterator.SampleWise(bfile_path, iterator, cov_path, pheno_path, 'pheno')

    #     for idx in iterator:
    #         bed = self.bed.read(index=idx)
    #         bim = self.bim.loc[idx[1]]
    #         fam = self.fam.loc[idx[0]]
    #         cov = self.cov.loc[idx[0]]
    #         ans = gwasprs.gwasdata.GWASData(bed, fam, bim, cov)

    #         result = GWASDataIterator.get_chunk(len(idx[0]))

    #         self.assertEqual(ans, result)

    # def test_snp_sample_get_chunk(self):
    #     iterator = gwasprs.gwasdata.SNPIterator(self.n_SNP, self.snp_default_step).samples(self.n_sample, self.sample_default_step)
    #     GWASDataIterator = gwasprs.gwasdata.GWASDataIterator.SNPWise(bfile_path, iterator, cov_path, pheno_path, 'pheno')

    #     for idx in iterator:
    #         bed = self.bed.read(index=idx)
    #         bim = self.bim.loc[idx[1]]
    #         fam = self.fam.loc[idx[0]]
    #         cov = self.cov.loc[idx[0]]
    #         ans = gwasprs.gwasdata.GWASData(bed, fam, bim, cov)

    #         result = GWASDataIterator.get_chunk(len(idx[1]))

    #         self.assertEqual(ans, result)

    # def test_sample_iterator_with_cov(self):
    #     iterator = gwasprs.gwasdata.SampleIterator(self.n_sample, self.sample_default_step)
    #     GWASDataIterator = gwasprs.gwasdata.GWASDataIterator(bfile_path, cov_path, pheno_path, 'pheno',
    #                                                        style="sample",
    #                                                        sample_step=self.sample_default_step,
    #                                                        snp_step=self.snp_default_step)

    #     pd.testing.assert_frame_equal(self.cov, result)

    # def test_sample_snp_with_cov_get_chunk(self):
    #     iterator = gwasprs.gwasdata.SampleIterator(self.n_sample, self.sample_chunk_size).snps(self.n_SNP, self.snp_chunk_size)
    #     cov_iterator = gwasprs.gwasdata.CovIterator(bfile_path, cov_path, pheno_path, 'pheno', iterator)
    #     total_blocks = [cov_iterator.get_chunk(chunk_size) for chunk_size in self.sample_step_list]
    #     result = pd.concat(total_blocks, axis=0)

    #     pd.testing.assert_frame_equal(self.cov, result)

    # def test_only_sample_with_cov_next(self):
    #     iterator = gwasprs.gwasdata.SampleIterator(self.n_sample, self.sample_chunk_size)
    #     cov_iterator = gwasprs.gwasdata.CovIterator(bfile_path, cov_path, pheno_path, 'pheno', iterator)
    #     total_blocks = [cov for cov in cov_iterator]
    #     result = pd.concat(total_blocks, axis=0)

    #     pd.testing.assert_frame_equal(self.cov, result)

    # def test_sample_snp_with_cov_next(self):


    #     ans = self.bed.read()

    #     iterator = gwasprs.gwasdata.SNPIterator(self.n_SNP, self.snp_default_step).samples(self.n_sample, self.sample_default_step)
    #     pass
