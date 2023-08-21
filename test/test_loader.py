import unittest
import numpy as np
import pandas as pd
import gwasprs

bfile_path = './data/test_bfile/hapmap1_100'
cov_path = './data/test_bfile/hapmap1_100.cov'
pheno_path = './data/test_bfile/hapmap1_100.pheno'

class GWASData_Standard_TestCase(unittest.TestCase):

    def setUp(self):
        # General data
        self.bed = gwasprs.loader.read_bed(bfile_path)
        self.bim = gwasprs.loader.read_bim(bfile_path)
        fam = gwasprs.loader.read_fam(bfile_path)
        self.fam = gwasprs.loader.format_fam(fam, pheno_path, 'pheno')
        cov = gwasprs.loader.read_cov(cov_path)
        self.cov = gwasprs.loader.format_cov(cov, self.fam)

        # GWASData object
        self.gwasdata = gwasprs.loader.read_gwasdata(bfile_path, cov_path, pheno_path, 'pheno')

    def tearDown(self):
        self.bed = None
        self.bim = None
        self.fam = None
        self.cov = None
        self.gwasdata = None
    
    def test_data_formation(self):
        self.gwasdata.subset()
        pd.testing.assert_frame_equal(self.fam, self.gwasdata.fam)
        pd.testing.assert_frame_equal(self.bim, self.gwasdata.bim)
        pd.testing.assert_frame_equal(self.cov, self.gwasdata.covariate)
        np.testing.assert_allclose(self.bed.read(), self.gwasdata.GT, equal_nan=True)

    def test_drop_missing_samples(self):
        missing_idx = list(set(gwasprs.loader.get_mask_idx(self.fam)).union(gwasprs.loader.get_mask_idx(self.cov)))
        keep_idx = list(set(self.fam.index).difference(missing_idx))
        self.fam = self.fam.iloc[keep_idx,:].reset_index(drop=True)
        self.cov = self.cov.iloc[keep_idx,:].reset_index(drop=True)
        # self.GT = self.bed.read()[keep_idx,:]
        
        self.gwasdata.subset()
        self.gwasdata.drop_missing_samples()

        pd.testing.assert_frame_equal(self.fam, self.gwasdata.fam)
        pd.testing.assert_frame_equal(self.cov, self.gwasdata.covariate)
        # np.testing.assert_allclose(self.GT, self.gwasdata.GT, equal_nan=True)

    def test_standard(self):
        missing_idx = list(set(gwasprs.loader.get_mask_idx(self.fam)).union(gwasprs.loader.get_mask_idx(self.cov)))
        keep_idx = list(set(self.fam.index).difference(missing_idx))
        self.fam = self.fam.iloc[keep_idx,:].reset_index(drop=True)
        self.cov = self.cov.iloc[keep_idx,:].reset_index(drop=True)
        # self.GT = self.bed.read()[keep_idx,:]

        self.gwasdata.standard()

        pd.testing.assert_frame_equal(self.fam, self.gwasdata.fam)
        pd.testing.assert_frame_equal(self.cov, self.gwasdata.covariate)
        # np.testing.assert_allclose(self.GT, self.gwasdata.GT, equal_nan=True)

class GWASData_UnmatchedSamples_TestCase(unittest.TestCase):
    pass