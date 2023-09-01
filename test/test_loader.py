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

class IteratorTestCase(unittest.TestCase):
    
    def setUp(self):
        self.bed = gwasprs.loader.read_bed(bfile_path)
        self.n_SNP = self.bed.sid_count
        self.n_sample = self.bed.iid_count
        self.snp_chunk_size = 15
        self.sample_chunk_size = 11

    def tearDown(self):
        self.bed = None

    def test_SNPIterator(self):
        ans = self.bed.read()

        iterator = gwasprs.loader.SNPIterator(self.n_SNP, chunk_size=self.snp_chunk_size)
        total_blocks = [self.bed.read(index=idx) for idx in iterator]
        result = np.concatenate(total_blocks, axis=1)

        np.testing.assert_allclose(ans, result, equal_nan=True)
    
    def test_SNPIterator_samples(self):
        ans = self.bed.read()

        iterator = gwasprs.loader.SNPIterator(self.n_SNP, chunk_size=self.snp_chunk_size).samples(self.n_sample, chunk_size=self.sample_chunk_size)
        total_blocks = [self.bed.read(index=idx) for idx in iterator]
        n_blocks = (self.n_sample//self.sample_chunk_size)+1
        snp_concat = [np.concatenate(total_blocks[i:i+n_blocks], axis=0) for i in range(0, len(total_blocks), n_blocks)]
        result = np.concatenate(snp_concat, axis=1)

        np.testing.assert_allclose(ans, result, equal_nan=True)

    def test_SampleIterator(self):
        ans = self.bed.read()

        iterator = gwasprs.loader.SampleIterator(self.n_sample, chunk_size=self.sample_chunk_size)
        total_blocks = [self.bed.read(index=idx) for idx in iterator]
        result = np.concatenate(total_blocks, axis=0)

        np.testing.assert_allclose(ans, result, equal_nan=True)

    def test_SampleIterator_snps(self):
        ans = self.bed.read()

        iterator = gwasprs.loader.SampleIterator(self.n_sample, chunk_size=self.sample_chunk_size).snps(self.n_SNP, chunk_size=self.snp_chunk_size)
        total_blocks = [self.bed.read(index=idx) for idx in iterator]
        n_blocks = (self.n_SNP//self.snp_chunk_size)+1
        sample_concat = [np.concatenate(total_blocks[i:i+n_blocks], axis=1) for i in range(0, len(total_blocks), n_blocks)]
        result = np.concatenate(sample_concat, axis=0)

        np.testing.assert_allclose(ans, result, equal_nan=True)


class BedIteratorTestCase(unittest.TestCase):

    def setUp(self):
        self.bed = gwasprs.loader.read_bed(bfile_path)
        self.n_SNP = self.bed.sid_count
        self.n_sample = self.bed.iid_count
        self.snp_chunk_size = 15
        self.sample_chunk_size = 11
        self.get_chunk_sample_list = [12, 10, 8, 20, 11] # 61
        self.get_chunk_snp_list = [23, 17, 2, 11, 14, 34] # 101

    def tearDown(self):
        self.bed = None

    def test_only_sample_get_chunk(self):
        ans = self.bed.read()
        
        iterator = gwasprs.loader.SampleIterator(self.n_sample, self.sample_chunk_size)
        bed_iterator = gwasprs.loader.BedIterator(bfile_path, iterator)
        total_blocks = [bed_iterator.get_chunk(chunk_size) for chunk_size in self.get_chunk_sample_list]
        result = np.concatenate(total_blocks, axis=0)

        np.testing.assert_allclose(ans, result, equal_nan=True)

    def test_only_snp_get_chunk(self):
        ans = self.bed.read()

        iterator = gwasprs.loader.SNPIterator(self.n_SNP, self.snp_chunk_size)
        bed_iterator = gwasprs.loader.BedIterator(bfile_path, iterator)
        total_blocks = [bed_iterator.get_chunk(chunk_size) for chunk_size in self.get_chunk_snp_list]
        result = np.concatenate(total_blocks, axis=1)

        np.testing.assert_allclose(ans, result, equal_nan=True)

    def test_sample_snp_get_chunk(self):
        ans = self.bed.read()

        iterator = gwasprs.loader.SampleIterator(self.n_sample, self.sample_chunk_size).snps(self.n_SNP, self.snp_chunk_size)
        bed_iterator = gwasprs.loader.BedIterator(bfile_path, iterator)
        snp_blocks = (self.n_SNP//self.snp_chunk_size)+1
        total_blocks = [bed_iterator.get_chunk(chunk_size, reset=False) for chunk_size in self.get_chunk_sample_list for _ in range(snp_blocks)]
        snp_concat = [np.concatenate(total_blocks[i:i+snp_blocks], axis=1) for i in range(0, len(total_blocks), snp_blocks)]
        result = np.concatenate(snp_concat, axis=0)

        np.testing.assert_allclose(ans, result, equal_nan=True)

    def test_snp_sample_get_chunk(self):
        ans = self.bed.read()

        iterator = gwasprs.loader.SNPIterator(self.n_SNP, self.snp_chunk_size).samples(self.n_sample, self.sample_chunk_size)
        bed_iterator = gwasprs.loader.BedIterator(bfile_path, iterator)
        sample_blocks = (self.n_sample//self.sample_chunk_size)+1
        total_blocks = [bed_iterator.get_chunk(chunk_size, reset=False) for chunk_size in self.get_chunk_snp_list for _ in range(sample_blocks)]
        sample_concat = [np.concatenate(total_blocks[i:i+sample_blocks], axis=0) for i in range(0, len(total_blocks), sample_blocks)]
        result = np.concatenate(sample_concat, axis=1)

        np.testing.assert_allclose(ans, result, equal_nan=True)

    def test_only_sample_next(self):
        ans = self.bed.read()
        
        iterator = gwasprs.loader.SampleIterator(self.n_sample, self.sample_chunk_size)
        bed_iterator = gwasprs.loader.BedIterator(bfile_path, iterator)
        total_blocks = [bed for bed in bed_iterator]
        result = np.concatenate(total_blocks, axis=0)

        np.testing.assert_allclose(ans, result, equal_nan=True)

    def test_only_snp_next(self):
        ans = self.bed.read()

        iterator = gwasprs.loader.SNPIterator(self.n_SNP, self.snp_chunk_size)
        bed_iterator = gwasprs.loader.BedIterator(bfile_path, iterator)
        total_blocks = [bed for bed in bed_iterator]
        result = np.concatenate(total_blocks, axis=1)

        np.testing.assert_allclose(ans, result, equal_nan=True)

    def test_sample_snp_next(self):
        ans = self.bed.read()

        iterator = gwasprs.loader.SampleIterator(self.n_sample, self.sample_chunk_size).snps(self.n_SNP, self.snp_chunk_size)
        bed_iterator = gwasprs.loader.BedIterator(bfile_path, iterator)
        total_blocks = [bed for bed in bed_iterator]
        snp_blocks = (self.n_SNP//self.snp_chunk_size)+1
        snp_concat = [np.concatenate(total_blocks[i:i+snp_blocks], axis=1) for i in range(0, len(total_blocks), snp_blocks)]
        result = np.concatenate(snp_concat, axis=0)

        np.testing.assert_allclose(ans, result, equal_nan=True)

    def test_snp_sample_next(self):
        ans = self.bed.read()

        iterator = gwasprs.loader.SNPIterator(self.n_SNP, self.snp_chunk_size).samples(self.n_sample, self.sample_chunk_size)
        bed_iterator = gwasprs.loader.BedIterator(bfile_path, iterator)
        total_blocks = [bed for bed in bed_iterator]
        sample_blocks = (self.n_sample//self.sample_chunk_size)+1
        sample_concat = [np.concatenate(total_blocks[i:i+sample_blocks], axis=0) for i in range(0, len(total_blocks), sample_blocks)]
        result = np.concatenate(sample_concat, axis=1)

        np.testing.assert_allclose(ans, result, equal_nan=True)


class FamIteratorTestCase(unittest.TestCase):
    
    def setUp(self):
        bed = gwasprs.loader.read_bed(bfile_path)
        self.n_SNP = bed.sid_count
        self.n_sample = bed.iid_count
        self.snp_chunk_size = 15
        self.sample_chunk_size = 11
        self.get_chunk_sample_list = [12, 10, 8, 20, 11] # 61
        self.get_chunk_snp_list = [23, 17, 2, 11, 14, 34] # 101

        fam = gwasprs.loader.read_fam(bfile_path)
        self.fam = gwasprs.loader.format_fam(fam, pheno_path, 'pheno')
        cov = gwasprs.loader.read_cov(cov_path)
        self.cov = gwasprs.loader.format_cov(cov, self.fam)
    
    def tearDown(self):
        self.fam = None
        self.cov = None
    
    def test_only_sample_get_chunk(self):
        iterator = gwasprs.loader.SampleIterator(self.n_sample, self.sample_chunk_size)
        fam_iterator = gwasprs.loader.FamIterator(bfile_path, None, pheno_path, 'pheno', iterator)
        total_blocks = [fam_iterator.get_chunk(chunk_size) for chunk_size in self.get_chunk_sample_list]
        result = pd.concat(total_blocks, axis=0)
    
        pd.testing.assert_frame_equal(self.fam, result)

    def test_sample_snp_get_chunk(self):
        iterator = gwasprs.loader.SampleIterator(self.n_sample, self.sample_chunk_size).snps(self.n_SNP, self.snp_chunk_size)
        fam_iterator = gwasprs.loader.FamIterator(bfile_path, None, pheno_path, 'pheno', iterator)
        total_blocks = [fam_iterator.get_chunk(chunk_size) for chunk_size in self.get_chunk_sample_list]
        result = pd.concat(total_blocks, axis=0)

        pd.testing.assert_frame_equal(self.fam, result)

    def test_only_sample_next(self):
        iterator = gwasprs.loader.SampleIterator(self.n_sample, self.sample_chunk_size)
        fam_iterator = gwasprs.loader.FamIterator(bfile_path, None, pheno_path, 'pheno', iterator)
        total_blocks = [fam for fam in fam_iterator]
        result = pd.concat(total_blocks, axis=0)

        pd.testing.assert_frame_equal(self.fam, result)

    def test_sample_snp_next(self):
        iterator = gwasprs.loader.SampleIterator(self.n_sample, self.sample_chunk_size).snps(self.n_SNP, self.snp_chunk_size)
        fam_iterator = gwasprs.loader.FamIterator(bfile_path, None, pheno_path, 'pheno', iterator)
        total_blocks = [fam for fam in fam_iterator]
        result = pd.concat(total_blocks, axis=0)

        pd.testing.assert_frame_equal(self.fam, result)

    def test_only_sample_with_cov_get_chunk(self):
        iterator = gwasprs.loader.SampleIterator(self.n_sample, self.sample_chunk_size)
        fam_iterator = gwasprs.loader.FamIterator(bfile_path, cov_path, pheno_path, 'pheno', iterator)
        total_blocks = [fam_iterator.get_chunk(chunk_size) for chunk_size in self.get_chunk_sample_list]
        result = pd.concat(total_blocks, axis=0)

        pd.testing.assert_frame_equal(self.fam, result)

    def test_sample_snp_with_cov_get_chunk(self):
        iterator = gwasprs.loader.SampleIterator(self.n_sample, self.sample_chunk_size).snps(self.n_SNP, self.snp_chunk_size)
        fam_iterator = gwasprs.loader.FamIterator(bfile_path, cov_path, pheno_path, 'pheno', iterator)
        total_blocks = [fam_iterator.get_chunk(chunk_size) for chunk_size in self.get_chunk_sample_list]
        result = pd.concat(total_blocks, axis=0)

        pd.testing.assert_frame_equal(self.fam, result)

    def test_only_sample_with_cov_next(self):
        iterator = gwasprs.loader.SampleIterator(self.n_sample, self.sample_chunk_size)
        fam_iterator = gwasprs.loader.FamIterator(bfile_path, cov_path, pheno_path, 'pheno', iterator)
        total_blocks = [fam for fam in fam_iterator]
        result = pd.concat(total_blocks, axis=0)

        pd.testing.assert_frame_equal(self.fam, result)

    def test_sample_snp_with_cov_next(self):
        iterator = gwasprs.loader.SampleIterator(self.n_sample, self.sample_chunk_size).snps(self.n_SNP, self.snp_chunk_size)
        fam_iterator = gwasprs.loader.FamIterator(bfile_path, cov_path, pheno_path, 'pheno', iterator)
        total_blocks = [fam for fam in fam_iterator]
        result = pd.concat(total_blocks, axis=0)

        pd.testing.assert_frame_equal(self.fam, result)



class CovIteratorTestCase(unittest.TestCase):
    
    def setUp(self):
        bed = gwasprs.loader.read_bed(bfile_path)
        self.n_SNP = bed.sid_count
        self.n_sample = bed.iid_count
        self.snp_chunk_size = 15
        self.sample_chunk_size = 11
        self.get_chunk_sample_list = [12, 10, 8, 20, 11] # 61
        self.get_chunk_snp_list = [23, 17, 2, 11, 14, 34] # 101
        
        fam = gwasprs.loader.read_fam(bfile_path)
        self.fam = gwasprs.loader.format_fam(fam, pheno_path, 'pheno')
        cov = gwasprs.loader.read_cov(cov_path)
        self.cov = gwasprs.loader.format_cov(cov, self.fam)
    
    def tearDown(self):
        self.fam = None
        self.cov = None
    
    def test_only_sample_get_chunk(self):
        iterator = gwasprs.loader.SampleIterator(self.n_sample, self.sample_chunk_size)
        cov_iterator = gwasprs.loader.CovIterator(bfile_path, None, pheno_path, 'pheno', iterator)
        total_blocks = [cov_iterator.get_chunk(chunk_size) for chunk_size in self.get_chunk_sample_list]
        
        np.testing.assert_array_equal(np.array(total_blocks), None)

    def test_sample_snp_get_chunk(self):
        iterator = gwasprs.loader.SampleIterator(self.n_sample, self.sample_chunk_size).snps(self.n_SNP, self.snp_chunk_size)
        cov_iterator = gwasprs.loader.CovIterator(bfile_path, None, pheno_path, 'pheno', iterator)
        total_blocks = [cov_iterator.get_chunk(chunk_size) for chunk_size in self.get_chunk_sample_list]

        np.testing.assert_array_equal(np.array(total_blocks), None)

    def test_only_sample_next(self):
        iterator = gwasprs.loader.SampleIterator(self.n_sample, self.sample_chunk_size)
        cov_iterator = gwasprs.loader.CovIterator(bfile_path, None, pheno_path, 'pheno', iterator)
        total_blocks = [cov for cov in cov_iterator]

        np.testing.assert_array_equal(np.array(total_blocks), None)

    def test_sample_snp_next(self):
        iterator = gwasprs.loader.SampleIterator(self.n_sample, self.sample_chunk_size).snps(self.n_SNP, self.snp_chunk_size)
        cov_iterator = gwasprs.loader.CovIterator(bfile_path, None, pheno_path, 'pheno', iterator)
        total_blocks = [cov for cov in cov_iterator]
        
        np.testing.assert_array_equal(np.array(total_blocks), None)

    def test_only_sample_with_cov_get_chunk(self):
        iterator = gwasprs.loader.SampleIterator(self.n_sample, self.sample_chunk_size)
        cov_iterator = gwasprs.loader.CovIterator(bfile_path, cov_path, pheno_path, 'pheno', iterator)
        total_blocks = [cov_iterator.get_chunk(chunk_size) for chunk_size in self.get_chunk_sample_list]
        result = pd.concat(total_blocks, axis=0)

        pd.testing.assert_frame_equal(self.cov, result)

    def test_sample_snp_with_cov_get_chunk(self):
        iterator = gwasprs.loader.SampleIterator(self.n_sample, self.sample_chunk_size).snps(self.n_SNP, self.snp_chunk_size)
        cov_iterator = gwasprs.loader.CovIterator(bfile_path, cov_path, pheno_path, 'pheno', iterator)
        total_blocks = [cov_iterator.get_chunk(chunk_size) for chunk_size in self.get_chunk_sample_list]
        result = pd.concat(total_blocks, axis=0)

        pd.testing.assert_frame_equal(self.cov, result)

    def test_only_sample_with_cov_next(self):
        iterator = gwasprs.loader.SampleIterator(self.n_sample, self.sample_chunk_size)
        cov_iterator = gwasprs.loader.CovIterator(bfile_path, cov_path, pheno_path, 'pheno', iterator)
        total_blocks = [cov for cov in cov_iterator]
        result = pd.concat(total_blocks, axis=0)

        pd.testing.assert_frame_equal(self.cov, result)

    def test_sample_snp_with_cov_next(self):
        iterator = gwasprs.loader.SampleIterator(self.n_sample, self.sample_chunk_size).snps(self.n_SNP, self.snp_chunk_size)
        cov_iterator = gwasprs.loader.CovIterator(bfile_path, cov_path, pheno_path, 'pheno', iterator)
        total_blocks = [cov for cov in cov_iterator]
        result = pd.concat(total_blocks, axis=0)

        pd.testing.assert_frame_equal(self.cov, result)


class BimIteratorTestCase(unittest.TestCase):

    def setUp(self):
        bed = gwasprs.loader.read_bed(bfile_path)
        self.n_SNP = bed.sid_count
        self.n_sample = bed.iid_count
        self.snp_chunk_size = 15
        self.sample_chunk_size = 11
        self.get_chunk_sample_list = [12, 10, 8, 20, 11] # 61
        self.get_chunk_snp_list = [23, 17, 2, 11, 14, 34] # 101
        
        self.bim = gwasprs.loader.read_bim(bfile_path)
    
    def tearDown(self):
        self.bim = None

    def test_only_snp_get_chunk(self):
        iterator = gwasprs.loader.SNPIterator(self.n_SNP, self.snp_chunk_size)
        bim_iterator = gwasprs.loader.BimIterator(bfile_path, iterator)
        total_blocks = [bim_iterator.get_chunk(chunk_size) for chunk_size in self.get_chunk_snp_list]
        result = pd.concat(total_blocks, axis=0)
        pd.testing.assert_frame_equal(self.bim, result)

    def test_snp_sample_get_chunk(self):
        iterator = gwasprs.loader.SNPIterator(self.n_SNP, self.snp_chunk_size).samples(self.n_sample, self.sample_chunk_size)
        bim_iterator = gwasprs.loader.BimIterator(bfile_path, iterator)
        total_blocks = [bim_iterator.get_chunk(chunk_size) for chunk_size in self.get_chunk_snp_list]
        result = pd.concat(total_blocks, axis=0)

        pd.testing.assert_frame_equal(self.bim, result)

    def test_only_snp_next(self):
        iterator = gwasprs.loader.SNPIterator(self.n_SNP, self.snp_chunk_size)
        bim_iterator = gwasprs.loader.BimIterator(bfile_path, iterator)
        total_blocks = [bim for bim in bim_iterator]
        result = pd.concat(total_blocks, axis=0)

        pd.testing.assert_frame_equal(self.bim, result)

    def test_snp_sample_next(self):
        iterator = gwasprs.loader.SNPIterator(self.n_SNP, self.snp_chunk_size).samples(self.n_sample, self.sample_chunk_size)
        bim_iterator = gwasprs.loader.BimIterator(bfile_path, iterator)
        total_blocks = [bim for bim in bim_iterator]
        result = pd.concat(total_blocks, axis=0)

        pd.testing.assert_frame_equal(self.bim, result)


class GWASDataIteratorTestCase(unittest.TestCase):

    def setUp(self):
        self.bed = gwasprs.loader.read_bed(bfile_path)
        self.n_SNP = self.bed.sid_count
        self.n_sample = self.bed.iid_count
        self.snp_chunk_size = 15
        self.sample_chunk_size = 11
        self.get_chunk_sample_list = [12, 10, 8, 20, 11] # 61
        self.get_chunk_snp_list = [23, 17, 2, 11, 14, 34] # 101

        fam = gwasprs.loader.read_fam(bfile_path)
        self.fam = gwasprs.loader.format_fam(fam, pheno_path, 'pheno')
        cov = gwasprs.loader.read_cov(cov_path)
        self.cov = gwasprs.loader.format_cov(cov, self.fam)
        self.bim = gwasprs.loader.read_bim(bfile_path)

    def tearDown(self):
        self.bed = None
        self.fam = None
        self.cov = None
        self.bim = None
    
    def test_only_sample_get_chunk(self):
        iterator = gwasprs.loader.SampleIterator(self.n_sample, self.sample_chunk_size)
        GWASDataIterator = gwasprs.loader.GWASDataIterator.SampleWise(bfile_path, iterator, cov_path, pheno_path, 'pheno')

        for idx in iterator:
            bed = self.bed.read(index=idx)
            bim = self.bim
            fam = self.fam.loc[idx[0]]
            cov = self.cov.loc[idx[0]]
            ans = gwasprs.loader.GWASData(bed, fam, bim, cov)

            result = GWASDataIterator.get_chunk(len(idx[0]))

            gwasprs.loader.assert_GWASData_is_equal(ans, result)
            
    def test_only_snp_get_chunk(self):
        iterator = gwasprs.loader.SNPIterator(self.n_SNP, self.snp_chunk_size)
        GWASDataIterator = gwasprs.loader.GWASDataIterator.SNPWise(bfile_path, iterator, cov_path, pheno_path, 'pheno')
        
        for idx in iterator:
            bed = self.bed.read(index=idx)
            bim = self.bim.loc[idx[1]]
            fam = self.fam
            cov = self.cov
            ans = gwasprs.loader.GWASData(bed, fam, bim, cov)

            result = GWASDataIterator.get_chunk(len(idx[1]))

            gwasprs.loader.assert_GWASData_is_equal(ans, result)

    def test_sample_snp_get_chunk(self):
        iterator = gwasprs.loader.SampleIterator(self.n_sample, self.sample_chunk_size).snps(self.n_SNP, self.snp_chunk_size)
        GWASDataIterator = gwasprs.loader.GWASDataIterator.SampleWise(bfile_path, iterator, cov_path, pheno_path, 'pheno')

        for idx in iterator:
            bed = self.bed.read(index=idx)
            bim = self.bim.loc[idx[1]]
            fam = self.fam.loc[idx[0]]
            cov = self.cov.loc[idx[0]]
            ans = gwasprs.loader.GWASData(bed, fam, bim, cov)

            result = GWASDataIterator.get_chunk(len(idx[0]))

            gwasprs.loader.assert_GWASData_is_equal(ans, result)

    def test_snp_sample_get_chunk(self):
        iterator = gwasprs.loader.SNPIterator(self.n_SNP, self.snp_chunk_size).samples(self.n_sample, self.sample_chunk_size)
        GWASDataIterator = gwasprs.loader.GWASDataIterator.SNPWise(bfile_path, iterator, cov_path, pheno_path, 'pheno')

        for idx in iterator:
            bed = self.bed.read(index=idx)
            bim = self.bim.loc[idx[1]]
            fam = self.fam.loc[idx[0]]
            cov = self.cov.loc[idx[0]]
            ans = gwasprs.loader.GWASData(bed, fam, bim, cov)

            result = GWASDataIterator.get_chunk(len(idx[1]))

            gwasprs.loader.assert_GWASData_is_equal(ans, result)

    def test_only_sample_next(self):
        ans = self.bed.read()
        
        iterator = gwasprs.loader.SampleIterator(self.n_sample, self.sample_chunk_size)
        pass

    def test_only_snp_next(self):
        ans = self.bed.read()

        iterator = gwasprs.loader.SNPIterator(self.n_SNP, self.snp_chunk_size)
        pass

    def test_sample_snp_next(self):
        ans = self.bed.read()

        iterator = gwasprs.loader.SampleIterator(self.n_sample, self.sample_chunk_size).snps(self.n_SNP, self.snp_chunk_size)
        pass

    def test_snp_sample_next(self):
        ans = self.bed.read()

        iterator = gwasprs.loader.SNPIterator(self.n_SNP, self.snp_chunk_size).samples(self.n_sample, self.sample_chunk_size)
        pass

    
