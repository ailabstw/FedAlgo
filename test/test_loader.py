import sys, os
import unittest
import logging
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/..')

from gwasprs.loader import GwasDataLoader, GwasSnpIterator

logging.basicConfig(level=logging.DEBUG)


class BedReaderTestCase(unittest.TestCase):

    def setUp(self):
        self.bed_path = "/mnt/prsdata/prs-data/Test/Data/DEMO_REG/demo_hg38"

    def tearDown(self):
        self.bed_path = None

    def test_reader(self):

        gwas_data_loader = GwasDataLoader(self.bed_path)
        gwas_data_loader.read_in()
        BED = gwas_data_loader.get_geno()
        
        idx_list = np.s_[2:32,5:20]
        GT = BED.read(index=idx_list)
        logging.info( GT.shape)

    def test_iterator(self):

        gwas_data_loader = GwasDataLoader(self.bed_path)
        gwas_data_loader.read_in()
        gwas_snp_iter = GwasSnpIterator(gwas_data_loader, batch_size=32)

        GT, BIM = next(gwas_snp_iter)
        logging.info(f"IterGenoStep for {gwas_snp_iter.cc}/{len(gwas_snp_iter)}")


# cd /yilun/CODE/fed-algo
# python3 -m unittest test.test_io.BedReaderTestCase 

# nosetests /yilun/CODE/fed-algo/test/main.py


