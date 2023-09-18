import unittest
import os

from bed_reader import open_bed
import pandas as pd
import numpy as np
import jax.numpy as jnp

import gwasprs 


# The plink2 result was saved in plink_out_path
bfile_source_path = "../data/test_bfile"
plink_out_path = "/tmp"
bfile = "hapmap1_100"


def _cal_plink_eigenvec(bed, delete):
    # the standardization removes some SNPs (var=0)
    snps = np.delete(bed.sid, delete)
    pd.DataFrame({'ID':snps}).to_csv(os.path.join(plink_out_path, 'snp_list'), index=None)

    pca_prefix = os.path.join(plink_out_path, bfile)

    # Make bfile and perform pca
    os.system(f"plink2 --extract {os.path.join(plink_out_path, 'snp_list')} \
                       --bed {os.path.join(bfile_source_path, f'{bfile}.bed')} \
                       --bim {os.path.join(bfile_source_path, f'{bfile}.bim')} \
                       --fam {os.path.join(bfile_source_path, f'{bfile}.fam')} \
                       --allow-extra-chr \
                       --chr 1-22 \
                       --make-bed \
                       --out {pca_prefix}")

    os.system(f'plink2 --bfile {pca_prefix} \
                       --pca 20 allele-wts \
                       --out {pca_prefix}')

    G = pd.read_csv(pca_prefix+".eigenvec", sep='\t').iloc[:,2:]

    H = pd.read_csv(pca_prefix+".eigenvec.allele", sep='\t')
    H = H.drop_duplicates('ID').iloc[:,5:]
    
    return np.array(G), np.array(H)

def _cal_np_eigenvec(std_A):
    G, S, Ht = np.linalg.svd(std_A, full_matrices=False)
    return G[:,:20], Ht.T[:,:20]

def _result_report(func_name, mtx1, mtx2, mtx_name):
    func_name = ' '.join(func_name.split(sep='_')[1:])
    print(f'\n{"="*30} {func_name} {"="*30}')
    try:
        gwasprs.linalg.eigenvec_concordance_estimation(mtx1, mtx2)
    except AssertionError as error:
            print(f'plink could give unprecised result, so the Assertionerror should be read manually.\n{error}')

def _standardize(A):
    # Impute NAs with mean
    col_mean = np.nanmean(A, axis=0)
    inds = np.where(np.isnan(A))
    A[inds] = np.take(col_mean, inds[1])
    
    # Standardize and remove SNPs with var=0
    A = (A-np.mean(A,axis=0))/np.nanstd(A,axis=0,ddof=1)
    delete = np.isnan(A[0])
    A = np.delete(A, delete, axis=1)
    return A, delete

class PcaUsecase(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        BED = open_bed(os.path.join(bfile_source_path, f"{bfile}.bed"))
        A = BED.read()
        self.std_A_ans, delete = _standardize(A)

        # plink2
        self.plink_G, self.plink_H = _cal_plink_eigenvec(BED, delete)
        
        # np.linalg.svd
        self.numpy_G, self.numpy_H = _cal_np_eigenvec(self.std_A_ans)
        
        # federated-svd
        params = {
            'As':[jnp.array(A)],
            'k1':20,
            'epsilon':1e-9,
            'max_iterations':20,
            'k2':20,
        }
        self.usecase_G, self.usecase_H = gwasprs.linalg.FederatedSVD.standalone(**params)
        # self.usecase_G, self.usecase_H = gwasprs.linalg.federated_svd(**params)

        self.usecase_G = self.usecase_G[0]

    @classmethod
    def tearDownClass(self):
        self.usecase = None

    """
    !!!!!!!!!!!!!!!!!!!!!!!!!! The plink performance is worse !!!!!!!!!!!!!!!!!!!!!!!!!!
    The result was proved by performing np.linalg.svd, pca_usecase and plink2 on standardized genotype matrix.
    Taking the directly SVD algorithm as the real measures (np.linalg.svd).

    Under tolerated decimal = 1e-4:
    %diagonal %G matched elements/400 % H matched elements/400
    The inner products: np.linalg.svd vs plink2      ~0.9x / 0.005 / 0.0
                        np.linalg.svd vs pca_usecase 1.0   / 1.0   / 1.0
                        pca_usecase vs plink2        ~0.9x / 0.005 / 0.005
    """
    
    def test_G_MTX_plink_vs_usecase(self):
        _result_report(self.test_G_MTX_plink_vs_usecase.__name__, self.plink_G, self.usecase_G, 'G')
    
    def test_H_MTX_plink_vs_usecase(self):
        _result_report(self.test_H_MTX_plink_vs_usecase.__name__, self.plink_H, self.usecase_H, 'H')

    def test_G_MTX_numpy_vs_plink(self):
        _result_report(self.test_G_MTX_numpy_vs_plink.__name__, self.plink_G, self.numpy_G, 'G')
    
    def test_H_MTX_numpy_vs_plink(self):
        _result_report(self.test_H_MTX_numpy_vs_plink.__name__, self.plink_H, self.numpy_H, 'H')

    def test_G_MTX_numpy_vs_usecase(self):
        _result_report(self.test_G_MTX_numpy_vs_usecase.__name__, self.numpy_G, self.usecase_G, 'G')

    def test_H_MTX_numpy_vs_usecase(self):
        _result_report(self.test_H_MTX_numpy_vs_usecase.__name__, self.numpy_H, self.usecase_H, 'H')