
from federatedprs.io.loader import GwasDataLoader



def scan_bfile(  
    bed_path: str,
    pheno_path: str = None,
    pheno_name: str = 'PHENO1',
    cov_path: str = None,
    snp_list: str = None, 
    ind_list: str = None, 
    mean_fill_na_flag: bool = False,
    read_all_gt_flag: bool = False,
    rename_snp_flag: bool = True,
):
    bed_loader = GwasDataLoader(bed_path, pheno_path, pheno_name, 
            cov_path, snp_list, ind_list, 
            mean_fill_na_flag, read_all_gt_flag, rename_snp_flag) 
    
    bed_loader

