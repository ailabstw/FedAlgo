from typing import List, Set
import pandas as pd
import os
import logging
import numpy as np
from bed_reader import open_bed

from .utils import rename_snp, AUTOSOME_LIST



class GwasDataLoader():
    
    def __init__(self, 
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
        '''
        Read GWAS data
        '''
        self.bed_path = bed_path
        self.pheno_path = pheno_path
        self.pheno_name = pheno_name
        self.cov_path = cov_path
        self.snp_list = snp_list
        self.ind_list = ind_list
        
        self.mean_fill_na_flag = mean_fill_na_flag
        self.read_all_gt_flag = read_all_gt_flag
        self.rename_snp_flag = rename_snp_flag

        # init value
        self.BED = None
        self.FAM = None
        self.BIM = None
        self.COV = None
        self.snp_idx_list: Set[str] = set()
        self.ind_idx_list: Set[str] = set()
        self.read_in_flag = False


    ####### READ FILE #######
    def read_in(self):
        logging.info(f"Start read file")

        self.BED = self._read_bed()
        self.FAM = self._read_fam()
        self.BIM = self._read_bim()
        if self.cov_path and os.path.exists(str(self.cov_path)): 
            self.COV = self._read_cov()
        if self.pheno_path and os.path.exists(str(self.pheno_path)):
            self._read_pheno()
            
        # filter
        if self.snp_list:
            self.snp_idx_list = self._read_snp_list()
        else:
            self.snp_idx_list = list(range(len(self.BIM.index)))
        if self.ind_list:
            self.ind_idx_list = self._read_ind_list()
        else:
            self.ind_idx_list = set(range(len(self.FAM.index)))

        self._check_pheno_missing()
        self._filter()
        self.read_in_flag = True
    
    
    
    ####### Get info #######
    def get_geno(self):
        if self.read_all_gt_flag:
            logging.info(f"Read in complete genotype matrix")
            index = np.s_[list(self.ind_idx_list),self.snp_idx_list]
            GT = self.BED.read(index=index,)

            logging.info(f"return {GT.shape[0]} individaul")
            logging.info(f"return {GT.shape[1]} snp")

            return GT
        else:
            return self.BED

    def get_pheno(self):
        return self.FAM.PHENO1.values
    
    def get_sample(self):
        fid_iid_list = list(zip(self.FAM.FID.values, self.FAM.IID.values))
        return fid_iid_list

    def get_snp(self, autosome_only = False):
        # need to reparse BIM.ID for inference of string type
        if autosome_only:
            BIM = self.BIM.loc[self.BIM.CHR.isin(AUTOSOME_LIST)]
            SNP = BIM.ID.values.tolist()
        else:
            SNP = self.BIM.ID.values.tolist()

        return np.array(SNP)

    def get_old_snp(self, autosome_only = False):
        assert self.rename_snp_flag
        if autosome_only:
            BIM = self.BIM.loc[self.BIM.CHR.isin(AUTOSOME_LIST)]
            SNP = BIM.Original_ID.values.tolist()
        else:
            SNP = self.BIM.Original_ID.values.tolist()
        return np.array(SNP)

    def get_snp_table(self, autosome_only = True, dedup = True):
        assert self.rename_snp_flag
        snp_list_ori = self.get_snp(autosome_only=autosome_only) 
        snp_list_old = self.get_old_snp(autosome_only=autosome_only)
        # remove dup
        if dedup:

            snp_id_table = {}
            snp_list = []
            for i in range(len(snp_list_ori)):
                snp = snp_list_ori[i]
                if snp not in snp_id_table:
                    snp_id_table[snp] = snp_list_old[i]
                    snp_list.append(snp)

        return np.array(snp_list), snp_id_table

    def get_allele(self):
        return self.BIM.A1.values, self.BIM.A2.values

    def get_cov(self, add_bias_flag = True):
        ll = len(self.FAM.FID.values)
        
        COV = None
        if os.path.exists(str(self.cov_path)):
            COV = self.COV.iloc[:,2:].values 
            if add_bias_flag:
                BIAS = np.ones((ll,1), dtype = COV.dtype)
                COV = np.concatenate(( BIAS, COV), axis=1)
        else:
            if add_bias_flag:
                COV = np.ones((ll,1), dtype = np.float32)

        return COV


    ####### HELPERS #######
    def _read_bed(self):
        BED = open_bed(f"{self.bed_path}.bed")
        logging.info(f"BED file: {self.bed_path}.bed")
        return BED

    def _read_fam(self):
        FAM = pd.read_csv(f"{self.bed_path}.fam", sep = '\s+', header = None)
        FAM.columns = ["FID","IID","P","M","SEX","PHENO1"]
        FAM.FID = FAM.FID.astype(str)
        FAM.IID = FAM.IID.astype(str)
        logging.debug(f"FAM file: {self.bed_path}.fam")
        self.raw_ind_num = len(FAM.index)
        return FAM

    def _read_bim(self):
        BIM = pd.read_csv(f"{self.bed_path}.bim", sep = '\s+', header = None)
        BIM.columns = ["CHR","ID","cM","POS","A1","A2"]
        BIM.A1 = BIM.A1.astype(str).replace("0",".")
        BIM.A2 = BIM.A2.astype(str).replace("0",".")
        BIM.ID = BIM.ID.astype(str)
        logging.debug(f"BIM file: {self.bed_path}.bim")
        self.raw_snp_num = len(BIM.index)
        
        return BIM

    def _read_cov(self):
        sep = '\s+'
        if '.csv' in self.cov_path:
            sep = ','
        COV = pd.read_csv(self.cov_path, sep = sep)
        # align to fam
        COV.FID = COV.FID.astype(str)
        COV.IID = COV.IID.astype(str)
        COV = COV.drop_duplicates(subset = ["FID","IID"])
        COV = self.FAM[["FID","IID"]].merge(COV, on = ["FID","IID"], how = 'left', sort = False)
        COV = COV.reset_index(drop = True)
        # fill with na
        if self.mean_fill_na_flag:
            for col in COV.columns[2:]:
                x = COV[col]
                col_mean = np.nanmean(col, axis = 0)
                na_idx = np.where(np.isnan(x))
                x[na_idx] = np.take(col_mean, na_idx[1])
                COV[col] = x

        logging.info(f"COV file: {self.cov_path}")
        cov_names = ','.join(COV.columns[2:])
        logging.info(f"COV name: {cov_names}")
        return COV

    def _read_pheno(self):
        sep = '\s+'
        if '.csv' in self.pheno_path:
            sep = ','
        PHENO = pd.read_csv(self.pheno_path, sep = sep)
        PHENO = PHENO[["FID","IID", self.pheno_name]]
        PHENO.columns = ["FID","IID", "PHENO1"]
        PHENO.FID = PHENO.FID.astype(str)
        PHENO.IID = PHENO.IID.astype(str)
        # align to fam
        self.FAM = self.FAM.drop(columns = "PHENO1")
        self.FAM = self.FAM.merge(PHENO, on = ["FID","IID"], how = 'left', sort = False)
        self.FAM = self.FAM.reset_index(drop = True)
        logging.info(f"PHENO file: {self.pheno_path}")

    def _read_snp_list(self, odered_flag = False):
        if isinstance(self.snp_list, str):
            logging.info(f"SNP list: {self.snp_list}")
            SNP = pd.read_csv(self.snp_list, sep = '\s+', header = None)
        elif isinstance(self.snp_list, list):
            tmp_file = f"{self.bed_path}.tmp_snp_list"
            with open(tmp_file, 'w') as FF:
                for i in self.snp_list:
                    FF.write(f"{i}\n")
            SNP = pd.read_csv(tmp_file, sep = '\s+', header = None)

        if odered_flag:
            DA = self.BIM.loc[:,["ID"]].copy()
            DA["INDEX"] = DA.index
            DA = DA.set_index("ID", drop = False)
            SNP = SNP.loc[SNP[0].isin(DA.ID)]
            snp_idx_list = DA.loc[SNP[0], "INDEX"].values            

        else:
            snp_idx_list = self.BIM[self.BIM.ID.isin(SNP[0])].index

        if len(snp_idx_list) == 0:
            logging.error("No SNP left after ind_list filter")
            raise IndexError

        logging.info(f"Read {len(SNP.index)} snps")
        return snp_idx_list

    def _read_ind_list(self):
        if isinstance(self.ind_list, str):
            IND = pd.read_csv(self.ind_list, sep = '\s+', header = None)
            logging.info(f"IND list: {self.ind_list}")
        elif isinstance(self.ind_list, list):
            tmp_file = f"{self.bed_path}.tmp_ind_list"
            with open(tmp_file, 'w') as FF:
                for i in self.ind_list:
                    FF.write(f"{i},{i}\n")
            IND = pd.read_csv(tmp_file, sep = '\s+', header = None)

        if len(IND.columns) > 2:
            IND = IND.iloc[:,:2]
        IND.columns = ["FID", "IID"]
        
        self.FAM["INDEX"] = self.FAM.index
        ind_idx_list = self.FAM.merge(IND, on = ["FID", "IID"], sort = False).loc[:,"INDEX"].values
        self.FAM = self.FAM.drop(columns="INDEX")
        
        if len(ind_idx_list) == 0:
            logging.error("No individual left after ind_list filter")
            raise IndexError
        logging.info(f"Read {len(IND.index)} individuals")
        return set(ind_idx_list)

    def _check_pheno_missing(self):
        rm_count = 0
        bad_ind_idx = set()
        
        if self.COV is not None:
            bad_ind_idx0 = self.COV.loc[self.COV.isnull().any(axis=1)].index
            bad_ind_idx1 = self.COV.loc[(self.COV == -9 ).any(axis=1)| \
                (self.COV == -9.0 ).any(axis=1)].index
            if len(bad_ind_idx1) > 0:
                logging.warning(f"-9 is found in cov column, which may be ambiguous in quantitative covariate. \
                    Suggested to change it as NaN. We will regard it as missing for now")
                bad_ind_idx0 = bad_ind_idx0.union(set(bad_ind_idx1))

            rm_count += len(bad_ind_idx0)
            bad_ind_idx = bad_ind_idx.union(set(bad_ind_idx0))
        
        bad_ind_idx0 = self.FAM.loc[self.FAM.PHENO1.isnull()].index
        bad_ind_idx1 = self.FAM.loc[(self.FAM.PHENO1 == -9 )| (self.FAM.PHENO1 == -9.0 )].index
        if len(bad_ind_idx1) > 0:
            logging.warning(f"-9 is found in fam PHENO column, which may be ambiguous in quantitative pheno. \
                Suggested to change it as NaN. We will regard it as missing for now")
            bad_ind_idx0 = bad_ind_idx0.union(set(bad_ind_idx1))

        rm_count += len(bad_ind_idx0)
        bad_ind_idx = bad_ind_idx.union(set(bad_ind_idx0))
        logging.warning(f"{rm_count} individuals to be remove due to missing value in pheno or covaraite")
        
        self.ind_idx_list = self.ind_idx_list.difference(bad_ind_idx)
        
    def _filter(self):
        self.FAM = self.FAM.iloc[list(self.ind_idx_list)]
        self.BIM = self.BIM.iloc[self.snp_idx_list]
        if self.COV is not None:
            self.COV = self.COV.iloc[list(self.ind_idx_list)]

        if self.rename_snp_flag:
            new_snp_list = rename_snp(self.BIM, to_byte = False, to_dict = False)
            self.BIM["Original_ID"] = self.BIM.ID
            self.BIM.ID = new_snp_list




class GwasSnpIterator():
    '''
    This class iter through snp
    iter read snp
    '''
    
    def __init__(self, GwasDataLoader: GwasDataLoader, batch_size: int, snp_name_list: List[str] = [], swap_flag: bool = True):
        if not GwasDataLoader.read_in_flag:
            GwasDataLoader.read_in()
            
        # GwasDataLoader data
        self.BED = GwasDataLoader.BED
        self.FAM = GwasDataLoader.FAM
        self.BIM = GwasDataLoader.BIM
        self.COV = GwasDataLoader.COV
        self.snp_idx_list = list(GwasDataLoader.snp_idx_list)
        self.ind_idx_list = list(GwasDataLoader.ind_idx_list)
        self.swap_flag = swap_flag
        
        # self arg
        self.batch_size = batch_size
        self.build_flag = False
        
        self.build(snp_name_list)

    def build(self, snp_name_list: List[str] = []):
        self.snp_name_list = snp_name_list
        if len(self.snp_name_list) > 0:
            self._update_snp()

        # derived argument
        self.snp_num = len(self.snp_idx_list)
        self.total_step = (self.snp_num // self.batch_size) + 1
        logging.info(f"SNP Genrator: Total num {self.snp_num}; Batch size {self.batch_size}; Total step {self.total_step}")

        # init
        self._start = 0
        self._end = self.batch_size
        self.cc = 0
        self.build_flag = True
    
    def __iter__(self):
        for _ in range(self.total_step):
            logging.debug(f"Get batch {self.cc}")
            yield self.__next__()

    def __next__(self):
            
        snp_idx_list = np.array(list(self.snp_idx_list))[self._start:self._end]
        idx_list = np.s_[self.ind_idx_list,snp_idx_list]
        GT = self.BED.read(index=idx_list)
        BIM = self.BIM.iloc[snp_idx_list,:].copy()
        self._update()
        
        if self.swap_flag:
            GT, BIM = self.swap_allele(GT, BIM)
        
        return GT, BIM
        
    def __len__(self):
        return self.total_step

    def _update(self):
        self._start += self.batch_size
        self._end += self.batch_size
        self._end = min(self._end, self.snp_num)
        self.cc += 1

    def _update_snp(self):
        self.BIM = self.BIM.loc[self.BIM.ID.isin(self.snp_name_list)]
        self.snp_idx_list = list(self.BIM.index)

    def swap_allele(self, GT, BIM):
        BIM.loc[:,"INDEX"] = BIM.index
        BIM = BIM.reset_index(drop = True)
        # get swap idx
        BIM.loc[:,"SWAP"] = BIM.A1 > BIM.A2
        
        # swap bim
        BIM.loc[:,"TMP"]  = BIM.A1
        BIM.loc[BIM.SWAP, "A1"] = BIM[BIM.SWAP].A2
        BIM.loc[BIM.SWAP, "A2"] = BIM[BIM.SWAP].TMP
        BIM = BIM.drop(columns = ["TMP"])

        # swap gt
        swap_index = BIM[BIM.SWAP].index
        GT[:,swap_index] = np.abs(GT[:,swap_index] - 2)
        
        # reset index
        BIM = BIM.set_index("INDEX", drop = True)
        return GT, BIM





class GwasIndIterator():
    '''
    This class iter through ind
    read in all snp first
    '''
    
    def __init__(self, GwasDataLoader: GwasDataLoader, batch_size: int ):
        if not GwasDataLoader.read_in_flag:
            GwasDataLoader.read_in()
            
        # GwasDataLoader data
        self.BED = GwasDataLoader.BED
        self.FAM = GwasDataLoader.FAM
        self.BIM = GwasDataLoader.BIM
        self.COV = GwasDataLoader.COV
        self.snp_idx_list = list(GwasDataLoader.snp_idx_list)
        self.ind_idx_list = list(GwasDataLoader.ind_idx_list)
        self.GwasDataLoader = GwasDataLoader

        # self arg
        self.batch_size = batch_size
        
        # derived argument
        self.ind_num = len(self.ind_idx_list)
        self.total_step = (self.ind_num // self.batch_size) + 1
        logging.info(f"IND Genrator: Total num {self.ind_num}; Batch size {self.batch_size}; Total step {self.total_step}")

        # init
        self._start = 0
        self._end = self.batch_size
        self.cc = 0

        
    def __iter__(self):
        for _ in range(self.total_step):
            logging.debug(f"Get batch {self.cc}")
            yield self.__next__()


    def __next__(self):
        idx_list = np.array(list(self.ind_idx_list))[self._start:self._end]
        GT = self.BED.read(index = np.s_[idx_list,self.snp_idx_list])
        self._update()
        
        return GT
        
    def __len__(self):
        return self.snp_num

    def _update(self):
        self._start += self.batch_size
        self._end += self.batch_size
        self._end = min(self._end, self.ind_num)
        self.cc += 1

