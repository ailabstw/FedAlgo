from typing import Any, List, Set
import pandas as pd
import os
import copy
import logging
import numpy as np
from bed_reader import open_bed
import abc

AUTOSOME_LIST = ()
for i in range(1,23):
    AUTOSOME_LIST += (i, str(i), f'chr{i}')

def read_bed(bfile_path):
    return open_bed(f'{bfile_path}.bed')

def fam_setter(FAM):
    FAM.columns = ['FID','IID','P','M','SEX','PHENO1']
    FAM.FID = FAM.FID.astype(str)
    FAM.IID = FAM.IID.astype(str)
    return FAM

def bim_setter(BIM):
    BIM.columns = ['CHR','ID','cM','POS','A1','A2']
    BIM.A1 = BIM.A1.astype(str).replace('0','.')
    BIM.A2 = BIM.A2.astype(str).replace('0','.')
    BIM.ID = BIM.ID.astype(str)
    return BIM

def cov_setter(COV):
    COV.FID = COV.FID.astype(str)
    COV.IID = COV.IID.astype(str)
    return COV.drop_duplicates(subset = ['FID','IID'])

def pheno_setter(PHENO):
    PHENO.columns = ['FID','IID','PHENO1']
    PHENO.FID = PHENO.FID.astype(str)
    PHENO.IID = PHENO.IID.astype(str)
    return PHENO

def read_fam(bfile_path):
    return fam_setter(pd.read_csv(f'{bfile_path}.fam', sep='\s+', header=None))

def read_bim(bfile_path):
    return bim_setter(pd.read_csv(f'{bfile_path}.bim', sep='\s+', header=None))

def read_cov(cov_path):
    sep = ',' if '.csv' in cov_path else '\s+'
    COV = pd.read_csv(cov_path, sep=sep)
    return cov_setter(COV)

def read_pheno(pheno_path, pheno_name):
    sep = ',' if '.csv' in pheno_path else '\s+'
    PHENO = pd.read_csv(pheno_path, sep=sep)
    PHENO = PHENO[['FID','IID',pheno_name]]
    return pheno_setter(PHENO)

def read_snp_list(snp_list_path):
    """
    Read a snp list from a given path.
    """
    SNP_df = pd.read_csv(snp_list_path, sep='\s+', header=None)
    SNP_df.columns = ['ID']
    return SNP_df

def read_ind_list(ind_list_path):
    """
    Read a sample list from a given path.
    """
    IND_df = pd.read_csv(ind_list_path, sep='\s+', header=None).iloc[:,:2]
    IND_df.columns = ['FID','IID']
    return IND_df

def format_cov(COV, FAM):
    """
    Read covarities from a given path and map to the corresponding FAM file.
    """
    # the samples in .fam missing covariate values will fill with NaNs
    return FAM[['FID','IID']].merge(COV, on=['FID','IID'], how='left', sort=False)

def format_fam(FAM, pheno_path=None, pheno_name=None):
    """
    Replace the FAM file with the corresponding pheno file.
    """
    if pheno_path is not None:
        PHENO = read_pheno(pheno_path, pheno_name)
        FAM.drop(columns='PHENO1', inplace=True)
        # the samples in .fam missing phenotypes values will fill with NaNs
        FAM = FAM.merge(PHENO, on=['FID','IID'], how='left', sort=False)
    return FAM

def get_mask_idx(df):
    """
    Get the index of missing values in a dataframe.
    """
    mask_miss1 = df.isnull().any(axis=1)
    mask_miss2 = (df == (-9 or -9.0)).any(axis=1)
    return df[mask_miss1 | mask_miss2].index

def impute_cov(COV):
    # nanmean imputation
    col_means = COV.iloc[:,2:].mean()
    COV.fillna(col_means, inplace=True)
    return COV

def create_unique_snp_id(BIM, to_byte=True, to_dict=False):
    """
    Create the unique ID as CHR:POS:A1:A2
    If the A1 and A2 are switched, record the SNP index for adjusting the genptype values (0 > 2, 2 > 0).
    """
    unique_id, sorted_snp_idx = [], []
    for line in BIM.iterrows():
        CHR = str(line[1]['CHR'])
        POS = str(line[1]['POS'])
        ALLELE = [str(line[1]['A1'])[:23], str(line[1]['A2'])[:23]]
        ALLELE_sorted = sorted(ALLELE)
        unique_id.append(f"{CHR}:{POS}:{ALLELE_sorted[0]}:{ALLELE_sorted[1]}")
        if ALLELE != ALLELE_sorted:
            sorted_snp_idx.append(line[0])
    if to_byte:
        unique_id = np.array(unique_id, dtype="S")
    if to_dict:
        unique_id = dict(zip(unique_id, BIM.ID))
    return unique_id, sorted_snp_idx

def redirect_genotype(GT, snp_idx):
    GT[:, snp_idx] = 2 - GT[:, snp_idx]
    return GT

def dropped_info(data, subset, cols):
    data_id = list(zip(*data[cols].to_dict('list').values()))
    subset_id = list(zip(*subset[cols].to_dict('list').values()))
    dropped_idx = [idx for idx, id in enumerate(data_id) if id not in subset_id]
    return data.iloc[dropped_idx,:]

def update_dropped(prev, update):
    return pd.concat([prev, update]).reset_index(drop=True)
        
def subset_samples(sample_list:(str, list, tuple), data, order=False, list_is_idx=False):
    """
    Args:
        sample_list (str, list, tuple) : could be a list of sample IDs, a path to sample list file or a list of sample indices.
        data (pd.DataFrame) : the data to be extracted.
    
    Returns:
        subset_data (pd.DataFrame) : the subset of the data (the index of the subset_data has been reset)
        sample_idx : the indices of sample list in "data", not the indices in "subset_data".
        dropped_data (pd.DataFrame) : the dropped subset.
    """
    # Format sample list
    if isinstance(sample_list, str):
        sample_df = read_ind_list(sample_list)
    elif isinstance(sample_list, (tuple, list)) and not list_is_idx:
        sample_df = pd.DataFrame({'FID':sample_list,'IID':sample_list})
    else:
        sample_df = data.iloc[sample_list,:2].reset_index(drop=True)

    # Follow the order in the ind_list
    if order:
        subset_data = sample_df.merge(data, on=['FID','IID'])
    else:
        subset_data = data.merge(sample_df, on=['FID','IID'])

    if len(subset_data) == 0:
        raise IndexError
    
    # Get the indices of samples in original data for getting the ordered genotype matrix.
    sample_idx = subset_data.merge(data.reset_index())['index'].to_list()

    # Dropped data
    dropped_data = dropped_info(data, subset_data, ['FID','IID'])
    
    return subset_data, sample_idx, dropped_data

def subset_snps(snp_list:(str, list, tuple), data, order=False, list_is_idx=False):
    """
    Args:
        snp_list (str, list, tuple) : could be a list of SNP IDs, a path to snp list file or a list of snp indices.
        data (pd.DataFrame) : the data to be extracted.
    
    Returns:
        subset_data (pd.DataFrame) : the subset of the data (the index of the subset_data has been reset)
        snp_idx : the indices of snp list in "data", not the indices in "subset_data".
        dropped_data (pd.DataFrame) : the dropped subset.
    """
    # Format SNP list
    if isinstance(snp_list, str):
        snp_df = read_snp_list(snp_list)
    elif isinstance(snp_list, (tuple, list)) and not list_is_idx:
        snp_df = pd.DataFrame({'ID':snp_list})
    else:
        snp_df = data.iloc[snp_list,1].reset_index(drop=True)
    
    # Follow the order of the snp_list
    if order:
        subset_snps = snp_df.merge(data, on=['ID'])
    else:
        subset_snps = data.merge(snp_df, on=['ID'])
    
    if len(subset_snps) == 0:
        raise IndexError
    
    # Get the indices of snps in original data for getting the ordered genotype matrix.
    snp_idx = subset_snps.merge(data.reset_index())['index'].to_list()

    # Dropped data
    dropped_data = dropped_info(data, subset_snps, ['ID'])

    return subset_snps, snp_idx, dropped_data

def index_non_missing_samples(FAM, COV=None):
    """
    Index samples without any missing values in FAM, pheno, or covariates.
    """
    FAM_rm_idx = set(get_mask_idx(FAM))
    if COV is not None:
        COV_rm_idx = get_mask_idx(COV)
        rm_sample_idx = FAM_rm_idx.union(COV_rm_idx)
    else:
        rm_sample_idx = FAM_rm_idx
    keep_ind_idx = set(FAM.index).difference(rm_sample_idx)

    return list(keep_ind_idx)

def create_snp_table(snp_id_list, rs_id_list):
    """
    Create the mapping table that unique IDs can be mapped to the rsIDs.
    """
    snp_id_table = {}
    for i in range(len(snp_id_list)):
        snp_id = snp_id_list[i]
        if snp_id not in snp_id_table:
            snp_id_table.setdefault(snp_id, rs_id_list[i])

    return list(snp_id_table.keys()), snp_id_table

class GWASData:
    """
    The GWASData performs three main operations, subsect extraction, dropping samples with missing values
    and add unique position ID for each SNP.

    subset()
        This function allows multiple times to extract the subset with the given sample list and SNP list.
        ex. subset(sample_list1)
                    :
            subset(sample_list9)

        Limitations: If the sample list is an index list, and the SNP list is not, \
                     please do it in two steps, the first step can be subset(sample_list, list_is_idx), \
                     the second step can be subset(snp_list). Switching the order is fine.
                     Similar situation is the same as `order`.

        Args:
            sample_list (optional, str, tuple, list, default=None) : Allows the path to the snp list, a list of sample IDs or a list of sample indices. \
                                                                     Note that the np.array dtype is not supported. The default is return the whole samples.

            snp_list (optional, str, tuple, list, default=None)    : Allows the path to the SNP list, a list of SNP IDs or a list of SNP indices. \
                                                                     Note that the np.array dtype is not supported. The default is return the whole SNPs.

            order (boolean, default=False) : Determines the sample and snp order in fam, bim, cov and genotype. \
                                             If True, the order follows the given sample/snp list.

            list_is_idx (boolean, default=False) : If the `sample_list` or the `snp_list` are indices, this parameter should be specified as True.

        Returns:
            Subset of fam (pd.DataFrame)
            Subset of cov (pd.DataFrame)
            Subset of bim (pd.DataFrame)
            Subset of genotype (np.ndarray)
            dropped_fam (pd.DataFrame)
            dropped_cov (pd.DataFrame)
            dropped_bim (pd.DataFrame)

    
    drop_missing_samples():
        Drop samples whose phenotype or covariates contain missing values ('', NaN, -9, -9.0).

        Returns:
            Subset of fam (pd.DataFrame) : samples without missing values
            Subset of cov (pd.DataFrame) : samples without missing values
            dropped_fam (pd.DataFrame) : samples with missing values
            dropped_cov (pd.DataFrame) : samples with missing values


    add_unique_snp_id():
        Add the unique IDs for each SNP.

        Returns:
            bim : With unique IDs and rsIDs
            genotype : If the A1 and A2 are switched, snp array = 2 - snp array
    """
    def __init__(self, GT, fam, bim, cov):
        self.__dict__.update(locals())
        self.__dict__.update({f'dropped_{data}':pd.DataFrame() for data in ['fam', 'bim', 'cov']})

    def standard(self):
        self.subset()
        self.drop_missing_samples()
        self.add_unique_snp_id()
    
    def custom(self, **kwargs):
        self.subset(**kwargs)

        if kwargs.get('impute_cov') is True:
            self.impute_covariates()

        self.drop_missing_samples()

        if kwargs.get('add_unique_snp_id') is True:
            self.add_unique_snp_id()

    def subset(self, sample_list=None, snp_list=None, order=False, list_is_idx=False):
        # Sample information
        if sample_list:
            self.fam, sample_idx, dropped_fam = subset_samples(sample_list, self.fam, order, list_is_idx)
            self.dropped_fam = update_dropped(self.dropped_fam, dropped_fam)

            if self.cov is not None:
                self.cov, _, dropped_cov = subset_samples(sample_list, self.cov, order, list_is_idx)
                self.dropped_cov = update_dropped(self.dropped_cov, dropped_cov)
        else:
            sample_idx = list(self.fam.index)

        # SNP information
        if snp_list:
            self.bim, snp_idx, dropped_bim = subset_snps(snp_list, self.bim, order, list_is_idx)
            self.dropped_bim = update_dropped(self.dropped_bim, dropped_bim)
        else:
            snp_idx = list(self.bim.index)

        # Genotype information
        self.GT = self.GT[np.ix_(sample_idx, snp_idx)]

    def impute_covariates(self):
        self.cov = impute_cov(self.cov)

    def drop_missing_samples(self):
        # Re-subset the samples without any missing values
        sample_idx = index_non_missing_samples(self.fam, self.cov)
        self.subset(sample_list=sample_idx, list_is_idx=True)

    def add_unique_snp_id(self):
        unique_id, sorted_snp_idx = create_unique_snp_id(self.bim, to_byte=False, to_dict=False)
        self.bim['rsID'] = self.bim['ID']
        self.bim['ID'] = unique_id
        self.GT = redirect_genotype(self.GT, sorted_snp_idx)

    @property
    def phenotype(self):
        return self.fam
    
    @property
    def sample_id(self):
        return list(zip(self.fam.FID, self.fam.IID))
    
    @property
    def covariate(self):
        return self.cov
    
    @property
    def snp_id(self):
        return list(self.bim.ID)
    
    @property
    def autosome_snp_id(self):
        return list(self.bim[self.bim.CHR.isin(AUTOSOME_LIST)].ID)
    
    @property
    def rsID(self):
        return list(self.bim.rsID)
    
    @property
    def autosome_rsID(self):
        return list(self.bim[self.bim.CHR.isin(AUTOSOME_LIST)].rsID)
    
    @property
    def allele(self):
        return list(zip(self.bim.A1, self.bim.A2))
    
    @property
    def genotype(self):
        return self.GT
    
    @property
    def snp_table(self):
        assert 'rsID' in self.bim.columns
        return create_snp_table(self.snp_id, self.rsID)

    @property
    def autosome_snp_table(self):
        assert 'rsID' in self.bim.columns
        return create_snp_table(self.autosome_snp_id, self.autosome_rsID)
    
    @property
    def dropped_phenotype(self):
        return self.dropped_fam
    
    @property
    def dropped_covariate(self):
        return self.dropped_cov
    
    @property
    def dropped_snp(self):
        return self.dropped_bim


def read_gwasdata(bfile_path, cov_path=None, pheno_path=None, pheno_name='PHENO1'):
    bed = read_bed(bfile_path)
    GT = bed.read()
    fam = read_fam(bfile_path)
    fam = format_fam(fam, pheno_path, pheno_name)
    bim = read_bim(bfile_path)
    cov = read_cov(cov_path)
    cov = format_cov(cov, fam)
    return GWASData(GT, fam, bim, cov)
    
def format_sample_metadata(bfile_path, cov_path=None, pheno_path=None, pheno_name='PHENO1'):
    """
    """
    fam = read_fam(bfile_path)
    fam = format_fam(fam, pheno_path, pheno_name)
    fam.to_csv('iterative.fam', index=None)
    if cov_path:
        cov = read_cov(cov_path)
        cov = format_cov(cov, fam)
        cov.to_csv('iterative.cov', index=None)

class IndexIterator(abc.ABC):
    def __init__(self, n_features, chunk_size:int=1000, skip_idx:(list, range)=None):
        self.n_features = n_features
        self.current_idx = 0
        self.chunk_size = chunk_size
        self.skip_idx = skip_idx
    
    def keep_features(self, chunk_size):
        interval = (self.current_idx, min(self.current_idx+chunk_size, self.n_features))
        if self.skip_idx:
            keep_features = list(set(range(*interval)).difference(self.skip_idx))
        else:
            keep_features = range(*interval)
        return keep_features
    
    def reset(self):
        self.current_idx = 0

    def is_end(self):
        return self.current_idx >= self.n_features

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_idx < self.n_features:
            idx = self.keep_features(self.chunk_size)
            self.current_idx += self.chunk_size
            return idx
        else:
            raise StopIteration

class SNPIterator(IndexIterator):
    def __init__(self, n_SNP, chunk_size:int=1000, skip_idx:(list, range)=None):
        super().__init__(n_SNP, chunk_size, skip_idx)
        self.n_SNP = n_SNP
        self.sample_iterator = None
    
    def keep_idx(self, chunk_size):
        if self.sample_iterator:
            sample_idx = next(self.sample_iterator)
            return np.s_[sample_idx, self.keep_features(chunk_size)]
        else:
            return np.s_[:, self.keep_features(chunk_size)]
        
    def samples(self, n_sample, chunk_size:int=1000, skip_idx:(list, range)=None):
        self.sample_iterator = IndexIterator(n_sample, chunk_size, skip_idx)
        return self
    
    def get_chunk(self, chunk_size, reset=True):
        """
        Usage:
            SNPIterator.get_chunk()
                the next SNP idx starts from current_idx + chunk_size

            SNPIterator.get_chunk(reset=False)
                the next SNP idx starts from current_idx + chunk_size

            SNPIterator.samples().get_chunk()
                the next SNP idx starts from current_idx + chunk_size
                the next Sample idx starts from 0

            SNPIterator.samples().get_chunk(reset=False)
                if Sample idx approch the end,
                the next SNP idx starts from current_idx + chunk_size
                the next Sample idx starts from 0

                otherwise,
                the next SNP idx starts from current_idx
                the next Sample idx starts from current_idx + chunk_size
        
        More explicit usage can be found in unittest.
        """
        if self.current_idx < self.n_SNP:
            idx = self.keep_idx(chunk_size)
            self.current_idx += chunk_size
            # reset was used to prevent StopIteration caused by sample_iterator
            if self.sample_iterator and (self.sample_iterator.is_end() or reset):
                self.sample_iterator.reset()
            # don't jump to the next chunk until reaching the end
            elif self.sample_iterator and not reset:
                self.current_idx -= chunk_size
            return idx
        else:
            raise StopIteration

    def __next__(self):
        return self.get_chunk(self.chunk_size, reset=False)
    
    @property
    def category(self):
        if self.sample_iterator:
            return 'snp-sample'
        else:
            return 'snp'

class SampleIterator(IndexIterator):
    def __init__(self, n_sample, chunk_size:int=1000, skip_idx:(list, range)=None):
        super().__init__(n_sample, chunk_size, skip_idx)
        self.n_sample = n_sample
        self.snp_iterator = None
    
    def keep_idx(self, chunk_size):
        if self.snp_iterator:
            snp_idx = next(self.snp_iterator)
            return np.s_[self.keep_features(chunk_size), snp_idx]
        else:
            return np.s_[self.keep_features(chunk_size), :]
    
    def snps(self, n_SNP, chunk_size:int=1000, skip_idx:(range, list)=None):
        self.snp_iterator = IndexIterator(n_SNP, chunk_size, skip_idx)
        return self

    def get_chunk(self, chunk_size, reset=True):
        """
        Usage:
            SampleIterator.get_chunk()
                the next Sample idx starts from current_idx + chunk_size

            SampleIterator.get_chunk(reset=False)
                the next Sample idx starts from current_idx + chunk_size

            SampleIterator.snps().get_chunk()
                the next Sample idx starts from current_idx + chunk_size
                the next SNP idx starts from 0

            SampleIterator.snps().get_chunk(reset=False)
                if SNP idx approch the end,
                the next Sample idx starts from current_idx + chunk_size
                the next SNP idx starts from 0

                otherwise,
                the next Sample idx starts from current_idx
                the next SNP idx starts from current_idx + chunk_size
        
        More explicit usage can be found in unittest.
        """
        if self.current_idx < self.n_sample:
            idx = self.keep_idx(chunk_size)
            self.current_idx += chunk_size
            # reset was used to prevent StopIteration caused by snp_iterator
            if self.snp_iterator and (self.snp_iterator.is_end() or reset):
                self.snp_iterator.reset()
            # don't jump to the next chunk until reaching the end
            elif self.snp_iterator and not reset:
                self.current_idx -= chunk_size
            return idx
        else:
            raise StopIteration

    def __next__(self):
        return self.get_chunk(self.chunk_size, reset=False)
    
    @property
    def category(self):
        if self.snp_iterator:
            return 'sample-snp'
        else:
            return 'sample'


class BedIterator:
    def __init__(self, bfile_path:str, iterator:(SampleIterator|SNPIterator)):
        self.bed = read_bed(bfile_path)
        self.iterator = iterator
    
    def read(self):
        return self.bed.read()
    
    def get_chunk(self, chunk_size, reset=True):
        # the chunk_size is the sample chunk size of SampleIterator(SNPIterator)
        # the chunk_size is the snp chunk size of SNPIterator(SampleIterator)
        idx = self.iterator.get_chunk(chunk_size, reset)
        return self.bed.read(index=idx)
        
    def __iter__(self):
        return self
    
    def __next__(self):
        idx = next(self.iterator)
        return self.bed.read(index=idx)
    
class FamIterator:
    def __init__(self, bfile_path, cov_path=None, pheno_path=None, pheno_name='PHENO1', iterator:(SampleIterator|None)=None):
        format_sample_metadata(bfile_path, cov_path, pheno_path, pheno_name)
        self.fam = pd.read_csv('iterative.fam', iterator=True)
        self.iterator = iterator
        if type(self.iterator) is SampleIterator or type(self.iterator) is type(None):
            self.static = False
        else:
            raise TypeError('iterator only supports SampleIterator and NoneType.')
    
    def read(self):
        return fam_setter(self.fam.read())
    
    def to_static(self):
        self.static = True
        self.fam = self.read()
    
    def reset(self):
        self.fam = pd.read_csv('iterative.fam', iterator=True)
    
    def get_chunk(self, chunk_size):
        if not self.static:
            fam = self.fam.get_chunk(chunk_size)
            if self.iterator:
                idx = self.iterator.get_chunk(chunk_size)
                idx = np.s_[idx[0],:] # ignore snp index
                return fam_setter(fam).loc[idx]
            else:
                return fam_setter(fam)
        else:
            # Static mode ignores chunk_size information
            return self.fam
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.static:
            return self.get_chunk(self.iterator.chunk_size)
        else:
            raise AttributeError("Static mode cannot perform iteration. Please use `get_chunk()` to get the static data.")
    
class CovIterator:
    def __init__(self, bfile_path, cov_path=None, pheno_path=None, pheno_name='PHENO1', iterator:(SampleIterator|None)=None):
        format_sample_metadata(bfile_path, cov_path, pheno_path, pheno_name)
        if cov_path:
            self.cov = pd.read_csv('iterative.cov', iterator=True)
        else:
            self.cov = None
            
        self.iterator = iterator
        if type(self.iterator) is SampleIterator or type(self.iterator) is type(None):
            if self.cov:
                self.static = False
            else:
                self.to_static()
        else:
            raise TypeError('iterator only supports SampleIterator and NoneType.')

    def read(self):
        if self.cov:
            return cov_setter(self.cov.read())
        else:
            return None
    
    def to_static(self):
        self.static = True
        self.cov = self.read()
    
    def reset(self):
        if self.cov: self.cov = pd.read_csv('iterative.cov', iterator=True)
    
    def get_chunk(self, chunk_size):
        if self.cov is not None and not self.static:
            cov = self.cov.get_chunk(chunk_size)
            if self.iterator:
                idx = self.iterator.get_chunk(chunk_size)
                idx = np.s_[idx[0],:] # ignore snp index
                return cov_setter(cov).loc[idx]
            else:
                return cov_setter(cov)
        else:
            # Allowing static mode iteration is because 
            # cov could be missing when fam iterator is dynamic mode, and they should share the same index iterator.
            if self.iterator:
                self.iterator.get_chunk(chunk_size)
            return self.cov

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.get_chunk(self.iterator.chunk_size)
    
class BimIterator:
    def __init__(self, bfile_path, iterator:(SNPIterator|None)=None):
        self.bim = pd.read_csv(f'{bfile_path}.bim', iterator=True, sep='\s+', header=None)
        self.bfile = bfile_path
        self.iterator = iterator
        
        if type(self.iterator) is SNPIterator or type(self.iterator) is type(None):
            self.static = False
        else:
            raise TypeError('iterator only supports SNPIterator and NoneType.')

    def read(self):
        bim = self.bim.read()
        return bim_setter(bim)

    def to_static(self):
        self.static = True
        self.bim = self.read()

    def reset(self):
        self.bim = pd.read_csv(f'{self.bfile_path}.bim', iterator=True, sep='\s+', header=None)
    
    def get_chunk(self, chunk_size):
        if not self.static:
            bim = self.bim.get_chunk(chunk_size)
            if self.iterator:
                idx = self.iterator.get_chunk(chunk_size)
                idx = np.s_[idx[1],:] # ignore sample index
                return bim_setter(bim).loc[idx]
            else:
                return bim_setter(bim)
        else:
            # Static mode ignores chunk_size information
            return self.bim

    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.static:
            return self.get_chunk(self.iterator.chunk_size)
        else:
            raise AttributeError("Static mode cannot perform iteration. Please use `get_chunk()` to get the static data.")



class GWASDataIterator:
    def __init__(self, bed:BedIterator, bim:BimIterator, fam:FamIterator, cov:CovIterator, iterator:(SampleIterator|SNPIterator)):
        self.__dict__.update(locals())

    @staticmethod
    def SampleWise(bfile_path, iterator:SampleIterator,
                  cov_path=None, pheno_path=None, pheno_name='PHENO1'):
        bed = BedIterator(bfile_path, copy.deepcopy(iterator))
        bim = BimIterator(bfile_path)
        fam = FamIterator(bfile_path, cov_path, pheno_path, pheno_name, copy.deepcopy(iterator))
        cov = CovIterator(bfile_path, cov_path, pheno_path, pheno_name, copy.deepcopy(iterator))
        if iterator.category == 'sample':
            bim.to_static()
        return GWASDataIterator(bed, bim, fam, cov, iterator)

    @staticmethod
    def SNPWise(bfile_path, iterator:SNPIterator,
                  cov_path=None, pheno_path=None, pheno_name='PHENO1'):
        bed = BedIterator(bfile_path, copy.deepcopy(iterator))
        bim = BimIterator(bfile_path, copy.deepcopy(iterator))
        fam = FamIterator(bfile_path, cov_path, pheno_path, pheno_name)
        cov = CovIterator(bfile_path, cov_path, pheno_path, pheno_name)
        if iterator.category == 'snp':
            fam.to_static()
            cov.to_static()
        return GWASDataIterator(bed, bim, fam, cov, iterator)
    
    def reset(self):
        if self.iterator.category == 'sample-snp' and self.iterator.snp_iterator.is_end():
            self.bim.reset()
        elif self.iterator.category == 'snp-sample' and self.iterator.sample_iterator.is_end():
            self.fam.reset()
            self.cov.reset()

    def get_chunk(self, chunk_size):

        if self.iterator.category == ('sample-snp' or 'snp-sample'):
            reset = False
        else:
            reset = True
        
        chunk_bed = self.bed.get_chunk(chunk_size, reset=reset)
        chunk_bim = self.bim.get_chunk(chunk_size)
        chunk_fam = self.fam.get_chunk(chunk_size)
        chunk_cov = self.cov.get_chunk(chunk_size)
        self.reset()
        print('chunk_bed', chunk_bed.shape)
        print('chunk_bim', chunk_bim.shape)
        print('chunk_fam', chunk_fam.shape)
        print('chunk_cov', chunk_cov.shape)
        return GWASData(chunk_bed, chunk_fam, chunk_bim, chunk_cov)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.get_chunk(self.iterator.chunk_size)


def assert_GWASData_is_equal(GWASData1, GWASData2, standard=True):
    # if standard:
    #     GWASData1.standard()
    #     GWASData2.standard()
    GWASData1_properties = vars(GWASData1)
    GWASData2_properties = vars(GWASData2)
    for key in GWASData1_properties.keys():
        v1 = GWASData1_properties[key]
        v2 = GWASData2_properties[key]
        if type(v1) is np.ndarray:
            np.testing.assert_allclose(v1, v2, equal_nan=True)
        elif type(v1) is pd.DataFrame:
            pd.testing.assert_frame_equal(v1, v2)




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
        self.__dict__.update(locals())

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

        self.BED = read_bed(self.bed_path)
        fam = read_fam(self.bed_path)
        self.FAM = format_fam(fam, self.pheno_path, self.pheno_name)
        self.BIM = read_bim(self.bed_path)
        if self.cov_path and os.path.exists(str(self.cov_path)):
            cov = read_cov(self.cov_path) 
            self.COV =  format_cov(cov, self.FAM)
            if self.mean_fill_na_flag:
                self.COV = impute_cov(self.COV)
            
        # filter
        if self.snp_list:
            _, self.snp_idx_list, __ = subset_snps(self.snp_list, self.BIM)
        else:
            self.snp_idx_list = list(self.BIM.index)
        if self.ind_list:
            _, self.ind_idx_list, __ = subset_samples(self.ind_list, self.FAM)
        else:
            self.ind_idx_list = list(self.FAM.index)

        self.ind_idx_list = index_non_missing_samples(self.FAM, self.COV)
        self.FAM, _, __ = subset_samples(self.ind_idx_list, self.FAM, list_is_idx=True)
        self.COV, _, __ = subset_samples(self.ind_idx_list, self.COV, list_is_idx=True)
        self.BIM, _, __ = subset_snps(self.snp_idx_list, self.BIM, list_is_idx=True)
        if self.rename_snp_flag:
            new_snp_id = create_unique_snp_id(self.BIM, to_byte=False, to_dict=False)[0]
            self.BIM['Original_ID'] = self.BIM['ID']
            self.BIM['ID'] = new_snp_id
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
        return create_snp_table(snp_list_ori, snp_list_old)

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
