import numpy as np
import subprocess 
import logging

def truncate(ss, limit = 5 ):
    ss = str(ss)
    if len(ss) > limit:
        ss = ss[:limit]
    return ss



def rename_snp(da, to_byte = True, to_dict = False):
    da = da.copy().reset_index(drop = True)
    snp_list = []
    for i in range(len(da.index)):
        
        CHR = str(da.CHR[i])
        POS = str(da.POS[i])
        #SNP_ID = str(da.ID[i])
        A0 = truncate(da.A1[i])
        A1 = truncate(da.A2[i])
        allele_list = [A0, A1]
        allele_list.sort()
        
        snp = f"{CHR}:{POS}:" + allele_list[0] + ":" + allele_list[1]
        snp_list.append(snp)
    
    if to_byte:
        snp_list = np.array(snp_list, dtype="S")
        
    if to_dict:
        snp_list = dict(zip(snp_list, da.ID))

    return snp_list

