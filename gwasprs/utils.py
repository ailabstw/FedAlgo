import subprocess 
import logging
import os
import pandas as pd
import numpy as np

def truncate(ss, limit = 5 ):
    ss = str(ss)
    if len(ss) > limit:
        ss = ss[:limit]
    return ss


AUTOSOME_LIST = [ i for i in range(1,22) ] + \
    [ str(i) for i in range(1,22) ] + \
    [ f"chr{i}" for i in range(1,22) ] 

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


def call_bash_cmd(command, *args, **kargs):
    PopenObj = subprocess.Popen(command, 
                        stdout = subprocess.PIPE, 
                        stderr = subprocess.PIPE, 
                        preexec_fn = os.setsid,
                        shell = True, 
                        executable = "/bin/bash",
                        *args, **kargs)
    out, err = PopenObj.communicate()
    out = out.decode("utf8").rstrip("\r\n").split("\n")
    err = err.decode("utf8").rstrip("\r\n").split("\n")
    if PopenObj.returncode != 0:
        logging.error(f"command failed")
        logging.error(command)
        for i in err:
            logging.error(i)
        raise RuntimeError
    return out, err


def read_bim(bfile_prefix):
    BIM = pd.read_csv(f"{bfile_prefix}", sep = "\s+", header = None)
    BIM.columns = ["CHR", "ID", "cM", "POS", "A1", "A2"]
    return BIM

def read_fam(bfile_prefix):
    FAM = pd.read_csv(f"{bfile_prefix}", sep = "\s+", header = None)
    FAM.columns = ["FID", "IID", "P", "M", "SEX", "PHENO"]
    return FAM

