import subprocess 
import logging
import os
import pandas as pd

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

