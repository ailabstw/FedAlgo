import pandas as pd
import numpy as np
from typing import List, Tuple
import numpy.typing as npt


from .utils import call_bash_cmd
from .setup import setup_plink2
from .hwe import read_hardy, cal_hwe_pvalue_vec


PLINK2_PATH = setup_plink2()


def cal_qc_client(bfile_path: str, out_path: str, snp_list: List[str],
        het_bin: int, het_range: Tuple[float, float]):
    if len(snp_list) > 0:
        write_snp_list(f"{out_path}.common_snp_list", snp_list)

    calculate_allele_freq_hwe(bfile_path, out_path, snp_list)
    calculate_homo_het_count(bfile_path, out_path, snp_list)

    allele_count = read_hardy(out_path)
    obs_count = get_obs_count(f"{out_path}.vmiss")
    het_hist, het = get_histogram(f"{out_path}.het", het_bin, het_range)

    return allele_count, het_hist, het, obs_count


def qc_stats(bfile_path: str, out_path: str, snp_list: List[str]):
    if len(snp_list) > 0:
        write_snp_list(f"{out_path}.common_snp_list", snp_list)

    calculate_allele_freq_hwe(bfile_path, out_path, snp_list)
    calculate_homo_het_count(bfile_path, out_path, snp_list)

    allele_count = read_hardy(out_path)
    obs_count = get_obs_count(f"{out_path}.vmiss")

    return allele_count, obs_count


def calculate_allele_freq_hwe(bfile_path: str, out_path: str, snp_list: List[str]):
    extract_cmd = f"--extract \"{out_path}.common_snp_list\"" if len(snp_list) > 0 else ""
    cmd0 = f"\"{PLINK2_PATH}\" --bfile \"{bfile_path}\" {extract_cmd} --rm-dup force-first --allow-extra-chr "
    cmd = f"{cmd0} --freq --hardy --missing --out \"{out_path}\" "
    call_bash_cmd(cmd)


def calculate_homo_het_count(bfile_path: str, out_path: str, snp_list: List[str]):
    # read-freq for sample size < 50
    extract_cmd = f"--extract \"{out_path}.common_snp_list\"" if len(snp_list) > 0 else ""
    cmd0 = f"\"{PLINK2_PATH}\" --bfile \"{bfile_path}\" {extract_cmd} --rm-dup force-first --allow-extra-chr "
    cmd = f"{cmd0} --out \"{out_path}\"  --het --read-freq \"{out_path}.afreq\" "
    call_bash_cmd(cmd)


def write_snp_list(filepath, snp_list):
    with open(filepath, "w") as file:
        for i in snp_list:
            file.write(f"{i}\n")


def get_histogram(het_path, bin, range):
    het = pd.read_csv(het_path, sep = r"\s+")
    het_hist, bin_edges = np.histogram(het.F, bins=bin, range=range)
    return het_hist, het.F.values


def get_obs_count(vmiss_path):
    vmiss = pd.read_csv(vmiss_path, sep = r"\s+")
    obs_count = vmiss.OBS_CT.max()
    return obs_count


# aggregator use het to filter ind
def filter_ind(HET, het_mean: float, heta_std: float, HETER_SD: float, sample_list: npt.NDArray[np.byte]):
    HET = np.abs((HET - het_mean ) / heta_std)
    # currently het is not filtered
    remove_idx = np.where(HET > HETER_SD)[0]
    remove_list = [ sample_list[i] for i in remove_idx ]

    return remove_list


# edge create bed after qc
def create_filtered_bed(
        bfile_path: str,
        out_path: str,
        include_snp_list: List[str],
        MIND_QC: float,
        keep_ind_list: List[Tuple[str,str]] = [] ):

    with open(f"{out_path}.ind_list", "w") as FF:
        FF.write("IND\tIND\n")
        for i,j in keep_ind_list:
            FF.write(f"{i}\t{j}\n")

    assert len(include_snp_list) > 0
    with open(f"{out_path}.snp_list", "w") as FF:
        FF.write("SNP\n")
        for i in include_snp_list:
            FF.write(f"{i}\n")

    cmd = f"\"{PLINK2_PATH}\" --allow-extra-chr  --rm-dup force-first  "
    rm_cmd = f"--keep \"{out_path}.ind_list\""
    out, err = call_bash_cmd(f"{cmd} --bfile \"{bfile_path}\" --mind {MIND_QC} --extract \"{out_path}.snp_list\" --out \"{out_path}\" --hardy --make-bed {rm_cmd}")

    return out_path


'''
def ld_prune(bfile_path, out_path, include_snp_list = [], remove_ind_list = []):
    cmd = f"\"{PLINK2_PATH}\" --bfile \"{bfile_path}\" --allow-extra-chr --autosome  --out \"{out_path}\" "
    cmd = cmd + f" --indep-pairwise {PRUNE_WINDOW} {PRUNE_STEP} {PRUNE_THRESHOLD} "

    if len(remove_ind_list) > 0:
        with open(f"{out_path}.ind_list", "w") as FF:
            FF.write("IND\tIND\n")
            for i,j in remove_ind_list:
                FF.write(f"{i}\t{j}\n")
        cmd = cmd + f"--remove \"{out_path}.ind_list\""

    if len(include_snp_list) > 0:
        with open(f"{out_path}.snp_list", "w") as FF:
            FF.write("SNP\n")
            for i in include_snp_list:
                FF.write(f"{i}\n")
        cmd = cmd + f"--extract \"{out_path}.snp_list\""

    out, err = call_bash_cmd(cmd)

    with open(f"{out_path}.QC.prune.in") as FF:
        snp_list = []
        for i in FF.readline():
            snp_list.append(i.rstrip())

    return np.array(snp_list)
'''



def filter_snp (
        ALLELE_COUNT: np.ndarray,
        SNP_ID: np.ndarray,
        SAMPLE_COUNT: int,
        save_path: str,
        GENO_QC: float,
        HWE_QC: float,
        MAF_QC: float,
        ):

    COUNT = ALLELE_COUNT.sum(axis = 1)

    # FREQ
    MAF = (ALLELE_COUNT[:,0]*2 + ALLELE_COUNT[:,1]) / (2*COUNT)

    # HWE
    PVALUE = cal_hwe_pvalue_vec(ALLELE_COUNT[:,1], ALLELE_COUNT[:,0], ALLELE_COUNT[:,2])

    # MISSING
    MISSING = 1 - ( COUNT/SAMPLE_COUNT )

    # SUMMARY
    DA = pd.DataFrame({
        "ID": SNP_ID,
        "MISSING": MISSING,
        "HWE": PVALUE,
        "MAF": MAF,
    })

    DA.loc[DA.MAF>0.5, "MAF"] = 1 - DA.loc[DA.MAF>0.5].MAF
    DA["PASS"] = False
    DA.MISSING = DA.MISSING.round(6)
    DA.MAF = DA.MAF.round(6)
    DA.loc[ (DA.MISSING <= GENO_QC) & (DA.HWE >= HWE_QC) & (DA.MAF >= MAF_QC), "PASS" ] = True
    SNP_ID = DA[DA.PASS].ID.values

    #DA.ID = snp_id_values2id(DA.ID.values)
    DA.to_csv(f"{save_path}", index = False)

    return SNP_ID



def cal_het_sd(HET_HIST: np.ndarray, HET_RANGE: Tuple[float,float], HET_BIN: int):
    bin_edges = np.linspace(HET_RANGE[0], HET_RANGE[1], num = HET_BIN+1)
    margin = bin_edges[1] - bin_edges[0]
    bin_edges = (bin_edges + margin)[:HET_BIN]

    HET_SUM = np.sum(HET_HIST)
    HET_MEAN = np.sum((HET_HIST*bin_edges))/HET_SUM
    HET_SD = np.sqrt(np.sum(((bin_edges - HET_MEAN)**2)*HET_HIST)/(HET_SUM-1))
    return HET_SD, HET_MEAN

