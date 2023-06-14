import pandas as pd

from ctypes import c_double, c_int32, c_uint32, cdll, POINTER, c_double



def read_hardy(out_path: str):
    HWE = pd.read_csv(f"{out_path}.hardy", sep = "\s+")
    HWE.AX = HWE.AX.astype(str)
    HWE.A1 = HWE.A1.astype(str)
    HWE["AA"] = HWE.TWO_AX_CT
    HWE["aa"] = HWE.HOM_A1_CT
    HWE.loc[HWE.A1 < HWE.AX, "AA"] = HWE.HOM_A1_CT    
    HWE.loc[HWE.A1 < HWE.AX, "aa"] = HWE.TWO_AX_CT
    ALLELE_COUNT = HWE[["AA", "HET_A1_CT", "aa"]].values
    return ALLELE_COUNT




def setup_plink_hwp(plink_hwp_so_path):
    # hwp_so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plink_hwp.so" )
    hwp_so = cdll.LoadLibrary(plink_hwp_so_path)
    hwp_so.HweP_py.argtypes = [ c_int32, c_int32, c_int32, c_uint32 ]
    hwp_so.HweP_py.restype = c_double
    hwp_so.HweP_vec_py.argtypes = [ POINTER(c_int32), POINTER(c_int32), POINTER(c_int32), POINTER(c_double), c_uint32 ]
    return hwp_so



def cal_hwe_pvalue(HET, HOM1, HOM2, hwp_so):
    pvalue_list = []
    for i in range(len(HET)):
        pvalue = hwp_so.HweP_py(int(HET[i]), int(HOM1[i]), int(HOM2[i]), 0)
        pvalue_list.append(pvalue)
    
    return pvalue_list


def series_tolist(HET, n):
    HET = HET.tolist()
    HET = (c_int32 * n)(*HET)
    return HET

def cal_hwe_pvalue_vec(HET, HOM1, HOM2, hwp_so):
    n = len(HET)
    HET = series_tolist(HET, n)
    HOM1 = series_tolist(HOM1, n)
    HOM2 = series_tolist(HOM2, n)
    pvalue_list = [0.] * n
    pvalue_list = (c_double * n)(*pvalue_list)
    
    hwp_so.HweP_vec_py(HET, HOM1, HOM2, pvalue_list, n, 0)
    pvalue_list = pvalue_list[:]
    return pvalue_list


