import urllib.request
import zipfile
import os
import sys
import stat

from setuptools import Extension
from setuptools_cpp import ExtensionBuilder

PLINK2_FILENAME = "plink.zip"
PLINK2_DST = "fedalgo/gwasprs/bin/"
PLINK2_BIN = os.path.join(PLINK2_DST, "plink2")
MODE = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH

def platform_plink2_url():
    platform = sys.platform
    if platform == "linux" or platform == "linux2":
        return "https://s3.amazonaws.com/plink2-assets/alpha5/plink2_linux_avx2_20240105.zip"
    elif platform == "darwin":
        return "https://s3.amazonaws.com/plink2-assets/alpha5/plink2_mac_20240105.zip"
    elif platform == "win32":
        return "https://s3.amazonaws.com/plink2-assets/alpha5/plink2_win64_20240105.zip"
    else:
        raise ValueError(f"Unsupported platform: {platform}")

def download_plink2(plink2_url, plink2_filename, plink2_dst):
    if not os.path.exists(plink2_dst):
        os.makedirs(plink2_dst)
    
    urllib.request.urlretrieve(plink2_url, plink2_filename)
    with zipfile.ZipFile(plink2_filename, 'r') as zip_ref:
        zip_ref.extractall(plink2_dst)
    
    os.remove(plink2_filename)

def build(setup_kwargs):
    if not os.path.exists(PLINK2_BIN):
        download_plink2(platform_plink2_url(), PLINK2_FILENAME, PLINK2_DST)
        os.chmod(PLINK2_BIN, MODE)
    
    setup_kwargs.update(
        {
            "ext_modules": [
                Extension(
                    "fedalgo.gwasprs.plink_hwp",
                    sources=["externals/plink_hwp.cc"],
                    library_dirs=['externals'],
                ),
            ],
            "cmdclass": {"build_ext": ExtensionBuilder},
            "package_data": {
                "fedalgo.gwasprs": ["bin/plink2"]
            }
        }
    )
