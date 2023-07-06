from setuptools import setup, Extension
import urllib.request
import zipfile
import os, sys, stat

def platform_plink2_url():
    platform = sys.platform
    if platform == "linux" or platform == "linux2":
        return "https://s3.amazonaws.com/plink2-assets/alpha4/plink2_linux_avx2_20230621.zip"
    elif platform == "darwin":
        return "https://s3.amazonaws.com/plink2-assets/alpha4/plink2_mac_20230621.zip"
    elif platform == "win32":
        return "https://s3.amazonaws.com/plink2-assets/alpha4/plink2_win64_20230621.zip"

# download plink2 executable
plink2_url = platform_plink2_url()
plink2_filename = "plink.zip"
plink2_dst = "gwasprs/bin/"

urllib.request.urlretrieve(plink2_url, plink2_filename)

with zipfile.ZipFile(plink2_filename, 'r') as zip_ref:
    zip_ref.extractall(plink2_dst)

os.remove(plink2_filename)

mode = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
os.chmod(os.path.join(plink2_dst, "plink2"), mode)


setup_args = dict(
    ext_modules = [
        Extension(
            'gwasprs.plink_hwp',
            sources = ['externals/plink_hwp.cc'],
            library_dirs = ['externals'],
        ),
    ],
    package_data = {'gwasprs.plink2': [os.path.join(plink2_dst, "plink2")]},
)
setup(**setup_args)
