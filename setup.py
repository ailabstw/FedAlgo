from setuptools import setup, Extension
import urllib.request
import zipfile
import os

# download plink2 executable
plink2_url = "https://s3.amazonaws.com/plink2-assets/alpha4/plink2_linux_avx2_20230621.zip"
plink2_filename = "plink2_linux.zip"
plink2_dst = "gwasprs/bin/"

urllib.request.urlretrieve(plink2_url, plink2_filename)

with zipfile.ZipFile(plink2_filename, 'r') as zip_ref:
    zip_ref.extractall(plink2_dst)

os.remove(plink2_filename)


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
