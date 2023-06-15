from setuptools import setup, Extension

setup_args = dict(
    ext_modules = [
        Extension(
            'gwasprs.plink_hwp',
            sources = ['externals/plink_hwp.cc'],
            library_dirs = ['externals'], 
        )
    ]
)
setup(**setup_args)


