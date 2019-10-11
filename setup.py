from setuptools import setup, find_packages
#import versioneer

DISTNAME = 'libgf2'
LICENSE = 'Apache License, Version 2.0 '
AUTHOR = "Jason M. Sachs"
URL = "https://github.com/dkenward/libgf2"
DOWNLOAD_URL = 'https://github.com/dkenward/libgf2/archive/master.zip'
CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'Environment :: Console',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Cython',
    'Topic :: Scientific/Engineering']

VERSION = 1.0
DESCRIPTION = ("``libgf2``: a Python module for computation in :math:`GF(2^n)`")
LONG_DESCRIPTION = """
This module is intended for exploration of linear
feedback shift registers (LFSRs) and other areas related
to binary Galois fields.
"""

setup(name=DISTNAME,
      maintainer=AUTHOR,
      version=VERSION,
      packages=find_packages(include=['libgf2', 'libgf2.*']),
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      classifiers=CLASSIFIERS,
      platforms='any',
      python_requires='>=3.1*'
      )