
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: lucidmode                                                                                  -- #
# -- Description: A Lightweight Framework with Transparent and Interpretable Machine Learning Models     -- #
# -- setup.py: python script with setup directives functions                                             -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/lucidmode/lucidmode                                                  -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

#! /usr/bin/env python

""" A Lucid Framework for Transparent and Interpretable Machine Learning Models. """

from setuptools import setup, find_packages
import lucidmode

DESCRIPTION = __doc__
VERSION = lucidmode.__version__

setup(name='lucidmode',
      version=VERSION,
      description=DESCRIPTION,
      long_description=open("README.rst").read(),
      classifiers=['Development Status :: 0.4 - Beta',
                   'Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'Topic :: Software Development',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: Unix',
                   'Operating System :: MacOS',
                   'Programming Language :: Python :: 3.6',
                   'Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8'],
      author='IFFranciscoME',
      author_email='if.francisco.me@gmail.com',
      url='https://https://github.com/lucidmode/lucidmode',
      license='GPL-3.0 License',
      packages=find_packages(exclude=['*.tests',
                                      '*.tests.*']),
      zip_safe=False,
      package_data={'': ['LICENSE']},
      install_requires=['pandas>=1.1.4', 'numpy>=1.20'])
