#!/usr/bin/python3
import os
from warnings import warn
from distutils.core import setup
from setuptools import find_packages

REQ_FILE = 'requirements.txt'

if not os.path.exists(REQ_FILE):
      warn("No requirements file found.  Using defaults deps")
      deps = [
            'numpy',
            'pandas', 
            'matplotlib',
            'scipy',
            'tensorflow',
            'pyyaml',
            'pynvml']
      warn(', '.join(deps))
else:
      with open(REQ_FILE, 'r') as f:
            deps = f.read().splitlines()


setup(name='pyutils',
      version='1.0.0',
      description='Python utilities for ease of workflow',
      author='Jason St George',
      author_email='stgeorge@brsc.com',
      packages=find_packages(),
      install_requires=deps)

