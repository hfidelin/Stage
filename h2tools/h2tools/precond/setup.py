#This script will build the main subpackages  

from distutils.util import get_platform 
import sys
from os.path import exists, getmtime
import os

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration, get_info
    config = Configuration('precond', parent_package, top_path)
    #import ipdb; ipdb.set_trace()
    #config.add_library()
    src = ['iluoo.f90','iluoo.pyf']
    config.add_extension('iluoo',sources=src)
    return config
    
#from distutils.core import setup
#from numpy.distutils.core import setup, Extension
#src = ['tt_f90.f90','tt_f90.pyf']
#inc_dir = ['tt-fort']
#lib = ['tt-fort/mytt.a']
#ext = Extension('tt_f90', src, include_dirs=inc_dir)
#setup(ext_modules = [ext])


if __name__ == '__main__':
    print('This is the wrong setup.py to run')

