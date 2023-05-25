#This script will build the main subpackages  
from distutils.util import get_platform 
import sys
from os.path import exists, getmtime
import os

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration, get_info
    config = Configuration('data', parent_package, top_path)
    data = []
    for ext in data:
        src = [ext+'.pyx']
        config.add_extension(ext,sources=src,extra_compile_args=['-undefined,dynamic_lookup'])
    src = ['integ.f90','integ.pyf']
    config.add_extension('integ',sources=src,extra_compile_args=['-undefined,dynamic_lookup']) 
    return config

if __name__ == '__main__':
    print('This is the wrong setup.py to run')
