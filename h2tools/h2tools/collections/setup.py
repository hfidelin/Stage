# -*- coding: utf-8 -*-
#This script will build the main subpackages
from __future__ import print_function, absolute_import 
from distutils.util import get_platform 
import sys
from os.path import exists, getmtime
import os

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration, get_info
    config = Configuration('collections', parent_package, top_path)
    
    #config.add_subpackage('particles')
    #config.add_subpackage('pcm')
    #config.add_subpackage('triangular_surface')
    #config.add_subpackage('integral')
    
    return config

if __name__ == '__main__':
    print('This is the wrong setup.py to run')
