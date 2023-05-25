"""
Python package `h2tools` contains routines to approximate matrices,
arising in different problems with non-local operators (nbody problem,
integral equations).
Main feature of it is MCBH method of approximation by H2-matrices.
"""

from __future__ import print_function, absolute_import, division

__all__ = ['ClusterTree', 'Problem', 'MinimalData']

from .cluster_tree import ClusterTree
from .problem import Problem
from .minimal_data import MinimalData
from .__version__ import __version__
