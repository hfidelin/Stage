# main imports and variables

# make this script work with both Python 2 and Python 3
from __future__ import print_function, absolute_import, division

# import all necessary modules and set variables
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
# at first, import predefined functions and dataclass for particles from h2py.collections
from h2tools.collections import particles
# import main container for problem and cluster tree
from h2tools import Problem, ClusterTree

import numpy as np

# set dimensionality of the problem (ndim), size of the problem (count),
# desired relative accuracy of far field approximation (tau),
# number of iterations for MCBH algorithm (iters),
# whether to hold special submatrices in memory or compute them on demand (onfly),
# symmetricity of the problem (symmetric),
# maximum size of leaf node of cluster trees (block_size) and
# verbosity (verbose)
ndim = 2
count = 16
tau = 1e-9
iters = 1
onfly = 0
symmetric = 1
block_size = 2
verbose = 0
random_init = 2

func = particles.inv_distance

#generate data
# set random seed so that results are repeatable
np.random.seed(0)
# generate positions of particles in shape (ndim, count)
position = np.random.randn(ndim, count)
#position = np.linspace(0, 1, count).reshape(ndim, count)
# create data object from given positions of particles
data = particles.Particles(ndim, count, position)
# initialize cluster tree from data object
tree = ClusterTree(data, block_size)
# create main problem object with interaction particles.inv_distance
problem = Problem(func, tree, tree, symmetric, verbose, None)

# import multicharge approximation method
from h2tools.mcbh import mcbh

# build approximation of the matrix with MCBH algorithm

matrix = mcbh(problem, tau, iters=iters, onfly=onfly, verbose=0, random_init=random_init, mpi_comm=None)

# Compress matrix
matrix.svdcompress(1e-2, verbose=0)



for i in range(problem.row_tree.num_nodes):
    T = matrix.row_transfer[i]
    print(f"Noeud n°{i}\n")
    try :
        print(np.linalg.norm(T @ T.T), '\n')
    except:
        print(f"Pas de matrice de transfert\n")
#print(matrix)
