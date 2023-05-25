"""
This is example of using `h2tools` with MPI parallel clusters. To run
example, simply type something like:

mpiexec -n 4 python example.py
"""


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
# import multicharge approximation method
from h2tools.mcbh import mcbh
from time import time

import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD
if comm.size == 1:
    comm = None
# set dimensionality of the problem (ndim), size of the problem (count),
# desired relative accuracy of far field approximation (tau),
# number of iterations for MCBH algorithm (iters),
# whether to hold special submatrices in memory or compute them on demand (onfly),
# symmetricity of the problem (symmetric),
# maximum size of leaf node of cluster trees (block_size) and
# verbosity (verbose)
ndim = 3
count = 5000
tau = 1e-3
iters = 1
onfly = 0
symmetric = 1
block_size = 20
verbose = 1
random_init = 5

func = particles.inv_distance

#generate data
# set random seed so that results are repeatable
np.random.seed(0)
# generate positions of particles in shape (ndim, count)
position = np.random.randn(ndim, count)
# create data object from given positions of particles
data = particles.Particles(ndim, count, position)
# initialize cluster tree from data object
tree = ClusterTree(data, block_size)
# create main problem object with interaction particles.inv_distance
problem = Problem(func, tree, tree, symmetric, verbose, comm)

# build approximation of the matrix with MCBH algorithm

matrix = mcbh(problem, tau, alpha=0., iters=iters, onfly=onfly, verbose=verbose,
        random_init=random_init, mpi_comm=comm)
np.random.seed(100)

t0 = time()
rel_err = matrix.diffnorm(far_only=1)
t0 = time()-t0
if comm is None or comm.rank == 0:
    print('approximation error:', rel_err, '(computed in', t0, 'seconds)')

x = np.random.randn(matrix.shape[1])
t0 = time()
matrix.dot(x)
t0 = time()-t0
if comm is None or comm.rank == 0:
    print('dot (1 column):', t0, 'seconds')

x = np.random.randn(matrix.shape[1], 100)
t0 = time()
matrix.dot(x)
t0 = time()-t0
if comm is None or comm.rank == 0:
    print('dot (100 columns):', t0, 'seconds')

x = np.random.randn(matrix.shape[0])
t0 = time()
matrix.rdot(x)
t0 = time()-t0
if comm is None or comm.rank == 0:
    print('rdot (1 column):', t0, 'seconds')

x = np.random.randn(matrix.shape[0], 100)
t0 = time()
matrix.rdot(x)
t0 = time()-t0
if comm is None or comm.rank == 0:
    print('rdot (100 columns):', t0, 'seconds')
"""
x = np.random.randn(matrix.shape[1])
t0 = time()
problem.dot(x)
t0 = time()-t0
if comm is None or comm.rank == 0:
    print('1 x problem.dot (1 column):', t0, 'seconds')

x = np.random.randn(matrix.shape[1], 100)
t0 = time()
problem.dot(x)
t0 = time()-t0
if comm is None or comm.rank == 0:
    print('1 x problem.dot (100 column):', t0, 'seconds')

x = np.random.randn(matrix.shape[0])
t0 = time()
problem.rdot(x)
t0 = time()-t0
if comm is None or comm.rank == 0:
    print('1 x problem.rdot (1 column):', t0, 'seconds')

x = np.random.randn(matrix.shape[0], 100)
t0 = time()
problem.rdot(x)
t0 = time()-t0
if comm is None or comm.rank == 0:
    print('1 x problem.rdot (100 column):', t0, 'seconds')

x = np.random.randn(matrix.shape[1])
t0 = time()
problem.far_dot(x)
t0 = time()-t0
if comm is None or comm.rank == 0:
    print('1 x problem.far_dot (1 column):', t0, 'seconds')

x = np.random.randn(matrix.shape[1], 100)
t0 = time()
problem.far_dot(x)
t0 = time()-t0
if comm is None or comm.rank == 0:
    print('1 x problem.far_dot (100 column):', t0, 'seconds')

x = np.random.randn(matrix.shape[0])
t0 = time()
problem.far_rdot(x)
t0 = time()-t0
if comm is None or comm.rank == 0:
    print('1 x problem.far_rdot (1 column):', t0, 'seconds')

x = np.random.randn(matrix.shape[0], 100)
t0 = time()
problem.far_rdot(x)
t0 = time()-t0
if comm is None or comm.rank == 0:
    print('1 x problem.far_rdot (100 column):', t0, 'seconds')

x = np.random.randn(matrix.shape[1], 100)
err = matrix.dot(x)-problem.dot(x)
y = matrix.dot(x)
rx = np.random.randn(matrix.shape[0], 100)
rerr = matrix.rdot(x)-problem.rdot(x)
ry = matrix.rdot(x)
if comm is None or comm.rank == 0:
    print('error of 100-column dot:', np.linalg.norm(err)/np.linalg.norm(y))
    print('error of 100-column rdot:', np.linalg.norm(rerr)/np.linalg.norm(ry))
"""

"""
matrix_seq = mcbh(problem, tau, iters=iters, onfly=onfly, verbose=verbose,
        random_init=random_init, mpi_comm=None)

t0 = time()
rel_err = matrix_seq.diffnorm(matrix, far_only=1)
t0 = time()-t0
if comm is None or comm.rank == 0:
    print('approximation error:', rel_err, '(computed in', t0, 'seconds)')
"""
