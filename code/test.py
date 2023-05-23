# At first, we generate 10000 particles's positions randomly with given
# random seed
import numpy as np
np.random.seed(0)

ndim = 3
count = 100
Y = np.ones(count)
position = np.random.randn(ndim, count)
# We use predefined `particles` submodule of `h2tools.collections`
# This submodule contains data class, cluster division code and
# interaction function

from h2tools.collections import particles
# Create particles data object
data = particles.Particles(ndim, count, vertex=position)
print(data)


# Initialize cluster tree with data object
from h2tools import ClusterTree
tree = ClusterTree(data, block_size=25)


# Use function inv_distance, which returns inverse of distance
# between particles
func = particles.inv_distance
# Create object for whole problem (block cluster tree + function)
from h2tools import Problem
problem = Problem(func, tree, tree, symmetric=1, verbose=0)


# Build approximation of matrix in H^2 format
# with relative accuracy parameter 1e-4,
# 0 (zero) iterations of MCBH algorithm
# and with True verbose flag
from h2tools.mcbh import mcbh
matrix = mcbh(problem, tau=1e-4, iters=0, verbose=0)
"""
print(matrix.dot(Y))
# If you have package `pypropack` installed,
# you can measure relative error of approximation
# in spectral norm
print(f"La différence relative est {matrix.diffnorm()}")
"""