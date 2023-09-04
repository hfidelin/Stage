"""
This file contains function for initializing easily particules problems
for h2tools.
"""
import numpy as np
from h2tools import ClusterTree
from h2tools import Problem
from h2tools.collections import particles

def init_particules_problem(position, func, block_size, full_matrix=False ):
    """
    Initialise un objet Python "Problem" pour un problème symmetrique 
    provenant d'un nuage de point
    """
    ndim, count = position.shape
    N = count
    data = particles.Particles(ndim, count, vertex=position)
    tree = ClusterTree(data, block_size=block_size)
    problem = Problem(func, tree, tree, symmetric=1, verbose=0)

    if full_matrix:
        row = np.arange(N)
        col = row
        A = problem.func(row, col)
        return problem, A

    else:
        return problem

def init_pos_2D(N):
    """
    Creates a uniform distribution of particles in [0, 1]²
    """

    print("WARNING : The input N must be a perfect square")
    N_x = int(np.sqrt(N))

    x = np.linspace(0, 1, N_x)
    grid = np.meshgrid(x, x)

    position = np.array(grid).reshape(len(x) ** 2, -1).T
    return position
