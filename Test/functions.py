import numpy as np
from h2tools import ClusterTree
from h2tools import Problem
from h2tools.collections import particles


def init_particules_problem(position, func, block_size=25, full_matrix=False ):
    """
    Initialise un objet Python "Problem" pour un problème symmetrique 
    provenant d'un nuage de point
    """
    ndim, count = position.shape
    N = count
    data = particles.Particles(ndim, count, vertex=position)
    tree = ClusterTree(data, block_size= block_size)
    problem = Problem(func, tree, tree, symmetric=1, verbose=0)

    if full_matrix:
        row = np.array([i for i in range(N)])
        col = row
        A = problem.func(row, col)
        return problem, A

    else:
        return problem
    