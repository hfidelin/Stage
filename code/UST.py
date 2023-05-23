"""
Fonctions permettant de vérifier (de préférence pour des problèmes linéaire
de petite taille) la validité d'un solveur en passant par un solveur itératif
de Krylov
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from functions import init_particules_problem
from h2tools.mcbh_2 import mcbh
from h2tools.collections import particles
from scipy.sparse import csc_matrix
from functions import init_C0




if __name__ == '__main__':
    
    start = time.time()

    N = 5
    ndim = 3
    position = np.random.randn(ndim, N)

    tau = 1e-8

    func = particles.inv_distance
    problem, tree, A = init_particules_problem(position, func, block_size=2, full_matrix=True)
    A_h2 = mcbh(problem, tau, iters=1, verbose=0 )
    A_plot = np.zeros((N, N))
    row = np.arange(N)
    x = np.ones(N)
    y = A.dot(x)
    close = A_h2.row_close
    y1 = A_h2.close_dot(x)
    y2 = A_h2.far_dot(x)
    row = A_h2.row_basis
    size = tree.level[-1]
    dumb = init_C0(problem, A)
    print(len(dumb))
    print(f"\nTemps d'exécution : {time.time() - start}")
    
    #plt.matshow(A_plot)
    #plt.colorbar()
    #plt.show()
    
    
        
