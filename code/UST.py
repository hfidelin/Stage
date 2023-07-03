"""
Fonctions permettant de vérifier (de préférence pour des problèmes linéaire
de petite taille) la validité d'un solveur en passant par un solveur itératif
de Krylov
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from functions import init_particules_problem
from h2tools.mcbh import mcbh
from h2tools.collections import particles
from scipy.sparse import csc_matrix
from functions import *




if __name__ == '__main__':
    
    start = time.time()

    N = 2 ** 3
    ndim = 1
    position = np.linspace(0, 1, N).reshape(ndim, N)
    #position = np.random.randn(ndim, N)

    tau = 1e-2
    block_size = 2
    func = particles.inv_distance
    problem, L, A = init_particules_problem(position, func, block_size=block_size, full_matrix=True)
    
    print(70 * '-')
    print(f"DONNÉES DU PROBLÈME :")
    print(f'\nN \t=\t{N}')
    print(f'\nDIM \t=\t{ndim}')
    print(f'\nB_SIZE \t=\t{block_size}')
    print(f'\nDEPTH \t=\t{L}')
    print(f'\nTAU \t=\t{tau}')
    print(70 * '-', '\n')

    A_h2 = mcbh(problem, tau, iters=1, verbose=0 )
    A_h2.svdcompress(1e-9)
    row_interact = A_h2.row_interaction
    col_interact = A_h2.col_interaction

    row_transfer = A_h2.row_transfer
    col_transfer = A_h2.col_transfer

    row_leaf, col_leaf = init_list_leaf(row_transfer, col_transfer, Block_size=block_size)
    print(row_leaf)
    C0 = init_C0(N, problem)
    U0 = init_U0(N, row_leaf, Block_size=block_size)
    
    print(f"\nTemps d'exécution : {time.time() - start}")

    plt.imshow(U0.todense())
    plt.colorbar()
    plt.show()
    
    #plt.matshow(A_plot)
    #plt.colorbar()
    #plt.show()
    
    
        
