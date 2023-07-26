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
from scipy.sparse.linalg import LinearOperator
import pandas as pd
from functions import *




if __name__ == '__main__':
    
    start = time.time()
    N = 2 ** 9
    ndim = 1
    if ndim == 1:
        position = np.linspace(0, 1, N).reshape(ndim, N)
    elif ndim == 2:
        position = init_pos_2D(N)
    elif ndim == 3:
        position = np.random.randn(ndim, N)
    else :
        raise ValueError('The dimension must be 1, 2 or 3')
    
    L = 2
    
    tau = 1e-1
    
    block_size = N // (2 ** L)

    if block_size == 0 :
        raise ValueError(f"The depth L is is giving a block size of {block_size}")

    func = particles.inv_distance
    problem, L, A = init_particules_problem(position, func, block_size=block_size, 
                                               full_matrix=True)
    
    print(70 * '-')
    print(f"DONNÉES DU PROBLÈME :")
    print(f'\nN \t=\t{N}')
    print(f'\nDIM \t=\t{ndim}')
    print(f'\nB_SIZE \t=\t{block_size}')
    print(f'\nDEPTH \t=\t{L}')
    print(f'\nTAU \t=\t{tau}')
    print(70 * '-', '\n')
    
    A_h2 = mcbh(problem, tau / 10, iters=1, verbose=0)


    print("PRÉ COMPRESSION\n")

    pre_row_transfer = A_h2.row_transfer
    pre_col_transfer = A_h2.col_transfer
    A_h2.svdcompress(tau, verbose=0)

    post_row_transfer = A_h2.row_transfer
    post_col_transfer = A_h2.col_transfer




    for i in range(1, problem.row_tree.num_nodes):
        a = 1
        #print(f"Noeud n°{i}\n")
        #print(pd.DataFrame(pre_row_transfer[i] @ pre_row_transfer[i].T), '\n')

    row_interact = A_h2.row_interaction
    col_interact = A_h2.col_interaction

    row_transfer = A_h2.row_transfer
    col_transfer = A_h2.col_transfer

    row_basis = init_vect_base(problem, row_transfer)
    col_basis = init_vect_base(problem, col_transfer)