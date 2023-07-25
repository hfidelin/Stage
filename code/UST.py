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
import pandas as pd
from functions import *




if __name__ == '__main__':
    
    start = time.time()
    N = 2 ** 4
    ndim = 1
    if ndim == 1:
        position = np.linspace(0, 1, N).reshape(ndim, N)
    elif ndim == 2:
        position = init_pos_2D(N)
    elif ndim == 3:
        position = np.random.randn(ndim, N)
    else :
        raise ValueError('The dimension must be 1, 2 or 3')
    
    L = 1
    
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

    A_h2 = mcbh(problem, tau, iters=1, verbose=0 )
    A_h2.svdcompress(tau)
    row_interact = A_h2.row_interaction
    col_interact = A_h2.col_interaction

    row_transfer = A_h2.row_transfer
    col_transfer = A_h2.col_transfer

    row_basis = init_vect_base(problem, row_transfer)
    col_basis = init_vect_base(problem, col_transfer)
    
    
    print(70 * '-', '\n')
    
    vect_U = []
    vect_V = []
    vect = [block_size * (2 **l) for l in range(L)]

    
    for l in vect:
        
        shape = (l, l)
        vect_l = []
        for j in range(1, len(row_basis)):
            base = row_basis[j]
            if base.shape[0] == shape[0]:
                vect_l.append(base) 
        U = init_U0(N, vect_l, l)
        V = init_V0(N, vect_l, l) 
        
        vect_U.append(U)
        vect_V.append(V)


    for inter in A_h2.row_interaction:
        print(inter, '\n')

    C0 = init_C0(N, problem)

    U0 = vect_U[0].todense()
    V0 = vect_V[0].todense()
    A1 = U0.T @ A @ V0
    print(70 * '-', '\n')
    plt.imshow(A1)
    plt.colorbar()
    plt.show()
    print(70 * '-', '\n')
    #print(type(A_h2.row_basis))

    """
    
    
    
    
    vect_U = []
    vect_V = []
    for l in reversed(range (1, L + 1)):
        shape = (2 ** l, 2 ** l)
        vect_l = []
        for j in range(1, len(row_basis)):
            base = row_basis[j]
            if base.shape[0] == shape[0]:
                vect_l.append(base) 
        U = init_U0(N, vect_l, 2 ** l)
        V = init_V0(N, vect_l, 2 ** l) 
        
        vect_U.append(U)
        vect_V.append(V)
    
    for mat in vect_U:
        print(pd.DataFrame(mat.todense()), '\n')
    
    """