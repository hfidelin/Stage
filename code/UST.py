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
    N = 2 ** 5
    ndim = 1
    position = np.linspace(0, 1, N).reshape(ndim, N)
    #position = np.random.randn(ndim, N)
    L = 4
    
    tau = 1e-3
    block_size = N // (2 ** L)

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
    for l in reversed(range (1, L + 1)):
        print(f"Level : {L + 1 - l}")
        shape = (2 ** l, 2 ** l)
        print(f"Block shape : {shape}")
        vect_l = []
        for j in range(1, len(row_basis)):
            base = row_basis[j]
            if base.shape[0] == shape[0]:
                vect_l.append(base) 
        U = init_U0(N, vect_l, 2 ** l)
        V = init_V0(N, vect_l, 2 ** l)
        
        vect_U.append(U)
        vect_V.append(V)
    

    A_h2 = la.LinearOperator((N, N), matvec=A_h2.dot, rmatvec=A_h2.rdot)
    
    A1 = vect_U[-1].T @ A_h2.matmat(vect_V[-1].todense())
    plt.imshow(A1)
    plt.colorbar()
    plt.show()
    
    
        
