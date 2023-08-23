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
import pandas as pd
from functions import *




if __name__ == '__main__':
    
    start = time.time()

    N = 2 ** 9
    X = []
    Y = []
    Y_gmres = []
    Y_opti = []
    N_vec = [2 ** i for i in range(8, 13)]
    for N in N_vec:
        X.append(N)
        ndim = 3
        if ndim == 1:
            position = np.linspace(0, 1, N).reshape(ndim, N)
        elif ndim == 2:
            position = init_pos_2D(N)
        elif ndim == 3:
            position = np.random.randn(ndim, N)
        else :
            raise ValueError('The dimension must be 1, 2 or 3')
        
        L = 6
        
        tau = 1e-3
        
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

        tau_compress = 1e-6
        A_h2 = mcbh(problem, tau, iters=1, verbose=0)
        A_h2.svdcompress(tau_compress, verbose=0)

        b = np.ones(N)

        x = A_h2.solve(b)
        x_gmres = A_h2.solve_gmres(b, eps=1e-6)

        M = A_h2.precond(1e-2)
        M = np.eye(N)
        x_opti = A_h2.solve_gmres(b, eps=1e-5, M=M)
        

        x_ref = np.linalg.solve(A, b)

        err = np.linalg.norm(x_ref - x)
        err_gmres = np.linalg.norm(x_ref - x_gmres)
        err_opti = np.linalg.norm(x_ref - x_gmres)
        Y.append(err)
        Y_gmres.append(err_gmres)
        Y_opti.append(err_opti)

        print(f"Conditionnement : {np.linalg.cond(A)}")
    
    plt.loglog(X, Y, label="Direct", linewidth=2, marker='^', c='b')
    plt.loglog(X, Y_gmres, label="GMRES", linewidth=2, marker='^', c='r')
    plt.loglog(X, Y_opti, label="GMRES & Direct", linewidth=2, marker='^', c='g')

    plt.title("Comparatif erreur solveur direct et GMRES")
    plt.xlabel("Nombre d'inconnu $N$")
    plt.ylabel(r"Valeur de $\|x - \tilde{x} \|_2$")    
    plt.ylim((1e-15, 1e4))
    plt.grid()
    plt.legend()
    plt.show()



    #y = np.linalg.solve(S, B)



    """
    




    
    post_row_transfer = A_h2.row_transfer
    post_col_transfer = A_h2.col_transfer




    for i in range(1, problem.row_tree.num_nodes):
        T = post_row_transfer[i]
        norm = np.linalg.norm(T @ T.T)
        print(f"Noeud {i} : \t {norm}")
        #print(f"Noeud n°{i}\n")
        #print(pd.DataFrame(pre_row_transfer[i] @ pre_row_transfer[i].T), '\n')

    row_interact = A_h2.row_interaction
    col_interact = A_h2.col_interaction

    row_transfer = A_h2.row_transfer
    col_transfer = A_h2.col_transfer

    row_basis = init_vect_base(problem, row_transfer)
    col_basis = init_vect_base(problem, col_transfer)

    row_leaf, col_leaf = init_list_leaf(row_transfer, col_transfer, Block_size=block_size)

    k = 0
    
    U0_T = init_Uk(N, row_basis, block_size, k)
    V0 = init_Vk(N, col_basis, block_size, k)
    
    A_linop = LinearOperator((N, N), matvec=A_h2.dot, rmatvec=A_h2.rdot)
    A1 = U0_T @ A_linop.matmat(V0.todense())
    
    for interact in row_interact:
        if (interact):
            print(interact[0], '\n')

    plt.imshow(V0.todense())
    plt.colorbar()
    plt.show()
    """