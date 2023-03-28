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

if __name__ == '__main__':
    
    start = time.time()

    N = 50
    ndim = 3
    position = np.random.randn(ndim, N)

    tau = 1e-8


    func = particles.log_distance

    problem, A = init_particules_problem(position, func, block_size=5, full_matrix=True)
    A_h2 = mcbh(problem, tau, iters=1, verbose=0 )
    A_plot = np.zeros((N, N))
    node_ROW = A_h2.row_basis
    node_COL = A_h2.col_basis

    ROW = A_h2.row_interaction
    COL = A_h2.col_interaction

    for row, col in zip(node_ROW, node_COL):
            try:
                for r, c in zip(row, col):
                    print(f"\n{r}\t{c}\n")
                    for r_vec, c_vec in zip(ROW, COL):
                        for a, b in zip(r_vec, c_vec):
                            print(f"\n Sous matrices :\n\n{a}\n{b}\n")
            except:
                 continue
        
            
    print(f"\nTemps d'exécution : {time.time() - start}")
    """
    plt.matshow(A)
    plt.colorbar()
    plt.show()
    """
    
        
