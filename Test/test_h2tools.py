"""
Fonctions permettant de vérifier (de préférence pour des problèmes linéaire
de petite taille) la validité d'un solveur en passant par un solveur itératif
de Krylov
"""
import numpy as np
import time
from functions import init_particules_problem
from h2tools.mcbh_2 import mcbh
from h2tools.collections import particles
from scipy.sparse.linalg import lgmres



def solveur_Krylov(A_h2, b, tol):

    x, exitCode = lgmres(A_h2, b, atol=tol)    #Résolution par itération Krylov
    
    print(f"\nConvergence du solveur de Krylov :\n{exitCode == 0}")
        
    return x

if __name__ == "__main__":

    np.random.seed(0)
    # Initialisation des paramètres du problème
    start = time.time()
    N = 50
    
    
    ndim = 3
    count = N
    position = np.random.randn(ndim, count)
    
    #Initialisation du problème H2
    func = particles.inv_distance
    problem, A = init_particules_problem(position, func, block_size=10, 
                                         full_matrix=True)
    
    #Mise en place des matrices étudiée

    A_h2 = mcbh(problem, tau=1e-4, iters=0, verbose=0)  #Matrice H2
    
    for t, s in zip(A_h2.row_interaction, A_h2.col_interaction):
        print(f"{t}\t{s}",'\n')
    #print(f"Test :\n{type(A_h2.col_basis)}")


    
    
    print(f"Temps d'exécution : {time.time() - start}")

    
