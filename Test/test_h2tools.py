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
from scipy.sparse.linalg import lgmres



def solveur_Krylov(A_h2, b, tol):

    x, exitCode = lgmres(A_h2, b, atol=tol)    #Résolution par itération Krylov
    
    print(f"\nConvergence du solveur de Krylov :\n{exitCode == 0}")
        
    return x


def err_h2(N_vec):

    for N in N_vec :
        print(f"N = {N}")
        ndim = 3
        count = N
        position = np.random.randn(ndim, count)
    
        #Initialisation du problème H2
        func = particles.inv_distance
        problem = init_particules_problem(position, func, block_size=20, 
                                               full_matrix=False)
    
        X_err = [(1e-1 ** i) for i in range(1,15)]
        Y_err = []
        
        for t in X_err :
            print(f"Tau = {t}")
            A_h2 = mcbh(problem, tau=t, iters=1, verbose=0)  #Matrice H2
            err = A_h2.diffnorm()
            Y_err.append(err)

        plt.loglog(X_err, Y_err, linewidth=2)
        
            
    plt.loglog(X_err, X_err, ls=':', label='Ordre 1')
    plt.grid()
    plt.legend()
    plt.title(f"Erreur en norme opérateur pour N = {N}")
    plt.xlabel(r"Valeur de $\tau$")
    plt.ylabel(r"Erreur $\|\|A-\hat{A} \|\|$")
    plt.show()
    


if __name__ == "__main__":

    np.random.seed(0)
    # Initialisation des paramètres du problème
    start = time.time()
    N = 50
    N_vec = [2000, 3000, 4000, 5000]
    N_vec = [5000]
    err_h2(N_vec)
    """
    ndim = 3
    count = N
    position = np.random.randn(ndim, count)
    
    #Initialisation du problème H2
    func = particles.inv_distance
    problem, tree, A = init_particules_problem(position, func, block_size=2, 
                                               full_matrix=True)
    
    #Mise en place des matrices étudiée

    A_h2 = mcbh(problem, tau=1e-4, iters=1, verbose=0)  #Matrice H2
    
    print(A_h2.diffnorm())
    """

    
    
    print(f"Temps d'exécution : {time.time() - start}")

    
