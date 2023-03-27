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

if __name__ == "__main__":

    np.random.seed(0)
    # Initialisation des paramètres du problème
    start = time.time()
    N_vec = [100, 150, 200, 250, 300, 350, 400]
    N_vec = [25 * i for i in range(1, 33)]
    X = []
    Y = []
    """
    N = 400
    if N > 400:
        raise Exception("Taille du problème trop grande, veuillez prendre N < 400")
    """
    for N in N_vec :
        print(f"\nN = {N}")
        ndim = 3
        count = N
        position = np.random.randn(ndim, count)
        
        #Initialisation du problème H2
        func = particles.inv_distance
        problem, A = init_particules_problem(position, func, full_matrix=True)
        

        #Mise en place des matrices étudiée

        #A = init_A(position)                                #Matrice pleine du problème
        
        A_h2 = mcbh(problem, tau=1e-5, iters=0, verbose=0)  #Matrice H2
        
        b = np.ones(N)                                      #Vecteur second membre du système linéaire
        
        #Résolution du système linéaire

        tol = 1e-9
        
        x1 = np.linalg.solve(A, b)    #Résolution par Numpy pour vérification
        
        #Vérification 
        try:
            x2 = solveur_Krylov(A_h2, b, tol)
            Y_err = np.linalg.norm(x1 - x2)
            #print(f"\nErreur commise par l'approximation :\n{Y_err}\n")
            X.append(N)
            Y.append(Y_err)
        except:
            continue
    
    plt.title(r"Erreur commise pour $\tilde{A} \tilde{x}= b$")
    plt.xlabel("Nombre de points $N$")
    plt.ylabel(r"Valeur de $\|\|x - \tilde{x} \|\|_2$")
    plt.loglog(X, Y, c='r', linewidth=2)
    plt.grid()
    plt.show()
    
    print(f"Temps d'exécution : {time.time() - start}")

        
