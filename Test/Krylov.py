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
from scipy.sparse.linalg import lgmres, gmres, spsolve, cg
from scipy.sparse import csc_matrix

def solveur_Krylov(A_h2, b, tol):

    x, exitCode = lgmres(A_h2, b, atol=tol)    #Résolution par itération Krylov
    
    print(f"\nConvergence du solveur de Krylov :\n{exitCode == 0}")
        
    return x

def init_mat(N):
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                A[i,j] = 2
            elif (i == j - 1) or (i == j + 1):
                A[i, j] = -1
 #   A[0, 0] = 1
 #   A[0, 1] = 0
 #   A[-1, -1] = 1
 #   A[-1, -2] = 0
    return A



if __name__ == "__main__":

    np.random.seed(0)
    # Initialisation des paramètres du problème
    start = time.time()
    N_vec = [100, 150, 200, 250, 300, 350, 400]
    #N_vec = [50 * i for i in range(1, 33)]
    #N_vec = [1000, 2000, 3000, 4000, 5000]
    #N_vec = [50, 100, 200]
    X = []
    Y_cg = []
    Y_gmres = []
    for N in N_vec:
        print(f"\nN = {N}\n")
        w = 1
        A = init_mat(N)
        #A_s = np.eye(N, N)
        print(f"Déterminant non nul : {np.linalg.det != 0}")
        A_s = csc_matrix(A)

        b = np.ones(N)
        
        x1 = spsolve(A_s, b)

        
        #print(f"\nNorme x1 : {np.linalg.norm(x1)}")
        
        x_cg, exitCode_cg = cg(A_s, b)
        x_gmres, exitCode_gmres = gmres(A_s, b)
        
        #x2, exitCode = lgmres(A_s, b, atol=1e-8)
        err_cg = np.linalg.norm(x1 - x_cg)
        err_gmres = np.linalg.norm(x1 - x_gmres)
        #print(f"\nNorme x2 : {np.linalg.norm(x2)}")
        print(f"CG :\t{exitCode_cg==0}\nGMRES :\t{exitCode_gmres==0} ")
        X.append(N)
        Y_cg.append(err_cg)
        Y_gmres.append(err_gmres)

    plt.title(r"Erreur commise pour $\tilde{A} \tilde{x}= b$")
    plt.xlabel("Nombre d'inconnu $N$")
    plt.ylabel(r"Valeur de $\|\|x - \tilde{x} \|\|_2$")
    plt.loglog(X, Y_cg, c='b', linewidth=2, label='CG')
    plt.loglog(X, Y_gmres, c='r', linewidth=2, label='GMRES')
    plt.loglog(X, X, ls=':', label='Ordre 1')
    plt.grid()
    plt.legend()
    plt.show()

    """
    if N > 400:
        raise Exception("Taille du problème trop grande, veuillez prendre N < 400")
    
    for N in N_vec :
        print(f"\nN = {N}")
        ndim = 3
        count = N
        position = np.random.randn(ndim, count)
        
        #Initialisation du problème H2
        func = particles.inv_distance
        problem, tree, A = init_particules_problem(position, func, full_matrix=True)
        

        #Mise en place des matrices étudiée

        #A = init_A(position)                                #Matrice pleine du problème
        
        #A_h2 = mcbh(problem, tau=1e-5, iters=0, verbose=0)  #Matrice H2
        A_h2 = A
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
    """
    print(f"Temps d'exécution : {time.time() - start}")

        
