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
from scipy.sparse.linalg import lgmres, gmres, spsolve, cg, LinearOperator
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
    #N_vec = [100, 150, 200, 250, 300, 350, 400]
    N_vec = [100 * i for i in range(1, 9)]
    #N_vec = [1000, 2000, 3000, 4000, 5000]
    #N_vec = [10, 25, 50, 75, 100]
    #N_vec = [10, 20]
    X = []
    Y_cg = []
    Y_gmres = []
    for N in N_vec:
        print(f"\nN = {N}\n")
        ndim = 1
        position = np.linspace(0, 1, N)
        position = position.reshape((ndim, N))
        tau = 1e-8
        x = np.random.randn(N)
        func = particles.inv_distance
        problem, A = init_particules_problem(position, func, block_size=2, 
                                               full_matrix=True)
        #print(A)
        A_h2 = mcbh(problem, tau=tau, iters=1, verbose=0)  #Matrice H2
        
        mv = problem.dot

        A_h2 = LinearOperator((N, N), matvec=mv)

        b = np.ones(N)
        print('Calcul du x_exact...')
        x_ref = np.linalg.solve(A, b)

        
        
        print('Calcul de x_gmres...')
        x_gmres, exitCode_gmres = gmres(A_h2, b, tol=1e-8)
        #x_ref, exit_ref = cg(A, b, tol=1e-8)
        #x_cg, exitCode_full = cg(A_h2, b, tol=1e-8)
        #x2, exitCode = lgmres(A_s, b, atol=1e-8)
        #err_cg = np.linalg.norm(x1 - x_cg)
        print("Calcul de l'erreur")
        err_gmres = np.linalg.norm(x_ref - x_gmres)
        #err_y = np.linalg.norm(y_ref - y_h2)
        #print(f"\nNorme x2 : {np.linalg.norm(x2)}")
        #print(f"CG :\t{exitCode_cg==0}\nGMRES :\t{exitCode_gmres==0} ")
        print(f"GMRES :\t{exitCode_gmres==0} ")
        X.append(N)
        #Y_cg.append(err_cg)
        Y_gmres.append(err_gmres)

    plt.title(r"Erreur commise pour $\tilde{A} \tilde{x}= b$")
    plt.xlabel("Nombre d'inconnu $N$")
    plt.ylabel(r"Valeur de $\|\|x - \tilde{x} \|\|_2$")
    #plt.loglog(X, Y_cg, c='b', linewidth=2, label='CG')
    plt.loglog(X, Y_gmres, c='r', linewidth=2, label='GMRES')
    plt.loglog(X, X, ls=':', label='Ordre 1')
    plt.grid()
    plt.legend()
    plt.show()

    
    print(f"Temps d'exécution : {time.time() - start}")

        
