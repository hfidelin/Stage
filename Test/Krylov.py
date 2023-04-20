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


class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        #if self._disp:
            #print('iter %3i\trk = %s' % (self.niter, str(rk)))



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


def solve_gmres(N, A, b, x_ref, eps, counter, verbose=False):
    restart = int(N / 2)

    print(" -- Resolution GMRES QTT  --- ")
    tr = time.time()
    x, info = gmres(A, b, tol=eps,restart=restart, maxiter=N, callback=counter)
    tr = time.time() - tr
    err = np.linalg.norm(x-x_ref) #/ np.linalg.norm(x_ref)
    if verbose:
        print("Temps Resolution GMRES QTT ",tr," s")
        print("Iterations = {}".format(counter.niter))
        print("erreur GMRES(QTT) = ", err)
        print(" ")
    return x, err

if __name__ == "__main__":

    np.random.seed(0)
    # Initialisation des paramètres du problème
    start = time.time()
    #N_vec = [100, 150, 200, 250, 300, 350, 400]
    #N_vec = [500 * i for i in range(1, 9)]
    N_vec = [50 * i for i in range(1, 22)]
    #N_vec = [1000, 2000, 3000, 4000, 5000]
    #N_vec = [10, 25, 50, 75, 100]
    #N_vec = [5000]
    X = []
    Y_cg = []
    Y_h2 = []
    Y_full = []
    for N in N_vec:

        print(f"\nN = {N}\n")

        ndim = 1
        position = np.linspace(0, 1, N)
        position = position.reshape((ndim, N))
        tau = 1e-12
        
        func = particles.inv_distance

        problem, A = init_particules_problem(position, func, block_size=2, 
                                               full_matrix=True)
        
        A_h2 = mcbh(problem, tau=tau, iters=1, verbose=0)  #Matrice H2
        
        mv = A_h2.dot

        A_h2 = LinearOperator((N, N), matvec=mv)

        b = np.ones(N)

        count = gmres_counter()

        x_ref = np.linalg.solve(A, b)
        x_gmres, err_gmres_full = solve_gmres(N, A, b, x_ref, eps=tau, counter = count)
        x_h2, err_gmres_h2 = solve_gmres(N, A_h2, b, x_ref, eps=tau, counter = count)
        """
        y_ref = A.dot(x)
        y_h2 = A_h2.matvec(x)
        
        x_full, exitCode = gmres(A, b, tol=1e-8)
        print('Calcul de x_gmres...')
        #x_gmres, exitCode_gmres = gmres(A_h2, b, tol=1e-8)
        #x_ref, exit_ref = cg(A, b, tol=1e-8)
        x_cg, exitCode_full = cg(A_h2, b, tol=1e-8)
        #x2, exitCode = lgmres(A_s, b, atol=1e-8)
        err_cg = np.linalg.norm(x_ref - x_cg)
        print("Calcul de l'erreur")
        err_gmres = np.linalg.norm(x_ref - x_full)
        #err_gmres = np.linalg.norm(x_ref - x_gmres)
        #err_y = np.linalg.norm(y_ref - y_h2)
        #print(f"\nNorme x2 : {np.linalg.norm(x2)}")
        #print(f"CG :\t{exitCode_cg==0}\nGMRES :\t{exitCode_gmres==0} ")
        #print(f"GMRES :\t{exitCode_gmres==0} ")
        """
        X.append(N)
        #Y_cg.append(err_cg)
        #Y_cg.append(err_y)
        Y_full.append(err_gmres_full)
        Y_h2.append(err_gmres_h2)

    print(f"Temps d'exécution : {time.time() - start}")

    plt.title(r"Erreur commise pour $\tilde{A} \tilde{x}= b par GMRES$")
    plt.xlabel("Nombre d'inconnu $N$")
    plt.ylabel(r"Valeur de $\|\|x - \tilde{x} \|\|_2$")
    #plt.loglog(X, Y_cg, c='b', linewidth=2, label='CG')
    plt.loglog(X, Y_full, c='r', linewidth=2, label='Full')
    plt.loglog(X, Y_h2, c='b', linewidth=2, label='H2')
    #plt.loglog(X, Y_cg, c='b', linewidth=2, label='y = Ax')
    plt.loglog(X, X, ls=':', label='Ordre 1')
    plt.grid()
    plt.legend()
    plt.show()

   

    
    

        
