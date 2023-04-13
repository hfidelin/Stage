import numpy as np
import matplotlib.pyplot as plt
import time
from h2tools.collections import particles
from h2tools import ClusterTree
from h2tools import Problem
from h2tools.mcbh_2 import mcbh
from functions import init_particules_problem


if __name__ == "__main__":

    np.random.seed(0)
    
    start = time.time()
    N_vec = [2000, 3000, 4000, 5000]
    for N in N_vec:
        print("N =",N)
        ndim = 3
        count = N
        position = np.random.randn(ndim, count)

        
        func = particles.inv_distance
        problem, A = init_particules_problem(position, func, full_matrix=True)
        
        A_h2 = mcbh(problem, tau=1e-4, iters=0, verbose=0)

        X = np.zeros(N)
        X[0] = 1
        X_err = [(1e-1 ** i) for i in range(1,15)]
        Y_err_dot = []
        Y_err_matvec = []
        res = A.dot(X)
        for t in X_err :
            print(chr(964), f"= {t}")
            A_h2 = mcbh(problem, tau=t, iters=1, verbose=0)
            A_h2.svdcompress(t)
            res_h2_dot = A_h2.dot(X)
            res_h2_matvec = A_h2.matvec(X)
            #print(np.linalg.norm(res ), np.linalg.norm(res_h2), "\n")
            err_dot = np.linalg.norm(res - res_h2_dot)
            err_matvec = np.linalg.norm(res - res_h2_matvec)
            Y_err_dot.append(err_dot)
            Y_err_matvec.append(err_matvec)
        
        print(f"Temps d'exécution : {time.time() - start}")
        
        if N == 2000 :
            plt.loglog(X_err, X_err, ls=':', label='Ordre 1')
        #plt.loglog(X_err, Y_err_dot, label=f"N={N}", linewidth=2)
        plt.loglog(X_err, Y_err_matvec, label=f"N={N}", linewidth=2)
        plt.title(r"Erreur commise pour $\tilde{y} = \tilde{A} x$")
        plt.xlabel(r"Valeurs de $\tau$")
        plt.ylabel(r"Valeurs de $|| y - \tilde{y}||_2$")
        plt.legend()
    plt.grid()
    plt.show()
        
        
