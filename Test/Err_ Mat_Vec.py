import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse.linalg import LinearOperator
from h2tools.collections import particles
from h2tools import ClusterTree
from h2tools import Problem
from h2tools.mcbh_2 import mcbh
from functions import init_particules_problem


if __name__ == "__main__":

    np.random.seed(0)
    
    start = time.time()
    #N_vec = [2000, 3000, 4000, 5000]
    N_vec = [2000]
    for N in N_vec:
        print("N =", N)
        ndim = 1
        count = N
        position = np.random.randn(ndim, count)
        block_size = 25

        
        func = particles.inv_distance
        problem, L, A = init_particules_problem(position, func, block_size, full_matrix=True)
        
        A_h2 = mcbh(problem, tau=1e-4, iters=0, verbose=0)

        X = np.random.randn(N)
        
        X_err = [(1e-1 ** i) for i in range(1,15)]
        Y_err_dot = []
        Y_err_matvec = []
        res = A.dot(X)
        for t in X_err :
            print(chr(964), f"= {t}")
            A_h2 = mcbh(problem, tau=t, iters=1, verbose=0)
            #A_h2.svdcompress(t)
            print('A shape :', A.shape)
            print('Ah2 shape :', A_h2.shape)
            print('Y shape :', X.shape)

            

            res_h2 = A_h2.matvec(X)
            #print(np.linalg.norm(res ), np.linalg.norm(res_h2), "\n")
            err_dot = np.linalg.norm(res - res_h2)
            Y_err_dot.append(err_dot)
        
        print(f"Temps d'exécution : {time.time() - start}")
        
        if N == 2000 :
            plt.loglog(X_err, X_err, ls=':', label='Ordre 1')
        plt.loglog(X_err, Y_err_dot, label=f"N={N}", linewidth=2)
        plt.title(r"Erreur commise pour $\tilde{y} = \tilde{A} x$")
        plt.xlabel(r"Valeurs de $\tau$")
        plt.ylabel(r"Valeurs de $|| y - \tilde{y}||_2$")
        plt.legend()
    plt.grid()
    plt.show()
        
        
