import numpy as np
import matplotlib.pyplot as plt
from h2tools.collections import particles
from scipy.sparse.linalg import LinearOperator
from h2tools.mcbh_2 import mcbh
import time
from functions import *

if __name__ == '__main__':
    
    start = time.time()
    N_vec = [1000, 2000, 3000]
    N_vec = [500 * i for i in range(1, 8, 2)]
    for N in N_vec:
        print(f'N = {N}\n')
        block_size = 50
        ndim = 1
        position = np.linspace(0, 1, N)
        position = position.reshape((ndim, N))
        position = np.random.randn(ndim, N)
        func = particles.inv_distance
        problem, L, A = init_particules_problem(position, func, block_size=block_size, 
                                                full_matrix=True)
        X = np.random.randn(N)
        tau_vect = [(1e-1) ** i for i in range(1, 15)]
        err_vect = []
        err_h2_vect = []
        for t in tau_vect:
            print(chr(964), f"= {t}\n")
            A_h2 = mcbh(problem, tau=t, iters=1, verbose=0)
            try:
                A_h2.svdcompress(t)
            except:
                pass
            
            

            Y_ref = A.dot(X)
            Y = A_h2.matvec(X)

            err_h2 = A_h2.diffnorm()
            err = np.linalg.norm(Y_ref - Y)

            err_vect.append(err)
            err_h2_vect.append(err_h2)

            print(60 * '-')

        
        
        plt.loglog(tau_vect, err_vect, label=f'N = {N}')
        if N == 500:
            plt.loglog(tau_vect, tau_vect, ls=':', c='b', label='Slope 1')
        
        plt.legend()
        plt.title(f"Erreur commise pour produit matrice-vecteur 1D ")
        plt.xlabel(r"Valeurs de $\tau$")
        plt.ylabel(r"Valeurs de $|| y - \tilde{y}||_2$")
        
    print(f"Temps d'exécution : {time.time() - start}")
    plt.grid()
    plt.show()