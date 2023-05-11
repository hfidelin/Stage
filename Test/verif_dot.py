import numpy as np
import matplotlib.pyplot as plt
from h2tools.collections import particles
from scipy.sparse.linalg import LinearOperator
from h2tools.mcbh_2 import mcbh
import time
from functions import *

if __name__ == '__main__':
    
    start = time.time()
    N_vec = [100, 200, 500, 1000]
    for N in N_vec:
        block_size = 50
        ndim = 1
        position = np.linspace(0, 1, N)
        position = position.reshape((ndim, N))
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
            A_h2.svdcompress(t)
            
            mv = A_h2.dot
            A_h2_dot = LinearOperator((N, N), matvec=mv)

            Y_ref = A.dot(X)
            Y = A_h2_dot.matvec(X)

            err_h2 = A_h2.diffnorm()
            err = np.linalg.norm(Y_ref - Y)

            err_vect.append(err)
            err_h2_vect.append(err_h2)

            print(60 * '-')

        print(f"Temps d'exécution : {time.time() - start}")
        
        plt.loglog(tau_vect, err_vect, label=f'N = {N}')
        if N == 100:
            plt.loglog(tau_vect, tau_vect, ls=':', c='b', label='Slope 1')
        plt.grid()
        plt.legend()
        plt.title(f"Erreur commise pour produit matrice-vecteur ")
        plt.xlabel(r"Valeurs de $\tau$")
        plt.ylabel(r"Valeurs de $|| y - \tilde{y}||_2$")

        """
        plt.clf()
        
        
        plt.loglog(tau_vect, err_h2_vect, label=f'N = {N}')
        if N == 100:
            plt.loglog(tau_vect, tau_vect, ls=':', c='b', label='Slope 1')
        
        plt.legend()
        plt.title(f"Erreur commise par l'approximation $H^2$")
        plt.xlabel(r"Valeurs de $\tau$")
        plt.ylabel(r"Valeurs de $| A - \hat{A}|$")
        """
    plt.grid()
    plt.show()