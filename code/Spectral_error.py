"""
This file print the relative spectral error for different value of tau and
different size of matrix N
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from h2tools.collections import particles
from h2tools.mcbh import mcbh
from functions import init_particules_problem, init_pos_2D


if __name__ == "__main__":

    
    start = time.time()
    N_vec = [100, 500, 1000, 2500]
    for N in N_vec:
        print("N =", N)
        ndim = 3
        if ndim == 1:
            position = np.linspace(0, 1, N).reshape(ndim, N)
        elif ndim == 2:
            position = init_pos_2D(N)
        elif ndim == 3:
            position = np.random.randn(ndim, N)
        else :
            raise ValueError('The dimension must be 1, 2 or 3')
        
        L = 6
        tau = 1e-3
        block_size = N // (2 ** L)
        func = particles.inv_distance
        problem = init_particules_problem(position, func, block_size=block_size)
        
        

        
        X_err = [(1e-1 ** i) for i in range(1,12)]
        Y_err_dot = []
        Y_err_matvec = []
        for t in X_err :
            print(chr(964), f"= {t}")
            A_h2 = mcbh(problem, tau=t, iters=1, verbose=0)
            A_h2.svdcompress(t)
    
            Y_err_dot.append(A_h2.diffnorm()[0])
        
        print(f"Temps d'exécution : {time.time() - start}")
        
        if N == 2500 :
            plt.loglog(X_err, X_err, ls=':', label='Slope 1')
        plt.loglog(X_err, Y_err_dot, label=f"N={N}", linewidth=2)
        plt.title(f"Erreur relative en norme spectrale, dimension = {ndim}")
        plt.xlabel(r"Valeurs de $\tau$")
        plt.ylabel(r"Valeurs de $|| A - A_{\mathcal{H}^2}||_2$")
        plt.legend()
    plt.grid()
    plt.show()
        
        
