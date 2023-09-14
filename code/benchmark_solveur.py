"""
This file plot a benchmark of the different solveur (direct and GMRES)
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from functions import init_particules_problem
from h2tools.mcbh import mcbh
from h2tools.collections import particles
from functions import *


if __name__ == "__main__":

    np.random.seed(314)

    # parameter for H² matrix
    N = 2500
    ndim = 3
    block_size = 50
    func = particles.inv_distance


    tau_vect = [1e-11, 1e-9, 1e-6, 1e-3, 1e-1]
    X = []
    Y_direct = []
    Y_gmres = []
    Y_h2 = []
    for tau in tau_vect:

        if ndim == 1:
            position = np.linspace(0, 1, N).reshape(ndim, N)
        elif ndim == 2:
            position = init_pos_2D(N)
        elif ndim == 3:
            position = np.random.randn(ndim, N)
        else :
            raise ValueError('The dimension must be 1, 2 or 3')


        problem, A = init_particules_problem(position, func, block_size, 
                                            full_matrix=True)
        
        A_h2 = mcbh(problem, tau, iters=1, verbose=0) #H² matrix
        A_h2.svdcompress(tau, verbose=False)
        b = np.random.randn(N) #random vector for linear system

        x = A_h2.solve(b)
        x_gmres = A_h2.solve_gmres(b, eps=1e-6)
        x_ref = np.linalg.solve(A, b)

        err_direct = np.linalg.norm(x_ref - x)
        print('Erreur solveur direct : '+str(err_direct))
        err_gmres = np.linalg.norm(x_ref - x_gmres)
        print('Erreur solveur direct : '+str(err_gmres))
        err_h2 = A_h2.diffnorm()[0]
        print('Erreur solveur direct : '+str(err_h2))

        X.append(tau)

        Y_direct.append(err_direct)
        Y_gmres.append(err_gmres)
        Y_h2.append(err_h2)

    plt.loglog(X, Y_direct, label="Direct", linewidth=2, marker='^', c='b')
    plt.loglog(X, Y_gmres, label="GMRES", linewidth=2, marker='^', c='r')
    plt.loglog(X, Y_h2, label=r"$\mathcal{H}^2$ error", marker='^', c='y', ls=":")

    plt.grid()
    plt.legend()
    plt.title(f"Benchmark solveur for N = {N}")
    plt.xlabel(r"Précision $\tau$")
    plt.ylabel("Relatives error")
    plt.show()
