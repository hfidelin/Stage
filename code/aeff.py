import numpy as np
from scipy.sparse import csc_matrix
import scipy.sparse.linalg as la
from h2tools import ClusterTree
from h2tools import Problem
from h2tools.collections import particles
from scipy.sparse import csc_matrix
from h2tools.mcbh import mcbh
from functions import * 
import time


def plot_err_H2(ndim, vect_N, vect_tau):
    for N in vect_N:
        vect_err = []
        print(70 * '-')
        print(f"N = {N}\n")
        if ndim == 1:
            position = np.linspace(0, 1, N).reshape(ndim, N)
        elif ndim == 2:
            position = init_pos_2D(N)
        elif ndim == 3:
            position = np.random.randn(ndim, N)
        else :
            raise ValueError('The dimension must be 1, 2 or 3')
        
        L = 5
        block_size = N // (2 ** L)
        func = particles.inv_distance
        problem, L, A = init_particules_problem(position, func, block_size=block_size, 
                                                full_matrix=True)
        for tau in vect_tau:
            print(f"tau = {tau}\n")
            A_h2 = mcbh(problem, tau=tau, iters=1, verbose=0)
            vect_err.append(A_h2.diffnorm())
    
        plt.loglog(vect_tau, vect_err, label=f'N = {N}')

    plt.loglog(vect_tau, vect_tau, label="Pente 1", ls = ':')
    plt.grid()
    plt.legend()
    plt.title("Erreur approximation H² en norme spectrale")
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$\|A - A_{H^2}\|$")

def plot_err_matvec(ndim, vect_N, vect_tau):
    for N in vect_N:
        vect_err = []
        print(70 * '-')
        print(f"N = {N}\n")
        if ndim == 1:
            position = np.linspace(0, 1, N).reshape(ndim, N)
        elif ndim == 2:
            position = init_pos_2D(N)
        elif ndim == 3:
            position = np.random.randn(ndim, N)
        else :
            raise ValueError('The dimension must be 1, 2 or 3')
        
        L = 5
        block_size = N // (2 ** L)
        func = particles.inv_distance
        problem, L, A = init_particules_problem(position, func, block_size=block_size, 
                                                full_matrix=True)
        X = np.random.randn(N)
        for tau in vect_tau:
            print(f"tau = {tau}\n")
            A_h2 = mcbh(problem, tau=tau, iters=1, verbose=0)
            y = A_h2.dot(X)
            y_ex = A.dot(X)
            vect_err.append(np.linalg.norm(y - y_ex))
            print(f'Vérif : {abs(y[0] - y_ex[0])}')
    
        plt.loglog(vect_tau, vect_err, label=f'N = {N}')

    plt.loglog(vect_tau, vect_tau, label="Pente 1", ls = ':')
    plt.grid()
    plt.legend()
    plt.title("Erreur approximation produit matrice-vecteur")
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$\|y - y_{H^2}\|$")
        


if __name__ == '__main__':

    vect_N = [10, 20, 30, 40, 50]
    vect_N = [2000, 3000, 4000, 5000]
    vect_N = [100, 500, 1000, 2500]
    vect_tau = [10 **(-i) for i in range(1,9)]

    ndim = 1

    plot_err_matvec(ndim, vect_N, vect_tau)
    plt.show()
    
    