import numpy as np
import matplotlib.pyplot as plt
import time
from h2tools.collections import particles
from h2tools import ClusterTree
from h2tools import Problem
from h2tools.mcbh_2 import mcbh


def init_A(position):
    ndim, N = position.shape
    A = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i != j :
                Xi = np.array([position[:, i]])
                Xj = np.array([position[:, j]])
                r = np.linalg.norm(Xi - Xj)
                A[i, j] = 1 / r
    return A

if __name__ == "__main__":

    np.random.seed(0)
    
    start = time.time()
    N = 500
    ndim = 3
    count = N
    position = np.random.randn(ndim, count)

    
    data = particles.Particles(ndim, count, vertex=position)
    tree = ClusterTree(data, block_size=25)
    func = particles.inv_distance
    problem = Problem(func, tree, tree, symmetric=1, verbose=0)
    
    A = init_A(position)
    A_h2 = mcbh(problem, tau=1e-4, iters=0, verbose=0)

    X = np.zeros(N)
    X[0] = 1
    X_err = [(1e-1 ** i) for i in range(1,15)]
    Y_err = []
    res = A.dot(X)
    for t in X_err :
        print(chr(964), f"= {t}")
        A_h2 = mcbh(problem, tau=t, iters=1, verbose=0)
        A_h2.svdcompress(1e-4)
        res_h2 = A_h2.dot(X)
        print(np.linalg.norm(res ), np.linalg.norm(res_h2), "\n")
        err = np.linalg.norm(res - res_h2)
        Y_err.append(err)
    
    print(f"Temps d'exécution : {time.time() - start}")
    
   
    
    plt.plot(X_err, Y_err, c='r')
    plt.title(r"Erreur commise pour $\tilde{y} = \tilde{A} x$ avec $N=5000$")
    plt.xlabel(r"Valeurs de $\tau$")
    plt.ylabel(r"Valeurs de $|| y - \tilde{y}||_2$")
    plt.grid()
    plt.show()

    plt.clf()

    plt.loglog(X_err, Y_err, c ='r')
    plt.title(r"Erreur commise pour $\tilde{y} = \tilde{A} x$ avec $N=5000$")
    plt.xlabel(r"Valeurs de $\tau$")
    plt.ylabel(r"Valeurs de $|| y - \tilde{y}||_2$")
    plt.grid()
    plt.show()
     
    
