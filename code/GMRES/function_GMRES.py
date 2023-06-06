import numpy as np
from h2tools import ClusterTree
from h2tools import Problem
from h2tools.collections import particles
from scipy.sparse.linalg import gmres, LinearOperator
from h2tools.mcbh import mcbh
import time

def init_particules_problem(position, func, block_size, full_matrix=False ):
    """
    Initialise un objet Python "Problem" pour un problème symmetrique 
    provenant d'un nuage de point
    """
    ndim, count = position.shape
    N = count
    tree_size = N
    data = particles.Particles(ndim, count, vertex=position)
    tree = ClusterTree(data, block_size=block_size)
    L = 0
    while tree_size > block_size:
        tree_size = np.ceil(tree_size / 2)
        L += 1 
    #L = tree.num_levels
    problem = Problem(func, tree, tree, symmetric=1, verbose=0)

    if full_matrix:
        row = np.arange(N)
        col = row
        A = problem.func(row, col)
        return problem, L, A

    else:
        return problem, L

def solve_gmres(N, A, b, x_ref, eps, Cond, counter, verbose=False):
    restart = int(N / 2)

    print(" -- Resolution GMRES QTT  --- ")
    tr = time.time()
    x, info = gmres(A, b, tol=eps,restart=restart, maxiter=300 , callback=counter, M=Cond)
    print(f'INFO = {info}')
    tr = time.time() - tr
    err = np.linalg.norm(x-x_ref) #/ np.linalg.norm(x_ref)
    if verbose:
        print("Temps Resolution GMRES QTT ",tr," s")
        print("Iterations = {}".format(counter.niter))
        print("erreur GMRES(QTT) = ", err)
        print(" ")
    return x, err

if __name__ == '__main__':
    
    start = time.time()
    N = 2 ** 3
    ndim = 1
    #position = np.random.randn(ndim, N)
    position = np.linspace(0, 1, N)
    position = position.reshape((ndim, N))
    tau = 1e-9
    block_size = 4

    func = particles.inv_distance
    problem, L, A = init_particules_problem(position, func, block_size=block_size, 
                                               full_matrix=True)
    
    print(70 * '-')
    print(f"DONNÉES DU PROBLÈME :")
    print(f'\nN \t=\t{N}')
    print(f'\nB_SIZE \t=\t{block_size}')
    print(f'\nDEPTH \t=\t{L}')
    print(f'\nTAU \t=\t{tau}')
    print(70 * '-', '\n')
    A_h2 = mcbh(problem, tau=tau, iters=1, verbose=0)  #Matrice H2
    A_h2.svdcompress(tau)
    """
    y_vec =[] 
    for i in range(N):
        c = np.zeros(N)
        c[i] = 1

        y = A_h2.dot(c)
        y_vec.append(y)
    
    A_dot = np.array(y_vec)
    
    print(np.linalg.norm(A - A_dot))
    """
      
  