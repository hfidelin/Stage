import numpy as np
from scipy.sparse import csc_matrix
import scipy.sparse.linalg as la
from h2tools import ClusterTree
from h2tools import Problem
from h2tools.collections import particles
from scipy.sparse import csc_matrix
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


def add_sp_list(vect_row, vect_column, vect_val, r, c, val):
    """
    Paramètre :
        - r, c : (integer) respectivement l'indice ligne et l'indice colonne
        - val : (float) valeur de la matrice sparse en [r, c]
        - vect_row, vect_column, vect_val : (ndarray) vecteurs respectif 
            de r, c et val
        - N_sp : (integer) indice des vecteurs row, column, vect_val 
            où ranger r, c et val

    Ajoute l'indice ligne 'r' dans le vecteur des indices ligne 'row'
    Ajoute l'indice coolonne 'c' dans le vecteur des indices colonne 'column'
    Ajoute la valeur 'val' dans le vecteur valeur 'vect_val'
    """
    val = np.array(val)  
    vect_row.append(r) 
    vect_column.append(c)
    vect_val.append(val) 

# Define a CSC matrix A


if __name__ == '__main__':
    start = time.time()
    N = 1000 
    ndim = 3
    #position = np.linspace(0, 1, N)
    #position = position.reshape(ndim, N)
    position = np.random.randn(ndim, N)

    tau = 1e-6
    block_size = 2

    func = particles.inv_distance
    problem, L, A = init_particules_problem(position, func, block_size=block_size, 
                                               full_matrix=True)
    I, J = A.shape
    print(70 * '-')
    print(f"DONNÉES DU PROBLÈME :")
    print(f'\nN \t=\t{N}')
    print(f'\nDIM \t=\t{ndim}')
    print(f'\nB_SIZE \t=\t{block_size}')
    print(f'\nDEPTH \t=\t{L}')
    print(f'\nTAU \t=\t{tau}')
    print(70 * '-', '\n')
    
    A_h2 = mcbh(problem, tau=tau, iters=1, verbose=0)  #Matrice H2
    #A_h2.svdcompress(tau)
    mv, rmv = A_h2.dot, A_h2.rdot
    A_h2_linop = la.LinearOperator((N, N), matvec=mv, rmatvec=rmv)



    B = csc_matrix(np.eye(N))
    print('\nMatrices initialisées')

    vect_row = []
    vect_col = []
    vect_data = []

    S= np.zeros((I, J))
    
    for i in range(I):
        print(f"i = {i}")
        B_col = B[:, i].toarray()
        C = A_h2.dot(B_col)
        if i == 0:
            S = C
        else:
            S = np.concatenate((S, C), axis=1)
        #print(S,'\n')
        #S[:, i] = C
        
    #print(S)
           
    print('\nErreur commise :\t',np.linalg.norm(S - A))

    print(f"\nTemps d'execution :\t{time.time() - start}")
    