import numpy as np
import matplotlib.pyplot as plt
from h2tools import ClusterTree
from h2tools import Problem
from h2tools.collections import particles
from scipy.sparse import csc_matrix
import time

def init_particules_problem(position, func, block_size=25, full_matrix=False ):
    """
    Initialise un objet Python "Problem" pour un problème symmetrique 
    provenant d'un nuage de point
    """
    ndim, count = position.shape
    N = count
    data = particles.Particles(ndim, count, vertex=position)
    tree = ClusterTree(data, block_size= block_size)
    problem = Problem(func, tree, tree, symmetric=1, verbose=0)

    if full_matrix:
        row = np.arange(N)
        col = row
        A = problem.func(row, col)
        return problem, tree, A

    else:
        return problem, tree


def add_sp(vect_row, vect_column, vect_val, r, c, val, N_sp):
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
        
    vect_row[N_sp] = r
    vect_column[N_sp] = c
    print(f"\nVal : {val}")
    vect_val[N_sp] = val
    print(f"{N_sp} : {vect_val}")

def add_sp_list(vect_row, vect_column, vect_val, r, c, val ):
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
        
    vect_row.append(r) 
    vect_column.append(c)
    print(f"\nVal : {val}")
    vect_val.append(val) 



def extract_close(row, col, i, k, tmp_matrix, vect_row, vect_col, vect_val, N_sp):
    """
    Extraire les sous matrices close dans une matrice sparse
    """
    m = 0
    for r in row.index[i]:
        n = 0
        for c in col.index[k]:
            val = tmp_matrix[m, n]
            add_sp_list(vect_row, vect_col, vect_val, r, c, val)
            N_sp += 1
            n += 1
        m += 1

                
    

def init_N_vect(problem):
    
    N_vect = 0

    func = problem.func
    row = problem.row_tree
    row_close = problem.row_close
    col = problem.col_tree
    col_close = problem.col_close
    row_size = row.level[-1]
    col_size = col.level[-1]
    row_close_interaction = [[None for j in row_close[i]]
                for i in range(row_size)]
    
    for i in range(row_size):
        for j in range(len(row_close[i])):
            if row_close_interaction[i][j] is None:
                k = row_close[i][j]
                for r in row.index[i]:
                    for c in col.index[k]:
                        N_vect += 1
    return N_vect

def init_C0(problem, A):
    """
    Renvoie la matrice close C0
    """

    func = problem.func
    row = problem.row_tree
    row_close = problem.row_close
    col = problem.col_tree
    col_close = problem.col_close
    row_size = row.level[-1]
    col_size = col.level[-1]
    row_close_interaction = [[None for j in row_close[i]]
                for i in range(row_size)]
    symmetric = problem.symmetric
    """
    N_vect = init_N_vect(problem)

    vect_row = np.zeros(N_vect)
    vect_col = np.zeros(N_vect)
    vect_val = np.zeros(N_vect)
    """
    N_sp = 0
    
    vect_row = []
    vect_col = []
    vect_val = []

    for i in range(row_size):
        print(f"BOUCLE {i} : {N_sp}")
        for j in range(len(row_close[i])):
            if row_close_interaction[i][j] is None:
                k = row_close[i][j]
                
                tmp_matrix = func(row.index[i], col.index[k])
                #print(f"Sub :\n{tmp_matrix}")
                extract_close(row, col, i, k, tmp_matrix, 
                              vect_row, vect_col, vect_val, N_sp)
            else :
                N_sp += 1   
            
    """
                row_close_interaction[i][j] = tmp_matrix
                if symmetric and k != i:
                    l = row_close[k].index(i)
                    row_close_interaction[k][l] = tmp_matrix.T
        if symmetric:
            col_close_interaction = row_close_interaction
        else:
            col_close_interaction = [[None for j in col_close[i]]
                    for i in range(col_size)]
            for i in range(col_size):
                for j in range(len(col_close[i])):
                    jj = col_close[i][j]
                    k = row_close[jj].index(i)
                    if row_close_interaction[jj][k] is not None:
                        col_close_interaction[i][j] =\
                                row_close_interaction[jj][k].T
    """
    #print(f"Row : {vect_row}")
    #print(f"Column : {vect_col}")
    print(f"Final : {vect_val}")
    
    C0_row = csc_matrix((vect_val, (vect_row, vect_col)), shape=(N, N))
    return C0_row


if __name__ == '__main__':
    start = time.time()
    N = 50
    ndim = 3
    position = np.random.randn(ndim, N)

    tau = 1e-8

    func = particles.inv_distance
    problem, tree, A = init_particules_problem(position, func, block_size=2, 
                                               full_matrix=True)
    

    C0 = init_C0(problem, A)
    #print(f"C0 :\n {C0.toarray()}")
    print(f"\nTemps d'exécution : {time.time() - start}")
    plot = True
    if plot :
        plt.imshow(C0.toarray())
        plt.colorbar()
        plt.show()
    