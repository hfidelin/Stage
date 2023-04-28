import numpy as np
import matplotlib.pyplot as plt
from h2tools import ClusterTree
from h2tools import Problem
from h2tools.collections import particles
from scipy.sparse import csc_matrix
from h2tools.mcbh_2 import mcbh
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
        return problem, A

    else:
        return problem


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
        
    vect_row.append(r) 
    vect_column.append(c)
    vect_val.append(val) 



def extract_close(row, col, i, k, tmp_matrix, vect_row, vect_col, vect_val):
    """
    Extraire les sous matrices close dans une matrice sparse
    """
    m = 0
    for r in row.index[i]:
        n = 0
        for c in col.index[k]:
            val = tmp_matrix[m, n]
            add_sp_list(vect_row, vect_col, vect_val, r, c, val)
            n += 1
        m += 1

def extract_far(row, col, i, k, tmp_matrix, vect_row, vect_col, vect_val):
    """
    Extraire les sous matrices far dans une matrice sparse
    """
    M, N = tmp_matrix.shape
    m = 0
    for r in row.index[i]:
        n = 0
        for c in col.index[k]:
            #print(f'Pré m,n : {m,n}')
            if m < M and n < N: 
                #print(f'Post m,n : {m,n}')
                val = tmp_matrix[m, n]
                add_sp_list(vect_row, vect_col, vect_val, r, c, val)
                n += 1
            else:
                pass
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

def init_C0(problem, plot=False):
    """
    Renvoie la matrice close C0
    """

    func = problem.func
    row = problem.row_tree
    row_close = problem.row_close
    col = problem.col_tree
    row_size = row.level[-1]
    
    
    vect_row = []
    vect_col = []
    vect_val = []

    for i in range(row_size):
        for j in range(len(row_close[i])):
            k = row_close[i][j]
                
            tmp_matrix = func(row.index[i], col.index[k])
            extract_close(row, col, i, k, tmp_matrix, 
                              vect_row, vect_col, vect_val)
              
            
    
    C0 = csc_matrix((vect_val, (vect_row, vect_col)), shape=(N, N))
    
    if plot :
        plt.spy(C0.toarray())
        plt.title(f"Squelette de $C_0$ pour $N={N}$")
        plt.show()
    
    return C0

def init_F0(problem, list_far, row_transfer, col_transfer, plot=False):
    """
    Renvoie la matrice far F0
    """

    func = problem.func
    row = problem.row_tree
    row_far = problem.row_far
    col = problem.col_tree
    col_size = row.level[-1]
    
    vect_row = []
    vect_col = []
    vect_val = []

    for i in range(col_size):
        for j in range(len(row_far[i])):
            k = row_far[i][j]
            #V = row_transfer[i][0]
            #W = col_transfer[i][0]
            
            S = list_far[i][0]#func(row.index[i], col.index[k])
            #print(f'V shape :\t{V.shape}')
            #print(f'S shape :\t{S.shape}')
            #print(f'W shape :\t{W.shape}')
            """
            try:
                S = V @ S @ W.T
                print(S)
            except:
                print(False)
            print(20 * '-')
            #print('Pré S =', S)
            #S = V @ S @ W.T
            #print('Post S =',S)
            if S.shape == ():
                S = np.array(S)
            print(S.shape)
            """
            extract_far(row, col, i, k, S, vect_row, vect_col, vect_val)
            
            

            
    
    F0 = csc_matrix((vect_val, (vect_row, vect_col)), shape=(N, N))
    
    if plot :
        plt.spy(F0.toarray())
        plt.title(f"Squelette de matrice far pour $N={N}$")
        plt.show()
    
    return F0
    

def plot_C0_F0(A, C0, F0):
    fig = plt.figure()
    st = fig.suptitle(r'Décomposition $A = C_0 + F_0$', fontsize="x-large")

    ax1 = fig.add_subplot(211)
    ax1.spy(A)
    ax1.set_title(r'Matrice dense $A$')

    ax1 = fig.add_subplot(223)
    ax1.spy(C0.toarray())
    ax1.set_title('Blocs close')
    
    ax1 = fig.add_subplot(224)
    ax1.spy(F0.toarray())
    ax1.set_title('Blocs far')
    
    fig.tight_layout()

    #fig.savefig('decomp.png')
    plt.show()
    



def init_list_leaf(row_transfer, col_transfer, Block_size):
    len_row_transfer = len(row_transfer)
    len_col_transfer = len(col_transfer)

    row_leaf = []
    col_leaf = []

    for i in range(1, len_row_transfer):
        M = col_transfer[i]
        if M.shape == (Block_size, Block_size):
            row_leaf.append(M)

    
    
    for i in range(1, len_col_transfer):
        M = col_transfer[i]
        if M.shape == (Block_size, Block_size):
            col_leaf.append(M)

    return row_leaf, col_leaf


def init_U0(N, row_leaf):
    
    vect_row = []
    vect_col = []
    vect_val = []

    B = len(row_leaf)
    for i in range(B):
        Block = row_leaf[i]
        I, J = Block.shape
        for ii in range(I):
            for jj in range(J):
                val = Block[ii,jj]
                r = i * B_size + ii
                c = i * B_size + jj
                add_sp_list(vect_row, vect_col, vect_val, r, c, val)
        
    U0 = csc_matrix((vect_val, (vect_row, vect_col)), shape=(N, N))
    return U0
        



def init_V0(N, col_leaf):

    vect_row = []
    vect_col = []
    vect_val = []

    B = len(col_leaf)
    for i in range(B):
        Block = row_leaf[i]
        I, J = Block.shape
        for ii in range(I):
            for jj in range(J):
                val = Block[ii,jj]
                r = i * B_size + ii
                c = i * B_size + jj
                add_sp_list(vect_row, vect_col, vect_val, r, c, val)
        
    V0 = csc_matrix((vect_val, (vect_row, vect_col)), shape=(N, N))
    return V0
        


    #fig.savefig('decomp.png')
    plt.show()

if __name__ == '__main__':
    
    start = time.time()
    N = 2 ** 3
    ndim = 1
    #position = np.random.randn(ndim, N)
    position = np.linspace(0, 1, N)
    position = position.reshape((ndim, N))
    tau = 1e-10
    B_size = 2

    func = particles.inv_distance
    problem, A = init_particules_problem(position, func, block_size=B_size, 
                                               full_matrix=True)
    
    A_h2 = mcbh(problem, tau=tau, iters=1, verbose=0)  #Matrice H2
    A_h2.svdcompress(1e-10)

    far = A_h2.col_interaction
    col_transfer = A_h2.col_transfer
    row_transfer = A_h2.row_transfer
    row_basis = A_h2.row_basis
    
    
    row_leaf, col_leaf = init_list_leaf(row_transfer, col_transfer, B_size)

    U0 = init_U0(N, row_leaf)
    V0 = init_V0(N, col_leaf)
    C0 = init_C0(problem)
    A1 = U0.T @ A @ V0
    
    plt.imshow(U0.todense())
    plt.colorbar()
    plt.title('Matrice de décomposition $U0$')
    plt.show()
    