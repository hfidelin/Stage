import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as la
from h2tools import ClusterTree
from h2tools import Problem
from h2tools.collections import particles
from scipy.sparse import csc_matrix, csr_matrix
from h2tools.mcbh import mcbh
import time
import pandas as pd

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

def init_C0(N, problem, plot=False):
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

def init_F1(problem, list_far, plot=False):
    """
    Renvoie la matrice far F0
    """

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

            S = list_far[i][0]
            
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
    #print(row_transfer)
    for i in range(1, len_row_transfer):
        M = row_transfer[i]
        #print('Row Transfert :\n',M.shape)
        if M is not None:
            if M.shape == (Block_size, Block_size):
                #print(M, '\n')
                row_leaf.append(M)
        

    
    
    for i in range(1, len_col_transfer):
        M = col_transfer[i]
        #print('Col Transfert :\n',M.shape)
        if M is not None:
            if M.shape == (Block_size, Block_size):
                #print(M, '\n')
                col_leaf.append(M)

    return row_leaf, col_leaf


def init_U0(N, row_leaf, Block_size):
    
    vect_row = []
    vect_col = []
    vect_val = []

    B = len(row_leaf)
    for i in range(B):
        Block = row_leaf[i].T
        #print('BLOCK :\n', Block, '\n')
        I, J = Block.shape
        for ii in range(I):
            for jj in range(J):
                val = Block[ii,jj]
                r = i * Block_size + ii
                c = i * Block_size + jj
                add_sp_list(vect_row, vect_col, vect_val, r, c, val)
        
    U0 = csc_matrix((vect_val, (vect_row, vect_col)), shape=(N, N))
    return U0
        

def init_V0(N, col_leaf, Block_size):

    vect_row = []
    vect_col = []
    vect_val = []

    B = len(col_leaf)
    for i in range(B):
        Block = col_leaf[i]
        I, J = Block.shape
        for ii in range(I):
            for jj in range(J):
                val = Block[ii,jj]
                r = i * Block_size + ii
                c = i * Block_size + jj
                add_sp_list(vect_row, vect_col, vect_val, r, c, val)
        
    V0 = csc_matrix((vect_val, (vect_row, vect_col)), shape=(N, N))
    return V0
        


    #fig.savefig('decomp.png')
    plt.show()

if __name__ == '__main__':
    
    start = time.time()
    N = 2 ** 3
    ndim = 1
    position = np.linspace(0, 1, N).reshape(ndim, N)
    #position = np.random.randn(ndim, N)

    tau = 1e-1
    block_size = 2

    func = particles.inv_distance
    problem, L, A = init_particules_problem(position, func, block_size=block_size, 
                                               full_matrix=True)
    
    print(70 * '-')
    print(f"DONNÉES DU PROBLÈME :")
    print(f'\nN \t=\t{N}')
    print(f'\nDIM \t=\t{ndim}')
    print(f'\nB_SIZE \t=\t{block_size}')
    print(f'\nDEPTH \t=\t{L}')
    print(f'\nTAU \t=\t{tau}')
    print(70 * '-', '\n')
    
    A_h2 = mcbh(problem, tau=tau, iters=1, verbose=0)  #Matrice H2
    
    M_h2 = np.zeros(A_h2.shape)
    
    row_far = A_h2.row_interaction
    col_far = A_h2.col_interaction


    row_transfer = A_h2.row_transfer
    col_transfer = A_h2.col_transfer

    row_basis, col_basis = init_list_leaf(row_transfer, col_transfer, Block_size=block_size)
    
    print(70 * '-', '\n')
    #print(row_transfer[3],'\n')
    #print(row_transfer[4],'\n')
    t = 0
    for mat in row_transfer:
        print(f"T{t} :\n")
        print(mat, '\n')
        t += 1

    V3 = row_transfer[3]
    V4 = row_transfer[4]
    V5 = row_transfer[5]
    V6 = row_transfer[6]
    
    #print(V3.shape, row_transfer[3].shape)
    V1 = V3 @ row_transfer[3] + V4 @ row_transfer[4]
    V2 = V5 @ row_transfer[5] + V6 @ row_transfer[6]
    print(70 * '-', '\n')
    df = pd.DataFrame(A)
    #print(df)
    """
    for mat in row_far:
        if len(mat) != 0:
            print(mat[0],'\n')
    print(70 * '-', '\n')
    for i in range(1, problem.row_tree.num_nodes):
        print(f"Noeud n°{i}\n")
        if (problem.row_far[i]):
            for j in problem.row_far[i]:

                T_r = row_transfer[i]
                print(T_r, '\n')
                S = row_far[i][0]
                T_c = col_transfer[j]
                print(f"Shape du produit : {T_r.shape}, {S.shape}, {T_c.T.shape}")
                F = T_r @ S @ T_c.T

                row_vect = problem.row_tree.index[i]
                col_vect = problem.row_tree.index[j]
                print(f"Index ligne : \t{row_vect}")
                print(f"Index colonne :\t{col_vect}")
                row_vect = list(row_vect)
                col_vect = list(col_vect)
                F_verif = A[np.ix_(row_vect, col_vect)]
                #M_h2[np.ix_(row_vect, col_vect)] = F
                try:
                    print(np.linalg.norm(F - F_verif))
                except:
                    print(f"ECHEC : {len(row_vect), len(col_vect)} pour {F.shape}")
                df = pd.DataFrame(F_verif)
                #print(df)
        print(70 * '-', '\n')

    #print(f"Erreur norme : {np.linalg.norm(A - M_h2)}")
    
    plt.imshow(M_h2)
    plt.colorbar()
    #plt.show()
    """

    