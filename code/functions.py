import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.sparse.linalg as la
from h2tools import ClusterTree
from h2tools import Problem
from h2tools.collections import particles
from scipy.sparse import csc_matrix,lil_matrix
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

    C0 = lil_matrix((N, N))
    for i in range(problem.row_tree.num_nodes):
        if (problem.row_close[i]):
            for j in problem.row_close[i]:
                row_vect = problem.row_tree.index[i]
                col_vect = problem.row_tree.index[j]
                C0[np.ix_(row_vect, col_vect)] = problem.func(row_vect, col_vect)
    
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


def init_Uk(N, row_basis, Block_size, k):

    diag = []

    B = len(row_basis)
    for i in range(1, B):
        Block = row_basis[i].T
        I, J = Block.shape
        if J == Block_size ** (k + 1):

            max_size = max(Block.shape)
            
            square_block = np.zeros((max_size, max_size))

            square_block[:Block.shape[0], :Block.shape[1]] = Block 

            diag.append(square_block)

    Uk = sp.sparse.block_diag(diag)
    return Uk
        

def init_Vk(N, row_basis, Block_size, k):

    diag = []

    B = len(row_basis)
    for i in range(1, B):
        Block = row_basis[i]
        I, J = Block.shape
        if I == Block_size ** (k + 1):

            max_size = max(Block.shape)
            
            square_block = np.zeros((max_size, max_size))

            square_block[:Block.shape[0], :Block.shape[1]] = Block 

            diag.append(square_block)

    Vk = sp.sparse.block_diag(diag)
    return Vk
        


    #fig.savefig('decomp.png')
    plt.show()


def init_vect_base(problem, list_transfer):
    vect_base = []
    ind_b = 0
    for i in reversed(range(1, problem.row_tree.num_nodes)):
        #print(f"Noeud n°{i}\n")
        
        if (problem.row_far[i]):
            if (problem.row_tree.child[i]):
                
                Base_c1 = vect_base[ind_b + 1]
                Base_c2 = vect_base[ind_b]
                U = lil_matrix((Base_c1.shape[0] + Base_c2.shape[0], Base_c1.shape[1] + Base_c2.shape[1]))
                U[:Base_c1.shape[0], :Base_c1.shape[1]] = Base_c1
                U[-Base_c2.shape[0]:, -Base_c2.shape[1]:] = Base_c2
                
                 
                Base = U @ list_transfer[i]
                ind_b += 2
                
            else:
                Base = list_transfer[i]
               
        
        vect_base.append(Base)

    vect_base = list(reversed(vect_base))        
    vect_base.insert(0, [])
    
    return vect_base
    
def build_A(N, problem, list_row_basis, list_col_basis, list_far):
    M_h2 = np.zeros((N, N))
    for i in range(1, problem.row_tree.num_nodes):
        #print(f"Noeud n°{i}\n")
        if (problem.row_far[i]):
            for j in problem.row_far[i]:
                row_ind = problem.row_tree.index[i]
                col_ind = problem.col_tree.index[j]
                #print(row_basis[i].shape, row_far[i][0].shape, row_basis[j].shape)
                F = list_row_basis[i] @ list_far[i][0] @ list_col_basis[j].T
                M_h2[np.ix_(row_ind, col_ind)] = F

                #F_verif = A[np.ix_(row_ind, col_ind)]
                #print('Erreur locale :', np.linalg.norm(F - F_verif) / np.linalg.norm(F), '\n')
    
    C0 = init_C0(N, problem)
    M_h2 += C0 
    return M_h2


def init_pos_2D(N):

    print("WARNING : The input N must be a perfect square")
    N_x = int(np.sqrt(N))

    x = np.linspace(0, 1, N_x)
    grid = np.meshgrid(x, x)

    position = np.array(grid).reshape(len(x) ** 2, -1).T
    return position

if __name__ == '__main__':
    
    start = time.time()
    N = (2 ** 4) ** 2
    ndim = 1
    if ndim == 1:
        position = np.linspace(0, 1, N).reshape(ndim, N)
    elif ndim == 2:
        position = init_pos_2D(N)
    elif ndim == 3:
        position = np.random.randn(ndim, N)
    else :
        raise ValueError('The dimension must be 1, 2 or 3')
    
    L = 2
    tau = 1e-3
    block_size = N // (2 ** L)
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

    
    """
    A_h2 = mcbh(problem, tau=tau, iters=1, verbose=0)  #Matrice H2
    A_h2.svdcompress(tau = 1e-2)
    
    row_far = A_h2.row_interaction
    col_far = A_h2.col_interaction


    row_transfer = A_h2.row_transfer
    col_transfer = A_h2.col_transfer

    row_basis = init_vect_base(problem, row_transfer)
    col_basis = init_vect_base(problem, col_transfer)
    
    M_h2 = build_A(N, problem, row_basis, col_basis, row_far)

    print(70 * '-', '\n')
    print(f"ERREUR H² : {A_h2.diffnorm()}\n")
    print("Norme de la reconstruction : ", np.linalg.norm(M_h2 - A) / np.linalg.norm(M_h2))
    
    vect_X = []
    vect_spec = []
    vect_recon = []
    eps = 1e-5
    for i in range(7, 14):
        if np.sqrt(2**i) == np.floor(np.sqrt(2**i)):
            N = 2 ** i
            print(f"N = {N}")
            ndim = 2
            if ndim == 1:
                position = np.linspace(0, 1, N).reshape(ndim, N)
            elif ndim == 2:
                position = init_pos_2D(N)
            elif ndim == 3:
                position = np.random.randn(ndim, N)
            else :
                raise ValueError('The dimension must be 1, 2 or 3')
            L = 3
            tau = 1e-8
            block_size = N // (2 ** L)
            func = particles.inv_distance
            problem, L, A = init_particules_problem(position, func, block_size=block_size, 
                                                    full_matrix=True)
            A_h2 = mcbh(problem, tau=tau, iters=1, verbose=0)  #Matrice H2         

            row_far = A_h2.row_interaction
            col_far = A_h2.col_interaction

            row_transfer = A_h2.row_transfer
            col_transfer = A_h2.col_transfer

            row_basis = init_vect_base(problem, row_transfer)
            col_basis = init_vect_base(problem, col_transfer)  

            M_h2 = build_A(N, problem, row_basis, col_basis, row_far) 

            vect_X.append(N)
            vect_spec.append(A_h2.diffnorm())
            vect_recon.append(np.linalg.norm(M_h2 - A) / np.linalg.norm(M_h2))
            print(f"ERREUR H² : {A_h2.diffnorm()}\n")
            print("Norme de la reconstruction : ", np.linalg.norm(M_h2 - A) / np.linalg.norm(M_h2))
            print(70 * '-', '\n')
        
    plt.loglog(vect_X, vect_spec, label="Erreur H2tools")
    plt.loglog(vect_X, vect_recon, label="Erreur rebuild")
    plt.legend()
    plt.grid()
    plt.show()
        
        



    
            
    
    for i in range(3, 14):
        N = 2 ** i
        print(f"N = {N}")
        ndim = 2
        if ndim == 1:
            position = np.linspace(0, 1, N).reshape(ndim, N)
        elif ndim == 2:
            position = init_pos_2D(N)
        elif ndim == 3:
            position = np.random.randn(ndim, N)
        else :
            raise ValueError('The dimension must be 1, 2 or 3')
        L = 3
        tau = 1e-8
        block_size = N // (2 ** L)
        func = particles.inv_distance
        problem, L, A = init_particules_problem(position, func, block_size=block_size, 
                                                full_matrix=True)
        A_h2 = mcbh(problem, tau=tau, iters=1, verbose=0)  #Matrice H2         

        row_far = A_h2.row_interaction
        col_far = A_h2.col_interaction

        row_transfer = A_h2.row_transfer
        col_transfer = A_h2.col_transfer

        row_basis = init_vect_base(problem, row_transfer)
        col_basis = init_vect_base(problem, col_transfer)  

        M_h2 = build_A(N, problem, row_basis, col_basis, row_far) 

        vect_X.append(N)
        vect_spec.append(A_h2.diffnorm())
        vect_recon.append(np.linalg.norm(M_h2 - A) / np.linalg.norm(M_h2))
        print(f"ERREUR H² : {A_h2.diffnorm()}\n")
        print("Norme de la reconstruction : ", np.linalg.norm(M_h2 - A) / np.linalg.norm(M_h2))
        print(70 * '-', '\n')
    
    plt.loglog(vect_X, vect_spec, label="Erreur H2tools")
    plt.loglog(vect_X, vect_recon, label="Erreur rebuild")
    plt.legend()
    plt.grid()
    plt.show()
    """ 
    



   