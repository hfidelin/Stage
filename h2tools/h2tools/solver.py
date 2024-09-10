import time
import numpy as np
import scipy.sparse as sp
from .h2_to_sparse import convert_h2_to_sparse




#quick class to print gmres evolution during computing

class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))



def sparse_matrix(h2_matrix):
    """
    Compute matrices from the sparse factorisation for direct solver

    Parameter:
    ----------

    h2_matrix : python object 'H2matrix' class

    Return:
    S, U, V :   _ S csc sparse matrix 
                _ U, V csc sparse orthogonal matrices
    """

    row_tree = h2_matrix.problem.row_tree
    level_count = len(row_tree.level)-1

    S, U, V = convert_h2_to_sparse(h2_matrix, check_error=True, show_process=True)

    S = S.res

    U_mat = U[0]
    V_mat = V[0]

    for j in range(level_count - 1):
        U_mat = U_mat.dot(U[j])
        V_mat = V[j].dot(V_mat)
    
    

    return S.tocsc(), U_mat.tocsc(), V_mat.tocsc()

def direct_solver(h2_matrix, b):
    """
    Compute the solution x of the linear system from h2_matrix and the vector b
    using direct method from sparse factorisation

    Parameters:
    ----------

    h2_matrix: python object 'H2matrix' class
    
    b : ndarray
        column vector used as right hand side of the linear system

    Returns:
    ----------

    x : ndarray
        solution of the linear system
    """

    S, U, V = sparse_matrix(h2_matrix)

    Ub = U.dot(b)

    y = sp.linalg.spsolve(S, Ub)

    x = V.dot(y)

    return x


def gmres_solver(h2_matrix, b, eps=1e-5, M=None):
    """
    Compute the solution x of the linear system from h2_matrix and the vector b
    using GMRES algorithm from Scipy

    Parameters:
    ----------

    h2_matrix: python object 'H2matrix' class
    
    b : ndarray
        column vector used as right hand side of the linear system

    eps : float
        tolerance for convergence

    M : sparse matrix, ndarray, LinearOperator
        inverse of the preconditionneur of h2_matrix (see notes from Scipy GMRES)

    Returns:
    ----------

    x : ndarray
        approximation of the solution of the linear system
    """

    count = gmres_counter()
    N = h2_matrix.shape[0]
    restart = int(N / 2)
    x, info = sp.linalg.gmres(h2_matrix, b, tol=eps,restart=restart, maxiter=300, M=M, callback=count )
    return x

"""
def inv_precond(h2_matrix):
    print("Solver appelé")
    S, U, V = sparse_matrix(h2_matrix)

    P = U @ S @ V
    print(f"TYPE DE P : {type(P)}")
    Px = lambda x : sp.linalg.spsolve(P, x)
    P_linop = sp.linalg.LinearOperator(P.shape, Px)
    print("Solveur renvoi effectué")
    return P_linop
"""
