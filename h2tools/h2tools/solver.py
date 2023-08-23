import time
import numpy as np
import scipy.sparse as sp
from .h2_to_sparse import convert_h2_to_sparse




class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))



def sparse_matrix(h2_matrix):


    row_tree = h2_matrix.problem.row_tree
    level_count = len(row_tree.level)-1

    S, U, V = convert_h2_to_sparse(h2_matrix)

    S = S.res

    U_mat = U[0]
    V_mat = V[0]

    for j in range(level_count - 1):
        U_mat = U_mat.dot(U[j])
        V_mat = V_mat.dot(V[j])
    
    

    return S, U_mat, V_mat

def direct_solver(h2_matrix, b):

    S, U, V = sparse_matrix(h2_matrix)

    Ub = U.dot(b)

    y = sp.linalg.spsolve(S, Ub)

    x = V.dot(y)

    return x


def gmres_solver(h2_matrix, b, eps, M=None):
    
    N = h2_matrix.shape[0]
    restart = int(N / 2)
    x, info = sp.linalg.gmres(h2_matrix, b, tol=eps,restart=restart, maxiter=300, M=M )
    return x

def inv_precond(h2_matrix):
    print("Solver appelé")
    S, U, V = sparse_matrix(h2_matrix)

    P = U @ S @ V
    print(f"TYPE DE P : {type(P)}")
    Px = lambda x : sp.linalg.spsolve(P, x)
    P_linop = sp.linalg.LinearOperator(P.shape, Px)
    print("Solveur renvoi effectué")
    return P_linop

