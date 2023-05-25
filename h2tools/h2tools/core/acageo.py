from time import time
import numpy as np
from .maxvolpy.maxvol import maxvol_svd, maxvol_qr
import copy
from sys import getsizeof
from math import pi
import ipdb

def cheb_points(n):
    angles = pi/(n-1)*np.arange(n)
    return np.cos(angles)

class Factor(object):
    """Class Factor"""
    def __new__(cls, *args, **kwargs):
        """__new__ docstring"""
        x = super(Factor, cls).__new__(cls)
        x._status = None
        return x

    def __init__(self, dtype, func, row_data, row_tree, col_data, col_tree, queue, tau, onfly=False, verbose=False, symmetric=False, rect_maxvol_tol = 10, num_points = 5):
        """Docstring for mcbh.Factor.__init__"""
        time0 = time()
        self.dtype = dtype
        self.func = func
        self.row_data = row_data
        self.row_tree = row_tree
        self.col_data = col_data
        self.col_tree = col_tree
        self.queue = queue
        self.tau = tau
        self.symmetric = symmetric
        self.rect_maxvol_tol = rect_maxvol_tol
        self._totaltime = 0
        self._functime = 0
        self._funccalls = 0
        self._elemscomputed = 0
        self._maxvoltime = 0
        self._zero_iteration(num_points)
        self._main_iteration()
        if not onfly:
            self.__matrix()
            self._status = 'fullmem mcbh'
        else:
            self._status = 'lowmem mcbh'
        self._totaltime = time()-time0
        if verbose:
            print('Function calls:', self._funccalls)
            print('Function values computed:', self._elemscomputed)
            print('Function time:', self._functime)
            print('Average time per function value:', self._functime/self._elemscomputed)
            print('Maxvol time:', self._maxvoltime)
            print('Total MCBH time:', self._totaltime)

    @staticmethod
    def reduce_basis(tree, index, num_points):
        #pdb.set_trace()
        if index.size == 0:
            return np.ndarray(0, dtype = np.int32)
        tmp_vertex = tree.data.vertex[:, index]
        center = tmp_vertex.mean(axis=1)
        tmp_vertex -= center.reshape(-1, 1)
        U, S, V = np.linalg.svd(tmp_vertex.dot(tmp_vertex.T))
        tmp_vertex = U.T.dot(tmp_vertex)
        bbox = np.array([tmp_vertex.min(axis=1), tmp_vertex.max(axis=1)])
        rel_points = np.cos(pi/(num_points-1)*np.arange(num_points))
        coef_value = 0.5*np.array([rel_points+1, 1-rel_points]).T
        tmp_basis_points = coef_value.dot(bbox)
        tmp_basis = np.ndarray(num_points**3, dtype=np.int32)
        for j1 in range(num_points):
            for j2 in range(num_points):
                for j3 in range(num_points):
                    cur_item = j1*num_points**2+j2*num_points+j3
                    center = np.array([[tmp_basis_points[j1, 0], tmp_basis_points[j2, 1], tmp_basis_points[j3, 2]]]).T
                    tmp_norms = np.linalg.norm(tmp_vertex-center, axis=0)
                    max_norm = np.max(tmp_norms)
                    tmp_norms[tmp_basis[:cur_item]] = max_norm
                    tmp_basis[cur_item] = np.argmin(tmp_norms)
        return index[tmp_basis]
#
    def _zero_iteration(self, num_points):
        row = self.row_tree
        col = self.col_tree
        sym = self.symmetric
        row_size = row.level[-1]
        self.row_prebasis = [np.ndarray(0, dtype = np.int32) for i in range(row_size)]
        self.row_basis = [np.ndarray(0, dtype = np.int32) for i in range(row_size)]
        if not sym:
            col_size = col.level[-1]
            self.col_prebasis = [np.ndarray(0, dtype = np.int32) for i in range(col_size)]
            self.col_basis = [np.ndarray(0, dtype = np.int32) for i in range(col_size)]
        else:
            col_size = row_size
            self.col_prebasis = self.row_prebasis
            self.col_basis = self.row_basis
        for i in range(len(row.level)-3, -1, -1):
            for j in range(row.level[i], row.level[i+1]):
                if not row.notransition[j]:
                    self.row_basis[j] = self.reduce_basis(row, row.index[j], num_points)
            if not sym:
                for j in range(col.level[i], col.level[i+1]):
                    if not col.notransition[j]:
                        self.col_basis[j] = self.reduce_basis(col, col.index[j], num_points)
            for j in range(row.level[i], row.level[i+1]):
                if not row.notransition[j]:
                    tmp_index = [self.reduce_basis(row, self.row_prebasis[row.parent[j]], num_points)]
                    for k in row.far[j]:
                        tmp_index.append(self.col_basis[k])
                    self.row_prebasis[j] = np.hstack(tmp_index)
            if not sym:
                for j in range(col.level[i], col.level[i+1]):
                    if not col.notransition[j]:
                        tmp_index = [self.reduce_basis(col, self.col_prebasis[col.parent[j]], num_points)]
                        for k in col.far[j]:
                            tmp_index.append(self.row_basis[k])
                        self.col_prebasis[j] = np.hstack(tmp_index)

    def _main_iteration(self):
        row = self.row_tree
        col = self.col_tree
        sym = self.symmetric
        row_size = row.level[-1]
        col_size = col.level[-1]
        self.row_coef = [np.ndarray((0, 0), dtype = np.float32) for i in range(row_size)]
        if not sym:
            self.col_coef = [np.ndarray((0, 0), dtype = np.float32) for i in range(col_size)]
        else:
            self.col_coef = self.row_coef
        for i in range(row.level[-1]-1, -1, -1):
            if not row.notransition[i]:
                tmp_basis = [self.row_basis[i]]
                if len(row.child[i]) == 0:
                    tmp_basis.append(row.index[i])
                else:
                    for j in row.child[i]:
                        tmp_basis.append(self.row_basis[j])
                if len(row.far[i]) == 0:
                    self.row_basis[i] = np.hstack(tmp_basis[1:])
                    self.row_coef[i] = np.ndarray((0, self.row_basis[i].size), dtype = np.float32)
                    continue
                time0 = time()
                tmp_matrix = self.func(np.hstack(tmp_basis), self.row_prebasis[i])
                self._functime += time()-time0
                self._funccalls += 1
                self._elemscomputed += tmp_matrix.size
                time0 = time()
                tmp_piv, tmp_C = maxvol_svd(tmp_matrix, self.tau, top_k_index = self.row_basis[i].size, job = 'R')
                self._maxvoltime += time()-time0
                self.row_coef[i] = tmp_C[self.row_basis[i].size:].copy()
                self.row_basis[i] = self.row_basis[i][tmp_piv]
        if not sym:
            for i in range(col.level[-1]-1, -1, -1):
                if not col.notransition[i]:
                    tmp_basis = [self.col_basis[i]]
                    if len(col.child[i]) == 0:
                        tmp_basis.append(col.index[i])
                    else:
                        for j in col.child[i]:
                            tmp_basis.append(self.col_basis[j])
                    if len(col.far[i]) == 0:
                        self.col_basis[i] = np.hstack(tmp_basis[1:])
                        self.col_coef[i] = np.ndarray((0, self.col_basis[i].size), dtype = np.float32)
                        continue
                    time0 = time()
                    tmp_matrix = self.func(self.col_prebasis[i], np.hstack(tmp_basis))
                    self._functime += time()-time0
                    self._funccalls += 1
                    self._elemscomputed += tmp_matrix.size
                    time0 = time()
                    tmp_piv, tmp_C = maxvol_svd(tmp_matrix, self.tau, top_k_index = self.col_basis[i].size, job = 'C')
                    self._maxvoltime += time()-time0
                    self.col_coef[i] = tmp_C[self.col_basis[i].size:].copy()
                    self.col_basis[i] = self.col_basis[i][tmp_piv]

    def __matrix(self):
        """Technical procedure, computes interaction matrices"""
        row = self.row_tree
        row_data = self.row_data
        col = self.col_tree
        col_data = self.col_data
        row_basis = self.row_basis
        col_basis = self.col_basis
        row_size = row.level[-1]
        col_size = col.level[-1]
        self.row_comp_far_matrix = [[] for i in range(row_size)]
        self.col_comp_far_matrix = [[] for i in range(col_size)]
        # Loop os query-independant
        for i in range(row_size):
            for j in row.far[i]:
                time0 = time()
                tmpmatrix = self.func(row_basis[i], col_basis[j])
                self._functime += time()-time0
                self._funccalls += 1
                self._elemscomputed += tmpmatrix.size
                self.row_comp_far_matrix[i].append(tmpmatrix)
                del tmpmatrix
        for i in range(col_size):
            for j in col.far[i]:
                k = row.far[j].index(i)
                self.col_comp_far_matrix[i].append(self.row_comp_far_matrix[j][k].T)

    @staticmethod
    def __dot_up(tree, transfer, x):
        """Technical procedure, computes basises 'weights' from all the 'particles', from bottom of the tree to top, SINGLE-CORE version"""
        size = tree.level[-1]
        level_count = len(tree.level)-1
        node_weight = [np.zeros((0, x.shape[1]), dtype = x.dtype) for i in range(size)]
        # Loop is query-dependant
        for i in range(level_count-1):
            for j in range(tree.level[level_count-i-2], tree.level[level_count-i-1]):
                if tree.notransition[j]:
                    continue
                if len(tree.child[j]) is 0:
                    tmp = x[tree.index[j]]
                else:
                    tmp = []
                    for k in tree.child[j]:
                        tmp.append(node_weight[k])
                    tmp = np.vstack(tmp)
                if transfer[j].shape[0] is 0:
                    node_weight[j] = tmp
                else:
                    node_weight[j] = transfer[j].T.dot(tmp)
        return node_weight

    @staticmethod
    def __dot_interact(tree, matrix, node_weight):
        """Technical procedure, computes basises 'potentials' from basises 'weights', uses precomputed interaction matrices, SINGLE-CORE version"""
        size = tree.level[-1]
        tmp = node_weight[-1]
        node_answer = [np.ndarray((0, tmp.shape[1]), dtype = tmp.dtype) if len(tree.far[i]) is 0 else np.zeros((matrix[i][0].shape[0], tmp.shape[1]), dtype = tmp.dtype) for i in range(size)]
        for i in range(size):
            if tree.notransition[i]:
                continue
            tmp = node_answer[i]
            for j in range(len(tree.far[i])):
                tmp += matrix[i][j].dot(node_weight[tree.far[i][j]])
        return node_answer

    @staticmethod
    def __dot_interact_onfly(tree, func, basis0, basis1, node_weight, T = False):
        """Technical procedure, computes basises 'potentials' from basises 'weights', computes interaction matrices on the fly, SINGLE-CORE version"""
        size = tree.level[-1]
        tmp = node_weight[-1]
        node_answer = [np.ndarray((0, tmp.shape[1]), dtype = tmp.dtype) if len(tree.far[i]) is 0 else np.zeros((func(basis0[i], basis1[tree.far[i][0]]).shape[0], tmp.shape[1]), dtype = tmp.dtype) for i in range(size)]
        for i in range(size):
            if tree.notransition[i]:
                continue
            tmp = node_answer[i]
            if T:
                for j in range(len(tree.far[i])):
                    tmp += func(basis1[tree.far[i][j]], basis0[i]).T.dot(node_weight[tree.far[i][j]])
            else:
                for j in range(len(tree.far[i])):
                    tmp += func(basis0[i], basis1[tree.far[i][j]]).dot(node_weight[tree.far[i][j]])
        return node_answer
    
    @staticmethod
    def __dot_down(tree, transfer, node_answer):
        """Techical procedure, computes all 'potentials' from basises 'potentials', from top of the tree to bottom, SINGLE-CORE version"""
        size = tree.level[-1]
        level_count = len(tree.level)-1
        tmp = node_answer[-1]
        dtype = tmp.dtype
        nrhs = tmp.shape[1]
        answer = np.zeros((tree.data.count, nrhs), dtype = dtype)
        for i in range(level_count-1):
            for j in range(tree.level[i], tree.level[i+1]):
                if tree.notransition[j]:
                    continue
                if node_answer[j].shape[0] is 0:
                    node_answer[j] = np.zeros((transfer[j].shape[1], nrhs), dtype = dtype)
        for i in range(level_count-1):
            for j in range(tree.level[i], tree.level[i+1]):
                if tree.notransition[j]:
                    continue
                if transfer[j].shape[0] is 0:
                    tmp = node_answer[j]
                else:
                    tmp = transfer[j].dot(node_answer[j])
                if len(tree.child[j]) is 0:
                    answer[tree.index[j]] = tmp
                else:
                    i1 = 0
                    for k in tree.child[j]:
                        i2 = i1 + node_answer[k].shape[0]
                        node_answer[k] += tmp[i1:i2]
                        i1 = i2
        return answer

    def dot(self, x0, dasha_debug=False):
        """Computes 'far h2matrix'-'vector' dot products"""
        if x0.shape[0] != self.col_data.count:
            raise ValueError('operands could not be broadcast together with shapes ({0:d}) ({1:d})').format(self.col_data.count, x0.shape[0])
        if x0.ndim is 1:
            x = x0.reshape(-1, 1)
        else:
            x = x0
        node_weight = self.__dot_up(self.col_tree, self.col_coef, x)
        if self._status == 'lowmem mcbh':
            node_answer = self.__dot_interact_onfly(self.row_tree, self.func, self.row_basis, self.col_basis, node_weight)
        else:
            node_answer = self.__dot_interact(self.row_tree, self.row_comp_far_matrix, node_weight)
        if dasha_debug:
            self.node_weight = node_weight
            self.node_answer = node_answer
        answer = self.__dot_down(self.row_tree, self.row_coef, node_answer)
        if x0.ndim is 1:
            answer = answer.reshape(-1)
        return answer

    def rdot(self, x0, dasha_debug=False):
        """Computes 'vector'-'far h2matrix' dot products"""
        if x0.shape[0] != self.row_data.count:
            raise ValueError('operands could not be broadcast together with shapes ({0:d}) ({1:d})').format(self.row_data.count, x0.shape[0])
        if x0.ndim is 1:
            x = x0.reshape(-1, 1)
        else:
            x = x0
        nrhs = x.shape[1]
        node_weight = self.__dot_up(self.row_tree, self.row_coef, x)
        if self._status == 'lowmem mcbh':
            node_answer = self.__dot_interact_onfly(self.col_tree, self.func, self.col_basis, self.row_basis, node_weight, 1)
        else:
            node_answer = self.__dot_interact(self.col_tree, self.col_comp_far_matrix, node_weight)
        if dasha_debug:
            self.node_weight = node_weight
            self.node_answer = node_answer
        answer = self.__dot_down(self.col_tree, self.col_coef, node_answer)
        if x0.ndim is 1:
            answer = answer.reshape(-1)
        return answer

    def nbytes(self, transfer=True, interaction=True, basis=True, python=True):
        nbytes = 0
        if transfer:
            for i in self.row_coef:
                for j in i:
                    nbytes += j.nbytes
            if not self.symmetric:
                for i in self.col_coef:
                    for j in i:
                        nbytes += j.nbytes
        if interaction and not self._status == 'lowmem mcbh':
            for i in self.row_comp_far_matrix:
                for j in i:
                    nbytes += j.nbytes
        if basis:
            for i in self.row_basis:
                nbytes += i.nbytes
            if not self.symmetric:
                for i in self.col_basis:
                    nbytes += i.nbytes
        if python:
            nbytes += getsizeof(self)
            nbytes += getsizeof(self.row_basis)
            for i in self.row_basis:
                nbytes += getsizeof(i)
            nbytes += getsizeof(self.row_coef)
            for i in self.row_coef:
                nbytes += getsizeof(i)
                for j in i:
                    nbytes += getsizeof(j)
            if not self.symmetric:
                nbytes += getsizeof(self.col_basis)
                for i in self.col_basis:
                    nbytes += getsizeof(i)
                nbytes += getsizeof(self.col_coef)
                for i in self.col_coef:
                    nbytes += getsizeof(i)
                    for j in i:
                        nbytes += getsizeof(j)
            if not self._status == 'lowmem mcbh':
                nbytes += getsizeof(self.row_comp_far_matrix)
                for i in self.row_comp_far_matrix:
                    nbytes += getsizeof(i)
                    for j in i:
                        nbytes += getsizeof(j)
                if not self.symmetric:
                    nbytes += getsizeof(self.col_comp_far_matrix)
                    for i in self.col_comp_far_matrix:
                        nbytes += getsizeof(i)
                        for j in i:
                            nbytes += getsizeof(j)
        return nbytes

    def svdcompress(self, tau, verbose=False):
        time0 = time()
        if self._status == 'lowmem mcbh':
            print('Current state is \'lowmem mcbh\', changing to \'fullmem mcbh\'')
            self.__matrix()
        if verbose:
            print('memory BEFORE SVD-compression: {0:.3f}MB'.format(self.nbytes()/1024./1024))
        transfer = self.row_coef
        for i in range(len(transfer)):
            if self.row_tree.notransition[i]:
                continue
            if transfer[i].size == 0:
                transfer[i] = np.eye(transfer[i].shape[1], dtype = transfer[i].dtype)
        if self.symmetric:
            # if symmetry flag is True, then each item of self.queue contains only 'row' tag
            self.__orthogonolize('row')
            self.__compress('row', tau)
            self.__orthogonolize('row')
        else:
            transfer = self.col_coef
            for i in range(len(transfer)):
                if self.col_tree.notransition[i]:
                    continue
                if transfer[i].size == 0:
                    transfer[i] = np.eye(transfer[i].shape[1], dtype = transfer[i].dtype)
            self.__orthogonolize('row')
            self.__compress('col', tau)
            self.__orthogonolize('col')
            self.__compress('row', tau)
            self.__orthogonolize('row')
        self._compresstime = time()-time0
        if verbose:
            print('memory AFTER SVD-compression: {0:.3f}MB'.format(self.nbytes()/1024./1024))
        print('recompression time:', self._compresstime)
        self._status = 'h2'

    def __orthogonolize(self, RC):
        if RC == 'row':
            tree = self.row_tree
            transfer = self.row_coef
            interaction = self.row_comp_far_matrix
            interaction2 = self.col_comp_far_matrix
            node_count = self.row_tree.level[-1]
        else:
            tree = self.col_tree
            transfer = self.col_coef
            interaction = self.col_comp_far_matrix
            interaction2 = self.row_comp_far_matrix
            node_count = self.col_tree.level[-1]
        diff = [0 for i in range(node_count)]
        for i in self.queue:
            for j in i:
                if RC == j[0]:
                    k = j[1]
                    # Update transfer matrix from children nodes
                    if len(tree.child[k]) > 0:
                        s = 0
                        for l in tree.child[k]:
                            e = s+diff[l].shape[1]
                            transfer[k][s:e] = diff[l].dot(transfer[k][s:e])
                            s = e
                    # Update transfer matrix with Q factor of QR factorization
                    transfer[k], r = np.linalg.qr(transfer[k])
                    # Apply R factor of QR factorization to interaction matrices
                    for l in range(len(interaction[k])):
                        interaction[k][l] = r.dot(interaction[k][l])
                    diff[k] = r
        if self.symmetric:
            for i in self.queue:
                for j in i:
                    if RC == j[0]:
                        k = j[1]
                        # Apply R factors to interaction matrices from the other side for symmetry
                        for l in range(len(interaction[k])):
                            interaction[k][l] = interaction[k][l].dot(diff[tree.far[k][l]].T)
        for i in self.queue:
            for j in i:
                if RC == j[0]:
                    k = j[1]
                    # Update interaction matrices for another tree
                    for l in range(len(interaction[k])):
                        m = tree.far[tree.far[k][l]].index(k)
                        interaction2[tree.far[k][l]][m] = interaction[k][l].T

    def __compress(self, RC, tau):
        if RC == 'row':
            tree = self.row_tree
            transfer = self.row_coef
            interaction = self.row_comp_far_matrix
            interaction2 = self.col_comp_far_matrix
            child_index = self.row_child_index
            node_count = self.row_tree.level[-1]
        else:
            tree = self.col_tree
            transfer = self.col_coef
            interaction = self.col_comp_far_matrix
            interaction2 = self.row_comp_far_matrix
            child_index = self.col_child_index
            node_count = self.col_tree.level[-1]
        diffS = [0 for i in range(node_count)]
        diffU = [0 for i in range(node_count)]
        for i in reversed(self.queue):
            for j in i:
                if RC == j[0]:
                    k = j[1]
                    p = tree.parent[k]
                    # Put part of parent transfer matrix, according to node itself, into 'tmp_matrix'
                    if tree.notransition[p]:
                        tmp_matrix = []
                    else:
                        l = tree.child[p].index(k)
                        ind = child_index[p][l]
                        tmp_matrix = [transfer[p][ind[0]:ind[1]]]
                    # Put all the interaction matrices into 'tmp_matrix'
                    tmp_matrix.extend(interaction[k])
                    tmp_matrix = np.hstack(tmp_matrix)
                    # Compute SVD of tmp_matrix
                    U, S, V = np.linalg.svd(tmp_matrix, full_matrices=0)
                    # Define new rank with relative tolerance 'tau'
                    new_rank = S.size
                    tmp_eps = tau*S[0]
                    for l in range(S.size):
                        if S[l] < tmp_eps:
                            new_rank = l
                            break
                    S = S[:new_rank].copy()
                    U = U[:, :new_rank].copy()
                    #V = V[:new_rank]
                    diffS[k] = S
                    diffU[k] = np.diag(1/S).dot(U.T.conj())
                    # Update transfer matrix according to SVD
                    transfer[k] = transfer[k].dot(U.dot(np.diag(S)))
        #return
        for i in self.queue:
            for j in i:
                if RC == j[0]:
                    k = j[1]
                    #Update transfer matrix
                    if (not tree.notransition[k]) and (len(tree.child[k]) > 0):
                        s = 0
                        tmp_matrix = []
                        last_row = 0
                        for l in range(len(tree.child[k])):
                            tmp_diff = diffU[tree.child[k][l]]
                            e = s+tmp_diff.shape[1]
                            tmp_matrix.append(tmp_diff.dot(transfer[k][s:e]))
                            s = e
                            child_index[k][l] = (last_row, last_row+tmp_diff.shape[0])
                            last_row += tmp_diff.shape[0]
                        if s != transfer[k].shape[0]:
                            print('hhj')
                        transfer[k] = np.vstack(tmp_matrix)
                    # Update interaction matrices according to SVD
                    for l in range(len(interaction[k])):
                        interaction[k][l] = diffU[k].dot(interaction[k][l].copy())
        if self.symmetric:
            for i in self.queue:
                for j in i:
                    if RC == j[0]:
                        k = j[1]
                        # Update interaction matrices from the other side for symmetry
                        for l in range(len(interaction[k])):
                            interaction[k][l] = interaction[k][l].dot(diffU[tree.far[k][l]].T)
        for i in self.queue:
            for j in i:
                if RC == j[0]:
                    k = j[1]
                    # Update interaction matrices for another tree
                    for l in range(len(interaction[k])):
                        m = tree.far[tree.far[k][l]].index(k)
                        interaction2[tree.far[k][l]][m] = interaction[k][l].T
        
    def mcbh(self, onfly=False):
        if self._status == 'lowmem mcbh' and not onfly:
            print('Already on required representation')
            return
        if self._status == 'fullmem mcbh' and onfly:
            print('Already on required representation')
            return
        if self._status == 'lowmem mcbh':
            self.__matrix()
            self._status = 'fullmem mcbh'
            return
        if self._status == 'fullmem mcbh':
            del self.row_comp_far_matrix
            del self.col_comp_far_matrix
            self._status = 'lowmem mcbh'
            return
        tol = self.rect_maxvol_tol
        row = self.row_tree
        col = self.col_tree
        level_count = len(row.level)-1
        row_size = row.level[-1]
        col_size = col.level[-1]
        row_r = [0 for i in range(row_size)]
        col_r = [0 for i in range(col_size)]
        for i in range(level_count-1, -1, -1):
            for j in range(col.level[i], col.level[i+1]):
                if col.notransition[j]:
                    continue
                if col.child[j] == []:
                    if self.col_coef[j].shape[0] is 0:
                        self.col_coef[j] = np.eye(self.col_coef[j].shape[1])
                    tmp = rect_maxvol_qr(self.col_coef[j], tol)
                    col_r[j] = self.col_coef[j][tmp[0]]
                    self.col_coef[j] = tmp[1]
                    self.col_basis[j] = self.col_tree.index[j][tmp[0]]
                else:
                    s = []
                    s2 = []
                    ind = 0
                    if self.col_coef[j].shape[0] is 0:
                        self.col_coef[j] = np.eye(self.col_coef[j].shape[1])
                    for k in col.child[j]:
                        p = col_r[k].shape[0]
                        s.append(col_r[k].dot(self.col_coef[j][ind:ind+p]))
                        s2.append(self.col_basis[k])
                        ind += p
                    tmp = rect_maxvol_qr(np.vstack(s), tol)
                    col_r[j] = np.vstack(s)[tmp[0]]
                    self.col_coef[j] = tmp[1]
                    self.col_basis[j] = np.hstack(s2)[tmp[0]]
            if i < level_count-1:
                for j in range(col.level[i+1], col.level[i+2]):
                    col_r[j] = 0
        for i in range(col.level[-3], col.level[-1]):
            col_r[i] = 0
        
        for i in range(level_count-1, -1, -1):
            for j in range(row.level[i], row.level[i+1]):
                if row.notransition[j]:
                    continue
                if row.child[j] == []:
                    if self.row_coef[j].shape[0] is 0:
                        self.row_coef[j] = np.eye(self.row_coef[j].shape[1])
                    tmp = rect_maxvol_qr(self.row_coef[j], tol)
                    row_r[j] = self.row_coef[j][tmp[0]]
                    self.row_coef[j] = tmp[1]
                    self.row_basis[j] = self.row_tree.index[j][tmp[0]]
                else:
                    s = []
                    s2 = []
                    ind = 0
                    if self.row_coef[j].shape[0] is 0:
                        self.row_coef[j] = np.eye(self.row_coef[j].shape[1])
                    for k in row.child[j]:
                        p = row_r[k].shape[0]
                        s.append(row_r[k].dot(self.row_coef[j][ind:ind+p]))
                        s2.append(self.row_basis[k])
                        ind += p
                    tmp = rect_maxvol_qr(np.vstack(s), tol)
                    row_r[j] = np.vstack(s)[tmp[0]]
                    self.row_coef[j] = tmp[1]
                    self.row_basis[j] = np.hstack(s2)[tmp[0]]
            if i < level_count-1:
                for j in range(row.level[i+1], row.level[i+2]):
                    row_r[j] = 0
        for i in range(row.level[-3], row.level[-1]):
            row_r[i] = 0
        if onfly:
            return
        self.__matrix()

    def copy(self):
        ans = Factor.__new__(Factor)
        ans._status = self._status
        ans.symmetric = self.symmetric
        ans.row_tree = self.row_tree
        ans.row_data = self.row_data
        ans.col_tree = self.col_tree
        ans.col_data = self.col_data
        ans.dtype = self.dtype
        ans.func = self.func
        ans.queue = self.queue
        ans.row_basis = copy.deepcopy(self.row_basis)
        ans.row_coef = copy.deepcopy(self.row_coef)
        ans.row_comp_far_matrix = copy.deepcopy(self.row_comp_far_matrix)
        ans.row_child_index = copy.deepcopy(self.row_child_index)
        ans.col_basis = copy.deepcopy(self.col_basis)
        ans.col_coef = copy.deepcopy(self.col_coef)
        ans.col_comp_far_matrix = copy.deepcopy(self.col_comp_far_matrix)
        ans.col_child_index = copy.deepcopy(self.col_child_index)
        return ans

    @property
    def T(self):
        return self.transpose()
    
    def transpose(self):
        ans = Factor.__new__(Factor)
        ans._status = self._status
        ans.symmetric = self.symmetric
        ans.row_tree = self.col_tree
        ans.row_data = self.col_data
        ans.col_tree = self.row_tree
        ans.col_data = self.row_data
        ans.dtype = self.dtype
        ans.func = self.func
        ans.queue = self.queue
        ans.row_basis = self.col_basis
        ans.row_coef = self.col_coef
        ans.row_comp_far_matrix = self.col_comp_far_matrix
        ans.row_child_index = copy.deepcopy(self.col_child_index)
        ans.col_basis = self.row_basis
        ans.col_coef = self.row_coef
        ans.col_comp_far_matrix = self.row_comp_far_matrix
        ans.col_child_index = copy.deepcopy(self.row_child_index)
        return ans
