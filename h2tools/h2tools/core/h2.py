"""Module h2py.core.h2"""
from __future__ import print_function, absolute_import, division

from time import time
import numpy as np
from .maxvolpy.maxvol import maxvol_svd, maxvol_qr
import copy
from sys import getsizeof

class Factor(object):
    """Class Factor"""
    def __new__(cls, *args, **kwargs):
        """__new__ docstring"""
        x = super(Factor, cls).__new__(cls)
        x._status = None
        return x

    def __init__(self, dtype, func, row_data, row_tree, row_far, row_notransition, col_data, col_tree, col_far, col_notransition, queue, tau, iters=1, onfly=False, verbose=False, symmetric=False, maxvol_tol = 1.05):
        """Docstring for mcbh.Factor.__init__"""
        time0 = time()
        self.dtype = dtype
        self.func = func
        self.row_data = row_data
        self.row_tree = row_tree
        self.row_far = row_far
        self.row_notransition = row_notransition
        self.col_data = col_data
        self.col_tree = col_tree
        self.col_far = col_far
        self.col_notransition = col_notransition
        self.queue = queue
        self.tau = tau
        self.symmetric = symmetric
        self.maxvol_tol = maxvol_tol
        self._totaltime = 0
        self._functime = 0
        self._funccalls = 0
        self._elemscomputed = 0
        self._maxvoltime = 0
        row_size = row_tree.level[-1]
        row_prebasis = [np.ndarray(0, dtype = np.int32) for i in range(row_size)]
        col_size = col_tree.level[-1]
        col_prebasis = [np.ndarray(0, dtype = np.int32) for i in range(col_size)]
        self.__factorup(tau, row_prebasis, col_prebasis, verbose)
        for i in range(iters):
            row_prebasis, col_prebasis = self.__factordown(verbose)
            self.__factorup(tau, row_prebasis, col_prebasis, verbose)
        if not onfly:
            self.__matrix()
            self._status = 'fullmem mcbh'
        else:
            self.row_comp_far_matrix = None
            self.col_comp_far_matrix = None
            self._status = 'lowmem mcbh'
        self._totaltime = time()-time0
        if verbose:
            print('Far interactions(MCBH method):')
            print('    Function calls: {}'.format(self._funccalls))
            print('    Function values computed: {}'.format(self._elemscomputed))
            print('    Function time: {:.2f} seconds'.format(self._functime))
            if self._elemscomputed > 0:
                print('    Average time per function value: {:.2e} seconds'.format(self._functime/self._elemscomputed))
            print('    Maxvol time: {:.2f} seconds'.format(self._maxvoltime))
            print('    Total MCBH time: {:.2f} seconds'.format(self._totaltime))
    
    def __factorup(self, tau, row_prebasis, col_prebasis, verbose):
        """Technical procedure, computes new basises and transfer matrices with given representor sets"""
        def _buildmatrix(ind, RC, tree0, far0, tree1, basis0, basis1, prebasis, func):
            """Technical function, returns index list and matrix, corresponding to node 'ind' of tree 'tree0'"""
            child = tree0.child[ind]
            if len(child) == 0:
                list0 = [tree0.index[ind]]
                index0 = list0[0]
            else:
                list0 = []
                for k in child:
                    list0.append(basis0[k])
                index0 = np.concatenate(list0)
            list1 = [prebasis]
            child = tree1.child
            for k in far0[ind]:
                if basis1[k].size > 0:
                    list1.append(basis1[k])
                elif len(child[k]) == 0:
                    list1.append(tree1.index[k])
                else:
                    for l in child[k]:
                        list1.append(basis1[l])
            if len(list1) > 1:
                index1 = np.concatenate(list1)
                if RC == 'row':
                    time0 = time()
                    matrix = func(index0, index1)
                    self._functime += time()-time0
                else:
                    time0 = time()
                    matrix = func(index1, index0)
                    self._functime += time()-time0
                self._funccalls += 1
                self._elemscomputed += matrix.size
                return index0, matrix
            else:
                return index0, 0
            
        row_size = self.row_tree.level[-1]
        col_size = self.col_tree.level[-1]
        self.row_basis = [np.ndarray(0, dtype = np.int32) for i in range(row_size)]
        self.row_coef = [np.ndarray((0, 0), dtype = np.int32) for i in range(row_size)]
        self.row_child_index = [[] for i in range(row_size)]
        if self.symmetric:
            self.col_basis = self.row_basis
            self.col_coef = self.row_coef
            self.col_child_index = self.row_child_index
        else:
            self.col_basis = [np.ndarray(0, dtype = np.int32) for i in range(col_size)]
            self.col_coef = [np.ndarray((0, 0), dtype = np.int32) for i in range(col_size)]
            self.col_child_index = [[] for i in range(col_size)]
        tol = self.maxvol_tol
        # Loop is query-dependant
        for i in self.queue:
            for j in i:
                ind = j[1]
                if j[0] == 'row':
                    row, matrix = _buildmatrix(ind, j[0], self.row_tree, self.row_far, self.col_tree, self.row_basis, self.col_basis, row_prebasis[self.row_tree.parent[ind]], self.func)
                    if matrix is 0:
                        self.row_basis[ind] = row.copy()
                        self.row_coef[ind] = np.ndarray((0, row.size), dtype = self.dtype)
                    else:
                        time0 = time()
                        basis, self.row_coef[ind] = maxvol_svd(matrix, tau, tol, job = 'R')
                        self._maxvoltime += time()-time0
                        self.row_basis[ind] = row[basis].copy()
                    del row, matrix
                    s = 0
                    self.row_child_index[ind] = []
                    for k in self.row_tree.child[ind]:
                        e = s+self.row_coef[k].shape[1]
                        self.row_child_index[ind].append((s, e))
                        s = e
                else:
                    col, matrix = _buildmatrix(ind, j[0], self.col_tree, self.col_far, self.row_tree, self.col_basis, self.row_basis, col_prebasis[self.col_tree.parent[ind]], self.func)
                    if matrix is 0:
                        self.col_basis[ind] = col.copy()
                        self.col_coef[ind] = np.ndarray((0, col.size), dtype = self.dtype)
                    else:
                        time0 = time()
                        basis, self.col_coef[ind] = maxvol_svd(matrix, tau, tol, job = 'C')
                        self._maxvoltime += time()-time0
                        self.col_basis[ind] = col[basis].copy()
                    del col, matrix
                    s = 0
                    self.col_child_index[ind] = []
                    for k in self.col_tree.child[ind]:
                        e = s+self.col_coef[k].shape[1]
                        self.col_child_index[ind].append((s, e))
                        s = e

    def __factordown(self, verbose):
        """Technical procedure, computes representor sets with given basises"""
        row_size = self.row_tree.level[-1]
        col_size = self.col_tree.level[-1]
        row_prebasis = [np.ndarray(0, dtype = np.int32) for i in range(row_size)]
        col_prebasis = [np.ndarray(0, dtype = np.int32) for i in range(col_size)]
        tol = self.maxvol_tol
        # Loop is query-dependant
        for i in reversed(self.queue):
            for j in i:
                ind = j[1]
                if j[0] == 'row':
                    row = self.row_basis[ind]
                    col_list = [row_prebasis[self.row_tree.parent[ind]]]
                    for k in self.row_far[ind]:
                        col_list.append(self.col_basis[k])
                    col = np.concatenate(col_list)
                    if row.size >= col.size:
                        row_prebasis[ind] = col.copy()
                    else:
                        time0 = time()
                        tmpmatrix = self.func(row, col).T
                        self._functime += time()-time0
                        self._funccalls += 1
                        self._elemscomputed += tmpmatrix.size
                        time0 = time()
                        tmpmatrix = maxvol_qr(tmpmatrix, tol)[0]
                        self._maxvoltime += time()-time0
                        row_prebasis[ind] = col[tmpmatrix].copy()
                        del tmpmatrix
                    del row, col, col_list
                else:
                    col = self.col_basis[ind]
                    row_list = [col_prebasis[self.col_tree.parent[ind]]]
                    for k in self.col_far[ind]:
                        row_list.append(self.row_basis[k])
                    row = np.concatenate(row_list)
                    if col.size >= row.size:
                        col_prebasis[ind] = row.copy()
                    else:
                        time0 = time()
                        tmpmatrix = self.func(row, col)
                        self._functime += time()-time0
                        self._funccalls += 1
                        self._elemscomputed += tmpmatrix.size
                        time0 = time()
                        tmpmatrix = maxvol_qr(tmpmatrix, tol)[0]
                        self._maxvoltime += time()-time0
                        col_prebasis[ind] = row[tmpmatrix].copy()
                        del tmpmatrix
                    del row, row_list, col
        return row_prebasis, col_prebasis
            
    def __matrix(self):
        """Technical procedure, computes interaction matrices"""
        row = self.row_tree
        row_far = self.row_far
        row_data = self.row_data
        col = self.col_tree
        col_far = self.col_far
        col_data = self.col_data
        row_basis = self.row_basis
        col_basis = self.col_basis
        row_size = row.level[-1]
        col_size = col.level[-1]
        self.row_comp_far_matrix = [[] for i in range(row_size)]
        self.col_comp_far_matrix = [[] for i in range(col_size)]
        # Loop os query-independant
        for i in range(row_size):
            for j in row_far[i]:
                time0 = time()
                tmpmatrix = self.func(row_basis[i], col_basis[j])
                #self._functime += time()-time0
                #self._funccalls += 1
                #self._elemscomputed += tmpmatrix.size
                self.row_comp_far_matrix[i].append(tmpmatrix)
                del tmpmatrix
        for i in range(col_size):
            for j in col_far[i]:
                k = row_far[j].index(i)
                self.col_comp_far_matrix[i].append(self.row_comp_far_matrix[j][k].T)

    @staticmethod
    def __dot_up(tree, notransition, transfer, x):
        """Technical procedure, computes basises 'weights' from all the 'particles', from bottom of the tree to top, SINGLE-CORE version"""
        size = tree.level[-1]
        level_count = len(tree.level)-1
        node_weight = [np.zeros((0, x.shape[1]), dtype = x.dtype) for i in range(size)]
        # Loop is query-dependant
        for i in range(level_count-1):
            for j in range(tree.level[level_count-i-2], tree.level[level_count-i-1]):
                if notransition[j]:
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
    def __dot_interact(tree, far, notransition, matrix, node_weight):
        """Technical procedure, computes basises 'potentials' from basises 'weights', uses precomputed interaction matrices, SINGLE-CORE version"""
        size = tree.level[-1]
        tmp = node_weight[-1]
        node_answer = [np.ndarray((0, tmp.shape[1]), dtype = tmp.dtype) if len(far[i]) is 0 else np.zeros((matrix[i][0].shape[0], tmp.shape[1]), dtype = tmp.dtype) for i in range(size)]
        for i in range(size):
            if notransition[i]:
                continue
            tmp = node_answer[i]
            for j in range(len(far[i])):
                tmp += matrix[i][j].dot(node_weight[far[i][j]])
        return node_answer

    @staticmethod
    def __dot_interact_onfly(tree, far, notransition, func, basis0, basis1, node_weight, T = False):
        """Technical procedure, computes basises 'potentials' from basises 'weights', computes interaction matrices on the fly, SINGLE-CORE version"""
        size = tree.level[-1]
        tmp = node_weight[-1]
        node_answer = [np.ndarray((0, tmp.shape[1]), dtype = tmp.dtype) if len(far[i]) is 0 else np.zeros((func(basis0[i], basis1[far[i][0]]).shape[0], tmp.shape[1]), dtype = tmp.dtype) for i in range(size)]
        for i in range(size):
            if notransition[i]:
                continue
            tmp = node_answer[i]
            if T:
                for j in range(len(far[i])):
                    tmp += func(basis1[far[i][j]], basis0[i]).T.dot(node_weight[far[i][j]])
            else:
                for j in range(len(far[i])):
                    tmp += func(basis0[i], basis1[far[i][j]]).dot(node_weight[far[i][j]])
        return node_answer
    
    @staticmethod
    def __dot_down(tree, notransition, transfer, node_answer):
        """Techical procedure, computes all 'potentials' from basises 'potentials', from top of the tree to bottom, SINGLE-CORE version"""
        size = tree.level[-1]
        level_count = len(tree.level)-1
        tmp = node_answer[-1]
        dtype = tmp.dtype
        nrhs = tmp.shape[1]
        answer = np.zeros((tree.data.count, nrhs), dtype = dtype)
        for i in range(level_count-1):
            for j in range(tree.level[i], tree.level[i+1]):
                if notransition[j]:
                    continue
                if node_answer[j].shape[0] is 0:
                    node_answer[j] = np.zeros((transfer[j].shape[1], nrhs), dtype = dtype)
        for i in range(level_count-1):
            for j in range(tree.level[i], tree.level[i+1]):
                if notransition[j]:
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
        node_weight = self.__dot_up(self.col_tree, self.col_notransition, self.col_coef, x)
        if self._status == 'lowmem mcbh':
            node_answer = self.__dot_interact_onfly(self.row_tree, self.row_far, self.row_notransition, self.func, self.row_basis, self.col_basis, node_weight)
        else:
            node_answer = self.__dot_interact(self.row_tree, self.row_far, self.row_notransition, self.row_comp_far_matrix, node_weight)
        if dasha_debug:
            self.node_weight = node_weight
            self.node_answer = node_answer
        answer = self.__dot_down(self.row_tree, self.row_notransition, self.row_coef, node_answer)
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
        node_weight = self.__dot_up(self.row_tree, self.row_notransition, self.row_coef, x)
        if self._status == 'lowmem mcbh':
            node_answer = self.__dot_interact_onfly(self.col_tree, self.col_far, self.col_notransition, self.func, self.col_basis, self.row_basis, node_weight, 1)
        else:
            node_answer = self.__dot_interact(self.col_tree, self.col_far, self.col_notransition, self.col_comp_far_matrix, node_weight)
        if dasha_debug:
            self.node_weight = node_weight
            self.node_answer = node_answer
        answer = self.__dot_down(self.col_tree, self.col_notransition, self.col_coef, node_answer)
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
        if basis and not self._status == 'h2':
            for i in self.row_basis:
                nbytes += i.nbytes
            if not self.symmetric:
                for i in self.col_basis:
                    nbytes += i.nbytes
        if python:
            nbytes += getsizeof(self)
            if not self._status == 'h2':
                nbytes += getsizeof(self.row_basis)
                for i in self.row_basis:
                    nbytes += getsizeof(i)
            nbytes += getsizeof(self.row_coef)
            for i in self.row_coef:
                nbytes += getsizeof(i)
                for j in i:
                    nbytes += getsizeof(j)
            if not self.symmetric:
                if not self._status == 'h2':
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
            if self.row_notransition[i]:
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
                if self.col_notransition[i]:
                    continue
                if transfer[i].size == 0:
                    transfer[i] = np.eye(transfer[i].shape[1], dtype = transfer[i].dtype)
            self.__orthogonolize('row')
            self.__compress('col', tau)
            self.__orthogonolize('col')
            self.__compress('row', tau)
            self.__orthogonolize('row')
        self._compresstime = time()-time0
        self.row_basis = None
        self.col_basis = None
        if verbose:
            print('memory AFTER SVD-compression: {0:.3f}MB'.format(self.nbytes()/1024./1024))
        print('recompression time:', self._compresstime)
        self._status = 'h2'

    def __orthogonolize(self, RC):
        if RC == 'row':
            tree = self.row_tree
            far = self.row_far
            transfer = self.row_coef
            interaction = self.row_comp_far_matrix
            interaction2 = self.col_comp_far_matrix
            node_count = self.row_tree.level[-1]
        else:
            tree = self.col_tree
            far = self.col_far
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
                            interaction[k][l] = interaction[k][l].dot(diff[far[k][l]].T)
        for i in self.queue:
            for j in i:
                if RC == j[0]:
                    k = j[1]
                    # Update interaction matrices for another tree
                    for l in range(len(interaction[k])):
                        m = far[far[k][l]].index(k)
                        interaction2[far[k][l]][m] = interaction[k][l].T

    def __compress(self, RC, tau):
        if RC == 'row':
            tree = self.row_tree
            far = self.row_far
            transfer = self.row_coef
            interaction = self.row_comp_far_matrix
            interaction2 = self.col_comp_far_matrix
            child_index = self.row_child_index
            node_count = self.row_tree.level[-1]
            notransition = self.row_notransition
        else:
            tree = self.col_tree
            far = self.col_far
            transfer = self.col_coef
            interaction = self.col_comp_far_matrix
            interaction2 = self.row_comp_far_matrix
            child_index = self.col_child_index
            node_count = self.col_tree.level[-1]
            notransition = self.col_notransition
        diffS = [0 for i in range(node_count)]
        diffU = [0 for i in range(node_count)]
        for i in reversed(self.queue):
            for j in i:
                if RC == j[0]:
                    k = j[1]
                    p = tree.parent[k]
                    # Put part of parent transfer matrix, according to node itself, into 'tmp_matrix'
                    if notransition[p]:
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
                    if (not notransition[k]) and (len(tree.child[k]) > 0):
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
                            interaction[k][l] = interaction[k][l].dot(diffU[far[k][l]].T)
        for i in self.queue:
            for j in i:
                if RC == j[0]:
                    k = j[1]
                    # Update interaction matrices for another tree
                    for l in range(len(interaction[k])):
                        m = far[far[k][l]].index(k)
                        interaction2[far[k][l]][m] = interaction[k][l].T

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
            self.row_comp_far_matrix = None
            self.col_comp_far_matrix = None
            self._status = 'lowmem mcbh'
            return
        tol = 1.05
        row = self.row_tree
        col = self.col_tree
        level_count = len(row.level)-1
        row_size = row.level[-1]
        col_size = col.level[-1]
        if self._status == 'h2':
            self.row_basis = [np.ndarray(0, dtype = np.int32) for i in range(row_size)]
            if not self.symmetric:
                self.col_basis = [np.ndarray(0, dtype = np.int32) for i in range(col_size)]
            else:
                self.col_basis = self.row_basis
            self._status = 'fullmem mcbh'
        row_r = [0 for i in range(row_size)]
        col_r = [0 for i in range(col_size)]
        for i in range(level_count-1, -1, -1):
            for j in range(col.level[i], col.level[i+1]):
                if self.col_notransition[j]:
                    continue
                if col.child[j] == []:
                    if self.col_coef[j].shape[0] is 0:
                        self.col_coef[j] = np.eye(self.col_coef[j].shape[1])
                    tmp = maxvol_qr(self.col_coef[j], tol)
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
                    tmp = maxvol_qr(np.vstack(s), tol)
                    col_r[j] = np.vstack(s)[tmp[0]]
                    self.col_coef[j] = tmp[1]
                    self.col_basis[j] = np.concatenate(s2)[tmp[0]]
            if i < level_count-1:
                for j in range(col.level[i+1], col.level[i+2]):
                    col_r[j] = 0
        for i in range(col.level[-3], col.level[-1]):
            col_r[i] = 0
        
        if not self.symmetric:
            for i in range(level_count-1, -1, -1):
                for j in range(row.level[i], row.level[i+1]):
                    if self.row_notransition[j]:
                        continue
                    if row.child[j] == []:
                        if self.row_coef[j].shape[0] is 0:
                            self.row_coef[j] = np.eye(self.row_coef[j].shape[1])
                        tmp = maxvol_qr(self.row_coef[j], tol)
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
                        tmp = maxvol_qr(np.vstack(s), tol)
                        row_r[j] = np.vstack(s)[tmp[0]]
                        self.row_coef[j] = tmp[1]
                        self.row_basis[j] = np.concatenate(s2)[tmp[0]]
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
        ans.row_far = copy.deepcopy(self.row_far)
        ans.row_notransition = copy.deepcopy(self.row_notransition)
        ans.row_basis = copy.deepcopy(self.row_basis)
        ans.row_coef = copy.deepcopy(self.row_coef)
        ans.row_comp_far_matrix = copy.deepcopy(self.row_comp_far_matrix)
        ans.row_child_index = copy.deepcopy(self.row_child_index)
        ans.dtype = self.dtype
        ans.func = self.func
        ans.queue = copy.deepcopy(self.queue)
        if not self.symmetric:
            ans.col_tree = self.col_tree
            ans.col_data = self.col_data
            ans.col_far = copy.deepcopy(self.col_far)
            ans.col_notransition = copy.deepcopy(self.col_notransition)
            ans.col_basis = copy.deepcopy(self.col_basis)
            ans.col_coef = copy.deepcopy(self.col_coef)
            ans.col_comp_far_matrix = copy.deepcopy(self.col_comp_far_matrix)
            ans.col_child_index = copy.deepcopy(self.col_child_index)
        else:
            ans.col_tree = ans.row_tree
            ans.col_data = ans.row_data
            ans.col_far = ans.row_far
            ans.col_notransition = ans.row_notransition
            ans.col_basis = ans.row_basis
            ans.col_coef = ans.row_coef
            ans.col_comp_far_matrix = ans.row_comp_far_matrix
            ans.col_child_index = ans.row_child_index
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
        ans.row_far = self.col_far
        ans.row_notransition = self.col_notransition
        ans.col_tree = self.row_tree
        ans.col_data = self.row_data
        ans.col_far = self.row_far
        ans.col_notransition = self.col_notransition
        ans.dtype = self.dtype
        ans.func = self.func
        ans.queue = self.queue
        ans.row_basis = self.col_basis
        ans.row_coef = self.col_coef
        ans.row_comp_far_matrix = self.col_comp_far_matrix
        ans.row_child_index = self.col_child_index
        ans.col_basis = self.row_basis
        ans.col_coef = self.row_coef
        ans.col_comp_far_matrix = self.row_comp_far_matrix
        ans.col_child_index = self.row_child_index
        return ans
