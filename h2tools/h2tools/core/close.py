"""Module h2py.core.close"""
from __future__ import print_function, absolute_import, division

import numpy as np
from time import time

class Factor(object):
    """class Factor"""
    def __new__(cls, *args, **kwargs):
        return super(Factor, cls).__new__(cls)
    
    def __init__(self, dtype, func, row_data, row_tree, row_close, col_data, col_tree, col_close, onfly, verbose):
        time0 = time()
        self.dtype = dtype
        self.func = func
        self.row_data = row_data
        self.col_data = col_data
        self.onfly = onfly
        row = self.row_tree = row_tree
        col = self.col_tree = col_tree
        self.row_close = row_close
        self.col_close = col_close
        if onfly:
            self.row_comp_far_matrix = None
            self.col_comp_far_matrix = None
            return
        self._totaltime = 0
        self._functime = 0
        self._funccalls = 0
        self._elemscomputed = 0
        row_size = row.level[-1]
        col_size = col.level[-1]
        self.row_matrix = [[] for i in range(row_size)]
        self.col_matrix = [[] for i in range(col_size)]
        for i in range(row_size):
            for j in row_close[i]:
                time1 = time()
                matrix = func(row.index[i], col.index[j])
                self._functime += time()-time1
                self._funccalls += 1
                self._elemscomputed += matrix.size
                self.row_matrix[i].append(matrix)
        for i in range(col_size):
            for j in col_close[i]:
                k = row_close[j].index(i)
                self.col_matrix[i].append(self.row_matrix[j][k].T)
        self._totaltime = time()-time0
        if verbose:
            print('Close interactions:')
            print('    Function calls: {}'.format(self._funccalls))
            print('    Function values computed: {}'.format(self._elemscomputed))
            print('    Function time: {:.2f} seconds'.format(self._functime))
            if self._elemscomputed > 0:
                print('    Average time per function value: {:.2e} seconds'.format(self._functime/self._elemscomputed))
            print('    Total close time: {:.2f} seconds'.format(self._totaltime))

    def dot(self, x):
        row = self.row_tree
        col = self.col_tree
        row_close = self.row_close
        col_close = self.col_close
        row_data =  self.row_data
        col_data = self.col_data
        func = self.func
        row_size = row.level[-1]
        if x.ndim == 1:
            x = x.reshape(x.shape[0], 1)
        nrhs = x.shape[1]
        answer = np.zeros((row.index[0].shape[0], nrhs), dtype = self.dtype)
        if self.onfly:
            for i in range(row_size):
                for j in range(len(row_close[i])):
                    answer[row.index[i]] += func(row.index[i], col.index[row_close[i][j]]).dot(x[col.index[row_close[i][j]]])
        else:
            matrix = self.row_matrix
            for i in range(row_size):
                for j in range(len(row_close[i])):
                    answer[row.index[i]] += matrix[i][j].dot(x[col.index[row_close[i][j]]])
        if nrhs == 1:
            answer = answer.reshape(answer.shape[0])
        return answer

    def rdot(self, x):
        row = self.row_tree
        col = self.col_tree
        row_close = self.row_close
        col_close = self.col_close
        row_data =  self.row_data
        col_data = self.col_data
        func = self.func
        col_size = col.level[-1]
        if x.ndim == 1:
            x = x.reshape(x.shape[0], 1)
        nrhs = x.shape[1]
        answer = np.zeros((col.index[0].shape[0], nrhs), dtype = self.dtype)
        if self.onfly:
            for i in range(col_size):
                for j in range(len(col_close[i])):
                    answer[col.index[i]] += func(row.index[col_close[i][j]], col.index[i]).T.dot(x[row.index[col_close[i][j]]])
        else:
            matrix = self.col_matrix
            for i in range(col_size):
                for j in range(len(col.close[i])):
                    answer[col.index[i]] += matrix[i][j].dot(x[row.index[col_close[i][j]]])
        if nrhs == 1:
            answer = answer.reshape(answer.shape[0])
        return answer

    def nbytes(self):
        if self.onfly:
            return 0
        row = self.row_tree
        row_size = row.level[-1]
        mem = 0
        for i in self.row_matrix:
            for j in i:
                mem += j.size
        return mem

    @property
    def T(self):
        return self.transpose()
        
    def transpose(self):
        raise NotImplementedError("Need to reimplement")
        answer = Factor.__new__(Factor)
        answer.dtype = self.dtype
        answer.func = self.func
        col = answer.row_tree = self.col_tree
        row = answer.col_tree = self.row_tree
        answer.row_data = self.col_data
        answer.col_data = self.row_data
        answer.onfly = self.onfly
        size = row.level[-1]
        answer.col_matrix = [[] for i in range(size)]
        for i in range(size):
            for j in range(len(row.close[i])):
                answer.col_matrix[i].append(self.row_matrix[i][j])
        col = self.col_tree
        size = col.level[-1]
        answer.row_matrix = [[] for i in range(size)]
        for i in range(size):
            for j in range(len(col.close[i])):
                answer.row_matrix[i].append(self.col_matrix[i][j])
        return answer
