import numpy as np
from time import time
from sys import getsizeof
from .rect_cross2d import rect_cross2d

class Factor(object):
    def __new__(cls, *args, **kwargs):
        return super(Factor, cls).__new__(cls)

    @staticmethod
    def from_file(f):
        pass
    
    def __init__(self, dtype, func, row_data, row_tree, col_data, col_tree, queue, tau, verbose = False):
        start_time = time()
        self.row_tree = row = row_tree
        self.row_data = row_data
        self.col_tree = col = col_tree
        self.col_data = col_data
        row_size = row.level[-1]
        self.factor = [[] for i in range(row_size)]
        self._totaltime = 0
        self._functime = 0
        self._funccalls = 0
        self._elemscomputed = 0
        self._crosstime = 0
        for i in range(row_size):
            for j in row.far[i]:
                time0 = time()
                tmp_matrix = func(row.index[i], col.index[j])
                self._functime += time()-time0
                self._funccalls += 1
                self._elemscomputed += tmp_matrix.size
                time0 = time()
                self.factor[i].append(rect_cross2d(tmp_matrix, tau, max_iters = 10, max_restarts = 1))
                self._crosstime += time()-time0
        self._totaltime = time()-start_time
        if verbose:
            print('Function calls: {}'.format(self._funccalls))
            print('Function values computed: {}'.format(self._elemscomputed))
            print('Function time:{}'.format(self._functime))
            print('Average time per function value:{}'.format(self._functime/self._elemscomputed))
            print('Cross time:{}'.format(self._crosstime))
            print('Total MSKEL time:{}'.format(self._totaltime))

    def dot(self, x, dasha_debug = False):
        row = self.row_tree
        col = self.col_tree
        row_size = row.level[-1]
        col_size = col.level[-1]
        answer = np.zeros(x.shape, dtype = np.float64)
        if x.ndim is 1:
            nrhs = 1
            x = x.reshape(-1,1)
        else:
            nrhs = x.shape[1]
        preanswer = [0 for i in range(row_size)]
        preweight = [0 for i in range(col_size)]
        for i in range(col_size-1, -1, -1):
            if len(col.child[i]) is 0:
                preweight[i] = x[col.index[i]]
            else:
                tmp = []
                for j in col.child[i]:
                    tmp.append(preweight[j])
                preweight[i] = np.concatenate(tmp)
        for i in range(row_size):
            maxj = len(row.far[i])
            tmp = np.zeros((row.index[i].size, nrhs), dtype = np.float64)
            for j in range(maxj):
                tmp += (self.factor[i][j][0]*self.factor[i][j][1].reshape(1, -1)).dot(self.factor[i][j][2].dot(preweight[row.far[i][j]]))
            preanswer[i] = tmp
        for i in range(row_size):
            if len(row.child[i]) is 0:
                answer[row.index[i]] = preanswer[i]
                continue
            s = 0
            for j in row.child[i]:
                e = s+row.index[j].size
                preanswer[j] += preanswer[i][s:e]
                s = e
        return answer

    def rdot(self, x, dasha_debug = False):
        row = self.row_tree
        col = self.col_tree
        row_size = row.level[-1]
        col_size = col.level[-1]
        answer = np.zeros(x.shape, dtype = np.float64)
        if x.ndim is 1:
            nrhs = 1
            x = x.reshape(-1,1)
        else:
            nrhs = x.shape[1]
        preanswer = [0 for i in range(col_size)]
        preweight = [0 for i in range(row_size)]
        for i in range(row_size-1, -1, -1):
            if len(row.child[i]) is 0:
                preweight[i] = x[row.index[i]]
            else:
                tmp = []
                for j in row.child[i]:
                    tmp.append(preweight[j])
                preweight[i] = np.concatenate(tmp)
        for i in range(col_size):
            maxj = len(col.far[i])
            tmp = np.zeros((col.index[i].size, nrhs), dtype = np.float64)
            for j in range(maxj):
                row_number = col.far[i][j]
                row_far_number = row.far[row_number].index(i)
                tmp += (self.factor[row_number][row_far_number][2].T*self.factor[row_number][row_far_number][1].reshape(1, -1)).dot(self.factor[row_number][row_far_number][0].T.dot(preweight[col.far[i][j]]))
            preanswer[i] = tmp
        for i in range(col_size):
            if len(col.child[i]) is 0:
                answer[col.index[i]] = preanswer[i]
                continue
            s = 0
            for j in col.child[i]:
                e = s+col.index[j].size
                preanswer[j] += preanswer[i][s:e]
                s = e
        return answer

    def nbytes(self, interaction = True, python = True):
        nbytes = 0
        if interaction:
            for i in self.factor:
                for j in i:
                    nbytes += j[0].nbytes+j[1].nbytes+j[2].nbytes
        if python:
            nbytes += getsizeof(self)
            nbytes += getsizeof(self.factor)
            for i in self.factor:
                nbytes += getsizeof(i)
                for j in i:
                    nbytes += getsizeof(j)
                    nbytes += getsizeof(j[0])
                    nbytes += getsizeof(j[1])
                    nbytes += getsizeof(j[2])
        return nbytes

    @property
    def T(self):
        return self.transpose()

    def transpose(self):
        ans = Factor.__new__(Factor)
        ans.row_tree = self.col_tree
        ans.row_data = self.col_data
        ans.col_tree = self.row_tree
        ans.col_data = self.row_data
        ans.func = self.func
        ans.eps = self.eps
        ans.max_rank = self.max_rank
        ans.ftype = self.ftype
        col_size = self.col_tree.level[-1]
        if self.ftype is 'separate':
            ans.factor = [[] for i in range(col_size)]
            ans.close_factor = [[] for i in range(col_size)]
            for i in range(col_size):
                for j in self.col_tree.far[i]:
                    ind = self.row_tree.far[j].index(i)
                    ans.factor[i].append((self.factor[j][ind][2], self.factor[j][ind][1], self.factor[j][ind][0]))
                for j in self.col_tree.close[i]:
                    ans.close_factor[i].append(self.close_factor[j][self.row_tree.close[j].index(i)].T)
        else:
            print('bad ftype')
            return
        return ans
