"""
Class Problem stores discretized problem and block cluster tree.
"""

from __future__ import print_function, absolute_import, division

from time import time
import numpy as np
from .cluster_tree import ClusterTree
try:
    from mpi4py import MPI
except:
    pass

class Problem(object):
    """
    Container for problem (as linear operator) and block cluster tree.

    Contains problem setting as linear operator, which is represented
    by geometric data, corresponding to domain and area of linear
    operator, and function of interaction of discrete elements of
    domain and area. Since geometric data is discontinuous, given
    linear operator is assumed as matrix of interactions.

    Domain and area of operator transform to columns and rows of matrix
    of interaction correspondingly. For convinience, domain (rows) is
    also called destination of interaction and area (columns) is also
    called source of interaction. It is to be more appropriate for
    N-body, electrostatics and other particle-to-particle interactions.

    Interaction of discrete elements can be a scalar, vector, matrix or
    even a tensor. If interaction is not scalar, corresponding function
    must return tensor of shape `(n_rows, ..., n_columns)`. There are
    examples of vector function of interactions in *examples* folder of
    *h2tools* git repository.

    Cluster trees, corresponding to domain and area (or to source and
    destination), are called column and row cluster trees accordingly.

    Parameters
    ----------
    func : callable
        Function of interaction. Should take 4 parameters `(row_data,
        row_index, column_data, columns)`.
    row_tree : ClusterTree
        Row cluster tree, corresponding to area of operator or
        destination of interaction.
    col_tree : ClusterTree
        Column cluster tree, corresponding to domain of operator or
        source of interaction.
    symmetric : boolean
        Shows if domain and area or source and destination are the
        same, while linear operator itself is symmetric. Reduces
        `mcbh` factorization time by a multiplier of 2.
    verbose : boolean
        If `True`, outputs additional information, such as number of
        levels, nodes and leaves of cluster trees.

    Attributes
    ----------
    _func: callable
        Function of interaction of objects (input parameter `func`).
    row_data : Python object
        Geometric data, corresponding to area of operator or
        destination of interaction.
    col_data : Python object
        Geometric data, corresponding to domain of operator or source
        of interaction.
    symmetric : boolean
        Shows if the problem is symmetric (input parameter
        `symmetric`).
    shape : tuple
        Shape of corresponding matrix of interactions.
    func_shape: tuple
        Shape of each element of matrix (scalar/vector/matrix
        interactions). Shape of function of interaction of 2 discrete
        elements.
    dtype : numpy.dtype
        Data type of corresponding matrix of interactions.
    row_tree : ClusterTree
        Row cluster tree.
    col_tree : ClusterTree
        Column cluster tree.
    row_block_size : integer
        Maximum size of leaf-node of row cluster tree with nonempty
        near-field interactions.
    col_block_size : integer
        Maximum size of leaf-node of row cluster tree with nonempty
        near-field interactions.
    row_far : list of list of integer
        List of lists of admissibly far nodes for each node of row
        cluster tree. `row_far[i]` is a list of admissibly far nodes,
        corresponding to `i`-th node of row cluster tree.
    col_far : list of list of integer
        List of lists of admissibly far nodes for each node of column
        cluster tree. `col_far[i]` is a list of admissibly far nodes,
        corresponding to `i`-th node of column cluster tree.
    """

    def __init__(self, func, row_tree, col_tree, symmetric, verbose=False,
            mpi_comm=None):
        self._func = func
        if symmetric and row_tree is not col_tree:
            raise ValueError("row_tree and col_tree parameters must be the "
                "same (as Python objects) if flag symmetric is `True`")
        self.symmetric = symmetric
        self.row_tree = row_tree
        self.col_tree = col_tree
        self.row_data = row_tree.data
        self.col_data = col_tree.data
        self.shape = (len(row_tree.data), len(col_tree.data))
        l = np.arange(1, dtype=np.uint64)
        tmp = self.func(l, l)
        self.func_shape = tmp.shape[1:-1]
        self.dtype = tmp.dtype
        self._build(verbose)
        self.mpi_comm = mpi_comm

    def func(self, row, col):
        """
        Matrix-like proxy to function.

        Parameters
        ----------
        row : numpy.ndarray(dtype=numpy.uint64)
            rows, where to compute submatrix
        col: numpy.ndarray(dtype=numpy.uint64)
            columns, where to compute submatrix

        Returns
        -------
        numpy.ndarray of shape `(row.size, self.func_shape, col.size)`
            submatrix on intersection of rows `row` and columns `col`.
            can be scalar, vector or even ndarray
        """
        return self._func(self.row_data, row, self.col_data, col)

    def _build(self, verbose=False):
        """
        Builds cluster trees and defines admissibly far and close nodes.
        
        Divides matrix into blocks hierarchically.

        Parameters
        ----------
        verbose: boolean
            whether or not to output cluster tree generation time,
            number of levels, number of nodes and number of leaves.
        """
        # Reinitialize row (corresponds to receivers)
        # and column (corresponds to transmitters) trees
        # start timer
        t0 = time()
        # row_check[i] stores nodes of column tree, corresponding to
        # self far or self close zone of i-th node of row tree
        row_check = [[0]]
        # row_far[i] will contain indexes of column cluster nodes,
        # that are far from i-th row cluster node
        self.row_far = []
        # row_close[i] will contain indexes of column cluster nodes,
        # that are close to i-th row cluster node
        self.row_close = []
        # row_notransition[i] is True, if i-th row cluster node
        # has empty far zone (including far zone of its parents)
        self.row_notransition = []
        # the same for col_row and col_close
        if self.row_tree is not self.col_tree:
            self.col_far = []
            self.col_close = []
            self.col_notransition = []
        else:
            self.col_far = self.row_far
            self.col_close = self.row_close
            self.col_notransition = self.row_notransition
        # row_tree.aux contains auxiliary data for each node of
        # row cluster tree
        self.row_tree.aux = [self.row_data.compute_aux(
            self.row_tree.index[0])]
        if self.row_tree is not self.col_tree:
            self.col_tree.aux = [self.col_data.compute_aux(
                self.col_tree.index[0])]
        # Start from top of cluster trees
        cur_level = 0
        while (self.row_tree.level[cur_level] <
                self.row_tree.level[cur_level+1]):
            # initialize *_far, *_close and *_notransition
            for i in range(self.row_tree.level[cur_level],
                    self.row_tree.level[cur_level+1]):
                self.row_far.append([])
                self.row_close.append([])
            if self.row_tree is not self.col_tree:
                for i in range(self.col_tree.level[cur_level],
                        self.col_tree.level[cur_level+1]):
                    self.col_far.append([])
                    self.col_close.append([])
            # at first, decide about far and close zones and
            # fill special variables
            for i in range(self.row_tree.level[cur_level],
                    self.row_tree.level[cur_level+1]):
                for j in row_check[i]:
                    if self.row_tree.is_far(i, self.col_tree, j):
                        self.row_far[i].append(j)
                        if self.row_tree is not self.col_tree:
                            self.col_far[j].append(i)
                    else:
                        self.row_close[i].append(j)
                        if self.row_tree is not self.col_tree:
                            self.col_close[j].append(i)
            # fill notransition varaible
            for i in range(self.row_tree.level[cur_level],
                    self.row_tree.level[cur_level+1]):
                if i == 0:
                    self.row_notransition.append(not self.row_far[i])
                else:
                    self.row_notransition.append(not(self.row_far[i] or
                        not self.row_notransition[self.row_tree.parent[i]]))
            if self.row_tree is not self.col_tree:
                for i in range(self.col_tree.level[cur_level],
                        self.col_tree.level[cur_level+1]):
                    if i == 0:
                        self.col_notransition.append(not self.col_far[i])
                    else:
                        self.col_notransition.append(not(self.col_far[i] or
                            not self.col_notransition[
                                self.col_tree.parent[i]]))
            # divide cluster into subclusters if there is something
            # in close zone
            # division for row tree
            for i in range(self.row_tree.level[cur_level],
                    self.row_tree.level[cur_level+1]):
                if (self.row_close[i] and not self.row_tree.child[i] and
                        self.row_tree.index[i].size >
                        self.row_tree.block_size):
                    nonzero_close = False
                    for j in self.row_close[i]:
                        if (self.col_tree.index[j].size >
                                self.col_tree.block_size):
                            nonzero_close = True
                            break
                    if nonzero_close:
                        self.row_tree.divide(i)
            # division for column tree
            if self.row_tree is not self.col_tree:
                for i in range(self.col_tree.level[cur_level],
                        self.col_tree.level[cur_level+1]):
                    if (self.col_close[i] and not self.col_tree.child[i] and
                            self.col_tree.index[i].size >
                            self.col_tree.block_size):
                        nonzero_close = False
                        for j in self.col_close[i]:
                            if (self.row_tree.index[j].size >
                                    self.row_tree.block_size):
                                nonzero_close = True
                                break
                        if nonzero_close:
                            self.col_tree.divide(i)
            # update row_check to check children for close zones
            for i in range(self.row_tree.level[cur_level],
                    self.row_tree.level[cur_level+1]):
                whom_to_check = []
                for j in self.row_close[i]:
                    whom_to_check.extend(self.col_tree.child[j])
                for j in self.row_tree.child[i]:
                    row_check.append(whom_to_check)
            # recompute actual close zone
            # recompute for row tree
            for i in range(self.row_tree.level[cur_level],
                    self.row_tree.level[cur_level+1]):
                tmp_close = []
                if self.row_tree.child[i]:
                    for j in self.row_close[i]:
                        if not self.col_tree.child[j]:
                            tmp_close.append(j)
                    self.row_close[i] = tmp_close
            # recompute for column tree
            if self.row_tree is not self.col_tree:
                for i in range(self.col_tree.level[cur_level],
                        self.col_tree.level[cur_level+1]):
                    tmp_close = []
                    if self.col_tree.child[i]:
                        for j in self.col_close[i]:
                            if not self.row_tree.child[j]:
                                tmp_close.append(j)
                        self.col_close[i] = tmp_close
            # update level of each tree
            self.row_tree.level.append(len(self.row_tree))
            if self.row_tree is not self.col_tree:
                self.col_tree.level.append(len(self.col_tree))
            cur_level += 1
        # update number of levels
        self.num_levels = len(self.row_tree.level)-1
        self.row_tree.num_levels = self.num_levels
        self.col_tree.num_levels = self.num_levels
        if verbose:
            print('Cluster trees are generated in {} seconds'.format(
                time()-t0))
            print('Depth level of each cluster tree: {}'.format(
                self.num_levels))
            print('Row cluster tree')
            print('    nodes : {}'.format(self.row_tree.num_nodes))
            print('    leaves: {}'.format(self.row_tree.num_leaves))
            print('Column cluster tree')
            print('    nodes : {}'.format(self.col_tree.num_nodes))
            print('    leaves: {}'.format(self.col_tree.num_leaves))

    def far_dot(self, x):
        """
        Applies farfield part of operator to vector from left side.

        Parameters
        ----------
        x : numpy.ndarray
            Vector or block-vector to multiply.

        Returns
        -------
        numpy.ndarray
            Result of farfield `A * x`.
        """
        row = self.row_tree
        col = self.col_tree
        row_far = self.row_far
        col_far = self.col_far
        row_data = self.row_data
        col_data = self.col_data
        row_size = row.level[-1]
        col_size = col.level[-1]
        func = self.func
        dtype = self.dtype
        answer_shape = [self.shape[0]]
        answer_shape.extend(self.func_shape)
        answer_shape.extend(x.shape[1:])
        answer = np.zeros(answer_shape, dtype=x.dtype)
        if self.mpi_comm is None:
            for i in range(row_size):
                for j in row_far[i]:
                    answer[row.index[i]] += np.tensordot(func(row.index[i],
                            col.index[j]), x[col.index[j]], 1)
        else:
            mpi_comm = self.mpi_comm
            mpi_rank = mpi_comm.rank
            mpi_size = mpi_comm.size
            lanswer = np.zeros(answer.shape, dtype=answer.dtype)
            for i in range(mpi_rank, row_size, mpi_size):
                for j in row_far[i]:
                    lanswer[row.index[i]] += np.tensordot(func(row.index[i],
                            col.index[j]), x[col.index[j]], 1)
            mpi_comm.Allreduce(lanswer, answer, op=MPI.SUM)
        return answer

    def far_rdot(self, x):
        """
        Applies farfield part of operator to vector from right side.

        Parameters
        ----------
        x : numpy.ndarray
            Vector or block-vector to multiply.

        Returns
        -------
        numpy.ndarray
            Result of farfield `x * A`.
        """
        row = self.row_tree
        col = self.col_tree
        row_far = self.row_far
        col_far = self.col_far
        row_data = self.row_data
        col_data = self.col_data
        row_size = row.level[-1]
        col_size = col.level[-1]
        func = self.func
        dtype = self.dtype
        answer_shape = [self.shape[1]]
        answer_shape.extend(reversed(self.func_shape))
        answer_shape.extend(x.shape[1:])
        answer = np.zeros(answer_shape, dtype=x.dtype)
        if self.mpi_comm is None:
            for i in range(col_size):
                for j in col_far[i]:
                    answer[col.index[i]] += np.tensordot(func(row.index[j],
                            col.index[i]).T, x[row.index[j]], 1)
        else:
            mpi_comm = self.mpi_comm
            mpi_rank = mpi_comm.rank
            mpi_size = mpi_comm.size
            lanswer = np.zeros(answer.shape, dtype=answer.dtype)
            for i in range(mpi_rank, col_size, mpi_size):
                for j in col_far[i]:
                    lanswer[col.index[i]] += np.tensordot(func(row.index[j],
                            col.index[i]).T, x[row.index[j]], 1)
            mpi_comm.Allreduce(lanswer, answer, op=MPI.SUM)
        return answer
    
    def dot(self, x, max_items=125000000):
        """
        Applies linear operator to a given vector from left side.

        Parameters
        ----------
        x : numpy.ndarray
            Vector or block-vector to multiply
        max_items : integer
            Size of buffer (number of elements, same data type as
            matrix elements).

        Returns
        -------
        numpy.ndarray
            Result of `A * x`.
        """
        answer_shape = [self.shape[0]]
        answer_shape.extend(self.func_shape)
        answer_shape.extend(x.shape[1:])
        answer = np.zeros(answer_shape, dtype=x.dtype)
        full_col = np.arange(self.shape[1], dtype=np.uint64)
        if self.mpi_comm is None:
            count = (self.shape[0]*self.shape[1]-1)//max_items+1
            row_step = (self.shape[0]-1)//count+1
            for i in range(count):
                l = np.arange(row_step*i, min(row_step*(i+1), self.shape[0]),
                        dtype=np.uint64)
                answer[l] = np.tensordot(self.func(l, full_col), x, 1)
        else:
            mpi_comm = self.mpi_comm
            mpi_rank = mpi_comm.rank
            mpi_size = mpi_comm.size
            my_start = self.shape[0]*mpi_rank//mpi_size
            my_end = self.shape[0]*(mpi_rank+1)//mpi_size
            count = ((my_end-my_start)*self.shape[1]-1)//max_items+1
            row_step = (my_end-my_start-1)//count+1
            lanswer = np.zeros(answer.shape, answer.dtype)
            for i in range(count):
                l = np.arange(my_start+row_step*i, min(my_start+row_step*(i+1),
                    my_end), dtype=np.uint64)
                lanswer[l] = np.tensordot(self.func(l, full_col), x, 1)
            mpi_comm.Allreduce(lanswer, answer, op=MPI.SUM)
        return answer
    
    def rdot(self, x, max_items=125000000):
        """
        Applies linear operator to a given vector from right side.

        Parameters
        ----------
        x : numpy.ndarray
            Vector or block-vector to multiply
        max_items : integer
            Size of buffer (number of elements, same data type as
            matrix elements).

        Returns
        -------
        numpy.ndarray
            Result of `x * A`.
        """
        answer_shape = [self.shape[0]]
        answer_shape.extend(self.func_shape)
        answer_shape.extend(x.shape[1:])
        answer = np.zeros(answer_shape, dtype=x.dtype)
        full_row = np.arange(self.shape[0], dtype=np.uint64)
        if self.mpi_comm is None:
            count = (self.shape[0]*self.shape[1]-1)//max_items+1
            col_step = (self.shape[1]-1)//count+1
            for i in range(count):
                l = np.arange(col_step*i, min(col_step*(i+1), self.shape[1]),
                        dtype=np.uint64)
                answer[l] = np.tensordot(self.func(full_row, l).T, x, 1)
        else:
            mpi_comm = self.mpi_comm
            mpi_rank = mpi_comm.rank
            mpi_size = mpi_comm.size
            my_start = self.shape[1]*mpi_rank//mpi_size
            my_end = self.shape[1]*(mpi_rank+1)//mpi_size
            count = ((my_end-my_start)*self.shape[0]-1)//max_items+1
            col_step = (my_end-my_start-1)//count+1
            lanswer = np.zeros(answer.shape, answer.dtype)
            for i in range(count):
                l = np.arange(my_start+col_step*i, min(my_start+col_step*(i+1),
                    my_end), dtype=np.uint64)
                lanswer[l] = np.tensordot(self.func(full_row, l).T, x, 1)
            mpi_comm.Allreduce(lanswer, answer, op=MPI.SUM)
        return answer
