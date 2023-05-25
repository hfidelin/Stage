"""
Each hierarchical matrix is defined by two cluster trees, corresponding
to rows and columns of matrix. Row and column cluster trees are
hierarchical, root nodes correspond to all rows and columns
correspondingly. 
"""

from __future__ import print_function, absolute_import, division

__all__ = ['ClusterTree']

import numpy as np
from time import time

##############################
##  class SmartIndex
##############################

class SmartIndex(object):
    """
    Stores only view to index and information about each node.

    It is only used in `ClusterTree` class for convenient work with
    indexes. Main reason this is implemented separately from
    `ClusterTree` is easily readable syntax: `index[key]` returns view
    to subarray of array `index`, corresponding to indices of node
    `key`.

    Parameters
    ----------
    size : integer
        Number of objects in cluster

    Attributes
    ----------
    index: 1-dimensional array
        Permutation array such, that indexes of objects, corresponding
        to the same subcluster, are located one after each other.
    node: list of tuples
        Indexes of `i`-th node of cluster tree are
        `index[node[i][0]:node[i][1]]`.
    """

    def __init__(self, size):
        self.index = np.arange(size, dtype=np.uint64)
        self.node = [(0, size)]

    def __getitem__(self, key):
        """Get indices for cluster `key`."""
        return self.index[slice(*self.node[key])]

    def __setitem__(self, key, value):
        """
        Set indices for cluster `key`.

        Changes only main index array.
        """
        self.index[slice(*self.node[key])] = value

    def add_node(self, parent, node):
        """Add node, that corresponds to `index[node[0]:node[1]]`."""
        start = self.node[parent][0]+node[0]
        stop = self.node[parent][0]+node[1]
        self.node.append((start, stop))

    def __len__(self):
        return len(self.node)

#################################
##    class ClusterTree
#################################

class ClusterTree(object):
    """
    Stores cluster tree as lists of parents and children.

    Parameters
    ----------
    data : Python object
        Data, that is required to be clustered with cluster tree.
    block_size : integer
        Maximum size of leaf-node (number of objects, corresponding to
        leaf-node) with near-field interactions.

    Attributes
    ----------
    block_size : integer
        Maximum size of leaf-node (number of objects, corresponding to
        leaf-node) with near-field interactions.
    data : Python object
        Data, that is required to be clustered with cluster tree.
    index : instance of `SmartIndex`
        Indexes of clusters. For `i`-th node of cluster tree, `index[i]`
        is a `numpy.ndarray(ndim=1,dtype=numpy.uint64)`, containing
        indexes of corresponding data objects.
    parent : list of integers
        List of parent nodes (defined by its number, `-1` for
        root-node).
    child : list of lists of integers
        For `i`-th node of cluster tree, `child[i]` is a list of
        child-nodes (defined by its number).
    leaf : list of integers
        List of leaf-nodes, defined by their numbers.
    level : list of integers
        For `l` level of depth of cluster tree, all corresponding nodes
        have numbers from `level[l]` (incslusively) to `level[l+1]`
        (exclusively).
    num_levels : integer
        Number of levels of depth of cluster tree.
    num_leaves : integer
        Number of total leaf-nodes in cluster tree.
    num_nodes : integer
        Number of total nodes in cluster tree.
    """

    def __init__(self, data, block_size):
        self.block_size = block_size
        self.data = data
        self.index = SmartIndex(len(data))
        self.parent = [-1]
        self.child = [[]]
        self.leaf = [0]
        self.level = [0, 1]
        self.num_levels = 0
        self.num_leaves = 1
        self.num_nodes = 1

    def is_consistent(self):
        """
        Check inner structure of cluster tree.

        Checks cluster for consistency (child-parent bonds, objects of each
        cluster node are next to each other). Currently, no need to use this
        function.
        """
        data_size, index, parent, child = len(self.data), self.index,\
                self.parent, self.child
        if index.index.dtype is not np.dtype(np.uint64):
            raise TypeError("index must be array of unsigned 64-bit integers")
        tree_size = len(parent)
        if len(child) != tree_size or len(index) != tree_size:
            raise IndexError("parent, child and node arrays must have the"
                " same length")
        sorted_index = index.index.copy()
        sorted_index.sort()
        for i in range(data_size-1):
            if sorted_index[i] == sorted_index[i+1]:
                raise IndexError("elements in index array must be different")
        if sorted_index[0] < 0 or sorted_index[-1] >= data_size:
            raise IndexError("elements in index array must be in interval"
                " [0;N), where N is a size of data")
        num_links = tree_size-1
        for i in range(tree_size):
            check_point = index.node[i][0]
            for j in child[i]:
                if parent[j] != i:
                    raise IndexError("each child of a node must have it as a"
                        " parent")
                if index.node[j][0] != check_point:
                    raise IndexError("indexes of children must go one after"
                        " other")
                check_point = index.node[j][1]
            if child[i] and index.node[i][1] != check_point:
                    raise IndexError("indexes of children must go one after"
                        " other")
            num_links -= len(child[i])
        if num_links != 0:
            raise IndexError("parent and child arrays do not correspond to"
                " each other")
        self.consistent = True
        return True

    def is_far(self, i, other_tree, j):
        """
        Checks if two clusters are far or close to each other.
        
        This check is done only with auxiliary information about each
        cluster (which is computed by `data.cluster_auxiliary_data_func`).
        """
        if i <= j:
            result = self.data.check_far(self.aux[i], other_tree.aux[j])
        else:
            result = other_tree.data.check_far(other_tree.aux[j], self.aux[i])
        return result

    def divide(self, key):
        """
        Divides cluster with given ID into sublcusters.
        
        Fills every required field (such as list of parents, children)
        and so on.
        """
        index = self.index[key]
        new_index, subclusters = self.data.divide(index)
        test_index = new_index.copy()
        new_index = index[new_index]
        test_index.sort()
        if (test_index != np.arange(index.size)).any():
            raise Error("bad error in tree construction")
        last_ind = subclusters[0]
        for i in range(len(subclusters)-1):
            next_ind = subclusters[i+1]
            if next_ind < last_ind:
                raise Error("children indices must be one after other")
            self.index.add_node(key, (last_ind, next_ind))
            last = len(self.parent)
            self.parent.append(key)
            if self.child[key]:
                self.num_leaves += 1
            self.num_nodes += 1
            self.child[key].append(last)
            self.child.append([])
            self.aux.append(self.data.compute_aux(
                new_index[last_ind:next_ind]))
            last_ind = next_ind
        if next_ind != test_index.size:
            raise Error("Sum of sizes of children must be the same as"
                " size of the parent")
        self.index[key] = new_index

    @staticmethod
    def fromfile(self, fname):
        """
        Reserved for function, that reads tree information from file.
        """
        pass

    def tofile(self, fname):
        """
        Reserved for function, that writes tree information to file.
        """
        pass

    def copy(self):
        """
        Reserved for copy function of tree object.
        """
        pass

    def __getitem__(self, key):
        """
        Reserved for purpose of tree visualization.
        """
        if key < 0 or key >= len(self):
            raise IndexError("wrong node index")

    def __len__(self):
        """
        Number of nodes in current tree.
        """
        return len(self.parent)

    def draw(self, key=0):
        """
        Reserved for drawing tree with accent on node `key`.
        """
        pass

    def __repr__(self):
        """
        Returns information on cluster tree in a string format.

        Returns following information about cluster tree: name of
        cluster tree class, name of data class, number of levels of
        depth of cluster tree, number of nodes and number of leaf-nodes.
        """
        return 'Class: {}\nData class: {}\nLevels: {}\nNodes: {}\nLeaves: {}'\
                .format(self.__class__.__name__, self.data.__class__.__name__,
                self.num_levels, len(self), self.num_leaves)
