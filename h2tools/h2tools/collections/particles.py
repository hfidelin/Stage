from __future__ import print_function, absolute_import, division

__all__ = ['Particles', 'inv_distance', 'log_distance']

import numpy as np
from time import time
from numba import jit
import math

###############################################################################
###                              class Particles                            ###
###############################################################################

class Particles(object):
    """
    Class for computing particle-to-particle interactions.

    Predefined class for N-body problems. Interacting object are simply
    called `particles` and are represented only by vertices (in
    n-dimensional space). Uses recursive inertial bisection to build
    cluster trees. Two subclusters of particles are considered to be
    far (far-field interactions), if distance between centers of
    bounding boxes of subclusters is greater, than diameters of bounding
    boxes.
    
    Parameters
    ----------
    ndim : int
        Dimensionality of space
    count : int
        Number of particles
    vertex : 2-dimensional array
        Coordinates of particles. Shape is `(ndim, count)`.
    """
    def __init__(self, ndim, count, vertex):
        self.ndim = ndim
        self.count = count
        self.vertex = vertex

    def check_far(self, self_aux, other_aux):
        """
        Returns True if bounding boxes are far.
        
        If maximum diagonal of bounding boxes is larger, than distance
        between centers of bounding boxes, then bounding boxes are
        assumed to be far.

        Parameters
        ----------
        self_aux, other_aux: 2-dimensional array of shape (2, *ndim*)
            Each bounding box contains minimum values of corresponding
            coordinates in [0,:] and maximum values of corresponding
            coordinates in [1,:].

        Returns
        -------
        boolean
            `True` if bounding boxes are far, `False` otherwise.
        """
        return Particles.fast_check_far_ndim(self_aux, other_aux, self.ndim)

    @staticmethod
    @jit(nopython=True)
    def fast_check_far(self_aux, other_aux):
        diam0 = self_aux[0, 0]-self_aux[1, 0]
        diam0 *= diam0
        tmp = self_aux[0, 1]-self_aux[1, 1]
        diam0 += tmp*tmp
        tmp = self_aux[0, 2]-self_aux[1, 2]
        diam0 += tmp*tmp
        diam1 = other_aux[0, 0]-other_aux[1, 0]
        diam1 *= diam1
        tmp = other_aux[0, 1]-other_aux[1, 1]
        diam1 += tmp*tmp
        tmp = other_aux[0, 2]-other_aux[1, 2]
        diam1 += tmp*tmp
        dist = self_aux[0, 0]+self_aux[1, 0]-other_aux[0, 0]-other_aux[1, 0]
        dist *= dist
        tmp = self_aux[0, 1]+self_aux[1, 1]-other_aux[0, 1]-other_aux[1, 1]
        dist += tmp*tmp
        tmp = self_aux[0, 2]+self_aux[1, 2]-other_aux[0, 2]-other_aux[1, 2]
        dist += tmp*tmp
        dist *= 0.25
        return dist > diam0 and dist > diam1

    @staticmethod
    @jit(nopython=True)
    def fast_check_far_ndim(self_aux, other_aux, ndim):
        """Fast check of bounding boxes."""
        diam0 = 0.
        diam1 = 0.
        dist = 0.
        for i in range(ndim):
            tmp = self_aux[0, i]-self_aux[1, i]
            diam0 += tmp*tmp
            tmp = other_aux[0, i]-other_aux[1, i]
            diam1 += tmp*tmp
            tmp = self_aux[0, i]+self_aux[1, i]-other_aux[0, i]-other_aux[1, i]
            dist += tmp*tmp
        dist *= 0.25
        return dist > diam0 and dist > diam1
    
    def compute_aux(self, index):
        """
        Computes bounding box of cluster, corresponding to `index`.
        """
        tmp_particles = self.vertex[:,index]
        return np.array([np.min(tmp_particles, axis=1),
            np.max(tmp_particles, axis=1)])

    def divide(self, index):
        """
        Divides cluster, corresponding to indexes `index`.

        Clusters is divided by inertial bisection procedure.
        """
        vertex = self.vertex[:, index]
        center = vertex.mean(axis=1)
        vertex -= center.reshape(-1, 1)
        normal = np.linalg.svd(vertex, full_matrices=0)[0][:,0]
        scal_dot = normal.dot(vertex)
        scal_sorted = scal_dot.argsort()
        scal_dot = scal_dot[scal_sorted]
        k = scal_dot.searchsorted(0)
        return scal_sorted, [0, k, scal_sorted.size]

    def __len__(self):
        return self.count

###############################################################################
###                   interactions for Particles                            ###
###############################################################################

def inv_distance(data1, list1, data2, list2):
    """
    Returns 1/r for each pair of particles from two sets.

    Function 1/r is used as interaction between two particles.

    Parameters
    ----------
    data1 : Python object
        Destination of interactions
    list1 : array
        Indices of particles from `data1` to compute interactions
    data2 : Python object
        Source of interactions
    list2 : array
        Indices of particles from `data1` to compute interactions

    Returns
    -------
    numpy.ndarray(ndim=2)
        Array of interactions of corresponding particles.
    """
    ans = np.ndarray((list1.size, list2.size), dtype=np.float64)
    return inv_distance_numba(data1.ndim, data1.vertex, list1, data2.vertex,
            list2, ans)

@jit(nopython=True)
def inv_distance_numba(ndim, vertex1, list1, vertex2, list2, ans):
    n = list1.size
    m = list2.size
    for i in range(n):
        for j in range(m):
            tmp_l = 0.0
            for k in range(ndim):
                tmp_v = vertex1[k, list1[i]]-vertex2[k, list2[j]]
                tmp_l += tmp_v*tmp_v
            if tmp_l <= 0:
                ans[i, j] = 0
            else:
                ans[i, j] = 1./math.sqrt(tmp_l)
    return ans

def log_distance(data1, list1, data2, list2):
    """
    Returns -log(r) for each pair of particles from two sets.

    Function -log(r) is used as interaction between two particles.

    Parameters
    ----------
    data1 : Python object
        Destination of interactions
    list1 : array
        Indices of particles from `data1` to compute interactions
    data2 : Python object
        Source of interactions
    list2 : array
        Indices of particles from `data1` to compute interactions

    Returns
    -------
    numpy.ndarray(ndim=2)
        Array of interactions of corresponding particles.
    """
    ans = np.ndarray((list1.size, list2.size), dtype=np.float64)
    return log_distance_numba(data1.ndim, data1.vertex, list1, data2.vertex,
            list2, ans)

@jit(nopython=True)
def log_distance_numba(ndim, vertex1, list1, vertex2, list2, ans):
    n = list1.size
    m = list2.size
    for i in range(n):
        for j in range(m):
            tmp_l = 0.0
            for k in range(ndim):
                tmp_v = vertex1[k, list1[i]]-vertex2[k, list2[j]]
                tmp_l += tmp_v*tmp_v
            if tmp_l <= 0:
                ans[i, j] = 0
            else:
                ans[i, j] = -0.5*math.log(tmp_l)
    return ans
