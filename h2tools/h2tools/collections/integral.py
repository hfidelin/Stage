"""
Predefined class for volume and boundary integral equations in
2-dimensional or 3-dimensional space, discretized by one of three
typical methods: Galerkin, collocation or Nystrom.
"""

from __future__ import print_function, absolute_import, division

__all__ = ['DiscretizedArea']

import numpy as np
from numba import jit
import math

###############################################################################
###                         class DiscretizedArea                           ###
###############################################################################

class DiscretizedArea(object):
    """
    Stores area as discrete elements, normals and collocation points.
    
    Helpful in discretizations of integral equations by one
    of 3 typical methods: Galerkin, collocation or Nystrom.
    Every parameter, except `ndim` and `count` is optional.
    Non-optional parameters depend on discretization type.
    
    Uses inertial bisection of collocation points for cluster tree
    generation. If collocation points are not set up, they are computed
    as geometrical centers of discrete elements.

    Parameters
    ----------
    ndim : int
        Dimensionality of space.
    count : int
        Number of discrete elements.
    vertex : 2-dimensional array
        Coordinates of vertices. Shape is `(ndim, count)`.
    element: 1-dimensional array
        Information on discrete elements is one after another.
        Each discrete element is presented as number of corresponding
        vertices, followed by their indexes.
    normal : 2-dimensional array
        Normals for each discrete element. Shape is `(ndim, count)`.
    area : 1-dimensional array
        Area of each discrete element.
    collocation : 2-dimensional array
        Coordinates of collocation points. Shape is `(ndim, count)`.

    Attributes
    ----------
    element_size : array of int
        Contains number of vertices, corresponding to discrete element
    element_start : array of int
        Where indexes of corresponding vertices are located in
        `elements`

    See Also
    --------

    Examples
    --------

    """

    def __init__(self, ndim, count, vertex=None, element=None, normal=None,
            area=None, collocation=None):
        self.ndim = ndim
        self.count = count
        if vertex is None and element is None and collocation is None:
            raise ValueError("Boundary must be set as by pair of `vertex` and "
                "`elements` and/or by `collocation` points.")
        if vertex is not None and element is None:
            raise ValueError("`polygon` parameter must not be `None`.")
        if vertex is None and element is not None:
            raise ValueError("`elements` parameter must not be `None`.")
        self.vertex = vertex
        self.element = element
        if element is not None:
            self.element_size = np.zeros(count, dtype=np.int)
            self.element_start = np.zeros(count, dtype=np.int)
            cur_pointer = 0
            for i_element in range(count):
                self.element_size[i_element] = self.element[cur_pointer]
                self.element_start[i_element] = cur_pointer+1
                cur_pointer += self.element[cur_pointer]+1
        else:
            self.element_size = None
            self.element_start = None
        if collocation is None:
            self._compute_collocation()
        else:
            self.collocation = collocation
        self.normal = normal
        self.area = area

    def _compute_collocation(self):
        """Computes collocation points as centers of polygons."""
        collocation = np.zeros((ndim, count), dtype=self.vertex.dtype)
        for i in range(self.count):
            start = self.element_start[i]
            end = start+self.element_size[i]
            my_vertices = self.element[start:end]
            for j in my_vertices:
                collocation[:, i] += self.vertex[:, j]
        self.collocation = collocation

    def __len__(self):
        return self.count

    @staticmethod
    def from_dat(fname):
        pass

    def check_far(self, self_aux, other_aux):
        """
        Checks if two bounding boxes are far or close.
        
        If maximum diagonal of bounding boxes is larger, than distance
        between centers of bounding boxes, then bounding boxes are
        assumed to be far and function returns `True`.

        Parameters
        ----------
        self_aux, other_aux : `numpy.ndarray` of shape `(2, ndim)`
            Each bounding box contains minimum values of corresponding
            coordinates in [0,:] and maximum values of corresponding
            coordinates in [1,:].

        Returns
        -------
        boolean
            `True` if bounding boxes are far, `False` otherwise.
        """
        return Volume.fast_check_far_ndim(self_aux, other_aux, self.ndim)

    @staticmethod
    @jit(nopython=True)
    def fast_check_far(self_aux, other_aux):
        """Fast check of bounding boxes."""
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
        tmp_particles = self.collocation[:,index]
        return np.array([np.min(tmp_particles, axis=1),
            np.max(tmp_particles, axis=1)])

    def divide(self, index):
        """
        Divides cluster, corresponding to indexes `index`.

        Clusters is divided by inertial bisection procedure, applied to
        collocation points of discrete elements.
        """
        vertex = self.collocation[:, index]
        center = vertex.mean(axis=1)
        vertex -= center.reshape(self.dim, 1)
        normal = np.linalg.svd(vertex, full_matrices=0)[0][:,0]
        scal_dot = normal.dot(vertex)
        scal_sorted = scal_dot.argsort()
        scal_dot = scal_dot[scal_sorted]
        k = scal_dot.searchsorted(0)
        return scal_sorted, [0, k, scal_sorted.size]
