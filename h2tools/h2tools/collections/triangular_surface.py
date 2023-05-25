"""
Predefined class for boundary integral equation in 3-dimensional space.
Stores triangulation as well as normals. Uses inertial bisection of
collocation points to build cluster tree.
"""

from __future__ import print_function, absolute_import, division

import numpy as np
from numba import jit, vectorize
import math

###############################################################################
###                       class TriangularSurface                           ###
###############################################################################

class TriangularSurface(object):
    """
    Triangulization of surface, defined by triangles and normals.

    Surface is assumed to be in 3-dimensional space.
    Uses inertial bisection of collocation points to build cluster
    tree.

    Parameters
    ----------
    vertex : 2-dimensional array
        Coordinates of vertices. Shape is `(3, vertex_count)`,
        where `vertex_count` is a total number of vertices.
    triangle : 1-dimensioanl array
        `triangle[3*i:3*i+3]` correspond to indexes of vertices of
        `i`-th triangle.
    collocation : 2-dimensional array, optional
        Coordinates of collocation points. Shape is `(3, count)`,
        where `count` is a total number of triangles.
    area : 1-dimensional array, optional
        area of each triangle.
    normal : 2-dimensional array, optional
        normal of each triangle. Shape is the same as `collocation`.

    Attributes
    ----------

    See Also
    --------

    Examples
    --------

    """
    def __init__(self, vertex, triangle, collocation=None, area=None,
            normal=None):
        self.vertex = vertex
        self.triangle = triangle
        self.ndim = 3
        self.count = triangle.shape[1]
        if collocation is None:
            self.collocation = np.ndarray((3, self.count), dtype=vertex.dtype)
            for i in range(self.count):
                self.collocation[:, i] = self.vertex[:,
                        self.triangle[:, i]].mean(axis=1)
        else:
            self.collocation = collocation
        if area is None or normal is None:
            self.area = np.ndarray(self.count, dtype=np.float)
            self.normal = np.ndarray((3, self.count), dtype=np.float,
                    order='F')
            for i in range(self.count):
                tri = self.triangle[:, i]
                x = self.vertex[:,tri[1:3]]-self.vertex[:,tri[0]].reshape(-1,1)
                tmp_normal = np.array([x[1, 1]*x[2, 0]-x[2, 1]*x[1, 0],
                    x[2, 1]*x[0, 0]-x[0, 1]*x[2, 0],
                    x[0, 1]*x[1, 0]-x[1, 1]*x[0, 0]])
                norm = np.linalg.norm(tmp_normal)
                self.normal[:, i] = tmp_normal/norm
                self.area[i] = 0.5*norm
        if area is not None:
            self.area = area
        if normal is not None:
            self.normal = normal

    def __len__(self):
        """Returns number of triangles in surface grid."""
        return self.count

    @staticmethod
    def from_dat(f):
        """Reads triangular surface from .dat files (in special format)."""
        if type(f) is str:
            f = open(f, 'r')
        p_count = 0
        v_count = 0
        polygon = []
        vertex = []
        n_objects = int(f.readline())
        for i_obj in range(n_objects):
            n_modules = int(f.readline())
            for i_mod in range(n_modules):
                f.readline()
                n_quad = int(f.readline())
                f.readline()
                for i_quad in range(n_quad):
                    tmp = f.readline().split(' ')
                    tmp.pop()
                    for i in range(len(tmp)):
                        tmp[i] = float(tmp[i])
                    stmp = np.array(tmp).reshape(4,3)
                    b01 = int((stmp[0] == stmp[1]).all())
                    b02 = int((stmp[0] == stmp[2]).all())
                    b03 = int((stmp[0] == stmp[3]).all())
                    b12 = int((stmp[1] == stmp[2]).all())
                    b13 = int((stmp[1] == stmp[3]).all())
                    b23 = int((stmp[2] == stmp[3]).all())
                    if (b01+b02+b03+b12+b13+b23)/int(True) > 1:
                        print('nothing should be added')
                        continue
                    if b01 == 1 or b02 == 1:
                        tmp[:3] = tmp[-3:]
                    if b12 == 1:
                        tmp[3:6] = tmp[-3:]
                    if (b01+b02+b03+b12+b13+b23)/int(True) == 1:
                        vertex.extend(tmp[:-3])
                        polygon.extend([(v_count, v_count+1, v_count+2)])
                        v_count += 3
                        p_count += 1
                        continue
                    vertex.extend(tmp)
                    polygon.extend([(v_count, v_count+1, v_count+2),
                        (v_count+2, v_count+3, v_count)])
                    v_count += 4
                    p_count += 2
                #f.readline()
        p = np.array(polygon, dtype=np.uint64).T.copy()
        v = np.array(vertex).reshape(len(vertex)/3, 3).T.copy()
        f.close()
        self = TriangularSurface(v, p)
        return self

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
        return TriangularSurface.fast_check_far(self_aux, other_aux)

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
        center = vertex.mean(axis = 1)
        vertex -= center.reshape(-1, 1)
        normal = np.linalg.svd(vertex, full_matrices=0)[0][:,0]
        scal_dot = normal.dot(vertex)
        scal_sorted = scal_dot.argsort()
        scal_dot = scal_dot[scal_sorted]
        k = scal_dot.searchsorted(0)
        return scal_sorted, [0, k, scal_sorted.size]

###############################################################################
###                  interactions of TriangularSurface                      ###
###############################################################################

def integral_inverse_r3(data1, list1, data2, list2):
    """
    Returns pairwise integrals of r^{-3} over corresponding areas.

    Each integral is calculated from collocation point of triangles of
    `data1` over areas of triangles of `data2`.

    Parameters
    ----------
    data1 : TringularSurface
        Surface grid.
    list1 : 1-dimensional array
        Indexes of triangles from `data1`.
    data2 : TriangularSurface
        Surface grid.
    list2 : 1-dimensional array
        Indexes of triangles from `data2`.
    """
    ans = np.ndarray((list1.size, list2.size), dtype=np.float64)
    tmp = np.ndarray((7, 3), dtype=np.float64)
    return integral_inverse_r3_numba(data1.collocation, data1.area,
            data1.normal, list1, data2.vertex, data2.triangle, list2, tmp, ans)

@jit(nopython=True)
def integral_inverse_r3_3d_side_numba(x1, x2, out):
    d210 = x2[0]-x1[0]
    d211 = x2[1]-x1[1]
    d212 = x2[2]-x1[2]
    r01 = x1[0]*x1[0]+x1[1]*x1[1]+x1[2]*x1[2]
    r02 = x2[0]*x2[0]+x2[1]*x2[1]+x2[2]*x2[2]
    r21 = d210*d210+d211*d211+d212*d212
    a0 = x1[1]*d212-x1[2]*d211
    a1 = x1[2]*d210-x1[0]*d212
    a2 = x1[0]*d211-x1[1]*d210
    ra = a0*a0+a1*a1+a2*a2
    b1 = x1[0]*d210+x1[1]*d211+x1[2]*d212
    b2 = x2[0]*d210+x2[1]*d211+x2[2]*d212
    b = (b1/math.sqrt(r01)-b2/math.sqrt(r02))/ra
    out[0] -= b*a0
    out[1] -= b*a1
    out[2] -= b*a2
    
@jit(nopython=True)
def integral_inverse_r3_numba(dst_collocation, dst_area, dst_normal, dst_list,
        src_vertex, src_triangle, src_list, tmp, ans):
    n = dst_list.size
    m = src_list.size
    for j in range(m):
        tri = src_list[j]
        v0 = src_triangle[0, tri]
        v1 = src_triangle[1, tri]
        v2 = src_triangle[2, tri]        
        for l in range(3):
            tmp[0, l] = src_vertex[l, v0]
            tmp[1, l] = src_vertex[l, v1]
            tmp[2, l] = src_vertex[l, v2]
        for i in range(n):
            dst = dst_list[i]
            for l in range(3):
                tmp[3, l] = dst_collocation[l, dst]-tmp[0, l]
                tmp[4, l] = dst_collocation[l, dst]-tmp[1, l]
                tmp[5, l] = dst_collocation[l, dst]-tmp[2, l]
                tmp[6, l] = 0.0
            integral_inverse_r3_3d_side_numba(tmp[3], tmp[4], tmp[6])
            integral_inverse_r3_3d_side_numba(tmp[4], tmp[5], tmp[6])
            integral_inverse_r3_3d_side_numba(tmp[5], tmp[3], tmp[6])
            ans[i, j] = dst_area[dst]*(tmp[6, 0]*dst_normal[0, dst]+
                    tmp[6, 1]*dst_normal[1, dst]+tmp[6, 2]*dst_normal[2, dst])
    return ans
