from __future__ import print_function, absolute_import, division

import numpy as np
import math
from numba import jit
from scipy.sparse.linalg import gmres, LinearOperator
from h2tools import Problem, ClusterTree
from h2tools.mcbh import mcbh

###############################################################################
###                             class PCM_Data                              ###
###############################################################################

class PCM_Data(object):
    """
    Methods to find desolvation energy of molecule.
    
    Contains molecule surface as a number of discrete elements, each
    represented by collocation point with area and normal, and a number
    of molecule atoms, represented by coordinates and charges. Also,
    contains routines to solve integral equation and find solvation
    energy in the framework of polarized continuum model (PCM).

    Parameters
    ----------
    count : int
        Number of discrete elements.
    collocation : 2-dimensional array
        Coordinates of collocation points. Shape is `(3, count)`.
    area : 1-dimensional array
        Area of each discrete element.
    normal : 2-dimensional array
        Normals for each discrete element. Shape is `(3, count)`.
    atom_count : int
        Number of atoms.
    atom_position : 2-dimensional array.
        Radius vectors of atoms. Shape is `(3, atom_count)`.
    atom_charge : 1-dimensional array.
        Values of charges of each atom in molecule.
    eps : float
        Permittivity of solvent.
    """
    def __init__(self, count, collocation, area, normal, atom_count,
            atom_position, atom_charge, eps):
        self.count = count
        self.collocation = collocation
        self.area = area
        self.normal = normal
        self.atom_count = atom_count
        self.atom_position = atom_position
        self.atom_charge = atom_charge
        self.eps = eps
        self.dim = 3

    def __len__(self):
        """Number of discrete elements of surface."""
        return self.count

    @staticmethod
    def from_file(fname, eps):
        """
        Reads molecule configuration from file.
        
        Format of input file is following: first line contains number of
        atoms, each following line contains 3 coordinates and charge of
        atom (split by space), then first line after row with last atom
        contains number of discrete surface elements, followed by rows,
        containing 3 coordinates, 3 coordinates of normal and area for
        each discrete element. Permittivity of solvent is given by `eps`
        parameter.
        
        Parameters
        ----------
        fname : string
            Name of file to read.
        eps : float
            Permittivity of solvent.
        """
        f = open(fname, 'r')
        acount = (int)(f.readline()[:-1])
        atom = np.ndarray((3,acount), dtype=np.float64)
        acharge = np.ndarray(acount, dtype=np.float64)
        for i in range(acount):
            line = f.readline()
            t = line.split()
            atom[0,i], atom[1,i], atom[2,i] =\
                    (float)(t[0]), (float)(t[1]), (float)(t[2])
            acharge[i] = (float)(t[3])
        count = (int)(f.readline()[:-1])
        surface = np.ndarray((3, count), dtype=np.float64)
        normal = np.ndarray((3, count), dtype=np.float64)
        area = np.ndarray(count, dtype=np.float64)
        for i in range(count):
            line = f.readline()
            t = line.split()
            surface[0,i], surface[1,i], surface[2,i] =\
                    (float)(t[0]), (float)(t[1]), (float)(t[2])
            normal[0,i], normal[1,i], normal[2,i] =\
                    (float)(t[3]), (float)(t[4]), (float)(t[5])
            area[i] = (float)(t[6])
        f.close()
        return PCM_Data(count, surface, area, normal, acount, atom, acharge,
                eps)

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
        return PCM_Data.fast_check_far(self_aux, other_aux)

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

    def compute_rhs(self, max_items=100000000):
        """
        Computes right hand side to find solvation energy.
        """
        block_count = (self.count*self.atom_count-1)//max_items+1
        block_size = (self.count-1)//block_count+1
        l1 = np.arange(self.atom_count, dtype=np.uint64)
        ans = np.ndarray(self.count, dtype=np.float64)
        for i in range(block_count):
            start = block_size*i
            end = min(block_size*(i+1), self.count)
            l = np.arange(start, end, dtype=np.uint64)
            tmp = rhs(self, l, self, l1)
            ans[start:end] = tmp.dot(self.atom_charge)
        return ans

    def compute_qtd(self, max_items=100000000):
        """
        Computes auxiliary value, required to find solvation energy.
        """
        block_count = (self.count*self.atom_count-1)//max_items+1
        block_size = (self.count-1)//block_count+1
        l1 = np.arange(self.atom_count, dtype=np.uint64)
        ans = np.ndarray(self.count, dtype=np.float64)
        for i in range(block_count):
            start = block_size*i
            end = min(block_size*(i+1), self.count)
            l = np.arange(start, end, dtype=np.uint64)
            tmp = energy(self, l, self, l1)
            ans[start:end] = tmp.dot(self.atom_charge)
        return ans

    def compute_diag(self, matrix):
        """
        Computes diagonal of a matrix by special integral identity.
        """
        return self.eps/(self.eps+1.)-matrix.rdot(np.ones(self.count))

    def find_solvation_energy(self, eps=1e-6, block_size=25, onfly=True,
            maxiter=999, verbose=False):
        """
        Computes solvation energy of a given molecule.

        Parameters
        ----------
        eps : float
            Relative tolerance.
        block_size : int
            Maximum size of leaf nodes in cluster tree.
        onfly : boolean
            If True, does not store some submatrices (saves a lot of
            memory, but each matrix-vector operation requires
            computation of these submatrices on demand).
        maxiter : int
            Maximum number of iterations for GMRES.
        verbose : boolean
            If True, MCBH factorization will produce additional output.

        Returns
        -------
        float
            Solvation energy of molecule.
        """
        def dot(x):
            return matrix.dot(x)+diag*x
        def simple_print(r):
            print(simple_print.__dict__['i'], r)
            simple_print.__dict__['i'] += 1
        rhs = self.compute_rhs()
        lo = LinearOperator((self.count, self.count), dot, dtype=rhs.dtype)
        simple_print.__dict__['i'] = 1
        tree = ClusterTree(self, block_size=block_size)
        problem = Problem(electro, tree, tree, symmetric=0, verbose=verbose)
        matrix = mcbh(problem, tau=eps, iters=1, onfly=onfly, verbose=verbose)
        diag = self.compute_diag(matrix)
        sol = gmres(lo, rhs, tol=eps, callback=simple_print, maxiter=maxiter)
        w = sol[0]
        special_const = 0.5*1.602176565e-19*1.602176565e-19*\
                8.9875517873681764e+9*1.0e+10*6.02214129e+23/4184.0
        qtd = self.compute_qtd()
        return special_const*qtd.dot(w), simple_print.__dict__['i']-1

###############################################################################
###                      Interaction of PCM_Data                            ###
###############################################################################

def electro(data1, list1, data2, list2):
    """
    Interactions of discrete elements of surface of molecule.
    """
    ans = np.ndarray((list1.size, list2.size), dtype=np.float64)
    return electro_numba(data1.collocation, data1.area, data1.normal,
            data1.eps, list1, list2, ans)

@jit(nopython=True)
def electro_numba(collocation, area, normal, eps, list1, list2, ans):
    n = list1.size
    m = list2.size
    for j in range(m):
        for i in range(n):
            if list1[i] == list2[j]:
                ans[i, j] = 0
                continue
            vec0 = collocation[0, list1[i]]-collocation[0, list2[j]]
            vec1 = collocation[1, list1[i]]-collocation[1, list2[j]]
            vec2 = collocation[2, list1[i]]-collocation[2, list2[j]]
            norm = vec0*vec0+vec1*vec1+vec2*vec2
            norm = 1.0/(norm*math.sqrt(norm))
            vec0 *= norm
            vec1 *= norm
            vec2 *= norm
            norm = vec0*normal[0, list1[i]]+vec1*normal[1, list1[i]]+\
                    vec2*normal[2, list1[i]]
            ans[i, j] = norm*area[list1[i]]*(eps-1)/(4*math.pi*(eps+1))
    return ans

def rhs(data1, list1, data2, list2):
    """
    Interactions of discrete elements of surface and atoms of molecule.
    """
    ans = np.ndarray((list1.size, list2.size), dtype=np.float64)
    return rhs_numba(data1.collocation, data1.area, data1.normal,
            data1.atom_position, data1.eps, list1, list2, ans)

@jit(nopython=True)
def rhs_numba(collocation, area, normal, atom_position, eps, list1, list2,
        ans):
    n = list1.size
    m = list2.size
    for j in range(m):
        for i in range(n):
            vec0 = collocation[0, list1[i]]-atom_position[0, list2[j]]
            vec1 = collocation[1, list1[i]]-atom_position[1, list2[j]]
            vec2 = collocation[2, list1[i]]-atom_position[2, list2[j]]
            norm = vec0*vec0+vec1*vec1+vec2*vec2
            norm = 1.0/(norm*math.sqrt(norm))
            vec0 *= norm
            vec1 *= norm
            vec2 *= norm
            norm = vec0*normal[0, list1[i]]+vec1*normal[1, list1[i]]+\
                    vec2*normal[2, list1[i]]
            ans[i, j] = norm*area[list1[i]]*(1-eps)/(4*math.pi*(eps+1))
    return ans

def energy(data1, list1, data2, list2):
    """
    Auxiliary interactions of discrete elements and atoms of molecule.
    """
    ans = np.ndarray((list1.size, list2.size), dtype=np.float64)
    return energy_numba(data1.collocation, data1.atom_position, list1, list2,
            ans)

@jit(nopython=True)
def energy_numba(collocation, atom_position, list1, list2, ans):
    n = list1.size
    m = list2.size
    for j in range(m):
        for i in range(n):
            vec0 = collocation[0, list1[i]]-atom_position[0, list2[j]]
            vec1 = collocation[1, list1[i]]-atom_position[1, list2[j]]
            vec2 = collocation[2, list1[i]]-atom_position[2, list2[j]]
            norm = vec0*vec0+vec1*vec1+vec2*vec2
            ans[i, j] = 1.0/math.sqrt(norm)
    return ans
