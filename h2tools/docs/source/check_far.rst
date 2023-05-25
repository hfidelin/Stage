Check_far function
==================


check_far function must be coded with help of Cython or Numba to make it as fast, as possible, since it is most used function in cluster tree generation.
Even cluster division code runs much times less.
For example, you have 5000 nodes of cluster tree and each cluster tree is divided into 2 sublcusters.
It means that cluster division function was called 2500 times.
But function check_far runs several times for each node!
So, if tree generation takes too much time -- consider accelerating it by means of Cython or Numba.

We compared 3 functions:

.. code-block:: python

    import numpy as np
    def check_far(self_aux, other_aux):
        """Input arguments contain bounding box of clusters in
	following format: [min_coordinate_for_each_axis,
	max_coordinate_for_each_axis].
	Returns True if maximum of diameters of bounding boxes is
	less, than distance between centers of bounding boxes.
	Uses numpy.linalg.norm to measure diameters and distance."""
        diam0 = np.linalg.norm(self_aux[0]-self_aux[1])
        diam1 = np.linalg.norm(other_aux[0]-other_aux[1])
        dist = 0.5*np.linalg.norm(self_aux[0]+self_aux[1]-other_aux[0]-other_aux[1])
        return dist > max(diam0, diam1)
	
    def check_far2(self_aux, other_aux):
        """Only difference with check_far function is that check_far2
	uses scalar dot (numpy.dot) instead of numpy.linalg.norm"""
	diam0 = self_aux[0]-self_aux[1]
        diam0 = diam0.dot(diam0)
        diam1 = other_aux[0]-other_aux[1]
        diam1 = diam1.dot(diam1)
        dist = self_aux[0]+self_aux[1]-other_aux[0]-other_aux[1]
        dist = 0.25*dist.dot(dist)
        return dist > max(diam0, diam1)
	
    from numba import jit
    @jit(nopython=True)
    def check_far3(self_aux, other_aux):
        """Optimized with numba"""
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

Simple comparison for N-body problem (20000 particles, block_size=12):

.. code-block:: python

    check_far: 12.8 seconds
    check_far2: 7.1 seconds
    check_far3: 1.5 seconds
