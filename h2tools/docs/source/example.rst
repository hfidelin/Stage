Simple example of how to work with ``h2tools``
==============================================

Let assume, we need to approximate functional matrix
:math:`A_{ij} = f(x_i, y_j)` for some given sets of objects
:math:`X=\{x_1, \ldots, x_N\}` and :math:`Y=\{y_1, \ldots, y_M\}`.
Obviously, :math:`X` corresponds to rows and :math:`Y` corresponds to columns.
Let additionally denote :math:`X` and :math:`Y` as a destination and source
sets accordingly.
For simplicity, we assume interaction function :math:`f` returns only scalar values.
Then, following steps are required to build approximation with MCBH method:

1. Prepare cluster trees :math:`\mathcal{T}_{I}` and :math:`\mathcal{T}_{J}`
   for destination and source sets correspondingly,
2. Prepare block cluster tree by finding admissibly far and close nodes for
   :math:`\mathcal{T}_{I}` and :math:`\mathcal{T}_{J}`,
3. Build approximation with given parameters.

To make each step clearer, we show it in following example:

.. code-block:: python

    # At first, we generate 10000 particles's positions randomly with given
    # random seed
    import numpy as np
    np.random.seed(0)
    position = np.random.randn(3, 10000)
    # We use predefined `particles` submodule of `h2tools.collections`
    # This submodule contains data class, cluster division code and
    # interaction function
    from h2tools.collections import particles
    # Create particles data object
    data = particles.ParticlesData(position)
    # Initialize cluster tree with data object
    from h2tools import ClusterTree
    tree = ClusterTree(data, block_size=25)
    # Use function inv_distance, which returns inverse of distance
    # between particles
    func = particles.inv_distance
    # Create object for whole problem (block cluster tree + function)
    from h2tools import Problem
    problem = Problem(func, tree, tree, symmetric=1, verbose=0)
    # Build approximation of matrix in H^2 format
    # with relative accuracy parameter 1e-4,
    # 0 (zero) iterations of MCBH algorithm
    # and with True verbose flag
    from h2tools.mcbh import mcbh
    matrix = mcbh(problem, tau=1e-4, iters=0, verbose=1)
    # If you have package `pypropack` installed,
    # you can measure relative error of approximation
    # in spectral norm
    matrix.diffnorm()
