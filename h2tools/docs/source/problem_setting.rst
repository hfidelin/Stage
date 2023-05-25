Problem setting or how to use ``h2tools``
=========================================

Main feature of ``h2tools`` is a special method of approximation of given
matrix by :math:`\mathcal{H}^2`-matrix directly from some of its entries.
So, problem setting here is about constructing approximation and working
with approximant. To build any hierarchical approximation we need
**hierarchical** (**mosaic**) partitioning, which is usually based on
**block cluster tree**.

Assume we have matrix :math:`A`, which corresponds to our initial problem.
Since size of the initial problem can be incredibly large, we cannot store
whole matrix :math:`A` in memory and number of operations for a simple
matrix-vector operation can be too big (linear in size of matrix :math:`A`).
So, we need a function, so-called **block function**, that will return
submatrix of :math:`A`.

Since matrix :math:`A` can be assumed as linear operator, we denote rows
and columns of :math:`A` as range of values and domain of corresponding
linear operator (discretized initial problem). In accordance to N-body
problems, we also name rows and columns of :math:`A` as destinations and
sources of corresponding interaction.

To make it short, there are following virtual stages of working with
``h2tools``:

1. Build discretization of initial problem, so that it has corresponding matrix
   :math:`A` and **block function**.
2. Build cluster tree for destination discrete elements (rows of :math:`A`) and
   and for source discrete elements (columns of :math:`A`). Resulting cluster
   trees are also called row and column cluster trees.
3. Link row and column cluster trees into **block cluster tree** by acquiring
   near-field and far-field relations between row and column cluster trees.
4. Build hierarchical approximation by passing **block function** and **block
   cluster tree** as arguments to approximation procedure.
5. Use resulting approximant in any way you like.

Easiest way to understand following documentation is to read
``minimal_example`` in examples directory with ``ipython notebook``.


1a. Building discretization
---------------------------

There is no special procedure in ``h2tools``, which will build discretization.
However, **Python** objects, containing discretizations (range of values and
domain) or destinations and sources, should have special methods. These
methods are used to build **hierarchical partitioning**.

Assume we have data object ``data`` (i.e. corresponding to
destinations/range of values), consisting of some discrete elements
or particles. Then, ``data`` should have method ``__len__``, which returns
number of discrete elements in ``data``. If you plan to build **hierarchical
partitioning** automatically by ``h2tools``, ``data`` object should
additionally have following methods:
  
- ``divide``: divides cluster of discrete elements into subclusters
  of discrete elements,
- ``check_far``: returns :code:`True` if two clusters of discrete
  elements are geometrically far from each other,
- ``compute_aux``: returns auxiliary data for given cluster (i.e. bounding
  box).

Both sources and destinations (domain and range of values) should have
corresponding ``data`` object, except for case of symmetric problem (one
object is enough). To understand minimal requirements on ``data``, see this
example:

.. toctree::

    minimal_data.rst

1b. Providing block function
----------------------------

**Block function** must have following definition:

.. code:: python

    def block_func(row_data, rows, col_data, cols):
        # row_data is a data, corresponding to destinations (range of values)
        # col_data is a data, corresponding to sources (domain)
        # rows and cols are arrays of indexes of rows and columns,
        # corresponding to required submatrix
        return submatrix

``h2tools`` supports vector/matrix/tensor kernels, but to use such kernels
**block function** must return submatrix of a special shape. If `element_shape`
is a shape of kernel element, then entire submatrix must have shape `(n_rows,
element_shape[0], ..., element_shape[-1], n_columns)`. Here is simple example:

.. code:: python

    def block_func(row_data, rows, col_data, cols):
        # Function, that returns radius-vectors from sources (col_data) to
        #   destinations (row_data)
        shape = (rows.size, 3, cols.size)
        submatrix = np.zeros(shape)
        for i in range(rows.size):
            for j in range(cols.size):
                submatrix[i, :, j] = row_data[rows[i]]-col_data[cols[j]]
        return submatrix

2. Building cluster trees
-------------------------

Current implementation of cluster trees generation is built into procedure of
linking cluster trees into block cluster tree. This stage requires only
initialization of row and column cluster trees.

Assume ``row_data`` corresponds to destinations and ``col_data`` corresponds to
sources. Then, following simple example shows cluster tree initialization:

.. code:: python

    from h2tools import ClusterTree
    # row_tree_block_size and col_tree_block_size stand for maximum size
    # of leaf nodes of row and column cluster trees
    row_tree_block_size = 50
    col_tree_block_size = 50
    row_tree = ClusterTree(row_data, row_tree_block_size)
    col_tree = ClusterTree(col_data, col_tree_block_size)

More information on ``ClusterTree`` class is here:

.. toctree::

    cluster_tree.rst


3. Linking block cluster tree
-----------------------------

If cluster trees are already initialized, **hierarchical partitioning** can be
easily computed with following command:

.. code:: python

    from h2tools import Problem
    # func is a block function, returning submatrix of initial matrix
    # symmetric shows if initial matrix is symmetric
    # in symmetric case row_tree and col_tree must be the same Python object
    # (not two different instances)
    # verbose is a flag of additional output
    problem = Problem(func, row_tree, col_tree, symmetric, verbose)

More information on ``Problem`` class is here:

.. toctree::

    problem.rst

    
4. Building approximation
-------------------------

After ``Problem`` object, containing source data, destination data, **block
function** and **block cluster tree**, is ready, MCBH-approximation by
:math:`\mathcal{H}^2`-matrix can be computed by following command:

.. code:: python

    from h2tools.mcbh import mcbh
    # tau is a parameter of relative spectral tolerance of approximation
    # iters is a number of iterations of MCBH-method
    # onfly is a boolean flag, showing if basis submatrices should be computed
    # on fly or saved in memory (onfly=1 dramatically reduces memory
    # consumption by approximant)
    # verbose is a boolean flag, showing if additional information on
    # approximation should go to output
    matrix = mcbh(problem, tau, iters, onfly, verbose)
    # matrix is an instance of H2matrix

More information on multicharge Barnes-Hut (MCBH) method is here:

.. toctree::

    mcbh.rst


5. Working with provided approximant
------------------------------------

MCBH approximation procedure returns instance of ``H2matrix`` class. Full
description of ``H2matrix`` class is here:

.. toctree::

    h2matrix.rst


Gaining Performance boost
-------------------------

As was mentioned previously, approximation requires **block function** and
**block cluster tree**. **Block function** is a numerical bottleneck of entire
approximation problem and can be accelerated by means of **Cython** or
**Numba** (becomes up to 100 times faster). Bottleneck of **block cluster
tree** generation is ``check_far`` method of source and destination data
objects. More information on accelerating tree generation is here:

.. toctree::

    check_far.rst
