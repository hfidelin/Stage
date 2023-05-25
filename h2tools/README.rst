.. |--| unicode:: U+2013   .. en dash
.. |---| unicode:: U+2014  .. em dash, trimming surrounding whitespace
   :trim:

What is ``h2tools``?
--------------------

It is an open-source software, designed to work with :math:`\mathcal{H}^2`
-matrices. Matrices, close to :math:`\mathcal{H}^2`-matrices, often appear
in different physical problems (i.e. described as integral equations or
particle-to-particle interactions). Special structure of such matrices enables
representation with relatively small number of parameters and sparse-like
arithmetics. You can get more information on it in the book [H2matrix-book]_.

``h2tools`` is distributed as a **Python** module, which works with both major
revisions of **Python**  (i.e. 2.7.10 and 3.5 versions are succesfully tested
for compatibility). It has ``h2tools.collections`` submodule with predefined
classes for different types of problems.

``h2tools`` supports vector/matrix/tensor kernels and MPI parallel
factorization and matrix-vector operations.

.. [H2matrix-book] Hackbusch W., Khoromskij B., Sauter S.A. On
    :math:`\mathcal{H}^2`-matrices. |---| Springer, 2000.


Why ``h2tools`` and not any other library?
------------------------------------------

Main feature of ``h2tools`` is an algebraic method of constructing
:math:`\mathcal{H}^2`-approximation using only matrix entries [MCBH]_.
This method does not require preliminary approximation by
:math:`\mathcal{H}`-matrix, is linear in means of problem size and has lower
complexity in terms of memory and operations, than any method of
:math:`\mathcal{H}`-approximation. It also uses less matrix entries, than any
:math:`\mathcal{H}`-approximation method.

.. [MCBH] Mikhalev A.Yu., Oseledets I.V. Iterative representing set selection
    for nested cross approximation // Numer. Linear Algebra Appl. (available
    online `here`__)

__ http://onlinelibrary.wiley.com/doi/10.1002/nla.2021/abstract


Requirements
------------

``h2tools`` extensively uses following **Python** modules:

- `numpy <http://numpy.org/>`_ for matrix operations,
- `maxvolpy <http://bitbucket.org/muxas/maxvolpy/>`_ for selection of
  representing sets,
- `numba <http://numba.pydata.org/>`_ to run examples (which are optimized
  with help of ``numba``),
- `cython <http://cython.org/>`_, required by ``maxvolpy``.

If there is no predefined class in ``h2tools.collections`` for your problem,
consider using ``numba`` and/or ``cython`` to accelerate **Python** code. In
example, such an optimization reduces time of matrix entry computation by a
factor of about 100, which is usually a bottleneck of approximation procedure.


Installation
------------

Easiest way to install ``h2tools`` is to use ``pip`` command:

.. code:: shell

    pip install h2tools

To enable approximation error measurement, install `pypropack
<http://github.com/jakevdp/pypropack>`_. If you want to run examples or use
``h2tools.collections`` submodule, additionally install ``numba`` (i.e. via
``pip install numba``).

If you want to use latest version of ``h2tools``, you can install it from git:

.. code:: shell

    git clone https://bitbucket.org/muxas/h2tools
    cd h2tools
    python setup.py install


Examples
--------

Examples in ``ipython notebook`` format can be found in **examples** folder
of git repository.


Documentation
-------------

Latest documentation is available at http://pythonhosted.org/h2tools
