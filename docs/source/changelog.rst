Changelog
=========

Unreleased
----------

**Fixed**

* Update sklearn dependency to scikit-learn

0.4.6 - 2022-01-11
------------------

**Fixed**

* `Issue #46 <https://github.com/pmacg/py-sgtl/issues/46>`_ - speed up weight calculation

0.4.5 - 2022-01-09
------------------

**Added**

* `Issue #32 <https://github.com/pmacg/py-sgtl/issues/32>`_ - add tensor product method for combining graphs
* `Issue #27 <https://github.com/pmacg/py-sgtl/issues/27>`_ - add option to plot spectrum of graph
* `Issue #31 <https://github.com/pmacg/py-sgtl/issues/31>`_ - add graphs together

0.4.4 - 2022-01-07
------------------
For brevity, the log below includes changes made in versions ``0.4.1``, ``0.4.2``, ``0.4.3``, and ``0.4.4``.

**Added**

* `Issue #25 <https://github.com/pmacg/py-sgtl/issues/25>`_ - add methods for reading and writing edgelist files
* Added ``Graph.adjacency_matrix()`` method.
* `Issue #29 <https://github.com/pmacg/py-sgtl/issues/29>`_ - add spectrum methods for adjacency and laplacian matrices.
* `Issue #35 <https://github.com/pmacg/py-sgtl/issues/35>`_ - add method to construct k nearest neighbour graph.
* `Issue #41 <https://github.com/pmacg/py-sgtl/issues/41>`_ - construct graph using gaussian kernel function with threshold.

**Fixed**

* `Issue #39 <https://github.com/pmacg/py-sgtl/issues/39>`_ - KNN graph construction should work with sparse data matrices.

0.4 - 2021-12-14
----------------

**Changed**

* `Issue #3 <https://github.com/pmacg/py-sgtl/issues/3>`_ - move members ``graph.num_edges`` and ``graph.num_vertices`` to new methods ``graph.number_of_edges`` and ``graph.number_of_vertices``.

**Added**

* `Issue #3 <https://github.com/pmacg/py-sgtl/issues/3>`_ - add ``total_volume`` method to ``sgtl.graph.Graph``
* `Issue #2 <https://github.com/pmacg/py-sgtl/issues/2>`_ - add ``_check_vert_num`` method to ``sgtl.graph.Graph``
* `Issue #9 <https://github.com/pmacg/py-sgtl/issues/9>`_ - add methods for converting to and from ``networkx`` graph objects.
* `Issue #6 <https://github.com/pmacg/py-sgtl/issues/6>`_ - add cheeger cut algorithm

**Fixed**

* `Issue #10 <https://github.com/pmacg/py-sgtl/issues/10>`_ - correct ``graph.weight`` calculation when edges have floating point weights or self-loops
* `Issue #3 <https://github.com/pmacg/py-sgtl/issues/3>`_ - correct number of edges for weighted graphs
* `Issue #8 <https://github.com/pmacg/py-sgtl/issues/8>`_ - return a meaningful error when computing the conductance or bipartiteness of the empty set

0.3.3 - 2021-11-12
------------------

**Fixed**

* `Issue #4 <https://github.com/pmacg/py-sgtl/issues/4>`_ - definition of normalised laplacian was incorrect

0.3.2 - 2021-11-11
------------------

**Fixed**

* The ``ssbm`` method will not allow non-float probabilities
* The ``Graph.weight`` method will give the correct value when sets are equal
* Prevent ``sbm`` method from generating duplicate edges

0.3 - 2021-11-11
----------------

**Changed**

* Changed the name of the ``sbm`` module to ``random``.

**Added**

* Generate standard graphs: complete, cycle, path, star.
* Generate Erdos-Renyi graphs

**Fixed**

* Spectral clustering checks that number of clusters and eigenvectors are positive integers.

0.2 - 2021-11-10
----------------

**Added**

* Changelog page to documentation
* ``sbm`` module for generating graphs from the stochastic block model

0.1.1 - 2021-11-10
------------------

**Added**

* ``Graph`` class for representing graphs
* ``clustering.spectral_clustering`` clustering method
