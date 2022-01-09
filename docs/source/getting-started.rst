Getting Started
===============

Installation
------------

To use SGTL, first install it using pip.

.. code-block:: console

   $ pip install sgtl

Quick Start
-----------

Creating Graphs
~~~~~~~~~~~~~~~~

Graphs in the SGTL are represented using their sparse adjacency matrix.
To create a new graph, you can use the following pattern.

   >>> import scipy
   >>> import sgtl
   >>> adjacency_matrix = scipy.sparse.csr_matrix([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
   >>> graph = sgtl.Graph(adjacency_matrix)
   >>> graph.number_of_vertices()
   4
   >>> graph.number_of_edges()
   3

For more information about the ``Graph`` class, see :any:`sgtl.graph.Graph`.

You can also generate some graphs with standard shapes.
For example:

    >>> import sgtl.graph
    >>> graph = sgtl.graph.cycle_graph(5)
    >>> graph.adjacency_matrix().toarray()
    array([[0., 1., 0., 0., 1.],
           [1., 0., 1., 0., 0.],
           [0., 1., 0., 1., 0.],
           [0., 0., 1., 0., 1.],
           [1., 0., 0., 1., 0.]])

Or, you can load a more complex graph from an edgelist file like this.

    >>> import sgtl.graph
    >>> graph = sgtl.graph.from_edgelist("my_graph.edgelist")

See the documentation of the :any:`sgtl.graph.from_edgelist` method for more information on the
required format of the edgelist file.

Viewing the spectrum of a graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Investigating the spectrum of a graph is very simple. For example, you could visualise
the spectrum of the tensor product of two graphs like this.

.. code-block:: python

    >>> import sgtl.graph
    >>> import sgtl.spectrum
    >>> g1 = sgtl.graph.cycle_graph(5)
    >>> g2 = sgtl.graph.complete_graph(5)
    >>> g3 = g1.tensor_product(g2)
    >>> spectrum = sgtl.spectrum.normalised_adjacency_spectrum(g3, plot=True)

The stochastic block model
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To generate a random graph from the stochastic block model with 1000 vertices and 5 clusters,
you can use the following pattern.

   >>> import sgtl.random
   >>> p = 0.5
   >>> q = 0.1
   >>> graph = sgtl.random.ssbm(1000, 5, p, q)

For more information about the methods for generating graphs from the stochastic block model, see
:any:`sgtl.random`.

Spectral clustering
~~~~~~~~~~~~~~~~~~~

Finding clusters in a graph using spectral clustering is as easy as this:

   >>> import sgtl.random
   >>> import sgtl.clustering
   >>> graph = sgtl.random.ssbm(10, 2, 1, 0.1)
   >>> clusters = sgtl.clustering.spectral_clustering(graph, 2)
   >>> sorted(clusters)
   [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

For more information, see :any:`sgtl.clustering`.