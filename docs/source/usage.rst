Usage
=====

Installation
------------

To use SGTL, first install it using pip:

.. code-block:: console

   $ pip install sgtl

Quick Start
-----------
Graphs in the SGTL are represented using their sparse adjacency matrix.
To create a new graph, you can use the following pattern.

   >>> import scipy
   >>> import sgtl
   >>> adjacency_matrix = scipy.sparse.csr_matrix([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
   >>> graph = sgtl.Graph(adjacency_matrix)
   >>> graph.num_vertices
   4
   >>> graph.num_edges
   3
