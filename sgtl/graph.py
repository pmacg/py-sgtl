"""
Provides the Graph object, which is our core representation of a graph within the SGTL library.
"""
import scipy as sp
import scipy.sparse
import numpy as np
import math


class Graph(object):
    """
    Represents a graph. We keep things very simple - a graph is represented by its sparse adjacency matrix.

    In the general case, this allows for
      - directed and undirected graphs
      - self-loops

    If you'd like to store meta-data about the graph, such as node or edge labels, you should implement a subclass of
    this one, and add that information yourself.

    This graph cannot be dynamically updated. It must be initialised with the complete adjacency matrix.

    Vertices are referred to by their index in the adjacency matrix.
    """

    def __init__(self, adj_mat):
        """
        Initialise the graph with an adjacency matrix.

        :param adj_mat: A sparse scipy matrix.
        """
        # The graph is represented by the sparse adjacency matrix. We store the adjacency matrix in two sparse formats.
        self.adj_mat = adj_mat.tocsr()
        self.lil_adj_mat = adj_mat.tolil()

        # For convenience, and to speed up operations on the graph, we precompute various pieces of information about
        # the graph.

        # Store the degrees of the vertices in the graph.
        self.degrees = adj_mat.sum(axis=0).tolist()[0]
        self.inv_degrees = list(map(lambda x: 1 / x if x != 0 else 0, self.degrees))
        self.sqrt_degrees = list(map(math.sqrt, self.degrees))
        self.inv_sqrt_degrees = list(map(lambda x: 1 / x if x > 0 else 0, self.sqrt_degrees))

        # Store the number of edges and vertices in the graph.
        self.num_vertices = self.adj_mat.shape[0]
        self.num_edges = round(sum(self.degrees) / 2)

    def degree_matrix(self):
        """Construct the diagonal degree matrix of the graph."""
        return sp.sparse.spdiags(self.degrees, [0], self.num_vertices, self.num_vertices, format="csr")

    def inverse_degree_matrix(self):
        """Construct the inverse of the diagonal degree matrix of the graph."""
        return sp.sparse.spdiags(self.inv_degrees, [0], self.num_vertices, self.num_vertices, format="csr")

    def inverse_sqrt_degree_matrix(self):
        """Construct the square root of the inverse of the diagonal degree matrix of the graph."""
        return sp.sparse.spdiags(self.inv_sqrt_degrees, [0], self.num_vertices, self.num_vertices, format="csr")

    def laplacian_matrix(self):
        """
        Construct the Laplacian matrix of the graph. The Laplacian matrix is defined to be

        .. math::
           L = D - A

        where D is the diagonal degree matrix and A is the adjacency matrix of the graph.
        """
        return self.degree_matrix() - self.adj_mat

    def normalised_laplacian_matrix(self):
        """
        Construct the normalised Laplacian matrix of the graph. The normalised Laplacian matrix is defined to be

        .. math::
            \\mathcal{L} = D^{-1/2} L D^{-1/2} =  I - D^{-1/2} A D^{-1/2}

        where I is the identity matrix and D is the diagonal degree matrix of the graph.
        """
        return self.inverse_sqrt_degree_matrix() @ self.laplacian_matrix() @ self.inverse_degree_matrix()

    def random_walk_laplacian_matrix(self):
        """
        Construct the random walk Laplacian matrix of the graph. The random walk Laplacian matrix is defined to be

        .. math::
            L_{\\mathrm{RW}} = D^{-1} L =  I - D^{-1} A

        where I is the identity matrix and D is the diagonal degree matrix of the graph.
        """
        return self.inverse_degree_matrix() @ self.laplacian_matrix()

    def volume(self, vertex_set):
        """
        Given a set of vertices, compute the volume of the set.

        :param vertex_set: an iterable collection of vertex indices
        :return: The volume of vertex_set
        """
        return sum([self.degrees[v] for v in vertex_set])

    def weight(self, vertex_set_l, vertex_set_r):
        """
        Compute the weight of all edges between the two given vertex sets.

        :param vertex_set_l: a collection of vertex indices corresponding to the set L
        :param vertex_set_r: a collection of vertex indices corresponding to the set R
        :return: The weight w(L, R)
        """
        return self.lil_adj_mat[vertex_set_l][:, vertex_set_r].sum()

    def conductance(self, vertex_set_s):
        """
        Compute the conductance of the given set of vertices.
        The conductance is defined to be

        .. math::
           \\phi(S) = 1 - \\frac{2 w(S, S)}{vol(S)}

        :param vertex_set_s: a collection of vertex indices corresponding to the set S
        :return: The conductance :math:`\\phi(S)`
        """
        return 1 - (2 * self.weight(vertex_set_s, vertex_set_s)) / self.volume(vertex_set_s)

    def bipartiteness(self, vertex_set_l, vertex_set_r):
        """
        Compute the bipartiteness of the two given vertex sets.
        The bipartiteness is defined as

        .. math::

           \\beta(L, R) = 1 - \\frac{2 w(L, R)}{vol(L \\cup R)}

        :param vertex_set_l: a collection of vertex indices corresponding to the set L
        :param vertex_set_r: a collection of vertex indices corresponding to the set R
        :return: The bipartiteness ratio \beta(L, R)
        """
        return 1 - 2 * self.weight(vertex_set_l, vertex_set_r) / self.volume(vertex_set_l + vertex_set_r)


def complete_graph(n: int) -> Graph:
    """
    Construct the complete unweighted graph on :math:`n` vertices.

    :param n: The number of vertices in the graph.
    :return: The complete graph on :math:`n` vertices, as a `Graph` object.
    :raises ValueError: if the number of vertices is not a positive integer.

    :Example:

       >>> import sgtl.graph
       >>> graph = sgtl.graph.complete_graph(4)
       >>> graph.adj_mat.toarray()
       array([[0., 1., 1., 1.],
              [1., 0., 1., 1.],
              [1., 1., 0., 1.],
              [1., 1., 1., 0.]])

    """
    if n <= 0:
        raise ValueError("The graph must contain at least one vertex.")

    # Generate the complete adjacency matrix - we generate a dense matrix first
    adj_mat = sp.sparse.csr_matrix(np.ones((n, n)) - np.eye(n))
    return Graph(adj_mat)


def cycle_graph(n: int) -> Graph:
    """
    Construct the unweighted cycle graph on :math:`n` vertices.

    :param n: The number of vertices in the graph
    :return: The cycle graph on :math:`n` vertices, as a `Graph` object.
    :raises ValueError: if the number of vertices is not a positive integer.

    :Example:

       >>> import sgtl.graph
       >>> graph = sgtl.graph.cycle_graph(5)
       >>> graph.adj_mat.toarray()
       array([[0., 1., 0., 0., 1.],
              [1., 0., 1., 0., 0.],
              [0., 1., 0., 1., 0.],
              [0., 0., 1., 0., 1.],
              [1., 0., 0., 1., 0.]])

    """
    if n <= 0:
        raise ValueError("The graph must contain at least one vertex.")

    # Generate the cycle graph adjacency matrix
    adj_mat = sp.sparse.diags([np.ones(n-1), np.ones(n-1), np.ones(1), np.ones(1)], [-1, 1, (n-1), (1-n)])
    return Graph(adj_mat)


def star_graph(n: int) -> Graph:
    """
    Construct the unweighted star graph on :math:`n` vertices.

    :param n: The number of vertices in the graph
    :return: The star graph on :math:`n` vertices, as a `Graph` object.
    :raises ValueError: if the number of vertices is not a positive integer.

    :Example:

       >>> import sgtl.graph
       >>> graph = sgtl.graph.star_graph(4)
       >>> graph.adj_mat.toarray()
       array([[0., 1., 1., 1.],
              [1., 0., 0., 0.],
              [1., 0., 0., 0.],
              [1., 0., 0., 0.]])

    """
    if n <= 0:
        raise ValueError("The graph must contain at least one vertex.")

    # Generate the cycle graph adjacency matrix
    adj_mat = sp.sparse.lil_matrix((n, n))
    for i in range(1, n):
        adj_mat[0, i] = 1
        adj_mat[i, 0] = 1
    return Graph(adj_mat)


def path_graph(n: int) -> Graph:
    """
    Construct the unweighted path graph on :math:`n` vertices.

    :param n: The number of vertices in the graph
    :return: The path graph on :math:`n` vertices, as a `Graph` object.
    :raises ValueError: if the number of vertices is not a positive integer.

    :Example:

       >>> import sgtl.graph
       >>> graph = sgtl.graph.path_graph(5)
       >>> graph.adj_mat.toarray()
       array([[0., 1., 0., 0., 0.],
              [1., 0., 1., 0., 0.],
              [0., 1., 0., 1., 0.],
              [0., 0., 1., 0., 1.],
              [0., 0., 0., 1., 0.]])

    """
    if n <= 0:
        raise ValueError("The graph must contain at least one vertex.")

    # Generate the cycle graph adjacency matrix
    adj_mat = sp.sparse.diags([np.ones(n-1), np.ones(n-1)], [-1, 1])
    return Graph(adj_mat)
