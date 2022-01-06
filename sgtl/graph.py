"""
Provides the Graph object, which is our core representation of a graph within the SGTL library.
"""
import math
from typing import List

import scipy
import scipy.sparse
from sklearn.neighbors import NearestNeighbors
import numpy as np
import networkx as nx
import pandas as pd


class Graph:
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
        # We can assume that there are no non-zero entries in the stored adjacency matrix.
        self.adj_mat = adj_mat.tocsr()
        self.adj_mat.eliminate_zeros()
        self.lil_adj_mat = adj_mat.tolil()

        # For convenience, and to speed up operations on the graph, we precompute the degrees of the vertices in the
        # graph.
        self.degrees = adj_mat.sum(axis=0).tolist()[0]
        self.inv_degrees = list(map(lambda x: 1 / x if x != 0 else 0, self.degrees))
        self.sqrt_degrees = list(map(math.sqrt, self.degrees))
        self.inv_sqrt_degrees = list(map(lambda x: 1 / x if x > 0 else 0, self.sqrt_degrees))

    @staticmethod
    def from_networkx(netx_graph: nx.Graph, edge_weight_attribute='weight'):
        """
        Given a networkx graph object, convert it to an SGTL graph object. Unless otherwise specified, this method
        will use the 'weight' attribute on the networkx edges to assign the weight of the edges. If no such attribute
        is present, the edges will be added with weight 1.

        :param netx_graph: The networkx graph object to be converted.
        :param edge_weight_attribute: (default 'weight') the edge attribute to be used to generate the edge weights.
        :return: An SGTL graph which is equivalent to the given networkx graph.
        """
        return Graph(nx.adjacency_matrix(netx_graph, weight=edge_weight_attribute))

    def to_networkx(self) -> nx.Graph:
        """
        Construct a networkx graph which is equivalent to this SGTL graph.
        """
        return nx.Graph(self.adjacency_matrix())

    def draw(self):
        """
        Plot the graph, by first converting to a networkx graph. This will use the default networkx plotting
        functionality. If you'd like to do something more fancy, then you should convert the graph to a networkx graph
        using the ``to_networkx`` method and use networkx directly.
        """
        nx_graph = self.to_networkx()
        nx.draw(nx_graph)

    def number_of_vertices(self) -> int:
        """The number of vertices in the graph."""
        return self.adjacency_matrix().shape[0]

    def total_volume(self) -> float:
        """The total volume of the graph."""
        return sum(self.degrees)

    def _number_of_self_loops(self) -> int:
        """Get the number of self-loops in the graph."""
        return np.count_nonzero(self.adjacency_matrix().diagonal())

    def _volume_of_self_loops(self) -> float:
        """Get the total weight of all self-loops in the graph."""
        return float(np.sum(self.adjacency_matrix().diagonal()))

    def number_of_edges(self) -> int:
        """The number of edges in the graph, ignoring any weights."""
        return int((self.adjacency_matrix().nnz + self._number_of_self_loops()) / 2)

    def degree_matrix(self):
        """Construct the diagonal degree matrix of the graph."""
        return scipy.sparse.spdiags(
            self.degrees, [0], self.number_of_vertices(), self.number_of_vertices(), format="csr")

    def inverse_degree_matrix(self):
        """Construct the inverse of the diagonal degree matrix of the graph."""
        return scipy.sparse.spdiags(self.inv_degrees, [0], self.number_of_vertices(), self.number_of_vertices(),
                                    format="csr")

    def inverse_sqrt_degree_matrix(self):
        """Construct the square root of the inverse of the diagonal degree matrix of the graph."""
        return scipy.sparse.spdiags(self.inv_sqrt_degrees, [0], self.number_of_vertices(), self.number_of_vertices(),
                                    format="csr")

    def adjacency_matrix(self):
        """
        Return the Adjacency matrix of the graph.
        """
        return self.adj_mat

    def laplacian_matrix(self):
        """
        Construct the Laplacian matrix of the graph. The Laplacian matrix is defined to be

        .. math::
           L = D - A

        where D is the diagonal degree matrix and A is the adjacency matrix of the graph.
        """
        return self.degree_matrix() - self.adjacency_matrix()

    def normalised_laplacian_matrix(self):
        """
        Construct the normalised Laplacian matrix of the graph. The normalised Laplacian matrix is defined to be

        .. math::
            \\mathcal{L} = D^{-1/2} L D^{-1/2} =  I - D^{-1/2} A D^{-1/2}

        where I is the identity matrix and D is the diagonal degree matrix of the graph.
        """
        return self.inverse_sqrt_degree_matrix() @ self.laplacian_matrix() @ self.inverse_sqrt_degree_matrix()

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
        self._check_vert_num(vertex_set)
        return sum([self.degrees[v] for v in vertex_set])

    def weight(self,
               vertex_set_l: List[int],
               vertex_set_r: List[int],
               check_for_overlap=True,
               sets_are_equal=False) -> float:
        """
        Compute the weight of all edges between the two given vertex sets.

        By default, this method needs to check whether the given two sets overlap, this is a somewhat expensive
        operation. There are two circumstances in which the caller can avoid this:

        * if the caller can guarantee that the sets do not overlap, then set ``check_for_overlap=False``
        * if the caller can guarantee that the sets are equal, then set ``sets_are_equal=True``

        :param vertex_set_l: a collection of vertex indices corresponding to the set L
        :param vertex_set_r: a collection of vertex indices corresponding to the set R
        :param check_for_overlap: set to ``False`` if the given sets are guaranteed not to overlap
        :param sets_are_equal: set to ``True`` if the given sets are guaranteed to be equal
        :return: The weight w(L, R)
        """
        self._check_vert_num(vertex_set_l, vertex_set_r)
        raw_weight = self.lil_adj_mat[vertex_set_l][:, vertex_set_r].sum()

        # If the two sets L and R overlap, we will have double counted any edges inside this overlap, save for the
        # self-loops
        if sets_are_equal:
            weight_to_remove = raw_weight / 2
            weight_to_remove -= sum([self.adjacency_matrix()[i, i] for i in vertex_set_l]) / 2
        elif not check_for_overlap:
            weight_to_remove = 0
        else:
            overlap = set.intersection(set(vertex_set_l), set(vertex_set_r))
            weight_to_remove = self.lil_adj_mat[list(overlap)][:, list(overlap)].sum() / 2
            weight_to_remove -= sum([self.adjacency_matrix()[i, i] for i in overlap]) / 2

        # Return the corrected weight
        return raw_weight - weight_to_remove

    def conductance(self, vertex_set_s):
        """
        Compute the conductance of the given set of vertices.
        The conductance is defined to be

        .. math::
           \\phi(S) = 1 - \\frac{2 w(S, S)}{vol(S)}

        :param vertex_set_s: a collection of vertex indices corresponding to the set S
        :return: The conductance :math:`\\phi(S)`
        :raises ValueError: if the vertex set is empty
        """
        self._check_vert_num(vertex_set_s)

        if len(vertex_set_s) == 0:
            raise ValueError("The conductance of the empty set is undefined.")

        return 1 - (2 * self.weight(vertex_set_s, vertex_set_s, sets_are_equal=True)) / self.volume(vertex_set_s)

    def bipartiteness(self, vertex_set_l, vertex_set_r):
        """
        Compute the bipartiteness of the two given vertex sets.
        The bipartiteness is defined as

        .. math::

           \\beta(L, R) = 1 - \\frac{2 w(L, R)}{vol(L \\cup R)}

        :param vertex_set_l: a collection of vertex indices corresponding to the set L
        :param vertex_set_r: a collection of vertex indices corresponding to the set R
        :return: The bipartiteness ratio :math:`\\beta(L, R)`
        :raises ValueError: if both vertex sets are empty
        """
        self._check_vert_num(vertex_set_l, vertex_set_r)

        if len(vertex_set_l) + len(vertex_set_r) == 0:
            raise ValueError("The bipartiteness of the empty set is undefined.")

        return 1 - 2 * self.weight(vertex_set_l, vertex_set_r) / self.volume(vertex_set_l + vertex_set_r)

    def _check_vert_num(self, *args):
        """
        Check that the input vertex set does not include indices greater than the number of vertices in the graph
        """
        for arg in args:
            for vert in arg:
                if vert >= self.number_of_vertices():
                    raise IndexError("Input vertex set includes indices larger than the number of vertices.")


def complete_graph(number_of_vertices: int) -> Graph:
    """
    Construct the complete unweighted graph on :math:`n` vertices.

    :param number_of_vertices: The number of vertices in the graph.
    :return: The complete graph on :math:`n` vertices, as a `Graph` object.
    :raises ValueError: if the number of vertices is not a positive integer.

    :Example:

       >>> import sgtl.graph
       >>> graph = sgtl.graph.complete_graph(4)
       >>> graph.adjacency_matrix().toarray()
       array([[0., 1., 1., 1.],
              [1., 0., 1., 1.],
              [1., 1., 0., 1.],
              [1., 1., 1., 0.]])

    """
    if number_of_vertices <= 0:
        raise ValueError("The graph must contain at least one vertex.")

    # Generate the complete adjacency matrix - we generate a dense matrix first
    adj_mat = scipy.sparse.csr_matrix(np.ones((number_of_vertices, number_of_vertices)) - np.eye(number_of_vertices))
    return Graph(adj_mat)


def cycle_graph(number_of_vertices: int) -> Graph:
    """
    Construct the unweighted cycle graph on :math:`n` vertices.

    :param number_of_vertices: The number of vertices in the graph
    :return: The cycle graph on :math:`n` vertices, as a `Graph` object.
    :raises ValueError: if the number of vertices is not a positive integer.

    :Example:

       >>> import sgtl.graph
       >>> graph = sgtl.graph.cycle_graph(5)
       >>> graph.adjacency_matrix().toarray()
       array([[0., 1., 0., 0., 1.],
              [1., 0., 1., 0., 0.],
              [0., 1., 0., 1., 0.],
              [0., 0., 1., 0., 1.],
              [1., 0., 0., 1., 0.]])

    """
    if number_of_vertices <= 0:
        raise ValueError("The graph must contain at least one vertex.")

    # Generate the cycle graph adjacency matrix
    adj_mat = scipy.sparse.diags([np.ones(number_of_vertices - 1),
                                  np.ones(number_of_vertices - 1),
                                  np.ones(1),
                                  np.ones(1)],
                                 [-1, 1, (number_of_vertices - 1), (1 - number_of_vertices)])
    return Graph(adj_mat)


def star_graph(number_of_vertices: int) -> Graph:
    """
    Construct the unweighted star graph on :math:`n` vertices.

    :param number_of_vertices: The number of vertices in the graph
    :return: The star graph on :math:`n` vertices, as a `Graph` object.
    :raises ValueError: if the number of vertices is not a positive integer.

    :Example:

       >>> import sgtl.graph
       >>> graph = sgtl.graph.star_graph(4)
       >>> graph.adjacency_matrix().toarray()
       array([[0., 1., 1., 1.],
              [1., 0., 0., 0.],
              [1., 0., 0., 0.],
              [1., 0., 0., 0.]])

    """
    if number_of_vertices <= 0:
        raise ValueError("The graph must contain at least one vertex.")

    # Generate the cycle graph adjacency matrix
    adj_mat = scipy.sparse.lil_matrix((number_of_vertices, number_of_vertices))
    for i in range(1, number_of_vertices):
        adj_mat[0, i] = 1
        adj_mat[i, 0] = 1
    return Graph(adj_mat)


def path_graph(number_of_vertices: int) -> Graph:
    """
    Construct the unweighted path graph on :math:`n` vertices.

    :param number_of_vertices: The number of vertices in the graph
    :return: The path graph on :math:`n` vertices, as a `Graph` object.
    :raises ValueError: if the number of vertices is not a positive integer.

    :Example:

       >>> import sgtl.graph
       >>> graph = sgtl.graph.path_graph(5)
       >>> graph.adjacency_matrix().toarray()
       array([[0., 1., 0., 0., 0.],
              [1., 0., 1., 0., 0.],
              [0., 1., 0., 1., 0.],
              [0., 0., 1., 0., 1.],
              [0., 0., 0., 1., 0.]])

    """
    if number_of_vertices <= 0:
        raise ValueError("The graph must contain at least one vertex.")

    # Generate the cycle graph adjacency matrix
    adj_mat = scipy.sparse.diags([np.ones(number_of_vertices - 1), np.ones(number_of_vertices - 1)], [-1, 1])
    return Graph(adj_mat)


def knn_graph(data: np.ndarray, k: int):
    """
    Construct the k-nearest neighbours graph from the given data.

    The ``data`` paramenter must have two dimensions. If ``data`` has dimension (n, d), then the resulting graph will
    have ``n`` vertices. Each vertex will be connected to the ``k`` vertices which are closest to it in the dataset.
    Notice that this does **not** necessarily result in a ``k``-regular graph since neighbours may or may not be
    mutually within the ``k`` nearest.

    The graph will have at most n * k edges.

    The running time of this construction is :math:`O\\left(d \\log(n) + n k\\right)` where
      - d is the dimensionality of each data point
      - n is the number of data points
      - k is the parameter k in the knn graph

    This is likely to be dominated by the :math:`O(n k)` term.

    :param data: the data to construct the graph from
    :param k: how many neighbours to connect to
    :return: An ``sgtl.Graph`` object representing the ``k``-nearest neighbour graph of the input.
    """
    # Create the nearest neighbours for each vertex using sklearn
    _, neighbours = NearestNeighbors(n_neighbors=(k+1)).fit(data).kneighbors(data)

    # Now, let's construct the adjacency matrix of the graph iteratively
    adj_mat = scipy.sparse.lil_matrix((len(data), len(data)))
    for vertex in range(len(data)):
        # Get the k nearest neighbours of this vertex
        for neighbour in neighbours[vertex][1:]:
            adj_mat[vertex, neighbour] = 1
            adj_mat[neighbour, vertex] = 1

    return Graph(adj_mat)


def from_edgelist(filename: str, directed=False, num_vertices=None, **kwargs) -> Graph:
    """
    Construct an ``sgtl.Graph`` object from an edgelist file.

    The edgelist file represents a graph by specifying each edge in the graph on a separate line in the file.
    Each line looks like
        id1 id2 <weight>
    where id1 and id2 are the vertex indices of the given edge, and an optional weight is given as a floating point
    number. If any edge has a specified weight, then you must specify the weight for every edge. That is, every line
    in the edgelist must contain the same number of elements (either 2 or 3). The file may also contain comment lines.

    By default, the elements on a given line are assumed to be separated by spaces, and comments should begin with a
    '#' character. These defaults can be changed by passing additional key-word arguments which will be passed to
    pandas.read_csv.

    :param filename: The edgelist filename.
    :param directed: Whether the graph is directerd.
    :param num_vertices: The number of vertices in the graph, if this is known in advance. Specifying this value will
                         speed up the method.
    :param kwargs: Key-word arguments to pass to the `pandas.read_csv` method which will be used to read in the file.
                   This can be used to set a delimiter other than the space character, or to change the comment
                   character.
    :return: the constructed `sgtl.Graph` object
    :raises TypeError: If the edgelist file cannot be parsed due to incorrect types.

    :Example:

    An example edgelist file is as follows::

        # This is a comment line
        0 1 0.5
        1 2 1
        2 0 0.5

    The above file gives a weighted triangle graph.
    """
    # Set the default values of the parameters
    if 'sep' not in kwargs:
        kwargs['sep'] = '\s+'
    if 'comment' not in kwargs:
        kwargs['comment'] = '#'
    if 'header' not in kwargs:
        kwargs['header'] = None

    # Read in the edgelist file, and create the adjacency matrix of the graph as we go
    if num_vertices is None:
        num_vertices = 1
    adj_mat = scipy.sparse.lil_matrix((num_vertices, num_vertices))
    maximum_node_index = num_vertices - 1
    edgelist_data = pd.read_csv(filename, **kwargs)
    for edge_row in edgelist_data.iterrows():
        # Get the vertex indices from the edgelist file
        v1 = int(edge_row[1][0])
        v2 = int(edge_row[1][1])

        # Check whether the weight of the edge is specified
        if len(edge_row[1]) > 2:
            weight = edge_row[1][2]
        else:
            weight = 1.0

        if not isinstance(weight, np.float):
            raise TypeError("Edge weights must be given as floating point numbers.")

        # Update the size of the adjacency matrix if we have encountered a larger vertex index than previously.
        if v1 > maximum_node_index or v2 > maximum_node_index:
            old_maximum_node_index = maximum_node_index
            maximum_node_index = max(v1, v2)

            # Update the shape of the lil matrix
            adj_mat._shape = (maximum_node_index + 1, maximum_node_index + 1)

            # Add rows to the data elements of the lil matrix
            for i in range(old_maximum_node_index + 1, maximum_node_index + 1):
                adj_mat.rows = np.append(adj_mat.rows, 1)
                adj_mat.rows[i] = []
                adj_mat.data = np.append(adj_mat.data, 1)
                adj_mat.data[i] = []

        # Update the adjacency matrix with this edge.
        adj_mat[v1, v2] = weight
        if not directed:
            adj_mat[v2, v1] = weight

    # Construct the graph object and return it
    return Graph(adj_mat)


def to_edgelist(graph: Graph, filename: str):
    """
    Save the given graph object as an edgelist file.

    :param graph: the graph to be saved.
    :param filename: the edgelist filename.
    """
    with open(filename, 'w') as fout:
        # Iterate through every edge in the graph, and add the edge to the edgelist.
        adj_mat = graph.adjacency_matrix().tocoo()
        for v1, v2, weight in zip(adj_mat.row, adj_mat.col, adj_mat.data):
            fout.write(f"{v1} {v2} {weight}\n")
