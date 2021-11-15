"""
Provides methods for generating random graphs.

Includes methods for generating Erdos-Renyi graphs and graphs from the stochastic block model.
"""
import random
import scipy.sparse
import numpy as np
from . import graph


def _get_num_pos_edges(c1_size, c2_size, same_cluster, self_loops, directed):
    """
    Compute the number of possible edges between two clusters.

    :param c1_size: The size of the first cluster
    :param c2_size: The size of the second cluster
    :param same_cluster: Whether these are the same cluster
    :param self_loops: Whether we will generate self loops
    :param directed: Whether we are generating a directed graph
    :return: the number of possible edges between these clusters
    """
    if not same_cluster:
        # The number is simply the product of the number of vertices
        return c1_size * c2_size

    # The base number is n choose 2
    possible_edges_between_clusters = int((c1_size * (c1_size - 1)) / 2)

    # If we are allowed self-loops, then add them on
    if self_loops:
        possible_edges_between_clusters += c1_size

    # The number is normally the same for undirected and directed graphs, unless the clusters are the same, in which
    # case the number for the directed graph is double since we need to consider both directions of each edge.
    if directed:
        possible_edges_between_clusters *= 2

    # But if we are allowed self-loops, then we shouldn't double them since there is only one 'direction'.
    if directed and self_loops:
        possible_edges_between_clusters -= c1_size

    return possible_edges_between_clusters


def _get_number_of_edges(c1_size, c2_size, prob, same_cluster, directed):
    """
    Compute the number of edges there will be between two clusters. If the two clusters are the same, then this method
    will assume we are generating self-loops.

    :param c1_size: The size of the first cluster
    :param c2_size: The size of the second cluster
    :param prob: The probability of an edge between the clusters
    :param same_cluster: Whether these are the same cluster
    :param directed: Whether we are generating a directed graph
    :return: the number of edges to generate between these clusters
    """
    # We need to compute the number of possible edges
    possible_edges_between_clusters = _get_num_pos_edges(c1_size, c2_size, same_cluster, True, directed)

    # Sample the number of edges from the binomial distribution
    return np.random.binomial(possible_edges_between_clusters, prob)


def _generate_sbm_edges(cluster_sizes, prob_mat_q, directed=False):
    """
    Given a list of cluster sizes, and a square matrix Q, generates edges for a graph in the following way.

    For two vertices u and v where u is in cluster i and v is in cluster j, there is an edge between u and v with
    probability Q_{i, j}.

    For the undirected case, we assume that the matrix Q is symmetric (and in practice look only at the upper triangle).
    For the directed case, we generate edges (u, v) and (v, u) with probabilities Q_{i, j} and Q_{j, i} respectively.

    May return self-loops. The calling code can decide what to do with them.

    Returns edges as pairs (u, v) where u and v are integers giving the index of the respective vertices.

    :param cluster_sizes: a list giving the number of vertices in each cluster
    :param prob_mat_q: A square matrix where Q_{i, j} is the probability of each edge between clusters i and j. Should
                       be symmetric in the undirected case.
    :param directed: Whether to generate a directed graph (default is false).
    :return: Edges (u, v).
    """
    # We will iterate over the clusters. This variable keeps track of the index of the first vertex in the current
    # cluster_1.
    c1_base_index = 0

    for cluster_1_idx, cluster_1_size in enumerate(cluster_sizes):
        # Keep track of the index of the first vertex in the current cluster_2
        c2_base_index = c1_base_index

        # If we are constructing a directed graph, we need to consider all values of cluster_2.
        # Otherwise, we will consider only the clusters with an index >= cluster_1.
        if directed:
            second_clusters = range(len(cluster_sizes))
            c2_base_index = 0
        else:
            second_clusters = range(cluster_1_idx, len(cluster_sizes))

        for cluster_2_idx in second_clusters:
            cluster_2_size = cluster_sizes[cluster_2_idx]
            if cluster_2_idx == cluster_1_idx:
                # If the clusters are the same, we will iterate through each vertex
                for vertex_2 in range(c1_base_index, c1_base_index + cluster_1_size):
                    # Get the number of edges leaving this vertex
                    num_edges = np.random.binomial(cluster_1_size - (vertex_2 - c1_base_index),
                                                   prob_mat_q[cluster_1_idx][cluster_2_idx])
                    # Sample this number of edges and yield them
                    for vertex_1 in random.sample(range(vertex_2, c1_base_index + cluster_1_size), num_edges):
                        yield vertex_2, vertex_1
            else:
                # Compute the number of edges between these two clusters
                num_edges = _get_number_of_edges(cluster_1_size,
                                                 cluster_2_size,
                                                 prob_mat_q[cluster_1_idx][cluster_2_idx],
                                                 cluster_1_idx == cluster_2_idx,
                                                 directed)

                # Sample this number of edges.
                num_possible_edges = (cluster_1_size * cluster_2_size) - 1
                for edge_idx in random.sample(range(num_possible_edges), num_edges):
                    vertex_1 = c1_base_index + int(edge_idx / cluster_1_size)
                    vertex_2 = c2_base_index + (edge_idx % cluster_1_size)
                    yield vertex_1, vertex_2

            # Update the base index for the second cluster
            c2_base_index += cluster_2_size

        # Update the base index of this cluster
        c1_base_index += cluster_1_size


def sbm(cluster_sizes, prob_mat_q, directed=False, self_loops=False):
    """
    Generate a graph from the general stochastic block model.

    The list cluster_sizes gives the number of vertices inside each cluster and the matrix Q gives the probability of
    each edge between pairs of clusters.

    For two vertices :math:`u` and :math:`v` where :math:`u` is in cluster :math:`i` and :math:`v` is in cluster
    :math:`j`, there is an edge between :math:`u` and :math:`v` with probability :math:`Q_{i, j}`.

    For the undirected case, we assume that the matrix :math:`Q` is symmetric (and in practice look only at the upper
    triangle). For the directed case, we generate edges :math:`(u, v)` and :math:`(v, u)` with probabilities
    :math:`Q_{i, j}` and :math:`Q_{j, i}` respectively.

    Returns an :doc:`sgtl.graph.Graph` object.

    :param cluster_sizes: The number of vertices in each cluster.
    :param prob_mat_q: A square matrix where :math:`Q_{i, j}` is the probability of each edge between clusters
                       :math:`i` and :math:`j`. Should be symmetric in the undirected case.
    :param directed: Whether to generate a directed graph (default is false).
    :param self_loops: Whether to generate self-loops (default is false).
    :return: The generated graph as an :doc:`sgtl.graph.Graph` object.

    :Example:

    To generate a graph with 4 clusters of different sizes, and a custom probability matrix :math:`Q`, you can use
    the following:

    .. code-block:: python

       import sgtl.random
       cluster_sizes = [20, 50, 100, 200]
       Q = [[0.6, 0.1, 0.1, 0.3], [0.1, 0.5, 0.2, 0.1], [0.1, 0.2, 0.7, 0.2], [0.3, 0.1, 0.2, 0.5]]
       graph = sgtl.random.sbm(cluster_sizes, Q)

    For convenience, in the common case when every cluster has the same size or the internal and external probabilities
    are all the same, you can instead use :doc:`sgtl.random.sbm_equal_clusters` or :doc:`sgtl.random.ssbm`.
    """
    # Initialize the adjacency matrix
    adj_mat = scipy.sparse.lil_matrix((sum(cluster_sizes), sum(cluster_sizes)))

    # Generate the edges in the graph
    for (vertex_1, vertex_2) in _generate_sbm_edges(cluster_sizes, prob_mat_q, directed=directed):
        if vertex_1 != vertex_2 or self_loops:
            # Add this edge to the adjacency matrix.
            adj_mat[vertex_1, vertex_2] = 1

            if not directed:
                adj_mat[vertex_2, vertex_1] = 1

    # Construct the graph and return it.
    return graph.Graph(adj_mat)


def sbm_equal_clusters(n, k, prob_mat_q, directed=False):
    """
    Generate a graph with equal cluster sizes from the stochastic block model.

    Generates a graph with n vertices and k clusters. Every cluster will have floor(n/k) vertices. The probability of
    each edge inside a cluster is given by the probability matrix Q as described in the ``sbm`` method.

    :param n: The number of vertices in the graph.
    :param k: The number of clusters.
    :param prob_mat_q: q[i][j] gives the probability of an edge between clusters i and j
    :param directed: Whether to generate a directed graph.
    :return: The generated graph as an ``sgtl.Graph`` object.
    """
    # We are ok with using the 'n', and 'k' variable names for the stochastic block model - these are standard notation
    # for this model.
    # pylint: disable=invalid-name
    return sbm([int(n/k)] * k, prob_mat_q, directed=directed)


def ssbm(n: int, k: int, p: float, q: float, directed=False):
    """
    Generate a graph from the symmetric stochastic block model.

    Generates a graph with n vertices and k clusters. Every cluster will have floor(n/k) vertices. The probability of
    each edge inside a cluster is given by p. The probability of an edge between two different clusters is q.

    :param n: The number of vertices in the graph.
    :param k: The number of clusters.
    :param p: The probability of an edge inside a cluster.
    :param q: The probability of an edge between clusters.
    :param directed: Whether to generate a directed graph.
    :return: The generated graph as an ``sgtl.Graph`` object.
    """
    # We are ok with using the 'n', 'k', 'p', and 'q' variable names for the stochastic block model - these are
    # standard notation for this model.
    # pylint: disable=invalid-name

    # Make sure that the value q is an integer or float
    try:
        p = float(p)
        q = float(q)
    except Exception as error:
        raise TypeError("The probabilities p and q must be numbers between 0 and 1.") from error

    # Every cluster has the same size.
    cluster_sizes = [int(n/k)] * k

    # Construct the k*k probability matrix Q. The off-diagonal entries are all q and the diagonal entries are all p.
    prob_mat_q = []
    for row_num in range(k):
        new_row = [q] * k
        new_row[row_num] = p
        prob_mat_q.append(new_row)

    # Call the general sbm method.
    return sbm(cluster_sizes, prob_mat_q, directed=directed)


def erdos_renyi(n, p):
    """
    Generate a random graph from the Erdos-Renyi model.

    :math:`G(n, p)` is a random graph on :math:`n` vertices such that for any pair of vertices :math:`u` and :math:`v`
    the edge :math:`(u, v)` is included with probability :math:`p`.

    :param n: The number of vertices in the graph.
    :param p: The probability of an edge between each pair of vertices.
    :return: The generated graph as an ``sgtl.Graph`` object.
    """
    # We are ok with using the 'n', and 'q' variable names for the stochastic block model - these are standard notation
    # for this model.
    # pylint: disable=invalid-name
    return ssbm(n, 1, p, p)
