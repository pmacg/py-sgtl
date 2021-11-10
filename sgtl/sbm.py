"""
Provides methods for generating graphs from the stochastic block model.
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
    else:
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


def _get_number_of_edges(c1_size, c2_size, prob, same_cluster, self_loops, directed):
    """
    Compute the number of edges there will be between two clusters.

    :param c1_size: The size of the first cluster
    :param c2_size: The size of the second cluster
    :param prob: The probability of an edge between the clusters
    :param same_cluster: Whether these are the same cluster
    :param self_loops: Whether we will generate self loops
    :param directed: Whether we are generating a directed graph
    :return: the number of edges to generate between these clusters
    """
    # We need to compute the number of possible edges
    possible_edges_between_clusters = _get_num_pos_edges(c1_size, c2_size, same_cluster, self_loops, directed)

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

    for cluster_1 in range(len(cluster_sizes)):
        # Keep track of the index of the first vertex in the current cluster_2
        c2_base_index = c1_base_index

        # If we are constructing a directed graph, we need to consider all values of cluster_2.
        # Otherwise, we will consider only the clusters with an index >= cluster_1.
        if directed:
            second_clusters = range(len(cluster_sizes))
            c2_base_index = 0
        else:
            second_clusters = range(cluster_1, len(cluster_sizes))

        for cluster_2 in second_clusters:
            # Compute the number of edges between these two clusters
            num_edges = _get_number_of_edges(cluster_sizes[cluster_1],
                                             cluster_sizes[cluster_2],
                                             prob_mat_q[cluster_1][cluster_2],
                                             cluster_1 == cluster_2,
                                             True,
                                             directed)

            # Sample this number of edges. TODO: correct for possible double-sampling of edges
            num_possible_edges = (cluster_sizes[cluster_1] * cluster_sizes[cluster_2]) - 1
            for i in range(num_edges):
                edge_idx = random.randint(0, num_possible_edges)
                u = c1_base_index + int(edge_idx / cluster_sizes[cluster_1])
                v = c2_base_index + (edge_idx % cluster_sizes[cluster_1])
                yield u, v

            # Update the base index for the second cluster
            c2_base_index += cluster_sizes[cluster_2]

        # Update the base index of this cluster
        c1_base_index += cluster_sizes[cluster_1]


def sbm(cluster_sizes, prob_mat_q, directed=False, self_loops=False):
    """
    Generate a graph from the general stochastic block model.

    The list cluster_sizes gives the number of vertices inside each cluster and the matrix Q gives the probability of
    each edge between pairs of clusters.

    For two vertices u and v where u is in cluster i and v is in cluster j, there is an edge between u and v with
    probability Q_{i, j}.

    For the undirected case, we assume that the matrix Q is symmetric (and in practice look only at the upper triangle).
    For the directed case, we generate edges (u, v) and (v, u) with probabilities Q_{i, j} and Q_{j, i} respectively.

    Returns an ``sgtl.Graph`` object.

    :param cluster_sizes: The number of vertices in each cluster.
    :param prob_mat_q: A square matrix where Q_{i, j} is the probability of each edge between clusters i and j. Should
                       be symmetric in the undirected case.
    :param directed: Whether to generate a directed graph (default is false).
    :param self_loops: Whether to generate self-loops (default is false).
    :return: The generated graph as an ``sgtl.Graph`` object.
    """
    # Initialize the adjacency matrix
    adj_mat = scipy.sparse.lil_matrix((sum(cluster_sizes), sum(cluster_sizes)))

    # Generate the edges in the graph
    for (u, v) in _generate_sbm_edges(cluster_sizes, prob_mat_q, directed=directed):
        if u != v or self_loops:
            # Add this edge to the adjacency matrix.
            adj_mat[u, v] = 1

            if not directed:
                adj_mat[v, u] = 1

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
    return sbm([int(n/k)] * k, prob_mat_q, directed=directed)


def ssbm(n, k, p, q, directed=False):
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
