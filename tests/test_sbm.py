"""
Tests for the sbm module.
"""
import scipy.sparse
from context import sgtl
import sgtl.sbm


def test_num_pos_edges():
    n = 250
    assert sgtl.sbm._get_num_pos_edges(n, n, True, True, False) == int((n * (n - 1)) / 2) + n


def test_sbm():
    # Generate a probability matrix for a directed graph
    prob_mat_q = [[0.4, 0.1, 0.01, 0], [0.2, 0.4, 0.01, 0], [0.01, 0.3, 0.6, 0.2], [0, 0.2, 0.1, 0.6]]

    # Generate a graph with this probability matrix, and the following cluster sizes
    cluster_sizes = [100, 50, 20, 100]
    graph = sgtl.sbm.sbm(cluster_sizes, prob_mat_q, directed=True)
    assert graph.num_vertices == sum(cluster_sizes)

    # Generate a graph with a fixed number of vertices
    n = 1000
    graph = sgtl.sbm.sbm_equal_clusters(n, 4, prob_mat_q, directed=True)
    assert graph.num_vertices == n

    # Generate a graph with a fixed p and q
    n = 2000
    p = 0.8
    q = 0.2
    graph = sgtl.sbm.ssbm(n, 4, p, q)

    # The adjacency matrix should be symmetric
    sym_diff = (graph.adj_mat - graph.adj_mat.transpose())
    sym_diff.eliminate_zeros()
    assert sym_diff.nnz == 0

    # The number of edges between two clusters should be about
    # (n/4) * (n/4) * q
    assert abs((graph.weight(list(range(500)), list(range(500, 1000))) / ((n/4) * (n/4) * q)) - 1) <= 0.1

    # And in a single cluster should be about
    # (n/4) * (n/4) * p
    assert abs((graph.weight(list(range(500)), list(range(500))) / (int(((n/4) * ((n/4) - 1)) / 2) + (n/4))) - 1) <= 0.1
