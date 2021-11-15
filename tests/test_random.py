"""
Tests for the random module.
"""
import pytest
from context import sgtl
import sgtl.random


def test_num_pos_edges():
    n = 250
    assert sgtl.random._get_num_pos_edges(n, n, True, True, False) == int((n * (n - 1)) / 2) + n


def test_get_num_edges():
    n = 250
    p = 0.8
    num_edges = sgtl.random._get_number_of_edges(n, n, p, True, True, False)
    expected_edges = p * ((n * (n - 1) / 2) + n)
    assert abs((num_edges / expected_edges) - 1) <= 0.1


def test_sbm():
    # Generate a probability matrix for a directed graph
    prob_mat_q = [[0.4, 0.1, 0.01, 0], [0.2, 0.4, 0.01, 0], [0.01, 0.3, 0.6, 0.2], [0, 0.2, 0.1, 0.6]]

    # Generate a graph with this probability matrix, and the following cluster sizes
    cluster_sizes = [100, 50, 20, 100]
    graph = sgtl.random.sbm(cluster_sizes, prob_mat_q, directed=True)
    assert graph.number_of_vertices() == sum(cluster_sizes)

    # Generate a graph with a fixed number of vertices
    n = 1000
    graph = sgtl.random.sbm_equal_clusters(n, 4, prob_mat_q, directed=True)
    assert graph.number_of_vertices() == n

    # Generate a graph with a fixed p and q
    n = 2000
    p = 0.8
    q = 0.2
    graph = sgtl.random.ssbm(n, 4, p, q)

    # The adjacency matrix should be symmetric
    sym_diff = (graph.adj_mat - graph.adj_mat.transpose())
    sym_diff.eliminate_zeros()
    assert sym_diff.nnz == 0

    # The number of edges between two clusters should be about
    # (n/4) * (n/4) * q
    assert abs((graph.weight(list(range(500)), list(range(500, 1000))) / ((n/4) * (n/4) * q)) - 1) <= 0.1

    # And in a single cluster should be about
    # (n/4) * (n/4) * p
    assert abs((graph.weight(list(range(500)), list(range(500))) /
                (p * sgtl.random._get_num_pos_edges(500, 500, True, False, False))) - 1) <= 0.1


def test_ssbm():
    # Make sure that ssbm does not allow you to pass an array of probabilities.
    with pytest.raises(TypeError):
        q = [[0.4, 0.1, 0.1, 0.1], [0.1, 0.4, 0.1, 0.1], [0.1, 0.1, 0.4, 0.1], [0.1, 0.1, 0.1, 0.4]]
        _ = sgtl.random.ssbm(1000, 4, 0.1, q)

    # We should be able to pass the value 0 to the ssbm function
    _ = sgtl.random.ssbm(1000, 4, 0.1, 0)


def test_erdos_renyi():
    # Generate a graph
    n = 1000
    graph = sgtl.random.erdos_renyi(n, 0.1)

    # Check that the graph has the expected number of vertices and edges.
    assert graph.number_of_vertices() == n
    assert abs((graph.volume(range(1000)) / (int(2 * 0.1 * (n * (n - 1)) / 2) + n)) - 1) <= 0.1
