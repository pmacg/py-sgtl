"""
Tests for the graph object.
"""
import numpy as np
import scipy as sp
import scipy.sparse
import math
import pytest
from context import sgtl
import sgtl.random

# Define the adjacency matrices of some useful graphs.
C4_ADJ_MAT = scipy.sparse.csr_matrix([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
K6_ADJ_MAT = scipy.sparse.csr_matrix([[0, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1],
                                      [1, 1, 1, 0, 1, 1], [1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 0]])
BARBELL5_ADJ_MAT = scipy.sparse.csr_matrix([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                            [1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                                            [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                                            [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                                            [1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
                                            [0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
                                            [0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
                                            [0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
                                            [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                                            ])

# Define a small constant for making sure floats are close
EPSILON = 0.00001


def test_graph_constructor():
    # Start by constructing the cycle graph on 4 vertices.
    graph = sgtl.Graph(C4_ADJ_MAT)

    # The graph has 4 vertices and 4 edges
    assert graph.number_of_vertices() == 4
    assert graph.number_of_edges() == 4

    # Check the vertex degrees
    for i in range(4):
        assert graph.degrees[i] == 2
        assert graph.inv_degrees[i] == 1/2
        assert graph.sqrt_degrees[i] == math.sqrt(2)
        assert graph.inv_sqrt_degrees[i] == 1 / math.sqrt(2)

    # Now, try creating the complete graph on 6 vertices.
    graph = sgtl.Graph(K6_ADJ_MAT)
    assert graph.number_of_vertices() == 6
    assert graph.number_of_edges() == 15
    for i in range(4):
        assert graph.degrees[i] == 5
        assert graph.inv_degrees[i] == 1/5
        assert graph.sqrt_degrees[i] == math.sqrt(5)
        assert graph.inv_sqrt_degrees[i] == 1 / math.sqrt(5)

    # Now, try the barbell graph
    graph = sgtl.Graph(BARBELL5_ADJ_MAT)
    assert graph.number_of_vertices() == 10
    assert graph.number_of_edges() == 21
    assert graph.degrees[2] == 4
    assert graph.degrees[4] == 5


def test_complete_graph():
    # Create a complete graph
    n = 4
    graph = sgtl.graph.complete_graph(n)
    expected_adjacency_matrix = sp.sparse.csr_matrix([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])

    assert graph.number_of_vertices() == 4
    adj_mat_diff = (graph.adj_mat - expected_adjacency_matrix)
    adj_mat_diff.eliminate_zeros()
    assert adj_mat_diff.nnz == 0

    # Make sure we can't do something stupid
    with pytest.raises(ValueError):
        _ = sgtl.graph.complete_graph(-10)
    with pytest.raises(ValueError):
        _ = sgtl.graph.complete_graph(0)


def test_cycle_graph():
    # Create a cycle graph
    n = 5
    graph = sgtl.graph.cycle_graph(n)
    expected_adjacency_matrix = sp.sparse.csr_matrix([[0, 1, 0, 0, 1],
                                                      [1, 0, 1, 0, 0],
                                                      [0, 1, 0, 1, 0],
                                                      [0, 0, 1, 0, 1],
                                                      [1, 0, 0, 1, 0]])

    assert graph.number_of_vertices() == 5
    adj_mat_diff = (graph.adj_mat - expected_adjacency_matrix)
    adj_mat_diff.eliminate_zeros()
    assert adj_mat_diff.nnz == 0

    # Make sure we can't do something stupid
    with pytest.raises(ValueError):
        _ = sgtl.graph.complete_graph(-10)
    with pytest.raises(ValueError):
        _ = sgtl.graph.complete_graph(0)


def test_star_graph():
    # Create a star graph
    n = 5
    graph = sgtl.graph.star_graph(n)
    expected_adjacency_matrix = sp.sparse.csr_matrix([[0, 1, 1, 1, 1],
                                                      [1, 0, 0, 0, 0],
                                                      [1, 0, 0, 0, 0],
                                                      [1, 0, 0, 0, 0],
                                                      [1, 0, 0, 0, 0]])

    assert graph.number_of_vertices() == 5
    adj_mat_diff = (graph.adj_mat - expected_adjacency_matrix)
    adj_mat_diff.eliminate_zeros()
    assert adj_mat_diff.nnz == 0

    # Make sure we can't do something stupid
    with pytest.raises(ValueError):
        _ = sgtl.graph.complete_graph(-10)
    with pytest.raises(ValueError):
        _ = sgtl.graph.complete_graph(0)


def test_path_graph():
    # Create a star graph
    n = 5
    graph = sgtl.graph.path_graph(n)
    expected_adjacency_matrix = sp.sparse.csr_matrix([[0, 1, 0, 0, 0],
                                                      [1, 0, 1, 0, 0],
                                                      [0, 1, 0, 1, 0],
                                                      [0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0]])

    assert graph.number_of_vertices() == 5
    adj_mat_diff = (graph.adj_mat - expected_adjacency_matrix)
    adj_mat_diff.eliminate_zeros()
    assert adj_mat_diff.nnz == 0

    # Make sure we can't do something stupid
    with pytest.raises(ValueError):
        _ = sgtl.graph.complete_graph(-10)
    with pytest.raises(ValueError):
        _ = sgtl.graph.complete_graph(0)


def test_volume():
    # Generate a known graph
    graph = sgtl.Graph(BARBELL5_ADJ_MAT)

    # Get the volume of a konwn set
    cluster = [0, 1, 2, 3, 4]
    volume = graph.volume(cluster)
    assert volume == 21


def test_weight():
    # Generate a known graph
    graph = sgtl.Graph(BARBELL5_ADJ_MAT)

    # Get the weight of edges from a set to itself
    cluster = [0, 1, 2, 3, 4]
    weight = graph.weight(cluster, cluster)
    assert weight == 10


def test_conductance():
    # Generate a known graph
    graph = sgtl.Graph(BARBELL5_ADJ_MAT)

    # Get the conductance of a known set
    cluster = [0, 1, 2, 3, 4]
    conductance = graph.conductance(cluster)
    assert abs(conductance - (1/21)) <= EPSILON


def test_symmetry():
    # Generate a large graph from the stochastic block model
    big_graph = sgtl.random.ssbm(1000, 5, 0.8, 0.2)

    # Check that all of the graph matrices are truly symmetric
    assert np.allclose(big_graph.adj_mat.toarray(), big_graph.adj_mat.toarray().T)

    lap_mat = big_graph.normalised_laplacian_matrix()
    lap_mat_dense = lap_mat.toarray()
    assert np.allclose(lap_mat_dense, lap_mat_dense.T)

    lap_mat = big_graph.laplacian_matrix()
    lap_mat_dense = lap_mat.toarray()
    assert np.allclose(lap_mat_dense, lap_mat_dense.T)


def test_num_edges():
    # Generate a known graph
    graph = sgtl.Graph(BARBELL5_ADJ_MAT)

    # Check the number of edges in the graph
    assert graph.number_of_vertices() == 10
    assert graph.number_of_edges() == 21
    assert graph.total_volume() == 21

    # Now create a weighted graph and check the number of edges method.
    adjacency_matrix = scipy.sparse.csr_matrix([[0, 2, 0, 1],
                                                [2, 0, 3, 0],
                                                [0, 3, 0, 1],
                                                [1, 0, 1, 0]])
    graph = sgtl.Graph(adjacency_matrix)
    assert graph.number_of_vertices() == 4
    assert graph.number_of_edges() == 4
    assert graph.total_volume() == 7

    # Test the number of edges for a graph with self-loops
    adjacency_matrix = scipy.sparse.csr_matrix([[0, 2, 0, 1],
                                                [2, 2, 3, 0],
                                                [0, 3, 0, 1],
                                                [1, 0, 1, 0]])
    graph = sgtl.Graph(adjacency_matrix)
    assert graph.number_of_vertices() == 4
    assert graph.number_of_edges() == 5
    assert graph.total_volume() == 9

    # Add more self-loops to a graph
    adjacency_matrix = scipy.sparse.csr_matrix([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                [1, 3, 1, 1, 1.5, 0, 0, 0, 0, 0],
                                                [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                                                [1, 1.5, 1, 0, 1, 0, 0, 0, 0, 0],
                                                [1, 1, 1, 1, 0.5, 1, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
                                                [0, 0, 0, 0, 0, 1, 0.5, 1, 2.5, 1],
                                                [0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
                                                [0, 0, 0, 0, 0, 1, 2.5, 1, 0, 0.5],
                                                [0, 0, 0, 0, 0, 1, 1, 1, 0.5, 0],
                                                ])
    graph = sgtl.Graph(adjacency_matrix)
    assert graph.number_of_vertices() == 10
    assert graph.number_of_edges() == 24
    assert graph.total_volume() == 26.5
