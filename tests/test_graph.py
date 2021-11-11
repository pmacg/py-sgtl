"""
Tests for the graph object.
"""
import scipy as sp
import scipy.sparse
import math
import pytest
from context import sgtl

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
    assert graph.num_vertices == 4
    assert graph.num_edges == 4

    # Check the vertex degrees
    for i in range(4):
        assert graph.degrees[i] == 2
        assert graph.inv_degrees[i] == 1/2
        assert graph.sqrt_degrees[i] == math.sqrt(2)
        assert graph.inv_sqrt_degrees[i] == 1 / math.sqrt(2)

    # Now, try creating the complete graph on 6 vertices.
    graph = sgtl.Graph(K6_ADJ_MAT)
    assert graph.num_vertices == 6
    assert graph.num_edges == 15
    for i in range(4):
        assert graph.degrees[i] == 5
        assert graph.inv_degrees[i] == 1/5
        assert graph.sqrt_degrees[i] == math.sqrt(5)
        assert graph.inv_sqrt_degrees[i] == 1 / math.sqrt(5)

    # Now, try the barbell graph
    graph = sgtl.Graph(BARBELL5_ADJ_MAT)
    assert graph.num_vertices == 10
    assert graph.num_edges == 21
    assert graph.degrees[2] == 4
    assert graph.degrees[4] == 5


def test_complete_graph():
    # Create a complete graph
    n = 4
    graph = sgtl.graph.complete_graph(n)
    expected_adjacency_matrix = sp.sparse.csr_matrix([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])

    assert graph.num_vertices == 4
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

    assert graph.num_vertices == 5
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

    assert graph.num_vertices == 5
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

    assert graph.num_vertices == 5
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
