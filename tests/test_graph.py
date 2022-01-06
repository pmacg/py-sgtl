"""
Tests for the graph object.
"""
import numpy as np
import scipy as sp
import scipy.sparse
import math
import networkx
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
    adj_mat_diff = (graph.adjacency_matrix() - expected_adjacency_matrix)
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
    adj_mat_diff = (graph.adjacency_matrix() - expected_adjacency_matrix)
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
    adj_mat_diff = (graph.adjacency_matrix() - expected_adjacency_matrix)
    adj_mat_diff.eliminate_zeros()
    assert adj_mat_diff.nnz == 0

    # Make sure we can't do something stupid
    with pytest.raises(ValueError):
        _ = sgtl.graph.complete_graph(-10)
    with pytest.raises(ValueError):
        _ = sgtl.graph.complete_graph(0)


def test_path_graph():
    # Create a path graph
    n = 5
    graph = sgtl.graph.path_graph(n)
    expected_adjacency_matrix = sp.sparse.csr_matrix([[0, 1, 0, 0, 0],
                                                      [1, 0, 1, 0, 0],
                                                      [0, 1, 0, 1, 0],
                                                      [0, 0, 1, 0, 1],
                                                      [0, 0, 0, 1, 0]])

    assert graph.number_of_vertices() == 5
    adj_mat_diff = (graph.adjacency_matrix() - expected_adjacency_matrix)
    adj_mat_diff.eliminate_zeros()
    assert adj_mat_diff.nnz == 0

    assert graph.total_volume() == 8

    # Make sure we can't do something stupid
    with pytest.raises(ValueError):
        _ = sgtl.graph.complete_graph(-10)
    with pytest.raises(ValueError):
        _ = sgtl.graph.complete_graph(0)


def test_adjacency_matrix():
    graph = sgtl.Graph(BARBELL5_ADJ_MAT)
    adj_mat_diff = (graph.adjacency_matrix() - BARBELL5_ADJ_MAT)
    adj_mat_diff.eliminate_zeros()
    assert adj_mat_diff.nnz == 0


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
    assert conductance == pytest.approx(1/21)


def test_symmetry():
    # Generate a large graph from the stochastic block model
    big_graph = sgtl.random.ssbm(1000, 5, 0.8, 0.2)

    # Check that all of the graph matrices are truly symmetric
    assert np.allclose(big_graph.adjacency_matrix().toarray(), big_graph.adjacency_matrix().toarray().T)

    lap_mat = big_graph.normalised_laplacian_matrix()
    lap_mat_dense = lap_mat.toarray()
    assert np.allclose(lap_mat_dense, lap_mat_dense.T)

    lap_mat = big_graph.laplacian_matrix()
    lap_mat_dense = lap_mat.toarray()
    assert np.allclose(lap_mat_dense, lap_mat_dense.T)


def test_out_of_range():
    # Create a graph
    graph = sgtl.graph.complete_graph(5)

    # Check the error message when trying to access a vertex which doesn't exist
    with pytest.raises(IndexError, match="Input vertex set includes indices larger than the number of vertices."):
        _ = graph.volume([6])
    with pytest.raises(IndexError, match="Input vertex set includes indices larger than the number of vertices."):
        _ = graph.volume([0, 1, 2, 3, 4, 5])
    with pytest.raises(IndexError, match="Input vertex set includes indices larger than the number of vertices."):
        _ = graph.weight([0, 1, 2], [6])


def test_cond_empty_set():
    # Create a graph
    graph = sgtl.graph.complete_graph(5)

    # Test the behaviour when asking for the conductance or bipartiteness of the empty set
    with pytest.raises(ValueError, match="The conductance of the empty set is undefined."):
        _ = graph.conductance([])
    with pytest.raises(ValueError, match="The bipartiteness of the empty set is undefined."):
        _ = graph.bipartiteness([], [])

    # If only one set is empty, then the bipartiteness is defined
    bip = graph.bipartiteness([0, 1], [])
    assert bip == 1

    bip = graph.bipartiteness([], [2, 3, 4])
    assert bip == 1


def test_num_edges():
    # Generate a known graph
    graph = sgtl.Graph(BARBELL5_ADJ_MAT)

    # Check the number of edges in the graph
    assert graph.number_of_vertices() == 10
    assert graph.number_of_edges() == 21
    assert graph.total_volume() == 42

    # Now create a weighted graph and check the number of edges method.
    adjacency_matrix = scipy.sparse.csr_matrix([[0, 2, 0, 1],
                                                [2, 0, 3, 0],
                                                [0, 3, 0, 1],
                                                [1, 0, 1, 0]])
    graph = sgtl.Graph(adjacency_matrix)
    assert graph.number_of_vertices() == 4
    assert graph.number_of_edges() == 4
    assert graph.total_volume() == 14

    # Test the number of edges for a graph with self-loops
    adjacency_matrix = scipy.sparse.csr_matrix([[0, 2, 0, 1],
                                                [2, 2, 3, 0],
                                                [0, 3, 0, 1],
                                                [1, 0, 1, 0]])
    graph = sgtl.Graph(adjacency_matrix)
    assert graph.number_of_vertices() == 4
    assert graph.number_of_edges() == 5
    assert graph.total_volume() == 16

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
    assert graph.total_volume() == 49


def test_float_weights():
    # Create a graph with floating-point edge weights.
    adjacency_matrix = scipy.sparse.csr_matrix([[0, 2.2, 0, 1],
                                                [2.2, 2.6, 3.1, 0],
                                                [0, 3.1, 0, 1.09],
                                                [1, 0, 1.09, 0]])
    graph = sgtl.Graph(adjacency_matrix)
    assert graph.number_of_vertices() == 4
    assert graph.number_of_edges() == 5
    assert graph.total_volume() == pytest.approx(17.38)

    # Check the weight between vertex sets
    assert graph.weight([0], [1]) == pytest.approx(2.2)
    assert graph.weight([0], [1, 2, 3]) == pytest.approx(3.2)
    assert graph.weight([0, 1], [0, 1]) == pytest.approx(4.8)


def test_networkx():
    # Test the methods for converting from and to networkx graphs.
    # Start by constructing a networkx graph
    netx_graph = networkx.generators.barbell_graph(4, 1)
    graph = sgtl.Graph.from_networkx(netx_graph)

    assert graph.number_of_vertices() == 9
    assert graph.number_of_edges() == 14

    expected_adjacency_matrix = sp.sparse.csr_matrix([[0, 1, 1, 1, 0, 0, 0, 0, 0],
                                                      [1, 0, 1, 1, 0, 0, 0, 0, 0],
                                                      [1, 1, 0, 1, 0, 0, 0, 0, 0],
                                                      [1, 1, 1, 0, 0, 0, 0, 0, 1],
                                                      [0, 0, 0, 0, 0, 1, 1, 1, 1],
                                                      [0, 0, 0, 0, 1, 0, 1, 1, 0],
                                                      [0, 0, 0, 0, 1, 1, 0, 1, 0],
                                                      [0, 0, 0, 0, 1, 1, 1, 0, 0],
                                                      [0, 0, 0, 1, 1, 0, 0, 0, 0]])
    adj_mat_diff = (graph.adjacency_matrix() - expected_adjacency_matrix)
    adj_mat_diff.eliminate_zeros()
    assert adj_mat_diff.nnz == 0

    # Now, construct a graph using the sgtl Graph object, and convert it to networkx
    graph = sgtl.Graph(expected_adjacency_matrix)
    netx_graph = graph.to_networkx()

    # Check that the networkx graph looks correct
    assert netx_graph.number_of_nodes() == 9
    assert netx_graph.number_of_edges() == 14
    assert netx_graph.has_edge(0, 1)
    assert netx_graph.has_edge(3, 8)
    assert netx_graph.has_edge(8, 4)
    assert not netx_graph.has_edge(2, 8)


def test_knn_graph():
    # Let's construct some data where we can calculate the knn graph by hand.
    raw_data = np.asarray([[1, 1], [4, 1], [1, 2], [2, 2], [2, 3]])
    expected_adj_mat = sp.sparse.csr_matrix([[0, 0, 1, 1, 0],
                                             [0, 0, 0, 1, 1],
                                             [1, 0, 0, 1, 1],
                                             [1, 1, 1, 0, 1],
                                             [0, 1, 1, 1, 0]])
    graph = sgtl.graph.knn_graph(raw_data, 2)

    adj_mat_diff = (graph.adjacency_matrix() - expected_adj_mat)
    adj_mat_diff.eliminate_zeros()
    assert adj_mat_diff.nnz == 0
    assert graph.number_of_vertices() == 5

    # Let's do one more, with k = 3 this time.
    raw_data = np.asarray([[1, 1], [2, 1], [3, 1], [5, 1], [2, 2], [3, 2], [3, 4]])
    expected_adj_mat = sp.sparse.csr_matrix([[0, 1, 1, 0, 1, 0, 0],
                                             [1, 0, 1, 1, 1, 1, 0],
                                             [1, 1, 0, 1, 1, 1, 1],
                                             [0, 1, 1, 0, 0, 1, 0],
                                             [1, 1, 1, 0, 0, 1, 1],
                                             [0, 1, 1, 1, 1, 0, 1],
                                             [0, 0, 1, 0, 1, 1, 0]])
    graph = sgtl.graph.knn_graph(raw_data, 3)

    adj_mat_diff = (graph.adjacency_matrix() - expected_adj_mat)
    adj_mat_diff.eliminate_zeros()
    assert adj_mat_diff.nnz == 0
    assert graph.number_of_vertices() == 7

    # Construct a knn graph with a sparse data matrix
    raw_data = sp.sparse.csr_matrix(raw_data)
    graph = sgtl.graph.knn_graph(raw_data, 3)
    adj_mat_diff = (graph.adjacency_matrix() - expected_adj_mat)
    adj_mat_diff.eliminate_zeros()
    assert adj_mat_diff.nnz == 0
    assert graph.number_of_vertices() == 7



def test_edgelist():
    ##########
    # TEST 1 #
    ##########
    # Let's load the different test graphs, and check that we get what we'd expect.
    expected_adj_mat = sp.sparse.csr_matrix([[0, 1, 1],
                                             [1, 0, 1],
                                             [1, 1, 0]])
    graph = sgtl.graph.from_edgelist("data/test1.edgelist")
    adj_mat_diff = (graph.adjacency_matrix() - expected_adj_mat)
    adj_mat_diff.eliminate_zeros()
    assert adj_mat_diff.nnz == 0

    # Now save and reload the graph and check that the adjacency matrix has not changed
    sgtl.graph.to_edgelist(graph, "data/temp.edgelist")
    graph = sgtl.graph.from_edgelist("data/temp.edgelist")
    adj_mat_diff = (graph.adjacency_matrix() - expected_adj_mat)
    adj_mat_diff.eliminate_zeros()
    assert adj_mat_diff.nnz == 0

    ##########
    # TEST 2 #
    ##########
    expected_adj_mat = sp.sparse.csr_matrix([[0, 0.5, 0.5],
                                             [0.5, 0, 1],
                                             [0.5, 1, 0]])
    graph = sgtl.graph.from_edgelist("data/test2.edgelist")
    adj_mat_diff = (graph.adjacency_matrix() - expected_adj_mat)
    adj_mat_diff.eliminate_zeros()
    assert adj_mat_diff.nnz == 0

    # Now save and reload the graph and check that the adjacency matrix has not changed
    sgtl.graph.to_edgelist(graph, "data/temp.edgelist")
    graph = sgtl.graph.from_edgelist("data/temp.edgelist")
    adj_mat_diff = (graph.adjacency_matrix() - expected_adj_mat)
    adj_mat_diff.eliminate_zeros()
    assert adj_mat_diff.nnz == 0

    ##########
    # TEST 3 #
    ##########
    expected_adj_mat = sp.sparse.csr_matrix([[0, 1, 0.5],
                                             [1, 0, 1],
                                             [0.5, 1, 0]])
    graph = sgtl.graph.from_edgelist("data/test3.edgelist", comment='/')
    adj_mat_diff = (graph.adjacency_matrix() - expected_adj_mat)
    adj_mat_diff.eliminate_zeros()
    assert adj_mat_diff.nnz == 0

    # Now save and reload the graph and check that the adjacency matrix has not changed
    sgtl.graph.to_edgelist(graph, "data/temp.edgelist")
    graph = sgtl.graph.from_edgelist("data/temp.edgelist")
    adj_mat_diff = (graph.adjacency_matrix() - expected_adj_mat)
    adj_mat_diff.eliminate_zeros()
    assert adj_mat_diff.nnz == 0

    ##########
    # TEST 4 #
    ##########
    expected_adj_mat = sp.sparse.csr_matrix([[0, 1, 0],
                                             [0, 0, 1],
                                             [0.5, 0, 0]])
    graph = sgtl.graph.from_edgelist("data/test4.edgelist", num_vertices=3, directed=True)
    adj_mat_diff = (graph.adjacency_matrix() - expected_adj_mat)
    adj_mat_diff.eliminate_zeros()
    assert adj_mat_diff.nnz == 0

    # Now save and reload the graph and check that the adjacency matrix has not changed
    sgtl.graph.to_edgelist(graph, "data/temp.edgelist")
    graph = sgtl.graph.from_edgelist("data/temp.edgelist", directed=True)
    adj_mat_diff = (graph.adjacency_matrix() - expected_adj_mat)
    adj_mat_diff.eliminate_zeros()
    assert adj_mat_diff.nnz == 0
