"""
Tests for the graph object.
"""
import scipy
import scipy.sparse
import math
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
