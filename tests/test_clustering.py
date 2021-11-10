"""Tests for the clustering algorithms."""
import scipy
import scipy.sparse
import math
from context import sgtl
import sgtl.clustering

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


def test_spectral_clustering():
    # Construct a graph object with the barbell adjacency matrix
    graph = sgtl.Graph(BARBELL5_ADJ_MAT)

    # Cluster the graph
    clusters = sgtl.clustering.spectral_clustering(graph, 2)

    # Assert that the correct clusters have been found.
    assert (clusters[0] == [0, 1, 2, 3, 4] or clusters[0] == [5, 6, 7, 8, 9])
    assert (clusters[1] == sorted(list({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} - set(clusters[0]))))
