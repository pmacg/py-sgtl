"""Tests for the clustering algorithms."""
import scipy.sparse
import pytest
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


def test_spectral_clustering_bad_arguments():
    # Construct a graph object with the barbell adjacency matrix
    graph = sgtl.Graph(BARBELL5_ADJ_MAT)

    # Passing invalid arguments should fail
    with pytest.raises(ValueError):
        _ = sgtl.clustering.spectral_clustering(graph, 2, num_eigenvectors=0)
    with pytest.raises(ValueError):
        _ = sgtl.clustering.spectral_clustering(graph, 0)
    with pytest.raises(ValueError):
        _ = sgtl.clustering.spectral_clustering(graph, -1, num_eigenvectors=2)
    with pytest.raises(ValueError):
        _ = sgtl.clustering.spectral_clustering(graph, 2, num_eigenvectors=-1)

    # Passing a float as the number of clusters or eigenvalues should fail
    with pytest.raises(TypeError):
        _ = sgtl.clustering.spectral_clustering(graph, 2.5)
    with pytest.raises(TypeError):
        _ = sgtl.clustering.spectral_clustering(graph, 3.0)
    with pytest.raises(TypeError):
        _ = sgtl.clustering.spectral_clustering(graph, 2, num_eigenvectors=2.4)
