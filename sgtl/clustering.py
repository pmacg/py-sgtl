"""
Various graph clustering algorithms based on spectral graph theory.
"""
from typing import List
import scipy
import scipy.sparse
import scipy.sparse.linalg
from sklearn.cluster import KMeans

import sgtl


def spectral_clustering(graph: sgtl.Graph, num_clusters: int, num_eigenvectors=None) -> List[List[int]]:
    """
    Perform spectral clustering on the given graph object.

    :param graph: an SGTL graph object
    :param num_clusters: the number of clusters to find
    :param num_eigenvectors: (optional) the number of eigenvectors to use to find the clusters
    :return: A list of lists. Each list corresponds to the indices of the vertex in that cluster.

    :raises ValueError: if the requested number of clusters or eigenvectors are not a positive integer
    """
    # If the number of eigenvectors is not specified, use the same number as the number of clusters we are looking for.
    if num_eigenvectors is None:
        num_eigenvectors = num_clusters

    # If the number of eigenvectors, or the number of clusters is 0, we should raise an error
    if num_eigenvectors <= 0:
        raise ValueError("You must use more than 0 eigenvectors for spectral clustering.")
    if num_clusters <= 0:
        raise ValueError("You must find at least 1 cluster when using spectral clustering.")
    if not isinstance(num_clusters, int) or not isinstance(num_eigenvectors, int):
        raise TypeError("The number of clusters and eigenvectors must be positive integers.")

    # Get the normalised laplacian matrix of the graph
    laplacian_matrix = graph.normalised_laplacian_matrix()

    # Find the bottom eigenvectors of the laplacian matrix
    _, eigenvectors = scipy.sparse.linalg.eigsh(laplacian_matrix, num_eigenvectors, which='SM')

    # Perform k-means on the eigenvectors to find the clusters
    labels = KMeans(n_clusters=num_clusters).fit_predict(eigenvectors)

    # Split the clusters.
    clusters = [[] for _ in range(num_clusters)]
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    return clusters
