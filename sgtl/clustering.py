"""
Various graph clustering algorithms based on spectral graph theory.
"""
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
from sklearn.cluster import KMeans


def spectral_clustering(graph, num_clusters, num_eigenvectors=None):
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
    if type(num_clusters) is not int or type(num_eigenvectors) is not int:
        raise TypeError("The number of clusters and eigenvectors must be positive integers.")

    # Get the normalised laplacian matrix of the graph
    laplacian_matrix = graph.normalised_laplacian_matrix()

    # Find the bottom eigenvectors of the laplacian matrix
    _, eigenvectors = sp.sparse.linalg.eigsh(laplacian_matrix, num_eigenvectors, which='SM')

    # Perform k-means on the eigenvectors to find the clusters
    labels = KMeans(n_clusters=num_clusters).fit_predict(eigenvectors)

    # Split the clusters.
    clusters = [[] for _ in range(num_clusters)]
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    return clusters
