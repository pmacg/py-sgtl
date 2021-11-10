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
    :return: A list of lists containing
    """
    # If the number of eigenvectors is not specified, use the same number as the number of clusters we are looking for.
    if not num_eigenvectors:
        num_eigenvectors = num_clusters

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
