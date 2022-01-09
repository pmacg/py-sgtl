"""
This module contains methods relating to the spectrum of a graph.
"""
from typing import List
import scipy
import scipy.linalg
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import sgtl


def _plot_spectrum(spectrum):
    """
    Plot the given list of floats as a matplotlib point plot. If the eigenvalues given are complex, we
    ignore the imaginary part!
    """
    # If the given eigenvalues are complex, ignore the imaginary part!
    if len(spectrum) > 0 and isinstance(spectrum[0], complex):
        spectrum = [x.real for x in spectrum]

    plt.plot(sorted(spectrum), '.')
    plt.show()


def adjacency_spectrum(graph: sgtl.Graph, num_eigenvalues=None, plot=False) -> List[complex]:
    """
    Return a list of the eigenvalues of the adjacency matrix of the given graph.

    Computing all of the eigenvalues of the adjacency matrix is an expensive operation. You should set num_eigenvalues
    to control the number of eigenvalues you would like to compute.

    If the plot argument is true, then the spectrum is plotted using matplotlib before being returned.

    :param graph: The SGTL graph object on which to operate.
    :param num_eigenvalues: How many eigenvalues to return. If this value is not none, the method will return the
                            eigenvalues with the maximum absolute values.
    :param plot: Whether to plot the spectrum before returning it
    """
    if num_eigenvalues is None:
        num_eigenvalues = graph.number_of_vertices()

    if num_eigenvalues >= graph.number_of_vertices() - 1:
        eigvals, _ = scipy.linalg.eig(graph.adjacency_matrix().toarray())
    else:
        eigvals, _ = scipy.sparse.linalg.eigs(graph.adjacency_matrix(), k=num_eigenvalues)

    # Plot if required
    if plot:
        _plot_spectrum(eigvals)

    return list(eigvals)


def normalised_adjacency_spectrum(graph: sgtl.Graph, num_eigenvalues=None, plot=False) -> List[complex]:
    """
    Return a list of the eigenvalues of the normalised adjacency matrix of the given graph.

    Computing all of the eigenvalues of the adjacency matrix is an expensive operation. You should set num_eigenvalues
    to control the number of eigenvalues you would like to compute.

    If the plot argument is true, then the spectrum is plotted using matplotlib before being returned.

    :param graph: The SGTL graph object on which to operate.
    :param num_eigenvalues: How many eigenvalues to return. If this value is not none, the method will return the
                            eigenvalues with the maximum absolute values.
    :param plot: Whether to plot the spectrum before returning it
    """
    if num_eigenvalues is None:
        num_eigenvalues = graph.number_of_vertices()

    if num_eigenvalues >= graph.number_of_vertices() - 1:
        eigvals, _ = scipy.linalg.eig(graph.normalised_adjacency_matrix().toarray())
    else:
        eigvals, _ = scipy.sparse.linalg.eigs(graph.normalised_adjacency_matrix(), k=num_eigenvalues)

    # Plot if required
    if plot:
        _plot_spectrum(eigvals)

    return list(eigvals)


def laplacian_spectrum(graph: sgtl.Graph, num_eigenvalues=None, magnitude='smallest', plot=False) -> List[complex]:
    """
    Return a list of the eigenvalues of the laplacian matrix of the given graph.

    Computing all of the eigenvalues of the laplacian matrix is an expensive operation. You should set num_eigenvalues
    to control the number of eigenvalues you would like to compute.

    When computing only a subset of the eigenvalues, the 'magnitude' parameter controls whether the eigenvalues with
    largest or smallest magnitude will be returned. This defaults to 'smallest'.

    If the plot argument is true, then the spectrum is plotted using matplotlib before being returned.

    :param graph: The SGTL graph object on which to operate.
    :param num_eigenvalues: How many eigenvalues to return.
    :param magnitude: Should be 'smallest' or 'largest' - whether to return the eigenvalues with smallest or largest
                      magnitude.
    :param plot: Whether to plot the spectrum before returning it
    """
    if num_eigenvalues is None:
        num_eigenvalues = graph.number_of_vertices()

    # Modify our magnitude parameter to the one used by scipy
    if magnitude not in ['smallest', 'largest']:
        raise ValueError("Magnitude parameter must be either 'smallest' or 'largest'.")
    mag_sp = 'LM' if magnitude == 'largest' else 'SM'

    if num_eigenvalues >= graph.number_of_vertices() - 1:
        eigvals, _ = scipy.linalg.eig(graph.laplacian_matrix().toarray())
    else:
        eigvals, _ = scipy.sparse.linalg.eigs(graph.laplacian_matrix(), k=num_eigenvalues, which=mag_sp)

    # Plot if required
    if plot:
        _plot_spectrum(eigvals)

    return list(eigvals)


def normalised_laplacian_spectrum(graph: sgtl.Graph,
                                  num_eigenvalues=None,
                                  magnitude='smallest',
                                  plot=False) -> List[complex]:
    """
    Return a list of the eigenvalues of the normalised laplacian matrix of the given graph.

    Computing all of the eigenvalues of the laplacian matrix is an expensive operation. You should set num_eigenvalues
    to control the number of eigenvalues you would like to compute.

    When computing only a subset of the eigenvalues, the 'magnitude' parameter controls whether the eigenvalues with
    largest or smallest magnitude will be returned. This defaults to 'smallest'.

    If the plot argument is true, then the spectrum is plotted using matplotlib before being returned.

    :param graph: The SGTL graph object on which to operate.
    :param num_eigenvalues: How many eigenvalues to return.
    :param magnitude: Should be 'smallest' or 'largest' - whether to return the eigenvalues with smallest or largest
                      magnitude.
    :param plot: Whether to plot the spectrum before returning it
    """
    if num_eigenvalues is None:
        num_eigenvalues = graph.number_of_vertices()

    # Modify our magnitude parameter to the one used by scipy
    if magnitude not in ['smallest', 'largest']:
        raise ValueError("Magnitude parameter must be either 'smallest' or 'largest'.")
    mag_sp = 'LM' if magnitude == 'largest' else 'SM'

    if num_eigenvalues >= graph.number_of_vertices() - 1:
        eigvals, _ = scipy.linalg.eig(graph.normalised_laplacian_matrix().toarray())
    else:
        eigvals, _ = scipy.sparse.linalg.eigs(graph.normalised_laplacian_matrix(), k=num_eigenvalues, which=mag_sp)

    # Plot if required
    if plot:
        _plot_spectrum(eigvals)

    return list(eigvals)
