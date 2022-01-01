"""
A collection of spectral graph algorithms.
"""
from typing import Tuple, List, Set
import numpy as np
import scipy.sparse.linalg

import sgtl


def _sweep_set(graph: sgtl.Graph, vector: List[float]) -> Tuple[Set[int], Set[int]]:
    """
    Given an SGTL graph and a vector, use the sweep set algorithm to find a sparse cut in the graph.

    :param graph: The graph on which to operate.
    :param vector: The vector on which to sweep.
    :return: The set of vertices corresponding to the optimal cut
    """
    # Calculate n here once
    num_vertices = graph.number_of_vertices()

    # Keep track of the best cut so far
    best_cut_index = None
    best_conductance = None

    # Keep track of the size of the set and the cut weight to make computing the conductance
    # straightforward
    total_volume = graph.total_volume()
    set_volume = 0.0
    cut_weight = 0.0

    # Normalise the vector with the degrees of each vertex
    degree_matrix = graph.degree_matrix()
    vector = degree_matrix.power(-(1 / 2)).dot(vector)

    # First, sort the vertices based on their value in the given vector
    sorted_vertices = [i for i, _ in sorted(enumerate(vector), key=(lambda y: y[1]))]

    # Keep track of which edges to add/subtract from the cut each time
    edges_to_add = np.ones(num_vertices)

    # Loop through the vertices in the graph
    for (i, vertex) in enumerate(sorted_vertices[:-1]):
        # Update the set size and cut weight
        set_volume += graph.degrees[vertex]

        # From now on, edges to this vertex will be removed from the cut at each iteration.
        edges_to_add[vertex] = -1

        additional_weight = graph.adjacency_matrix()[vertex, :].dot(edges_to_add)
        cut_weight += additional_weight

        # Calculate the conductance
        this_conductance = cut_weight / min(set_volume, total_volume - set_volume)

        # Check whether this conductance is the best
        if best_conductance is None or this_conductance < best_conductance:
            best_cut_index = i
            best_conductance = this_conductance

    # Return the best cut
    return set(sorted_vertices[:best_cut_index + 1]), set(sorted_vertices[best_cut_index + 1:num_vertices])


def cheeger_cut(graph: sgtl.Graph) -> Tuple[Set[int], Set[int]]:
    """
    Given a graph G, find the cheeger cut. Returns a pair of lists containing the vertex indices of the two sides of
    the cut.

    :param graph: The graph on which to operate.
    :return: Two sets containing the vertices on each side of the cheeger cut.

    :Example:

    >>> import sgtl.graph
    >>> import sgtl.algorithms
    >>> graph = sgtl.graph.path_graph(10)
    >>> cut = sgtl.algorithms.cheeger_cut(graph)
    >>> sorted(cut, key=(lambda x: min(x)))
    [{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}]

    """
    # Compute the second smallest eigenvalue of the laplacian matrix
    laplacian_matrix = graph.normalised_laplacian_matrix()
    _, eig_vecs = scipy.sparse.linalg.eigsh(laplacian_matrix, which="SM", k=2)
    v_2 = eig_vecs[:, 1]

    # Perform the sweep set operation to find the sparsest cut
    return _sweep_set(graph, v_2)
