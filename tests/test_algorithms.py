"""
Tests for the algorithms module.
"""
from context import sgtl
import sgtl.algorithms

import test_graph


def test_cheeger_cut():
    graph = sgtl.graph.path_graph(10)
    cut = sgtl.algorithms.cheeger_cut(graph)
    assert sorted(cut) == [{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}]

    graph = sgtl.Graph(test_graph.BARBELL5_ADJ_MAT)
    cut = sgtl.algorithms.cheeger_cut(graph)
    assert sorted(cut) == [{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}]

    graph = sgtl.graph.complete_graph(10)
    cut = sgtl.algorithms.cheeger_cut(graph)
    assert len(cut[0]) == 5
    assert len(cut[1]) == 5
