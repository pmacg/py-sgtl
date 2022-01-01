"""
Tests for the spectrum module.
"""
from context import sgtl
import sgtl.spectrum
import pytest


def test_adjacency_spectrum():
    # Check the spectrum of a complete graph.
    graph = sgtl.graph.complete_graph(10)
    spectrum = sorted(sgtl.spectrum.adjacency_spectrum(graph))

    assert spectrum[0] == pytest.approx(-1)
    assert spectrum[2] == pytest.approx(-1)
    assert spectrum[9] == pytest.approx(9)

    # Check that we get the correct number of eigenvalues
    spectrum = sgtl.spectrum.adjacency_spectrum(graph, num_eigenvalues=3)
    assert len(spectrum) == 3
    assert spectrum[0] == pytest.approx(9)


def test_laplacian_spectrum():
    # Check the spectrum of a complete graph.
    graph = sgtl.graph.complete_graph(10)
    spectrum = sorted(sgtl.spectrum.laplacian_spectrum(graph))

    assert spectrum[0] == pytest.approx(0)
    assert spectrum[2] == pytest.approx(10)
    assert spectrum[9] == pytest.approx(10)

    # Check that we get the correct number of eigenvalues, and that we can request from either end of the spectrum.
    spectrum = sgtl.spectrum.laplacian_spectrum(graph, num_eigenvalues=3)
    assert len(spectrum) == 3
    assert spectrum[0] == pytest.approx(0)      # By default, we take from the 'bottom end' of the spectrum.

    spectrum = sgtl.spectrum.laplacian_spectrum(graph, num_eigenvalues=4, magnitude='largest')
    assert len(spectrum) == 4
    assert spectrum[0] == pytest.approx(10)


def test_normalised_laplacian_spectrum():
    # Check the spectrum of a complete graph.
    graph = sgtl.graph.complete_graph(10)
    spectrum = sorted(sgtl.spectrum.normalised_laplacian_spectrum(graph))

    assert spectrum[0] == pytest.approx(0)
    assert spectrum[2] == pytest.approx(10/9)
    assert spectrum[9] == pytest.approx(10/9)

    # Check that we get the correct number of eigenvalues, and that we can request from either end of the spectrum.
    spectrum = sgtl.spectrum.normalised_laplacian_spectrum(graph, num_eigenvalues=3)
    assert len(spectrum) == 3
    assert spectrum[0] == pytest.approx(0)      # By default, we take from the 'bottom end' of the spectrum.

    spectrum = sgtl.spectrum.normalised_laplacian_spectrum(graph, num_eigenvalues=4, magnitude='largest')
    assert len(spectrum) == 4
    assert spectrum[0] == pytest.approx(10/9)


def test_compare_spectra():
    # Test that the adjacency and laplacian spectra relate in the way we'd expect.
    graph = sgtl.graph.cycle_graph(10)
    a_spectrum = sorted(sgtl.spectrum.adjacency_spectrum(graph))
    l_spectrum = sorted(sgtl.spectrum.laplacian_spectrum(graph))[::-1]
    for i in range(10):
        assert l_spectrum[i] == pytest.approx(2 - a_spectrum[i])
