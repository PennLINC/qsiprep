"""Ensure phase-correction citations exist in the boilerplate bibliography."""

from qsiprep.data import load as load_data


def test_phase_citations_present():
    bib = load_data('boilerplate.bib').read_text()
    assert '@article{eichner2015real' in bib
    assert '@article{sprenger2017real' in bib
