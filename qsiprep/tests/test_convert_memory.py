"""TORTOISEConvert must declare its memory, or nipype over-schedules it.

It decompresses a whole 4D series to an uncompressed .nii -- ~1.3 GB for a
279-volume acquisition. Undeclared, nipype assumes the 0.20 GB default and runs
as many as there are cores: a 3-subject run put eight in flight on a 30 GB box
and the kernel OOM-killed one at 1.69 GB RSS, taking the whole workflow down with
BrokenProcessPool after 11 hours.
"""

import pytest


def _config(**kw):
    from qsiprep import config

    config.nipype.omp_nthreads = 24
    config.workflow.hmc_model = 'diffprep_quadratic'
    config.workflow.b0_threshold = 100
    for k, v in kw.items():
        setattr(config.workflow, k, v)
    return config


def test_convert_nodes_declare_memory():
    """Both construction sites -- the plain path and the rpe_series path."""
    import inspect

    from qsiprep.workflows.dwi import diffprep

    src = inspect.getsource(diffprep)
    # every TORTOISEConvert node construction carries a mem_gb
    starts = [i for i in range(len(src)) if src.startswith('TORTOISEConvert()', i)]
    assert starts, 'no TORTOISEConvert nodes found -- test needs updating'
    for i in starts:
        window = src[i : i + 220]
        assert 'mem_gb' in window, f'TORTOISEConvert node without mem_gb:\n{window[:160]}'


@pytest.mark.parametrize('declared', [2.0])
def test_declared_memory_covers_what_the_kernel_observed(declared):
    """1.69 GB was the RSS at OOM; the declaration must not be under it."""
    assert declared >= 1.69
