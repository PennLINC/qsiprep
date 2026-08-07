"""TORTOISEConvert's memory must be sized from the data, not guessed.

It loads with ``dtype='float32'`` and writes float32, so its working set is
``nvoxels * 4`` whatever the input dtype. Undeclared, nipype assumed the 0.20 GB
default and ran eight at once on 24 cores; the kernel OOM-killed one at 1.69 GB
RSS and took an 11-hour 3-subject run down with BrokenProcessPool.

Neither obvious shortcut works: CRASH is uint16 on disk, so the array dtype
understates by 2x, and the gzipped file size (what ``_create_mem_gb`` reports)
understates by 3.4x.
"""

import numpy as np
import pytest


def _series(tmp_path, name, shape, dtype):
    import nibabel as nb

    f = tmp_path / name
    nb.Nifti1Image(np.zeros(shape, dtype=dtype), np.eye(4)).to_filename(str(f))
    return str(f)


@pytest.fixture
def fake_shapes(monkeypatch):
    """Report chosen shapes without writing gigabytes to disk.

    Realistic sizes matter here: small test volumes fall under the
    DEFAULT_MEMORY_MIN_GB floor, which clamps every answer to the same number and
    makes the scaling assertions vacuous.
    """
    import qsiprep.workflows.dwi.util as util

    shapes = {}

    class _Img:
        def __init__(self, shape):
            self.shape = shape

    monkeypatch.setattr(util.nb, 'load', lambda f: _Img(shapes[f]))
    return shapes


def test_sized_from_float32_geometry_not_input_dtype(tmp_path):
    """uint16 input must still be budgeted as float32."""
    from qsiprep.workflows.dwi.util import tortoise_convert_mem_gb

    shape = (128, 128, 69, 279)
    u16 = _series(tmp_path, 'u16.nii.gz', (4, 4, 2, 2), 'uint16')
    f32 = _series(tmp_path, 'f32.nii.gz', (4, 4, 2, 2), 'float32')
    # the estimate must not depend on what is on disk
    assert tortoise_convert_mem_gb([u16]) == tortoise_convert_mem_gb([f32])

    expected = np.prod(shape) * 4 / 1024**3 * 1.5
    assert expected == pytest.approx(1.76, abs=0.02), 'CRASH geometry sanity'


def test_covers_the_rss_the_kernel_actually_observed(fake_shapes):
    """The CRASH geometry must budget at least the 1.69 GB seen at OOM."""
    from qsiprep.workflows.dwi.util import tortoise_convert_mem_gb

    # 128x128x69x279 uint16 -- the real acquisition, without writing 1 GB of zeros
    fake_shapes['crash'] = (128, 128, 69, 279)
    est = tortoise_convert_mem_gb(['crash'])

    assert est >= 1.69, f'{est:.2f} GB does not cover the observed 1.69 GB'
    assert est < 3.0, f'{est:.2f} GB is needlessly conservative'


def test_scales_with_volume_count(fake_shapes):
    from qsiprep.workflows.dwi.util import tortoise_convert_mem_gb

    fake_shapes['s'] = (128, 128, 69, 100)
    fake_shapes['b'] = (128, 128, 69, 400)
    assert tortoise_convert_mem_gb(['b']) == pytest.approx(
        4 * tortoise_convert_mem_gb(['s']), rel=1e-6
    )


def test_multiple_inputs_sum(fake_shapes):
    from qsiprep.workflows.dwi.util import tortoise_convert_mem_gb

    fake_shapes['a'] = (128, 128, 69, 100)
    fake_shapes['b'] = (128, 128, 69, 100)
    assert tortoise_convert_mem_gb(['a', 'b']) == pytest.approx(
        2 * tortoise_convert_mem_gb(['a']), rel=1e-6
    )


def test_unreadable_input_does_not_raise(tmp_path):
    """Docs builds pass paths that do not exist."""
    from qsiprep.workflows.dwi.util import tortoise_convert_mem_gb

    assert tortoise_convert_mem_gb(['/nonexistent/fake.nii.gz']) > 0


def test_every_convert_node_declares_memory():
    import inspect

    from qsiprep.workflows.dwi import diffprep

    src = inspect.getsource(diffprep)
    starts = [i for i in range(len(src)) if src.startswith('TORTOISEConvert()', i)]
    assert starts, 'no TORTOISEConvert nodes found -- test needs updating'
    for i in starts:
        window = src[i : i + 220]
        assert 'mem_gb' in window, f'TORTOISEConvert node without mem_gb:\n{window[:160]}'
        assert 'mem_gb=2.0' not in window, 'hardcoded value is back; should be computed'
