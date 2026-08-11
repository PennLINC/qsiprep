"""Every node that uses the GPU must be visible to nipype's GPU scheduler.

nipype decides whether a node competes for GPU slots with, and only with:

    def is_gpu_node(self):
        return bool(getattr(self.inputs, 'use_cuda', False)) or bool(
            getattr(self.inputs, 'use_gpu', False))

SynthStrip spells its request `gpu` and SynthSeg spells it `cpu` (opt-out), so
neither matched. With --gpu synthseg/synthstrip they ran on the device while the
scheduler treated them as pure CPU nodes and freely scheduled DIFFPREP, DRBUDDI
or eddy alongside them.
"""

import pytest


@pytest.mark.parametrize('name', ['DIFFPREP', 'DRBUDDI'])
def test_tortoise_nodes_are_visible(name):
    import qsiprep.interfaces.tortoise as t

    assert 'use_cuda' in getattr(t, name).input_spec().trait_names()


def test_eddy_is_visible():
    from qsiprep.interfaces.eddy import ExtendedEddy

    assert 'use_cuda' in ExtendedEddy.input_spec().trait_names()


@pytest.mark.parametrize('name', ['SynthSeg', 'SynthStrip'])
def test_freesurfer_gpu_nodes_are_visible(name):
    import qsiprep.interfaces.freesurfer as fs

    assert 'use_gpu' in getattr(fs, name).input_spec().trait_names()


@pytest.mark.parametrize('flag', [True, False])
def test_is_gpu_node_tracks_use_gpu(flag):
    from nipype.pipeline import engine as pe

    from qsiprep.interfaces.freesurfer import SynthStrip

    node = pe.Node(SynthStrip(use_gpu=flag), name=f'ss_{flag}')
    assert node.is_gpu_node() is flag


def test_use_gpu_is_not_passed_on_the_command_line():
    """It exists for resource accounting only; the tool has its own flag."""
    from qsiprep.interfaces.freesurfer import SynthStrip

    spec = SynthStrip.input_spec()
    assert spec.trait('use_gpu').argstr in (None, '')
