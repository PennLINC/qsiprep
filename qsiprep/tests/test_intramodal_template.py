"""Intramodal template transform selection and centre-of-mass initialization.

Two defects:

1. ``--intramodal-template-transform`` and ``--intramodal-template-iters`` were
   never passed to the workflow, so every template was BSplineSyN with 2
   iterations regardless of what the user asked for -- silently warping genuine
   between-session differences into agreement for anyone who chose a linear
   transform to avoid exactly that.
2. ``Rigid`` is not in antsMultivariateTemplateConstruction2's enum
   (BSplineSyN/SyN/Affine), so the CLI advertised a choice the backend could not
   honour. Linear templates now go through ``init_b0_hmc_wf`` instead.
"""

import pytest


def _config():
    from qsiprep import config

    config.execution.sloppy = False
    config.nipype.omp_nthreads = 1
    return config


def _build(transform, num_iterations=2, name=None):
    from qsiprep.workflows.dwi.intramodal_template import init_intramodal_template_wf

    _config()
    return init_intramodal_template_wf(
        inputs_list=['group_a', 'group_b'],
        t1w_source_file='/data/sub-01_T1w.nii.gz',
        transform=transform,
        num_iterations=num_iterations,
        name=name or f'imt_{transform}',
    )


def _names(wf):
    return wf.list_node_names()


@pytest.mark.parametrize('transform', ['Rigid', 'Affine'])
def test_linear_transforms_use_the_b0_hmc_workflow(transform):
    """antsMultivariateTemplateConstruction2 cannot do Rigid at all."""
    names = _names(_build(transform))
    assert any('intramodal_linear_template' in n for n in names)
    assert not any('ants_mvtc2' in n for n in names)


@pytest.mark.parametrize('transform', ['BSplineSyN', 'SyN'])
def test_nonlinear_transforms_still_use_mvtc2(transform):
    names = _names(_build(transform))
    assert any('ants_mvtc2' in n for n in names)
    assert not any('intramodal_linear_template' in n for n in names)


def test_requested_transform_reaches_the_nonlinear_backend():
    """The transform used to be dropped, leaving mvtc2 on its BSplineSyN default."""
    wf = _build('SyN')
    node = next(n for n in wf._get_all_nodes() if n.name == 'ants_mvtc2')
    assert node.inputs.transform == 'SyN'


@pytest.mark.parametrize('transform', ['Rigid', 'Affine'])
def test_linear_template_initializes_by_centre_of_mass(transform):
    """Sessions can differ by centimetres of table position.

    The shoreline settings carry no initialization and only two resolution
    levels, so without a centre-of-mass start a Rigid metric can fail to recover
    a large offset.
    """
    wf = _build(transform, name=f'com_{transform}')
    regs = [n for n in wf._get_all_nodes() if n.name.startswith('reg_')]
    assert regs, 'no registration nodes found in the linear template'
    for node in regs:
        assert node.inputs.initial_moving_transform_com == 1


def test_dwi_b0_alignment_does_not_initialize_by_com_by_default():
    """The b=0 HMC callers must be unaffected: volumes there already overlap."""
    from qsiprep.workflows.dwi.hmc import init_b0_hmc_wf

    _config()
    wf = init_b0_hmc_wf(align_to='iterative', transform='Rigid', name='plain_b0_hmc')
    regs = [n for n in wf._get_all_nodes() if n.name.startswith('reg_')]
    assert regs
    for node in regs:
        from nipype.interfaces.base import isdefined

        assert not isdefined(node.inputs.initial_moving_transform_com)


def test_iteration_count_is_honoured():
    """--intramodal-template-iters was ignored; the count was always 2."""
    wf = _build('BSplineSyN', num_iterations=5, name='iters_nonlinear')
    node = next(n for n in wf._get_all_nodes() if n.name == 'ants_mvtc2')
    assert node.inputs.iteration_limit == 5
