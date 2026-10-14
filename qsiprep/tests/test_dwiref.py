"""Intramodal template transform selection and centre-of-mass initialization.

Two defects:

1. ``--dwiref-construction-transform`` and ``--dwiref-construction-iters`` were
   never passed to the workflow, so every template was BSplineSyN with 2
   iterations regardless of what the user asked for -- silently warping genuine
   between-session differences into agreement for anyone who chose a linear
   transform to avoid exactly that.
2. ``Rigid`` is not in antsMultivariateTemplateConstruction2's enum
   (BSplineSyN/SyN/Affine), so the CLI advertised a choice the backend could not
   honour. Linear templates now go through ``init_b0_hmc_wf`` instead.
"""

import pytest


def _config(dwi2anat_dof=6):
    from qsiprep import config

    config.execution.sloppy = False
    config.nipype.omp_nthreads = 1
    config.workflow.dwi2anat_dof = dwi2anat_dof
    return config


def _build(transform, num_iterations=2, name=None, dwi2anat_dof=6):
    from qsiprep.workflows.dwi.dwiref import init_dwiref_wf

    _config(dwi2anat_dof)
    return init_dwiref_wf(
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
    """Test that linear transforms use the b=0 HMC workflow.

    antsMultivariateTemplateConstruction2 cannot do Rigid at all.
    """
    names = _names(_build(transform))
    assert any('dwiref_linear_template' in n for n in names)
    assert not any('ants_mvtc2' in n for n in names)


@pytest.mark.parametrize('transform', ['BSplineSyN', 'SyN'])
def test_nonlinear_transforms_still_use_mvtc2(transform):
    names = _names(_build(transform))
    assert any('ants_mvtc2' in n for n in names)
    assert not any('dwiref_linear_template' in n for n in names)


def test_requested_transform_reaches_the_nonlinear_backend():
    """Test that the requested transform reaches the nonlinear backend.

    The transform used to be dropped, leaving mvtc2 on its BSplineSyN default.
    """
    wf = _build('SyN')
    node = next(n for n in wf._get_all_nodes() if n.name == 'ants_mvtc2')
    assert node.inputs.transform == 'SyN'


@pytest.mark.parametrize('transform', ['Rigid', 'Affine'])
def test_linear_template_initializes_by_centre_of_mass(transform):
    """Test that the linear template initializes by centre of mass.

    Sessions can differ by centimetres of table position.

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
    """Test that b=0 alignment does not initialize by centre of mass by default.

    The b=0 HMC callers must be unaffected: volumes there already overlap.
    """
    from qsiprep.workflows.dwi.hmc import init_b0_hmc_wf

    _config()
    wf = init_b0_hmc_wf(align_to='iterative', transform='Rigid', name='plain_b0_hmc')
    regs = [n for n in wf._get_all_nodes() if n.name.startswith('reg_')]
    assert regs
    for node in regs:
        from nipype.interfaces.base import isdefined

        assert not isdefined(node.inputs.initial_moving_transform_com)


def test_iteration_count_is_honoured():
    """Test that the iteration count is honoured.

    --dwiref-construction-iters was ignored; the count was always 2.
    """
    wf = _build('BSplineSyN', num_iterations=5, name='iters_nonlinear')
    node = next(n for n in wf._get_all_nodes() if n.name == 'ants_mvtc2')
    assert node.inputs.iteration_limit == 5


@pytest.mark.parametrize(('dof', 'expected'), [(6, 'Rigid'), (12, 'Affine')])
def test_dwi2anat_dof_reaches_the_template_coregistration(dof, expected):
    """Test that --dwi2anat-dof reaches the template coregistration.

    The config value must drive the ANTs node, not just the mapping constant.

    Asserting ``DWI2ANAT_DOF_TO_TRANSFORM[dof] == expected`` would pass with this
    production consumer still reading the removed attribute, so set the config and
    build the real workflow.
    """
    from qsiprep.utils.misc import DWI2ANAT_DOF_TO_TRANSFORM

    assert DWI2ANAT_DOF_TO_TRANSFORM[dof] == expected

    wf = _build('Affine', name=f'imt_dof{dof}', dwi2anat_dof=dof)
    coreg = wf.get_node('b0_anat_coreg').get_node('b0_to_anat')
    assert coreg.inputs.transforms == [expected]


@pytest.mark.parametrize('transform', ['Rigid', 'BSplineSyN'])
def test_single_input_skips_template_construction(transform):
    """Test that a single distortion group is its own dwiref.

    mvtc2 needs at least two inputs, and a one-image template would only resample
    the b=0, so the reference passes straight through, as fMRIPrep's single-run
    subject boldref does.
    """
    from qsiprep.workflows.dwi.dwiref import init_dwiref_wf

    _config()
    wf = init_dwiref_wf(
        inputs_list=['group_a'],
        t1w_source_file='/data/sub-01_T1w.nii.gz',
        transform=transform,
        name=f'imt_single_{transform}',
    )
    names = _names(wf)
    assert not any('ants_mvtc2' in n or 'dwiref_linear_template' in n for n in names)
    assert not any('template_qc' in n for n in names)

    # The reference is the dwiref, and coregistration still runs on it.
    edge = wf._graph.get_edge_data(wf.get_node('inputnode'), wf.get_node('outputnode'))
    assert ('group_a_b0_template', 'dwiref') in edge['connect']
    assert any('b0_anat_coreg' in n for n in names)


def test_single_input_transform_is_a_binary_identity_affine(tmp_path, monkeypatch):
    """Test that the single-group transform is an identity in ITK's .mat format.

    It is written under the same .mat name as antsRegistration's affines, so it
    must be one, not ITK's text format renamed.
    """
    import numpy as np
    import SimpleITK as sitk

    from qsiprep.workflows.dwi.dwiref import _write_identity_affine

    monkeypatch.chdir(tmp_path)
    out_file = _write_identity_affine()

    assert out_file.endswith('.mat')
    with open(out_file, 'rb') as fobj:
        assert not fobj.read().startswith(b'#Insight Transform File')
    transform = sitk.ReadTransform(out_file)
    np.testing.assert_allclose(transform.GetParameters(), [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
