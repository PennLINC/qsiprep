"""The dwiref space must be reachable from BIDS and from ACPC.

Both transforms were computed and fed straight into ComposeTransforms for
resampling, then discarded. That made the dwiref space a dead end: a user
could not map a session's native b=0 into it, or the template back to ACPC --
the two hops that make a cross-session template interpretable at all.

It also meant the space of the written template could not be checked
independently, which is how a template got written in the wrong space while
carrying a correct-looking space-ACPC name.
"""

import json

from bids.layout.writing import build_path

from qsiprep.data import load as load_data


def _patterns():
    return json.loads(load_data('io_spec.json').read_text())['default_path_patterns']


# The rendered paths for both hops are asserted in test_dwiref_derivatives.py,
# across both --subject-anatomical-reference modes. They used to be asserted here
# against the anat/ datatype and the from-intramodal spelling, which this branch
# replaced; keeping a second copy only invited the two to drift apart.


def test_existing_transform_paths_unchanged():
    """The anatomical round trip must keep working."""
    pats = _patterns()
    for ents, expected in (
        (
            dict(
                subject='01',
                session='3',
                datatype='anat',
                suffix='xfm',
                mode='image',
                extension='.mat',
                **{'from': 'orig', 'to': 'anat'},
            ),
            'sub-01/ses-3/anat/sub-01_ses-3_from-orig_to-anat_mode-image_xfm.mat',
        ),
        (
            dict(
                subject='01',
                datatype='anat',
                suffix='xfm',
                mode='image',
                extension='.h5',
                **{'from': 'ACPC', 'to': 'MNI152NLin2009cAsym'},
            ),
            'sub-01/anat/sub-01_from-ACPC_to-MNI152NLin2009cAsym_mode-image_xfm.h5',
        ),
    ):
        assert build_path(ents, pats, strict=False) == expected


def test_both_sinks_are_wired_in_base():
    """Guard against the transforms silently going unwritten again."""
    import inspect

    from qsiprep.workflows import base

    src = inspect.getsource(base)
    assert 'ds_dwiref_to_acpc' in src
    assert 'ds_distortiongroup_to_dwiref' in src
    assert "'outputnode.dwiref_to_t1_affine', 'in_file'" in src


def test_single_group_subject_skips_the_template_instead_of_failing():
    """A one-session subject must not fail the run.

    Cohorts routinely mix single- and multi-session subjects: in CRASH, 24 of 59
    subjects have one session. Raising here meant a single
    --dwiref-construction-iters flag failed 41% of the dataset outright.
    """
    import inspect

    from qsiprep.workflows import base

    src = inspect.getsource(base.init_single_subject_wf)
    assert "raise Exception('Cannot make an intramodal with less than 2 groups.')" not in src
    assert 'Falling back to --dwiref-definition distortion-group' in src
    # and the flag must still be honoured when there ARE enough groups
    assert 'make_dwiref = True' in src
