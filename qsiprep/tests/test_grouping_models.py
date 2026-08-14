"""Unit tests for the qsiprep.grouping value objects."""

import pytest

from qsiprep.grouping.models import DistortionSignature, derive_output_name


class TestDistortionSignature:
    def test_pe_axis_and_polarity(self):
        sig = DistortionSignature(pe_dir='j-')
        assert sig.pe_axis == 'j'
        assert sig.pe_polarity == -1
        assert DistortionSignature(pe_dir='i').pe_polarity == 1
        assert DistortionSignature().pe_axis is None
        assert DistortionSignature().pe_polarity is None

    def test_compatible_shim(self):
        shimmed = DistortionSignature(shim=(1.0,))
        other = DistortionSignature(shim=(2.0,))
        assert not shimmed.compatible_shim(other)
        assert shimmed.compatible_shim(other, ignore_shims=True)
        assert shimmed.compatible_shim(DistortionSignature())
        assert shimmed.compatible_shim(shimmed)

    def test_key_excludes_informational_fields(self):
        sig_a = DistortionSignature(pe_dir='j', readout_time=0.05, parallel_factor=2)
        sig_b = DistortionSignature(pe_dir='j', readout_time=0.05, parallel_factor=3)
        assert sig_a.key == sig_b.key
        assert sig_a == sig_b  # compare=False on informational fields


class TestDeriveOutputName:
    def test_single_file(self):
        assert (
            derive_output_name(['/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz'])
            == 'sub-1_dir-AP_run-1'
        )

    def test_common_entities(self):
        assert (
            derive_output_name(
                [
                    '/data/sub-1/dwi/sub-1_dir-AP_run-1_dwi.nii.gz',
                    '/data/sub-1/dwi/sub-1_dir-AP_run-2_dwi.nii.gz',
                ]
            )
            == 'sub-1_dir-AP'
        )

    def test_only_subject_in_common(self):
        assert (
            derive_output_name(
                [
                    '/data/sub-1/dwi/sub-1_dir-AP_dwi.nii.gz',
                    '/data/sub-1/dwi/sub-1_dir-PA_dwi.nii.gz',
                ]
            )
            == 'sub-1'
        )

    def test_session_preserved(self):
        assert (
            derive_output_name(
                [
                    '/d/sub-1/ses-2/dwi/sub-1_ses-2_dir-AP_dwi.nii.gz',
                    '/d/sub-1/ses-2/dwi/sub-1_ses-2_dir-PA_dwi.nii.gz',
                ]
            )
            == 'sub-1_ses-2'
        )

    @pytest.mark.parametrize('extension', ['.nii', '.nii.gz'])
    def test_extensions(self, extension):
        assert derive_output_name([f'/d/sub-1/dwi/sub-1_dir-AP_dwi{extension}']) == 'sub-1_dir-AP'


class TestEvaluateShells:
    """The shelled/non-shelled classifier (resurrected from the retired
    _side_is_shelled detector in workflows/dwi/diffprep.py)."""

    def test_multi_shell(self):
        from qsiprep.grouping.metadata import evaluate_shells

        shelled, shells = evaluate_shells([0] + [1000] * 6 + [2000] * 6)
        assert shelled is True
        assert shells == (1000, 2000)

    def test_single_high_b_shell(self):
        """A lone b=2000 shell is shelled for eddy, unlike the old DRBUDDI
        detector, which required a tensor-fittable low-b shell."""
        from qsiprep.grouping.metadata import evaluate_shells

        shelled, shells = evaluate_shells([0] + [2000] * 12)
        assert shelled is True
        assert shells == (2000,)

    def test_grid(self):
        from qsiprep.grouping.metadata import evaluate_shells

        shelled, _ = evaluate_shells([0] + list(range(200, 3000, 150)))
        assert shelled is False

    def test_hasc55_regression(self):
        """A real CS-DSI HASC55 scheme has a dense low-b cluster that a bare
        population count would mis-read as shelled; the grid guard must not."""
        from qsiprep.grouping.metadata import evaluate_shells

        hasc55 = (
            '5 5 3395 3400 2595 4395 3795 2795 1995 4190 3600 3395 2795 1595 5 3790 '
            '4390 800 3400 3990 1195 3590 2195 4190 4000 2790 5000 5 1795 1795 4195 '
            '3395 1195 2795 595 3590 3395 1990 2795 4195 5 3390 3600 4395 4985 4195 '
            '3390 3990 3400 2590 3590 995 2790 5000 2395 2000 1795 2190 1195 1195 '
            '2595 3790 5'
        )
        shelled, _ = evaluate_shells([float(v) for v in hasc55.split()])
        assert shelled is False

    def test_b0_only_is_undetermined(self):
        from qsiprep.grouping.metadata import evaluate_shells

        shelled, shells = evaluate_shells([0, 0, 5])
        assert shelled is None
        assert shells == ()


class TestGridInfo:
    """Grid comparison classification for field-of-view checks."""

    @staticmethod
    def _grid(shape=(4, 4, 4), shift=(0, 0, 0), rot_x_deg=0.0, zooms=(2.0, 2.0, 2.0)):
        import math

        import numpy as np

        from qsiprep.grouping.models import GridInfo

        affine = np.diag([*zooms, 1.0])
        affine[:3, 3] = np.array([-40.0, -40.0, -40.0]) + np.asarray(shift, dtype=float)
        if rot_x_deg:
            theta = math.radians(rot_x_deg)
            rot_x = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, math.cos(theta), -math.sin(theta)],
                    [0.0, math.sin(theta), math.cos(theta)],
                ]
            )
            affine[:3, :3] = rot_x @ affine[:3, :3]
        return GridInfo(
            shape=tuple(shape),
            zooms=tuple(zooms),
            affine=tuple(tuple(float(v) for v in row) for row in affine),
        )

    def test_match(self):
        assert self._grid().compare(self._grid()) == 'match'
        # sub-tolerance jitter still matches
        assert self._grid().compare(self._grid(shift=(0.01, 0, 0))) == 'match'

    def test_shifted(self):
        grid_a, grid_b = self._grid(), self._grid(shift=(10.0, 0.0, 5.0))
        assert grid_a.compare(grid_b) == 'shifted'
        assert abs(grid_a.shift_mm(grid_b) - 11.18) < 0.01

    def test_oblique(self):
        grid_a, grid_b = self._grid(), self._grid(rot_x_deg=5.0)
        assert grid_a.compare(grid_b) == 'oblique'
        assert abs(grid_a.rotation_deg(grid_b) - 5.0) < 0.1

    def test_grid_mismatch(self):
        assert self._grid().compare(self._grid(shape=(4, 4, 6))) == 'grid'
        assert self._grid().compare(self._grid(zooms=(1.5, 1.5, 1.5))) == 'grid'

    def test_oblique_outranks_shift(self):
        """A rotated AND shifted grid classifies as oblique (the worse problem)."""
        grid_a = self._grid()
        grid_b = self._grid(shift=(10.0, 0.0, 0.0), rot_x_deg=5.0)
        assert grid_a.compare(grid_b) == 'oblique'


class TestShimEvidence:
    """The fov-shifted warning cites ShimSetting evidence when available."""

    @staticmethod
    def _run(shim_a, shim_b):
        from qsiprep.grouping.models import (
            ConcatenationGroup,
            DistortionSignature,
            FileRecord,
            Provenance,
        )
        from qsiprep.grouping.validation import check_data_compatibility

        def record(path, shim, shift):
            return FileRecord(
                path=path,
                datatype='dwi',
                suffix='dwi',
                session=None,
                signature=DistortionSignature(pe_dir='j-', readout_time=0.05, shim=shim),
                grid=TestGridInfo._grid(shift=shift),
            )

        records = {
            '/d/a_dwi.nii.gz': record('/d/a_dwi.nii.gz', shim_a, (0, 0, 0)),
            '/d/b_dwi.nii.gz': record('/d/b_dwi.nii.gz', shim_b, (10.0, 0, 0)),
        }
        concat = ConcatenationGroup(
            multipart_id='auto+concat+0',
            provenance=Provenance.INFERRED,
            distortion_groups=('a', 'b'),
            dwi_files=tuple(records),
            output_name='sub-01',
        )
        issues = check_data_compatibility(records, {'auto+concat+0': concat})
        (issue,) = [i for i in issues if i.code == 'fov-shifted']
        return issue.message

    def test_matching_shims_soften(self):
        message = self._run((1.0, 2.0), (1.0, 2.0))
        assert 'ShimSetting values match' in message

    def test_differing_shims_confirm(self):
        message = self._run((1.0, 2.0), (9.0, 9.0))
        assert 'confirming a re-shim' in message

    def test_absent_shims_unverifiable(self):
        message = self._run(None, None)
        assert 'cannot be verified' in message
