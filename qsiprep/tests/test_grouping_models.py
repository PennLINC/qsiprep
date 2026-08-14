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

    def test_opposes(self):
        ap = DistortionSignature(pe_dir='j-', readout_time=0.05)
        pa = DistortionSignature(pe_dir='j', readout_time=0.05)
        lr = DistortionSignature(pe_dir='i', readout_time=0.05)
        assert ap.opposes(pa)
        assert pa.opposes(ap)
        assert not ap.opposes(lr)
        assert not ap.opposes(ap)
        assert not DistortionSignature().opposes(pa)

    def test_opposes_respects_shims(self):
        ap = DistortionSignature(pe_dir='j-', shim=(1.0, 2.0))
        pa_same = DistortionSignature(pe_dir='j', shim=(1.0, 2.0))
        pa_othershim = DistortionSignature(pe_dir='j', shim=(9.0, 9.0))
        pa_noshim = DistortionSignature(pe_dir='j')
        assert ap.opposes(pa_same)
        assert not ap.opposes(pa_othershim)
        assert ap.opposes(pa_othershim, ignore_shims=True)
        # Unknown shim is a wildcard
        assert ap.opposes(pa_noshim)

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
