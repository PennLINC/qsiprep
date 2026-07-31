"""Tests for qsiprep.utils.misc."""

import logging

import numpy as np

from qsiprep.utils.misc import safe_unit_vector


def test_safe_unit_vector_zero_magnitude_substitutes_x_axis():
    result = safe_unit_vector(np.array([0.0, 0.0, 0.0]))
    assert np.array_equal(result, np.array([1.0, 0.0, 0.0]))


def test_safe_unit_vector_normalizes_nonzero_vector():
    result = safe_unit_vector(np.array([0.0, 3.0, 0.0]))
    assert np.allclose(result, np.array([0.0, 1.0, 0.0]))
    assert np.isclose(np.linalg.norm(result), 1.0)


def test_safe_unit_vector_no_nan_on_zero():
    result = safe_unit_vector(np.array([0.0, 0.0, 0.0]))
    assert not np.any(np.isnan(result))


def test_safe_unit_vector_warns_on_zero_magnitude(caplog):
    with caplog.at_level(logging.WARNING, logger='nipype.interface'):
        safe_unit_vector(np.array([0.0, 0.0, 0.0]))
    assert any('zero-magnitude' in record.message for record in caplog.records)


def test_average_bvec_no_nan_with_zero_magnitude_pair():
    from qsiprep.interfaces.dwi_merge import average_bvec

    # Antipodal vectors average to a zero-magnitude vector, which the old
    # normalization turned into NaN. The guard must keep the result finite.
    bvec1 = np.array([1.0, 0.0, 0.0])
    bvec2 = np.array([-1.0, 0.0, 0.0])
    averaged, _ = average_bvec(bvec1, bvec2)
    assert not np.any(np.isnan(averaged))


def test_angle_between_finite_for_zero_vector():
    from qsiprep.interfaces.dwi_merge import angle_between

    angle = angle_between(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert np.isfinite(angle)
