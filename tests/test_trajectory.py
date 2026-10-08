import pytest

from so101_controls.trajectory import quintic_step
from so101_controls.lab1 import periodic_reference


def test_quintic_has_rest_boundary_conditions():
    start_position, start_velocity = quintic_step(0.0, 2.0, -1.0, 1.0)
    end_position, end_velocity = quintic_step(2.0, 2.0, -1.0, 1.0)
    assert (start_position, start_velocity) == pytest.approx((-1.0, 0.0))
    assert (end_position, end_velocity) == pytest.approx((1.0, 0.0))


def test_quintic_midpoint_is_halfway():
    position, _ = quintic_step(1.0, 2.0, -1.0, 1.0)
    assert position == pytest.approx(0.0)


def test_quintic_rejects_nonpositive_duration():
    with pytest.raises(ValueError):
        quintic_step(0.0, 0.0, 0.0, 1.0)


def test_periodic_reference_starts_from_rest():
    assert periodic_reference(0.0) == pytest.approx((0.0, 0.0))


def test_periodic_reference_reaches_amplitude_halfway_through_cycle():
    position, velocity = periodic_reference(1.25, amplitude=0.6, frequency=0.4)
    assert (position, velocity) == pytest.approx((0.6, 0.0), abs=1e-12)
