import pytest

from so101_controls.trajectory import quintic_step


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
