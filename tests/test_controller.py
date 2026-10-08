import pytest

from so101_controls.controller import PIDController


def test_proportional_response():
    controller = PIDController(kp=2.0)
    assert controller.update(0.5, 0.0, 0.0, 0.01) == pytest.approx(1.0)


def test_velocity_feedforward_is_added():
    controller = PIDController(kp=1.0, velocity_feedforward=0.5)
    assert controller.update(0.2, 0.0, 2.0, 0.01) == pytest.approx(1.2)


def test_anti_windup_stops_integrator_during_saturation():
    controller = PIDController(kp=10.0, ki=2.0, output_limit=1.0)
    for _ in range(100):
        assert controller.update(1.0, 0.0, 0.0, 0.01) == pytest.approx(1.0)
    assert controller.integral == pytest.approx(0.0)


def test_reset_clears_integrator():
    controller = PIDController(kp=0.0, ki=1.0)
    controller.update(1.0, 0.0, 0.0, 0.1)
    controller.reset()
    assert controller.integral == pytest.approx(0.0)
