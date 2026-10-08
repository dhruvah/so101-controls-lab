import pytest

from so101_controls.joint_plant import JointPlant


def test_velocity_command_is_limited():
    plant = JointPlant(dt=0.01, command_delay=0.0, velocity_limit=1.0, load_acceleration=0.0)
    _, _, command = plant.step(10.0)
    assert command == pytest.approx(1.0)


def test_load_bias_moves_uncontrolled_joint():
    plant = JointPlant(dt=0.01, command_delay=0.0, load_acceleration=-1.0)
    position, velocity, _ = plant.step(0.0)
    assert position < 0.0
    assert velocity < 0.0
