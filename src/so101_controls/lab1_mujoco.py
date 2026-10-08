"""Compare outer-loop controllers on the complete MuJoCo SO-101 model."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco_menagerie
import numpy as np

from so101_controls.controller import PIDController
from so101_controls.lab1 import CONTROLLERS


START_QPOS = np.array([0.0, -1.2, 1.0, 0.7, 0.0, 0.2])


def periodic_reference(time, center=-1.2, amplitude=0.5, frequency=0.25):
    omega = 2.0 * np.pi * frequency
    position = center + 0.5 * amplitude * (1.0 - np.cos(omega * time))
    velocity = 0.5 * amplitude * omega * np.sin(omega * time)
    return position, velocity


def run_controller(name, gains, duration=8.0, control_dt=0.02):
    model = mujoco_menagerie.load("robotstudio_so101", "so101")
    data = mujoco.MjData(model)
    data.qpos[:] = START_QPOS
    data.ctrl[:] = START_QPOS
    mujoco.mj_forward(model, data)

    # Let gravity and the internal position servos reach the same initial state.
    for _ in range(round(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    joint_id = model.joint("shoulder_lift").id
    actuator_id = model.actuator("shoulder_lift").id
    qpos_id = model.jnt_qposadr[joint_id]
    dof_id = model.jnt_dofadr[joint_id]
    position_command = float(data.qpos[qpos_id])
    velocity_limit = 1.0
    force_limit = float(model.actuator_forcerange[actuator_id, 1])
    controller = PIDController(**gains, output_limit=velocity_limit)
    physics_steps = round(control_dt / model.opt.timestep)
    if not np.isclose(physics_steps * model.opt.timestep, control_dt):
        raise ValueError("control_dt must be an integer multiple of the MuJoCo timestep")

    records = []
    previous_velocity_command = 0.0
    for step in range(round(duration / control_dt)):
        time = step * control_dt
        desired_position, desired_velocity = periodic_reference(time)
        error = desired_position - data.qpos[qpos_id]
        error_rate = desired_velocity - data.qvel[dof_id]
        velocity_command = controller.update(error, error_rate, desired_velocity, control_dt)
        position_command += velocity_command * control_dt
        position_command = float(np.clip(position_command, *model.jnt_range[joint_id]))
        data.ctrl[actuator_id] = position_command
        for _ in range(physics_steps):
            mujoco.mj_step(model, data)

        records.append(
            (
                time,
                desired_position,
                desired_velocity,
                data.qpos[qpos_id],
                data.qvel[dof_id],
                velocity_command,
                position_command,
                data.actuator_force[actuator_id],
                velocity_command - previous_velocity_command,
            )
        )
        previous_velocity_command = velocity_command

    values = np.asarray(records)
    error = values[:, 1] - values[:, 3]
    metrics = {
        "controller": name,
        "rms_error_rad": float(np.sqrt(np.mean(error**2))),
        "max_error_rad": float(np.max(np.abs(error))),
        "final_error_rad": float(error[-1]),
        "command_smoothness_rms": float(np.sqrt(np.mean(values[:, 8] ** 2))),
        "outer_velocity_saturation_fraction": float(
            np.mean(np.abs(values[:, 5]) >= velocity_limit - 1e-9)
        ),
        "inner_force_saturation_fraction": float(
            np.mean(np.abs(values[:, 7]) >= force_limit - 1e-6)
        ),
    }
    return values, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/lab1/mujoco_periodic"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    results = {}
    summaries = []
    header = (
        "time_s", "desired_position_rad", "desired_velocity_rad_s", "position_rad",
        "velocity_rad_s", "outer_velocity_command_rad_s", "inner_position_target_rad",
        "actuator_force_nm", "command_change_rad_s",
    )
    for name, gains in CONTROLLERS.items():
        values, metrics = run_controller(name, gains)
        results[name] = values
        summaries.append(metrics)
        filename = name.lower().replace("+", "_plus_") + ".csv"
        with (args.output / filename).open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(values)

    (args.output / "metrics.json").write_text(json.dumps(summaries, indent=2) + "\n")
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 9))
    first = next(iter(results.values()))
    axes[0].plot(first[:, 0], first[:, 1], "k--", linewidth=2, label="reference")
    for name, values in results.items():
        axes[0].plot(values[:, 0], values[:, 3], label=name)
        axes[1].plot(values[:, 0], 1000.0 * (values[:, 1] - values[:, 3]), label=name)
        axes[2].plot(values[:, 0], values[:, 7], label=name)
    axes[0].set_ylabel("shoulder position [rad]")
    axes[1].set_ylabel("tracking error [mrad]")
    axes[2].set(xlabel="time [s]", ylabel="actuator force [N m]")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(ncol=2)
    figure.suptitle("Complete MuJoCo SO-101: shoulder trajectory tracking")
    figure.tight_layout()
    figure.savefig(args.output / "controller_comparison.png", dpi=180)
    plt.close(figure)

    print(json.dumps(summaries, indent=2))
    print(f"Results written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
