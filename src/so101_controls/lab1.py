"""Run the first trajectory-controller comparison experiment."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from so101_controls.controller import PIDController
from so101_controls.joint_plant import JointPlant
from so101_controls.trajectory import quintic_step


CONTROLLERS = {
    "P": dict(kp=5.0),
    "PD": dict(kp=6.0, kd=0.18),
    "PID": dict(kp=5.0, ki=3.0, kd=0.18),
    "FF+PID": dict(kp=5.0, ki=2.0, kd=0.18, velocity_feedforward=1.0),
}


def out_and_back_reference(time):
    if time < 2.0:
        return quintic_step(time, 2.0, 0.0, 0.8)
    if time < 3.0:
        return 0.8, 0.0
    if time < 5.0:
        return quintic_step(time - 3.0, 2.0, 0.8, 0.0)
    return 0.0, 0.0


def periodic_reference(time, amplitude=0.6, frequency=0.4):
    """Smooth periodic motion starting from rest at zero position."""
    omega = 2.0 * np.pi * frequency
    position = 0.5 * amplitude * (1.0 - np.cos(omega * time))
    velocity = 0.5 * amplitude * omega * np.sin(omega * time)
    return position, velocity


def run_controller(name, gains, reference_function=out_and_back_reference, duration=6.0, dt=0.005):
    plant = JointPlant(dt=dt)
    controller = PIDController(**gains, output_limit=plant.velocity_limit)
    records = []
    previous_command = 0.0
    for time in np.arange(0.0, duration, dt):
        desired_position, desired_velocity = reference_function(time)
        error = desired_position - plant.position
        error_rate = desired_velocity - plant.velocity
        raw_command = controller.update(error, error_rate, desired_velocity, dt)
        position, velocity, command = plant.step(raw_command)
        records.append(
            (time, desired_position, desired_velocity, position, velocity, command, command - previous_command)
        )
        previous_command = command

    values = np.asarray(records)
    error = values[:, 1] - values[:, 3]
    metrics = {
        "controller": name,
        "rms_error_rad": float(np.sqrt(np.mean(error**2))),
        "max_error_rad": float(np.max(np.abs(error))),
        "final_error_rad": float(error[-1]),
        "command_smoothness_rms": float(np.sqrt(np.mean(values[:, 6] ** 2))),
        "saturation_fraction": float(np.mean(np.abs(values[:, 5]) >= plant.velocity_limit - 1e-9)),
    }
    return values, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", choices=("out-and-back", "periodic"), default="out-and-back")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reference_function = (
        out_and_back_reference if args.trajectory == "out-and-back" else periodic_reference
    )
    duration = 6.0 if args.trajectory == "out-and-back" else 7.5
    if args.output is None:
        args.output = Path("results/lab1") / args.trajectory
    args.output.mkdir(parents=True, exist_ok=True)

    all_results = {}
    metrics = []
    for name, gains in CONTROLLERS.items():
        values, summary = run_controller(
            name, gains, reference_function=reference_function, duration=duration
        )
        all_results[name] = values
        metrics.append(summary)
        filename = name.lower().replace("+", "_plus_") + ".csv"
        with (args.output / filename).open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ("time_s", "desired_position_rad", "desired_velocity_rad_s", "position_rad",
                 "velocity_rad_s", "velocity_command_rad_s", "command_change_rad_s")
            )
            writer.writerows(values)

    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
    first = next(iter(all_results.values()))
    axes[0].plot(first[:, 0], first[:, 1], "k--", linewidth=2, label="reference")
    for name, values in all_results.items():
        axes[0].plot(values[:, 0], values[:, 3], label=name)
        axes[1].plot(values[:, 0], 1000.0 * (values[:, 1] - values[:, 3]), label=name)
    axes[0].set_ylabel("joint position [rad]")
    axes[1].set(xlabel="time [s]", ylabel="tracking error [mrad]")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle(f"SO-101 joint controller comparison: {args.trajectory}")
    figure.tight_layout()
    figure.savefig(args.output / "controller_comparison.png", dpi=180)
    plt.close(figure)

    print(json.dumps(metrics, indent=2))
    print(f"Results written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
