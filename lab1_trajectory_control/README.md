# Lab 1 — Trajectory Controller Showdown

## Question

How do feedback structure, reference smoothness, and payload uncertainty affect
SO-101 trajectory tracking?

## Controlled experiment

Every controller receives the same initial state, path, duration, control rate,
joint limits, and simulated payload. We compare:

1. P
2. PD
3. PID with anti-windup
4. Velocity feedforward + PID

The first reference will be a smooth joint-space trajectory. The second will be
the Cartesian Lemniscate used in the ETH exercise.

## Measurements

- RMS and maximum joint tracking error
- RMS end-effector tracking error
- overshoot and settling time
- peak velocity and acceleration
- fraction of commands at a limit
- command variation as a smoothness proxy

## Two-day boundary

**Day 1:** implement and verify the four controller equations against one
simulated joint trajectory.

**Day 2:** run the full arm on the Lemniscate, add one payload condition, and
generate the comparison plot and conclusions.

Hardware deployment is a separate follow-up and is not required for this lab's
first complete result.

## Run the first experiment

```bash
uv sync --extra dev
uv run python -m so101_controls.lab1 --trajectory out-and-back
uv run python -m so101_controls.lab1 --trajectory periodic
```

The first experiment deliberately uses a reduced-order joint model with finite
bandwidth, 30 ms command delay, velocity saturation, and a constant load bias.
This isolates feedback behavior before we add full-arm kinematics and contact.

## Preliminary result

| Controller | RMS error | Maximum error | Final error |
|---|---:|---:|---:|
| P | 78.0 mrad | 161.4 mrad | 12.8 mrad |
| PD | 64.1 mrad | 132.5 mrad | 10.7 mrad |
| PID | 73.8 mrad | 137.9 mrad | 33.5 mrad |
| Feedforward + PID | **15.4 mrad** | **31.1 mrad** | **-0.2 mrad** |

The result is not a claim that feedforward is universally best or that these
gains are optimal. It shows that a known desired velocity can provide most of
the motion command, leaving feedback to correct delay, load, and model error.
The untuned integral gain also produces visible overshoot and slow unwinding,
which will be the first tuning exercise.
