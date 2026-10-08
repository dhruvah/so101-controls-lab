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
uv run python -m so101_controls.lab1_mujoco
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

## Complete MuJoCo arm

`lab1_mujoco` runs the same controller comparison on Google DeepMind's
MuJoCo Menagerie SO-101. The outer controller produces a velocity command,
which is integrated into the position target accepted by the model's internal
joint servo. This nested structure mirrors the interface of a smart-servo arm:

```text
trajectory -> outer P/PD/PID -> position target -> internal servo -> mechanism
```

The shoulder-lift joint repeatedly moves from -1.2 to -0.7 rad while the full
arm experiences gravity, coupled rigid-body motion, force limits, and joint
friction. The other joints hold a fixed posture.

| Controller | RMS error | Maximum error | Inner force saturation |
|---|---:|---:|---:|
| P | 47.8 mrad | 67.6 mrad | 16.5% |
| PD | 31.9 mrad | 46.2 mrad | 16.8% |
| PID | 38.4 mrad | 62.6 mrad | 31.0% |
| Feedforward + PID | **14.6 mrad** | **21.7 mrad** | 23.8% |

Feedforward + PID tracks best, but it is not free: it commands earlier and
uses the shoulder's available force aggressively. The untuned integral term
both tracks worse than PD and spends the most time at the force limit. This is
why controller comparisons must include actuator effort and saturation rather
than tracking error alone.
