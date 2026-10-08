# SO-101 Controls Lab

A compact, experiment-driven study of robot control using the SO-101 in
MuJoCo and on physical hardware.

The repository contains three independent mini-projects:

1. **Trajectory Controller Showdown** — compare P, PD, PID, and
   feedforward + PID on identical trajectories and payloads.
2. **Estimation and Failure Lab** — study filtering, delay, jitter, dropped
   commands, watchdogs, and safe fallback behavior.
3. **Sim-to-Real Safe Policy Stack** — identify effective actuator parameters
   and place a safety supervisor around classical and learned controllers.

Each lab must end with a repeatable experiment, quantitative metrics, one
comparison plot, and a concise engineering conclusion.

## Current milestone

Lab 1 begins with one fixed Lemniscate trajectory in MuJoCo. Success means the
same runner can compare controllers using:

- RMS and maximum tracking error
- overshoot and settling behavior
- command smoothness
- actuator saturation
- completion time

Physical-arm testing follows only after conservative limits and an emergency
stop have been verified in simulation.

## Repository layout

```text
configs/                    Experiment configurations
results/                    Small, reproducible result summaries
src/so101_controls/         Shared control and experiment code
tests/                      Unit tests for controller mathematics
lab1_trajectory_control/    Lab 1 notes and reports
lab2_estimation_failures/   Lab 2 notes and reports
lab3_sim2real_policy/       Lab 3 notes and reports
```

## Safety

Simulation is the default backend. Physical experiments must use joint and
velocity limits, a command timeout, a reachable emergency stop, and an
unobstructed workspace.
