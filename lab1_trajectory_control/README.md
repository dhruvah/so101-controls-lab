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
