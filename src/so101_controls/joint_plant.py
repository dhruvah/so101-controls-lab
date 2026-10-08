"""Reduced-order model of a position-controlled smart-servo joint."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class JointPlant:
    """Velocity-command joint with bandwidth, delay, saturation, and load bias."""

    dt: float = 0.005
    velocity_time_constant: float = 0.08
    command_delay: float = 0.03
    velocity_limit: float = 1.8
    load_acceleration: float = -0.8
    position: float = 0.0
    velocity: float = 0.0
    _commands: deque = field(init=False, repr=False)

    def __post_init__(self):
        delay_steps = max(0, round(self.command_delay / self.dt))
        self._commands = deque([0.0] * (delay_steps + 1), maxlen=delay_steps + 1)

    def step(self, velocity_command):
        limited_command = float(np.clip(velocity_command, -self.velocity_limit, self.velocity_limit))
        self._commands.append(limited_command)
        delayed_command = self._commands[0]
        acceleration = (
            (delayed_command - self.velocity) / self.velocity_time_constant
            + self.load_acceleration
        )
        self.velocity += acceleration * self.dt
        self.position += self.velocity * self.dt
        return self.position, self.velocity, limited_command
