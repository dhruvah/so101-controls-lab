"""Small feedback controller used by the Lab 1 comparisons."""

from dataclasses import dataclass

import numpy as np


@dataclass
class PIDController:
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    velocity_feedforward: float = 0.0
    output_limit: float | None = None

    def __post_init__(self):
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def update(self, error, error_rate, desired_velocity, dt):
        previous_integral = self.integral
        self.integral += error * dt
        output = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * error_rate
            + self.velocity_feedforward * desired_velocity
        )
        if self.output_limit is None:
            return output

        limited = float(np.clip(output, -self.output_limit, self.output_limit))
        pushing_further_into_saturation = output != limited and np.sign(error) == np.sign(output)
        if pushing_further_into_saturation:
            self.integral = previous_integral
        return limited
