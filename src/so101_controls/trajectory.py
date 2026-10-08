"""Reference trajectories for repeatable controller comparisons."""


def quintic_step(time, duration, start, end):
    """Return position and velocity for a rest-to-rest quintic motion."""
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    phase = min(max(time / duration, 0.0), 1.0)
    blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
    blend_rate = (30.0 * phase**2 - 60.0 * phase**3 + 30.0 * phase**4) / duration
    displacement = end - start
    return start + displacement * blend, displacement * blend_rate
