"""Interactive SO-101 controls arcade for the MuJoCo viewer."""

import argparse
from collections import deque
import sys
import time

import glfw
import mujoco
import mujoco.viewer
import mujoco_menagerie
import numpy as np

from so101_controls.controller import PIDController
from so101_controls.lab1_mujoco import START_QPOS, periodic_reference


HELP = """
SO-101 Controls Arcade

  [ / ]   decrease / increase Kp
  ; / '   decrease / increase Kd
  , / .   decrease / increase Ki
  - / =   decrease / increase feedforward
  1 / 2   slower / faster trajectory
  SPACE   disturb the gripper
  R       reset the robot and metrics
  H       print these controls
"""


def add_sphere(scene, position, radius, color):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.full(3, radius),
        np.asarray(position, dtype=np.float64),
        np.eye(3).reshape(-1),
        np.asarray(color, dtype=np.float32),
    )
    scene.ngeom += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=0.0, help="Auto-close after N seconds")
    args = parser.parse_args()

    model = mujoco_menagerie.load("robotstudio_so101", "so101")
    data = mujoco.MjData(model)
    desired_data = mujoco.MjData(model)
    joint_id = model.joint("shoulder_lift").id
    actuator_id = model.actuator("shoulder_lift").id
    qpos_id = model.jnt_qposadr[joint_id]
    dof_id = model.jnt_dofadr[joint_id]
    gripper_body_id = model.body("gripper").id
    force_limit = model.actuator_forcerange[actuator_id, 1]
    control_dt = 0.02
    physics_steps = round(control_dt / model.opt.timestep)

    settings = {"kp": 5.0, "ki": 2.0, "kd": 0.18, "ff": 1.0, "speed": 1.0}
    events = {"reset": False, "disturb": False}

    def key_callback(key):
        increments = {
            glfw.KEY_LEFT_BRACKET: ("kp", -0.5),
            glfw.KEY_RIGHT_BRACKET: ("kp", 0.5),
            glfw.KEY_SEMICOLON: ("kd", -0.02),
            glfw.KEY_APOSTROPHE: ("kd", 0.02),
            glfw.KEY_COMMA: ("ki", -0.25),
            glfw.KEY_PERIOD: ("ki", 0.25),
            glfw.KEY_MINUS: ("ff", -0.1),
            glfw.KEY_EQUAL: ("ff", 0.1),
            glfw.KEY_1: ("speed", -0.1),
            glfw.KEY_2: ("speed", 0.1),
        }
        if key in increments:
            parameter, change = increments[key]
            settings[parameter] = max(0.0, settings[parameter] + change)
        elif key == glfw.KEY_SPACE:
            events["disturb"] = True
        elif key == glfw.KEY_R:
            events["reset"] = True
        elif key == glfw.KEY_H:
            print(HELP)

    def reset():
        mujoco.mj_resetData(model, data)
        data.qpos[:] = START_QPOS
        data.ctrl[:] = START_QPOS
        mujoco.mj_forward(model, data)
        for _ in range(round(0.5 / model.opt.timestep)):
            mujoco.mj_step(model, data)
        return PIDController(
            kp=settings["kp"], ki=settings["ki"], kd=settings["kd"],
            velocity_feedforward=settings["ff"], output_limit=1.0,
        ), float(data.qpos[qpos_id])

    controller, position_command = reset()
    errors = deque(maxlen=250)
    saturations = deque(maxlen=250)
    actual_trace = deque(maxlen=160)
    desired_trace = deque(maxlen=160)
    experiment_time = 0.0
    disturbance_steps = 0
    last_status = -1.0

    print(HELP)
    started = time.monotonic()
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = (0.15, 0.0, 0.20)
        viewer.cam.distance = 0.8
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20

        while viewer.is_running():
            loop_started = time.monotonic()
            if args.duration > 0.0 and loop_started - started >= args.duration:
                break
            if events["reset"]:
                controller, position_command = reset()
                errors.clear(); saturations.clear(); actual_trace.clear(); desired_trace.clear()
                experiment_time = 0.0
                events["reset"] = False

            controller.kp = settings["kp"]
            controller.ki = settings["ki"]
            controller.kd = settings["kd"]
            controller.velocity_feedforward = settings["ff"]
            desired_position, desired_velocity = periodic_reference(
                experiment_time * settings["speed"]
            )
            desired_velocity *= settings["speed"]
            error = desired_position - data.qpos[qpos_id]
            error_rate = desired_velocity - data.qvel[dof_id]
            velocity_command = controller.update(error, error_rate, desired_velocity, control_dt)
            position_command = float(np.clip(
                position_command + velocity_command * control_dt,
                *model.jnt_range[joint_id],
            ))
            data.ctrl[actuator_id] = position_command

            if events["disturb"]:
                disturbance_steps = max(1, round(0.15 / model.opt.timestep))
                events["disturb"] = False
            for _ in range(physics_steps):
                data.xfrc_applied[gripper_body_id, :3] = (0.0, 8.0, 0.0) if disturbance_steps else 0.0
                mujoco.mj_step(model, data)
                disturbance_steps = max(0, disturbance_steps - 1)
            data.xfrc_applied[gripper_body_id] = 0.0

            errors.append(error)
            saturations.append(abs(data.actuator_force[actuator_id]) >= force_limit - 1e-6)
            actual_trace.append(data.site("gripperframe").xpos.copy())
            desired_data.qpos[:] = data.qpos
            desired_data.qpos[qpos_id] = desired_position
            mujoco.mj_forward(model, desired_data)
            desired_trace.append(desired_data.site("gripperframe").xpos.copy())

            viewer.user_scn.ngeom = 0
            for point in desired_trace:
                add_sphere(viewer.user_scn, point, 0.0025, (0.1, 1.0, 0.2, 0.55))
            for point in actual_trace:
                add_sphere(viewer.user_scn, point, 0.0025, (0.1, 0.45, 1.0, 0.8))
            viewer.sync()

            if experiment_time - last_status >= 0.5:
                rms_mrad = 1000.0 * np.sqrt(np.mean(np.square(errors))) if errors else 0.0
                saturation = 100.0 * np.mean(saturations) if saturations else 0.0
                sys.stdout.write(
                    f"\rKp={settings['kp']:4.1f} Ki={settings['ki']:4.2f} "
                    f"Kd={settings['kd']:4.2f} FF={settings['ff']:3.1f} "
                    f"speed={settings['speed']:3.1f}x | RMS={rms_mrad:5.1f} mrad "
                    f"force saturation={saturation:4.1f}%   "
                )
                sys.stdout.flush()
                last_status = experiment_time

            experiment_time += control_dt
            remaining = control_dt - (time.monotonic() - loop_started)
            if remaining > 0.0:
                time.sleep(remaining)
    print("\nControls arcade closed.")


if __name__ == "__main__":
    main()
