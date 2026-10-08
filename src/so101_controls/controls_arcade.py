"""Interactive SO-101 controls arcade for the MuJoCo viewer."""

import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import webbrowser

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


CONTROL_PANEL = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>SO-101 Controls Arcade</title>
<style>
body{font:15px system-ui;margin:0;background:#10141c;color:#eef3fb}main{max-width:780px;margin:auto;padding:24px}
h1{margin:0 0 4px}.sub{color:#9daabe;margin-bottom:22px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.panel{background:#19212d;border:1px solid #303c4c;border-radius:12px;padding:18px}.control{margin:14px 0}
label{display:flex;justify-content:space-between;font-weight:600}input,select{width:100%;margin-top:8px}select{padding:8px;background:#10141c;color:#eef3fb;border:1px solid #435167;border-radius:6px}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.metric{background:#10141c;padding:14px;border-radius:8px}.value{font-size:24px;font-weight:700;display:block}.unit{color:#9daabe;font-size:12px}
button{padding:11px 16px;border:0;border-radius:7px;font-weight:700;cursor:pointer;margin:8px 8px 0 0}.disturb{background:#f3a83b}.reset{background:#6da7ff}
@media(max-width:620px){.grid{grid-template-columns:1fr}.metrics{grid-template-columns:1fr}}
</style></head><body><main><h1>SO-101 Controls Arcade</h1><div class="sub">Tune the outer loop while MuJoCo runs.</div>
<div class="grid"><section class="panel"><div class="control"><label>Trajectory</label><select id="trajectory"><option value="periodic">Periodic</option><option value="step">Step sequence</option><option value="chirp">Frequency sweep</option></select></div>
<div id="sliders"></div><button class="disturb" id="disturb">Apply disturbance</button><button class="reset" id="reset">Reset experiment</button></section>
<section class="panel"><div class="metrics"><div class="metric"><span class="value" id="rms">0.0</span><span class="unit">RMS error · mrad</span></div><div class="metric"><span class="value" id="sat">0.0</span><span class="unit">force saturation · %</span></div><div class="metric"><span class="value" id="err">0.0</span><span class="unit">current error · mrad</span></div></div>
<p class="sub" style="margin-top:18px">Green trail: desired gripper path<br>Blue trail: actual gripper path</p></section></div></main>
<script>
const specs=[['kp','Kp',0,15,.1],['ki','Ki',0,8,.05],['kd','Kd',0,1,.01],['ff','Feedforward',0,1.5,.05],['speed','Trajectory speed',.2,3,.1]];
const sliders=document.getElementById('sliders');
for(const [id,label,min,max,step] of specs){sliders.insertAdjacentHTML('beforeend',`<div class="control"><label>${label}<span id="${id}Value"></span></label><input id="${id}" type="range" min="${min}" max="${max}" step="${step}"></div>`)}
async function sendSettings(){const body={trajectory:document.getElementById('trajectory').value};for(const [id] of specs)body[id]=Number(document.getElementById(id).value);await fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})}
for(const [id] of specs){document.getElementById(id).addEventListener('input',e=>{document.getElementById(id+'Value').textContent=e.target.value;sendSettings()})}
document.getElementById('trajectory').addEventListener('change',sendSettings);
for(const action of ['disturb','reset'])document.getElementById(action).onclick=()=>fetch('/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})});
async function refresh(){try{const s=await(await fetch('/state')).json();for(const [id] of specs){const el=document.getElementById(id);if(document.activeElement!==el)el.value=s.settings[id];document.getElementById(id+'Value').textContent=Number(s.settings[id]).toFixed(id==='kp'||id==='speed'?1:2)}document.getElementById('trajectory').value=s.settings.trajectory;document.getElementById('rms').textContent=s.metrics.rms_mrad.toFixed(1);document.getElementById('sat').textContent=s.metrics.saturation_percent.toFixed(1);document.getElementById('err').textContent=s.metrics.current_error_mrad.toFixed(1)}catch(e){}setTimeout(refresh,200)}refresh();
</script></body></html>"""


def start_control_server(settings, events, metrics, lock, port):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, body, content_type="application/json"):
            payload = body.encode()
            self.send_response(code); self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", len(payload)); self.end_headers(); self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/":
                self.reply(200, CONTROL_PANEL, "text/html; charset=utf-8")
            elif self.path == "/state":
                with lock:
                    self.reply(200, json.dumps({"settings": settings, "metrics": metrics}))
            else:
                self.reply(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                with lock:
                    if self.path == "/settings":
                        for key in ("kp", "ki", "kd", "ff", "speed"):
                            if key in body:
                                settings[key] = max(0.0, float(body[key]))
                        if body.get("trajectory") in ("periodic", "step", "chirp"):
                            settings["trajectory"] = body["trajectory"]
                    elif self.path == "/action" and body.get("action") in events:
                        events[body["action"]] = True
                self.reply(200, json.dumps({"ok": True}))
            except (ValueError, json.JSONDecodeError):
                self.reply(400, json.dumps({"error": "invalid request"}))

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def selected_reference(name, time_value, speed):
    scaled_time = time_value * speed
    if name == "periodic":
        position, velocity = periodic_reference(scaled_time)
        return position, velocity * speed
    if name == "step":
        return (-1.2 if int(scaled_time / 2.0) % 2 == 0 else -0.7), 0.0
    # A chirp raises frequency from 0.08 Hz to 0.8 Hz over 20 seconds.
    frequency_start, frequency_rate = 0.08, 0.036
    chirp_time = min(scaled_time, 20.0)
    phase = 2.0 * np.pi * (frequency_start * chirp_time + 0.5 * frequency_rate * chirp_time**2)
    phase_rate = 2.0 * np.pi * (frequency_start + frequency_rate * chirp_time) * speed
    return -1.2 + 0.25 * (1.0 - np.cos(phase)), 0.25 * np.sin(phase) * phase_rate


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
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
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

    settings = {
        "kp": 5.0, "ki": 2.0, "kd": 0.18, "ff": 1.0,
        "speed": 1.0, "trajectory": "periodic",
    }
    events = {"reset": False, "disturb": False}
    metrics = {"rms_mrad": 0.0, "saturation_percent": 0.0, "current_error_mrad": 0.0}
    state_lock = threading.Lock()
    server = start_control_server(settings, events, metrics, state_lock, args.port)
    panel_url = f"http://127.0.0.1:{args.port}"
    print(f"Control panel: {panel_url}")
    if not args.no_browser:
        webbrowser.open(panel_url)

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
            desired_position, desired_velocity = selected_reference(
                settings["trajectory"], experiment_time, settings["speed"]
            )
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
            with state_lock:
                metrics["rms_mrad"] = 1000.0 * float(np.sqrt(np.mean(np.square(errors))))
                metrics["saturation_percent"] = 100.0 * float(np.mean(saturations))
                metrics["current_error_mrad"] = 1000.0 * float(error)
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

            experiment_time += control_dt
            remaining = control_dt - (time.monotonic() - loop_started)
            if remaining > 0.0:
                time.sleep(remaining)
    print("Controls arcade closed.")
    server.shutdown()


if __name__ == "__main__":
    main()
