"""Interactive Cartesian painting challenge for the MuJoCo SO-101."""

import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import webbrowser

import mujoco
import mujoco.viewer
import mujoco_menagerie
import numpy as np

from so101_controls.controller import PIDController
from so101_controls.controls_arcade import add_sphere


PANEL = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>SO-101 Painting Lab</title>
<style>body{font:15px system-ui;margin:0;background:#10141c;color:#eef3fb}main{max-width:980px;margin:auto;padding:24px}h1{margin:0}.sub{color:#9daabe;margin:4px 0 18px}.grid{display:grid;grid-template-columns:360px 1fr;gap:16px}.panel{background:#19212d;border:1px solid #303c4c;border-radius:12px;padding:18px}.control{margin:13px 0}label{display:flex;justify-content:space-between;font-weight:600}input,select{width:100%;margin-top:7px}select{padding:8px;background:#10141c;color:#eef3fb;border:1px solid #435167;border-radius:6px}button{padding:10px 13px;border:0;border-radius:7px;font-weight:700;margin:7px 5px 0 0;cursor:pointer}.preset{background:#303c4c;color:#eef3fb}.reset{background:#6da7ff}.disturb{background:#f3a83b}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.metric{background:#10141c;padding:12px;border-radius:8px}.value{font-size:22px;font-weight:700;display:block}.unit{font-size:11px;color:#9daabe}canvas{width:100%;height:440px;background:#f4f1e8;border-radius:8px;margin-top:14px}.status{min-height:38px}@media(max-width:760px){.grid{grid-template-columns:1fr}.metrics{grid-template-columns:1fr}canvas{height:340px}}</style></head>
<body><main><h1>SO-101 Painting Lab</h1><div class="sub">Green guide versus blue painted stroke</div><div class="grid"><section class="panel"><strong>Controller</strong><br><button class="preset" data-preset="p">P</button><button class="preset" data-preset="pd">PD</button><button class="preset" data-preset="pid">PID</button><button class="preset" data-preset="ff">FF + PID</button><div class="control"><label>Painting</label><select id="shape"><option value="circle">Circle</option><option value="lemniscate">Lemniscate</option><option value="square">Square</option></select></div><div id="sliders"></div><button class="disturb" id="disturb">Bump the brush</button><button class="reset" id="reset">Clear and restart</button></section>
<section class="panel"><div class="metrics"><div class="metric"><span class="value" id="rms">0</span><span class="unit">path RMS · mm</span></div><div class="metric"><span class="value" id="max">0</span><span class="unit">maximum error · mm</span></div><div class="metric"><span class="value" id="sat">0</span><span class="unit">force saturation · %</span></div></div><p class="status" id="status">Starting the painting…</p><canvas id="canvas"></canvas></section></div></main>
<script>const specs=[['kp','Kp',0,15,.1],['ki','Ki',0,8,.05],['kd','Kd',0,1,.01],['ff','Feedforward',0,1.5,.05],['speed','Painting speed',.2,2.5,.1],['size','Painting size',.02,.07,.005]],box=document.getElementById('sliders');for(const [id,label,min,max,step] of specs)box.insertAdjacentHTML('beforeend',`<div class="control"><label>${label}<span id="${id}Value"></span></label><input id="${id}" type="range" min="${min}" max="${max}" step="${step}"></div>`);
async function send(){const b={shape:document.getElementById('shape').value};for(const [id] of specs)b[id]=Number(document.getElementById(id).value);await fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)})}for(const [id] of specs){const e=document.getElementById(id);e.oninput=()=>document.getElementById(id+'Value').textContent=e.value;e.onchange=send}document.getElementById('shape').onchange=send;for(const a of ['reset','disturb'])document.getElementById(a).onclick=()=>fetch('/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:a})});
const presets={p:{kp:5,ki:0,kd:0,ff:0},pd:{kp:6,ki:0,kd:.18,ff:0},pid:{kp:5,ki:3,kd:.18,ff:0},ff:{kp:5,ki:2,kd:.18,ff:1}};for(const b of document.querySelectorAll('[data-preset]'))b.onclick=()=>{for(const [k,v] of Object.entries(presets[b.dataset.preset]))document.getElementById(k).value=v;send()};
function draw(d){const c=document.getElementById('canvas'),r=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*r;c.height=h*r;const x=c.getContext('2d');x.scale(r,r);x.clearRect(0,0,w,h);const all=d.guide.concat(d.paint);if(!all.length)return;const ys=all.map(p=>p[0]),zs=all.map(p=>p[1]),loY=Math.min(...ys)-.015,hiY=Math.max(...ys)+.015,loZ=Math.min(...zs)-.015,hiZ=Math.max(...zs)+.015;function line(v,color,width,dash=[]){x.strokeStyle=color;x.lineWidth=width;x.setLineDash(dash);x.beginPath();v.forEach((p,i)=>{const px=(p[0]-loY)/(hiY-loY)*w,py=h-(p[1]-loZ)/(hiZ-loZ)*h;i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();x.setLineDash([])}line(d.guide,'#37b65f',2,[7,5]);line(d.paint,'#347fe5',4)}function explain(m){if(m.saturation_percent>30)return 'The joints are force-limited often—slow down or reduce gains.';if(m.rms_mm<4)return 'Clean stroke. Increase speed or choose the square to make it harder.';if(m.rms_mm<10)return 'Recognizable painting, but tracking error is visible.';return 'The brush is far from the guide. Try PD or FF + PID and reduce speed.'}
async function refresh(){try{const s=await(await fetch('/state')).json();for(const [id] of specs){const e=document.getElementById(id);if(document.activeElement!==e)e.value=s.settings[id];document.getElementById(id+'Value').textContent=Number(s.settings[id]).toFixed(id==='size'?3:id==='kp'||id==='speed'?1:2)}document.getElementById('shape').value=s.settings.shape;document.getElementById('rms').textContent=s.metrics.rms_mm.toFixed(1);document.getElementById('max').textContent=s.metrics.max_mm.toFixed(1);document.getElementById('sat').textContent=s.metrics.saturation_percent.toFixed(1);document.getElementById('status').textContent=explain(s.metrics);draw(s.drawing)}catch(e){}setTimeout(refresh,200)}refresh();</script></body></html>"""


def shape_point(name, phase, size):
    center = np.array([0.26, 0.0, 0.20])
    angle = 2.0 * np.pi * (phase % 1.0)
    if name == "circle":
        return center + np.array([0.0, size * np.cos(angle), size * np.sin(angle)])
    if name == "lemniscate":
        sine, cosine = np.sin(angle), np.cos(angle)
        denominator = 1.0 + sine**2
        return center + np.array([0.0, size * cosine / denominator, size * cosine * sine / denominator])
    segment = int((phase % 1.0) * 4.0)
    local = (phase * 4.0) % 1.0
    corners = ((1, 1), (-1, 1), (-1, -1), (1, -1), (1, 1))
    start, end = np.array(corners[segment]), np.array(corners[segment + 1])
    yz = size * (start + local * (end - start))
    return center + np.array([0.0, yz[0], yz[1]])


def solve_ik(model, data, target, seed, iterations=20):
    data.qpos[:] = seed
    for _ in range(iterations):
        mujoco.mj_forward(model, data)
        error = target - data.site("gripperframe").xpos
        if np.linalg.norm(error) < 1e-4:
            break
        jacobian = np.zeros((3, model.nv))
        rotational = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacobian, rotational, model.site("gripperframe").id)
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-4 * np.eye(3), error)
        data.qpos[:] = np.clip(data.qpos + 0.5 * delta, model.jnt_range[:, 0], model.jnt_range[:, 1])
    return data.qpos.copy()


def start_server(settings, events, metrics, drawing, lock, port):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, value, content_type="application/json"):
            payload = value.encode(); self.send_response(code); self.send_header("Content-Type", content_type); self.send_header("Content-Length", len(payload)); self.end_headers(); self.wfile.write(payload)
        def do_GET(self):
            if self.path == "/": self.reply(200, PANEL, "text/html; charset=utf-8")
            elif self.path == "/state":
                with lock: self.reply(200, json.dumps({"settings":settings,"metrics":metrics,"drawing":{k:list(v) for k,v in drawing.items()}}))
            else: self.reply(404, "{}")
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers.get("Content-Length",0))))
            with lock:
                if self.path == "/settings":
                    for key in ("kp","ki","kd","ff","speed","size"):
                        if key in body: settings[key]=max(0.0,float(body[key]))
                    if body.get("shape") in ("circle","lemniscate","square"): settings["shape"]=body["shape"]
                    events["reset"]=True
                elif self.path == "/action" and body.get("action") in events: events[body["action"]]=True
            self.reply(200,'{"ok":true}')
        def log_message(self,*_): pass
    server=ThreadingHTTPServer(("127.0.0.1",port),Handler); threading.Thread(target=server.serve_forever,daemon=True).start(); return server


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--port",type=int,default=8766); parser.add_argument("--duration",type=float,default=0.0); parser.add_argument("--no-browser",action="store_true"); args=parser.parse_args()
    model=mujoco_menagerie.load("robotstudio_so101","so101"); data=mujoco.MjData(model); ik_data=mujoco.MjData(model)
    settings={"kp":5.0,"ki":2.0,"kd":0.18,"ff":1.0,"speed":1.0,"size":0.05,"shape":"circle"}; events={"reset":False,"disturb":False}; metrics={"rms_mm":0.0,"max_mm":0.0,"saturation_percent":0.0}; drawing={"guide":deque(maxlen=240),"paint":deque(maxlen=1000)}; lock=threading.Lock(); server=start_server(settings,events,metrics,drawing,lock,args.port)
    url=f"http://127.0.0.1:{args.port}"; print(f"Painting panel: {url}"); webbrowser.open(url) if not args.no_browser else None
    control_dt=.02; physics_steps=round(control_dt/model.opt.timestep); force_limits=model.actuator_forcerange[:,1]; errors=deque(maxlen=250); saturation=deque(maxlen=250); disturbance_steps=0

    def reset():
        nonlocal errors,saturation
        for key in drawing: drawing[key].clear()
        errors.clear(); saturation.clear(); metrics.update(rms_mm=0.0,max_mm=0.0,saturation_percent=0.0)
        for sample in np.linspace(0,1,240,endpoint=False): drawing["guide"].append(shape_point(settings["shape"],sample,settings["size"])[1:].tolist())
        first=shape_point(settings["shape"],0,settings["size"]); seed=np.array([0,-1.2,1,.7,0,.2]); desired=solve_ik(model,ik_data,first,seed)
        mujoco.mj_resetData(model,data); data.qpos[:]=desired; data.ctrl[:]=desired; mujoco.mj_forward(model,data)
        controllers=[PIDController(settings["kp"],settings["ki"],settings["kd"],settings["ff"],1.0) for _ in range(5)]
        return controllers,desired.copy(),desired.copy(),0.0

    controllers,desired_q,position_command,experiment_time=reset(); started=time.monotonic()
    with mujoco.viewer.launch_passive(model,data) as viewer:
        viewer.cam.lookat[:]=(.24,0,.20); viewer.cam.distance=.65; viewer.cam.azimuth=90; viewer.cam.elevation=0
        while viewer.is_running():
            tick=time.monotonic()
            if args.duration and tick-started>=args.duration: break
            if events["reset"]: controllers,desired_q,position_command,experiment_time=reset(); events["reset"]=False
            target=shape_point(settings["shape"],experiment_time*settings["speed"]*.12,settings["size"]); previous_q=desired_q.copy(); desired_q=solve_ik(model,ik_data,target,desired_q); desired_velocity=(desired_q-previous_q)/control_dt
            for index,controller in enumerate(controllers):
                controller.kp=settings["kp"]; controller.ki=settings["ki"]; controller.kd=settings["kd"]; controller.velocity_feedforward=settings["ff"]
                command=controller.update(desired_q[index]-data.qpos[index],desired_velocity[index]-data.qvel[index],desired_velocity[index],control_dt); position_command[index]=np.clip(position_command[index]+command*control_dt,*model.jnt_range[index]); data.ctrl[index]=position_command[index]
            if events["disturb"]: disturbance_steps=round(.15/model.opt.timestep); events["disturb"]=False
            for _ in range(physics_steps): data.xfrc_applied[model.body("gripper").id,:3]=(0,8,0) if disturbance_steps else 0; mujoco.mj_step(model,data); disturbance_steps=max(0,disturbance_steps-1)
            data.xfrc_applied[model.body("gripper").id]=0; actual=data.site("gripperframe").xpos.copy(); error_mm=1000*np.linalg.norm(target-actual); errors.append(error_mm); saturation.append(np.any(np.abs(data.actuator_force[:5])>=force_limits[:5]-1e-6)); drawing["paint"].append(actual[1:].tolist()); metrics.update(rms_mm=float(np.sqrt(np.mean(np.square(errors)))),max_mm=float(max(errors)),saturation_percent=100*float(np.mean(saturation)))
            viewer.user_scn.ngeom=0
            for yz in drawing["guide"]:
                add_sphere(viewer.user_scn,(.26,*yz),.0015,(.1,1,.2,.45))
            for yz in list(drawing["paint"])[-500:]:
                add_sphere(viewer.user_scn,(.26,*yz),.002,(.1,.45,1,.75))
            viewer.sync(); experiment_time+=control_dt; remaining=control_dt-(time.monotonic()-tick); time.sleep(max(0,remaining))
    server.shutdown(); print("Painting lab closed.")


if __name__ == "__main__": main()
