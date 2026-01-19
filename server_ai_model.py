#!/usr/bin/env python3
from flask import (
    Flask, request, jsonify, render_template_string, send_from_directory, abort
)
from datetime import datetime
import os
import uuid
import json

app = Flask(__name__)

# Load YOLO model globally (assuming it's available; adjust path as needed)
try:
    from ultralytics import YOLO
    MODEL = YOLO(r"C:\Users\Administrator\Documents\Python\weights\best.pt")# Load a custom YOLO model
    print("[SERVER] YOLO model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load YOLO: {e}")
    MODEL = None

# ============================================================
# CONFIG  (absolute paths so saving/serving always matches)
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_SAVE_DIR = os.path.join(BASE_DIR, "received_photos")
MISSIONS_DIR = os.path.join(BASE_DIR, "missions")

os.makedirs(BASE_SAVE_DIR, exist_ok=True)
os.makedirs(MISSIONS_DIR, exist_ok=True)

# ============================================================
# GLOBAL STATE
# ============================================================

MISSION_STATE = "idle"              # "idle" or "start"
LAST_STATUS = {}                    # last status JSON from robot
CURRENT_MISSION_ID = None           # e.g. "mission_20251210_123456_ABCD12"
MISSIONS = {}                       # mission_id -> mission dict
CONTROL_COMMAND = "none"            # "none" | "abort" | "go_home"
LAST_TERMINAL_STATUS = None         # "mission_complete" | "home_unreachable" | "mission_aborted_by_operator"


# ============================================================
# HELPERS: MISSIONS
# ============================================================

def _new_mission_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6].upper()
    return f"mission_{ts}_{short}"


def _start_new_mission() -> str:
    """Create and register a new mission, set as current."""
    global CURRENT_MISSION_ID, MISSIONS, LAST_TERMINAL_STATUS

    mission_id = _new_mission_id()
    now = datetime.now().isoformat(timespec="seconds")

    MISSIONS[mission_id] = {
        "id": mission_id,
        "started_at": now,
        "ended_at": None,
        "status": "pending",  # pending|running|complete|home_unreachable|aborted
        "waypoints_reached": set(),
        "waypoints_unreachable": set(),
        "images_count": 0,
        "last_status": None,
    }
    CURRENT_MISSION_ID = mission_id
    LAST_TERMINAL_STATUS = None

    print(f"[SERVER] New mission created: {mission_id}")
    return mission_id


def _update_mission_on_status(status_payload: dict):
    """Update mission data based on status from the robot."""
    global CURRENT_MISSION_ID, MISSIONS, LAST_TERMINAL_STATUS

    if not CURRENT_MISSION_ID:
        if status_payload.get("status") == "mission_started":
            _start_new_mission()
        else:
            return

    mission = MISSIONS.get(CURRENT_MISSION_ID)
    if mission is None:
        return

    mission["last_status"] = status_payload
    status = status_payload.get("status", "")
    now = datetime.now().isoformat(timespec="seconds")

    if status == "mission_started":
        mission["status"] = "running"

    if status == "waypoint_reached":
        idx = status_payload.get("index")
        if idx is not None:
            mission["waypoints_reached"].add(int(idx))
    elif status == "waypoint_unreachable":
        idx = status_payload.get("index")
        if idx is not None:
            mission["waypoints_unreachable"].add(int(idx))

    if status in ("mission_complete", "mission_complete_after_abort"):
        mission["status"] = "complete"
        mission["ended_at"] = now
        LAST_TERMINAL_STATUS = "mission_complete"

    elif status.startswith("home_unreachable"):
        mission["status"] = "home_unreachable"
        mission["ended_at"] = now
        LAST_TERMINAL_STATUS = "home_unreachable"

    elif status == "mission_aborted_by_operator":
        mission["status"] = "aborted"
        mission["ended_at"] = now
        LAST_TERMINAL_STATUS = "mission_aborted_by_operator"


def _register_image_for_current_mission():
    """Increment image counter for current mission."""
    global CURRENT_MISSION_ID, MISSIONS
    if not CURRENT_MISSION_ID:
        _start_new_mission()
    mission = MISSIONS.get(CURRENT_MISSION_ID)
    if mission is None:
        return
    mission["images_count"] += 1
    if mission["status"] == "pending":
        mission["status"] = "running"


def _mission_summary(mission: dict) -> dict:
    return {
        "id": mission["id"],
        "started_at": mission["started_at"],
        "ended_at": mission["ended_at"],
        "status": mission["status"],
        "waypoints_reached": sorted(list(mission["waypoints_reached"])),
        "waypoints_unreachable": sorted(list(mission["waypoints_unreachable"])),
        "images_count": mission["images_count"],
    }


# ============================================================
# HTML / JS DASHBOARD
# ============================================================

INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Robot Mission Control+</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      :root {
        --bg1: #050816;
        --bg2: #020617;
        --accent: #38bdf8;
        --danger: #f97373;
        --success: #4ade80;
        --text-main: #e5e7eb;
        --text-muted: #9ca3af;
        --card-bg: rgba(15, 23, 42, 0.96);
        --border-subtle: rgba(148, 163, 184, 0.25);
        --shadow-soft: 0 22px 45px rgba(15, 23, 42, 0.9);
        --radius-xl: 20px;
      }
      * { box-sizing: border-box; margin: 0; padding: 0; }
      body {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 16px;
        color: var(--text-main);
        background:
          radial-gradient(circle at top left, #1d4ed8 0, transparent 55%),
          radial-gradient(circle at bottom right, #14b8a6 0, transparent 55%),
          linear-gradient(135deg, var(--bg1), var(--bg2));
      }
      .container { width: 100%; max-width: 1120px; }
      .card {
        background: var(--card-bg);
        border-radius: var(--radius-xl);
        box-shadow: var(--shadow-soft);
        border: 1px solid var(--border-subtle);
        padding: 18px 20px;
        backdrop-filter: blur(20px);
      }
      @media (min-width: 768px) { .card { padding: 24px 26px; } }
      .card-header {
        display: flex; flex-wrap: wrap;
        align-items: center; justify-content: space-between;
        gap: 12px; margin-bottom: 18px;
      }
      .title-block { display: flex; flex-direction: column; gap: 2px; }
      .title {
        font-size: 1.4rem; font-weight: 700; letter-spacing: 0.03em;
        display: flex; align-items: center; gap: 8px;
      }
      .logo-dot {
        width: 9px; height: 9px; border-radius: 999px;
        background: linear-gradient(135deg, #38bdf8, #a855f7);
        box-shadow: 0 0 12px rgba(56,189,248,0.9);
      }
      .subtitle { font-size: 0.85rem; color: var(--text-muted); }

      .status-pill {
        padding: 6px 12px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em;
        text-transform: uppercase; display: inline-flex; align-items: center;
        gap: 6px; border: 1px solid var(--border-subtle);
        background: rgba(15, 23, 42, 0.9);
      }
      .status-dot {
        width: 8px; height: 8px; border-radius: 999px;
        background: var(--text-muted);
      }
      .status-pill[data-level="idle"] .status-dot { background: #facc15; }
      .status-pill[data-level="running"] .status-dot { background: #38bdf8; }
      .status-pill[data-level="ok"] .status-dot { background: #4ade80; }
      .status-pill[data-level="error"] .status-dot { background: #f97373; }

      .layout { display: grid; gap: 18px; }
      @media (min-width: 900px) {
        .layout { grid-template-columns: minmax(0,1.8fr) minmax(0,1.2fr); }
      }

      .panel {
        border-radius: 16px;
        border: 1px solid var(--border-subtle);
        background: radial-gradient(circle at top left, rgba(56,189,248,0.09), transparent 55%),
                    linear-gradient(135deg, rgba(15,23,42,0.98), rgba(15,23,42,0.98));
        padding: 14px 14px 12px;
      }
      .panel + .panel {
        background: radial-gradient(circle at top right, rgba(168,85,247,0.12), transparent 55%),
                    linear-gradient(135deg, rgba(15,23,42,0.98), rgba(15,23,42,0.98));
      }
      .panel-header {
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 10px;
      }
      .panel-title {
        font-size: 0.9rem; text-transform: uppercase;
        letter-spacing: 0.14em; color: var(--text-muted);
      }
      .panel-pill {
        font-size: 0.7rem; padding: 4px 8px;
        border-radius: 999px; background: rgba(15,23,42,0.9);
        border: 1px solid rgba(148,163,184,0.35); color: var(--text-muted);
      }

      .btn {
        position: relative; display: inline-flex; align-items: center;
        justify-content: center; gap: 8px; padding: 10px 22px;
        border-radius: 999px; border: none; outline: none; cursor: pointer;
        font-size: 0.9rem; font-weight: 600; letter-spacing: 0.1em;
        text-transform: uppercase; color: #0b1120;
        transition: transform 0.12s, box-shadow 0.12s, filter 0.12s, opacity 0.12s;
      }
      .btn-main {
        background: linear-gradient(135deg, #22c55e, #22d3ee);
        box-shadow: 0 10px 30px rgba(34,197,94,0.55),
                    0 0 0 1px rgba(15,23,42,0.85);
      }
      .btn-danger {
        background: linear-gradient(135deg, #f97373, #f97316);
        box-shadow: 0 10px 26px rgba(248,113,113,0.45),
                    0 0 0 1px rgba(15,23,42,0.85);
      }
      .btn-home {
        background: linear-gradient(135deg, #22d3ee, #3b82f6);
        box-shadow: 0 10px 26px rgba(59,130,246,0.45),
                    0 0 0 1px rgba(15,23,42,0.85);
      }
      .btn:hover:not(:disabled) { transform: translateY(-1px) scale(1.01); filter: brightness(1.04); }
      .btn:active:not(:disabled) { transform: translateY(1px) scale(0.99); }
      .btn:disabled { opacity: 0.45; cursor: not-allowed; box-shadow: none; }

      .status-main { font-size: 0.95rem; margin-top: 16px; min-height: 44px; }
      .tag-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
      .tag {
        font-size: 0.72rem; padding: 4px 8px; border-radius: 999px;
        border: 1px solid rgba(148,163,184,0.4); background: rgba(15,23,42,0.95);
        color: var(--text-muted);
      }
      .tag strong { color: var(--text-main); }
      .hint { font-size: 0.75rem; color: var(--text-muted); margin-top: 8px; }

      .log-box {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 0.8rem; line-height: 1.6; color: var(--text-muted);
        background: rgba(15,23,42,0.9); border-radius: 12px;
        padding: 9px 10px; border: 1px solid rgba(55,65,81,0.7);
        max-height: 220px; overflow: auto;
      }
      .log-box pre { white-space: pre-wrap; word-wrap: break-word; }

      .sub-layout { display: grid; gap: 12px; margin-top: 12px; }
      @media (min-width: 900px) {
        .sub-layout { grid-template-columns: repeat(3, minmax(0,1fr)); }
      }

      .thumb-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(70px,1fr));
        gap: 8px;
      }
      .thumb-card {
        border-radius: 10px; overflow: hidden;
        border: 1px solid rgba(148,163,184,0.35);
        background: rgba(15,23,42,0.96);
      }
      .thumb-card img {
        width: 100%; height: 64px; object-fit: cover; display: block;
      }
      .thumb-card span {
        display: block; font-size: 0.7rem; padding: 4px 6px; color: var(--text-muted);
      }
      .thumb-card .detections {
        color: #38bdf8; font-weight: 500;
      }

      .mission-list { list-style: none; font-size: 0.78rem; color: var(--text-muted); }
      .mission-item { padding: 4px 0; border-bottom: 1px solid rgba(31,41,55,0.9); }
      .mission-item:last-child { border-bottom: none; }
      .mission-id { color: var(--text-main); font-weight: 500; }

      .badge {
        display: inline-block; padding: 2px 6px; border-radius: 999px;
        font-size: 0.68rem; margin-left: 6px;
        border: 1px solid rgba(148,163,184,0.45);
      }
      .badge.complete { color: #4ade80; border-color: rgba(74,222,128,0.7); }
      .badge.home_unreachable { color: #f97373; border-color: rgba(248,113,113,0.7); }
      .badge.running { color: #38bdf8; border-color: rgba(56,189,248,0.7); }
      .badge.pending { color: #facc15; border-color: rgba(250,204,21,0.7); }
      .badge.aborted { color: #f97373; border-color: rgba(248,113,113,0.7); }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="card">
        <div class="card-header">
          <div class="title-block">
            <div class="title">
              <span class="logo-dot"></span>
              Robot Mission Control+
            </div>
            <div class="subtitle">
              Start missions, monitor Nav2 status, inspect photos, AI predictions & mission history.
            </div>
          </div>
          <div class="status-pill" id="status-pill" data-level="idle">
            <span class="status-dot"></span>
            <span id="status-pill-text">IDLE</span>
          </div>
        </div>

        <div class="layout">
          <!-- LEFT -->
          <div class="panel">
            <div class="panel-header">
              <div class="panel-title">Mission Control</div>
              <div class="panel-pill" id="mission-state-pill">STATE: idle</div>
            </div>

            <div style="display:flex; flex-wrap:wrap; gap:8px;">
              <button id="btn-start" class="btn btn-main" onclick="startMission()">
                🔴 Start Mission
              </button>
              <button id="btn-abort" class="btn btn-danger" onclick="abortMission()" disabled>
                ⛔ Abort
              </button>
              <button id="btn-home" class="btn btn-home" onclick="returnHome()" disabled>
                🏠 Return Home
              </button>
            </div>

            <div class="status-main" id="status-main">
              Waiting for robot to connect...
            </div>

            <div class="tag-row">
              <div class="tag"><strong>Mission:</strong> <span id="tag-mission">—</span></div>
              <div class="tag"><strong>Waypoint:</strong> <span id="tag-waypoint">—</span></div>
              <div class="tag"><strong>Attempt:</strong> <span id="tag-attempt">—</span></div>
              <div class="tag"><strong>Phase:</strong> <span id="tag-phase">—</span></div>
            </div>

            <div class="sub-layout">
              <div>
                <div class="panel-header" style="margin-top:10px;">
                  <div class="panel-title">Latest Photos</div>
                  <div class="panel-pill">LIVE</div>
                </div>
                <div class="thumb-grid" id="thumb-grid">
                  <div class="thumb-card">
                    <span>No images yet.</span>
                  </div>
                </div>
                <div class="hint">Thumbnails update as the robot uploads photos from each waypoint.</div>
              </div>

              <div>
                <div class="panel-header" style="margin-top:10px;">
                  <div class="panel-title">AI Predictions</div>
                  <div class="panel-pill">YOLO LIVE</div>
                </div>
                <div class="thumb-grid" id="pred-grid">
                  <div class="thumb-card">
                    <span>No predictions yet.</span>
                  </div>
                </div>
                <div class="hint">YOLO detections on the latest photo per waypoint (boxes & classes).</div>
              </div>

              <div>
                <div class="panel-header" style="margin-top:10px;">
                  <div class="panel-title">Recent Missions</div>
                  <div class="panel-pill">HISTORY</div>
                </div>
                <ul class="mission-list" id="mission-list">
                  <li class="mission-item">No missions recorded yet.</li>
                </ul>
              </div>
            </div>
          </div>

          <!-- RIGHT -->
          <div class="panel">
            <div class="panel-header">
              <div class="panel-title">Robot Status (RAW)</div>
              <div class="panel-pill">LIVE JSON</div>
            </div>
            <div class="log-box">
              <pre id="status-json">No status received yet.</pre>
            </div>
          </div>
        </div>
      </div>
    </div>

    <script>
      async function startMission(){ try{ await fetch('/start_mission',{method:'POST'});}catch(e){console.error(e);}}
      async function abortMission(){ try{ await fetch('/abort_mission',{method:'POST'});}catch(e){console.error(e);}}
      async function returnHome(){ try{ await fetch('/return_home',{method:'POST'});}catch(e){console.error(e);}}

      function mapStatusToLevel(status){
        if(!status) return 'idle';
        const s=status.toLowerCase();
        if(s.includes('error')||s.includes('fail')||s.includes('unreachable')||s.includes('aborted')) return 'error';
        if(s.includes('started')||s.includes('moving')||s.includes('returning')) return 'running';
        if(s.includes('complete')||s.includes('reached')) return 'ok';
        return 'idle';
      }
      function prettifyStatus(status){ if(!status) return 'UNKNOWN'; return status.replace(/_/g,' ').toUpperCase(); }
      function badgeClass(status){
        if(!status) return 'pending';
        const s=status.toLowerCase();
        if(s.includes('complete')) return 'complete';
        if(s.includes('home_unreachable')) return 'home_unreachable';
        if(s.includes('running')) return 'running';
        if(s.includes('aborted')) return 'aborted';
        return 'pending';
      }

      // ====== BUTTON STATE MACHINE ======
      function updateButtons(effectiveStatus){
        // effectiveStatus is either terminal_status (if present) or current status
        const s=(effectiveStatus || '').toLowerCase();

        const btnStart=document.getElementById('btn-start');
        const btnAbort=document.getElementById('btn-abort');
        const btnHome=document.getElementById('btn-home');

        // defaults: idle
        let startEnabled=true;
        let abortEnabled=false;
        let homeEnabled=false;

        // Mission running: only Abort
        if(s.includes('moving_to_waypoint') || s.includes('mission_started')){
          startEnabled=false;
          abortEnabled=true;
          homeEnabled=false;

        // Returning home: only Abort
        } else if(s.includes('returning_home')){
          startEnabled=false;
          abortEnabled=true;
          homeEnabled=false;

        // Aborted: Abort + Return home
        } else if(s.includes('mission_aborted_by_operator')){
          startEnabled=false;
          abortEnabled=true;
          homeEnabled=true;

        // Home unreachable: only Return home (so you can retry) – Start disabled
        } else if(s.includes('home_unreachable')){
          startEnabled=false;
          abortEnabled=false;
          homeEnabled=true;

        // Mission complete: back to idle, only Start
        } else if(s.includes('mission_complete')){
          startEnabled=true;
          abortEnabled=false;
          homeEnabled=false;
        }

        btnStart.disabled=!startEnabled;
        btnAbort.disabled=!abortEnabled;
        btnHome.disabled=!homeEnabled;
      }

      async function pollStatus(){
        try{
          const r=await fetch('/last_status'); if(!r.ok) return;
          const js=await r.json();

          const status=js.status || 'mission_idle';
          const terminal=js.terminal_status || '';
          const effForButtons = terminal || status;  // use terminal state if present

          const wp=(js.index!==undefined)?js.index:'—';
          const attempt=js.attempt ?? '—';
          const missionId=js.mission_id || '—';

          let phase='—'; const s=status;
          if(s.includes('moving_to_waypoint')) phase='Moving';
          else if(s.includes('waypoint_reached')) phase='At waypoint';
          else if(s.includes('returning_home')) phase='Returning home';
          else if(s.includes('mission_complete')) phase='Done';
          else if(s.includes('mission_started')) phase='Running';
          else if(s.includes('mission_idle')) phase='Idle';
          else if(s.includes('home_unreachable')) phase='Home unreachable';
          else if(s.includes('aborted')) phase='Aborted';

          const pill=document.getElementById('status-pill');
          const pillText=document.getElementById('status-pill-text');
          pill.setAttribute('data-level', mapStatusToLevel(status));
          pillText.textContent = prettifyStatus(status);

          document.getElementById('mission-state-pill').textContent = 'STATE: ' + (js.mission_state || 'idle');
          document.getElementById('status-main').textContent = prettifyStatus(status);
          document.getElementById('tag-mission').textContent = missionId;
          document.getElementById('tag-waypoint').textContent = wp;
          document.getElementById('tag-attempt').textContent = attempt;
          document.getElementById('tag-phase').textContent = phase;
          document.getElementById('status-json').textContent = JSON.stringify(js, null, 2);

          updateButtons(effForButtons);
        }catch(e){ console.error('pollStatus', e); }
      }

      async function pollMissions(){
        try{
          const r=await fetch('/missions'); if(!r.ok) return;
          const js=await r.json(); const missions=js.missions || [];
          const list=document.getElementById('mission-list'); list.innerHTML='';
          if(!missions.length){
            const li=document.createElement('li');
            li.className='mission-item'; li.textContent='No missions recorded yet.';
            list.appendChild(li); return;
          }
          missions.slice(-6).reverse().forEach(m=>{
            const li=document.createElement('li'); li.className='mission-item';
            const spanId=document.createElement('span'); spanId.className='mission-id'; spanId.textContent=m.id;
            const badge=document.createElement('span'); badge.className='badge ' + badgeClass(m.status); badge.textContent=m.status.toUpperCase();
            const meta=document.createElement('div');
            meta.textContent=(m.started_at || '—') +
              (m.ended_at ? ' → ' + m.ended_at : '') +
              ' · imgs: ' + m.images_count +
              ' · wp✓ ' + m.waypoints_reached.length +
              ' / wp✗ ' + m.waypoints_unreachable.length;
            li.appendChild(spanId); li.appendChild(badge); li.appendChild(document.createElement('br')); li.appendChild(meta);
            list.appendChild(li);
          });
        }catch(e){ console.error('pollMissions', e); }
      }

      async function pollThumbnails(){
        try{
          const r=await fetch('/latest_photos'); if(!r.ok) return;
          const js=await r.json(); const grid=document.getElementById('thumb-grid'); grid.innerHTML='';
          const items=js.photos || [];
          if(!items.length){
            const card=document.createElement('div'); card.className='thumb-card';
            const span=document.createElement('span'); span.textContent='No images yet.';
            card.appendChild(span); grid.appendChild(card); return;
          }
          items.forEach(p=>{
            const card=document.createElement('div'); card.className='thumb-card';
            const img=document.createElement('img'); img.src=p.url; img.alt=p.label || '';
            const span=document.createElement('span'); span.textContent=p.label || '';
            card.appendChild(img); card.appendChild(span); grid.appendChild(card);
          });
        }catch(e){ console.error('pollThumbnails', e); }
      }

      async function pollPredictions(){
        try{
          const r=await fetch('/latest_predictions'); if(!r.ok) return;
          const js=await r.json(); const grid=document.getElementById('pred-grid'); grid.innerHTML='';
          const items=js.photos || [];
          if(!items.length){
            const card=document.createElement('div'); card.className='thumb-card';
            const span=document.createElement('span'); span.textContent='No predictions yet.';
            card.appendChild(span); grid.appendChild(card); return;
          }
          items.forEach(p=>{
            const card=document.createElement('div'); card.className='thumb-card';
            const img=document.createElement('img'); img.src=p.url; img.alt=p.label || '';
            const span=document.createElement('span'); span.textContent=p.label || '';
            const detSpan=document.createElement('span'); detSpan.className='detections';
            detSpan.textContent = p.detections && p.detections.length ? p.detections.join(', ') : 'No detections';
            card.appendChild(img); card.appendChild(span); card.appendChild(detSpan); grid.appendChild(card);
          });
        }catch(e){ console.error('pollPredictions', e); }
      }

      setInterval(pollStatus,1500);
      setInterval(pollMissions,3000);
      setInterval(pollThumbnails,3500);
      setInterval(pollPredictions,3500);
      pollStatus(); pollMissions(); pollThumbnails(); pollPredictions();
    </script>
  </body>
</html>
"""


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/start_mission", methods=["POST"])
def start_mission():
    global MISSION_STATE, LAST_STATUS, LAST_TERMINAL_STATUS
    MISSION_STATE = "start"
    mission_id = _start_new_mission()
    LAST_STATUS.setdefault("status", "mission_idle")
    LAST_STATUS["mission_id"] = mission_id
    LAST_TERMINAL_STATUS = None
    print("[SERVER] Mission start requested via UI")
    return jsonify({"ok": True, "mission_id": mission_id})


@app.route("/mission_state", methods=["GET"])
def mission_state():
    global MISSION_STATE
    state = MISSION_STATE
    if MISSION_STATE == "start":
        MISSION_STATE = "idle"   # one-shot
    return jsonify({"mission_state": state})


@app.route("/abort_mission", methods=["POST"])
def abort_mission():
    global CONTROL_COMMAND
    CONTROL_COMMAND = "abort"
    print("[SERVER] Abort requested by operator")
    return jsonify({"ok": True, "command": "abort"})


@app.route("/return_home", methods=["POST"])
def return_home():
    global CONTROL_COMMAND
    CONTROL_COMMAND = "go_home"
    print("[SERVER] Return-home requested by operator")
    return jupytext({"ok": True, "command": "go_home"})


@app.route("/control_state", methods=["GET"])
def control_state():
    global CONTROL_COMMAND
    cmd = CONTROL_COMMAND
    CONTROL_COMMAND = "none"   # one-shot
    return jsonify({"command": cmd})


@app.route("/status_update", methods=["POST"])
def status_update():
    global LAST_STATUS, MISSION_STATE
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        payload = {"status": "bad_status_payload"}

    status = payload.get("status", "")

    if status == "mission_idle":
        MISSION_STATE = "idle"

    payload.setdefault("mission_state", MISSION_STATE)
    payload.setdefault("mission_id", CURRENT_MISSION_ID or "—")

    LAST_STATUS = payload
    print("[STATUS]", LAST_STATUS)

    _update_mission_on_status(payload)
    return jsonify({"ok": True})


@app.route("/last_status", methods=["GET"])
def last_status():
    global LAST_STATUS, LAST_TERMINAL_STATUS
    if not LAST_STATUS:
        return jsonify({
            "status": "no_status_yet",
            "mission_state": MISSION_STATE,
            "mission_id": CURRENT_MISSION_ID or "—",
            "terminal_status": LAST_TERMINAL_STATUS,
        })
    if "mission_state" not in LAST_STATUS:
        LAST_STATUS["mission_state"] = MISSION_STATE
    if "mission_id" not in LAST_STATUS:
        LAST_STATUS["mission_id"] = CURRENT_MISSION_ID or "—"
    LAST_STATUS["terminal_status"] = LAST_TERMINAL_STATUS
    return jsonify(LAST_STATUS)


@app.route("/missions", methods=["GET"])
def missions():
    items = [_mission_summary(m) for m in MISSIONS.values()]
    items_sorted = sorted(items, key=lambda x: x["started_at"] or "", reverse=True)
    return jsonify({"missions": items_sorted})


@app.route("/latest_photos", methods=["GET"])
def latest_photos():
    photos = []
    if not CURRENT_MISSION_ID:
        return jsonify({"photos": photos})

    mission_root = os.path.join(BASE_SAVE_DIR, CURRENT_MISSION_ID)
    if not os.path.isdir(mission_root):
        return jsonify({"photos": photos})

    for wp_name in sorted(os.listdir(mission_root)):
        wp_path = os.path.join(mission_root, wp_name)
        if not os.path.isdir(wp_path):
            continue
        files = [f for f in sorted(os.listdir(wp_path)) if f.startswith("img") and f.endswith(".jpg")]
        if not files:
            continue
        last_file = files[-1]
        url = f"/photo/{CURRENT_MISSION_ID}/{wp_name}/{last_file}"
        label = f"{wp_name} · {last_file}"
        photos.append({"url": url, "label": label})

    photos = photos[-8:]
    return jsonify({"photos": photos})


@app.route("/latest_predictions", methods=["GET"])
def latest_predictions():
    photos = []
    if not CURRENT_MISSION_ID:
        return jsonify({"photos": photos})

    mission_root = os.path.join(BASE_SAVE_DIR, CURRENT_MISSION_ID)
    if not os.path.isdir(mission_root):
        return jsonify({"photos": photos})

    for wp_name in sorted(os.listdir(mission_root)):
        wp_path = os.path.join(mission_root, wp_name)
        if not os.path.isdir(wp_path):
            continue
        files = [f for f in sorted(os.listdir(wp_path)) if f.startswith("pred_") and f.endswith(".jpg")]
        if not files:
            continue
        last_file = files[-1]
        # Extract image_index from last_file (e.g., pred_img3_20240114_124600.jpg -> image_index=3)
        try:
            img_part = last_file.split('_')[1]  # 'img3'
            image_index = img_part[3:]  # '3'
        except Exception:
            image_index = "unknown"
        url = f"/photo/{CURRENT_MISSION_ID}/{wp_name}/{last_file}"
        label = f"{wp_name} · {last_file.replace('pred_', '')}"
        # Load detections if available
        dets_file = f"pred_dets_{image_index}.json"
        dets_path = os.path.join(wp_path, dets_file)
        detections = []
        if os.path.isfile(dets_path):
            with open(dets_path, "r") as f:
                detections = json.load(f)
        photos.append({"url": url, "label": label, "detections": detections})

    photos = photos[-8:]
    return jsonify({"photos": photos})


@app.route("/photo/<mission_id>/<wp_folder>/<filename>")
def serve_photo(mission_id, wp_folder, filename):
    folder = os.path.join(BASE_SAVE_DIR, mission_id, wp_folder)
    full_path = os.path.join(folder, filename)
    print(f"[PHOTO_REQ] mission={mission_id} wp={wp_folder} file={filename}")
    print(f"           -> FS path: {full_path}")

    if not os.path.isfile(full_path):
        print("[PHOTO_REQ] FILE NOT FOUND")
        abort(404)

    return send_from_directory(folder, filename)


@app.route("/upload_photo", methods=["POST"])
def upload_photo():
    waypoint_index = request.form.get("waypoint_index", "unknown")
    image_index = request.form.get("image_index", "0")
    file = request.files.get("image")

    if file is None:
        return jsonify({"status": "error", "message": "no image field"}), 400

    _register_image_for_current_mission()
    mission_id = CURRENT_MISSION_ID or "nomission"

    mission_folder = os.path.join(BASE_SAVE_DIR, mission_id)
    wp_folder_name = f"wp{waypoint_index}"
    wp_folder_path = os.path.join(mission_folder, wp_folder_name)
    os.makedirs(wp_folder_path, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"img{image_index}_{ts}.jpg"
    full_path = os.path.join(wp_folder_path, filename)

    file.save(full_path)
    print(f"[PHOTO_SAVE] {full_path}")

    # Run YOLO inference if model is loaded (on every photo, but UI shows only latest per waypoint)
    if MODEL:
        try:
            results = MODEL(full_path)[0]
            pred_filename = f"pred_{filename}"
            pred_path = os.path.join(wp_folder_path, pred_filename)
            results.save(filename=pred_path)
            print(f"[YOLO_SAVE] {pred_path}")

            # Save detected classes as JSON
            detected = []
            if results.boxes:
                detected = list(set(results.names[int(cls)] for cls in results.boxes.cls))
            dets_path = os.path.join(wp_folder_path, f"pred_dets_{image_index}.json")
            with open(dets_path, "w") as f:
                json.dump(detected, f)
            print(f"[DETS_SAVE] {dets_path} - {detected}")
        except Exception as e:
            print(f"[YOLO_ERROR] {e}")

    return jsonify({
        "status": "ok",
        "filename": filename,
        "waypoint_folder": wp_folder_name,
        "mission_id": mission_id,
    })


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print(f"[SERVER] BASE_SAVE_DIR = {BASE_SAVE_DIR}")
    app.run(host="0.0.0.0", port=5000, debug=False)