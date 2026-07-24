"""
dashboard.py
============
GymGuardian web dashboard — a minimal Flask app served on the LOCAL network
so gym staff can watch status and acknowledge incidents from a phone.

PRIVACY BY DESIGN: this dashboard serves STATUS AND EVENTS ONLY. It never
serves video, frames, or images of any kind — the "no video ever leaves the
device" claim holds. Everything is plain text/JSON generated on-device.

Graceful degradation: if Flask is not installed, start_dashboard() prints a
note and returns None — the monitor keeps running exactly as before.
"""

import threading

import config


# The whole UI is one self-contained page: inline CSS + JS, no external
# CDNs or fonts (the app must work with no internet at all). The page polls
# /api/status every 1.5 s and re-renders — no websockets needed for an MVP.
PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GymGuardian</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; margin: 0; }
  body { background: #0e1116; color: #e6e9ef; font-family: -apple-system,
         'Segoe UI', Roboto, Helvetica, Arial, sans-serif; padding: 20px; }
  .wrap { max-width: 860px; margin: 0 auto; }
  header { display: flex; align-items: baseline; gap: 12px; margin-bottom: 20px; }
  header h1 { font-size: 26px; letter-spacing: 0.5px; }
  header .tag { color: #8b93a5; font-size: 13px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
           gap: 14px; margin-bottom: 22px; }
  .card { background: #171b23; border: 1px solid #232936; border-radius: 12px;
          padding: 16px 18px; }
  .card h3 { font-size: 12px; text-transform: uppercase; letter-spacing: 1px;
             color: #8b93a5; margin-bottom: 8px; }
  .big { font-size: 22px; font-weight: 600; }
  .sub { color: #8b93a5; font-size: 13px; margin-top: 4px; }
  .state-normal    { color: #4ade80; }
  .state-suspicious{ color: #fbbf24; }
  .state-fall      { color: #f87171; }
  .state-entrap    { color: #e879f9; }
  .state-lost      { color: #fb923c; }
  .state-none      { color: #8b93a5; }
  .esc { border-color: #7f1d1d; background: #1f1416; }
  .esc.stage0 { border-color: #92600a; background: #1d1810; }
  .ackbtn { margin-top: 10px; background: #16a34a; color: white; border: 0;
            padding: 9px 18px; border-radius: 8px; font-size: 15px;
            font-weight: 600; cursor: pointer; }
  .ackbtn:active { background: #15803d; }
  table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #232936; }
  th { color: #8b93a5; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
  .pill { padding: 2px 9px; border-radius: 999px; font-size: 11.5px; font-weight: 600; }
  .pill.FALL       { background: #7f1d1d; color: #fecaca; }
  .pill.ENTRAPMENT { background: #701a75; color: #f5d0fe; }
  .st-active { color: #f87171; font-weight: 600; }
  .st-ack    { color: #4ade80; }
  .st-cancel { color: #8b93a5; }
  footer { margin-top: 26px; color: #566072; font-size: 12px; line-height: 1.6; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>&#128737;&#65039; GymGuardian</h1>
    <span class="tag">local safety monitor &mdash; no video leaves this device</span>
  </header>

  <div class="cards">
    <div class="card">
      <h3>Status</h3>
      <div class="big" id="state">loading&hellip;</div>
      <div class="sub" id="activity"></div>
    </div>
    <div class="card">
      <h3>Session</h3>
      <div class="big" id="uptime">&ndash;</div>
      <div class="sub" id="fps"></div>
    </div>
    <div class="card esc" id="esccard" style="display:none">
      <h3>Escalation</h3>
      <div class="big" id="escstage"></div>
      <div class="sub" id="escinfo"></div>
      <button class="ackbtn" onclick="ack()">Acknowledge</button>
    </div>
  </div>

  <div class="card">
    <h3>Incident log</h3>
    <table>
      <thead><tr><th>Time</th><th>Type</th><th>Stage</th><th>Status</th><th>Metrics</th></tr></thead>
      <tbody id="rows"><tr><td colspan="5" class="sub">no incidents yet</td></tr></tbody>
    </table>
  </div>

  <footer>
    All pose analysis happens on the gym's own computer. This dashboard shows
    status and event text only &mdash; no camera frames are stored, streamed,
    or displayed. Stage&nbsp;2 escalation is <b>simulated</b>: this prototype
    never contacts real emergency services.
  </footer>
</div>

<script>
const stateClass = (s) => {
  if (s === "Normal") return "state-normal";
  if (s === "Suspicious Movement") return "state-suspicious";
  if (s === "Possible Fall Detected") return "state-fall";
  if (s === "Possible Entrapment Detected") return "state-entrap";
  if (s === "Person Lost After Incident") return "state-lost";
  return "state-none";
};
const fmtUp = (s) => {
  s = Math.floor(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return (h ? h + "h " : "") + m + "m " + (s % 60) + "s";
};
async function ack() {
  try { await fetch("/api/ack", {method: "POST"}); refresh(); } catch (e) {}
}
async function refresh() {
  try {
    const r = await fetch("/api/status");
    const d = await r.json();
    const st = document.getElementById("state");
    st.textContent = d.state;
    st.className = "big " + stateClass(d.state);
    document.getElementById("activity").textContent = d.activity || "";
    document.getElementById("uptime").textContent = fmtUp(d.uptime_s);
    document.getElementById("fps").textContent = d.fps ? d.fps.toFixed(1) + " fps" : "";

    const card = document.getElementById("esccard");
    if (d.escalation && d.escalation.active) {
      const a = d.escalation.active;
      card.style.display = "";
      card.className = "card esc" + (a.stage === 0 ? " stage0" : "");
      document.getElementById("escstage").textContent =
        "Stage " + a.stage + ": " + a.stage_name;
      document.getElementById("escinfo").textContent = a.message +
        (a.next_stage_in > 0 ? " - next stage in " + Math.ceil(a.next_stage_in) + "s" : "");
    } else { card.style.display = "none"; }

    const rows = document.getElementById("rows");
    const incs = (d.escalation && d.escalation.incidents) || [];
    if (incs.length) {
      rows.innerHTML = incs.map(i => {
        const stCls = i.status === "ACTIVE" ? "st-active"
                    : i.status.startsWith("ACK") ? "st-ack" : "st-cancel";
        const m = i.metrics || {};
        const mtxt = "angle " + (m.torso_angle ?? "-") + "&deg; / motion " + (m.motion ?? "-");
        return "<tr><td>" + i.created + "</td>" +
               "<td><span class='pill " + i.type + "'>" + i.type + "</span></td>" +
               "<td>" + i.stage_name + "</td>" +
               "<td class='" + stCls + "'>" + i.status + "</td>" +
               "<td class='sub'>" + mtxt + "</td></tr>";
      }).join("");
    }
  } catch (e) { /* monitor may be restarting; keep polling */ }
}
refresh();
setInterval(refresh, 1500);
</script>
</body>
</html>"""


def start_dashboard(manager, get_status):
    """
    Start the Flask dashboard in a daemon thread. `manager` is the
    IncidentManager; `get_status` is a callable returning the live status
    dict (state, activity, uptime_s, fps). Returns the thread, or None if
    Flask is unavailable (the monitor keeps working without it).
    """
    try:
        from flask import Flask, jsonify
    except ImportError:
        print("[dashboard] Flask not installed - dashboard disabled. "
              "Run: pip install flask")
        return None

    # Silence per-request logging so alert banners stay readable.
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    app = Flask("gymguardian")

    @app.route("/")
    def index():
        return PAGE

    @app.route("/api/status")
    def status():
        data = get_status()
        data["escalation"] = manager.dashboard_state()
        # Lift the active incident up for the JS (kept nested for incidents).
        data["escalation"]["active"] = data["escalation"].get("active")
        return jsonify(data)

    @app.route("/api/ack", methods=["POST"])
    def ack():
        return jsonify({"acknowledged": manager.acknowledge("dashboard")})

    def _run():
        try:
            app.run(host=config.DASHBOARD_HOST, port=config.DASHBOARD_PORT,
                    debug=False, use_reloader=False, threaded=True)
        except Exception as exc:
            # e.g. port already in use — never take the monitor down with us.
            print(f"[dashboard] Dashboard stopped: {exc}")

    thread = threading.Thread(target=_run, daemon=True, name="gymguardian-dash")
    thread.start()
    print(f"[dashboard] GymGuardian dashboard: http://localhost:{config.DASHBOARD_PORT}"
          f"  (LAN: http://<this-computer's-IP>:{config.DASHBOARD_PORT})")
    return thread
