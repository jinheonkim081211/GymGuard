"""
alert.py
========
Alert channels + the staged escalation manager for GymGuardian.

ESCALATION FLOW (all timings in config.py):
    incident confirmed (FALL or ENTRAPMENT)
        v
    Stage 0 "Voice check"     — the computer asks "Are you OK?" out loud.
        |   person moves again within STAGE0_RESPONSE_WINDOW
        |     -> incident CANCELLED ("recovered after voice check")
        v   no response
    Stage 1 "Staff alert"     — loud alarm + desktop notification + dashboard red.
        v   still no response after STAGE1_RESPONSE_WINDOW
    Stage 2 "Emergency contact (SIMULATED)"
        — displayed and logged ONLY. This prototype NEVER contacts real
          emergency services, SMS, or phone systems.

    Acknowledge (dashboard button or 'a' key) stops escalation at any stage.

All external effects (sound, speech, notifications, file writes) are wrapped
in try/except so a missing dependency or OS quirk never crashes monitoring.

PRIVACY: we log only timestamps, states, and numeric metrics. No video
frames, no images, and no identity information are ever written or uploaded.
"""

import csv
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

import config


def _path(name):
    """Absolute path next to this file, so it works regardless of CWD."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


# ---------------------------------------------------------------------------
# Sound / speech channels (best effort, never blocking, never crashing)
# ---------------------------------------------------------------------------
def _beep_blocking(times=3):
    """Simple beep, cross-platform, tried in order of preference."""
    if sys.platform.startswith("win"):
        try:
            import winsound
            for _ in range(times):
                winsound.Beep(880, 250)
                time.sleep(0.1)
            return
        except Exception:
            pass
    if sys.platform == "darwin":
        try:
            for _ in range(times):
                os.system("afplay /System/Library/Sounds/Sosumi.aiff")
            return
        except Exception:
            pass
    try:
        sys.stdout.write("\a" * times)
        sys.stdout.flush()
    except Exception:
        pass


def play_beep():
    """Short beep on a background thread (non-blocking)."""
    if config.ENABLE_SOUND_ALERT:
        threading.Thread(target=_beep_blocking, args=(1,), daemon=True).start()


def play_alarm():
    """Loud/insistent alarm for Stage 1, on a background thread."""
    if config.ENABLE_SOUND_ALERT:
        threading.Thread(target=_beep_blocking, args=(4,), daemon=True).start()


def speak_voice_check():
    """
    Stage 0: speak "Are you OK?" via macOS `say`. Falls back to a beep on
    any failure (other OS, `say` missing, muted audio hardware...).
    """
    if not config.ENABLE_VOICE_CHECK:
        play_beep()
        return
    try:
        if sys.platform == "darwin":
            # Popen = non-blocking; the video loop must never wait on audio.
            subprocess.Popen(["say", config.VOICE_CHECK_TEXT],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
    except Exception:
        pass
    play_beep()   # fallback for non-macOS or any error


# ---------------------------------------------------------------------------
# Console / desktop notification channels
# ---------------------------------------------------------------------------
def console_banner(title, message, metrics=None):
    """Print a big, hard-to-miss banner to the terminal."""
    if not config.ENABLE_CONSOLE_ALERT:
        return
    line = "=" * 62
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + line)
    print(f"  !!  {title}  !!")
    print(f"  {message}")
    print(f"  Time: {stamp}")
    if metrics:
        print("  Metrics: "
              f"drop_speed={metrics.get('drop_speed', 0):.2f}  "
              f"torso_angle={metrics.get('torso_angle', 0):.1f}deg  "
              f"center_y={metrics.get('center_y', 0):.2f}  "
              f"motion={metrics.get('motion', 0):.3f}/s")
    print(line + "\n")


def desktop_notification(message):
    """Best-effort OS notification; silently does nothing if unsupported."""
    if not config.ENABLE_DESKTOP_NOTIFICATION:
        return
    try:
        from plyer import notification
        notification.notify(title="GymGuardian", message=message, timeout=5)
        return
    except Exception:
        pass
    if sys.platform == "darwin":
        try:
            safe = message.replace('"', "'")
            os.system(f'osascript -e \'display notification "{safe}" '
                      f'with title "GymGuardian"\'')
        except Exception:
            pass


# ---------------------------------------------------------------------------
# File logging (alerts.log + events.csv)
# ---------------------------------------------------------------------------
def log_alert(message, metrics=None):
    """Append a timestamped entry to the alert log file."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metric_str = ""
    if metrics:
        metric_str = (f" | drop_speed={metrics.get('drop_speed', 0):.2f}"
                      f" torso_angle={metrics.get('torso_angle', 0):.1f}"
                      f" center_y={metrics.get('center_y', 0):.2f}"
                      f" motion={metrics.get('motion', 0):.3f}")
    try:
        with open(_path(config.LOG_FILE), "a", encoding="utf-8") as fh:
            fh.write(f"[{stamp}] {message}{metric_str}\n")
    except Exception as exc:
        print(f"[alert] Could not write log file: {exc}")


CSV_COLUMNS = ["timestamp", "event", "incident_type", "stage",
               "drop_speed", "torso_angle", "center_y", "motion", "outcome"]


def log_csv_event(event, incident_type="", stage="", metrics=None, outcome=""):
    """
    Append one row to events.csv: state transitions, alerts, escalation
    stages, cancels, acknowledgments. These rows are the raw numbers behind
    a measured false-alarm rate.
    """
    m = metrics or {}
    row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           event, incident_type, stage,
           f"{m.get('drop_speed', 0):.3f}", f"{m.get('torso_angle', 0):.1f}",
           f"{m.get('center_y', 0):.3f}", f"{m.get('motion', 0):.3f}",
           outcome]
    path = _path(config.EVENTS_CSV)
    try:
        new_file = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if new_file:
                writer.writerow(CSV_COLUMNS)
            writer.writerow(row)
    except Exception as exc:
        print(f"[alert] Could not write events.csv: {exc}")


# ---------------------------------------------------------------------------
# The staged escalation / incident manager
# ---------------------------------------------------------------------------
STAGE_NAMES = ["Voice check", "Staff alert", "Emergency contact (SIMULATED)"]


class IncidentManager:
    """
    Owns the incident list and drives the staged escalation. main.py calls
    update() every frame with the detector's clock; the dashboard thread
    calls dashboard_state() and acknowledge() — a lock keeps them safe.

    Incidents persist to incidents.json (text only) so the log survives
    restarts.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.incidents = self._load()
        self.active = None          # the incident currently escalating
        self._last_now = None       # detector clock at the last update()
        # Session counters (printed in the quit summary).
        self.counters = {"incidents": 0, "voice_recoveries": 0,
                         "other_cancels": 0, "acknowledged": 0,
                         "stage1_reached": 0, "stage2_reached": 0}

    # ------------------------------------------------------------------
    # Incident lifecycle
    # ------------------------------------------------------------------
    def on_alert(self, now, incident_type, message, metrics):
        """
        Called when the detector fires an alert. Starts a new incident at
        Stage 0 (voice check) unless one is already escalating.
        """
        with self._lock:
            if self.active is not None:
                return None   # already escalating; don't stack incidents
            incident = {
                "id": datetime.now().strftime("inc_%Y%m%d_%H%M%S"),
                "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "t0": now,
                "type": incident_type or "FALL",
                "message": message,
                "stage": 0,
                "stage_name": STAGE_NAMES[0],
                "status": "ACTIVE",
                "metrics": {k: round(float(v), 3) for k, v in (metrics or {}).items()
                            if isinstance(v, (int, float))},
                "history": [self._h("confirmed - Stage 0 voice check")],
            }
            self.incidents.insert(0, incident)
            self.active = incident
            self.counters["incidents"] += 1
            self._save()

        # Stage 0 side effects (outside the lock — audio may take a moment).
        console_banner("INCIDENT", f"{message}  [Stage 0: voice check]", metrics)
        speak_voice_check()
        log_alert(f"STAGE 0 (voice check): {message}", metrics)
        log_csv_event("alert", incident["type"], "0", metrics)
        return incident

    def update(self, now, motion=None, recovered_reason=None):
        """
        Advance the escalation clock. Called every frame with the detector's
        `now` (video time in --video mode, wall clock on webcam) so stage
        timings track the same clock as detection.
        """
        actions = []
        with self._lock:
            self._last_now = now
            inc = self.active
            if inc is None:
                return actions

            # --- Cancel: the detector saw a genuine recovery ---------------
            if recovered_reason:
                self._close(inc, f"CANCELLED - recovered ({recovered_reason})")
                self.counters["other_cancels"] += 1
                actions.append(("cancelled", recovered_reason))

            # --- Stage 0: waiting for a response to the voice check --------
            elif inc["stage"] == 0:
                elapsed = now - inc["t0"]
                # Motion counts as "responded" for FALL incidents only. An
                # ENTRAPMENT victim struggling under the bar produces motion
                # too — entrapment cancels only when the wrists actually come
                # free (the detector reports that via recovered_reason).
                if (inc["type"] == "FALL" and motion is not None
                        and motion > config.RECOVERY_MOTION_THRESHOLD):
                    self._close(inc, "CANCELLED - recovered after voice check")
                    self.counters["voice_recoveries"] += 1
                    actions.append(("cancelled", "recovered after voice check"))
                elif elapsed >= config.STAGE0_RESPONSE_WINDOW:
                    inc["stage"] = 1
                    inc["stage_name"] = STAGE_NAMES[1]
                    inc["history"].append(self._h("no response - Stage 1 staff alert"))
                    self.counters["stage1_reached"] += 1
                    self._save()
                    actions.append(("stage1", inc["message"]))

            # --- Stage 1: waiting before the simulated emergency stage -----
            elif inc["stage"] == 1:
                elapsed = now - inc["t0"] - config.STAGE0_RESPONSE_WINDOW
                if elapsed >= config.STAGE1_RESPONSE_WINDOW:
                    inc["stage"] = 2
                    inc["stage_name"] = STAGE_NAMES[2]
                    inc["history"].append(self._h("no response - Stage 2 SIMULATED"))
                    self.counters["stage2_reached"] += 1
                    self._save()
                    actions.append(("stage2", inc["message"]))
            # Stage 2: nothing further — stays ACTIVE until ack or recovery.

        # Side effects outside the lock.
        for kind, msg in actions:
            if kind == "stage1":
                console_banner("STAGE 1 - STAFF ALERT", msg)
                play_alarm()
                desktop_notification(f"STAFF ALERT: {msg}")
                log_alert(f"STAGE 1 (staff alert): {msg}")
                log_csv_event("escalation", self._last_type(), "1")
            elif kind == "stage2":
                sim = ("SIMULATED: emergency contact would now be notified - "
                       "never auto-dialed in this prototype")
                console_banner("STAGE 2 - EMERGENCY (SIMULATED)", sim)
                log_alert(f"STAGE 2 (SIMULATED emergency contact): {msg}")
                log_csv_event("escalation", self._last_type(), "2", outcome="simulated")
            elif kind == "cancelled":
                console_banner("INCIDENT CANCELLED", f"Recovered: {msg}")
                log_alert(f"CANCELLED (recovered): {msg}")
                log_csv_event("cancel", self._last_type(), outcome=f"recovered: {msg}")
        return actions

    def acknowledge(self, source="dashboard"):
        """
        Mark the active incident handled (dashboard button or 'a' key).
        Stops escalation at whatever stage it reached. Returns True if an
        incident was actually acknowledged.
        """
        with self._lock:
            inc = self.active
            if inc is None:
                return False
            self._close(inc, "ACKNOWLEDGED")
            inc["history"].append(self._h(f"acknowledged via {source}"))
            self.counters["acknowledged"] += 1
            self._save()
        console_banner("ALERT ACKNOWLEDGED", f"Handled via {source}.")
        log_alert(f"ACKNOWLEDGED via {source}")
        log_csv_event("acknowledge", self._last_type(),
                      outcome=f"acknowledged via {source}")
        return True

    # ------------------------------------------------------------------
    # Dashboard view (called from the Flask thread)
    # ------------------------------------------------------------------
    def dashboard_state(self):
        """Thread-safe snapshot of escalation + incident history."""
        with self._lock:
            active = None
            if self.active is not None:
                inc = self.active
                elapsed = (self._last_now - inc["t0"]) if self._last_now else 0.0
                if inc["stage"] == 0:
                    remaining = max(0.0, config.STAGE0_RESPONSE_WINDOW - elapsed)
                elif inc["stage"] == 1:
                    remaining = max(0.0, config.STAGE0_RESPONSE_WINDOW
                                    + config.STAGE1_RESPONSE_WINDOW - elapsed)
                else:
                    remaining = 0.0
                active = {"id": inc["id"], "type": inc["type"],
                          "message": inc["message"], "stage": inc["stage"],
                          "stage_name": inc["stage_name"],
                          "elapsed": round(elapsed, 1),
                          "next_stage_in": round(remaining, 1)}
            # Shallow copies are fine: the dashboard only reads.
            return {"active": active, "incidents": [dict(i) for i in self.incidents[:50]]}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _h(text):
        return {"t": datetime.now().strftime("%H:%M:%S"), "event": text}

    def _last_type(self):
        return self.incidents[0]["type"] if self.incidents else ""

    def _close(self, incident, status):
        incident["status"] = status
        incident["history"].append(self._h(status))
        self.active = None
        self._save()

    def _load(self):
        try:
            with open(_path(config.INCIDENTS_FILE), "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    # Anything loaded from disk is a past incident; never
                    # resume escalation across restarts.
                    for inc in data:
                        if inc.get("status") == "ACTIVE":
                            inc["status"] = "CANCELLED - program restarted"
                    return data
        except FileNotFoundError:
            pass
        except Exception as exc:
            print(f"[alert] Could not read incidents.json: {exc}")
        return []

    def _save(self):
        try:
            with open(_path(config.INCIDENTS_FILE), "w", encoding="utf-8") as fh:
                json.dump(self.incidents[:200], fh, indent=2)
        except Exception as exc:
            print(f"[alert] Could not write incidents.json: {exc}")
