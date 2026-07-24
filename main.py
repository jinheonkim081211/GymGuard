"""
main.py
=======
Entry point for the Gym Safety Monitor / GymGuardian MVP.

Responsibilities:
  - Open the webcam (default) or a video file (--video clip.mp4).
  - Run each frame through the FallDetector.
  - Draw pose landmarks, bounding box, status text, SYSTEM VIEW panel,
    and the escalation banner.
  - Drive the staged escalation (voice check -> staff alert -> SIMULATED
    emergency) via the IncidentManager.
  - Serve the GymGuardian dashboard on the local network.
  - Optionally record the ANNOTATED output with --record out.mp4 (opt-in
    demo recording only; normal operation never records anything).
  - Log every state transition / alert / stage to events.csv and print a
    session summary on quit.

Keys in the video window:  q = quit,  a = acknowledge the active incident.

Run:
    python main.py                     # live webcam
    python main.py --video demo.mp4    # process a recorded clip
    python main.py --record out.mp4    # save annotated demo footage

PRIVACY: All processing happens locally. No video is stored or uploaded
(unless YOU pass --record for demo footage). Logs contain text only.
"""

import argparse
import sys
import time

import cv2

import config
import alert
import dashboard
from detector import (
    FallDetector,
    INCIDENT_STATES,
    STATE_NORMAL,
    STATE_SUSPICIOUS,
    STATE_FALL,
    STATE_ENTRAPMENT,
    STATE_NO_PERSON,
    STATE_PERSON_LOST,
)


# Color per state (BGR) for the on-screen status banner.
STATE_COLORS = {
    STATE_NO_PERSON: (150, 150, 150),   # gray
    STATE_NORMAL: (0, 200, 0),          # green
    STATE_SUSPICIOUS: (0, 165, 255),    # orange
    STATE_PERSON_LOST: (0, 165, 255),   # orange — vanished after an incident
    STATE_FALL: (0, 0, 255),            # red
    STATE_ENTRAPMENT: (255, 0, 255),    # magenta — pinned under the bar
}

# Colors for the reasoning lines in the SYSTEM VIEW panel (BGR).
THINKING_COLORS = {
    "good": (0, 210, 0),        # green  — condition looks safe
    "info": (200, 200, 200),    # gray   — neutral observation
    "warn": (0, 165, 255),      # orange — worth watching
    "bad": (0, 0, 255),         # red    — incident evidence
}


def parse_args():
    p = argparse.ArgumentParser(
        description="GymGuardian — local gym safety monitor (MVP)")
    p.add_argument("--video", metavar="PATH", default=None,
                   help="process a video file instead of the webcam")
    p.add_argument("--record", metavar="OUT.mp4", default=None,
                   help="save the ANNOTATED output to a video file "
                        "(opt-in demo recording; normal runs never record)")
    return p.parse_args()


def open_source(video_path):
    """
    Open the webcam or a video file. Returns (capture, is_file, source_fps)
    or (None, ..., ...) on failure.
    """
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap or not cap.isOpened():
            print(f"[main] ERROR: Could not open video file: {video_path}")
            return None, True, 30.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1 or fps > 240:
            fps = 30.0
        return cap, True, fps

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap or not cap.isOpened():
        print(f"[main] ERROR: Could not open camera index {config.CAMERA_INDEX}.")
        print("       - Is a webcam connected and not used by another app?")
        print("       - Try changing CAMERA_INDEX in config.py (0, 1, 2 ...).")
        return None, False, 30.0
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap, False, 30.0


def draw_side_panel(frame, result):
    """
    Draw the "SYSTEM VIEW" panel on the right: the recognized activity plus
    the live reasoning lines that show exactly what the system is thinking.
    """
    h, w = frame.shape[:2]
    panel_w = 330
    x0 = w - panel_w

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, 48),
                  (w, 48 + 46 + 24 * (len(result["thinking"]) + 1) + 16),
                  (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "SYSTEM VIEW", (x0 + 12, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, result["activity"], (x0 + 12, 96),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    y = 122
    for text, level in result["thinking"]:
        color = THINKING_COLORS.get(level, (255, 255, 255))
        cv2.putText(frame, text, (x0 + 12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        y += 24
    return frame


def draw_overlay(frame, result, fps, alert_text, escalation_line, ack_flash):
    """Status banner, metrics, escalation line, and warning border."""
    state = result["state"]
    color = STATE_COLORS.get(state, (255, 255, 255))
    h, w = frame.shape[:2]

    # --- Status banner across the top ---
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(frame, f"STATUS: {state}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:4.1f} FPS", (w - 130, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    # --- Debug metrics (bottom-left) so you can tune thresholds live ---
    if config.SHOW_DEBUG_METRICS and result["metrics"]["person"]:
        m = result["metrics"]
        lines = [
            f"drop_speed : {m['drop_speed']:.2f}  (thr {config.FALL_SPEED_THRESHOLD})",
            f"torso_angle: {m['torso_angle']:.1f}  (thr {config.TORSO_HORIZONTAL_ANGLE})",
            f"center_y   : {m['center_y']:.2f}  (thr {config.LOW_POSTURE_Y_THRESHOLD})",
            # motion is per SECOND (frame-rate independent).
            f"motion/s   : {m['motion']:.3f}  (thr {config.MOTIONLESS_THRESHOLD})",
        ]
        y0 = h - 20 - (len(lines) - 1) * 22
        for i, text in enumerate(lines):
            y = y0 + i * 22
            cv2.putText(frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # --- Help text (bottom-right) ---
    cv2.putText(frame, "q = quit   a = acknowledge", (w - 260, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # --- Escalation status / acknowledgment feedback ---
    if ack_flash:
        cv2.putText(frame, "ALERT ACKNOWLEDGED", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 210, 0), 2, cv2.LINE_AA)
    elif escalation_line:
        cv2.putText(frame, escalation_line, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA)

    # --- Big flashing warning while an incident is active ---
    if state in (STATE_FALL, STATE_ENTRAPMENT, STATE_PERSON_LOST):
        border = STATE_COLORS[state]
        if int(time.time() * 2) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border, 12)
        cv2.putText(frame, alert_text or config.ALERT_MESSAGE, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, border, 2, cv2.LINE_AA)

    return frame


def escalation_status_line(manager, now):
    """
    One-line escalation status for the on-screen overlay. Reads the active
    incident without the lock — display only, a stale frame is harmless.
    """
    inc = manager.active
    if inc is None:
        return None
    elapsed = now - inc["t0"]
    if inc["stage"] == 0:
        remain = max(0.0, config.STAGE0_RESPONSE_WINDOW - elapsed)
        return f"Stage 0 voice check - waiting {remain:.0f}s   (press 'a' to acknowledge)"
    if inc["stage"] == 1:
        remain = max(0.0, config.STAGE0_RESPONSE_WINDOW
                     + config.STAGE1_RESPONSE_WINDOW - elapsed)
        return f"Stage 1 STAFF ALERT - escalating in {remain:.0f}s   (press 'a')"
    return "Stage 2 EMERGENCY (SIMULATED) - press 'a' to acknowledge"


def main():
    args = parse_args()

    print("[main] Starting GymGuardian (MVP)...")
    print("[main] Privacy: video is processed locally and never uploaded.")
    if args.record:
        print(f"[main] DEMO RECORDING ON: annotated output -> {args.record}")
    print("[main] Keys: 'q' quit, 'a' acknowledge.\n")

    cap, is_file, src_fps = open_source(args.video)
    if cap is None:
        sys.exit(1)

    detector = FallDetector()
    manager = alert.IncidentManager()

    # --- Dashboard (optional, degrades gracefully without Flask) ---
    session_start = time.time()
    live_status = {"state": STATE_NO_PERSON, "activity": "-", "fps": 0.0}

    def get_status():
        return {"state": live_status["state"],
                "activity": live_status["activity"],
                "fps": live_status["fps"],
                "uptime_s": time.time() - session_start}

    if config.DASHBOARD_ENABLED:
        dashboard.start_dashboard(manager, get_status)

    # --- Optional annotated demo recording (--record) ---
    recorder = None   # created lazily once we know the frame size

    # --- Session statistics ---
    frames = 0
    suspicious_episodes = 0
    fps = 0.0
    prev_time = time.time()
    last_alert_text = None
    ack_flash_until = 0.0
    frame_index = 0

    # Pace file playback at the file's own FPS; webcam runs free.
    wait_ms = max(1, int(1000.0 / src_fps)) if is_file else 1

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if is_file:
                    print("[main] End of video file.")
                    break
                print("[main] WARNING: Dropped a frame from the camera.")
                time.sleep(0.05)
                continue
            frame_index += 1
            frames += 1

            # Webcam is mirrored for a natural feel; files are played as-is.
            if not is_file and config.FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            # THE clock: video files use their own timestamps so every
            # duration-based rule (stillness, entrapment, stages) behaves
            # identically to a live run; webcam uses the wall clock.
            if is_file:
                msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                now = msec / 1000.0 if msec and msec > 0 else frame_index / src_fps
            else:
                now = time.time()

            # ---- Core detection ----
            result = detector.process(frame, now=now)

            # ---- Log state transitions to events.csv ----
            while detector.transitions:
                t_tr, from_s, to_s = detector.transitions.popleft()
                alert.log_csv_event("state_change", outcome=f"{from_s} -> {to_s}",
                                    metrics=result["metrics"])
                if to_s == STATE_SUSPICIOUS:
                    suspicious_episodes += 1

            # ---- New incident? Start Stage 0 escalation ----
            if result["alert"]:
                last_alert_text = result["alert_message"]
                manager.on_alert(now, result["incident_type"],
                                 result["alert_message"], result["metrics"])

            # ---- Drive the escalation clock ----
            actions = manager.update(now, motion=result["metrics"]["motion"],
                                     recovered_reason=result["recovered"])
            for kind, why in actions:
                if kind == "cancelled" and detector.state in INCIDENT_STATES:
                    # Voice-check recovery: the manager cancelled, so snap
                    # the detector back to NORMAL for a fresh start too.
                    detector.reset_incident(now)

            if result["state"] == STATE_NORMAL and manager.active is None:
                last_alert_text = None

            # ---- Feed the dashboard ----
            live_status["state"] = result["state"]
            live_status["activity"] = result["activity"]
            live_status["fps"] = fps

            # ---- Draw everything ----
            detector.draw(frame, result)
            draw_overlay(frame, result, fps, last_alert_text,
                         escalation_status_line(manager, now),
                         time.time() < ack_flash_until)
            draw_side_panel(frame, result)

            # ---- Demo recording (opt-in only; see privacy note) ----
            if args.record:
                if recorder is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    rec_fps = src_fps if is_file else 20.0
                    recorder = cv2.VideoWriter(args.record, fourcc, rec_fps, (w, h))
                    if not recorder.isOpened():
                        print("[main] WARNING: could not open --record output; "
                              "recording disabled.")
                        recorder = False   # sentinel: don't retry every frame
                if recorder:
                    recorder.write(frame)

            cv2.imshow("GymGuardian - Gym Safety Monitor", frame)

            # ---- FPS update ----
            t2 = time.time()
            dt = t2 - prev_time
            prev_time = t2
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # ---- Keyboard ----
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord("q"):
                print("[main] Quit requested by user.")
                break
            if key == ord("a"):
                if manager.acknowledge("keyboard"):
                    ack_flash_until = time.time() + 4.0
                    detector.reset_incident(now)

    except KeyboardInterrupt:
        print("\n[main] Interrupted (Ctrl+C).")
    finally:
        detector.close()
        cap.release()
        if recorder:
            recorder.release()
            print(f"[main] Annotated demo video saved to {args.record}")
        cv2.destroyAllWindows()

        # ---- Session summary: the numbers behind a false-alarm rate ----
        c = manager.counters
        duration = time.time() - session_start
        print("\n" + "=" * 52)
        print("  SESSION SUMMARY")
        print(f"  Duration            : {duration/60:.1f} min ({frames} frames)")
        print(f"  Suspicious episodes : {suspicious_episodes}")
        print(f"  Incidents confirmed : {c['incidents']}")
        print(f"    voice-check recoveries : {c['voice_recoveries']}")
        print(f"    other cancels          : {c['other_cancels']}")
        print(f"    acknowledged           : {c['acknowledged']}")
        print(f"    reached Stage 1 / 2    : {c['stage1_reached']} / {c['stage2_reached']}")
        print(f"  Full event log      : {config.EVENTS_CSV}, incidents: "
              f"{config.INCIDENTS_FILE}")
        print("=" * 52)


if __name__ == "__main__":
    main()
