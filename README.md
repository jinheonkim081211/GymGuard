# GymGuardian 🛡️ — Gym Safety Monitor (MVP)

A **privacy-preserving**, fully **on-device** prototype that watches a gym
through an ordinary webcam and escalates when someone **appears** to fall,
collapse, or get **pinned under a barbell**.

> ⚠️ **This is an MVP demonstration prototype, not a medical device.**
> It does **not** diagnose injuries. Alerts say *"Possible fall or injury
> detected"* / *"Possible barbell entrapment detected"* — prompts for a human
> to check, nothing more. The final escalation stage is **SIMULATED**: this
> prototype never contacts real emergency services.

Built for a youth entrepreneurship competition: a clear, working,
explainable demo.

---

## ✨ What it does

- **Live pose monitoring** (MediaPipe Pose, 33 landmarks, on-device).
- **Two incident types**, detected by explainable rules:
  - **FALL** — sudden drop / collapse / motionless on the floor / down too
    long / vanished right after a fall.
  - **ENTRAPMENT** — a lifter pinned under the bar (*the differentiating
    feature*), inferred from **pose only** — no barbell detection needed.
- **Activity recognition** (squats, push-ups, burpees, bench press, walking…)
  with a live **SYSTEM VIEW** reasoning panel — the system narrates exactly
  what it is thinking, including confirmation countdowns.
- **Staged escalation** with voice check, staff alarm, and a *simulated*
  emergency stage. Acknowledge from the keyboard or a phone.
- **GymGuardian dashboard** — dark web UI on your local network showing
  status + incident log (never video).
- **Session statistics** — events.csv + a quit-time summary: the numbers
  behind a measured false-alarm rate.
- **Video-file input & demo recording** for building your pitch video.

### 🔒 Privacy by design
- **All processing is local.** No cloud, no uploads, no external APIs.
- **No footage is ever stored** in normal operation — only text logs
  (`alerts.log`, `events.csv`, `incidents.json`). `--record` exists solely
  for making demo videos and is **opt-in**.
- **No facial recognition, no identity tracking** — motion and posture only.
- The dashboard serves **status text only**, never frames.

---

## 📁 Project structure

```
gym_safety_monitor/
├── main.py            # Webcam/video loop, drawing, escalation wiring, CLI
├── detector.py        # Pose signals, activity classifier, state machine
├── alert.py           # Alert channels + staged IncidentManager
├── dashboard.py       # GymGuardian Flask dashboard (local network)
├── config.py          # ALL tunable thresholds, with comments
├── requirements.txt   # Dependencies (mediapipe pinned to 0.10.14!)
└── README.md          # This file
```

---

## 🛠️ Installation & run

```bash
cd gym_safety_monitor
python3.11 -m venv venv                # 3.9–3.12 work; mediapipe is pinned
source venv/bin/activate               # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py                         # live webcam
```

Other modes:

```bash
python main.py --video demo_clip.mp4   # process a recorded file instead
python main.py --record out.mp4        # save ANNOTATED demo footage (opt-in)
```

**Keys in the video window:** `q` quit · `a` acknowledge the active incident.

**Dashboard:** open `http://localhost:5000` on the same machine, or
`http://<computer-IP>:5000` from a phone on the same Wi-Fi. If Flask isn't
installed the monitor still runs — the dashboard just stays off.

---

## 🧠 How detection works

Every frame, four simple signals come from the shoulders/hips/wrists:

| Signal | Question it answers |
|---|---|
| Drop speed | Is the body falling fast? (normalized height/sec) |
| Torso angle | Is the torso horizontal? (0° standing, 90° flat) |
| Body height | Is the body low in the frame (near the floor)? |
| Motion /sec | Has the person stopped moving? (frame-rate independent) |

An **activity classifier** watches a 4-second window: *exercise is rhythmic*
(the body — or for pressing, the wrists — bounces repeatedly), while an
incident is *one event followed by stillness*. Recognized exercise
suppresses false suspicion; every conclusion is shown live in the SYSTEM
VIEW panel.

### FALL route

```
NORMAL ──(sudden drop OR collapsed posture, persisting)──▶ SUSPICIOUS
SUSPICIOUS ──(on floor + motionless 3s)──▶ POSSIBLE FALL DETECTED
```
Backstops: **down too long** (flat 45 s without exercising — covers seizures,
which are never "motionless") and **person lost** (vanishing right after a
suspected fall — covers falling behind equipment).

### ENTRAPMENT route — "bar-person-duration logic"

A pinned lifter never falls: nothing drops, and they are *on a bench,
mid-frame*. So entrapment is its own path:

```
pressing set recognized (wrists pumping rhythmically, torso flat)
        ▼
wrist rhythm STOPS while wrists stay AT CHEST LEVEL, torso stays flat
        ▼  sustained ENTRAPMENT_CONFIRM_DURATION (4 s)
POSSIBLE ENTRAPMENT DETECTED
```

The insight: a **failed rep resolves in seconds** — you rack the bar or roll
it off, and the wrists leave chest level. **Wrists pinned + duration** is
what distinguishes entrapment. Trembling or kicking does NOT cancel it
(struggle is expected); it cancels only when the wrists actually come free
for a couple of seconds.

### Staged escalation

```
FALL or ENTRAPMENT confirmed
   │
   ▼
Stage 0  VOICE CHECK      "Are you OK?" (macOS `say`; beep fallback)
   │        └─ person moves again within 10 s → CANCELLED (recovered), logged
   ▼  no response
Stage 1  STAFF ALERT      loud alarm + desktop notification + dashboard red
   ▼  no response after 30 s
Stage 2  EMERGENCY (SIMULATED)   displayed + logged ONLY — never auto-dialed
```

`a` key or the dashboard **Acknowledge** button stops escalation at any
stage. A cooldown stops alert spam; an alert that comes due during the
cooldown is queued, never dropped. Note: for ENTRAPMENT, stage-0 motion does
**not** count as recovery (struggling under a bar is motion!) — only the
wrists coming free cancels it.

---

## 🎛️ Key config knobs (config.py has ALL of them, commented)

- `FALL_SPEED_THRESHOLD`, `TORSO_HORIZONTAL_ANGLE`, `LOW_POSTURE_Y_THRESHOLD`,
  `MOTIONLESS_THRESHOLD` (per **second**), `MOTIONLESS_DURATION`
- `ENTRAPMENT_CONFIRM_DURATION`, `ENTRAPMENT_WRIST_BAND`,
  `ENTRAPMENT_WRIST_STILL`, `ENTRAPMENT_MEMORY_TIME`, `ENTRAPMENT_RELEASE_TIME`
- `STAGE0_RESPONSE_WINDOW`, `STAGE1_RESPONSE_WINDOW`, `RECOVERY_MOTION_THRESHOLD`
- `DOWN_TOO_LONG_DURATION`, `ALERT_COOLDOWN`
- `DASHBOARD_PORT`, `DASHBOARD_ENABLED`

The on-screen metrics panel shows each live value next to its threshold so
you can tune while watching real movements.

---

## 🧪 How to test it safely (staged movements)

**Never use real weight for the entrapment test. Use a broomstick or empty
hands.** Keep a mat nearby; don't actually fall.

1. **Baseline** — stand, walk, wave, squats, push-ups, burpees. Status stays
   green; the panel shows the recognized exercise and "suspicion suppressed".
2. **Staged fall** — slowly lie down and hold still ~3 s → FALL confirms →
   the computer asks *"Are you OK?"* → **start moving within 10 s** → watch
   the incident cancel as "recovered after voice check" (logged, no alarm).
3. **Full escalation** — repeat but stay still: Stage 1 alarm at 10 s,
   simulated Stage 2 at 40 s. Acknowledge with `a` or from your phone.
4. **Entrapment (safe!)** — lie on a bench or the floor, pump a broomstick
   (or empty fists) above your chest for a few reps — panel shows
   *Bench / Floor Press* — then **hold it still at chest level ~4 s** →
   ENTRAPMENT alert (magenta, its own message). Then push the stick up /
   sit up → "wrists free" recovery is logged.
5. **Racking control test** — same as 4, but after the reps move your hands
   up past your face / to your sides: NO alert (that's the rack).
6. Watch false alarms in `events.csv` and the quit-time session summary —
   tune one threshold at a time.

---

## ⚠️ Limitations

- **Single person only** — one athlete in frame; a busy gym needs the
  multi-camera / multi-person roadmap.
- **Camera angle matters** — the entrapment logic wants a roughly side-on
  view of the bench; "low in frame" depends on camera height. Re-tune per
  placement.
- **Lighting & occlusion** degrade pose quality (the person-lost state
  catches the worst case, but not all).
- **Look-alike poses**: long floor stretches can resemble "down too long";
  seated fainting (upright, on a bench) is currently invisible.
- **Entrapment is inferred from pose, not the bar** — a person just holding
  a stick perfectly still on their chest for 4 s triggers it by design
  (that's the demo!).
- **Not medical.** An alert means "a human should look," nothing more.

## 🚀 Future improvements

- Multiple cameras and multi-person pose tracking for real gym floors.
- A learned fall/entrapment classifier on skeleton sequences, validated on
  real gym-movement datasets (rules stay as the explainable fallback).
- Real staff push notifications (kept out of the MVP to preserve the
  no-external-services guarantee) and emergency-contact workflows.
- Zone awareness (benches, racks, mat areas) for per-zone thresholds.

---

## 🧾 License / intent

Educational prototype for a youth entrepreneurship competition. Use
responsibly; not a substitute for trained supervision or a spotter.
