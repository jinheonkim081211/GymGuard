"""
config.py
=========
Central place for all tunable thresholds and settings for the
Gym Safety Monitor / GymGuardian MVP.

Everything that affects detection sensitivity lives here so you can tune
the system WITHOUT touching the detection logic in detector.py.

Tip for demos:
  - If you get too many false alarms, RAISE the thresholds
    (e.g. FALL_SPEED_THRESHOLD, MOTIONLESS_DURATION).
  - If real falls are missed, LOWER them.
"""

# ---------------------------------------------------------------------------
# Camera / video settings
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0          # 0 = default built-in webcam. Try 1, 2... for external cams.
FRAME_WIDTH = 960         # Requested capture width (camera may override).
FRAME_HEIGHT = 540        # Requested capture height.
FLIP_HORIZONTAL = True    # Mirror the image (webcam only; --video files are not flipped).

# ---------------------------------------------------------------------------
# MediaPipe Pose settings
# ---------------------------------------------------------------------------
# Model complexity: 0 = fastest/least accurate, 1 = balanced, 2 = most accurate.
POSE_MODEL_COMPLEXITY = 1
# Minimum confidence for the pose to be considered "detected".
POSE_DETECTION_CONFIDENCE = 0.5
# Minimum confidence for landmark tracking between frames.
POSE_TRACKING_CONFIDENCE = 0.5
# Minimum visibility for a single landmark to be trusted in our math.
LANDMARK_VISIBILITY_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Fall / collapse detection thresholds
# ---------------------------------------------------------------------------
# --- Signal A: Sudden vertical drop ---------------------------------------
# We track the body center (normalized 0.0 top -> 1.0 bottom of frame).
# "Speed" = how far the center dropped per second (in normalized units).
# 0.55 means the center fell through 55% of the frame height in one second.
FALL_SPEED_THRESHOLD = 0.55      # normalized units / second
# Over how many seconds we measure the drop (short window = "sudden").
FALL_SPEED_WINDOW = 0.5          # seconds

# --- Signal B: Body orientation (torso angle) -----------------------------
# Torso angle measured from vertical (0 deg = standing upright,
# 90 deg = lying flat / horizontal). Above this = "horizontal".
TORSO_HORIZONTAL_ANGLE = 55.0    # degrees from vertical

# --- Signal C: Low posture / floor-level position -------------------------
# If the body center sits below this fraction of the frame, the person is
# considered "low" (near the floor). 0.60 = lower 40% of the frame.
LOW_POSTURE_Y_THRESHOLD = 0.60   # normalized (0 top -> 1 bottom)

# --- Signal D: Motionlessness after collapse ------------------------------
# Motion is measured per SECOND (frame-rate independent), so the threshold
# means the same thing on a 15 FPS laptop and a 30 FPS webcam. Average
# landmark movement (normalized frame units / second) below this = "motionless".
MOTIONLESS_THRESHOLD = 0.30      # normalized movement per SECOND
# How long the person must stay motionless (after a suspected fall) to alert.
MOTIONLESS_DURATION = 3.0        # seconds

# ---------------------------------------------------------------------------
# Activity recognition (the "what is the system thinking" panel)
# ---------------------------------------------------------------------------
# The classifier looks at a short sliding window of body motion and labels the
# activity (squats, push-ups, burpees, bench press, walking, ...). Recognized
# rhythmic exercise also SUPPRESSES false fall-suspicion.
ACTIVITY_WINDOW = 4.0            # seconds of motion history to analyze
# "Rhythmic" = the body center reverses direction at least this many times...
RHYTHM_MIN_REVERSALS = 2
# ...with bounces at least this big (normalized frame heights).
RHYTHM_MIN_AMPLITUDE = 0.05
# Burpees: big vertical bounces AND the torso swings between upright/flat.
BURPEE_MIN_AMPLITUDE = 0.18      # vertical bounce size
BURPEE_TORSO_SWING = 45.0        # torso-angle range (degrees) within the window
# Bench/floor press: body still + horizontal, but wrists pump up and down.
WRIST_RHYTHM_AMPLITUDE = 0.04    # wrist bounce size (relative to shoulders)
# Walking: how much horizontal travel across the frame counts as walking.
WALK_X_RANGE = 0.12
# After rhythmic exercise is recognized, ignore posture-only suspicion for
# this long (prevents "burpee rest position" from looking like a collapse).
EXERCISE_SUPPRESS_TIME = 2.0     # seconds

# ---------------------------------------------------------------------------
# Barbell ENTRAPMENT detection ("bar-person-duration logic")
# ---------------------------------------------------------------------------
# The differentiating scenario: a lifter pinned under the bar. There is no
# fall — nothing drops — so this is its OWN detection path, inferred from
# POSE ONLY (no barbell object detection). The logic:
#   a pressing set was recently active, then the wrist rhythm STOPS while
#   the wrists stay AT CHEST LEVEL and the torso stays horizontal.
# A failed rep resolves in seconds (racking or rolling the bar off);
# entrapment is wrists pinned + DURATION.
#
# How long after the last recognized pressing (ACT_BENCH) the "pressing
# session" memory stays active:
ENTRAPMENT_MEMORY_TIME = 10.0    # seconds
# "At chest level": wrist height within this band around the shoulder line
# (normalized frame units)...
ENTRAPMENT_WRIST_BAND = 0.10
# ...AND wrists within this 2D distance of the chest, in torso-lengths.
# (Kills the false positive of resting with arms at the sides, which for a
# lying person has the same wrist HEIGHT as a pinned bar.)
ENTRAPMENT_CHEST_RADIUS = 0.8    # torso-lengths from the chest point
# Wrist movement below this (normalized units / SECOND) = rhythm has stopped.
ENTRAPMENT_WRIST_STILL = 0.10
# Overall body motion must stay below this "struggle threshold". Deliberately
# GENEROUS: a pinned lifter may tremble or kick, and that must still count.
# Only big whole-body movement (sitting up, rolling away) breaks it.
ENTRAPMENT_STRUGGLE_MOTION = 1.0  # normalized movement per SECOND
# The pinned condition must be sustained this long before we alert.
ENTRAPMENT_CONFIRM_DURATION = 4.0  # seconds
# Recovery: the pinned condition must stay BROKEN this long (wrists moved
# away from chest / person sat up) before we cancel the incident.
ENTRAPMENT_RELEASE_TIME = 2.0    # seconds

# ---------------------------------------------------------------------------
# "Down too long" detection (seizures, slow collapses)
# ---------------------------------------------------------------------------
# A convulsing person is never motionless, and a slow faint has no fast drop.
# Backstop: if someone is flat/horizontal for this long without doing a
# recognized exercise, alert regardless of motion.
DOWN_TOO_LONG_DURATION = 45.0    # seconds
# NOTE: this rule requires a HORIZONTAL body, not merely "low in frame" —
# sitting on the floor to rest or stretch is normal in a gym and must not
# count as "down". (Seated fainting is a known limitation; see README.)

# ---------------------------------------------------------------------------
# Person-lost-after-incident detection (occlusion / falling out of frame)
# ---------------------------------------------------------------------------
# Pose estimation often fails on fully prone bodies, and a person can fall
# behind equipment — the fall itself can erase the detection. So:
# How many CONSECUTIVE frames without a pose before the person is "gone"
# (brief single-frame dropouts are ignored and state is kept).
PERSON_LOST_FRAMES = 15
# If the person vanishes within this long after a fast-drop frame (or while
# state is Suspicious), we treat the disappearance itself as suspicious.
VANISH_MEMORY_TIME = 3.0         # seconds
# If they do not reappear upright within this long, alert.
VANISH_ALERT_TIMEOUT = 10.0      # seconds

# ---------------------------------------------------------------------------
# State machine / false-positive reduction
# ---------------------------------------------------------------------------
# A single suspicious frame is NOT enough. Suspicious conditions must persist
# for this long before we move from NORMAL -> SUSPICIOUS.
SUSPICIOUS_PERSIST_TIME = 0.4    # seconds
# After a suspected fall, we watch for motionlessness for up to this long.
# If the person gets up (moves) within this window, we cancel the alert.
FALL_CONFIRM_WINDOW = 6.0        # seconds
# After firing an alert, ignore new alerts for this long (anti-spam).
# NOTE: an alert that comes due during the cooldown is QUEUED, not dropped —
# it fires the moment the cooldown expires if the person is still down.
ALERT_COOLDOWN = 15.0            # seconds

# ---------------------------------------------------------------------------
# Staged escalation (the alert-routing story of the pitch)
# ---------------------------------------------------------------------------
# Stage 0 "Voice check": the computer asks "Are you OK?" out loud and waits.
# If the person starts moving again within this window, the incident is
# cancelled and logged as a recovery — no staff alarm, no spam.
STAGE0_RESPONSE_WINDOW = 10.0    # seconds
# Motion above this (normalized units / SECOND) during the voice-check window
# counts as "responded". Set well ABOVE tremor level so a struggling pinned
# lifter is NOT mistaken for a recovery (entrapment never cancels on motion —
# only on the wrists actually coming free; see detector.py).
RECOVERY_MOTION_THRESHOLD = 0.9
# Stage 1 "Staff alert": loud alarm + desktop notification + dashboard red.
# If still no response this long after Stage 1, go to Stage 2:
STAGE1_RESPONSE_WINDOW = 30.0    # seconds
# Stage 2 "Emergency contact" is SIMULATED ONLY: displayed and logged, never
# actually dialed/sent. This prototype must never contact real services.
ENABLE_VOICE_CHECK = True        # speak "Are you OK?" via macOS `say`
VOICE_CHECK_TEXT = "Are you OK?"

# ---------------------------------------------------------------------------
# Web dashboard (GymGuardian) — local network only, no video ever served
# ---------------------------------------------------------------------------
DASHBOARD_ENABLED = True
DASHBOARD_HOST = "0.0.0.0"       # LAN-viewable (e.g. from a phone on the same Wi-Fi)
DASHBOARD_PORT = 5000
# Incident history file (survives restarts). Text/JSON only — no images.
INCIDENTS_FILE = "incidents.json"

# ---------------------------------------------------------------------------
# Alert channels & logging
# ---------------------------------------------------------------------------
ENABLE_SOUND_ALERT = True        # Play a beep/alarm sound at Stage 1.
ENABLE_CONSOLE_ALERT = True      # Print a big banner to the console.
ENABLE_DESKTOP_NOTIFICATION = True  # Try an OS desktop notification (best effort).
LOG_FILE = "alerts.log"          # Timestamped alert log (created next to main.py).
EVENTS_CSV = "events.csv"        # One row per state transition / alert / stage.
# Alert wording deliberately says "possible" — this system does NOT diagnose.
ALERT_MESSAGE = "Possible fall or injury detected."
ALERT_MESSAGE_ENTRAPMENT = "Possible barbell entrapment detected."
ALERT_MESSAGE_DOWN_LONG = "Person down for extended time."
ALERT_MESSAGE_VANISHED = "Person disappeared after suspected fall - check blind spot."

# ---------------------------------------------------------------------------
# Debug / tuning display
# ---------------------------------------------------------------------------
# Show live metrics (drop speed, torso angle, motion) on screen so you can
# tune the thresholds above while watching real movements.
SHOW_DEBUG_METRICS = True
# Also print those metrics to the console every N frames (0 = never).
DEBUG_PRINT_EVERY = 0
