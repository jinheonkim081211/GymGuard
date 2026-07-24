"""
detector.py
===========
Pose processing, activity recognition, and incident detection for the
Gym Safety Monitor / GymGuardian.

HOW IT WORKS (high level)
-------------------------
1. Each frame is passed to MediaPipe Pose, which returns 33 body landmarks
   (shoulders, hips, wrists, etc.) as normalized (x, y) points in [0, 1]
   (0,0 = top-left, 1,1 = bottom-right).

2. From those landmarks we compute simple, human-understandable signals:
     A. Vertical drop speed  -> is the body falling fast?
     B. Torso angle          -> is the torso horizontal (lying down)?
     C. Low posture          -> is the body near the floor (bottom of frame)?
     D. Motionlessness        -> has the person stopped moving? (per SECOND,
                                 so behavior is the same at 15 or 30 FPS)

3. An ACTIVITY CLASSIFIER watches a few seconds of motion history and labels
   what the person seems to be doing (squats, push-ups, burpees, bench press,
   walking, standing, lying still...). Exercise is RHYTHMIC; a fall is ONE
   drop followed by stillness. Recognized exercise suppresses false suspicion.

4. A STATE MACHINE combines the signals over time so no single noisy frame
   can trigger an alert. There are TWO incident types and several routes:

   FALL incidents:
     a) Classic fall:   sudden drop / collapsed posture -> down + motionless
     b) Down too long:  flat on the floor for a long time without exercising
                        (covers seizures/writhing, which are never motionless)
     c) Vanished:       person disappears right after a suspected fall and
                        does not come back (covers occlusion / blind spots)

   ENTRAPMENT incidents ("bar-person-duration logic", pose only — no barbell
   object detection):
     d) A pressing set was recently active (ACT_BENCH), then the wrist
        rhythm STOPS while the wrists stay pinned at chest level and the
        torso stays horizontal, sustained for ENTRAPMENT_CONFIRM_DURATION.
        A failed rep resolves in seconds (racking or rolling the bar off);
        entrapment is wrists pinned + DURATION.

All timestamps flow through a single `now` value passed to process(), so the
detector works identically on a live webcam and on recorded video files.

This is intentionally rule-based and lightweight so it runs on a laptop and
is easy to explain in a demo. It does NOT diagnose injury.
"""

import time
from collections import deque

import numpy as np
import cv2
import mediapipe as mp

import config


# ---------------------------------------------------------------------------
# System states (shown on screen and used by the state machine)
# ---------------------------------------------------------------------------
STATE_NO_PERSON = "No Person"
STATE_NORMAL = "Normal"
STATE_SUSPICIOUS = "Suspicious Movement"
STATE_FALL = "Possible Fall Detected"
STATE_ENTRAPMENT = "Possible Entrapment Detected"
STATE_PERSON_LOST = "Person Lost After Incident"

# States that mean "an incident is in progress".
INCIDENT_STATES = (STATE_FALL, STATE_ENTRAPMENT, STATE_PERSON_LOST)

# ---------------------------------------------------------------------------
# Activity labels (shown in the "system thinking" panel)
# ---------------------------------------------------------------------------
ACT_STANDING = "Standing / Idle"
ACT_WALKING = "Walking / Moving"
ACT_ACTIVE = "Active Movement"
ACT_SQUATS = "Squats (exercise)"
ACT_PUSHUPS = "Push-ups (exercise)"
ACT_BURPEES = "Burpees (exercise)"
ACT_BENCH = "Bench / Floor Press"
ACT_FLOOR = "On Floor (moving)"
ACT_LYING = "Lying Still"
ACT_FALLING = "Falling ?!"
ACT_PINNED = "Pinned under bar ?!"

# Activities that count as deliberate rhythmic exercise (used to suppress
# false fall-suspicion while the person is clearly working out).
EXERCISE_ACTIVITIES = {ACT_SQUATS, ACT_PUSHUPS, ACT_BURPEES, ACT_BENCH}


# MediaPipe landmark indices we care about (from the Pose model).
mp_pose = mp.solutions.pose
L_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
R_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
L_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
R_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value
L_WRIST = mp_pose.PoseLandmark.LEFT_WRIST.value
R_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST.value


def _count_reversals(values, hysteresis):
    """
    Count how many times a 1-D signal changes direction, ignoring wiggles
    smaller than `hysteresis`. This is how we tell RHYTHMIC motion
    (squats, push-ups: many reversals) from a FALL (one big move, none).
    """
    if len(values) < 3:
        return 0
    reversals = 0
    direction = 0          # +1 rising, -1 falling, 0 unknown yet
    anchor = values[0]     # last extremum we committed to
    for v in values[1:]:
        if v > anchor + hysteresis:
            if direction == -1:
                reversals += 1
            direction = 1
            anchor = v
        elif v < anchor - hysteresis:
            if direction == 1:
                reversals += 1
            direction = -1
            anchor = v
        else:
            # Keep tracking the running extremum in the current direction.
            if direction == 1:
                anchor = max(anchor, v)
            elif direction == -1:
                anchor = min(anchor, v)
    return reversals


class FallDetector:
    """
    Wraps MediaPipe Pose, the activity classifier, and the incident-detection
    state machine.

    Usage:
        detector = FallDetector()
        result = detector.process(frame)          # live webcam (wall clock)
        result = detector.process(frame, now=t)   # video file (media time)
        detector.close()
    """

    def __init__(self):
        # Create the MediaPipe Pose estimator once and reuse it.
        self.pose = mp_pose.Pose(
            model_complexity=config.POSE_MODEL_COMPLEXITY,
            min_detection_confidence=config.POSE_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.POSE_TRACKING_CONFIDENCE,
        )
        self.drawing = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles

        # --- History buffers for time-based signals ---
        # (timestamp, body_center_y) for the vertical-drop signal.
        self._center_history = deque(maxlen=120)
        # Motion history for the activity classifier:
        # each entry = (t, center_y, center_x, torso_angle, wrist_rel_y or None)
        self._activity_history = deque(maxlen=300)
        # Previous frame's landmarks + timestamp for per-SECOND motion.
        self._prev_landmarks = None
        self._prev_wrist_point = None
        self._prev_sample_time = None
        self._last_motion = 0.0
        self._last_wrist_motion = None
        self._pending_wrist_delta = None

        # --- State machine bookkeeping ---
        self.state = STATE_NORMAL
        self._suspicious_since = None      # when suspicious conditions began
        self._fall_candidate_since = None  # when a possible fall was first seen
        self._motionless_since = None      # when the person became motionless
        self._last_alert_time = -1e9       # for cooldown
        self._last_exercise_time = -1e9    # last time we recognized exercise

        # An alert that comes due DURING the cooldown is queued here and
        # fired the moment the cooldown expires (never silently dropped).
        self._pending_alert_message = None

        # Entrapment ("bar-person-duration") tracking.
        self._bench_seen_until = -1e9      # pressing-session memory expires at t
        self._entrap_since = None          # when the wrists became pinned
        self._entrap_release_since = None  # when the pinned condition broke

        # "Down too long" tracking (independent of state/motion).
        self._down_since = None            # when the person went flat

        # Person-lost-after-incident tracking.
        self._missing_frames = 0           # consecutive frames without a pose
        self._last_fast_drop_time = -1e9   # last frame with a fast drop
        self._person_lost_since = None     # when the person officially vanished
        self._vanish_alerted = False       # vanish alert sent for this episode

        # Set when an incident ends because the person recovered; process()
        # hands it to main.py exactly once so the recovery can be logged.
        self._recovered_reason = None

        # State-transition log (feeds events.csv via main.py).
        self.transitions = deque(maxlen=40)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def process(self, frame, now=None):
        """
        Process one BGR frame. `now` is the timestamp in seconds: leave it
        None for live webcam (wall clock) or pass the video's own time for
        file playback so every duration-based rule stays correct.

        Returns a dict:
            {
              "state": str,
              "alert": bool,              # True only on the frame an alert fires
              "alert_message": str|None,  # which alert fired (distinct per type)
              "incident_type": str|None,  # "FALL" or "ENTRAPMENT" on alert frames
              "recovered": str|None,      # set once when an incident self-cancels
              "landmarks": mp results,    # for drawing (or None)
              "bbox": (x1, y1, x2, y2) or None,
              "metrics": {...},           # numbers for on-screen debugging
              "activity": str,            # what the person seems to be doing
              "thinking": [(text, level), ...],  # the system's reasoning,
                          # level in {"good", "info", "warn", "bad"}
            }
        """
        if now is None:
            now = time.time()

        # MediaPipe expects RGB; OpenCV gives us BGR.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False   # small perf optimization
        results = self.pose.process(rgb)

        metrics = {
            "drop_speed": 0.0,
            "torso_angle": 0.0,
            "center_y": 0.0,
            "motion": 0.0,
            "person": False,
        }

        # -----------------------------------------------------------------
        # No pose this frame. Do NOT wipe everything immediately — pose
        # estimation often fails on prone bodies, and a person can fall
        # behind equipment. Brief dropouts keep the current state; a real
        # disappearance right after an incident becomes PERSON_LOST.
        # -----------------------------------------------------------------
        if not results.pose_landmarks:
            return self._handle_no_person(now, metrics)

        landmarks = results.pose_landmarks.landmark
        metrics["person"] = True

        # --- Reappearance handling after a gap of missing frames ----------
        gap = self._missing_frames
        self._missing_frames = 0
        if gap > 0:
            # Never compare landmarks across a gap — it looks like a huge
            # instantaneous motion spike.
            self._prev_landmarks = None
            self._prev_wrist_point = None
            self._prev_sample_time = None
            if gap >= config.PERSON_LOST_FRAMES:
                # Long gap: the drop-speed / rhythm histories are stale.
                self._center_history.clear()
                self._activity_history.clear()

        # -----------------------------------------------------------------
        # Compute the raw per-frame signals from landmarks.
        # -----------------------------------------------------------------
        center_y = self._body_center_y(landmarks)
        center_x = self._body_center_x(landmarks)
        drop_speed = self._vertical_drop_speed(center_y, now)
        torso_angle = self._torso_angle(landmarks)
        motion = self._motion_amount(landmarks, now)          # per SECOND
        wrist_rel, chest_dist_ratio = self._press_signals(landmarks)
        wrist_motion = self._last_wrist_motion                # per SECOND or None

        metrics["center_y"] = center_y
        metrics["drop_speed"] = drop_speed
        metrics["torso_angle"] = torso_angle
        metrics["motion"] = motion

        bbox = self._bounding_box(landmarks, frame.shape)

        # Instantaneous conditions used by classifier and state machine.
        is_fast_drop = drop_speed >= config.FALL_SPEED_THRESHOLD
        is_horizontal = torso_angle >= config.TORSO_HORIZONTAL_ANGLE
        is_low = center_y >= config.LOW_POSTURE_Y_THRESHOLD
        is_motionless = motion <= config.MOTIONLESS_THRESHOLD
        if is_fast_drop:
            self._last_fast_drop_time = now   # remembered for vanish logic

        # -----------------------------------------------------------------
        # Activity recognition over the sliding motion-history window.
        # -----------------------------------------------------------------
        self._activity_history.append((now, center_y, center_x, torso_angle, wrist_rel))
        activity, act_features = self._classify_activity(
            now, is_fast_drop, is_horizontal, is_low, is_motionless, motion
        )
        if activity in EXERCISE_ACTIVITIES:
            self._last_exercise_time = now
        exercising = (now - self._last_exercise_time) < config.EXERCISE_SUPPRESS_TIME

        # -----------------------------------------------------------------
        # ENTRAPMENT signals ("bar-person-duration logic").
        # Step 1: remember that a pressing set was recently active.
        # -----------------------------------------------------------------
        if activity == ACT_BENCH:
            self._bench_seen_until = now + config.ENTRAPMENT_MEMORY_TIME
        bench_recent = now < self._bench_seen_until

        # Step 2: the instantaneous "pinned" condition — pressing was recent,
        # torso still horizontal, wrists AT CHEST (height band + 2D proximity,
        # so arms-at-sides rest does not count), wrist rhythm stopped, and no
        # big whole-body movement (tremor/kicking is allowed; sitting up or
        # rolling away is not).
        entrap_cond = bool(
            bench_recent
            and is_horizontal
            and wrist_rel is not None
            and abs(wrist_rel) <= config.ENTRAPMENT_WRIST_BAND
            and chest_dist_ratio is not None
            and chest_dist_ratio <= config.ENTRAPMENT_CHEST_RADIUS
            and wrist_motion is not None
            and wrist_motion <= config.ENTRAPMENT_WRIST_STILL
            and motion <= config.ENTRAPMENT_STRUGGLE_MOTION
        )

        # -----------------------------------------------------------------
        # Reappearance while in PERSON_LOST: upright cancels the episode;
        # reappearing on the floor keeps us suspicious (they may have fallen
        # behind something and crawled back into view).
        # -----------------------------------------------------------------
        if self.state == STATE_PERSON_LOST:
            if is_horizontal or is_low:
                self._set_state(now, STATE_SUSPICIOUS)
                self._fall_candidate_since = now
                self._motionless_since = None
            else:
                self._reset_fall_tracking()
                self._set_state(now, STATE_NORMAL)
                self._recovered_reason = "person reappeared upright"
            self._person_lost_since = None
            self._vanish_alerted = False
        elif self.state == STATE_NO_PERSON:
            self._set_state(now, STATE_NORMAL)

        # -----------------------------------------------------------------
        # Feed everything into the state machine.
        # -----------------------------------------------------------------
        fired_message = self._update_state_machine(now, {
            "fast_drop": is_fast_drop,
            "flat": is_horizontal,
            "low": is_low,
            "motionless": is_motionless,
            "motion": motion,
            "exercising": exercising,
            "bench_recent": bench_recent,
            "entrap_cond": entrap_cond,
        })

        # Activity override: a pinned press is its own (alarming) activity.
        pinned_time = (now - self._entrap_since) if self._entrap_since else 0.0
        if pinned_time >= config.ENTRAPMENT_CONFIRM_DURATION / 2:
            activity = ACT_PINNED

        # Hand the one-shot recovery notice to the caller (for logging).
        recovered = self._recovered_reason
        self._recovered_reason = None

        # -----------------------------------------------------------------
        # Build the human-readable "what am I thinking" reasoning lines.
        # -----------------------------------------------------------------
        thinking = self._build_thinking(
            now, drop_speed, torso_angle,
            is_fast_drop, is_horizontal, is_low, is_motionless,
            activity, act_features, exercising,
            bench_recent, entrap_cond, pinned_time,
        )

        return {
            "state": self.state,
            "alert": fired_message is not None,
            "alert_message": fired_message,
            "incident_type": self._incident_type_for(fired_message),
            "recovered": recovered,
            "landmarks": results.pose_landmarks,
            "bbox": bbox,
            "metrics": metrics,
            "activity": activity,
            "thinking": thinking,
        }

    def reset_incident(self, now=None):
        """
        Called by main.py when the escalation manager cancels an incident
        (e.g. the person responded to the voice check by moving). Returns
        the detector to NORMAL so monitoring starts fresh — if the person
        is genuinely still down, the normal rules will re-confirm within a
        few seconds (respecting the alert cooldown).
        """
        if now is None:
            now = time.time()
        if self.state in INCIDENT_STATES:
            self._reset_fall_tracking()
            self._pending_alert_message = None
            self._entrap_since = None
            self._entrap_release_since = None
            self._person_lost_since = None
            self._vanish_alerted = False
            self._set_state(now, STATE_NORMAL)

    def draw(self, frame, result):
        """Draw pose landmarks and a bounding box onto the frame (in place)."""
        if result["landmarks"] is not None:
            self.drawing.draw_landmarks(
                frame,
                result["landmarks"],
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.drawing_styles.get_default_pose_landmarks_style(),
            )
        if result["bbox"] is not None:
            x1, y1, x2, y2 = result["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        return frame

    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()

    # ---------------------------------------------------------------------
    # Missing-person handling
    # ---------------------------------------------------------------------
    def _handle_no_person(self, now, metrics):
        """Called when MediaPipe finds no pose in the current frame."""
        self._missing_frames += 1
        fired = None

        if self.state == STATE_NO_PERSON:
            thinking = [("No person in view", "info")]

        elif self._missing_frames < config.PERSON_LOST_FRAMES:
            # Brief dropout — keep ALL state and history; pose estimation
            # flickers constantly, especially on prone bodies.
            thinking = [(
                f"Tracking dropped ({self._missing_frames}/"
                f"{config.PERSON_LOST_FRAMES} frames) - holding state",
                "info",
            )]

        else:
            # The person is officially gone. Was the disappearance suspicious?
            recently_dropped = (now - self._last_fast_drop_time) <= config.VANISH_MEMORY_TIME
            if self.state == STATE_PERSON_LOST:
                pass  # already tracking the vanish episode below
            elif self.state in (STATE_SUSPICIOUS, STATE_FALL, STATE_ENTRAPMENT) \
                    or recently_dropped:
                self._set_state(now, STATE_PERSON_LOST)
                self._person_lost_since = now
                self._vanish_alerted = False
            else:
                # Calm exit (walked out of frame) — full reset, no alarm.
                self._full_reset(now)

            if self.state == STATE_PERSON_LOST:
                waited = now - (self._person_lost_since or now)
                if waited >= config.VANISH_ALERT_TIMEOUT and not self._vanish_alerted:
                    fired = self._fire_alert(now, config.ALERT_MESSAGE_VANISHED)
                    if fired:
                        self._vanish_alerted = True
                elif self._pending_alert_message:
                    fired = self._flush_pending(now)
                    if fired == config.ALERT_MESSAGE_VANISHED:
                        self._vanish_alerted = True
                thinking = [
                    ("Person lost right after incident!", "bad"),
                    (f"Waiting for reappearance "
                     f"{min(waited, config.VANISH_ALERT_TIMEOUT):.0f}/"
                     f"{config.VANISH_ALERT_TIMEOUT:.0f}s", "bad"
                     if waited >= config.VANISH_ALERT_TIMEOUT else "warn"),
                ]
                if self._vanish_alerted:
                    thinking.append(("VERDICT: check the blind spot NOW", "bad"))
            else:
                thinking = [("No person in view", "info")]

        return {
            "state": self.state,
            "alert": fired is not None,
            "alert_message": fired,
            "incident_type": self._incident_type_for(fired),
            "recovered": None,
            "landmarks": None,
            "bbox": None,
            "metrics": metrics,
            "activity": "-",
            "thinking": thinking,
        }

    def _full_reset(self, now):
        """Person calmly left the frame: clear everything, no alarm."""
        self._prev_landmarks = None
        self._prev_wrist_point = None
        self._prev_sample_time = None
        self._center_history.clear()
        self._activity_history.clear()
        self._reset_fall_tracking()
        self._entrap_since = None
        self._entrap_release_since = None
        self._down_since = None
        self._person_lost_since = None
        self._vanish_alerted = False
        self._set_state(now, STATE_NO_PERSON)

    @staticmethod
    def _incident_type_for(message):
        """Map an alert message to its incident type for the dashboard/log."""
        if message is None:
            return None
        if message == config.ALERT_MESSAGE_ENTRAPMENT:
            return "ENTRAPMENT"
        return "FALL"

    # ---------------------------------------------------------------------
    # Signal computations (the "what we measure" part)
    # ---------------------------------------------------------------------
    def _visible(self, lm):
        """True if a landmark is confident enough to use."""
        return lm.visibility >= config.LANDMARK_VISIBILITY_THRESHOLD

    def _body_center_y(self, landmarks):
        """
        Vertical center of the body = average y of shoulders and hips.
        y is normalized: 0.0 = top of frame, 1.0 = bottom (near the floor).
        """
        pts = [landmarks[i].y for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
               if self._visible(landmarks[i])]
        if not pts:
            pts = [landmarks[i].y for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)]
        return float(np.mean(pts))

    def _body_center_x(self, landmarks):
        """Horizontal center of the body (used to detect walking)."""
        pts = [landmarks[i].x for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
               if self._visible(landmarks[i])]
        if not pts:
            pts = [landmarks[i].x for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)]
        return float(np.mean(pts))

    def _press_signals(self, landmarks):
        """
        Signals for entrapment detection. Returns:
          wrist_rel        : mean wrist y RELATIVE to the shoulder line
                             (None if wrists aren't visible), and
          chest_dist_ratio : 2D distance from the mean wrist point to the
                             upper-torso "chest" point, in torso-lengths
                             (None if not computable).
        Also stages the per-second wrist-motion estimate used for the
        "wrist rhythm stopped" check.
        """
        wrists = [landmarks[i] for i in (L_WRIST, R_WRIST) if self._visible(landmarks[i])]
        if not wrists:
            self._prev_wrist_point = None
            self._last_wrist_motion = None
            self._pending_wrist_delta = None
            return None, None

        wrist_pt = np.array([float(np.mean([lm.x for lm in wrists])),
                             float(np.mean([lm.y for lm in wrists]))])
        shoulder_mid = np.array([(landmarks[L_SHOULDER].x + landmarks[R_SHOULDER].x) / 2.0,
                                 (landmarks[L_SHOULDER].y + landmarks[R_SHOULDER].y) / 2.0])
        hip_mid = np.array([(landmarks[L_HIP].x + landmarks[R_HIP].x) / 2.0,
                            (landmarks[L_HIP].y + landmarks[R_HIP].y) / 2.0])

        wrist_rel = float(wrist_pt[1] - shoulder_mid[1])

        # "Chest" = a quarter of the way from the shoulders toward the hips.
        chest_pt = shoulder_mid + 0.25 * (hip_mid - shoulder_mid)
        torso_len = float(np.linalg.norm(shoulder_mid - hip_mid))
        chest_dist_ratio = (float(np.linalg.norm(wrist_pt - chest_pt)) / torso_len
                            if torso_len > 1e-4 else None)

        # Per-second wrist motion; finished in _motion_amount() which knows dt.
        if self._prev_wrist_point is not None and self._prev_sample_time is not None:
            self._pending_wrist_delta = float(np.linalg.norm(wrist_pt - self._prev_wrist_point))
        else:
            self._pending_wrist_delta = None
        self._prev_wrist_point = wrist_pt
        return wrist_rel, chest_dist_ratio

    def _vertical_drop_speed(self, center_y, now):
        """
        Signal A: how fast the body center is moving DOWNWARD.
        Compares the current center to where it was ~FALL_SPEED_WINDOW
        seconds ago. Returns normalized frame-heights per second (>= 0).
        """
        self._center_history.append((now, center_y))
        window_start = now - config.FALL_SPEED_WINDOW
        past = None
        for t, y in self._center_history:
            if t >= window_start:
                past = (t, y)
                break
        if past is None:
            return 0.0
        dt = now - past[0]
        if dt <= 1e-3:
            return 0.0
        # Downward movement = y increases, so (current - past) is positive.
        return max(0.0, (center_y - past[1]) / dt)

    def _torso_angle(self, landmarks):
        """
        Signal B: torso angle from VERTICAL (hip midpoint -> shoulder midpoint).
            ~0 deg  = standing upright,  ~90 deg = lying flat.
        """
        sx = (landmarks[L_SHOULDER].x + landmarks[R_SHOULDER].x) / 2.0
        sy = (landmarks[L_SHOULDER].y + landmarks[R_SHOULDER].y) / 2.0
        hx = (landmarks[L_HIP].x + landmarks[R_HIP].x) / 2.0
        hy = (landmarks[L_HIP].y + landmarks[R_HIP].y) / 2.0
        dx = sx - hx
        dy = sy - hy   # y grows downward in image coordinates
        return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))

    def _motion_amount(self, landmarks, now):
        """
        Signal D: average landmark movement per SECOND. Dividing by the
        frame interval makes MOTIONLESS_THRESHOLD mean the same thing on a
        15 FPS laptop and a 30 FPS webcam.
        """
        current = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
        prev_t = self._prev_sample_time
        prev_lm = self._prev_landmarks
        self._prev_landmarks = current
        self._prev_sample_time = now

        if prev_lm is None or prev_t is None or prev_lm.shape != current.shape:
            self._last_motion = 0.0
            self._last_wrist_motion = None
            return 0.0
        dt = now - prev_t
        if dt <= 1e-3:
            return self._last_motion   # duplicate timestamp; keep last value
        disp = float(np.mean(np.linalg.norm(current - prev_lm, axis=1)))
        self._last_motion = disp / dt

        # Finish the wrist-motion computation started in _press_signals().
        pending = self._pending_wrist_delta
        self._last_wrist_motion = (pending / dt) if pending is not None else None
        return self._last_motion

    def _bounding_box(self, landmarks, frame_shape):
        """Pixel-space bounding box around all visible landmarks."""
        h, w = frame_shape[:2]
        xs = [lm.x for lm in landmarks if self._visible(lm)]
        ys = [lm.y for lm in landmarks if self._visible(lm)]
        if not xs or not ys:
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
        return (int(max(0, min(xs) * w) - 10), int(max(0, min(ys) * h) - 10),
                int(min(w, max(xs) * w) + 10), int(min(h, max(ys) * h) + 10))

    # ---------------------------------------------------------------------
    # Activity recognition (the "what is the person doing" part)
    # ---------------------------------------------------------------------
    def _classify_activity(self, now, is_fast_drop, is_horizontal, is_low,
                           is_motionless, motion):
        """
        Label the current activity from the sliding window of motion history.

        Core idea: EXERCISE IS RHYTHMIC. Squats/push-ups/burpees make the body
        center bounce up and down repeatedly; a bench press makes the WRISTS
        bounce while the body stays still. A real fall is one downward move
        with no bounce afterwards.

        Returns (label, features_dict) — features are exposed for the
        reasoning panel so the user can see WHY we chose the label.
        """
        # Drop samples that fell out of the analysis window.
        cutoff = now - config.ACTIVITY_WINDOW
        while self._activity_history and self._activity_history[0][0] < cutoff:
            self._activity_history.popleft()

        cys = [s[1] for s in self._activity_history]
        cxs = [s[2] for s in self._activity_history]
        angles = [s[3] for s in self._activity_history]
        wrists = [s[4] for s in self._activity_history if s[4] is not None]

        # --- Window features ---
        amp = (max(cys) - min(cys)) if cys else 0.0            # vertical bounce size
        revs = _count_reversals(cys, config.RHYTHM_MIN_AMPLITUDE / 2.0)
        angle_range = (max(angles) - min(angles)) if angles else 0.0
        x_range = (max(cxs) - min(cxs)) if cxs else 0.0        # horizontal travel
        wrist_amp = (max(wrists) - min(wrists)) if len(wrists) >= 3 else 0.0
        wrist_revs = _count_reversals(wrists, config.WRIST_RHYTHM_AMPLITUDE / 2.0)

        rhythmic = revs >= config.RHYTHM_MIN_REVERSALS and amp >= config.RHYTHM_MIN_AMPLITUDE
        wrist_rhythmic = (wrist_revs >= config.RHYTHM_MIN_REVERSALS
                          and wrist_amp >= config.WRIST_RHYTHM_AMPLITUDE)

        features = {
            "amp": amp, "revs": revs, "angle_range": angle_range,
            "x_range": x_range, "wrist_amp": wrist_amp, "wrist_revs": wrist_revs,
            "rhythmic": rhythmic, "wrist_rhythmic": wrist_rhythmic,
        }

        # --- Decision rules, most urgent first ---
        if is_fast_drop:
            return ACT_FALLING, features

        # Burpees: big body bounces AND the torso keeps flipping between
        # upright and flat — no other gym movement swings the torso that much.
        if (rhythmic and amp >= config.BURPEE_MIN_AMPLITUDE
                and angle_range >= config.BURPEE_TORSO_SWING):
            return ACT_BURPEES, features

        # Pressing does NOT require being LOW in the frame — a person on a
        # bench sits mid-frame. Horizontal + pumping wrists + still body =
        # pressing, at any height.
        if (is_horizontal and wrist_rhythmic
                and amp < config.RHYTHM_MIN_AMPLITUDE * 1.5):
            return ACT_BENCH, features

        if is_horizontal and is_low:
            # On the floor: still, doing push-ups, or just moving around?
            if is_motionless and not rhythmic and not wrist_rhythmic:
                return ACT_LYING, features
            if rhythmic:
                return ACT_PUSHUPS, features
            return ACT_FLOOR, features

        if is_horizontal:
            # Horizontal but not low: e.g. lying on a bench without pressing.
            return (ACT_LYING if is_motionless else ACT_FLOOR), features

        # Upright activities.
        if rhythmic and x_range < config.WALK_X_RANGE:
            return ACT_SQUATS, features
        if x_range >= config.WALK_X_RANGE and motion > config.MOTIONLESS_THRESHOLD:
            return ACT_WALKING, features
        if motion > config.MOTIONLESS_THRESHOLD * 2.5:
            return ACT_ACTIVE, features
        return ACT_STANDING, features

    # ---------------------------------------------------------------------
    # Reasoning lines for the on-screen "SYSTEM VIEW" panel
    # ---------------------------------------------------------------------
    def _build_thinking(self, now, drop_speed, torso_angle,
                        is_fast_drop, is_horizontal, is_low, is_motionless,
                        activity, f, exercising,
                        bench_recent, entrap_cond, pinned_time):
        """
        Turn the raw checks into short human-readable lines with a severity
        level for coloring: "good" (green), "info" (gray), "warn" (orange),
        "bad" (red). This is literally the decision process, verbalized.
        """
        lines = []

        # 1. The core fall signals, as pass/fail checks.
        lines.append((
            f"Sudden drop: {drop_speed:.2f} " + ("!! FAST" if is_fast_drop else "(calm)"),
            "bad" if is_fast_drop else "good",
        ))
        lines.append((
            f"Torso angle: {torso_angle:.0f}deg " + ("(flat)" if is_horizontal else "(upright)"),
            "warn" if is_horizontal else "good",
        ))
        lines.append((
            f"Body height: {'LOW in frame' if is_low else 'normal'}",
            "warn" if is_low else "good",
        ))

        # 2. Rhythm evidence — the exercise-vs-incident discriminator.
        if f["rhythmic"]:
            lines.append((f"Rhythm: {f['revs']} bounces -> exercise", "good"))
        elif f["wrist_rhythmic"]:
            lines.append(("Rhythm: wrists pumping -> pressing", "good"))
        else:
            lines.append(("Rhythm: none detected", "info"))

        # 3. Entrapment watch — narrate the bar-person-duration check.
        if entrap_cond and pinned_time > 0.5:
            lines.append((
                f"Wrists pinned {pinned_time:.1f}/"
                f"{config.ENTRAPMENT_CONFIRM_DURATION:.0f}s -> entrapment?",
                "bad" if pinned_time >= config.ENTRAPMENT_CONFIRM_DURATION / 2 else "warn",
            ))
        elif self.state == STATE_ENTRAPMENT and self._entrap_release_since is not None:
            held = now - self._entrap_release_since
            lines.append((
                f"Wrists moving again {held:.1f}/"
                f"{config.ENTRAPMENT_RELEASE_TIME:.0f}s -> recovering?", "warn",
            ))
        elif bench_recent:
            lines.append(("Pressing set detected (watching bar)", "info"))

        # 4. Down-too-long watch.
        if self._down_since is not None and not exercising and not bench_recent:
            floor_time = now - self._down_since
            if floor_time > 5.0:
                lines.append((
                    f"On floor for {floor_time:.0f}/"
                    f"{config.DOWN_TOO_LONG_DURATION:.0f}s (no exercise) -> concern",
                    "bad" if floor_time >= 0.8 * config.DOWN_TOO_LONG_DURATION else "warn",
                ))

        # 5. Stillness / fall-confirmation countdown.
        if self.state == STATE_SUSPICIOUS and self._motionless_since is not None:
            held = now - self._motionless_since
            lines.append((
                f"Still for {held:.1f}/{config.MOTIONLESS_DURATION:.0f}s -> confirming fall...",
                "bad",
            ))
        elif is_motionless and (is_horizontal or is_low):
            lines.append(("Motionless while down", "warn"))
        elif is_motionless:
            lines.append(("Holding still (upright, ok)", "info"))
        else:
            lines.append(("Moving normally", "good"))

        # 6. Exercise suppression notice — shows WHY we're not alarmed.
        if exercising and self.state == STATE_NORMAL:
            lines.append(("Exercise detected: suspicion suppressed", "good"))

        # 7. Queued alert notice.
        if self._pending_alert_message:
            lines.append(("Alert queued (cooldown active)", "warn"))

        # 8. Final verdict line.
        if self.state == STATE_ENTRAPMENT:
            lines.append(("VERDICT: possible entrapment - assist NOW!", "bad"))
        elif self.state == STATE_FALL:
            lines.append(("VERDICT: possible incident - check on them!", "bad"))
        elif self.state == STATE_SUSPICIOUS:
            lines.append(("VERDICT: watching closely...", "warn"))
        else:
            lines.append((f"VERDICT: {activity} - looks safe", "good"))

        return lines

    # ---------------------------------------------------------------------
    # Alert firing with cooldown queueing
    # ---------------------------------------------------------------------
    def _fire_alert(self, now, message):
        """
        Fire an alert, or queue it if the cooldown is active. Returns the
        message when it actually fires (this frame), else None.

        Without the queue, an incident confirmed during the cooldown would
        move the state but never alert — a person who stayed down could sit
        in an incident state with no alert ever sent.
        """
        if (now - self._last_alert_time) < config.ALERT_COOLDOWN:
            self._pending_alert_message = message
            return None
        self._last_alert_time = now
        self._pending_alert_message = None
        return message

    def _flush_pending(self, now):
        """Fire the queued alert once the cooldown has expired."""
        if (self._pending_alert_message
                and (now - self._last_alert_time) >= config.ALERT_COOLDOWN):
            message = self._pending_alert_message
            self._pending_alert_message = None
            self._last_alert_time = now
            return message
        return None

    def _set_state(self, now, new_state):
        """Change state, recording the transition for events.csv."""
        if new_state != self.state:
            self.transitions.append((now, self.state, new_state))
            self.state = new_state

    # ---------------------------------------------------------------------
    # State machine (the "how we decide" part)
    # ---------------------------------------------------------------------
    def _update_state_machine(self, now, s):
        """
        Combine the signals over time. `s` is a dict of instantaneous
        conditions (see process()); all TIMING lives here so the whole
        decision process can be tested without a camera.

        Returns the alert message on exactly the frame an alert fires,
        else None.
        """
        alert_out = None

        # --- ENTRAPMENT route (bar-person-duration logic) -------------------
        # Deliberately separate from the fall route: a pinned lifter never
        # goes "motionless" (they struggle), nothing "drops", and it must
        # fire DURING a recognized pressing session — so it cannot use the
        # fall confirmation and must not be suppressed by `exercising`.
        if self.state != STATE_ENTRAPMENT:
            if s["entrap_cond"]:
                if self._entrap_since is None:
                    self._entrap_since = now
                elif (now - self._entrap_since) >= config.ENTRAPMENT_CONFIRM_DURATION:
                    # Wrists pinned at chest long enough: a failed rep would
                    # have been racked/rolled off by now => entrapment.
                    self._set_state(now, STATE_ENTRAPMENT)
                    self._entrap_release_since = None
                    alert_out = self._fire_alert(now, config.ALERT_MESSAGE_ENTRAPMENT)
            else:
                self._entrap_since = None

        # --- "Down too long" timer ------------------------------------------
        # Runs independent of the SUSPICIOUS state and of motion. Requires a
        # FLAT body (not merely low): sitting on the gym floor is normal.
        if s["flat"]:
            if self._down_since is None:
                self._down_since = now
            if s["exercising"] or s["bench_recent"]:
                # Deliberate exercise: keep sliding the timer forward so the
                # countdown starts from the END of the workout (otherwise
                # finishing push-ups then resting would alert immediately).
                self._down_since = now
            elif (now - self._down_since) >= config.DOWN_TOO_LONG_DURATION \
                    and self.state not in (STATE_FALL, STATE_ENTRAPMENT):
                self._set_state(now, STATE_FALL)
                alert_out = self._fire_alert(now, config.ALERT_MESSAGE_DOWN_LONG)
        else:
            self._down_since = None

        # --- Classic fall route ---------------------------------------------
        # "Suspicious" trigger: a sudden drop always counts. A collapsed-
        # looking posture (flat AND low) counts ONLY if we haven't just
        # recognized deliberate exercise — this is what stops burpees and
        # push-ups from tripping the system. NOTE: even during exercise, a
        # fast drop followed by stillness will still alert (safety first).
        looks_suspicious = s["fast_drop"] or (s["flat"] and s["low"] and not s["exercising"])
        on_the_floor = s["flat"] or s["low"]

        if self.state == STATE_NORMAL:
            if looks_suspicious:
                if self._suspicious_since is None:
                    self._suspicious_since = now
                elif (now - self._suspicious_since) >= config.SUSPICIOUS_PERSIST_TIME:
                    self._set_state(now, STATE_SUSPICIOUS)
                    self._fall_candidate_since = now
                    self._motionless_since = None
            else:
                self._suspicious_since = None

        elif self.state == STATE_SUSPICIOUS:
            elapsed = now - (self._fall_candidate_since or now)

            if on_the_floor and s["motionless"]:
                if self._motionless_since is None:
                    self._motionless_since = now
                elif (now - self._motionless_since) >= config.MOTIONLESS_DURATION:
                    # Confirmed: down + still for long enough => alert.
                    self._set_state(now, STATE_FALL)
                    alert_out = self._fire_alert(now, config.ALERT_MESSAGE)
            else:
                # They moved again -> reset the motionless timer.
                self._motionless_since = None
                if not on_the_floor and elapsed >= config.SUSPICIOUS_PERSIST_TIME:
                    self._reset_fall_tracking()
                    self._set_state(now, STATE_NORMAL)

            # Safety valve: never get stuck in SUSPICIOUS forever. (The
            # "down too long" route still covers a person who stays flat
            # but keeps moving, e.g. a seizure.)
            if self.state == STATE_SUSPICIOUS and elapsed >= config.FALL_CONFIRM_WINDOW:
                self._reset_fall_tracking()
                self._set_state(now, STATE_NORMAL)

        elif self.state == STATE_FALL:
            # If the alert for this episode was queued during a cooldown,
            # fire it the moment the cooldown expires.
            if alert_out is None:
                alert_out = self._flush_pending(now)
            # Recovery: upright and clearly moving -> back to NORMAL.
            if not on_the_floor and s["motion"] > config.MOTIONLESS_THRESHOLD * 1.5:
                self._reset_fall_tracking()
                self._pending_alert_message = None   # they're fine; drop queue
                self._recovered_reason = "stood up and moving"
                self._set_state(now, STATE_NORMAL)

        elif self.state == STATE_ENTRAPMENT:
            if alert_out is None:
                alert_out = self._flush_pending(now)
            # Recovery: the pinned condition must stay BROKEN for a couple of
            # seconds (wrists moved away from chest / person sat up) — a one-
            # frame flicker of the signals must not cancel a real entrapment.
            if s["entrap_cond"]:
                self._entrap_release_since = None
            else:
                if self._entrap_release_since is None:
                    self._entrap_release_since = now
                elif (now - self._entrap_release_since) >= config.ENTRAPMENT_RELEASE_TIME:
                    self._entrap_since = None
                    self._entrap_release_since = None
                    self._pending_alert_message = None
                    self._recovered_reason = "wrists free / sat up (entrapment released)"
                    self._set_state(now, STATE_NORMAL)

        return alert_out

    def _reset_fall_tracking(self):
        """Clear the transient timers used while tracking a possible fall."""
        self._suspicious_since = None
        self._fall_candidate_since = None
        self._motionless_since = None
