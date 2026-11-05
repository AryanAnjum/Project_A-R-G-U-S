"""Live sign language translation (SLT) for the ARGUS server.

A client (the dashboard webcam or the phone) streams camera frames at about
12-15 fps while SLT mode is on. Each client gets a SignSession that

1. runs MediaPipe Holistic on every frame to get the 107 keypoints the
   ARGUS-SLT model was trained on (sign_language/keypoints.py),
2. cuts the stream into sentences: signing starts when a hand appears and
   ends when no hand has been seen for ARGUS_SLT_END_GAP seconds (or when the
   user presses "Translate now"),
3. resamples the sentence to the 25 fps the model was trained on, and
4. translates it with the checkpoint for the selected sign language.

Languages are registered in LANGUAGES. A language whose checkpoint folder does
not exist is reported as "not ready" to the clients, so the Bangla model can be
added later by dropping its checkpoint at ARGUS_SLT_BN_CKPT (or the default
path) and restarting the server: no code change needed.
"""

import os
import queue
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
DATA_ROOT = os.path.abspath(os.getenv("ARGUS_HOME", REPO_ROOT))  # models and checkpoints

from sign_language.keypoints import NUM_POINTS, PARTS, from_holistic, normalize, to_features  # noqa: E402

MODEL_FPS = 25.0  # PHOENIX-2014T and the keypoint extraction both use 25 fps
END_GAP_S = float(os.getenv("ARGUS_SLT_END_GAP", "1.0"))
MIN_SIGN_S = float(os.getenv("ARGUS_SLT_MIN_SECONDS", "0.8"))
MAX_SIGN_S = float(os.getenv("ARGUS_SLT_MAX_SECONDS", "15.0"))
NUM_BEAMS = int(os.getenv("ARGUS_SLT_BEAMS", "5"))
HOLISTIC_TASK = os.getenv("ARGUS_SLT_HOLISTIC", os.path.join(DATA_ROOT, "models", "holistic_landmarker.task"))
I3D_CKPT = os.getenv("ARGUS_SLT_I3D", os.path.join(DATA_ROOT, "models", "bsl1k.pth.tar"))
I3D_RESIZE, I3D_CROP = 256, 224


@dataclass
class Language:
    code: str           # mBART language key used in training ("de", "bn")
    label: str          # shown in the UI
    sign_language: str  # shown in the UI
    checkpoint: str     # folder written by sign_language.train (model.pt + vocab.json)
    tts_lang: str       # language of the spoken caption

    @property
    def ready(self):
        return os.path.isfile(os.path.join(self.checkpoint, "model.pt"))


LANGUAGES = {
    "de": Language(
        "de", "German", "DGS",
        os.getenv("ARGUS_SLT_DE_CKPT", os.path.join(DATA_ROOT, "runs", "ablation_small_decoder_s1", "best")),
        "de",
    ),
    # Bangla is still training. Put its checkpoint here (or set ARGUS_SLT_BN_CKPT) and it turns on.
    "bn": Language(
        "bn", "Bangla", "BdSL",
        os.getenv("ARGUS_SLT_BN_CKPT", os.path.join(DATA_ROOT, "runs", "bansign_small", "best")),
        "bn",
    ),
}


def language_list():
    return [{"code": l.code, "label": l.label, "sign_language": l.sign_language, "ready": l.ready}
            for l in LANGUAGES.values()]


def resample_keypoints(frames, times, fps=MODEL_FPS):
    """Resample a (N, V, 3) keypoint sequence captured at irregular times to a fixed fps.

    Points present in both neighbouring frames are interpolated linearly; a point
    missing in one of them takes the nearer frame's value, so missing stays zero
    instead of being smeared towards the origin.
    """
    frames = np.asarray(frames, np.float32)
    times = np.asarray(times, np.float64)
    if len(frames) == 1:
        return frames.copy()
    n = max(1, int(round((times[-1] - times[0]) * fps)) + 1)
    target = times[0] + np.arange(n) / fps
    hi = np.clip(np.searchsorted(times, target, side="right"), 1, len(times) - 1)
    lo = hi - 1
    span = np.maximum(times[hi] - times[lo], 1e-6)
    w = np.clip((target - times[lo]) / span, 0.0, 1.0)[:, None, None]
    a, b = frames[lo], frames[hi]
    present_a = np.abs(a).sum(-1, keepdims=True) > 0
    present_b = np.abs(b).sum(-1, keepdims=True) > 0
    mixed = a * (1 - w) + b * w
    nearest = np.where(w < 0.5, a, b)
    return np.where(present_a & present_b, mixed, nearest).astype(np.float32)


def resample_index(times, fps=MODEL_FPS):
    """Indices of the captured frames nearest to each fixed-fps time step (used for RGB clips)."""
    times = np.asarray(times, np.float64)
    n = max(1, int(round((times[-1] - times[0]) * fps)) + 1)
    target = times[0] + np.arange(n) / fps
    idx = np.clip(np.searchsorted(times, target), 0, len(times) - 1)
    prev = np.clip(idx - 1, 0, len(times) - 1)
    closer_prev = np.abs(times[prev] - target) < np.abs(times[idx] - target)
    return np.where(closer_prev, prev, idx)


def hands_visible(kp):
    present = np.abs(kp).sum(-1) > 0
    return bool(present[PARTS["lhand"]].any() or present[PARTS["rhand"]].any())


class SLTEngine:
    """Loads translation checkpoints on first use and runs them one at a time."""

    def __init__(self, device):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.models = {}
        self.i3d = None
        self.lock = threading.Lock()

    def _load(self, code):
        if code in self.models:
            return self.models[code]
        lang = LANGUAGES[code]
        if not lang.ready:
            raise RuntimeError(f"The {lang.label} sign model isn't ready yet.")
        from sign_language.model import load_checkpoint
        print(f"[SLT] Loading {lang.label} ({lang.sign_language}) model from {lang.checkpoint}")
        model, vocab, args = load_checkpoint(lang.checkpoint, self.device)
        needs_rgb = bool(args.get("rgb_dim", 0))
        if needs_rgb and self.i3d is None:
            if not os.path.isfile(I3D_CKPT):
                raise RuntimeError(
                    f"This {lang.label} model also needs RGB features, but the I3D checkpoint "
                    f"is missing ({I3D_CKPT}). See sign_language/README.md, step 2b."
                )
            from sign_language.extract_i3d import load_i3d
            self.i3d = load_i3d(I3D_CKPT, self.device)
            print("[SLT] I3D RGB feature extractor loaded.")
        self.models[code] = (model, vocab, needs_rgb)
        return self.models[code]

    def needs_rgb(self, code):
        with self.lock:
            return self._load(code)[2]

    def translate(self, code, keypoints, rgb_frames=None):
        """keypoints: (T, V, 3) at 25 fps. rgb_frames: (T, 224, 224, 3) uint8 RGB or None."""
        torch = self.torch
        with self.lock:
            model, vocab, needs_rgb = self._load(code)
            seq = normalize(keypoints)
            feats = torch.from_numpy(to_features(seq))[None].to(self.device)
            lengths = torch.tensor([feats.shape[1]])
            rgb = None
            if needs_rgb:
                from sign_language.dataset import resample
                from sign_language.extract_i3d import features
                from sign_language.model import PoseEncoder
                clips = features(self.i3d, rgb_frames, self.device).astype(np.float32)
                n = int(PoseEncoder.out_lengths(torch.tensor(feats.shape[1])))
                rgb = torch.from_numpy(resample(clips, n))[None].to(self.device)
            with torch.no_grad():
                out = model.translate(feats, lengths, num_beams=NUM_BEAMS, rgb=rgb)
            return vocab.decode(out.tolist())[0]


class SignSession:
    """Turns one client's frame stream into translated sentences.

    on_event(dict) is called from the session's worker thread for status updates
    and captions; the server forwards those to the client.
    """

    def __init__(self, engine, on_event, language="de"):
        self.engine = engine
        self.on_event = on_event
        self.language = language
        self.frames = queue.Queue(maxsize=8)
        self.running = True
        self.flush_requested = False
        self._landmarker = None
        self._last_ts_ms = -1
        self._last_error = None
        self._reset_buffer()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    # ---- called from the WebSocket handler -------------------------------
    def push_frame(self, frame_bgr, timestamp=None):
        item = (frame_bgr, time.time() if timestamp is None else timestamp)
        try:
            self.frames.put_nowait(item)
        except queue.Full:
            # MediaPipe is behind: drop the oldest frame so latency doesn't build up
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            self.frames.put_nowait(item)

    def set_language(self, code):
        if code in LANGUAGES:
            self.language = code
            self._reset_buffer()

    def translate_now(self):
        self.flush_requested = True

    def clear(self):
        self._reset_buffer()
        self.on_event({"event": "slt_status", "state": "waiting", "seconds": 0})

    def close(self):
        self.running = False

    # ---- worker ----------------------------------------------------------
    def _reset_buffer(self):
        self.kp, self.times, self.rgb = [], [], []
        self.signing = False
        self.last_hand_time = 0.0
        self._last_status = 0.0

    def _get_landmarker(self):
        if self._landmarker is None:
            from mediapipe.tasks.python import BaseOptions, vision
            from sign_language.extract_keypoints import ensure_model
            options = vision.HolisticLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=ensure_model(HOLISTIC_TASK)),
                running_mode=vision.RunningMode.VIDEO,
            )
            self._landmarker = vision.HolisticLandmarker.create_from_options(options)
        return self._landmarker

    def _keypoints(self, frame_bgr, t):
        import mediapipe as mp
        ts_ms = max(int(t * 1000), self._last_ts_ms + 1)  # VIDEO mode needs increasing timestamps
        self._last_ts_ms = ts_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        return from_holistic(self._get_landmarker().detect_for_video(image, ts_ms))

    @staticmethod
    def _rgb_crop(frame_bgr):
        f = cv2.resize(frame_bgr, (I3D_RESIZE, I3D_RESIZE), interpolation=cv2.INTER_LINEAR)
        o = (I3D_RESIZE - I3D_CROP) // 2
        return cv2.cvtColor(f[o:o + I3D_CROP, o:o + I3D_CROP], cv2.COLOR_BGR2RGB)

    def _run(self):
        while self.running:
            try:
                frame, t = self.frames.get(timeout=0.2)
            except queue.Empty:
                frame, t = None, time.time()
            try:
                if frame is not None:
                    self._on_frame(frame, t)
                self._maybe_finish(t)
            except Exception as exc:  # keep the session alive, but report each distinct problem once
                message = f"{type(exc).__name__}: {exc}"
                if message != self._last_error:
                    self._last_error = message
                    print(f"[SLT ERROR] {message}")
                    self.on_event({"event": "slt_error", "text": str(exc)})
                self._reset_buffer()
                time.sleep(0.5)
        if self._landmarker is not None:
            self._landmarker.close()

    def _on_frame(self, frame, t):
        kp = self._keypoints(frame, t)
        hand = hands_visible(kp)
        if hand:
            self.last_hand_time = t
            if not self.signing:
                self.signing = True
                self.kp, self.times, self.rgb = [], [], []
        if self.signing:
            self.kp.append(kp)
            self.times.append(t)
            if self._wants_rgb():
                self.rgb.append(self._rgb_crop(frame))
        if t - self._last_status > 0.5:
            self._last_status = t
            seconds = (self.times[-1] - self.times[0]) if self.signing and self.times else 0
            self.on_event({"event": "slt_status", "state": "signing" if self.signing else "waiting",
                           "hands": hand, "seconds": round(seconds, 1)})

    def _wants_rgb(self):
        try:
            return self.engine.needs_rgb(self.language)
        except Exception:
            return False

    def _maybe_finish(self, now):
        if self.flush_requested:
            self.flush_requested = False
            if self.signing and self.kp:
                self._finish()
            return
        if not self.signing or not self.times:
            return
        duration = self.times[-1] - self.times[0]
        if now - self.last_hand_time >= END_GAP_S or duration >= MAX_SIGN_S:
            self._finish()

    def _finish(self):
        kp, times, rgb = self.kp, self.times, self.rgb
        self._reset_buffer()
        # drop the hands-down tail after the last hand so the model sees only signing
        last = max(i for i, k in enumerate(kp) if hands_visible(k)) if any(hands_visible(k) for k in kp) else len(kp) - 1
        kp, times, rgb = kp[:last + 1], times[:last + 1], rgb[:last + 1]
        duration = times[-1] - times[0]
        if duration < MIN_SIGN_S:
            self.on_event({"event": "slt_status", "state": "waiting", "seconds": 0,
                           "note": "Too short to translate."})
            return
        self.on_event({"event": "slt_status", "state": "translating", "seconds": round(duration, 1)})
        start = time.time()
        seq = resample_keypoints(np.stack(kp), times)
        rgb_frames = None
        if rgb:
            rgb_frames = np.stack(rgb)[resample_index(times)]
        text = self.engine.translate(self.language, seq, rgb_frames)
        ms = int((time.time() - start) * 1000)
        lang = LANGUAGES[self.language]
        print(f"[SLT] {lang.sign_language} {duration:.1f}s -> {text!r} ({ms} ms)")
        self.on_event({"event": "slt_caption", "language": lang.code, "sign_language": lang.sign_language,
                       "text": text, "seconds": round(duration, 1), "translate_ms": ms,
                       "frames": len(seq), "tts_lang": lang.tts_lang})
        self.on_event({"event": "slt_status", "state": "waiting", "seconds": 0})


__all__ = ["LANGUAGES", "NUM_POINTS", "SLTEngine", "SignSession", "language_list",
           "resample_keypoints", "resample_index"]
