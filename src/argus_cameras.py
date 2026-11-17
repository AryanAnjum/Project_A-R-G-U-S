"""Cameras plugged into the ARGUS PC, above all the head-mount camera.

The head-mount camera is a USB webcam worn on the head and plugged into the PC. The server opens it
itself (OpenCV), so the phone apps, the dashboard and the desktop app can all use it, even when no
browser window is open:

  * list_cameras()        the PC's cameras: [{"index": 0, "name": "Rapoo Camera"}, ...]
  * HeadCamera            keeps the chosen camera open while somebody uses it and hands each
                          frame to a callback (the server stores it as source "head")

Names come from DirectShow through the small optional `pygrabber` package on Windows. Without it the
cameras are called "Camera 0", "Camera 1", ...

Settings:
  ARGUS_HEAD_CAMERA   "auto" (default), a camera number (0, 1, ...) or part of its name ("rapoo")
  ARGUS_HEAD_FPS      frames handed on per second (default 15)
  ARGUS_HEAD_WIDTH / ARGUS_HEAD_HEIGHT   capture size (default 1920 x 1080, sharp enough for OCR)
"""
import json
import os
import sys
import threading
import time

import cv2

try:
    from pygrabber.dshow_graph import FilterGraph  # Windows only, optional
except Exception:
    FilterGraph = None

WINDOWS = sys.platform == "win32"
BUILT_IN_HINTS = ("integrated", "built-in", "builtin", "facetime", "ir camera", "infrared", "front", "rear",
                  "virtual", "obs", "snap camera", "droidcam", "iriun", "nvidia broadcast", "windows hello")
EXTERNAL_HINTS = ("usb", "rapoo", "logitech", "c270", "c920", "webcam")


def _env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def list_cameras(probe=4, skip=()):
    """The PC's cameras. Without camera names, numbers 0..probe-1 are tried (skip: numbers in use)."""
    if FilterGraph is not None:
        try:
            names = FilterGraph().get_input_devices()
            return [{"index": i, "name": n} for i, n in enumerate(names)]
        except Exception:
            pass
    found = []
    for i in range(probe):
        if i in skip:
            found.append({"index": i, "name": f"Camera {i}"})
            continue
        cap = _open(i)
        ok = cap is not None and cap.isOpened()
        if cap is not None:
            cap.release()
        if ok:
            found.append({"index": i, "name": f"Camera {i}"})
    return found


def guess_head_camera(cameras):
    """The camera most likely to be the head-mount one: a USB webcam rather than the laptop's own."""
    if not cameras:
        return None
    named = [c for c in cameras if not c["name"].startswith("Camera ")]
    if named:
        for c in named:
            if any(h in c["name"].lower() for h in EXTERNAL_HINTS) and \
                    not any(h in c["name"].lower() for h in BUILT_IN_HINTS):
                return c["index"]
        others = [c for c in named if not any(h in c["name"].lower() for h in BUILT_IN_HINTS)]
        if others:
            return others[-1]["index"]
    # Laptops list their own camera first, so a second camera is usually the USB one.
    return cameras[-1]["index"] if len(cameras) > 1 else cameras[0]["index"]


def _open(index):
    try:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if WINDOWS else cv2.VideoCapture(index)
    except Exception:
        return None
    return cap


class HeadCamera:
    """Keeps the head-mount camera open while it is needed and passes frames to on_frame(frame, ts)."""

    def __init__(self, on_frame, on_change, settings_path):
        self.on_frame = on_frame
        self.on_change = on_change          # called with state() whenever it changes
        self.settings_path = settings_path
        self.fps = max(1, _env_int("ARGUS_HEAD_FPS", 15))
        self.size = (_env_int("ARGUS_HEAD_WIDTH", 1920), _env_int("ARGUS_HEAD_HEIGHT", 1080))
        self.lock = threading.Lock()
        self.wanted = False
        self.running = False
        self.error = ""
        self.index = None
        self.name = ""
        self.frame_size = None
        self.cameras = []
        self.thread = None
        self._load_choice()

    # ---- which camera is the head-mount one ----
    def _load_choice(self):
        choice = os.getenv("ARGUS_HEAD_CAMERA", "auto").strip()
        try:
            with open(self.settings_path, encoding="utf-8") as f:
                saved = json.load(f).get("head_camera")
            if saved is not None and choice == "auto":
                choice = str(saved)
        except Exception:
            pass
        self.choice = choice

    def _resolve(self):
        self.cameras = list_cameras(skip={self.index} if self.running and self.index is not None else ())
        choice = self.choice.lower()
        if choice.isdigit():
            idx = int(choice)
        elif choice in ("", "auto"):
            idx = guess_head_camera(self.cameras)
        else:
            idx = next((c["index"] for c in self.cameras if choice in c["name"].lower()), None)
            if idx is None:
                idx = guess_head_camera(self.cameras)
        name = next((c["name"] for c in self.cameras if c["index"] == idx), f"Camera {idx}" if idx is not None else "")
        return idx, name

    def choose(self, index):
        """The dashboard picked which PC camera is the head-mount one (None: automatic)."""
        self.choice = "auto" if index is None else str(int(index))
        try:
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump({"head_camera": None if index is None else int(index)}, f)
        except Exception:
            pass
        with self.lock:
            restart = self.wanted
        if restart:
            self.set_wanted(False)
            time.sleep(0.3)
            self.set_wanted(True)
        else:
            self._changed()

    # ---- on / off ----
    def set_wanted(self, on):
        with self.lock:
            if on == self.wanted:
                return
            self.wanted = on
            if on and (self.thread is None or not self.thread.is_alive()):
                self.thread = threading.Thread(target=self._run, daemon=True)
                self.thread.start()
        if not on:
            self.error = ""  # a problem is reported again if it is switched back on and still there
            self._changed()

    def state(self):
        return {"running": self.running, "wanted": self.wanted, "error": self.error, "index": self.index,
                "name": self.name, "size": list(self.frame_size) if self.frame_size else None,
                "choice": self.choice}

    def _changed(self):
        try:
            self.on_change(self.state())
        except Exception:
            pass

    def _set(self, running=None, error=None):
        changed = False
        if running is not None and running != self.running:
            self.running, changed = running, True
        if error is not None and error != self.error:
            self.error, changed = error, True
        if changed:
            self._changed()

    def _run(self):
        while True:
            with self.lock:
                if not self.wanted:
                    break
            idx, name = self._resolve()
            self.index, self.name = idx, name
            if idx is None:
                self._set(False, "No camera is plugged into the PC. Plug in the head-mount camera.")
                if self._wait(3.0):
                    break
                continue
            cap = _open(idx)
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                self._set(False, f"The head-mount camera ({name}) couldn't be opened. Check its cable, and close "
                                 "other apps (or the PC app's own webcam) that use it.")
                if self._wait(3.0):
                    break
                continue
            # MJPG lets a USB 2 webcam deliver 1080p at full frame rate.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"[CAMERA] Head-mount camera on: {name} (camera {idx})")
            fails, last = 0, 0.0
            gap = 1.0 / self.fps
            while True:
                with self.lock:
                    if not self.wanted:
                        break
                ok, frame = cap.read()
                if not ok or frame is None:
                    fails += 1
                    if fails >= 30:
                        self._set(False, f"The head-mount camera ({name}) stopped sending pictures. "
                                         "Check that it is plugged in.")
                        break
                    time.sleep(0.05)
                    continue
                fails = 0
                now = time.time()
                if now - last < gap:
                    continue
                last = now
                self.frame_size = (frame.shape[1], frame.shape[0])
                self._set(True, "")
                try:
                    self.on_frame(frame, now)
                except Exception as exc:
                    print(f"[CAMERA] Frame handler failed: {exc}")
            cap.release()
            self._set(False)
            print("[CAMERA] Head-mount camera off")
            with self.lock:
                if not self.wanted:
                    break
            if self._wait(2.0):
                break
        self._set(False)
        with self.lock:
            if self.wanted:  # turned back on while it was closing
                self.thread = threading.Thread(target=self._run, daemon=True)
                self.thread.start()
            else:
                self.thread = None

    def _wait(self, seconds):
        """Sleeps, returning True early if the camera is no longer wanted."""
        end = time.time() + seconds
        while time.time() < end:
            with self.lock:
                if not self.wanted:
                    return True
            time.sleep(0.1)
        return False
