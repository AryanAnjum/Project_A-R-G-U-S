"""A.R.G.U.S. server: vision-language assistant, speech in/out and live sign language translation.

Run from the src folder:

    python allnew.py

then open http://localhost:8000 for the dashboard, or run `python argus_desktop.py` to
get the same dashboard in a native desktop window. The Android app connects to
ws://<this PC>:8000/ws.

Modes
    screen     answers about what is on this PC's screen (DXCam / MSS capture)
    live       answers about the camera (dashboard webcam or phone)
    assistive  short, safety-first descriptions for a visually impaired user
    slt        sign language translation from the camera (German now, Bangla later)

OCR: a question that asks to read something ("read this", "what does this say",
"how much money is this") or that arrives with the OCR button on is answered
from a full-resolution image with a reading-focused prompt. Other questions use
a smaller image so they answer faster.

Settings are environment variables; see CONFIG below and the README.
"""

import asyncio
import base64
import csv
import json
import os
import queue
import re
import string
import tempfile
import threading
import time
import traceback
from difflib import SequenceMatcher
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import cv2
import numpy as np
import pygame
import speech_recognition as sr
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
from peft import PeftModel
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (AutoProcessor, Qwen2VLForConditionalGeneration, StoppingCriteria,
                          StoppingCriteriaList, TextIteratorStreamer)

import argus_tts

try:
    import winsound
except ImportError:
    winsound = None

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    NVML_HANDLE = None
    print("[WARNING] NVML unavailable (pip install nvidia-ml-py). GPU temperature will show '--'.")


# ==========================================
# CONFIG
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
# Models, checkpoints and logs live here. The installed desktop app points this at its data folder.
DATA_ROOT = os.path.abspath(os.getenv("ARGUS_HOME", REPO_ROOT))


def env_float(name, default):
    return float(os.getenv(name, str(default)))


BASE_MODEL_ID = os.getenv("ARGUS_BASE_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
# Phased LoRA fine-tune (PEFT format). Defaults to the last phase.
ADAPTER_PATH = os.path.abspath(os.getenv(
    "ARGUS_ADAPTER_PATH", os.path.join(DATA_ROOT, "argus_phased_adapters", "part_5_adapter")))
MERGE_ADAPTER = os.getenv("ARGUS_MERGE_ADAPTER", "1") == "1"

# Longest image side sent to the model. OCR keeps the full 1920 px so small
# text stays readable; everyday questions use a smaller image to answer faster.
MAX_IMAGE_PX = int(os.getenv("ARGUS_MAX_IMAGE_PX", "1024"))
OCR_MAX_PX = int(os.getenv("ARGUS_OCR_MAX_PX", "1920"))
MAX_NEW_TOKENS = int(os.getenv("ARGUS_MAX_NEW_TOKENS", "160"))
OCR_MAX_NEW_TOKENS = int(os.getenv("ARGUS_OCR_MAX_NEW_TOKENS", "256"))
FRAME_MAX_AGE_S = env_float("ARGUS_FRAME_MAX_AGE", 3.0)

WHISPER_MODEL_NAME = os.getenv("ARGUS_WHISPER_MODEL", "small.en")
WHISPER_DEVICE = os.getenv("ARGUS_WHISPER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu").lower()
WHISPER_COMPUTE_TYPE = os.getenv("ARGUS_WHISPER_COMPUTE_TYPE", "float16" if WHISPER_DEVICE == "cuda" else "int8")
# Seconds of silence that end a spoken sentence. Lower answers sooner but may cut you off mid-pause.
PAUSE_THRESHOLD = env_float("ARGUS_PAUSE_THRESHOLD", 0.45)
# Seconds to ignore the mic after ARGUS stops talking, so room echo isn't heard as a question.
# 0 = listen again the moment ARGUS stops (best with earphones). With loudspeakers, 0.5-0.8 helps.
ECHO_GUARD_S = env_float("ARGUS_ECHO_GUARD", 0.0)
# While ARGUS talks: "all" = any new question interrupts it, "stop" = only stop / goodbye commands.
BARGE_IN = os.getenv("ARGUS_BARGE_IN", "all").lower()

# Sign language captions: also translate into English with Qwen, and which text to speak.
SLT_ENGLISH = os.getenv("ARGUS_SLT_ENGLISH", "1") == "1"
SLT_SPEAK = os.getenv("ARGUS_SLT_SPEAK", "en").lower()  # en | source | off

LATENCY_LOG = os.getenv("ARGUS_LATENCY_LOG", os.path.join(DATA_ROOT, "logs", "latency.csv"))

MODES = ("screen", "live", "assistive", "slt")
MODE_ALIASES = {"assist": "assistive", "sign": "slt", "lecture": "slt"}

pygame.mixer.init()

# Older transformers + newer peft ask for this attribute on every module; answer
# with "no tied weights" instead of crashing. Harmless on versions that have it.
_orig_getattr = torch.nn.Module.__getattr__


def _patched_getattr(self, name):
    if name == "all_tied_weights_keys":
        return {}
    return _orig_getattr(self, name)


torch.nn.Module.__getattr__ = _patched_getattr


# ==========================================
# 1. CLIENTS & WEBSOCKET BROADCAST
# ==========================================
@dataclass
class Client:
    ws: WebSocket
    kind: str = "dashboard"  # "dashboard" or "phone"
    mode: str = "screen"
    slt: object = None       # SignSession while the client is in SLT mode
    audio_queue: bool = False  # phone app can queue several audio clips per reply


class ConnectionManager:
    def __init__(self):
        self.clients = {}
        self.loop = None
        self.pc_mode = "screen"  # mode of the dashboard and the PC microphone

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.clients[websocket] = Client(websocket, mode=self.pc_mode)
        print(f"[SERVER] Client connected ({len(self.clients)} total)")

    def disconnect(self, websocket):
        client = self.clients.pop(websocket, None)
        if client and client.slt:
            client.slt.close()
        if client:
            print(f"[SERVER] {client.kind.capitalize()} disconnected")

    def dashboards(self):
        return [c.ws for c in list(self.clients.values()) if c.kind == "dashboard"]

    def has_dashboard(self):
        return any(c.kind == "dashboard" for c in list(self.clients.values()))

    def broadcast_sync(self, message: dict, dashboards_only=False):
        targets = self.dashboards() if dashboards_only else list(self.clients)
        for ws in targets:
            self.send_to_sync(ws, message)

    def send_to_sync(self, websocket, message: dict):
        if self.loop and websocket is not None:
            asyncio.run_coroutine_threadsafe(self._send_to(websocket, message), self.loop)

    async def _send_to(self, websocket, message: dict):
        if websocket not in self.clients:
            return
        try:
            await websocket.send_text(json.dumps(message))
        except Exception:
            pass

    def reply_to(self, origin, requester, message):
        """Phone requests are answered on the phone only; everything else on all dashboards."""
        if origin == "phone" and requester is not None:
            self.send_to_sync(requester, message)
        else:
            self.broadcast_sync(message, dashboards_only=True)


ws_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ws_manager.loop = asyncio.get_running_loop()
    yield


app = FastAPI(title="A.R.G.U.S. Multimodal Engine", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def log_event(engine, text):
    print(f"[{engine}] {text}")
    ws_manager.broadcast_sync({"event": "perception_log", "engine": engine, "text": text}, dashboards_only=True)


def set_status(state):
    """idle | listening | thinking | speaking, shown on the dashboard."""
    ws_manager.broadcast_sync({"event": "assistant_status", "state": state}, dashboards_only=True)


# ==========================================
# 2. MODELS & SENSORS
# ==========================================
print("\n[SYSTEM] Booting A.R.G.U.S. Multimodal Engine...")
device = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
GPU_TOTAL_GB = (torch.cuda.get_device_properties(0).total_memory / 1024 ** 3) if torch.cuda.is_available() else 0.0

print(f"[SYSTEM] Loading Qwen2-VL backbone ({BASE_MODEL_ID})...")
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map=device, low_cpu_mem_usage=True)

if not os.path.isfile(os.path.join(ADAPTER_PATH, "adapter_config.json")):
    raise FileNotFoundError(
        f"No adapter_config.json in {ADAPTER_PATH}. "
        "Set ARGUS_ADAPTER_PATH to your argus_phased_adapters/part_X_adapter folder.")

print(f"[SYSTEM] Loading A.R.G.U.S. fine-tuned adapter from {ADAPTER_PATH}...")
vlm = PeftModel.from_pretrained(base_model, ADAPTER_PATH, adapter_name="argus_general")
if MERGE_ADAPTER:
    vlm = vlm.merge_and_unload()  # no per-layer adapter overhead at inference, no extra VRAM
    print("[SYSTEM] Adapter merged into backbone.")
vlm.eval()
vlm_lock = threading.Lock()  # the assistant and the SLT English translation share the model

try:
    processor = AutoProcessor.from_pretrained(ADAPTER_PATH)
except Exception:
    print("[WARNING] Adapter has no processor files; using the base model's processor.")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

print(f"[SYSTEM] Loading Whisper {WHISPER_MODEL_NAME} ({WHISPER_DEVICE}, {WHISPER_COMPUTE_TYPE})...")
audio_model = WhisperModel(WHISPER_MODEL_NAME, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)

print("[SYSTEM] Initializing screen capture...")
try:
    import dxcam
    screen_camera = dxcam.create(output_color="BGR")
    if screen_camera is None:
        raise RuntimeError("DXCam returned None")
    screen_camera.start(target_fps=30)
    print("[SYSTEM] DXCam ready.")
except Exception:
    print("[WARNING] DXCam unavailable. Using MSS capture.")
    import mss

    class MSSCamera:
        def __init__(self):
            self.local = threading.local()  # mss handles are per-thread

        def get_latest_frame(self):
            if not hasattr(self.local, "sct"):
                self.local.sct = mss.mss()
            sct = self.local.sct
            return np.array(sct.grab(sct.monitors[1]))[:, :, :3]

    screen_camera = MSSCamera()

slt_engine = None
try:
    import argus_slt
    slt_engine = argus_slt.SLTEngine(device)
    ready = [f"{l['label']} ({l['sign_language']})" for l in argus_slt.language_list() if l["ready"]]
    print(f"[SYSTEM] Sign language models ready: {', '.join(ready) or 'none'}")
except Exception as exc:
    argus_slt = None
    print(f"[WARNING] Sign language mode disabled ({type(exc).__name__}: {exc}). "
          "Install mediapipe and sentencepiece in this environment to enable it.")

argus_tts.warm_up("en")


def warm_up_models():
    """Run each model once at startup. The very first CUDA generate pays one-time costs
    (kernel loading, cuBLAS/cuDNN setup, vision encoder init) of 10-20 s on Windows, and
    without this the user's first question absorbs them."""
    if os.getenv("ARGUS_WARMUP", "1") != "1":
        return
    print("[SYSTEM] Warming up the models so the first reply is fast...")
    start = time.time()
    try:
        img = Image.new("RGB", (MAX_IMAGE_PX, MAX_IMAGE_PX * 9 // 16), (40, 40, 40))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img}, {"type": "text", "text": "What is on the screen?"}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(vlm.device)
        with vlm_lock, torch.inference_mode():
            vlm.generate(**inputs, max_new_tokens=4, do_sample=False)
    except Exception as exc:
        print(f"[WARNING] Model warm-up failed ({type(exc).__name__}: {exc}); the first reply may be slow.")
    try:
        segments, _ = audio_model.transcribe(np.zeros(16000, dtype=np.float32), language="en")
        list(segments)
    except Exception as exc:
        print(f"[WARNING] Whisper warm-up failed ({type(exc).__name__}: {exc}).")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        used = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"[SYSTEM] Warm-up done in {time.time() - start:.1f} s ({used:.1f} GB VRAM reserved by PyTorch).")
    else:
        print(f"[SYSTEM] Warm-up done in {time.time() - start:.1f} s.")


warm_up_models()


# ==========================================
# 3. SHARED STATE
# ==========================================
# Latest camera frame per source ("pc" = dashboard webcam, "phone" = Android app).
camera_frames = {}
camera_frames_lock = threading.Lock()

assistant_state = "idle"   # PC microphone: "idle" waits for "Hey Argus", "active" takes commands
mic_enabled = True
mic_available = None       # None until the microphone has been tried
system_is_running = True
last_first_audio_ms = None


@dataclass
class PromptRequest:
    text: str
    origin: str = "pc"         # "pc" (dashboard or PC mic) or "phone"
    requester: object = None   # the phone's WebSocket, for phone requests
    mode: str = "screen"
    ocr: bool = False
    image: object = None       # BGR frame sent with the request (OCR photo)
    t0: float = field(default_factory=time.time)  # end of speech, or when a typed prompt arrived
    stt_ms: int = None
    stream_audio: bool = True  # speak sentence by sentence (older phone apps play one clip per reply)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


prompt_queue = queue.Queue()
cancel_generation = threading.Event()


# ==========================================
# 4. LATENCY LOG
# ==========================================
class LatencyTracker:
    """Collects per-stage timings for each reply and writes them to logs/latency.csv.

    All times are milliseconds from the end of the user's speech (PC mic) or from
    when a typed / phone prompt reached the server.
    """
    FIELDS = ["time", "origin", "mode", "ocr", "image_px", "stt_ms", "queue_ms", "first_token_ms",
              "first_sentence_ms", "tts_synth_ms", "first_audio_ms", "generation_done_ms", "new_tokens"]

    def __init__(self, path):
        self.path = path
        self.records = {}
        self.lock = threading.Lock()

    def start(self, req, **values):
        with self.lock:
            self.records[req.id] = {"t0": req.t0, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "origin": req.origin, "mode": req.mode, "stt_ms": req.stt_ms, **values}

    def mark(self, req_id, key, when=None, value=None):
        with self.lock:
            rec = self.records.get(req_id)
            if rec is None or rec.get(key) is not None:
                return
            rec[key] = value if value is not None else int(((when or time.time()) - rec["t0"]) * 1000)

    def done(self, req_id, part):
        """part is "audio" (first audio started, or no audio) or "text" (generation finished).

        The row is written once both have happened, so it holds every stage.
        """
        with self.lock:
            rec = self.records.get(req_id)
            if rec is None:
                return
            rec.setdefault("_done", set()).add(part)
            ready = {"audio", "text"} <= rec["_done"]
        if ready:
            self.finish(req_id)

    def finish(self, req_id):
        global last_first_audio_ms
        with self.lock:
            rec = self.records.pop(req_id, None)
        if rec is None:
            return
        if rec.get("first_audio_ms") is not None:
            last_first_audio_ms = rec["first_audio_ms"]
        row = {k: rec.get(k) for k in self.FIELDS}
        ws_manager.broadcast_sync({"event": "latency", **row}, dashboards_only=True)
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            new = not os.path.exists(self.path)
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDS)
                if new:
                    writer.writeheader()
                writer.writerow(row)
        except OSError as exc:
            print(f"[LATENCY] Could not write {self.path}: {exc}")
        print("[LATENCY] " + " | ".join(f"{k}={row[k]}" for k in self.FIELDS[5:] if row[k] is not None))


latency = LatencyTracker(LATENCY_LOG)


# ==========================================
# 5. AUDIO OUTPUT (BEEPS & TTS WITH BARGE-IN)
# ==========================================
def play_beep(frequency=1800, duration=120):
    if winsound is None:
        return
    try:
        winsound.Beep(frequency, duration)
    except Exception:
        pass


def play_capture_chime():
    play_beep(2200, 80)
    time.sleep(0.04)
    play_beep(2600, 100)


def play_hazard_alert():
    for duration in (80, 80, 120):
        play_beep(1200, duration)
        time.sleep(0.03)


@dataclass
class SpeechItem:
    text: str
    lang: str = "en"
    rate: str = "+8%"
    req_id: str = None
    origin: str = "pc"
    requester: object = None
    seq: int = 0
    epoch: int = 0


# Interrupting bumps the epoch; audio from an older epoch is thrown away.
tts_epoch = 0
pc_synth_queue = queue.Queue()
pc_play_queue = queue.Queue(maxsize=4)
phone_tts_queue = queue.Queue()
tts_playing = threading.Event()  # set while PC speech is queued or playing
pc_generating = threading.Event()  # set while a PC reply is being generated
pending_pc_speech = 0
pending_pc_lock = threading.Lock()
last_tts_end = 0.0
last_tts_start = 0.0
recent_speech = deque(maxlen=30)  # (time, normalized text) of what ARGUS said, for the echo filter


def pc_busy():
    """True while ARGUS is answering on the PC; the mic then only listens for 'stop' and 'goodbye'."""
    return tts_playing.is_set() or pc_generating.is_set()


def _pc_speech_done(count=1):
    global pending_pc_speech, last_tts_end
    with pending_pc_lock:
        pending_pc_speech = max(0, pending_pc_speech - count)
        if pending_pc_speech == 0 and tts_playing.is_set():
            tts_playing.clear()
            last_tts_end = time.time()
            set_status("listening" if assistant_state == "active" else "idle")


def speak(text, req=None, lang="en", rate="+8%", seq=0, origin=None, requester=None):
    """Queue one sentence. PC replies play on this PC; phone replies are sent to the phone."""
    origin = origin or (req.origin if req else "pc")
    requester = requester if requester is not None else (req.requester if req else None)
    item = SpeechItem(text, lang, rate, req.id if req else None, origin, requester, seq, tts_epoch)
    global pending_pc_speech
    if origin == "phone" and requester is not None:
        phone_tts_queue.put(item)
    else:
        with pending_pc_lock:
            pending_pc_speech += 1
            tts_playing.set()
        pc_synth_queue.put(item)


def _drain(q):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return


def interrupt_tts():
    """Stop talking now: drop queued sentences and stop the reply being generated."""
    global tts_epoch
    tts_epoch += 1
    cancel_generation.set()
    for q in (pc_synth_queue, pc_play_queue, phone_tts_queue):
        _drain(q)
    try:
        pygame.mixer.music.stop()
    except Exception:
        pass
    _pc_speech_done(count=10 ** 6)


def clear_pending_prompts():
    _drain(prompt_queue)


def pc_synth_worker():
    """Synthesizes the next sentence while the previous one is still playing."""
    while system_is_running:
        item = pc_synth_queue.get()
        if item.epoch != tts_epoch:
            continue  # interrupt_tts already reset the pending count
        try:
            start = time.time()
            audio, mime = argus_tts.synthesize(item.text, item.lang, item.rate)
            if item.seq == 0 and item.req_id:
                latency.mark(item.req_id, "tts_synth_ms", value=int((time.time() - start) * 1000))
            if item.epoch == tts_epoch:
                pc_play_queue.put((item, audio, mime))
        except Exception as exc:
            print(f"[TTS ERROR] {type(exc).__name__}: {exc}")
            if item.req_id:
                latency.done(item.req_id, "audio")
            _pc_speech_done()


def pc_play_worker():
    global last_tts_start
    while system_is_running:
        item, audio, mime = pc_play_queue.get()
        if item.epoch != tts_epoch:
            continue
        suffix = ".wav" if mime == "audio/wav" else ".mp3"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            tmp.write(audio)
            tmp.close()
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(tmp.name)
            set_status("speaking")
            last_tts_start = time.time()
            recent_speech.append((last_tts_start, normalize_command_text(item.text)))
            pygame.mixer.music.play()
            if item.seq == 0 and item.req_id:
                latency.mark(item.req_id, "first_audio_ms")
                latency.done(item.req_id, "audio")
            while pygame.mixer.music.get_busy():
                if item.epoch != tts_epoch:
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.02)
        except Exception as exc:
            print(f"[TTS PLAYBACK ERROR] {type(exc).__name__}: {exc}")
        finally:
            try:
                pygame.mixer.music.unload()
            except Exception:
                pass
            try:
                os.remove(tmp.name)
            except OSError:
                pass
            if item.epoch == tts_epoch:
                _pc_speech_done()


def phone_tts_worker():
    while system_is_running:
        item = phone_tts_queue.get()
        if item.epoch != tts_epoch:
            continue
        try:
            start = time.time()
            audio, mime = argus_tts.synthesize(item.text, item.lang, item.rate)
            ws_manager.send_to_sync(item.requester, {
                "event": "tts_audio", "mime_type": mime, "request_id": item.req_id, "seq": item.seq,
                "audio_base64": base64.b64encode(audio).decode("ascii")})
            if item.seq == 0 and item.req_id:
                latency.mark(item.req_id, "tts_synth_ms", value=int((time.time() - start) * 1000))
                latency.mark(item.req_id, "first_audio_ms")  # sent to the phone; playback starts on arrival
                latency.done(item.req_id, "audio")
        except Exception as exc:
            print(f"[PHONE TTS ERROR] {type(exc).__name__}: {exc}")
            if item.req_id:
                latency.done(item.req_id, "audio")


# ==========================================
# 6. SPEECH COMMANDS
# ==========================================
def normalize_command_text(text):
    clean = text.lower().strip().replace("’", "'")
    clean = clean.translate(str.maketrans("", "", string.punctuation))
    return " ".join(clean.split())


# Hint that helps Whisper spell "Argus". On noise Whisper sometimes just repeats this hint,
# which would look like a wake word followed by a question, so those transcripts are dropped.
WHISPER_PROMPT = "Hey Argus."


def is_prompt_echo(text):
    clean = normalize_command_text(text)
    words = clean.split()
    return bool(words) and all(w in {"hey", "argus", "assistant", "a", "r", "g", "u", "s"} for w in words) \
        and len(words) > 2


WHISPER_HALLUCINATIONS = {"thank you", "thanks", "thanks for watching", "subscribe", "you",
                          "okay", "ok", "amaraorg", "yeah", "you know", "bye", "so", "um", "uh", "hmm",
                          "thank you for watching", "please subscribe", "thank you very much",
                          "subtitles by the amaraorg community", "argus assistant"}


def is_hallucination(text):
    clean = normalize_command_text(text)
    return clean in WHISPER_HALLUCINATIONS or len(clean) < 2


ARGUS_PHONETIC_VARIANTS = r"argus|argas|argos|august|aragus|argis|arkus|ar\s+gus|our\s+guess|are\s+guess|artis"
WAKE_PREFIXES = r"hey|hi|hello|ok|okay|yo"
STOP_PHRASES = {"stop", "stop it", "stop stop", "no stop", "wait", "argus stop", "ok stop", "okay stop", "stop talking", "stop speaking", "be quiet", "quiet", "shut up",
                "cancel", "hold on", "pause", "enough", "thats enough", "please stop", "okay thats enough",
                "ok thats enough", "please be quiet"}
SLEEP_PHRASES = {"bye", "goodbye", "good bye", "sleep", "go to sleep", "thats all", "im done", "im finished",
                 "end conversation", "okay bye", "ok bye", "okay goodbye", "ok goodbye"}


def is_stop_command(text):
    clean = normalize_command_text(text)
    if clean in STOP_PHRASES:
        return True
    pattern = (rf"^(?:(?:hey\s+)?(?:{ARGUS_PHONETIC_VARIANTS})\s+)?(?:please\s+)?"
               r"(?:stop|stop talking|stop speaking|be quiet|quiet|cancel|pause)(?:\s+now|\s+please)?$")
    return bool(re.fullmatch(pattern, clean))


def is_sleep_command(text):
    clean = normalize_command_text(text)
    if clean in SLEEP_PHRASES:
        return True
    pattern = (rf"^(?:(?:hey\s+|ok\s+|okay\s+)?(?:{ARGUS_PHONETIC_VARIANTS})\s+)?"
               rf"(?:goodbye|good bye|bye|sleep|go to sleep)(?:\s+(?:{ARGUS_PHONETIC_VARIANTS}))?$")
    return bool(re.fullmatch(pattern, clean))


def detect_wake_call(text):
    clean = normalize_command_text(text)
    if not clean:
        return False, ""
    match = re.match(rf"^(?:(?:{WAKE_PREFIXES})\s+)?(?:{ARGUS_PHONETIC_VARIANTS})(?:\s+|$)(.*)$", clean)
    if match:
        return True, " ".join(match.group(1).split())
    return False, ""


def is_own_echo(heard, window_s=30.0):
    """True if the mic most likely picked up ARGUS's own voice from the speakers."""
    clean = normalize_command_text(heard)
    words = clean.split()
    if not words:
        return False
    now = time.time()
    recent = [text for t, text in list(recent_speech) if now - t <= window_s and text]
    if not recent:
        return False
    joined = " ".join(recent)
    if any(SequenceMatcher(None, clean, r).ratio() >= 0.6 for r in recent):
        return True
    if len(clean) >= 12 and SequenceMatcher(None, clean, joined).find_longest_match(
            0, len(clean), 0, len(joined)).size >= 0.6 * len(clean):
        return True
    spoken = set(joined.split())
    return len(words) >= 3 and sum(w in spoken for w in words) / len(words) >= 0.75


def sanitize_transcription_for_ui(text):
    return re.sub(ARGUS_PHONETIC_VARIANTS, "Argus", text, flags=re.IGNORECASE)


# Questions that need the model to read text: answered from the full-resolution image.
OCR_PATTERN = re.compile(
    r"\b(read|reading|what does (it|this|that) say|what('s| is) written|written|text|label|sign says|"
    r"how much (money|is this)|money|cash|taka|banknote|bank note|note is this|bill|price|"
    r"medicine|expiry|expiration|dose|dosage|ingredients|equation|formula|number)\b")
CURRENCY_PATTERN = re.compile(r"\b(money|cash|taka|banknote|bank note|note is this|how much money|bill)\b")


def detect_ocr_intent(text):
    return bool(OCR_PATTERN.search(normalize_command_text(text)))


def is_currency_question(text):
    return bool(CURRENCY_PATTERN.search(normalize_command_text(text)))


def normalize_mode(mode):
    mode = (mode or "").lower().strip()
    mode = MODE_ALIASES.get(mode, mode)
    return mode if mode in MODES else None


# ==========================================
# 7. VISION-LANGUAGE ANSWERS
# ==========================================
NO_DIALOGUE = ("Only answer the user's actual request. Never write a 'User:' or 'Assistant:' dialogue, "
               "never ask or answer questions on the user's behalf, and do not continue the conversation.")

SYSTEM_PROMPTS = {
    "screen": ("You are A.R.G.U.S., a friendly copilot looking at the user's screen, talking casually like a "
               "friend on a call. Answer in one or two short spoken sentences. Never use markdown, bullet "
               "points, or phrases like 'Certainly', 'As an AI' or 'I can see'. " + NO_DIALOGUE),
    "live": ("You are A.R.G.U.S., a friendly companion looking through the user's camera, talking casually "
             "like on a video call. Answer in one or two short spoken sentences. Never use markdown, lists "
             "or robotic AI phrases. " + NO_DIALOGUE),
    "assistive": ("You are A.R.G.U.S. Assistive Voice for a visually impaired user. Describe the key objects, "
                  "read exact labels, and call out any immediate physical hazard first. Be direct, calm and "
                  "concise (one or two sentences). " + NO_DIALOGUE),
    "ocr": ("You are A.R.G.U.S. reading text for the user. Read the requested text, numbers or label exactly as "
            "printed in the image, then answer the question in one or two plain spoken sentences. If part of "
            "the text is too blurry or cut off, say which part you cannot read instead of guessing. Never use "
            "markdown. " + NO_DIALOGUE),
    "currency": ("You are a strict currency reader for BANGLADESHI TAKA (BDT) only. State the value of the "
                 "banknote in exactly one short sentence, for example 'This is a 20 Taka note.' Do not describe "
                 "the person, background or room, and never use markdown. If the numbers are unreadable, say "
                 "'Please hold the note steady in better light.'"),
}

GROUNDING_RULES = ("The image attached to this message is the only visual evidence. Do not rely on earlier "
                   "images or answers. Answer using only details actually visible in this image; if the image "
                   "is unclear or the requested content cannot be read, say so plainly.")

HAZARD_PATTERN = re.compile(r"\b(danger|dangerous|hazard|caution|careful|watch out|warning|stairs ahead|"
                            r"step down|step up|obstacle|traffic|oncoming)\b", re.IGNORECASE)
# The model sometimes keeps going and writes both sides of a conversation: cut at the first
# speaker label after the start of the reply.
DIALOGUE_CUT = re.compile(r"(?:^|\n|\s)(?:User|Human|Assistant|You|Me|ARGUS|A\.R\.G\.U\.S\.?)\s*:", re.IGNORECASE)
LEADING_LABEL = re.compile(r"^\s*(?:Assistant|ARGUS|A\.R\.G\.U\.S\.?)\s*:\s*", re.IGNORECASE)


def dialogue_cut(text):
    """Index where an invented dialogue starts in text, or None."""
    lead = LEADING_LABEL.match(text)
    start = lead.end() if lead else 0
    m = DIALOGUE_CUT.search(text, start)
    return m.start() if m else None
# A sentence is ready to speak once it ends in . ! ? (or the Bangla danda) followed by a space.
# Sentences shorter than 12 characters ("Hmm...") are merged with the next one.
SENTENCE_END = re.compile(r"(.{12,}?[.!?\u0964][\"')\]]*)(\s+)", re.S)


def decode_image(data_url):
    """Base64 JPEG/PNG (optionally a data: URL) -> BGR array, or None."""
    if not data_url:
        return None
    if "," in data_url[:100]:
        data_url = data_url.split(",", 1)[1]
    try:
        arr = np.frombuffer(base64.b64decode(data_url), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def store_camera_frame(origin, frame):
    with camera_frames_lock:
        camera_frames[origin] = (frame, time.time())


def latest_camera_frame(origin):
    """Freshest camera frame, preferring the requester's own camera. None if all are stale."""
    with camera_frames_lock:
        candidates = dict(camera_frames)
    now = time.time()
    order = [origin] + [o for o in candidates if o != origin]
    for o in order:
        if o in candidates:
            frame, ts = candidates[o]
            if now - ts <= FRAME_MAX_AGE_S:
                return frame.copy(), o
    return None, None


def pick_frame(req):
    if req.image is not None:
        return req.image, "request"
    if req.mode == "screen":
        frame = screen_camera.get_latest_frame()
        if frame is None:
            time.sleep(0.05)
            frame = screen_camera.get_latest_frame()
        return frame, "screen"
    return latest_camera_frame(req.origin)


class CancelCriteria(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        return cancel_generation.is_set()


def clean_reply(text):
    cut = dialogue_cut(text)
    if cut is not None:
        text = text[:cut]
    text = LEADING_LABEL.sub("", text)
    return text.strip().strip('"').strip()


def send_reply(req, payload):
    ws_manager.reply_to(req.origin, req.requester, payload)


def answer_with_error(req, message):
    send_reply(req, {"event": "argus_reply", "id": req.id, "text": message, "mode": req.mode})
    latency.start(req)
    latency.done(req.id, "text")
    speak(message, req)


def process_prompt(req: PromptRequest):
    if req.origin != "phone":
        pc_generating.set()
    try:
        _process_prompt(req)
    finally:
        pc_generating.clear()


def _process_prompt(req: PromptRequest):
    cancel_generation.clear()
    queue_ms = int((time.time() - req.t0) * 1000)
    frame, source = pick_frame(req)
    if frame is None:
        msg = ("I can't see the screen right now." if req.mode == "screen" else
               "I can't see the camera right now. Turn the camera on and ask again.")
        answer_with_error(req, msg)
        return

    currency = req.mode == "assistive" and is_currency_question(req.text)
    ocr = req.ocr or currency or detect_ocr_intent(req.text)
    prompt_key = "currency" if currency else ("ocr" if ocr else ("live" if req.mode == "slt" else req.mode))
    max_px = OCR_MAX_PX if ocr else MAX_IMAGE_PX

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pil_image.thumbnail((max_px, max_px), Image.LANCZOS)
    task = req.text or ("Read the text in front of me." if ocr else "What is in front of me?")
    log_event("ROUTER", f"{req.mode} | {'OCR ' if ocr else ''}{pil_image.width}x{pil_image.height} from {source}")

    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_image},
        {"type": "text", "text": f"{SYSTEM_PROMPTS[prompt_key]}\n\n{GROUNDING_RULES}\n"
                                 f"Current user request: {task}\nRespond only to this request."},
    ]}]
    latency.start(req, ocr=int(ocr), image_px=max(pil_image.size), queue_ms=queue_ms)
    set_status("thinking")
    send_reply(req, {"event": "argus_reply_start", "id": req.id, "mode": req.mode, "ocr": ocr})

    rate = "+15%" if req.mode == "assistive" else "+8%"
    with vlm_lock:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(vlm.device)
        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
        gen_kwargs = dict(**inputs, streamer=streamer, do_sample=False, repetition_penalty=1.1,
                          max_new_tokens=OCR_MAX_NEW_TOKENS if ocr else MAX_NEW_TOKENS,
                          stopping_criteria=StoppingCriteriaList([CancelCriteria()]))
        gen_error = []

        def run_generate():
            try:
                with torch.inference_mode():
                    vlm.generate(**gen_kwargs)
            except Exception as exc:
                gen_error.append(exc)
                streamer.end()

        worker = threading.Thread(target=run_generate, daemon=True)
        worker.start()

        full, pending, seq, n_chunks = "", "", 0, 0
        for chunk in streamer:
            if n_chunks == 0:
                latency.mark(req.id, "first_token_ms")
            n_chunks += 1
            full += chunk
            if dialogue_cut(full) is not None:  # the model started writing a fake dialogue: stop there
                cancel_generation.set()
                pending += chunk
                break
            send_reply(req, {"event": "argus_reply_delta", "id": req.id, "text": chunk})
            pending += chunk
            # speak each finished sentence while the rest is still being generated
            while True:
                m = SENTENCE_END.match(pending)
                if not m or not req.stream_audio:
                    break
                sentence = clean_reply(m.group(1))
                pending = pending[m.end():]
                if sentence and not cancel_generation.is_set():
                    if seq == 0:
                        latency.mark(req.id, "first_sentence_ms")
                    speak(sentence, req, rate=rate, seq=seq)
                    seq += 1
        worker.join()
        latency.mark(req.id, "generation_done_ms")
        latency.mark(req.id, "new_tokens", value=n_chunks)
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if gen_error:
        print("[VLM ERROR]", "".join(traceback.format_exception(gen_error[0])))
    interrupted = cancel_generation.is_set() and dialogue_cut(full) is None
    if interrupted:
        latency.finish(req.id)
        return

    answer = clean_reply(full) or "Sorry, I didn't get a clear answer. Please ask me again."
    tail = clean_reply(pending)
    if tail:
        if seq == 0:
            latency.mark(req.id, "first_sentence_ms")
        speak(tail, req, rate=rate, seq=seq)
    elif seq == 0:
        speak(answer, req, rate=rate, seq=0)

    if req.mode == "assistive" and HAZARD_PATTERN.search(answer):
        play_hazard_alert()

    send_reply(req, {"event": "argus_reply", "id": req.id, "text": answer, "mode": req.mode, "ocr": ocr})
    log_event("Qwen2-VL", f"{'Phone reply' if req.origin == 'phone' else 'Reply'}: {answer}")
    latency.done(req.id, "text")


def translate_caption_to_english(text, source_label):
    """Short text-only Qwen call for SLT captions (about 100-300 ms)."""
    messages = [{"role": "user", "content": [{"type": "text", "text": (
        f"Translate this {source_label} sentence into natural English. Reply with the English sentence only.\n\n"
        f"{text}")}]}]
    with vlm_lock:
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(vlm.device)
        with torch.inference_mode():
            out = vlm.generate(**inputs, max_new_tokens=60, do_sample=False)
        reply = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return clean_reply(reply.split("\n")[0])


# ==========================================
# 8. SIGN LANGUAGE (SLT) SESSIONS
# ==========================================
def start_slt(client, language="de"):
    if argus_slt is None:
        ws_manager.send_to_sync(client.ws, {"event": "slt_error", "text": (
            "Sign language mode isn't available: install mediapipe and sentencepiece in the server environment.")})
        return
    if client.slt is None:
        client.slt = argus_slt.SignSession(slt_engine, lambda ev, c=client: on_slt_event(c, ev), language)
    ws_manager.send_to_sync(client.ws, {"event": "slt_languages", "languages": argus_slt.language_list(),
                                        "language": client.slt.language})


def stop_slt(client):
    if client.slt is not None:
        client.slt.close()
        client.slt = None


def on_slt_event(client, event):
    ws_manager.send_to_sync(client.ws, event)
    if event.get("event") != "slt_caption":
        return
    if client.kind == "phone":  # keep the dashboard's log in sync with what the phone translated
        log_event("SLT", f"Phone: {event['text']}")
    threading.Thread(target=finish_caption, args=(client, event), daemon=True).start()


def finish_caption(client, event):
    lang = argus_slt.LANGUAGES[event["language"]]
    english = None
    if SLT_ENGLISH and lang.code != "en" and event["text"]:
        try:
            start = time.time()
            english = translate_caption_to_english(event["text"], lang.label)
            ws_manager.send_to_sync(client.ws, {"event": "slt_caption_english", "text": english,
                                                "source": event["text"], "ms": int((time.time() - start) * 1000)})
        except Exception as exc:
            print(f"[SLT] English translation failed: {exc}")
    log_event("SLT", f"{lang.sign_language}: {event['text']}" + (f" | EN: {english}" if english else ""))
    if SLT_SPEAK == "off" or not event["text"]:
        return
    if SLT_SPEAK == "en" and english:
        text, tts_lang = english, "en"
    else:
        text, tts_lang = event["text"], lang.tts_lang
    origin = "phone" if client.kind == "phone" else "pc"
    speak(text, lang=tts_lang, origin=origin, requester=client.ws if origin == "phone" else None)


# ==========================================
# 9. BACKGROUND WORKERS
# ==========================================
def hardware_metrics_worker():
    while system_is_running:
        try:
            temp, used_gb = "--", None
            if NVML_HANDLE is not None:
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(NVML_HANDLE, pynvml.NVML_TEMPERATURE_GPU)
                    used_gb = pynvml.nvmlDeviceGetMemoryInfo(NVML_HANDLE).used / 1024 ** 3  # includes Whisper
                except Exception:
                    pass
            if used_gb is None and torch.cuda.is_available():
                used_gb = torch.cuda.memory_reserved() / 1024 ** 3
            ws_manager.broadcast_sync({
                "event": "system_metrics",
                "gpu_name": GPU_NAME,
                "vram_used": round(used_gb or 0.0, 1),
                "vram_total": round(GPU_TOTAL_GB, 1),
                "temperature": temp,
                "latency": last_first_audio_ms if last_first_audio_ms is not None else "--",
            })
        except Exception:
            pass
        time.sleep(1.0)


def audio_worker():
    global assistant_state, mic_available

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 280
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = PAUSE_THRESHOLD
    recognizer.non_speaking_duration = min(0.3, PAUSE_THRESHOLD)

    try:
        mic = sr.Microphone()
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1.2)
        print("[AUDIO] Microphone calibrated and active.")
        mic_available = True
    except Exception as e:
        print(f"[AUDIO ERROR] No microphone detected: {e}")
        mic_available = False
        ws_manager.broadcast_sync({"event": "mic_status", "available": False, "enabled": False})
        return

    def transcribe(audio_data):
        pcm = np.frombuffer(audio_data.get_raw_data(convert_rate=16000, convert_width=2), np.int16)
        segments, _ = audio_model.transcribe(
            pcm.astype(np.float32) / 32768.0, beam_size=2, best_of=1, language="en", vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 250}, condition_on_previous_text=False,
            temperature=0.0, no_speech_threshold=0.6, initial_prompt=WHISPER_PROMPT)
        kept = []
        for seg in segments:
            # Whisper invents text for noise or silence; those segments look unsure or "no speech".
            if seg.no_speech_prob > 0.5 or seg.avg_logprob < -1.0 or seg.compression_ratio > 2.4:
                print(f"[VOICE] Dropped unsure segment {seg.text.strip()!r} (no_speech={seg.no_speech_prob:.2f}, "
                      f"logprob={seg.avg_logprob:.2f})")
                continue
            kept.append(seg.text.strip())
        text = " ".join(kept).strip()
        if is_prompt_echo(text):
            print(f"[VOICE] Dropped {text!r}: Whisper repeated its own prompt (no one spoke)")
            return ""
        return text

    def handle_stop_or_sleep(heard):
        global assistant_state
        if is_sleep_command(heard):
            interrupt_tts()
            clear_pending_prompts()
            assistant_state = "idle"
            set_status("idle")
            log_event("SYSTEM", "Going to sleep. Say 'Hey Argus' to wake me.")
            return True
        if is_stop_command(heard):
            interrupt_tts()
            clear_pending_prompts()
            assistant_state = "active"
            set_status("listening")
            log_event("SYSTEM", "Stopped speaking. Listening.")
            return True
        return False

    while system_is_running:
        try:
            if not mic_enabled:
                time.sleep(0.2)
                continue

            if ECHO_GUARD_S > 0 and not pc_busy() and time.time() - last_tts_end < ECHO_GUARD_S:
                time.sleep(0.02)  # optional pause for loudspeaker setups (0 by default)
                continue

            # The mic is always open, also while ARGUS is talking, so you can interrupt at any time.
            idle = assistant_state == "idle"
            try:
                with mic as source:
                    audio = recognizer.listen(source, timeout=8.0 if not idle else 2.0,
                                              phrase_time_limit=12.0 if not idle else 6.0)
            except sr.WaitTimeoutError:
                continue
            now = time.time()
            speech_end = now - PAUSE_THRESHOLD
            # When did this phrase start? (the clip runs from the first sound to the end of the pause)
            clip_s = len(audio.frame_data) / float(audio.sample_rate * audio.sample_width)
            phrase_start = now - clip_s
            # Was ARGUS talking at any point while this phrase was spoken?
            overlapped = pc_busy() or last_tts_end >= phrase_start

            stt_start = time.time()
            heard = transcribe(audio)
            stt_ms = int((time.time() - stt_start) * 1000)
            if not heard:
                continue
            print(f"[VOICE] Heard {heard!r} ({stt_ms} ms, state={assistant_state}"
                  f"{', while ARGUS was talking' if overlapped else ''})")

            if handle_stop_or_sleep(heard):
                continue

            if is_own_echo(heard):
                print(f"[VOICE] Ignored (sounds like ARGUS's own reply): {heard!r}")
                continue
            if overlapped and BARGE_IN != "all":
                continue  # ARGUS_BARGE_IN=stop: only "stop" / "goodbye" interrupt

            if assistant_state == "idle":
                is_wake, remainder = detect_wake_call(heard)
                if not is_wake:
                    continue
                assistant_state = "active"
                play_capture_chime()
                set_status("listening")
                log_event("SYSTEM", "Awake.")
                if not remainder:
                    ws_manager.broadcast_sync({"event": "speech_recognized", "text": "Hey Argus"},
                                              dashboards_only=True)
                    continue
                heard = remainder

            if is_hallucination(heard):
                continue

            if overlapped and pc_busy():
                # You spoke over ARGUS: stop the current answer and take the new question.
                interrupt_tts()
                clear_pending_prompts()
                log_event("SYSTEM", "Interrupted. Answering your new question.")

            play_capture_chime()
            ws_manager.broadcast_sync({"event": "speech_recognized", "text": sanitize_transcription_for_ui(heard)},
                                      dashboards_only=True)
            mode = ws_manager.pc_mode if ws_manager.pc_mode != "slt" else "live"
            prompt_queue.put(PromptRequest(heard, "pc", None, mode, ocr=detect_ocr_intent(heard),
                                           t0=speech_end, stt_ms=stt_ms))
        except Exception as e:
            print(f"[AUDIO WORKER ERROR] {type(e).__name__}: {e}")
            time.sleep(0.2)


def llm_worker():
    while system_is_running:
        req = prompt_queue.get()
        try:
            process_prompt(req)
        except Exception:
            traceback.print_exc()
            answer_with_error(req, "Sorry, something went wrong while answering.")
        finally:
            if not pc_busy():
                set_status("listening" if assistant_state == "active" else "idle")


def screen_stream_worker():
    """Live preview of the PC screen for the dashboard (the model always gets the full frame)."""
    while system_is_running:
        if ws_manager.pc_mode == "screen" and ws_manager.has_dashboard():
            frame = screen_camera.get_latest_frame()
            if frame is not None:
                preview = cv2.resize(frame, (960, int(960 * frame.shape[0] / frame.shape[1])))
                ok, buffer = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                if ok:
                    ws_manager.broadcast_sync({
                        "event": "screen_frame",
                        "image_data": "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii"),
                    }, dashboards_only=True)
        time.sleep(0.1)


for target in (pc_synth_worker, pc_play_worker, phone_tts_worker, audio_worker, llm_worker,
               screen_stream_worker, hardware_metrics_worker):
    threading.Thread(target=target, daemon=True).start()
print("[VOICE] PC wake word: 'Hey Argus' | Stop: 'Okay stop' | Sleep: 'Goodbye' | OCR: 'Read this'")


# ==========================================
# 10. ROUTES & WEBSOCKET
# ==========================================
@app.get("/")
async def serve_dashboard():
    html_path = os.path.join(BASE_DIR, "new.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>new.html not found. Keep it next to allnew.py.</h1>", status_code=500)


def hello_payload(client):
    return {
        "event": "hello",
        "modes": list(MODES),
        "pc_mode": ws_manager.pc_mode,
        "mode": client.mode,
        "mic_enabled": mic_enabled,
        "mic_available": mic_available is not False,
        "gpu_name": GPU_NAME,
        "vram_total": round(GPU_TOTAL_GB, 1),
        "slt_available": argus_slt is not None,
        "slt_languages": argus_slt.language_list() if argus_slt else [],
        "max_image_px": MAX_IMAGE_PX,
        "ocr_max_px": OCR_MAX_PX,
    }


async def handle_message(client, data):
    global mic_enabled
    event = data.get("event")
    ws = client.ws
    if data.get("origin") == "phone" or data.get("client") == "phone":
        client.kind = "phone"

    if event == "hello":
        client.audio_queue = "audio_queue" in (data.get("features") or [])
        if client.kind == "phone":
            client.mode = normalize_mode(data.get("mode")) or "live"
        await ws.send_text(json.dumps(hello_payload(client)))

    elif event == "set_mode":
        mode = normalize_mode(data.get("mode"))
        if mode is None:
            await ws.send_text(json.dumps({"event": "perception_log", "engine": "SYSTEM",
                                           "text": f"Unknown mode {data.get('mode')!r}."}))
            return
        client.mode = mode
        if client.kind == "dashboard":
            ws_manager.pc_mode = mode
            for other in list(ws_manager.clients.values()):  # every dashboard follows the PC mode
                if other.kind == "dashboard":
                    other.mode = mode
                    if mode != "slt":
                        stop_slt(other)
            ws_manager.broadcast_sync({"event": "mode_changed", "mode": mode}, dashboards_only=True)
        else:
            ws_manager.send_to_sync(ws, {"event": "mode_changed", "mode": mode})
        if mode == "slt":
            start_slt(client, data.get("language", "de"))
        else:
            stop_slt(client)
        print(f"[SYSTEM] {client.kind.capitalize()} mode -> {mode.upper()}")

    elif event == "set_mic":
        mic_enabled = bool(data.get("enabled", True))
        if not mic_enabled:
            interrupt_tts()
        ws_manager.broadcast_sync({"event": "mic_status", "available": True, "enabled": mic_enabled},
                                  dashboards_only=True)

    elif event == "video_frame":
        frame = decode_image(data.get("image_data", ""))
        if frame is not None:
            store_camera_frame("phone" if client.kind == "phone" else "pc", frame)

    elif event == "slt_frame":
        frame = decode_image(data.get("image_data", ""))
        if frame is None:
            return
        store_camera_frame("phone" if client.kind == "phone" else "pc", frame)
        if client.slt is None and client.mode == "slt":
            start_slt(client)
        if client.slt is not None:
            ts = data.get("ts")
            client.slt.push_frame(frame, ts / 1000.0 if isinstance(ts, (int, float)) else None)

    elif event == "slt_language":
        code = data.get("language", "de")
        langs = {l["code"]: l for l in (argus_slt.language_list() if argus_slt else [])}
        if code not in langs:
            return
        if not langs[code]["ready"]:
            await ws.send_text(json.dumps({"event": "slt_error", "text": (
                f"The {langs[code]['label']} sign model isn't ready yet. It turns on once its checkpoint is added.")}))
            return
        if client.slt:
            client.slt.set_language(code)
        await ws.send_text(json.dumps({"event": "slt_languages", "languages": list(langs.values()),
                                       "language": code}))

    elif event == "slt_control" and client.slt is not None:
        if data.get("action") == "translate":
            client.slt.translate_now()
        elif data.get("action") == "clear":
            client.slt.clear()

    elif event == "stop":
        interrupt_tts()
        clear_pending_prompts()

    elif event == "user_prompt":
        text = (data.get("text") or "").strip()
        if not text:
            return
        mode = normalize_mode(data.get("mode")) or client.mode
        if mode == "slt":
            mode = "live"
        origin = "phone" if client.kind == "phone" else "pc"
        image = decode_image(data.get("image_data", ""))
        ocr = bool(data.get("ocr")) or detect_ocr_intent(text)
        print(f"[PROMPT] {origin} ({mode}{', OCR' if ocr else ''}): {text}")
        interrupt_tts()
        clear_pending_prompts()
        if origin == "pc":  # show typed questions on every open dashboard
            ws_manager.broadcast_sync({"event": "speech_recognized", "text": text, "typed": True},
                                      dashboards_only=True)
        prompt_queue.put(PromptRequest(text, origin, ws if origin == "phone" else None, mode, ocr, image,
                                       stream_audio=origin == "pc" or client.audio_queue))

    # "trigger_lecture_evaluation" from older clients is ignored: Lecture mode was replaced by SLT.


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    client = ws_manager.clients[websocket]
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            try:
                await handle_message(client, data)
            except Exception:
                traceback.print_exc()
    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    print("\n[SYSTEM] A.R.G.U.S. running at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("ARGUS_PORT", "8000")), ws_max_size=32 * 1024 * 1024)
