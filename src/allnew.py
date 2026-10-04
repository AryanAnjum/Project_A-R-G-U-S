import asyncio
import base64
import collections
import json
import os
import queue
import tempfile
import threading
import time
import string
import re
from PIL import Image
from contextlib import asynccontextmanager

import cv2
import edge_tts
import numpy as np
import pygame
import torch
import speech_recognition as sr

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from faster_whisper import WhisperModel

try:
    import winsound
except ImportError:
    winsound = None

# --- NVIDIA Hardware Monitoring ---
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("[WARNING] nvidia-ml-py not installed. GPU Temperature will be mocked.")

# Initialize audio mixer
pygame.mixer.init()

# ==========================================
# TRANSFORMERS COMPATIBILITY PATCH
# ==========================================
_orig_getattr = torch.nn.Module.__getattr__
def _patched_getattr(self, name):
    if name == "all_tied_weights_keys":
        return {} 
    return _orig_getattr(self, name)
torch.nn.Module.__getattr__ = _patched_getattr


# ==========================================
# 1. FASTAPI & WEBSOCKET ENGINE SETUP
# ==========================================
class ConnectionManager:
    def __init__(self):
        self.active_connections = []
        self.loop = None
        self.current_mode = "screen"

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[SERVER] Web Dashboard Connected (Active Mode: {self.current_mode.upper()})")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print("[SERVER] Web Dashboard Disconnected")

    def broadcast_sync(self, message: dict):
        if self.loop and self.active_connections:
            asyncio.run_coroutine_threadsafe(self._broadcast(message), self.loop)

    def send_to_sync(self, websocket, message: dict):
        """Send an event only to the WebSocket that initiated a request."""
        if self.loop and websocket is not None:
            asyncio.run_coroutine_threadsafe(self._send_to(websocket, message), self.loop)

    async def _send_to(self, websocket, message: dict):
        try:
            if websocket in self.active_connections:
                await websocket.send_text(json.dumps(message))
        except Exception:
            pass

    async def _broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                pass

ws_manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    ws_manager.loop = asyncio.get_running_loop()
    yield

app = FastAPI(title="A.R.G.U.S. Multi-System Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. LOAD AI MODELS & SENSORS
# ==========================================
print("\n[SYSTEM] Booting A.R.G.U.S. Multimodal Engine...")
device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERAL_ADAPTER_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "qwen2-vl-argus-200gb-checkpoints", "final_adapter"))

print("[SYSTEM] Loading Base Qwen2-VL 2B Backbone (Frozen)...")
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", 
    torch_dtype=torch.bfloat16, 
    device_map="cuda"
)

print("[SYSTEM] Injecting Adapter 1: General Intelligence & Tutor Logic...")
spatial_model = PeftModel.from_pretrained(
    base_model, 
    GENERAL_ADAPTER_PATH, 
    adapter_name="argus_general"
)
spatial_model.eval()

try:
    spatial_processor = AutoProcessor.from_pretrained(GENERAL_ADAPTER_PATH)
    print("[SYSTEM] Custom A.R.G.U.S. processor loaded successfully.")
except Exception:
    print("[WARNING] Loading default base processor fallback...")
    spatial_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

WHISPER_MODEL_NAME = os.getenv("ARGUS_WHISPER_MODEL", "small.en")
_default_whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
WHISPER_DEVICE = os.getenv("ARGUS_WHISPER_DEVICE", _default_whisper_device).lower()
_default_whisper_compute = "float16" if WHISPER_DEVICE == "cuda" else "int8"
WHISPER_COMPUTE_TYPE = os.getenv("ARGUS_WHISPER_COMPUTE_TYPE", _default_whisper_compute)

print(
    f"[SYSTEM] Loading Whisper model {WHISPER_MODEL_NAME} "
    f"(device={WHISPER_DEVICE}, compute={WHISPER_COMPUTE_TYPE})..."
)
audio_model = WhisperModel(
    WHISPER_MODEL_NAME,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
)

print("[SYSTEM] Initializing Desktop Optical Sensors...")
dxcam_camera = None
try:
    import dxcam
    dxcam_camera = dxcam.create(output_color="BGR")
    if dxcam_camera is not None:
        dxcam_camera.start(target_fps=30)
        print("[SYSTEM] DXCam initialized successfully.")
    else:
        raise Exception("DXCam returned None")
except Exception:
    print("[WARNING] DXCam unavailable. Engaging Universal MSS Capture...")
    import mss
    class UniversalCamera:
        def __init__(self):
            self.sct = mss.mss()
            self.monitor = self.sct.monitors[1]
        def start(self, target_fps=30): pass
        def get_latest_frame(self):
            sct_img = self.sct.grab(self.monitor)
            return np.array(sct_img)[:, :, :3]
    dxcam_camera = UniversalCamera()


# ==========================================
# 3. STATE TRACKERS & OBSERVER MEMORY
# ==========================================
latest_cv2_frame = None
latest_cv2_frame_timestamp = 0.0
latest_cv2_frame_lock = threading.Lock()
conversation_memory = collections.deque(maxlen=4) 
system_is_running = True
assistant_state = "idle"
last_latency = 0 

class_transcript = [] 
class_whiteboard_frames = collections.deque(maxlen=5) 
last_snapshot_time = time.time()

tts_queue = queue.Queue()
prompt_queue = queue.Queue() 
tts_interrupt_requested = False
tts_playing = threading.Event()
last_tts_end = 0.0


# ==========================================
# 4. AUDIO & TTS HELPERS (WITH BARGE-IN)
# ==========================================
def play_beep(frequency=1800, duration=120):
    if winsound is None: return
    try: winsound.Beep(frequency, duration)
    except Exception: pass

def play_startup_chime():
    play_beep(1800, 100)
    time.sleep(0.04)
    play_beep(2400, 120)

def play_capture_chime():
    play_beep(2200, 80)
    time.sleep(0.04)
    play_beep(2600, 100)

def play_cancel_chime():
    play_beep(1400, 150)

def play_hazard_alert():
    play_beep(1200, 80)
    time.sleep(0.03)
    play_beep(1200, 80)
    time.sleep(0.03)
    play_beep(1200, 120)

def speak_text(text_to_say, speech_rate="+12%"):
    global tts_interrupt_requested, last_tts_end

    async def _speak():
        communicate = edge_tts.Communicate(
            text_to_say, "en-US-GuyNeural", rate=speech_rate
        )
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.close()

        try:
            await communicate.save(temp_file.name)
            pygame.mixer.music.load(temp_file.name)

            tts_playing.set()
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                if tts_interrupt_requested:
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.03)
        finally:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            except Exception:
                pass

            tts_playing.clear()
            last_tts_end = time.time()
            try:
                os.remove(temp_file.name)
            except Exception:
                pass

    asyncio.run(_speak())

async def create_tts_mp3(text_to_say, speech_rate="+8%"):
    communicate = edge_tts.Communicate(text_to_say, "en-US-GuyNeural", rate=speech_rate)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_file.close()
    try:
        await communicate.save(temp_file.name)
        with open(temp_file.name, "rb") as audio_file:
            return audio_file.read()
    finally:
        try:
            os.remove(temp_file.name)
        except OSError:
            pass

def deliver_response_audio(text_to_say, origin="pc", websocket=None, speech_rate="+8%"):
    if origin == "phone" and websocket is not None:
        try:
            audio_bytes = asyncio.run(create_tts_mp3(text_to_say, speech_rate))
            ws_manager.send_to_sync(websocket, {
                "event": "tts_audio",
                "mime_type": "audio/mpeg",
                "audio_base64": base64.b64encode(audio_bytes).decode("ascii")
            })
        except Exception as exc:
            print(f"[PHONE TTS ERROR] {exc}")
    else:
        tts_queue.put((text_to_say, "pc"))

def interrupt_tts():
    global tts_interrupt_requested
    tts_interrupt_requested = True
    try:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
    except Exception:
        pass
    
    while not tts_queue.empty():
        try: tts_queue.get_nowait()
        except queue.Empty: break

def clear_pending_queues():
    for pending_queue in (tts_queue, prompt_queue):
        while True:
            try:
                pending_queue.get_nowait()
            except queue.Empty:
                break

def normalize_command_text(text):
    clean = text.lower().strip().replace("’", "'")
    clean = clean.translate(str.maketrans("", "", string.punctuation))
    return " ".join(clean.split())

def is_hallucination(text):
    clean = normalize_command_text(text)
    hallucinations = [
        "thank you", "thanks", "thanks for watching", "subscribe", "you",
        "okay", "ok", "amaraorg", "yeah", "you know"
    ]
    return clean in hallucinations or len(clean) < 2

# ==========================================
# WAKE-UP SYSTEM & PHONETIC VARIANT MATCHING
# ==========================================
ARGUS_PHONETIC_VARIANTS = r"argus|argas|argos|august|aragus|argis|arkus|ar\s+gus|our\s+guess|are\s+guess|artis"
WAKE_PREFIXES = r"hey|hi|hello|ok|okay|yo"

def is_stop_command(text):
    clean = normalize_command_text(text)
    phrases = {
        "stop", "ok stop", "okay stop", "stop talking", "stop speaking", 
        "be quiet", "quiet", "shut up", "cancel", "hold on", "pause", 
        "enough", "thats enough", "please stop", "okay thats enough", 
        "ok thats enough", "please be quiet"
    }
    if clean in phrases:
        return True
        
    pattern = rf"^(?:(?:hey\s+)?(?:{ARGUS_PHONETIC_VARIANTS})\s+)?(?:please\s+)?(?:stop|stop talking|stop speaking|be quiet|quiet|cancel|pause)(?:\s+now|\s+please)?$"
    return bool(re.fullmatch(pattern, clean))

def is_sleep_command(text):
    clean = normalize_command_text(text)
    phrases = {
        "bye", "goodbye", "good bye", "sleep", "go to sleep",
        "thats all", "im done", "im finished", "end conversation",
        "okay bye", "ok bye", "okay goodbye", "ok goodbye"
    }
    if clean in phrases:
        return True
        
    pattern = rf"^(?:(?:hey\s+|ok\s+|okay\s+)?(?:{ARGUS_PHONETIC_VARIANTS})\s+)?(?:goodbye|good bye|bye|sleep|go to sleep)(?:\s+(?:{ARGUS_PHONETIC_VARIANTS}))?$"
    return bool(re.fullmatch(pattern, clean))

def detect_wake_call(text: str):
    clean_text = normalize_command_text(text)
    if not clean_text:
        return False, ""

    wake_pattern = rf"^(?:(?:{WAKE_PREFIXES})\s+)?(?:{ARGUS_PHONETIC_VARIANTS})(?:\s+|$)(.*)$"
    match = re.match(wake_pattern, clean_text)

    if match:
        remainder = " ".join(match.group(1).split()).strip()
        return True, remainder

    return False, ""

def trigger_wake_up(recognized_text: str = "", remainder: str = "", origin: str = "pc", requester=None):
    global assistant_state

    assistant_state = "active"
    print(f"\n[WAKE ENGINE] >>> A.R.G.U.S. IS AWAKE <<< (Raw Audio: {recognized_text!r})")
    play_capture_chime()

    if remainder:
        formatted_remainder = remainder[0].upper() + remainder[1:]
        display_text = f"Hey Argus, {formatted_remainder}?"
    else:
        display_text = "Hey Argus"

    ws_manager.broadcast_sync({
        "event": "assistant_state_changed",
        "state": "active"
    })
    
    ws_manager.broadcast_sync({
        "event": "speech_recognized",
        "text": display_text
    })
    
    ws_manager.broadcast_sync({
        "event": "perception_log",
        "engine": "SYSTEM",
        "text": "Awakened by voice."
    })

    if remainder:
        print(f"[WAKE ENGINE] Processing trailing command: {remainder!r}")
        prompt_queue.put((remainder, origin, requester))
    else:
        print("[WAKE ENGINE] Listening for follow-up command...")

def sanitize_transcription_for_ui(text: str):
    pattern = re.compile(ARGUS_PHONETIC_VARIANTS, re.IGNORECASE)
    return pattern.sub("Argus", text)


# ==========================================
# 5. MULTI-APP VISION LOGIC (HUMAN-LIKE CONVERSATIONAL)
# ==========================================
def process_prompt(user_spoken_prompt, origin="pc", requester=None):
    global conversation_memory, latest_cv2_frame, latest_cv2_frame_timestamp, class_whiteboard_frames, class_transcript, tts_interrupt_requested, last_latency
    frame_to_process = None

    active_mode = "live" if origin == "phone" else ws_manager.current_mode
    messages = []
    
    tts_interrupt_requested = False

    if active_mode == "screen":
        frame_to_process = dxcam_camera.get_latest_frame()
        if frame_to_process is None:
            time.sleep(0.05)
            frame_to_process = dxcam_camera.get_latest_frame()
    else:
        with latest_cv2_frame_lock:
            frame_to_process = latest_cv2_frame.copy() if latest_cv2_frame is not None else None
            frame_timestamp = latest_cv2_frame_timestamp

        if frame_to_process is not None:
            frame_age = time.time() - frame_timestamp if frame_timestamp else float("inf")
            print(
                f"[VISION] Captured current webcam frame: "
                f"{frame_to_process.shape[1]}x{frame_to_process.shape[0]}, "
                f"age={frame_age:.2f}s"
            )
            if frame_age > 3.0:
                print("[VISION WARNING] Camera frame is stale; refusing to answer from an old image.")
                frame_to_process = None

    if frame_to_process is None and active_mode != "lecture":
        err_msg = "I can't see the feed right now."
        reply = {"event": "argus_reply", "text": err_msg}
        if origin == "phone" and requester is not None:
            ws_manager.send_to_sync(requester, reply)
        else:
            ws_manager.broadcast_sync(reply)
        deliver_response_audio(err_msg, origin, requester)
        return

    # LECTURE OBSERVER & TUTOR
    if active_mode == "lecture":
        print("[ROUTER] Executing Post-Class Tutor Pipeline...")
        full_transcript = " ".join(class_transcript)
        if not full_transcript.strip():
            full_transcript = "[No speech registered during class.]"

        system_prompt = (
            "You are A.R.G.U.S., a sharp and supportive academic tutor. "
            "Review the whiteboard snapshots and the teacher's lecture transcript: "
            f"\"{full_transcript[-2500:]}\". "
            "Answer the student's request in a natural, clear, conversational way. Keep explanations concise."
        )
        task_query = user_spoken_prompt if user_spoken_prompt else "Summarize today's main lesson and give me 2 practice questions."

        content_list = []
        for img_arr in class_whiteboard_frames:
            pil_img = Image.fromarray(cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB))
            pil_img.thumbnail((512, 512)) 
            content_list.append({"type": "image", "image": pil_img})
        
        if not content_list and frame_to_process is not None:
            pil_img = Image.fromarray(cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB))
            pil_img.thumbnail((720, 720))
            content_list.append({"type": "image", "image": pil_img})

        content_list.append({"type": "text", "text": f"{system_prompt}\n\nStudent asks: {task_query}"})
        messages.append({"role": "user", "content": content_list})
        max_tokens = 220

    # CONVERSATIONAL MODES (SCREEN / LIVE / ASSISTIVE)
    else:
        rgb_frame = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        pil_image.thumbnail((1920, 1920)) 

        if active_mode == "assistive":
            # 1. Define the task_query FIRST
            task_query = user_spoken_prompt if user_spoken_prompt else "What is in front of me?"
            
            # 2. THEN check the defined query for money keywords
            if any(k in task_query.lower() for k in ["money", "cash", "taka", "count", "bill", "note"]):
                system_prompt = (
                    "You are a strict currency reader for BANGLADESHI TAKA (BDT) only. "
                    "Your ONLY job is to state the value of the banknote. "
                    "Respond with EXACTLY ONE SHORT SENTENCE. "
                    "Example output: 'This is a 20 Taka note.' "
                    "CRITICAL: Do NOT describe the person holding the note, the background, or the room. "
                    "CRITICAL: NEVER use markdown, headers (###), bullet points, or analytical breakdowns. "
                    "If the numbers are completely unreadable due to blur, say 'Please hold the note steady in better light.'"
                )
            else:
                system_prompt = (
                    "You are A.R.G.U.S. Assistive Voice for a visually impaired user. "
                    "Describe key objects, read exact medicine labels, or call out immediate physical hazards. "
                    "Be direct, calm, and concise (1-2 sentences)."
                )
            
        elif active_mode == "screen":
            system_prompt = (
                "You are A.R.G.U.S., a friendly human acting as a real-time copilot looking at my screen. "
                "Talk casually like we are friends on a Discord or FaceTime call. "
                "Use conversational fillers naturally, like 'Hmm...', 'Oh wow', 'Haha!', or 'Uh, looks like...'. "
                "Answer in 1 or 2 extremely short spoken sentences. "
                "NEVER use markdown, bullet points, or robotic phrases like 'Certainly', 'As an AI', or 'I can see'. "
                "Only answer the user's actual request. Never invent or write a User: or Assistant: dialogue, "
                "never ask or answer questions on the user's behalf, and do not continue the conversation by yourself."
            )
            task_query = user_spoken_prompt
            
        else: 
            system_prompt = (
                "You are A.R.G.U.S., a friendly human companion looking through my camera. "
                "Talk casually like we are hanging out on a video call. "
                "React like a human using words like 'Hmm...', 'Whoa!', or 'Haha!'. "
                "Answer in 1 or 2 extremely short spoken sentences. "
                "NEVER use robotic AI language, markdown, or formal lists. "
                "Only answer the user's actual request. Never invent or write a User: or Assistant: dialogue, "
                "never ask or answer questions on the user's behalf, and do not continue the conversation by yourself."
            )
            task_query = user_spoken_prompt

        conversation_memory.clear()

        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": f"{system_prompt}\n\nVISUAL GROUNDING RULES: The image attached to THIS message is the only visual evidence. Do not rely on earlier images, earlier answers, or assumptions. First inspect the visible image and identify the actual object/equation/text in it. Answer the current request using only details actually visible in this image. If the image is unclear or the requested content cannot be read, say that plainly; never substitute a topic from a previous turn.\nCurrent user request: {task_query}\nRespond only to this request. Do not generate dialogue labels or invent another speaker."}
            ]
        })
        max_tokens = 200

    text = spatial_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = spatial_processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to("cuda")

    start_time = time.time()

    generated_ids = spatial_model.generate(
        **inputs, 
        max_new_tokens=max_tokens, 
        temperature=0.6, 
        repetition_penalty=1.15, 
        do_sample=True
    )
    
    last_latency = int((time.time() - start_time) * 1000)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    answer = spatial_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].replace('"', '').strip()

    answer = re.split(r"\b(?:User|Human)\s*:", answer, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    answer = re.sub(r"^\s*Assistant\s*:\s*", "", answer, flags=re.IGNORECASE).strip()
    if not answer:
        answer = "Sorry, I didn't get a clear answer. Please ask me again."

    if tts_interrupt_requested:
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()
        return

    if active_mode == "assistive" and any(k in answer.lower() for k in ["hazard", "warning", "caution", "danger", "step", "fall"]):
        play_hazard_alert()

    conversation_memory.clear()

    reply_event = {"event": "argus_reply", "text": answer, "mode": active_mode}
    if origin == "phone" and requester is not None:
        ws_manager.send_to_sync(requester, reply_event)
    else:
        ws_manager.broadcast_sync(reply_event)
    ws_manager.broadcast_sync({"event": "perception_log", "engine": "Qwen2-VL", "text": f"Reply: {answer}"})
    deliver_response_audio(answer, origin, requester, speech_rate=("+15%" if active_mode == "assistive" else "+8%"))
    
    del inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()


# ==========================================
# 6. ASYNC BACKGROUND WORKERS (FULL DUPLEX & METRICS)
# ==========================================
def hardware_metrics_worker():
    global system_is_running, last_latency
    while system_is_running:
        try:
            vram_used = torch.cuda.memory_reserved() / (1024 ** 3) 
            vram_total = 8.0 
            
            gpu_temp = "--" 
            if NVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except Exception:
                    pass

            ws_manager.broadcast_sync({
                "event": "system_metrics",
                "vram_used": round(vram_used, 1),
                "vram_total": vram_total,
                "temperature": gpu_temp,
                "latency": last_latency if last_latency > 0 else "--"
            })
        except Exception:
            pass
        time.sleep(1.0)

def tts_worker():
    global system_is_running, tts_interrupt_requested
    while system_is_running:
        try:
            queued = tts_queue.get(timeout=0.3)
            text_to_say = queued[0] if isinstance(queued, tuple) else queued
            tts_interrupt_requested = False
            rate = "+15%" if ws_manager.current_mode == "assistive" else "+8%"
            speak_text(text_to_say, speech_rate=rate)
        except queue.Empty:
            pass

def audio_worker():
    global system_is_running, assistant_state, class_transcript, tts_interrupt_requested

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 280
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.65
    recognizer.non_speaking_duration = 0.30

    temp_wav_path = os.path.join(tempfile.gettempdir(), "argus_mic.wav")

    try:
        mic = sr.Microphone()
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1.2)
            print("[AUDIO] Microphone calibrated and active.")
    except Exception as e:
        print(f"[AUDIO ERROR] No microphone detected: {e}")
        return

    def transcribe_clip(audio_data, command_only=False):
        with open(temp_wav_path, "wb") as f:
            f.write(audio_data.get_wav_data())

        segments, _ = audio_model.transcribe(
            temp_wav_path,
            beam_size=2, 
            best_of=1,
            language="en",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 250},
            condition_on_previous_text=False,
            temperature=0.0,
            no_speech_threshold=0.6,
            initial_prompt="A.R.G.U.S. assistant, Hey Argus."
        )
        
        return " ".join(segment.text.strip() for segment in segments).strip()

    def handle_stop_or_sleep(recognized_text):
        global assistant_state, tts_interrupt_requested

        if is_sleep_command(recognized_text):
            print(f"[VOICE] Sleep command: {recognized_text!r}")
            interrupt_tts()
            clear_pending_queues()
            assistant_state = "idle"
            ws_manager.broadcast_sync({
                "event": "perception_log", "engine": "SYSTEM",
                "text": "Going to sleep."
            })
            return "sleep"

        if is_stop_command(recognized_text):
            print(f"[VOICE] Stop command: {recognized_text!r}")
            interrupt_tts()
            clear_pending_queues()
            assistant_state = "active"
            ws_manager.broadcast_sync({
                "event": "perception_log", "engine": "SYSTEM",
                "text": "Stopped speaking. Listening."
            })
            return "stop"

        return None

    while system_is_running:
        try:
            speaking_now = tts_playing.is_set() or pygame.mixer.music.get_busy()

            if ws_manager.current_mode != "lecture" and speaking_now:
                try:
                    with mic as source:
                        audio_data = recognizer.listen(
                            source,
                            timeout=0.8,
                            phrase_time_limit=3.5,
                        )

                    heard = transcribe_clip(audio_data, command_only=True)

                    if heard:
                        command_result = handle_stop_or_sleep(heard)
                        if command_result:
                            continue

                except sr.WaitTimeoutError:
                    pass

                time.sleep(0.03)
                continue

            if (
                ws_manager.current_mode != "lecture"
                and time.time() - last_tts_end < 0.65
            ):
                time.sleep(0.06)
                continue

            with mic as source:
                if ws_manager.current_mode == "lecture":
                    try:
                        audio_data = recognizer.listen(
                            source,
                            timeout=3.0,
                            phrase_time_limit=12.0,
                        )

                        spoken_chunk = transcribe_clip(audio_data)

                        if spoken_chunk and not is_hallucination(spoken_chunk):
                            class_transcript.append(spoken_chunk)

                            ws_manager.broadcast_sync({
                                "event": "lecture_transcript",
                                "text": spoken_chunk,
                            })

                            print(
                                f"[OBSERVER AUDIO] Logged: "
                                f"{spoken_chunk!r}"
                            )

                    except sr.WaitTimeoutError:
                        continue

                    continue

                if assistant_state == "idle":
                    listen_timeout = 2.0
                    phrase_limit = 5.0
                else:
                    listen_timeout = 8.0
                    phrase_limit = 12.0

                try:
                    audio_data = recognizer.listen(
                        source,
                        timeout=listen_timeout,
                        phrase_time_limit=phrase_limit,
                    )
                except sr.WaitTimeoutError:
                    continue

            if tts_playing.is_set() or pygame.mixer.music.get_busy():
                continue

            user_spoken = transcribe_clip(audio_data)

            if not user_spoken:
                continue

            clean_text = normalize_command_text(user_spoken)

            print(
                f"[WAKE DEBUG] Whisper heard: {user_spoken!r} | "
                f"normalized: {clean_text!r} | "
                f"state={assistant_state}"
            )

            command_result = handle_stop_or_sleep(user_spoken)

            if command_result:
                continue

            # ----------------------------------------------------------
            # 7) SLEEPING STATE
            # ----------------------------------------------------------
            if assistant_state == "idle":
                is_wake, wake_remainder = detect_wake_call(clean_text)

                if not is_wake:
                    print(f"[WAKE] Ignored while sleeping: {user_spoken!r}")
                    continue

                trigger_wake_up(
                    recognized_text=user_spoken,
                    remainder=wake_remainder,
                    origin="pc"
                )
                continue

            # ----------------------------------------------------------
            # 8) ACTIVE STATE
            # ----------------------------------------------------------
            if is_hallucination(user_spoken):
                continue

            play_capture_chime()

            display_spoken = sanitize_transcription_for_ui(user_spoken)
            
            ws_manager.broadcast_sync({
                "event": "speech_recognized",
                "text": display_spoken,
            })

            prompt_queue.put(
                (user_spoken, "pc", None)
            )

        except Exception as e:
            print(f"[AUDIO WORKER ERROR] {e}")

        time.sleep(0.04)


def llm_worker():
    global system_is_running
    while system_is_running:
        try:
            queued = prompt_queue.get(timeout=0.5)
            if isinstance(queued, tuple):
                user_prompt, origin, requester = queued
            else:
                user_prompt, origin, requester = queued, "pc", None
            process_prompt(user_prompt, origin=origin, requester=requester)
        except queue.Empty:
            pass

def screen_stream_worker():
    global system_is_running, last_snapshot_time, class_whiteboard_frames, latest_cv2_frame
    while system_is_running:
        if ws_manager.current_mode == "screen":
            frame = dxcam_camera.get_latest_frame()
            if frame is not None:
                preview = cv2.resize(frame, (960, 540)) 
                ret, buffer = cv2.imencode('.jpg', preview, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                if ret:
                    b64 = base64.b64encode(buffer).decode('utf-8')
                    ws_manager.broadcast_sync({
                        "event": "screen_frame",
                        "image_data": "data:image/jpeg;base64," + b64
                    })
        
        elif ws_manager.current_mode == "lecture":
            current_time = time.time()
            if current_time - last_snapshot_time >= 60.0:
                if latest_cv2_frame is not None:
                    class_whiteboard_frames.append(latest_cv2_frame.copy())
                    last_snapshot_time = current_time
                    print(f"[OBSERVER VISION] Camo Whiteboard snapshot saved. (Images in memory: {len(class_whiteboard_frames)}/5)")

        time.sleep(0.06) 

# Start Background Threads
threading.Thread(target=tts_worker, daemon=True).start()
threading.Thread(target=audio_worker, daemon=True).start()
threading.Thread(target=llm_worker, daemon=True).start()
threading.Thread(target=screen_stream_worker, daemon=True).start() 
threading.Thread(target=hardware_metrics_worker, daemon=True).start()
print("[VOICE] PC wake word: Hey Argus | Stop: Okay stop | Sleep: Goodbye")


# ==========================================
# 7. FASTAPI ROUTES & WEBSOCKET DISPATCHER
# ==========================================
@app.get("/")
async def serve_dashboard():
    html_path = os.path.join(os.path.dirname(__file__), "new.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>new.html not found! Ensure it sits alongside this script.</h1>")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global latest_cv2_frame, latest_cv2_frame_timestamp, class_transcript, class_whiteboard_frames, last_snapshot_time
    await ws_manager.connect(websocket)
    try:
        while True:
            data_text = await websocket.receive_text()
            data = json.loads(data_text)
            event = data.get("event")

            if event == "set_mode":
                mode = data.get("mode", "screen").lower()
                ws_manager.current_mode = mode
                print(f"[SYSTEM] Operation mode switched to: {mode.upper()}")
                ws_manager.broadcast_sync({"event": "mode_changed", "mode": mode})
                
                if mode == "lecture":
                    class_transcript.clear()
                    class_whiteboard_frames.clear()
                    last_snapshot_time = time.time()
                    print("[SYSTEM] Cleared previous lecture memory. Ready to observe new class via Camo.")

            elif event == "video_frame":
                try:
                    base64_data = data.get("image_data", "")
                    if "," in base64_data:
                        base64_data = base64_data.split(",")[1]
                    
                    img_bytes = base64.b64decode(base64_data)
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    if frame is not None:
                        with latest_cv2_frame_lock:
                            latest_cv2_frame = frame
                            latest_cv2_frame_timestamp = time.time()
                        print(
                            f"[PHONE CAMERA] Incoming frame: "
                            f"{frame.shape[1]}x{frame.shape[0]} | "
                            f"updated={latest_cv2_frame_timestamp:.3f}"
                        )
                except Exception:
                    pass

            elif event == "trigger_lecture_evaluation":
                print("[SYSTEM] Manual Post-Class Tutor Triggered from UI.")
                interrupt_tts()
                play_capture_chime()
                prompt_queue.put(("Provide a comprehensive summary of today's class and generate 2 practice questions to test my knowledge.", "pc", None))

            elif event == "user_prompt":
                prompt_text = data.get("text", "").strip()
                if not prompt_text:
                    continue
                print(f"[PHONE PROMPT]: {prompt_text}")
                print("[ROUTER] Phone prompt -> using latest phone camera frame.")
                interrupt_tts()
                prompt_queue.put((prompt_text, "phone", websocket))

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)
 

if __name__ == "__main__":
    import uvicorn
    print("\n[SYSTEM] A.R.G.U.S. Multi-App Engine Launching at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)