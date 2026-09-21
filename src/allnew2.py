import asyncio
import base64
import collections
import json
import os
import queue
import tempfile
import threading
import time
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

# Initialize audio mixer
pygame.mixer.init()

# ==========================================
# MOONDREAM2 / TRANSFORMERS COMPATIBILITY PATCH
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
        # Available modes: "screen", "live", "lecture", "assistive"
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
# 2. LOAD AI MODELS & SENSORS (MULTI-ADAPTER)
# ==========================================
print("\n[SYSTEM] Booting A.R.G.U.S. Multimodal Engine...")
device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERAL_ADAPTER_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "qwen2-vl-argus-200gb-checkpoints", "final_adapter"))
MATH_ADAPTER_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "qwen2-vl-argus-math-adapter", "final_adapter"))

print("[SYSTEM] Loading Base Qwen2-VL 2B Backbone (Frozen)...")
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", 
    torch_dtype=torch.bfloat16, 
    device_map="cuda"
)

print("[SYSTEM] Injecting Adapter 1: General Intelligence...")
spatial_model = PeftModel.from_pretrained(
    base_model, 
    GENERAL_ADAPTER_PATH, 
    adapter_name="argus_general"
)

print("[SYSTEM] Injecting Adapter 2: Mathematical / Lecture Logic...")
spatial_model.load_adapter(
    MATH_ADAPTER_PATH, 
    adapter_name="argus_math"
)
spatial_model.eval()

try:
    spatial_processor = AutoProcessor.from_pretrained(GENERAL_ADAPTER_PATH)
    print("[SYSTEM] Custom A.R.G.U.S. processor loaded successfully.")
except Exception:
    print("[WARNING] Loading default base processor fallback...")
    spatial_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

print("[SYSTEM] Loading Audio Brain (Whisper)...")
audio_model = WhisperModel("tiny.en", device=device, compute_type="float16" if device == "cuda" else "int8")

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
# 3. STATE TRACKERS & ROLLING BUFFERS
# ==========================================
latest_cv2_frame = None 
conversation_memory = collections.deque(maxlen=3) 
system_is_running = True
assistant_state = "idle"

lecture_audio_buffer = collections.deque(maxlen=8)
tts_queue = queue.Queue()
prompt_queue = queue.Queue() 
tts_done_event = threading.Event()
tts_interrupt_requested = False

# ==========================================
# 4. AUDIO & TTS HELPERS
# ==========================================
def play_beep(frequency=1800, duration=120):
    if winsound is None: return
    try: winsound.Beep(frequency, duration)
    except Exception: pass

def play_startup_chime():
    play_beep(1800, 140)
    time.sleep(0.08)
    play_beep(2200, 110)

def play_capture_chime():
    play_beep(2200, 110)
    time.sleep(0.05)
    play_beep(2600, 130)

def play_hazard_alert():
    play_beep(3000, 100)
    time.sleep(0.05)
    play_beep(3000, 200)

def speak_text(text_to_say, speech_rate="+10%"):
    global tts_interrupt_requested
    async def _speak():
        communicate = edge_tts.Communicate(text_to_say, "en-US-ChristopherNeural", rate=speech_rate)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.close()
        await communicate.save(temp_file.name)
        
        pygame.mixer.music.load(temp_file.name)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            if tts_interrupt_requested:
                pygame.mixer.music.stop()
                break
            time.sleep(0.05)
            
        pygame.mixer.music.unload()
        try: os.remove(temp_file.name)
        except Exception: pass

    asyncio.run(_speak())

def interrupt_tts():
    global tts_interrupt_requested
    tts_interrupt_requested = True
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()


# ==========================================
# 5. MULTI-APP VISION LOGIC (CORE ROUTER)
# ==========================================
def process_prompt(user_spoken_prompt):
    global conversation_memory, latest_cv2_frame
    frame_to_process = None
    active_mode = ws_manager.current_mode

    # --- CHANGED: Now "lecture" mode pulls from DXCam Screen Capture ---
    if active_mode in ["screen", "lecture"]:
        frame_to_process = dxcam_camera.get_latest_frame()
        if frame_to_process is None:
            time.sleep(0.05)
            frame_to_process = dxcam_camera.get_latest_frame()
    else:
        # "live" and "assistive" leverage the optical webcam feed
        frame_to_process = latest_cv2_frame

    if frame_to_process is None:
        err_msg = "Optical feed unavailable for processing."
        ws_manager.broadcast_sync({"event": "argus_reply", "text": err_msg})
        tts_queue.put(err_msg)
        return

    rgb_frame = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    pil_image.thumbnail((960, 960)) 

    # Dynamic Task Prompting based on Active Module
    if active_mode == "lecture":
        print("[ROUTER] Executing Smart Lecture Evaluator Pipeline (Screen Source)...")
        recent_lecture = " ".join(lecture_audio_buffer)
        if not recent_lecture.strip():
            recent_lecture = "[No speech registered in buffer.]"

        system_prompt = (
            "You are A.R.G.U.S. Smart Lecture Evaluator. "
            "Examine the whiteboard equations, diagrams, and handwriting in this image. "
            f"Cross-reference them against the teacher's spoken lecture transcript: \"{recent_lecture}\". "
            "Identify any mathematical inconsistencies, missing variables, or conceptual discrepancies. "
            "Respond with a concise, structured evaluation report."
        )
        task_query = user_spoken_prompt if user_spoken_prompt else "Audit the current screen content against lecture speech."

    elif active_mode == "assistive":
        print("[ROUTER] Executing Assistive Vision Pipeline (Blind Support)...")
        system_prompt = (
            "You are A.R.G.U.S. Assistive Eye, an AI assistant for a visually impaired user. "
            "Prioritize physical safety and high-accuracy text extraction. "
            "If analyzing a medicine container, read the exact medicine name, dosage, and warnings. "
            "If scanning the surrounding room, warn immediately about physical hazards, drops, or approaching obstacles. "
            "Answer directly and concisely without conversational fluff."
        )
        task_query = user_spoken_prompt if user_spoken_prompt else "Describe the object held or warn of hazards in view."

    elif active_mode == "screen":
        system_prompt = (
            "You are A.R.G.U.S. Desktop Copilot. "
            "Analyze the user's screen carefully and answer their question clearly in one or two short sentences."
        )
        task_query = user_spoken_prompt

    else: 
        system_prompt = (
            "You are A.R.G.U.S., a friendly multimodal companion. "
            "Look at the scene and answer the user's query naturally in one or two sentences."
        )
        task_query = user_spoken_prompt

    messages = []
    if active_mode not in ["lecture", "assistive"]:
        for past_q, past_a in conversation_memory:
            messages.append({"role": "user", "content": [{"type": "text", "text": past_q}]})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": past_a}]})
    
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": f"{system_prompt}\n\nQuery: {task_query}"}
        ]
    })

    text = spatial_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = spatial_processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to("cuda")

    if active_mode == "lecture":
        print("[SYSTEM] 🧠 Switching Brain to MATH Mode...")
        spatial_model.set_adapter("argus_math")
    else:
        print("[SYSTEM] 🧠 Switching Brain to GENERAL Mode...")
        spatial_model.set_adapter("argus_general")
    
    max_tokens = 150 if active_mode == "lecture" else 70
    generated_ids = spatial_model.generate(
        **inputs, max_new_tokens=max_tokens, temperature=0.2, repetition_penalty=1.1, do_sample=True
    )
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    answer = spatial_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].replace('"', '').strip()

    if active_mode == "assistive" and any(k in answer.lower() for k in ["hazard", "warning", "caution", "danger", "step"]):
        play_hazard_alert()

    if active_mode not in ["lecture", "assistive"]:
        conversation_memory.append((user_spoken_prompt, answer))

    ws_manager.broadcast_sync({"event": "argus_reply", "text": answer, "mode": active_mode})
    ws_manager.broadcast_sync({"event": "perception_log", "engine": "Qwen2-VL", "text": f"Processed ({active_mode.upper()}): {task_query}"})
    
    tts_queue.put(answer)
    
    del inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()


# ==========================================
# 6. ASYNC BACKGROUND WORKERS
# ==========================================
def tts_worker():
    global system_is_running, tts_done_event, tts_interrupt_requested
    while system_is_running:
        try:
            text_to_say = tts_queue.get(timeout=0.5)
            tts_interrupt_requested = False
            tts_done_event.clear()
            rate = "+15%" if ws_manager.current_mode == "assistive" else "+10%"
            speak_text(text_to_say, speech_rate=rate)
            tts_done_event.set()
        except queue.Empty:
            pass

def audio_worker():
    global system_is_running, assistant_state
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 3000  
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8

    temp_wav_path = os.path.join(tempfile.gettempdir(), "argus_mic.wav")

    try:
        mic = sr.Microphone()
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1.2)
            print("[AUDIO] Microphone calibrated and online.")
    except Exception as e:
        print(f"[AUDIO ERROR] No microphone detected: {e}")
        return

    wake_variants = ["argus", "aragus", "arghas", "argas", "arghus", "argos", "hey argus"]

    while system_is_running:
        try:
            with mic as source:
                if ws_manager.current_mode == "lecture":
                    try:
                        audio_data = recognizer.listen(source, timeout=3.0, phrase_time_limit=6.0)
                        with open(temp_wav_path, "wb") as f:
                            f.write(audio_data.get_wav_data())
                        
                        segments, _ = audio_model.transcribe(temp_wav_path, beam_size=3, language="en")
                        spoken_chunk = "".join([s.text for s in segments]).strip()
                        
                        if len(spoken_chunk) > 2:
                            lecture_audio_buffer.append(spoken_chunk)
                            ws_manager.broadcast_sync({"event": "lecture_transcript", "text": spoken_chunk})

                            if any(trigger in spoken_chunk.lower() for trigger in ["evaluate", "check board", "argus"]):
                                play_capture_chime()
                                prompt_queue.put("Evaluate the whiteboard against current lecture notes.")
                    except sr.WaitTimeoutError:
                        continue

                elif assistant_state == "active":
                    try:
                        command_audio = recognizer.listen(source, timeout=6.0, phrase_time_limit=10.0)
                    except sr.WaitTimeoutError:
                        assistant_state = "idle"
                        ws_manager.broadcast_sync({"event": "perception_log", "engine": "SYSTEM", "text": "Auditory listener entered idle state."})
                        continue

                    with open(temp_wav_path, "wb") as f:
                        f.write(command_audio.get_wav_data())

                    segments, _ = audio_model.transcribe(temp_wav_path, beam_size=5, language="en")
                    user_spoken_prompt = "".join([s.text for s in segments]).strip().lower()

                    if len(user_spoken_prompt) > 2:
                        play_capture_chime()
                        ws_manager.broadcast_sync({"event": "speech_recognized", "text": user_spoken_prompt})
                        prompt_queue.put(user_spoken_prompt)

                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)

                elif assistant_state == "idle":
                    try:
                        audio_data = recognizer.listen(source, timeout=2.0, phrase_time_limit=4.0)
                    except sr.WaitTimeoutError:
                        continue

                    with open(temp_wav_path, "wb") as f:
                        f.write(audio_data.get_wav_data())

                    segments, _ = audio_model.transcribe(temp_wav_path, beam_size=3, language="en")
                    transcription = "".join([s.text for s in segments]).strip().lower()

                    if any(variant in transcription for variant in wake_variants):
                        interrupt_tts()
                        play_startup_chime()
                        ws_manager.broadcast_sync({"event": "speech_recognized", "text": transcription})
                        ws_manager.broadcast_sync({"event": "perception_log", "engine": "WHISPER", "text": "Wake Word Detected!"})
                        
                        greeting = "I'm with you. What do you need?" if ws_manager.current_mode != "assistive" else "A.R.G.U.S. ready. Point the camera."
                        tts_queue.put(greeting)
                        assistant_state = "active"

        except Exception:
            pass
        time.sleep(0.05)

def llm_worker():
    global system_is_running
    while system_is_running:
        try:
            user_prompt = prompt_queue.get(timeout=0.5)
            process_prompt(user_prompt)
        except queue.Empty:
            pass

def screen_stream_worker():
    global system_is_running
    while system_is_running:
        # --- CHANGED: Broadcast DXCam stream to UI when in Lecture Mode too ---
        if ws_manager.current_mode in ["screen", "lecture"]:
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
        time.sleep(0.06) 

# Start Threads
threading.Thread(target=tts_worker, daemon=True).start()
threading.Thread(target=audio_worker, daemon=True).start()
threading.Thread(target=llm_worker, daemon=True).start()
threading.Thread(target=screen_stream_worker, daemon=True).start() 


# ==========================================
# 7. FASTAPI ROUTES & WEBSOCKET DISPATCHER
# ==========================================
@app.get("/")
async def serve_dashboard():
    html_path = os.path.join(os.path.dirname(__file__), "new1.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>new1.html not found! Ensure it sits alongside this script.</h1>")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global latest_cv2_frame
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

            elif event == "video_frame":
                try:
                    base64_data = data.get("image_data", "")
                    if "," in base64_data:
                        base64_data = base64_data.split(",")[1]
                    
                    img_bytes = base64.b64decode(base64_data)
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    if frame is not None:
                        latest_cv2_frame = frame
                except Exception:
                    pass

            elif event == "trigger_lecture_evaluation":
                print("[SYSTEM] Manual Lecture Evaluation Triggered from UI.")
                play_capture_chime()
                prompt_queue.put("Cross-reference the whiteboard equations against spoken lecture speech and highlight errors.")

            elif event == "user_prompt":
                prompt_text = data.get("text", "")
                print(f"[WEBSITE PROMPT]: {prompt_text}")
                prompt_queue.put(prompt_text)

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    print("\n[SYSTEM] A.R.G.U.S. Multi-App Engine Launching at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)