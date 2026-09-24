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

# --- RE-ADDED: NVIDIA Hardware Monitoring ---
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

print("[SYSTEM] Loading Audio Brain (Whisper base.en)...")
audio_model = WhisperModel("base.en", device=device, compute_type="float16" if device == "cuda" else "int8")

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
conversation_memory = collections.deque(maxlen=4) 
system_is_running = True
assistant_state = "idle"
last_latency = 0  # <--- RE-ADDED: Global tracker for UI latency gauge

class_transcript = [] 
class_whiteboard_frames = collections.deque(maxlen=5) 
last_snapshot_time = time.time()

tts_queue = queue.Queue()
prompt_queue = queue.Queue() 
tts_interrupt_requested = False


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

def speak_text(text_to_say, speech_rate="+12%"):
    global tts_interrupt_requested
    async def _speak():
        communicate = edge_tts.Communicate(text_to_say, "en-US-GuyNeural", rate=speech_rate)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.close()
        await communicate.save(temp_file.name)
        
        pygame.mixer.music.load(temp_file.name)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            if tts_interrupt_requested:
                pygame.mixer.music.stop()
                break
            time.sleep(0.03)
            
        pygame.mixer.music.unload()
        try: os.remove(temp_file.name)
        except Exception: pass

    asyncio.run(_speak())

def interrupt_tts():
    """Immediately halts audio playback and drains pending speech."""
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

def is_hallucination(text):
    clean = text.translate(str.maketrans('', '', string.punctuation)).lower().strip()
    hallucinations = ["thank you", "thanks", "thanks for watching", "subscribe", "you", "bye", "okay", "ok", "amaraorg", "yeah"]
    return clean in hallucinations or len(clean) < 2

def is_stop_command(text):
    clean = text.translate(str.maketrans('', '', string.punctuation)).lower().strip()
    stop_words = ["stop", "shut up", "be quiet", "cancel", "never mind", "no need", "enough", "stop talking", "dont answer", "hold on", "pause", "thanks stop"]
    return any(w in clean for w in stop_words)


# ==========================================
# 5. MULTI-APP VISION LOGIC (HUMAN-LIKE CONVERSATIONAL)
# ==========================================
def process_prompt(user_spoken_prompt):
    global conversation_memory, latest_cv2_frame, class_whiteboard_frames, class_transcript, tts_interrupt_requested, last_latency
    frame_to_process = None
    active_mode = ws_manager.current_mode
    messages = []
    
    tts_interrupt_requested = False

    if active_mode == "screen":
        frame_to_process = dxcam_camera.get_latest_frame()
        if frame_to_process is None:
            time.sleep(0.05)
            frame_to_process = dxcam_camera.get_latest_frame()
    else:
        frame_to_process = latest_cv2_frame

    if frame_to_process is None and active_mode != "lecture":
        err_msg = "I can't see the feed right now."
        ws_manager.broadcast_sync({"event": "argus_reply", "text": err_msg})
        tts_queue.put(err_msg)
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
        pil_image.thumbnail((960, 960)) 

        if active_mode == "assistive":
            system_prompt = (
                "You are A.R.G.U.S. Assistive Voice for a visually impaired user. "
                "Describe key objects, read exact medicine labels, or call out immediate physical hazards. "
                "Be direct, calm, and concise (1-2 sentences)."
            )
            task_query = user_spoken_prompt if user_spoken_prompt else "What is in front of me?"
            
        elif active_mode == "screen":
            system_prompt = (
                "You are A.R.G.U.S., a friendly human acting as a real-time copilot looking at my screen. "
                "Talk casually like we are friends on a Discord or FaceTime call. "
                "Use conversational fillers naturally, like 'Hmm...', 'Oh wow', 'Haha!', or 'Uh, looks like...'. "
                "Answer in 1 or 2 extremely short spoken sentences. "
                "NEVER use markdown, bullet points, or robotic phrases like 'Certainly', 'As an AI', or 'I can see'."
            )
            task_query = user_spoken_prompt
            
        else: 
            system_prompt = (
                "You are A.R.G.U.S., a friendly human companion looking through my camera. "
                "Talk casually like we are hanging out on a video call. "
                "React like a human using words like 'Hmm...', 'Whoa!', or 'Haha!'. "
                "Answer in 1 or 2 extremely short spoken sentences. "
                "NEVER use robotic AI language, markdown, or formal lists."
            )
            task_query = user_spoken_prompt

        for past_q, past_a in conversation_memory:
            messages.append({"role": "user", "content": [{"type": "text", "text": past_q}]})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": past_a}]})
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": f"{system_prompt}\n\nUser: {task_query}"}
            ]
        })
        max_tokens = 60 

    # Execute Inference
    text = spatial_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = spatial_processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to("cuda")

    # --- RE-ADDED: Start Tracking Inference Time ---
    start_time = time.time()

    generated_ids = spatial_model.generate(
        **inputs, 
        max_new_tokens=max_tokens, 
        temperature=0.6, 
        repetition_penalty=1.15, 
        do_sample=True
    )
    
    # --- RE-ADDED: End Tracking Inference Time ---
    last_latency = int((time.time() - start_time) * 1000)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    answer = spatial_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].replace('"', '').strip()

    if tts_interrupt_requested:
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()
        return

    if active_mode == "assistive" and any(k in answer.lower() for k in ["hazard", "warning", "caution", "danger", "step", "fall"]):
        play_hazard_alert()

    if active_mode not in ["lecture", "assistive"]:
        conversation_memory.append((user_spoken_prompt, answer))

    ws_manager.broadcast_sync({"event": "argus_reply", "text": answer, "mode": active_mode})
    ws_manager.broadcast_sync({"event": "perception_log", "engine": "Qwen2-VL", "text": f"Reply: {answer}"})
    
    tts_queue.put(answer)
    
    del inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()


# ==========================================
# 6. ASYNC BACKGROUND WORKERS (FULL DUPLEX & METRICS)
# ==========================================
# --- RE-ADDED: Hardware Polling Worker ---
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
            text_to_say = tts_queue.get(timeout=0.3)
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

    temp_wav_path = os.path.join(tempfile.gettempdir(), "argus_mic.wav")

    try:
        mic = sr.Microphone() 
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1.2)
            print("[AUDIO] Headphone Microphone calibrated and active (Full Duplex).")
    except Exception as e:
        print(f"[AUDIO ERROR] No microphone detected: {e}")
        return

    wake_variants = ["argus", "aragus", "arghas", "argas", "arghus", "argos", "hey argus"]

    while system_is_running:
        try:
            with mic as source:
                if ws_manager.current_mode == "lecture":
                    try:
                        audio_data = recognizer.listen(source, timeout=3.0, phrase_time_limit=8.0)
                        with open(temp_wav_path, "wb") as f:
                            f.write(audio_data.get_wav_data())
                        
                        segments, _ = audio_model.transcribe(temp_wav_path, beam_size=3, language="en", vad_filter=True, initial_prompt="Argus.")
                        spoken_chunk = "".join([s.text for s in segments]).strip()
                        
                        if not is_hallucination(spoken_chunk):
                            class_transcript.append(spoken_chunk)
                            ws_manager.broadcast_sync({"event": "lecture_transcript", "text": spoken_chunk})
                            print(f"[OBSERVER AUDIO] Logged: '{spoken_chunk}'")
                    except sr.WaitTimeoutError:
                        continue

                else:
                    listen_timeout = 5.0 if assistant_state == "active" else 2.0
                    try:
                        audio_data = recognizer.listen(source, timeout=listen_timeout, phrase_time_limit=8.0)
                    except sr.WaitTimeoutError:
                        if assistant_state == "active" and not pygame.mixer.music.get_busy():
                            assistant_state = "idle"
                            ws_manager.broadcast_sync({"event": "perception_log", "engine": "SYSTEM", "text": "Listening timed out. Standing by."})
                        continue

                    with open(temp_wav_path, "wb") as f:
                        f.write(audio_data.get_wav_data())

                    segments, _ = audio_model.transcribe(temp_wav_path, beam_size=3, language="en", vad_filter=True, initial_prompt="Argus.")
                    user_spoken = "".join([s.text for s in segments]).strip()
                    clean_text = user_spoken.translate(str.maketrans('', '', string.punctuation)).lower()

                    if is_hallucination(user_spoken):
                        continue

                    is_currently_speaking = pygame.mixer.music.get_busy() or not tts_queue.empty()

                    if is_stop_command(clean_text):
                        print(f"[INTERRUPT] User commanded: '{user_spoken}' -> Stopping immediately.")
                        interrupt_tts()
                        play_cancel_chime()
                        assistant_state = "idle"
                        ws_manager.broadcast_sync({"event": "perception_log", "engine": "SYSTEM", "text": "Interrupted and silenced."})
                        continue

                    if is_currently_speaking:
                        print(f"[BARGE-IN] Interrupted speech with new question: '{user_spoken}'")
                        interrupt_tts()
                        play_capture_chime()
                        ws_manager.broadcast_sync({"event": "speech_recognized", "text": user_spoken})
                        prompt_queue.put(user_spoken)
                        assistant_state = "active"
                        continue

                    if assistant_state == "idle":
                        if any(variant in clean_text for variant in wake_variants):
                            play_startup_chime()
                            ws_manager.broadcast_sync({"event": "speech_recognized", "text": user_spoken})
                            
                            remainder = clean_text
                            for v in wake_variants:
                                remainder = remainder.replace(v, "").strip()

                            if len(remainder) > 3:
                                prompt_queue.put(remainder)
                            else:
                                greeting = "Yeah? I'm here." if ws_manager.current_mode != "assistive" else "I'm listening."
                                tts_queue.put(greeting)
                            
                            assistant_state = "active"

                    elif assistant_state == "active":
                        play_capture_chime()
                        ws_manager.broadcast_sync({"event": "speech_recognized", "text": user_spoken})
                        prompt_queue.put(user_spoken)

        except Exception:
            pass
        time.sleep(0.04)

def llm_worker():
    global system_is_running
    while system_is_running:
        try:
            user_prompt = prompt_queue.get(timeout=0.5)
            process_prompt(user_prompt)
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
threading.Thread(target=hardware_metrics_worker, daemon=True).start() # <--- RE-ADDED


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
    global latest_cv2_frame, class_transcript, class_whiteboard_frames, last_snapshot_time
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
                        latest_cv2_frame = frame
                except Exception:
                    pass

            elif event == "trigger_lecture_evaluation":
                print("[SYSTEM] Manual Post-Class Tutor Triggered from UI.")
                interrupt_tts()
                play_capture_chime()
                prompt_queue.put("Provide a comprehensive summary of today's class and generate 2 practice questions to test my knowledge.")

            elif event == "user_prompt":
                prompt_text = data.get("text", "")
                print(f"[WEBSITE PROMPT]: {prompt_text}")
                interrupt_tts()
                prompt_queue.put(prompt_text)

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    print("\n[SYSTEM] A.R.G.U.S. Multi-App Engine Launching at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)