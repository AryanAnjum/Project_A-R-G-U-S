import asyncio
import base64
import collections
import json
import os
import queue
import tempfile
import threading
import time
import urllib.request
from PIL import Image

import cv2
import edge_tts
import numpy as np
import pygame
import torch

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
from faster_whisper import WhisperModel
import speech_recognition as sr

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


from contextlib import asynccontextmanager

# ==========================================
# 1. FASTAPI & WEBSOCKET ENGINE SETUP
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs when the server starts up
    ws_manager.loop = asyncio.get_running_loop()
    yield
    # Anything after yield runs on shutdown (we don't need anything here)

app = FastAPI(title="A.R.G.U.S. Live Multimodal Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.loop = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print("[SERVER] Web Dashboard Connected")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print("[SERVER] Web Dashboard Disconnected")

    def broadcast_sync(self, message: dict):
        """Thread-safe sync wrapper to broadcast events to frontend WebSocket."""
        if self.loop and self.active_connections:
            asyncio.run_coroutine_threadsafe(self._broadcast(message), self.loop)

    async def _broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                pass

ws_manager = ConnectionManager()


# ==========================================
# 2. LOAD AI MODELS & GLOBALS
# ==========================================
print("\n[SYSTEM] Booting A.R.G.U.S. Live Multimodal Engine...")
device = "cuda" if torch.cuda.is_available() else "cpu"

print("[SYSTEM] Loading Temporal Brain (X3D)...")
vision_model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True).to(device).eval()

url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
if not os.path.exists("kinetics_classnames.json"):
    urllib.request.urlretrieve(url, "kinetics_classnames.json")
with open("kinetics_classnames.json", "r") as f:
    kinetics_classnames = json.load(f)
kinetics_id_to_classname = {v: str(k).replace('"', "").replace('_', ' ') for k, v in kinetics_classnames.items()}

print("[SYSTEM] Loading Spatial Brain (Qwen2-VL 2B)...")
spatial_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", 
    torch_dtype=torch.bfloat16, 
    device_map="cuda"
)
spatial_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

print("[SYSTEM] Loading Audio Brain (Whisper)...")
audio_model = WhisperModel("tiny.en", device=device, compute_type="float16" if device == "cuda" else "int8")


# ==========================================
# 3. STATE TRACKERS & ROLLING FRAME BUFFERS
# ==========================================
# Rolling 30-frame buffer storing real-time webcam frames for X3D action recognition
frame_buffer = collections.deque(maxlen=30) 
latest_cv2_frame = None 

prediction_text = "VISION: INITIALIZING..."
current_action_label = "nothing" 
argus_reply_text = "" 

memory_log = [] 
conversation_memory = collections.deque(maxlen=3) 
system_is_running = True
assistant_state = "idle"

tts_queue = queue.Queue()
prompt_queue = queue.Queue() 
tts_done_event = threading.Event()
tts_interrupt_requested = False


# ==========================================
# 4. HELPER FUNCTIONS & TTS
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

def speak_text(text_to_say):
    global tts_interrupt_requested
    async def _speak():
        communicate = edge_tts.Communicate(text_to_say, "en-US-ChristopherNeural", rate="+10%")
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
        except: pass

    asyncio.run(_speak())

def process_prompt(user_spoken_prompt):
    global argus_reply_text, conversation_memory
    if latest_cv2_frame is not None:
        rgb_frame = cv2.cvtColor(latest_cv2_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        print(f"[QWEN-VL] Scanning live webcam frame for: '{user_spoken_prompt}'")
        
        memory_context = ""
        if len(conversation_memory) > 0:
            memory_context = "Recent conversation context:\n"
            for past_q, past_a in conversation_memory:
                memory_context += f"User: {past_q}\nYou: {past_a}\n"
        
        system_prompt = (
            "You are A.R.G.U.S., a highly observant and friendly AI companion. "
            "Look very closely at the fine details in the live view. "
            "Answer the user's question naturally and conversationally in one or two short sentences."
        )
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": f"{system_prompt}\n\n{memory_context}\nQuestion: {user_spoken_prompt}"}
            ]
        }]
        
        text = spatial_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = spatial_processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to("cuda")
        
        generated_ids = spatial_model.generate(**inputs, max_new_tokens=60, temperature=0.3, do_sample=True)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        
        answer = spatial_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].replace('"', '').strip()

        if len(answer) > 200: answer = answer[:200] + "..."

        conversation_memory.append((user_spoken_prompt, answer))
        argus_reply_text = f"{answer}"

        # Sync reply to both Website UI Chat and Vocal Speaker
        ws_manager.broadcast_sync({"event": "argus_reply", "text": answer})
        ws_manager.broadcast_sync({"event": "perception_log", "engine": "Qwen2-VL", "text": f"Inferred prompt: '{user_spoken_prompt}'"})
        
        tts_queue.put(answer)
        
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()
    else:
        ws_manager.broadcast_sync({"event": "argus_reply", "text": "I cannot see the live video feed right now."})


# ==========================================
# 5. BACKGROUND AI WORKERS
# ==========================================

# --- WORKER 1: LIVE X3D ACTION RECOGNITION ---
def vision_worker():
    global prediction_text, system_is_running, current_action_label
    
    # Pre-transforms for X3D
    x3d_transform = Compose([
        UniformTemporalSubsample(13),
        Lambda(lambda x: x / 255.0),
        NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
        ShortSideScale(size=160),
        CenterCropVideo(crop_size=(160, 160)),
    ])

    while system_is_running:
        # Require at least 13 frames in the live stream buffer
        if len(frame_buffer) >= 13:
            try:
                # Extract 13 frames temporally from the live buffer
                frames_list = list(frame_buffer)
                step = max(1, len(frames_list) // 13)
                sampled_frames = frames_list[::step][:13]

                # Convert BGR frames into (C, T, H, W) PyTorch Tensor
                tensor_frames = [torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2, 0, 1) for f in sampled_frames]
                video_tensor = torch.stack(tensor_frames).permute(1, 0, 2, 3).float() # [C, T, H, W]

                # Apply X3D normalization
                video_tensor = x3d_transform({"video": video_tensor})["video"].to(device)[None, ...]

                with torch.no_grad():
                    preds = vision_model(video_tensor)
                    preds = torch.nn.Softmax(dim=1)(preds)
                    top_idx = preds.topk(k=1).indices[0][0].item()
                    confidence = preds.topk(k=1).values[0][0].item() * 100

                current_action_label = kinetics_id_to_classname[top_idx]
                prediction_text = f"{current_action_label.upper()} ({confidence:.1f}%)"

                # Send log to website
                if confidence > 50.0:
                    ws_manager.broadcast_sync({
                        "event": "perception_log", 
                        "engine": "X3D", 
                        "text": f"Action Detected: {current_action_label.title()} ({confidence:.1f}%)"
                    })

            except Exception as e:
                pass

        time.sleep(1.0) # Run action recognition every 1 second


# --- WORKER 2: VOCAL TTS ENGINE ---
def tts_worker():
    global system_is_running, tts_done_event, tts_interrupt_requested
    while system_is_running:
        try:
            text_to_say = tts_queue.get(timeout=0.5)
            tts_interrupt_requested = False
            tts_done_event.clear()
            speak_text(text_to_say)
            tts_done_event.set()
        except queue.Empty:
            pass


# --- WORKER 3: AUDIO / MICROPHONE LISTENER ---
def audio_worker():
    global system_is_running, assistant_state
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 3000  
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 1.0 

    # Route the wav file to your OS Temp folder so VS Code doesn't refresh the page!
    temp_wav_path = os.path.join(tempfile.gettempdir(), "argus_temp_mic.wav")

    try:
        mic = sr.Microphone()
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1.5)
            print("[AUDIO] Hardware microphone calibrated and ready.")
    except Exception as e:
        print(f"[AUDIO ERROR] No microphone found: {e}")
        return

    assistant_state = "idle"
    wake_variants = ["argus", "argos", "august", "arghus", "argas", "aragus", "hey argus"]

    while system_is_running:
        try:
            with mic as source:
                if assistant_state == "active":
                    try:
                        command_audio = recognizer.listen(source, timeout=8.0, phrase_time_limit=12.0)
                    except sr.WaitTimeoutError:
                        assistant_state = "idle"
                        ws_manager.broadcast_sync({"event": "perception_log", "engine": "SYSTEM", "text": "Listening timed out. Going to sleep."})
                        continue

                    # Save to the hidden temp folder
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
                        audio_data = recognizer.listen(source, timeout=2.0, phrase_time_limit=5.0)
                    except sr.WaitTimeoutError:
                        continue 

                    # Save to the hidden temp folder
                    with open(temp_wav_path, "wb") as f:
                        f.write(audio_data.get_wav_data())

                    segments, _ = audio_model.transcribe(temp_wav_path, beam_size=5, language="en")
                    transcription = "".join([s.text for s in segments]).strip().lower()

                    if transcription:
                        print(f"[DEBUG MIC] Heard: '{transcription}'") 

                    if any(variant in transcription for variant in wake_variants):
                        print("[AUDIO] Wake word triggered!")
                        play_startup_chime()
                        ws_manager.broadcast_sync({"event": "speech_recognized", "text": transcription})
                        ws_manager.broadcast_sync({"event": "perception_log", "engine": "WHISPER", "text": "Wake Word Detected!"})
                        tts_queue.put("I am listening, how can I help?")
                        assistant_state = "active"

        except Exception as e:
            pass
        time.sleep(0.1)


# --- WORKER 4: LLM INFERENCE DISPATCHER ---
def llm_worker():
    global system_is_running
    while system_is_running:
        try:
            user_prompt = prompt_queue.get(timeout=0.5)
            process_prompt(user_prompt)
        except queue.Empty:
            pass


# Start Background Threads
threading.Thread(target=vision_worker, daemon=True).start()
threading.Thread(target=tts_worker, daemon=True).start()
threading.Thread(target=audio_worker, daemon=True).start()
threading.Thread(target=llm_worker, daemon=True).start()


# ==========================================
# 6. FASTAPI WEB ROUTES & WEBSOCKETS
# ==========================================
@app.get("/")
async def serve_dashboard():
    """Serves the main A.R.G.U.S. HUD Interface"""
    # Assumes index.html is in the exact same folder as your Python script
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html not found!</h1><p>Please place index.html in the same directory as this Python script.</p>")
    
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
                mode = data.get("mode", "live")
                print(f"[SYSTEM] Frontend mode set to: {mode}")

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
                        frame_buffer.append(frame) 
                except Exception as e:
                    print(f"[VISION ERROR] Frame drop: {e}")

            elif event == "user_prompt":
                prompt_text = data.get("text", "")
                print(f"[WEBSITE PROMPT]: {prompt_text}")
                prompt_queue.put(prompt_text)

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        print(f"[WS ERROR] {e}")
        ws_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    print("\n[SYSTEM] Server launching at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)