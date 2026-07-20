import torch
import torch.nn as nn
import json
import urllib.request
import os
import cv2
import threading
import time
import queue
import textwrap 
import collections
import numpy as np
import speech_recognition as sr
import pyttsx3
from PIL import Image
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo

try:
    import winsound
except ImportError:
    winsound = None

# ==========================================
# MOONDREAM2 COMPATIBILITY PATCH
# ==========================================
_orig_getattr = torch.nn.Module.__getattr__
def _patched_getattr(self, name):
    if name == "all_tied_weights_keys":
        return {} 
    return _orig_getattr(self, name)
torch.nn.Module.__getattr__ = _patched_getattr


# ==========================================
# A.R.G.U.S. MASTER ENGINE 
# ==========================================
print("\n[SYSTEM] Booting A.R.G.U.S. Multimodal Engine...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------
# 1. LOAD AI MODELS & ENGINES
# ------------------------------------------
print("[SYSTEM] Loading Temporal Brain (X3D with Custom Weights)...")
vision_model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)

# Swap head for your 5 custom classes and load the .pth you trained!
NUM_CUSTOM_CLASSES = 5
in_features = vision_model.blocks[5].proj.in_features
vision_model.blocks[5].proj = nn.Linear(in_features, NUM_CUSTOM_CLASSES)

custom_weights_path = "models/argus_x3d_finetuned.pth"
if os.path.exists(custom_weights_path):
    vision_model.load_state_dict(torch.load(custom_weights_path, map_location=device))
    print(f"[SYSTEM] Successfully loaded custom weights from {custom_weights_path}")
else:
    print(f"[WARNING] Could not find {custom_weights_path}! Using random weights for custom classes.")

vision_model = vision_model.to(device).eval()

# Update these to match the 5 classes you used during fine-tuning!
custom_classnames = {
    0: "Washing Dishes",
    1: "Chopping Vegetables",
    2: "Opening Fridge",
    3: "Wiping Counter",
    4: "Cooking on Stove"
}

print("[SYSTEM] Loading Spatial Brain (Moondream2)...")
md_model_id = "vikhyatk/moondream2"
moondream = AutoModelForCausalLM.from_pretrained(
    md_model_id, trust_remote_code=True, torch_dtype=torch.float16
).to(device)
md_tokenizer = AutoTokenizer.from_pretrained(md_model_id, trust_remote_code=True)

print("[SYSTEM] Loading Audio Brain (Faster Whisper)...")
audio_model = WhisperModel("tiny.en", device=device, compute_type="float16" if device == "cuda" else "int8")

# ------------------------------------------
# 2. CONFIGURATION & VARIABLES
# ------------------------------------------
target_video = r"F:\Project CSE499\Project_A-R-G-U-S\videos\P01_05.mp4" 

transform = ApplyTransformToKey(
    key="video",
    transform=Compose([
        UniformTemporalSubsample(13),
        Lambda(lambda x: x / 255.0),
        NormalizeVideo([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
        ShortSideScale(size=160),
        CenterCropVideo(crop_size=(160, 160)),
    ]),
)

prediction_text = "VISION: BUFFERING FRAMES..."
subtitle_text = "AUDIO: LISTENING TO MICROPHONE..."
current_action_label = "nothing" 
argus_reply_text = "" 
reply_timestamp = 0.0  # Added to clear text after 8 seconds

# Global state trackers
memory_log = [] 
current_cv2_frame = None 
frame_buffer = collections.deque(maxlen=60) # 2 seconds of rolling video frames in RAM
system_is_running = True
assistant_state = "idle"

# Audio Synchronization Tools
tts_queue = queue.Queue()
tts_done_event = threading.Event()
tts_engine = None
tts_interrupt_requested = False
thinking_active = False

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
    global tts_engine
    try:
        tts_engine = pyttsx3.init(driverName="sapi5")
        tts_engine.setProperty("rate", 170)
        tts_engine.say(text_to_say)
        tts_engine.runAndWait()
        tts_engine.stop()
        tts_engine = None
    except Exception:
        pass

def process_prompt(user_spoken_prompt):
    global argus_reply_text, thinking_active, reply_timestamp
    thinking_active = True
    if current_cv2_frame is not None:
        rgb_frame = cv2.cvtColor(current_cv2_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        print(f"[MOONDREAM] Analyzing spatial frame for prompt: '{user_spoken_prompt}'")
        encoded_image = moondream.encode_image(pil_image)
        answer = moondream.answer_question(encoded_image, user_spoken_prompt, md_tokenizer)

        argus_reply_text = f"A.R.G.U.S: {answer.upper()}"
        reply_timestamp = time.time()  # Start the 8-second timer
        tts_queue.put(answer)
        
        del encoded_image
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
        tts_queue.put("I cannot see the video feed right now.")
    thinking_active = False

def interrupt_tts():
    global tts_interrupt_requested, tts_engine
    tts_interrupt_requested = True
    if tts_engine is not None:
        try: tts_engine.stop()
        except: pass

def detect_wake_word(transcription):
    text = (transcription or "").lower().replace("-", " ").replace(",", " ").replace(".", " ").replace("!", " ").replace("?", " ")
    words = text.split()
    wake_variants = ["argus", "aragus", "arghas", "argas", "arghus", "argos", "argyz"]
    return any(variant in words for variant in wake_variants)

def extract_followup_prompt(transcription):
    text = (transcription or "").lower()
    wake_variants = ["hey argus", "hi argus", "okay argus", "argus"]
    for marker in wake_variants:
        if marker in text:
            return text.replace(marker, "", 1).strip()
    return text.strip()

# ------------------------------------------
# 3. BACKGROUND THREAD 1: VISION WORKER (RAM BUFFERED)
# ------------------------------------------
def vision_worker():
    global prediction_text, system_is_running, current_action_label, memory_log
    
    while system_is_running:
        if len(frame_buffer) == 60:  # Only run if we have a full 2-second buffer
            try:
                # Convert RAM frames directly to PyTorch Tensor (No Hard Drive usage)
                frames_list = list(frame_buffer)
                video_tensor = torch.from_numpy(np.stack(frames_list))
                video_tensor = video_tensor.permute(3, 0, 1, 2).float() # [C, T, H, W]

                video_data = {"video": video_tensor}
                video_data = transform(video_data)
                inputs = video_data["video"].to(device)[None, ...]

                with torch.no_grad():
                    preds = vision_model(inputs)
                    preds = torch.nn.Softmax(dim=1)(preds)
                    top_idx = preds.topk(k=1).indices[0][0].item()
                    confidence = preds.topk(k=1).values[0][0].item() * 100
                    
                current_action_label = custom_classnames.get(top_idx, "Unknown")
                prediction_text = f"TARGET LOCKED: {current_action_label.upper()} ({confidence:.1f}%)"
                
                if confidence > 50.0:
                    if len(memory_log) == 0 or memory_log[-1] != current_action_label:
                        memory_log.append(current_action_label)
                        
            except Exception as e:
                print(f"[VISION ERROR] {e}")
                
            # Sleep for 1 second before sampling the next 2-second window
            time.sleep(1.0) 
        else:
            time.sleep(0.1)

# ------------------------------------------
# 4. BACKGROUND THREAD 2: VOCAL ENGINE (pyttsx3)
# ------------------------------------------
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

# ------------------------------------------
# 5. BACKGROUND THREAD 3: AUDIO WORKER (RAM PROCESSING)
# ------------------------------------------
def audio_worker():
    global subtitle_text, system_is_running, current_action_label, argus_reply_text, memory_log, current_cv2_frame, tts_done_event, thinking_active, assistant_state, reply_timestamp
    recognizer = sr.Recognizer()

    try:
        mic = sr.Microphone()
        print("[AUDIO] Microphone detected successfully.")
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("[AUDIO] Calibration complete. Listening active.")
    except Exception as e:
        subtitle_text = "AUDIO ERROR: No microphone found!"
        return

    while system_is_running:
        try:
            with mic as source:
                if assistant_state == "awaiting_prompt":
                    subtitle_text = "MIC: (I AM LISTENING - SPEAK NOW!)"
                    try:
                        recognizer.adjust_for_ambient_noise(source, duration=0.2)
                        command_audio = recognizer.listen(source, timeout=5.0, phrase_time_limit=5.0)
                    except sr.WaitTimeoutError:
                        tts_queue.put("I didn't hear a command.")
                        assistant_state = "idle"
                        continue

                    # Direct RAM audio conversion (No hard drive thrashing!)
                    audio_data = np.frombuffer(command_audio.get_raw_data(), np.int16).astype(np.float32) / 32768.0
                    segments, _ = audio_model.transcribe(audio_data, beam_size=1)
                    user_spoken_prompt = "".join([segment.text for segment in segments]).strip().lower()
                    
                    if user_spoken_prompt:
                        subtitle_text = f"MIC: {user_spoken_prompt}"
                        play_capture_chime()
                    else:
                        user_spoken_prompt = "Describe the main objects in this scene."
                        
                    assistant_state = "idle"
                    process_prompt(user_spoken_prompt)
                    continue

                audio_data = recognizer.listen(source, timeout=2.0, phrase_time_limit=4.0)
                
                # Direct RAM audio conversion
                raw_np = np.frombuffer(audio_data.get_raw_data(), np.int16).astype(np.float32) / 32768.0
                segments, _ = audio_model.transcribe(raw_np, beam_size=1)
                transcription = "".join([segment.text for segment in segments]).strip().lower()

                if transcription:
                    subtitle_text = f"MIC: {transcription}"

                    if "what is" in transcription and "doing" in transcription:
                        answer = f"Right now, he is {current_action_label}."
                        argus_reply_text = f"A.R.G.U.S: {answer.upper()}"
                        reply_timestamp = time.time()
                        tts_queue.put(answer)

                    elif "first" in transcription or "start" in transcription:
                        if len(memory_log) > 0:
                            answer = f"At the start of the video, he was {memory_log[0]}."
                        else:
                            answer = "I haven't recorded any actions yet."
                        argus_reply_text = f"A.R.G.U.S: {answer.upper()}"
                        reply_timestamp = time.time()
                        tts_queue.put(answer)

                    elif detect_wake_word(transcription):
                        interrupt_tts()
                        user_spoken_prompt = extract_followup_prompt(transcription)

                        subtitle_text = "MIC: HEY ARGUS..."
                        tts_done_event.clear()
                        tts_queue.put("I am listening.")
                        threading.Thread(target=play_startup_chime, daemon=True).start()

                        tts_done_event.wait(timeout=3.0)
                        time.sleep(0.25)

                        if user_spoken_prompt.strip():
                            subtitle_text = f"MIC: {user_spoken_prompt}"
                            play_capture_chime()
                            assistant_state = "idle"
                            process_prompt(user_spoken_prompt)
                        else:
                            assistant_state = "awaiting_prompt"

        except sr.WaitTimeoutError: pass
        except Exception: pass
        time.sleep(0.1)

# Start all three AI brains
threading.Thread(target=vision_worker, daemon=True).start()
threading.Thread(target=tts_worker, daemon=True).start()
threading.Thread(target=audio_worker, daemon=True).start()

# ------------------------------------------
# 6. THE STREAMING HUD LOOP (MAIN THREAD)
# ------------------------------------------
print("[SYSTEM] Launching Real-Time HUD. Press 'q' to close window.")
cap = cv2.VideoCapture(target_video)

cv2.namedWindow("A.R.G.U.S. Vision Engine", cv2.WINDOW_NORMAL)
cv2.resizeWindow("A.R.G.U.S. Vision Engine", 1280, 720) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        # Loop the video for continuous testing
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue 

    frame_height, frame_width, _ = frame.shape
    current_cv2_frame = frame.copy() 
    
    # Push the frame to the RAM buffer for X3D (Resized for performance)
    rgb_small = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2RGB)
    frame_buffer.append(rgb_small)

    # 1. Draw Top HUD
    cv2.rectangle(frame, (10, 10), (800, 60), (0, 0, 0), -1) 
    cv2.putText(frame, prediction_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # 2. Draw Bottom HUD 
    cv2.rectangle(frame, (10, frame_height - 50), (frame_width - 10, frame_height - 10), (0, 0, 0), -1)
    cv2.putText(frame, subtitle_text, (65, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # 3. Clear Stale Answers (Timeout after 8 seconds)
    if argus_reply_text != "" and (time.time() - reply_timestamp > 8.0):
        argus_reply_text = ""

    # 4. Draw Thinking Animation
    if thinking_active:
        spinner = [".  ", ".. ", "...", ".. "]
        dots = spinner[int(time.time() * 6) % len(spinner)]
        cv2.putText(frame, f"THINKING{dots}", (frame_width - 220, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # 5. Dynamic Text Wrapping
    if argus_reply_text != "":
        wrapped_lines = textwrap.wrap(argus_reply_text, width=65)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 2
        line_height = cv2.getTextSize("Ay", font, font_scale, thickness)[0][1] + 15
        total_height = len(wrapped_lines) * line_height
        
        start_y = (frame_height - total_height) // 2
        
        max_width = max([cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in wrapped_lines])
        box_x1 = (frame_width - max_width) // 2 - 20
        box_y1 = start_y - line_height
        box_x2 = (frame_width + max_width) // 2 + 20
        box_y2 = start_y + total_height + 5
        
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
        
        for i, line in enumerate(wrapped_lines):
            line_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            line_x = (frame_width - line_size[0]) // 2
            line_y = start_y + (i * line_height)
            cv2.putText(frame, line, (line_x, line_y), font, font_scale, (255, 150, 100), thickness, cv2.LINE_AA)

    cv2.imshow("A.R.G.U.S. Vision Engine", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'): break

system_is_running = False 
cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Shutdown complete.")