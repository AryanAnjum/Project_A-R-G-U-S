import torch
import json
import urllib.request
import os
import cv2
import threading
import queue
import time
import speech_recognition as sr
import pyttsx3
from faster_whisper import WhisperModel
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo

# ==========================================
# A.R.G.U.S. MULTIMODAL DEMO ENGINE (VISION + AUDIO + MEMORY)
# ==========================================
print("\n[SYSTEM] Booting A.R.G.U.S. Multimodal Engine...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------
# 1. LOAD AI MODELS & ENGINES
# ------------------------------------------
print("[SYSTEM] Loading Vision Brain (X3D)...")
vision_model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True).to(device).eval()

url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
if not os.path.exists("kinetics_classnames.json"):
    urllib.request.urlretrieve(url, "kinetics_classnames.json")
with open("kinetics_classnames.json", "r") as f:
    kinetics_classnames = json.load(f)
kinetics_id_to_classname = {v: str(k).replace('"', "").replace('_', ' ') for k, v in kinetics_classnames.items()}

print("[SYSTEM] Loading Audio Brain (Faster Whisper)...")
audio_model = WhisperModel("tiny.en", device=device, compute_type="float16" if device == "cuda" else "int8")

# ------------------------------------------
# 2. CONFIGURATION & VARIABLES
# ------------------------------------------
target_video = r"../videos/P01_06.mp4" 

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

prediction_text = "VISION: INITIALIZING..."
subtitle_text = "AUDIO: LISTENING TO MICROPHONE..."
current_action_label = "nothing" 
argus_reply_text = "" 

# --- SHORT-TERM MEMORY SYSTEM ---
memory_log = [] 

current_time_sec = 0.0
system_is_running = True

# ------------------------------------------
# 3. BACKGROUND THREAD 1: VISION WORKER
# ------------------------------------------
def vision_worker():
    global prediction_text, current_time_sec, system_is_running, current_action_label, memory_log
    ai_video_reader = EncodedVideo.from_path(target_video)
    last_inferred_time = -1.0
    
    while system_is_running:
        if current_time_sec - last_inferred_time >= 1.0:
            infer_time = current_time_sec
            try:
                video_data = ai_video_reader.get_clip(start_sec=infer_time, end_sec=infer_time + 2.0)
                video_data = transform(video_data)
                inputs = video_data["video"].to(device)[None, ...]

                with torch.no_grad():
                    preds = vision_model(inputs)
                    preds = torch.nn.Softmax(dim=1)(preds)
                    top_idx = preds.topk(k=1).indices[0][0].item()
                    confidence = preds.topk(k=1).values[0][0].item() * 100
                    
                current_action_label = kinetics_id_to_classname[top_idx]
                prediction_text = f"TARGET LOCKED: {current_action_label.upper()} ({confidence:.1f}%)"
                last_inferred_time = infer_time
                
                # --- MEMORY LOGIC ---
                if confidence > 50.0:
                    if len(memory_log) == 0 or memory_log[-1] != current_action_label:
                        memory_log.append(current_action_label)
                        print(f"[MEMORY] Logged new event: {current_action_label}")
                        
            except Exception:
                pass
        time.sleep(0.05)

# # ------------------------------------------
# 4. BACKGROUND THREAD 2 & 3: AUDIO & VOICE
# ------------------------------------------
# Create a secure pipeline between the listener and the speaker
tts_queue = queue.Queue()

def tts_worker():
    """This thread ONLY handles speaking. We initialize per-sentence to stop Windows driver lockups."""
    global system_is_running
    
    while system_is_running:
        try:
            # Check the queue for new things to say
            text_to_say = tts_queue.get(timeout=0.5)
            
            # --- THE FIX ---
            # Boot the engine, speak, and shut down the engine inside the loop.
            # This prevents SAPI5 from freezing after the first run.
            engine = pyttsx3.init()
            engine.setProperty('rate', 170)
            
            engine.say(text_to_say)
            engine.runAndWait()
            
            # Delete the engine to force a clean slate for the next sentence
            del engine 
            
        except queue.Empty:
            pass # Nothing to say right now, loop again

def audio_worker():
    global subtitle_text, system_is_running, current_action_label, argus_reply_text, memory_log
    recognizer = sr.Recognizer()
    
    try:
        mic = sr.Microphone()
        print("[AUDIO] Microphone detected successfully.")
        with mic as source:
            print("[AUDIO] Calibrating for background noise... (Please stay quiet for 1s)")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("[AUDIO] Calibration complete. Listening active.")
    except Exception as e:
        print(f"[AUDIO ERROR] Failed to initialize microphone: {e}")
        subtitle_text = "AUDIO ERROR: No microphone found!"
        return

    while system_is_running:
        try:
            with mic as source:
                audio_data = recognizer.listen(source, timeout=2, phrase_time_limit=3)
                
                with open("temp_mic.wav", "wb") as f:
                    f.write(audio_data.get_wav_data())
                
                segments, _ = audio_model.transcribe("temp_mic.wav", beam_size=1)
                transcription = "".join([segment.text for segment in segments]).strip().lower()
                
                if transcription:
                    print(f"[AUDIO WHISPER] Transcribed: {transcription}")
                    subtitle_text = f"MIC: {transcription}"
                    
                    # --- TRIGGER 1: LIVE STATUS ---
                    if "what is" in transcription and "doing" in transcription:
                        answer = f"Right now, he is {current_action_label}."
                        argus_reply_text = f"A.R.G.U.S: {answer.upper()}"
                        
                        # INSTEAD OF SPEAKING DIRECTLY, SEND IT TO THE QUEUE
                        tts_queue.put(answer)
                        
                    # --- TRIGGER 2: MEMORY RECALL ---
                    elif "first" in transcription or "start" in transcription:
                        if len(memory_log) > 0:
                            first_event = memory_log[0]
                            answer = f"At the start of the video, he was {first_event}."
                        else:
                            answer = "I haven't recorded any actions yet."
                            
                        argus_reply_text = f"A.R.G.U.S: {answer.upper()}"
                        
                        # SEND TO QUEUE
                        tts_queue.put(answer)
                        
        except sr.WaitTimeoutError:
            pass
        except Exception:
            pass
            
        time.sleep(0.1)

# Start all THREE AI brains in the background
threading.Thread(target=vision_worker, daemon=True).start()
threading.Thread(target=audio_worker, daemon=True).start()
threading.Thread(target=tts_worker, daemon=True).start() # NEW VOICE THREAD

# ------------------------------------------
# 5. THE STREAMING HUD LOOP (MAIN THREAD)
# ------------------------------------------
print("[SYSTEM] Launching Real-Time HUD. Press 'q' to close window.")
cap = cv2.VideoCapture(target_video)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break 

    current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    frame_height, frame_width, _ = frame.shape

    # 1. Draw Top HUD (Vision Prediction - Yellow)
    cv2.rectangle(frame, (10, 10), (800, 60), (0, 0, 0), -1) 
    cv2.putText(frame, prediction_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # 2. Draw Bottom HUD (Mic Subtitles - Green)
    cv2.rectangle(frame, (10, frame_height - 50), (frame_width - 10, frame_height - 10), (0, 0, 0), -1)
    cv2.putText(frame, subtitle_text, (20, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # 3. Draw AI Answer in the CENTER of the screen (Blue)
    if argus_reply_text != "":
        text_size = cv2.getTextSize(argus_reply_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (frame_width - text_size[0]) // 2
        text_y = (frame_height + text_size[1]) // 2
        
        cv2.rectangle(frame, (text_x - 15, text_y - text_size[1] - 15), 
                             (text_x + text_size[0] + 15, text_y + 15), (0, 0, 0), -1)
        cv2.putText(frame, argus_reply_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 150, 100), 3, cv2.LINE_AA)

    cv2.imshow("A.R.G.U.S. Vision Engine", frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Cleanup
system_is_running = False 
cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Shutdown complete.")