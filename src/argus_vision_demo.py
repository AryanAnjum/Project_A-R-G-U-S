import torch
import json
import urllib.request
import os
import cv2
import threading
import time
import speech_recognition as sr
from faster_whisper import WhisperModel
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo

# ==========================================
# A.R.G.U.S. MULTIMODAL DEMO ENGINE (VISION + AUDIO)
# ==========================================
print("\n[SYSTEM] Booting A.R.G.U.S. Multimodal Engine...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------
# 1. LOAD AI MODELS
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
# Using the "tiny" model so it runs instantly alongside the video model
audio_model = WhisperModel("tiny.en", device=device, compute_type="float16" if device == "cuda" else "int8")

# ------------------------------------------
# 2. CONFIGURATION & VARIABLES
# ------------------------------------------
target_video = r"videos\P01_06.mp4" 

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
current_time_sec = 0.0
system_is_running = True

# ------------------------------------------
# 3. BACKGROUND THREAD 1: VISION WORKER
# ------------------------------------------
def vision_worker():
    global prediction_text, current_time_sec, system_is_running
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
                    
                prediction_text = f"TARGET LOCKED: {kinetics_id_to_classname[top_idx].upper()} ({confidence:.1f}%)"
                last_inferred_time = infer_time
            except Exception:
                pass
        time.sleep(0.05)

# ------------------------------------------
# 4. BACKGROUND THREAD 2: AUDIO WORKER (DEBUG MODE)
# ------------------------------------------
def audio_worker():
    global subtitle_text, system_is_running
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
                print("[AUDIO] Listening...")
                # Listen in short bursts
                audio_data = recognizer.listen(source, timeout=2, phrase_time_limit=3)
                print("[AUDIO] Processing speech...")
                
                # Save to temporary file
                with open("temp_mic.wav", "wb") as f:
                    f.write(audio_data.get_wav_data())
                
                # Transcribe
                segments, _ = audio_model.transcribe("temp_mic.wav", beam_size=1)
                transcription = "".join([segment.text for segment in segments]).strip()
                
                if transcription:
                    print(f"[AUDIO WHISPER] Transcribed: {transcription}")
                    subtitle_text = f"MIC: {transcription}"
                else:
                    print("[AUDIO WHISPER] Heard sound, but no words detected.")
        except sr.WaitTimeoutError:
            print("[AUDIO] Listening timeout (silence)...")
        except Exception as e:
            print(f"[AUDIO LOOP ERROR]: {e}")
            
        time.sleep(0.1)

# Start both AI brains in the background
threading.Thread(target=vision_worker, daemon=True).start()
threading.Thread(target=audio_worker, daemon=True).start()

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

    # Draw Top HUD (Vision)
    cv2.rectangle(frame, (10, 10), (800, 60), (0, 0, 0), -1) 
    cv2.putText(frame, prediction_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Draw Bottom HUD (Audio / Subtitles)
    cv2.rectangle(frame, (10, frame_height - 60), (frame_width - 10, frame_height - 10), (0, 0, 0), -1)
    cv2.putText(frame, subtitle_text, (20, frame_height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("A.R.G.U.S. Vision Engine", frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Cleanup
system_is_running = False 
cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Shutdown complete.")