import collections
import os
import queue
import sys
import numpy as np
import pyaudio
import webrtcvad
import time
from datetime import datetime
from faster_whisper import WhisperModel

if sys.platform == "win32":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nvidia_cublas_bin = os.path.join(project_root, ".ARGUS_venv", "Lib", "site-packages", "nvidia", "cublas", "bin")
    if os.path.isdir(nvidia_cublas_bin) and nvidia_cublas_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{nvidia_cublas_bin};" + os.environ.get("PATH", "")

# ==========================================
# A.R.G.U.S. HARDWARE CONFIGURATION
# ==========================================
MODEL_SIZE = "base.en" 
DEVICE = "cuda"
COMPUTE_TYPE = "float16" 

RATE = 16000
CHUNK = 320 # 20ms audio chunk

# ANSI Color Codes for Terminal Polish
CYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'

print(f"\n{CYAN}=======================================")
print("[SYSTEM] Booting A.R.G.U.S. Hearing Engine...")
print(f"[SYSTEM] Allocating {MODEL_SIZE} to RTX 3060 ({COMPUTE_TYPE})...")
print(f"======================================={RESET}\n")

# 1. Load the Model
audio_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print(f"{GREEN}[SYSTEM] VRAM Allocation Successful. A.R.G.U.S. is listening...{RESET}\n")

vad = webrtcvad.Vad(1) 
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

def record_audio_chunk():
    frames = []
    silence_frames = 0
    is_speaking = False

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        is_speech = vad.is_speech(data, RATE)

        if is_speech:
            if not is_speaking:
                # Optional: You can uncomment the line below to see exactly when it hears you
                # print(f"{YELLOW}[VAD] Speech detected, buffering...{RESET}", end="\r")
                pass
            is_speaking = True
            silence_frames = 0
            frames.append(data)
        elif is_speaking:
            silence_frames += 1
            frames.append(data)
            if silence_frames > 50: 
                break
    
    return b''.join(frames)

# 3. Inference Loop with Telemetry and Filtering
try:
    while True:
        audio_data = record_audio_chunk()
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Start inference timer
        start_time = time.time()
        
        segments, info = audio_model.transcribe(audio_np, beam_size=5, language="en")
        
        # Calculate how fast the RTX 3060 processed it
        inference_time = time.time() - start_time
        current_time = datetime.now().strftime("%H:%M:%S")

        for segment in segments:
            text = segment.text.strip()
            
            # --- THE HALLUCINATION FILTER ---
            # Ignore weird Whisper artifacts and anything less than 3 characters
            bad_words = ["you", "you.", "thank you.", "thank you", "am.", "oh."]
            if len(text) > 2 and text.lower() not in bad_words:
                
                # The Tactical Output
                print(f"[{current_time}] {CYAN}[Lat: {inference_time:.2f}s]{RESET} --> {text}")

except KeyboardInterrupt:
    print(f"\n{YELLOW}[SYSTEM] Shutting down A.R.G.U.S. intake stream...{RESET}")
    stream.stop_stream()
    stream.close()
    audio.terminate()