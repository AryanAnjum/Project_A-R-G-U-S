import torch
import json
import urllib.request
import os
import cv2
import threading
import time
import asyncio
import edge_tts
import pygame
import tempfile
import collections

# Initialize the new high-quality audio mixer
pygame.mixer.init()
import queue
import speech_recognition as sr
import pyttsx3
from PIL import Image

from transformers import AutoModelForCausalLM, AutoTokenizer
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from faster_whisper import WhisperModel

# DO NOT pass the path to the .pt file. 
# Just use the model name, and the library will download the correct CTranslate2 format.
MODEL_SIZE = "large-v3"
# This will automatically download the correct files on the first run.
# compute_type="int8" forces it to compress, keeping your 8GB VRAM safe.
whisper_model = WhisperModel(
    MODEL_SIZE, 
    device="cuda", 
    compute_type="int8" 
)

try:
    import winsound
except ImportError:
    winsound = None

# ==========================================
# MOONDREAM2 COMPATIBILITY PATCH
# Prevents the "all_tied_weights_keys" crash in newer transformers libraries
# ==========================================
_orig_getattr = torch.nn.Module.__getattr__
def _patched_getattr(self, name):
    if name == "all_tied_weights_keys":
        return {} 
    return _orig_getattr(self, name)
torch.nn.Module.__getattr__ = _patched_getattr


# ==========================================
# A.R.G.U.S. MASTER ENGINE 
# (X3D Temporal + Moondream Spatial + Whisper + Memory + Synchronized TTS)
# ==========================================
print("\n[SYSTEM] Booting A.R.G.U.S. Multimodal Engine...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------
# 1. LOAD AI MODELS & ENGINES
# ------------------------------------------
print("[SYSTEM] Loading Temporal Brain (X3D)...")
vision_model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True).to(device).eval()

url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
if not os.path.exists("kinetics_classnames.json"):
    urllib.request.urlretrieve(url, "kinetics_classnames.json")
with open("kinetics_classnames.json", "r") as f:
    kinetics_classnames = json.load(f)
kinetics_id_to_classname = {v: str(k).replace('"', "").replace('_', ' ') for k, v in kinetics_classnames.items()}

print("[SYSTEM] Loading Spatial Brain (Moondream2)...")
md_model_id = "vikhyatk/moondream2"
moondream = AutoModelForCausalLM.from_pretrained(
    md_model_id, trust_remote_code=True, torch_dtype=torch.float16
).to(device)
md_tokenizer = AutoTokenizer.from_pretrained(md_model_id, trust_remote_code=True)

print("[SYSTEM] Loading Audio Brain (Whisper)...")
audio_model = WhisperModel("tiny.en", device=device, compute_type="float16" if device == "cuda" else "int8")


# ------------------------------------------
# 2. CONFIGURATION & VARIABLES
# ------------------------------------------
target_video = r"./videos/test.mp4" 

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

# Global state trackers
memory_log = [] 
# Global state trackers
memory_log = [] 
conversation_memory = collections.deque(maxlen=3) # <-- NEW: Remembers the last 3 things you talked about
current_cv2_frame = None
current_cv2_frame = None 
current_time_sec = 0.0
system_is_running = True
assistant_state = "idle"

# Audio Synchronization Tools
tts_queue = queue.Queue()
tts_done_event = threading.Event()
tts_engine = None
tts_interrupt_requested = False
thinking_active = False
thinking_text = "THINKING..."
mic_animation_phase = 0


def play_beep(frequency=1800, duration=120):
    if winsound is None:
        return
    try:
        winsound.Beep(frequency, duration)
    except Exception:
        try:
            winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
        except Exception as e:
            try:
                winsound.MessageBeep()
            except Exception as inner_error:
                print(f"[SFX] Beep failed: {e} / {inner_error}")


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
    
    # We use an async function to pull the Neural Voice from the cloud instantly
    async def _speak():
        # "ChristopherNeural" is a highly natural, conversational male voice
        communicate = edge_tts.Communicate(text_to_say, "en-US-ChristopherNeural", rate="+10%")
        
        # Save it to a temporary audio file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.close()
        
        await communicate.save(temp_file.name)
        
        # Play the audio using PyGame so we can interrupt it if you say "Hey Argus"
        pygame.mixer.music.load(temp_file.name)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            if tts_interrupt_requested:
                pygame.mixer.music.stop()
                break
            time.sleep(0.05)
            
        pygame.mixer.music.unload()
        try:
            os.remove(temp_file.name)
        except:
            pass

    asyncio.run(_speak())

def process_prompt(user_spoken_prompt):
    global argus_reply_text, thinking_active, conversation_memory
    thinking_active = True
    if current_cv2_frame is not None:
        rgb_frame = cv2.cvtColor(current_cv2_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        print(f"[MOONDREAM] Analyzing spatial frame for prompt: '{user_spoken_prompt}'")
        encoded_image = moondream.encode_image(pil_image)
        
        # --- THE MEMORY INJECTION ---
        # If Argus remembers talking to you recently, it loads it into his active brain
        memory_context = ""
        if len(conversation_memory) > 0:
            memory_context = "Recent context (use this if asked about something you just saw):\n"
            for past_q, past_a in conversation_memory:
                memory_context += f"- User asked: '{past_q}' | You answered: '{past_a}'\n"
        
        conversational_prompt = (
            f"Act as a friendly, highly observant AI companion. "
            f"{memory_context}"
            f"Answer the following question naturally and conversationally in one or two short sentences. "
            f"Question: {user_spoken_prompt}"
        )
        
        answer = moondream.answer_question(
            encoded_image, 
            conversational_prompt, 
            md_tokenizer,
            max_new_tokens=60
        )

        answer = answer.replace('"', '').strip()
        if len(answer) > 200:
            answer = answer[:200] + "..."

        # Save this exact interaction into his short-term memory buffer
        conversation_memory.append((user_spoken_prompt, answer))

        argus_reply_text = f"A.R.G.U.S: {answer.upper()}"
        tts_queue.put(answer)
    else:
        tts_queue.put("I cannot see the video feed right now.")
    thinking_active = False

def interrupt_tts():
    global tts_interrupt_requested
    tts_interrupt_requested = True
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()



def detect_wake_word(transcription):
    text = (transcription or "").lower().replace("-", " ").replace(",", " ").replace(".", " ").replace("!", " ").replace("?", " ")
    words = text.split()
    wake_variants = ["argus", "aragus", "arghas", "argas", "arghus", "argos", "argyz", "ardis", "aridus", "arigos", 'argers']
    for variant in wake_variants:
        if variant in words:
            return True
    if "hey" in words or "hi" in words or "okay" in words:
        for variant in wake_variants:
            if variant in words:
                return True
    return False


def extract_followup_prompt(transcription):
    text = (transcription or "").lower()
    wake_variants = [
        "hey argus", "hi argus", "okay argus", "argus",
        "hey aragus", "hi aragus", "okay aragus", "aragus",
        "hey arghas", "hi arghas", "okay arghas", "arghas",
        "hey argas", "hi argas", "okay argas", "argas",
        "hey arghus", "hi arghus", "okay arghus", "arghus",
        "hey argos", "hi argos", "okay argos", "argos",
        "hey ardis", "hi ardis", "okay ardis", "ardis",
        "hey aridus", "hi aridus", "okay aridus", "aridus",
        "hey arigos", "hi arigos", "okay arigos", "arigos",
        "hey argers", "hi argers", "okay argers", "argers",
    ]
    for marker in wake_variants:
        if marker in text:
            return text.replace(marker, "", 1).strip()
    return text.strip()


# ------------------------------------------
# 3. BACKGROUND THREAD 1: VISION WORKER (X3D)
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


# ------------------------------------------
# 4. BACKGROUND THREAD 2: VOCAL ENGINE (pyttsx3)
# ------------------------------------------
def tts_worker():
    """Isolated speaking thread that signals when it finishes speaking."""
    global system_is_running, tts_done_event, tts_interrupt_requested

    while system_is_running:
        try:
            text_to_say = tts_queue.get(timeout=0.5)
            tts_interrupt_requested = False
            tts_done_event.clear()
            speak_text(text_to_say)

            if tts_interrupt_requested:
                print("[TTS] Interrupted by new wake word.")
            tts_done_event.set()

        except queue.Empty:
            pass


# ------------------------------------------
# 5. BACKGROUND THREAD 3: AUDIO WORKER (Whisper + Moondream2)
# ------------------------------------------
def audio_worker():
    global subtitle_text, system_is_running, current_action_label, argus_reply_text, memory_log, current_cv2_frame, tts_done_event, thinking_active, thinking_text, assistant_state
    
    recognizer = sr.Recognizer()
    # 1. NEW: Strict microphone noise gating
    recognizer.energy_threshold = 4000  
    recognizer.dynamic_energy_threshold = True

    try:
        mic = sr.Microphone()
        print("[AUDIO] Microphone detected successfully.")
        with mic as source:
            print("[AUDIO] Calibrating for background noise... (Stay quiet for 2s)")
            # 2. NEW: Increased calibration duration to 2.0 seconds
            recognizer.adjust_for_ambient_noise(source, duration=2.0)
            print("[AUDIO] Calibration complete. Listening active.")
    except Exception as e:
        print(f"[AUDIO ERROR] Failed to initialize microphone: {e}")
        subtitle_text = "AUDIO ERROR: No microphone found!"
        return

    while system_is_running:
        try:
            with mic as source:
                if assistant_state == "awaiting_prompt":
                    subtitle_text = "MIC: (I AM LISTENING - SPEAK NOW!)"
                    print("[A.R.G.U.S] I AM LISTENING - SPEAK NOW!")
                    try:

                        command_audio = recognizer.listen(source, timeout=3.0, phrase_time_limit=7.0)
                    except sr.WaitTimeoutError:
                        tts_queue.put("I didn't hear a command.")
                        assistant_state = "idle"
                        continue

                    # Process the audio...
                    with open("temp_mic.wav", "wb") as f:
                        f.write(command_audio.get_wav_data())
                    
                    # ... [Your Whisper transcription code stays exactly the same here] ...

                    # 3. NEW: The Whisper accuracy constraints
                    segments, _ = audio_model.transcribe(
                        "temp_mic.wav", 
                        beam_size=5,
                        language="en", 
                        condition_on_previous_text=False,
                        temperature=0.0,
                        initial_prompt="A conversation with an AI engine named Argus. Technical terms: Moondream2, X3D, egocentric, VRAM.",
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500)
                    )
                    user_spoken_prompt = "".join([segment.text for segment in segments]).strip().lower()
                    if user_spoken_prompt:
                        subtitle_text = f"MIC: {user_spoken_prompt}"
                        play_capture_chime()
                    else:
                        user_spoken_prompt = "Describe the main objects in this scene."
                    assistant_state = "idle"
                    process_prompt(user_spoken_prompt)
                    continue

                audio_data = recognizer.listen(source, timeout=3.0, phrase_time_limit=10.0)

                with open("temp_mic.wav", "wb") as f:
                    f.write(audio_data.get_wav_data())

                # 4. NEW: The Whisper accuracy constraints (Applied to the passive listening loop as well)
                segments, _ = audio_model.transcribe(
                    "temp_mic.wav", 
                    beam_size=5,
                    language="en", 
                    condition_on_previous_text=False,
                    temperature=0.0,
                    initial_prompt="A conversation with an AI engine named Argus. Technical terms: Moondream2, X3D, egocentric, VRAM.",
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                transcription = "".join([segment.text for segment in segments]).strip().lower()

                if transcription:
                    print(f"[AUDIO WHISPER] Transcribed: {transcription}")
                    subtitle_text = f"MIC: {transcription}"

                    if "what is" in transcription and "doing" in transcription:
                        answer = f"Right now, he is {current_action_label}."
                        argus_reply_text = f"A.R.G.U.S: {answer.upper()}"
                        tts_queue.put(answer)

                    elif "first" in transcription or "start" in transcription:
                        if len(memory_log) > 0:
                            first_event = memory_log[0]
                            answer = f"At the start of the video, he was {first_event}."
                        else:
                            answer = "I haven't recorded any actions yet."
                        argus_reply_text = f"A.R.G.U.S: {answer.upper()}"
                        tts_queue.put(answer)

                    elif detect_wake_word(transcription):
                        interrupt_tts()
                        
                        print("[AUDIO] Wake word detected. Enforcing 2-step pipeline...")
                        subtitle_text = "MIC: HEY ARGUS..."
                        
                        # 1. Trigger the "I am listening" sequence
                        tts_done_event.clear()
                        tts_queue.put("I am listening.")
                        threading.Thread(target=play_startup_chime, daemon=True).start()

                        # 2. Wait for the AI to physically finish talking
                        tts_done_event.wait(timeout=1.0)
                        time.sleep(0.2) # Tiny buffer to ensure mic doesn't catch the speaker echo

                        # 3. FORCE the system into the 10-second recording state
                        # (We removed the extract_followup_prompt logic entirely)
                        assistant_state = "awaiting_prompt"

        except sr.WaitTimeoutError:
            pass
        except Exception:
            pass

        time.sleep(0.1)


# Start all three AI brains in the background
threading.Thread(target=vision_worker, daemon=True).start()
threading.Thread(target=tts_worker, daemon=True).start()
threading.Thread(target=audio_worker, daemon=True).start()


# ------------------------------------------
# 6. THE STREAMING HUD LOOP (MAIN THREAD)
# ------------------------------------------
print("[SYSTEM] Launching Real-Time HUD. Press 'q' to close window.")
cap = cv2.VideoCapture(target_video)

# --- RESIZABLE WINDOW FIX ---
cv2.namedWindow("A.R.G.U.S. Vision Engine", cv2.WINDOW_NORMAL)
cv2.resizeWindow("A.R.G.U.S. Vision Engine", 1280, 720) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break 

    current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 5000.0
    frame_height, frame_width, _ = frame.shape
    
    # Save the clean frame globally for Moondream2 to use
    current_cv2_frame = frame.copy() 

    # 1. Draw Top HUD (Vision Prediction - Yellow)
    cv2.rectangle(frame, (10, 10), (800, 60), (0, 0, 0), -1) 
    cv2.putText(frame, prediction_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # 2. Draw Bottom HUD (Mic Subtitles + Animated Microphone)
    cv2.rectangle(frame, (10, frame_height - 50), (frame_width - 10, frame_height - 10), (0, 0, 0), -1)
    mic_x, mic_y = 35, frame_height - 30
    mic_animation_phase = int(time.time() * 7) % 4
    mic_radius = 12 + (mic_animation_phase % 2)
    cv2.circle(frame, (mic_x, mic_y), mic_radius, (0, 255, 0), 2)
    cv2.line(frame, (mic_x, mic_y + mic_radius), (mic_x, mic_y + 18), (0, 255, 0), 2)
    cv2.line(frame, (mic_x - 8, mic_y + 18), (mic_x + 8, mic_y + 18), (0, 255, 0), 2)
    for bar_idx in range(3):
        bar_height = 5 + ((mic_animation_phase + bar_idx) % 3) * 3
        x = mic_x + 18 + bar_idx * 7
        cv2.rectangle(frame, (x, mic_y - bar_height // 2), (x + 3, mic_y + bar_height // 2), (0, 255, 0), -1)
    cv2.putText(frame, subtitle_text, (65, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # 3. Draw Thinking Animation while Moondream is generating
    if thinking_active:
        spinner = [".  ", ".. ", "...", ".. "]
        dots = spinner[int(time.time() * 6) % len(spinner)]
        cv2.putText(frame, f"THINKING{dots}", (frame_width - 220, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # 4. Draw AI Answer in the CENTER of the screen (Blue)
    if argus_reply_text != "":
        text_size = cv2.getTextSize(argus_reply_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        text_x = (frame_width - text_size[0]) // 2
        text_y = (frame_height + text_size[1]) // 2
        
        cv2.rectangle(frame, (text_x - 15, text_y - text_size[1] - 15), 
                             (text_x + text_size[0] + 15, text_y + 15), (0, 0, 0), -1)
        cv2.putText(frame, argus_reply_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 150, 100), 2, cv2.LINE_AA)

    cv2.imshow("A.R.G.U.S. Vision Engine", frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Cleanup
system_is_running = False 
cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Shutdown complete.")