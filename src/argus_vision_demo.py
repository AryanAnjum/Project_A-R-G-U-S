import torch
import json
import urllib.request
import os
import cv2
import threading
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo

# ==========================================
# A.R.G.U.S. VISUAL DEMO ENGINE (SAFE-THREADING MODE)
# ==========================================
print("\n[SYSTEM] Booting A.R.G.U.S. Vision Demo for Faculty Presentation...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load Pre-trained X3D Model & Kinetics Labels
model_name = 'x3d_s'
print("[SYSTEM] Loading Pre-trained Base Model...")
model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True).to(device).eval()

url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
if not os.path.exists("kinetics_classnames.json"):
    urllib.request.urlretrieve(url, "kinetics_classnames.json")
with open("kinetics_classnames.json", "r") as f:
    kinetics_classnames = json.load(f)
kinetics_id_to_classname = {v: str(k).replace('"', "").replace('_', ' ') for k, v in kinetics_classnames.items()}

# 2. Frame Transformations
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

# 3. Target Video Setup (Fixed string literal for Windows paths)
target_video = r"videos\P01_06.mp4" 

if not os.path.exists(target_video):
    print(f"\n[ERROR] Could not find video at: {target_video}")
    exit()

print(f"[SYSTEM] Establishing video feed from: {target_video}...")

# 4. The Streaming HUD Loop
print("[SYSTEM] Launching Real-Time HUD. Press 'q' to close window.")
cap = cv2.VideoCapture(target_video)

frame_count = 0
prediction_text = "INITIALIZING A.R.G.U.S..."
is_inferencing = False  # <--- THIS IS THE CRASH FIX

# The background AI function
def run_ai_inference(current_time):
    global prediction_text, is_inferencing
    try:
        # Create a safe, temporary video reader just for this thread
        local_video = EncodedVideo.from_path(target_video)
        video_data = local_video.get_clip(start_sec=current_time, end_sec=current_time + 2.0)
        video_data = transform(video_data)
        inputs = video_data["video"].to(device)[None, ...]

        with torch.no_grad():
            preds = model(inputs)
            preds = torch.nn.Softmax(dim=1)(preds)
            top_idx = preds.topk(k=1).indices[0][0].item()
            confidence = preds.topk(k=1).values[0][0].item() * 100
            
            prediction_text = f"TARGET LOCKED: {kinetics_id_to_classname[top_idx].upper()} ({confidence:.1f}%)"
    except Exception as e:
        pass 
    finally:
        # Unlock so the next frame can run!
        is_inferencing = False

# Read through the video continuously
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break # Video ended

    # ONLY run if the previous AI guess is finished (prevents crashes)
    if frame_count % 30 == 0 and not is_inferencing:
        is_inferencing = True # Lock it
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        # Start a background thread
        threading.Thread(target=run_ai_inference, args=(current_time,), daemon=True).start()

    # Draw the HUD Overlay
    cv2.rectangle(frame, (10, 10), (800, 60), (0, 0, 0), -1) 
    cv2.putText(frame, prediction_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Display the live feed
    cv2.imshow("A.R.G.U.S. Vision Engine", frame)
    frame_count += 1
    
    # Play speed (wait ~30ms between frames). Press 'q' to quit.
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()