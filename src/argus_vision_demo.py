import torch
import json
import urllib.request
import os
import cv2
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo

# ==========================================
# A.R.G.U.S. VISUAL DEMO ENGINE
# ==========================================
print("\n[SYSTEM] Booting A.R.G.U.S. Vision Demo...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load X3D Model & Kinetics Labels
model_name = 'x3d_s'
model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True).to(device).eval()

url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
if not os.path.exists("kinetics_classnames.json"):
    urllib.request.urlretrieve(url, "kinetics_classnames.json")
with open("kinetics_classnames.json", "r") as f:
    kinetics_classnames = json.load(f)
kinetics_id_to_classname = {v: str(k).replace('"', "").replace('_', ' ') for k, v in kinetics_classnames.items()}

# 2. Frame Transformations (Optimized for X3D)
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

# 3. Target Video (Update this path to one of your downloaded EPIC-KITCHENS videos)
target_video = "videos\P01_06.mp4" 

if not os.path.exists(target_video):
    print(f"\n[ERROR] Could not find video at: {target_video}")
    print("Please check the file path and try again.")
    exit()

print(f"[SYSTEM] Analyzing video: {target_video}...")

# 4. Extract Dynamic Center Clip & Run Inference
video = EncodedVideo.from_path(target_video)
duration = float(video.duration)
start_sec = max(0.0, (duration / 2.0) - 1.0) # Grab the middle 2 seconds for best action context
end_sec = min(duration, start_sec + 2.0)

video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
video_data = transform(video_data)
inputs = video_data["video"].to(device)[None, ...]

with torch.no_grad():
    preds = model(inputs)
    preds = torch.nn.Softmax(dim=1)(preds)
    top_idx = preds.topk(k=1).indices[0][0].item()
    confidence = preds.topk(k=1).values[0][0].item() * 100
    
prediction_text = f"TARGET LOCKED: {kinetics_id_to_classname[top_idx].upper()} ({confidence:.1f}%)"
print(f"\n[A.R.G.U.S.] Analysis Complete: {prediction_text}")

# 5. The E.D.I.T.H. Visual Overlay (OpenCV HUD)
print("[SYSTEM] Launching Heads-Up Display (HUD). Press 'q' to close window.")
cap = cv2.VideoCapture(target_video)
cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000) # Fast forward to the analyzed clip

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or cap.get(cv2.CAP_PROP_POS_MSEC) > end_sec * 1000:
        break # Stop when the 2-second clip ends

    # Draw the HUD
    cv2.rectangle(frame, (10, 10), (800, 60), (0, 0, 0), -1) # Black background box
    cv2.putText(frame, prediction_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("A.R.G.U.S. Vision Engine", frame)
    
    # Play at roughly normal speed (wait ~30ms between frames)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()