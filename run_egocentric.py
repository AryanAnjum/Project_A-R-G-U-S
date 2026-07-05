import torch
import json
import urllib.request
import os
import glob
import time
import gc

from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)

print("\n" + "="*66)
print("[SYSTEM] Initializing Optimized Egocentric Batch Pipeline...")
print("="*66)

# 1. Setup Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[SYSTEM] Hardware Verified. Routing compute to: {device.upper()}")

# 2. Load Model
model_name = 'x3d_s'
model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
model = model.to(device)
model = model.eval()

# 3. Load Labels
url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
urllib.request.urlretrieve(url, "kinetics_classnames.json")
with open("kinetics_classnames.json", "r") as f:
    kinetics_classnames = json.load(f)
kinetics_id_to_classname = {v: str(k).replace('"', "") for k, v in kinetics_classnames.items()}

# 4. Define Transformations
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

# 5. Strict File Discovery
video_folder = "videos"
all_files = list(set(glob.glob(os.path.join(video_folder, "*.mp4")) + 
                     glob.glob(os.path.join(video_folder, "*.MP4"))))
video_files = sorted(all_files)

if not video_files:
    print(f"[ERROR] No files found in '{video_folder}'.")
    exit()

print(f"[SYSTEM] Found {len(video_files)} unique files. Starting inference...")

# 6. Batch Processing Loop
for video_path in video_files:
    file_name = os.path.basename(video_path)
    print(f"\n[INFERENCE] Ingesting: {file_name} ...")
    
    video = None 
    try:
        video = EncodedVideo.from_path(video_path)
        video_data = video.get_clip(start_sec=0.0, end_sec=2.0)
        video_data = transform(video_data)
        inputs = video_data["video"].to(device)[None, ...]

        start_time = time.time()
        with torch.no_grad():
            preds = model(inputs)
        inference_time = (time.time() - start_time) * 1000

        # Output Results
        preds = torch.nn.Softmax(dim=1)(preds)
        top5 = preds.topk(k=5)
        
        print("-" * 66)
        print(f"| VIDEO TARGET: {file_name:<49} |")
        print(f"| GPU LATENCY:  {inference_time:>6.2f} ms |")
        print("-" * 66)
        for i in range(5):
            cls_name = kinetics_id_to_classname[int(top5.indices[0][i])].replace('_', ' ')
            print(f"| {i+1:<4} | {cls_name[:40]:<40} | {float(top5.values[0][i])*100:>6.2f}% |")
        print("=" * 66)

    except Exception as e:
        print(f"[WARNING] Skipping {file_name} due to error: {e}")
    
    finally:
        # Explicitly close and flush
        if video is not None:
            video._container.close()
            del video
        torch.cuda.empty_cache() 
        gc.collect()             
        time.sleep(1.5) 

print("\n[SYSTEM] Batch evaluation successfully completed.")