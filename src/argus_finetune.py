import torch
import torch.nn as nn
import pandas as pd
import os
from pytorchvideo.data import LabeledVideoDataset, make_clip_sampler
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from torch.utils.data import DataLoader

print("\n[SYSTEM] Initializing A.R.G.U.S. Transfer Learning Pipeline...")

# ==========================================
# 1. Hyperparameters & Hardware
# ==========================================
NUM_CUSTOM_CLASSES = 5 
BATCH_SIZE = 2 
LEARNING_RATE = 1e-4
EPOCHS = 1 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[SYSTEM] Hardware Check: Using {DEVICE.upper()}")

# ==========================================
# 2. Cross-Reference Manifest with Local Videos
# ==========================================
print("[SYSTEM] Reading training manifest...")
df = pd.read_csv("data/train_dataset.csv", names=['video_path', 'start', 'end', 'label_id'])

labeled_video_paths = []
for _, row in df.iterrows():
    vid_path = row['video_path']
    if os.path.exists(vid_path):
        labeled_video_paths.append((vid_path, {"label": int(row['label_id'])}))

if len(labeled_video_paths) == 0:
    print("[ERROR] No downloaded videos match the manifest.")
    print("Ensure your .MP4 files are inside the 'videos/' folder!")
    exit()

print(f"[SYSTEM] Ready to train on {len(labeled_video_paths)} local clips.")

# ==========================================
# 3. Model Surgery (Swapping the Output Head)
# ==========================================
print("[SYSTEM] Loading baseline X3D model and swapping classification head...")
model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)

in_features = model.blocks[5].proj.in_features
model.blocks[5].proj = nn.Linear(in_features, NUM_CUSTOM_CLASSES)
model = model.to(DEVICE)

# Freeze the backbone
for param in model.parameters():
    param.requires_grad = False
for param in model.blocks[5].proj.parameters():
    param.requires_grad = True 

# ==========================================
# 4. Data Loading Pipeline
# ==========================================
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

train_dataset = LabeledVideoDataset(
    labeled_video_paths=labeled_video_paths,
    clip_sampler=make_clip_sampler("uniform", 2.0),
    transform=transform,
    decode_audio=False
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0)

# ==========================================
# 5. The Training Loop (Corrected)
# ==========================================
# CORRECTED: Added Optimizer and Loss Function
optimizer = torch.optim.Adam(model.blocks[5].proj.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

print("\n[SYSTEM] Commencing Training Loop...")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0.0
    batch_count = 0
    
    for batch in train_loader:
        # Check for video_index (might return 'Unknown' if not present in your dataset version)
        vid_idx = batch.get("video_index", "Unknown") 
        
        inputs = batch["video"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_fn(predictions, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
        
        print(f" -> Batch {batch_count} | Video Index: {vid_idx} | Loss: {loss.item():.4f}")
        
    avg_loss = total_loss / max(1, batch_count)
    print(f"[EPOCH {epoch+1}/{EPOCHS} COMPLETE] Average Loss: {avg_loss:.4f}\n")
    
# ==========================================
# 6. Save the Custom Weights
# ==========================================
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/argus_x3d_finetuned.pth")
print("[SUCCESS] New A.R.G.U.S. Engine Weights Saved to models/argus_x3d_finetuned.pth!")