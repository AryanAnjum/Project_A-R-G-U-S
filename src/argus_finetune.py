import torch
import torch.nn as nn
import pandas as pd
import os
from pytorchvideo.data import LabeledVideoDataset, make_clip_sampler
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from torch.utils.data import DataLoader
from tqdm import tqdm  # <-- For the ETA progress bar

print("\n[SYSTEM] Initializing A.R.G.U.S. Transfer Learning Pipeline...")

# ==========================================
# 1. Hyperparameters, Hardware & Checkpoint Config
# ==========================================
NUM_CUSTOM_CLASSES = 5 
BATCH_SIZE = 2 
LEARNING_RATE = 1e-4
EPOCHS = 1 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- NEW: Checkpoint Settings ---
CHECKPOINT_PATH = "models/argus_checkpoint.pth"
SAVE_INTERVAL = 10  # Save progress every 10 batches

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
# 5. Fault-Tolerant Training Loop
# ==========================================
optimizer = torch.optim.Adam(model.blocks[5].proj.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

# --- NEW: Checkpoint Loading Logic ---
start_epoch = 0
start_batch = 0
os.makedirs("models", exist_ok=True)

if os.path.exists(CHECKPOINT_PATH):
    print(f"\n[SYSTEM] Found interrupted training state at {CHECKPOINT_PATH}!")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    start_batch = checkpoint['batch_count']
    print(f"[SYSTEM] Successfully restored memory. Resuming from Batch {start_batch}...")

print("\n[SYSTEM] Commencing Training Loop...")
model.train()

for epoch in range(start_epoch, EPOCHS):
    total_loss = 0.0
    batch_count = 0
    processed_this_epoch = 0
    
    # Wrap the loader in TQDM to create the visual progress bar
    progress_bar = tqdm(train_loader, desc=f"Training", unit="batch")
    
    for batch in progress_bar:
        batch_count += 1
        
        # --- FAST FORWARD LOGIC ---
        # If resuming from a crash, quickly skip the batches we already learned
        if epoch == start_epoch and batch_count <= start_batch:
            progress_bar.set_postfix({"Status": "Fast-forwarding to checkpoint..."})
            continue
            
        # Get current video names to display on screen
        vid_names = batch.get("video_name", ["Unknown"])
        current_videos = ", ".join([str(v) for v in vid_names])
        if len(current_videos) > 25:
            current_videos = current_videos[:22] + "..." # Keep it short so it fits nicely
        
        inputs = batch["video"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_fn(predictions, labels)
        
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        total_loss += loss_val
        processed_this_epoch += 1
        
        # Update the live progress bar text
        progress_bar.set_postfix({
            "Loss": f"{loss_val:.4f}",
            "Video": current_videos
        })
        
        # --- AUTO-SAVE CHECKPOINT ---
        if batch_count % SAVE_INTERVAL == 0:
            torch.save({
                'epoch': epoch,
                'batch_count': batch_count,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, CHECKPOINT_PATH)
            
    # Calculate average loss, preventing division by zero if all batches were skipped
    avg_loss = total_loss / max(1, processed_this_epoch)
    print(f"\n[EPOCH {epoch+1}/{EPOCHS} COMPLETE] Average Loss: {avg_loss:.4f}\n")
    
# ==========================================
# 6. Save the Final Custom Weights
# ==========================================
# Once fully complete, save the final ready-to-use .pth file
torch.save(model.state_dict(), "models/argus_x3d_finetuned.pth")
print("[SUCCESS] New A.R.G.U.S. Engine Weights Saved to models/argus_x3d_finetuned.pth!")

# Optional: Clean up the temporary checkpoint file since we are officially done
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)