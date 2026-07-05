import pandas as pd
import os

print("[SYSTEM] Parsing EPIC-KITCHENS Annotations...")

# 1. Load the official annotations
official_csv_path = "data/EPIC_100_train.csv"
if not os.path.exists(official_csv_path):
    print(f"[ERROR] Could not find {official_csv_path}. Please download it from GitHub.")
    exit()

df = pd.read_csv(official_csv_path)

# 2. Define the 5 custom actions for the A.R.G.U.S. micro-dataset
# We map EPIC-KITCHENS string verbs to our own numerical IDs (0-4)
target_actions = {
    'wash': 0,
    'cut': 1,
    'mix': 2,
    'pour': 3,
    'open': 4
}

# 3. Filter and Format the Data
training_data = []
downloaded_videos_dir = "videos/" # Update this to your exact video folder if different

for index, row in df.iterrows():
    verb = str(row['verb']).lower()
    
    if verb in target_actions:
        video_id = row['video_id'] # e.g., P01_01
        participant_id = row['participant_id'] # e.g., P01
        
        # Construct the expected local file path
        # Assuming format: EPIC-KITCHENS/P01/videos/P01_01.MP4 or similar
        video_filename = f"{video_id}.MP4"
        
        # (Optional but recommended) Check if you actually downloaded this specific video
        # If you haven't downloaded it, we shouldn't add it to the training manifest
        # if os.path.exists(os.path.join(downloaded_videos_dir, video_filename)):
        
        training_data.append({
            'video_path': f"videos/{video_filename}",
            'start_time': row['start_timestamp'], # Needs to be converted to seconds in some cases
            'end_time': row['stop_timestamp'],
            'label_id': target_actions[verb]
        })

# 4. Save the new PyTorchVideo-compatible manifest
new_df = pd.DataFrame(training_data)
new_df.to_csv("data/train_dataset.csv", index=False, header=False)

print(f"[SUCCESS] Extracted {len(new_df)} action clips!")
print("[SYSTEM] Saved clean manifest to data/train_dataset.csv. Ready for Transfer Learning.")