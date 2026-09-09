import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

# ==========================================
# 1. HARDWARE & PATH CONFIGURATION
# ==========================================
DATA_JSON = "math_data/train_samples_all_tuning.json"
IMG_DIR = "math_data/data_images"
OUTPUT_DIR = "qwen2-vl-argus-math-adapter"

# ==========================================
# 2. DATASET PARSER (LLaVA to Qwen2-VL)
# ==========================================
class MathV360KDataset(Dataset):
    def __init__(self, json_path, img_dir):
        print(f"[SYSTEM] Loading dataset from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.img_dir = img_dir
        
        # Optional: Slice the dataset if you only want to train on a subset (e.g., 50,000 samples) to save time
        # self.data = self.data[:50000] 
        print(f"[SYSTEM] Loaded {len(self.data)} mathematical samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Load the Image
        # Note: MathV360K json "image" paths sometimes include prefixes. Adjust if necessary.
        img_path = os.path.join(self.img_dir, item["image"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback to a blank image if corrupted to prevent training crash
            image = Image.new('RGB', (224, 224), color='white')

        # 2. Parse LLaVA conversational format
        # LLaVA format embeds "<image>\n" in the text. We must remove it for Qwen2-VL.
        human_text = item["conversations"][0]["value"].replace("<image>\n", "").replace("\n<image>", "").strip()
        assistant_text = item["conversations"][1]["value"].strip()

        # 3. Build Qwen2-VL standard ChatML Message List
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": human_text}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_text}
                ]
            }
        ]
        return messages

# ==========================================
# 3. CUSTOM DATA COLLATOR
# ==========================================
def custom_collate_fn(batch):
    batch_texts = []
    batch_images = []
    
    for messages in batch:
        # Format the prompt using Qwen's official chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        # Extract the PIL images from the dictionary
        image_inputs, _ = process_vision_info(messages)
        
        batch_texts.append(text)
        batch_images.extend(image_inputs)
        
    # Process text and images into tensors
    inputs = processor(
        text=batch_texts,
        images=batch_images,
        padding=True,
        return_tensors="pt"
    )
    
    # Create labels for the loss function (Standard Causal LM training)
    labels = inputs["input_ids"].clone()
    # Mask padding tokens so the model isn't penalized for them
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    
    return inputs

# ==========================================
# 4. MODEL BOOTSTRAP (8GB VRAM SAFE)
# ==========================================
print("[SYSTEM] Initializing 4-bit Base Qwen2-VL-2B...")

# Strict image resolution limits. Without this, complex charts will OOM your RTX 3060 instantly.
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", 
    min_pixels=256 * 28 * 28, 
    max_pixels=768 * 28 * 28 
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    quantization_config=bnb_config,
    device_map="cuda"
)

# Prepare for LoRA training by freezing base weights and enabling gradient checkpointing
base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

# Attach the new untrained LoRA Adapter matrices (Targeting ALL attention/MLP modules for max math reasoning)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model_math = get_peft_model(base_model, peft_config)
model_math.print_trainable_parameters()

# ==========================================
# 5. TRAINING LOOP
# ==========================================
dataset = MathV360KDataset(DATA_JSON, IMG_DIR)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,        # MUST be 1 for 8GB VRAM
    gradient_accumulation_steps=8,        # Accumulate gradients to simulate batch size of 8
    learning_rate=2e-5,
    num_train_epochs=1,
    fp16=False,
    bf16=True,                            # Use bfloat16 to match Qwen's native precision
    logging_steps=10,
    save_steps=500,
    optim="paged_adamw_8bit",             # Offloads optimizer states to CPU RAM to save VRAM
    remove_unused_columns=False,          # CRITICAL: Must be false for multimodal processing
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False} 
)

trainer = Trainer(
    model=model_math,
    args=training_args,
    train_dataset=dataset,
    data_collator=custom_collate_fn
)

print("\n[SYSTEM] Commencing A.R.G.U.S. Math Module Training...")
trainer.train()

print(f"\n[SYSTEM] Training Complete! Saving new adapter to {OUTPUT_DIR}")
trainer.model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("[SYSTEM] Adapter Saved Successfully.")